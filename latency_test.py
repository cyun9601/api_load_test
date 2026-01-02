import asyncio
import time
import statistics
from typing import List, Dict, Optional, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import json
import io
import random
import numpy as np
import wave
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용


@dataclass
class TestResult:
    """단일 테스트 결과"""
    response_time: float
    success: bool
    error: Optional[str] = None
    text: Optional[str] = None  # STT 예측 텍스트
    audio_duration: Optional[float] = None  # 오디오 길이 (초)
    rtf: Optional[float] = None  # Real-Time Factor (처리 시간 / 오디오 길이)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    avg_rtf: float  # 평균 RTF
    min_rtf: float  # 최소 RTF
    max_rtf: float  # 최대 RTF
    median_rtf: float  # 중앙값 RTF
    p95_rtf: float  # P95 RTF
    p99_rtf: float  # P99 RTF


class STTLoadTester:
    """STT 모델 부하 테스터"""
    
    def __init__(
        self,
        api_call_func,
        audio_generator_func,
        total_requests: int,
        warmup_requests: int,
        concurrent_requests: int = 1,
        request_delay: float = 0.0,
        save_audio_samples: bool = False
    ):
        """
        Args:
            api_call_func: STT API를 호출하는 비동기 함수 (audio_data: io.BytesIO를 인자로 받음)
            audio_generator_func: 오디오를 생성하는 함수 (io.BytesIO를 반환)
            total_requests: 총 요청 수 (N)
            warmup_requests: 버릴 warm-up 요청 수 (M)
            concurrent_requests: 동시 요청 수
            request_delay: 요청 간 지연 시간 (초)
            save_audio_samples: 오디오 샘플 저장 여부
        """
        self.api_call_func = api_call_func
        self.audio_generator_func = audio_generator_func
        self.total_requests = total_requests
        self.warmup_requests = warmup_requests
        self.concurrent_requests = concurrent_requests
        self.request_delay = request_delay
        self.results: List[TestResult] = []  # 성능 테스트 결과
        self.warmup_results: List[TestResult] = []  # Cold start (warmup) 결과
        self.save_audio_samples: bool = save_audio_samples  # 오디오 샘플 저장 여부
        self.saved_audio_count: int = 0  # 저장된 오디오 개수
        self.result_dir: str = "result"  # 결과 저장 디렉토리
        self.timestamp_dir: Optional[str] = None  # 타임스탬프 하위 디렉토리
        self.audio_duration: Optional[float] = None  # 랜덤 오디오 생성 시 오디오 길이
    
    def _save_audio_sample(self, audio_data: io.BytesIO, request_type: str, request_id: int):
        """오디오 샘플을 파일로 저장"""
        if not self.save_audio_samples:
            return
        
        # 첫 번째 warmup과 첫 번째 성능 테스트만 저장
        if request_type == "warmup" and request_id == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_sample_warmup_{timestamp}.wav"
            self._write_audio_file(audio_data, filename)
            self.saved_audio_count += 1
        elif request_type == "performance" and request_id == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_sample_performance_{timestamp}.wav"
            self._write_audio_file(audio_data, filename)
            self.saved_audio_count += 1
    
    def _ensure_result_dir(self):
        """result 폴더와 타임스탬프 하위 폴더가 없으면 생성"""
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        # 타임스탬프 하위 폴더가 없으면 생성
        if self.timestamp_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.timestamp_dir = os.path.join(self.result_dir, timestamp)
        
        if not os.path.exists(self.timestamp_dir):
            os.makedirs(self.timestamp_dir)
    
    def _write_audio_file(self, audio_data: io.BytesIO, filename: str):
        """오디오 데이터를 파일로 저장"""
        try:
            self._ensure_result_dir()
            filepath = os.path.join(self.timestamp_dir, filename)
            audio_data.seek(0)
            with open(filepath, 'wb') as f:
                f.write(audio_data.read())
            print(f"🎵 오디오 샘플 저장: {filepath}")
        except Exception as e:
            print(f"⚠️ 오디오 저장 실패: {e}")
    
    async def _make_request(self, request_id: int, is_warmup: bool = False) -> TestResult:
        """단일 요청 실행 (오디오 생성 시간 제외)"""
        # 오디오 생성 (시간 측정 제외)
        audio_data = self.audio_generator_func(is_warmup=is_warmup)
        
        # 오디오 길이 측정
        audio_duration = None
        if self.audio_duration is not None:
            # 랜덤 오디오 생성 모드
            audio_duration = self.audio_duration
        else:
            # Resource 폴더 모드 - 파일에서 측정
            file_path = getattr(audio_data, 'file_path', None)
            if file_path:
                audio_duration = get_audio_duration(audio_data, file_path)
                if audio_duration is None and request_id == 0:  # 첫 요청에서만 경고
                    print(f"⚠️ 오디오 길이를 측정할 수 없습니다: {file_path}")
                    print(f"   MP3 파일인 경우 'pip install mutagen'을 실행하세요.")
            else:
                # file_path가 없으면 filename으로 시도
                filename = getattr(audio_data, 'filename', None)
                if filename:
                    audio_duration = get_audio_duration(audio_data, filename)
                else:
                    audio_duration = get_audio_duration(audio_data)
        
        # 오디오 샘플 저장
        request_type = "warmup" if is_warmup else "performance"
        self._save_audio_sample(audio_data, request_type, request_id)
        
        # API 호출을 위해 오디오 데이터를 다시 읽을 수 있도록 복사
        audio_data.seek(0)
        audio_bytes = audio_data.read()
        audio_data_copy = io.BytesIO(audio_bytes)
        
        # API 호출만 시간 측정에 포함
        start_time = time.time()
        try:
            response = await self.api_call_func(audio_data_copy)
            response_time = time.time() - start_time
            
            # STT 예측 텍스트 추출
            text = None
            if isinstance(response, dict):
                # 다양한 응답 형식 지원
                text = response.get("text") or response.get("transcription") or response.get("result")
            elif isinstance(response, str):
                text = response
            
            # RTF 계산 (Real-Time Factor = 처리 시간 / 오디오 길이)
            rtf = None
            if audio_duration and audio_duration > 0:
                rtf = response_time / audio_duration
            
            return TestResult(
                response_time=response_time,
                success=True,
                text=text,
                audio_duration=audio_duration,
                rtf=rtf
            )
        except Exception as e:
            response_time = time.time() - start_time
            
            # RTF 계산 (실패한 경우에도)
            rtf = None
            if audio_duration and audio_duration > 0:
                rtf = response_time / audio_duration
            
            return TestResult(
                response_time=response_time,
                success=False,
                error=str(e),
                text=None,
                audio_duration=audio_duration,
                rtf=rtf
            )
    
    async def _run_requests(self, num_requests: int, is_warmup: bool = False) -> List[TestResult]:
        """요청 배치 실행"""
        results = []
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        async def bounded_request(request_id: int):
            async with semaphore:
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                result = await self._make_request(request_id, is_warmup=is_warmup)
                if is_warmup:
                    # Warmup (cold start) 결과 저장
                    self.warmup_results.append(result)
                else:
                    # 성능 테스트 결과 저장
                    results.append(result)
                    self.results.append(result)
                return result
        
        tasks = [bounded_request(i) for i in range(num_requests)]
        await asyncio.gather(*tasks)
        
        return results
    
    async def run(self) -> PerformanceMetrics:
        """로드 테스트 실행"""
        # 타임스탬프 폴더 생성
        self._ensure_result_dir()
        
        print(f"🚀 STT 모델 로드 테스트 시작")
        print(f"   총 요청 수: {self.total_requests}")
        print(f"   Warm-up 요청 수: {self.warmup_requests}")
        print(f"   동시 요청 수: {self.concurrent_requests}")
        print(f"   실제 측정 요청 수: {self.total_requests - self.warmup_requests}")
        print(f"   매 요청마다 새로운 음성과 유사한 오디오 생성")
        print(f"   결과 저장 경로: {self.timestamp_dir}")
        print()
        
        # Warm-up 단계
        if self.warmup_requests > 0:
            print(f"🔥 Warm-up 단계 ({self.warmup_requests}개 요청)...")
            warmup_start = time.time()
            await self._run_requests(self.warmup_requests, is_warmup=True)
            warmup_time = time.time() - warmup_start
            print(f"   Warm-up 완료 (소요 시간: {warmup_time:.2f}초)")
            print()
        
        # 실제 테스트 단계
        print(f"📊 성능 측정 단계 ({self.total_requests - self.warmup_requests}개 요청)...")
        test_start = time.time()
        await self._run_requests(self.total_requests - self.warmup_requests, is_warmup=False)
        test_time = time.time() - test_start
        
        # 메트릭 계산
        return self._calculate_metrics(test_time)
    
    def _calculate_metrics(self, total_time: float) -> PerformanceMetrics:
        """성능 메트릭 계산"""
        if not self.results:
            raise ValueError("테스트 결과가 없습니다.")
        
        response_times = [r.response_time for r in self.results]
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        successful_response_times = [r.response_time for r in successful_results]
        
        if not successful_response_times:
            raise ValueError("성공한 요청이 없습니다.")
        
        sorted_times = sorted(successful_response_times)
        n = len(sorted_times)
        
        # RTF 계산 (성공한 요청 중 RTF가 있는 것만)
        rtf_values = [r.rtf for r in successful_results if r.rtf is not None]
        
        if rtf_values:
            sorted_rtf = sorted(rtf_values)
            n_rtf = len(sorted_rtf)
        else:
            sorted_rtf = []
            n_rtf = 0
        
        return PerformanceMetrics(
            total_requests=len(self.results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            avg_response_time=statistics.mean(successful_response_times),
            min_response_time=min(successful_response_times),
            max_response_time=max(successful_response_times),
            median_response_time=statistics.median(sorted_times),
            p95_response_time=sorted_times[int(n * 0.95)] if n > 0 else 0,
            p99_response_time=sorted_times[int(n * 0.99)] if n > 0 else 0,
            requests_per_second=len(self.results) / total_time if total_time > 0 else 0,
            avg_rtf=statistics.mean(rtf_values) if rtf_values else 0.0,
            min_rtf=min(rtf_values) if rtf_values else 0.0,
            max_rtf=max(rtf_values) if rtf_values else 0.0,
            median_rtf=statistics.median(sorted_rtf) if sorted_rtf else 0.0,
            p95_rtf=sorted_rtf[int(n_rtf * 0.95)] if n_rtf > 0 else 0.0,
            p99_rtf=sorted_rtf[int(n_rtf * 0.99)] if n_rtf > 0 else 0.0
        )
    
    def print_results(self, metrics: PerformanceMetrics):
        """결과 출력"""
        print("\n" + "="*60)
        print("📈 성능 테스트 결과")
        print("="*60)
        print(f"총 요청 수: {metrics.total_requests}")
        print(f"성공한 요청: {metrics.successful_requests} ({metrics.successful_requests/metrics.total_requests*100:.1f}%)")
        print(f"실패한 요청: {metrics.failed_requests} ({metrics.failed_requests/metrics.total_requests*100:.1f}%)")
        print()
        print("응답 시간 통계:")
        print(f"  평균: {metrics.avg_response_time:.3f}초")
        print(f"  중앙값: {metrics.median_response_time:.3f}초")
        print(f"  최소: {metrics.min_response_time:.3f}초")
        print(f"  최대: {metrics.max_response_time:.3f}초")
        print(f"  P95: {metrics.p95_response_time:.3f}초")
        print(f"  P99: {metrics.p99_response_time:.3f}초")
        print()
        print("RTF (Real-Time Factor) 통계:")
        print(f"  평균: {metrics.avg_rtf:.3f}")
        print(f"  중앙값: {metrics.median_rtf:.3f}")
        print(f"  최소: {metrics.min_rtf:.3f}")
        print(f"  최대: {metrics.max_rtf:.3f}")
        print(f"  P95: {metrics.p95_rtf:.3f}")
        print(f"  P99: {metrics.p99_rtf:.3f}")
        print(f"  (RTF < 1.0: 실시간보다 빠름, RTF > 1.0: 실시간보다 느림)")
        print()
        print(f"처리량: {metrics.requests_per_second:.2f} 요청/초")
        print("="*60)
        
        # 실패한 요청 상세 정보
        if metrics.failed_requests > 0:
            print("\n❌ 실패한 요청 상세:")
            for i, result in enumerate(self.results):
                if not result.success:
                    print(f"  요청 #{i+1}: {result.error}")
    
    def save_histogram(self, filename: Optional[str] = None):
        """응답 시간 및 RTF 도수분포표(히스토그램)를 저장 (Cold start와 성능 테스트 구분)"""
        # Cold start (warmup)와 성능 테스트 결과 수집
        warmup_response_times = [r.response_time for r in self.warmup_results if r.success]
        performance_response_times = [r.response_time for r in self.results if r.success]
        warmup_rtf = [r.rtf for r in self.warmup_results if r.success and r.rtf is not None]
        performance_rtf = [r.rtf for r in self.results if r.success and r.rtf is not None]
        
        if not warmup_response_times and not performance_response_times:
            print("⚠️ 성공한 요청이 없어 히스토그램을 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "response_time_histogram.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        # Font settings
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 히스토그램 생성 (위아래 서브플롯: 위=응답 시간, 아래=RTF)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # === 위쪽: 응답 시간 히스토그램 ===
        all_times = warmup_response_times + performance_response_times
        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
            bins = np.linspace(min_time, max_time, 30)
        else:
            bins = 30
        
        # Cold start (warmup) histogram
        if warmup_response_times:
            ax1.hist(
                warmup_response_times,
                bins=bins,
                edgecolor='black',
                alpha=0.6,
                color='orange',
                label=f'Cold Start ({len(warmup_response_times)} requests)'
            )
        
        # Performance test histogram
        if performance_response_times:
            ax1.hist(
                performance_response_times,
                bins=bins,
                edgecolor='black',
                alpha=0.6,
                color='steelblue',
                label=f'Performance Test ({len(performance_response_times)} requests)'
            )
        
        # 통계 정보
        stats_lines = []
        if warmup_response_times:
            warmup_avg = statistics.mean(warmup_response_times)
            warmup_median = statistics.median(warmup_response_times)
            ax1.axvline(warmup_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                     label=f'Cold Start Avg: {warmup_avg:.3f}s')
            stats_lines.append(f'Cold Start: {len(warmup_response_times)} requests')
            stats_lines.append(f'  Avg: {warmup_avg:.3f}s')
            stats_lines.append(f'  Median: {warmup_median:.3f}s')
        
        if performance_response_times:
            perf_avg = statistics.mean(performance_response_times)
            perf_median = statistics.median(performance_response_times)
            ax1.axvline(perf_avg, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                     label=f'Performance Test Avg: {perf_avg:.3f}s')
            if not stats_lines:
                stats_lines.append('Performance Test:')
            stats_lines.append(f'  {len(performance_response_times)} requests')
            stats_lines.append(f'  Avg: {perf_avg:.3f}s')
            stats_lines.append(f'  Median: {perf_median:.3f}s')
        
        if all_times:
            stats_lines.append(f'\nOverall Min: {min(all_times):.3f}s')
            stats_lines.append(f'Overall Max: {max(all_times):.3f}s')
        
        ax1.set_xlabel('Response Time (seconds)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Response Time Histogram (Cold Start vs Performance Test)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        stats_text = '\n'.join(stats_lines)
        ax1.text(0.98, 0.98, stats_text,
                transform=ax1.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # === 아래쪽: RTF 히스토그램 ===
        all_rtf = warmup_rtf + performance_rtf
        if all_rtf:
            min_rtf = min(all_rtf)
            max_rtf = max(all_rtf)
            bins_rtf = np.linspace(min_rtf, max_rtf, 30)
        else:
            bins_rtf = 30
        
        # Cold start RTF histogram
        if warmup_rtf:
            ax2.hist(
                warmup_rtf,
                bins=bins_rtf,
                edgecolor='black',
                alpha=0.6,
                color='orange',
                label=f'Cold Start ({len(warmup_rtf)} requests)'
            )
        
        # Performance test RTF histogram
        if performance_rtf:
            ax2.hist(
                performance_rtf,
                bins=bins_rtf,
                edgecolor='black',
                alpha=0.6,
                color='steelblue',
                label=f'Performance Test ({len(performance_rtf)} requests)'
            )
        
        # RTF 통계 정보
        rtf_stats_lines = []
        if warmup_rtf:
            warmup_rtf_avg = statistics.mean(warmup_rtf)
            warmup_rtf_median = statistics.median(warmup_rtf)
            ax2.axvline(warmup_rtf_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                     label=f'Cold Start Avg: {warmup_rtf_avg:.3f}')
            rtf_stats_lines.append(f'Cold Start: {len(warmup_rtf)} requests')
            rtf_stats_lines.append(f'  Avg RTF: {warmup_rtf_avg:.3f}')
            rtf_stats_lines.append(f'  Median RTF: {warmup_rtf_median:.3f}')
        
        if performance_rtf:
            perf_rtf_avg = statistics.mean(performance_rtf)
            perf_rtf_median = statistics.median(performance_rtf)
            ax2.axvline(perf_rtf_avg, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                     label=f'Performance Test Avg: {perf_rtf_avg:.3f}')
            if not rtf_stats_lines:
                rtf_stats_lines.append('Performance Test:')
            rtf_stats_lines.append(f'  {len(performance_rtf)} requests')
            rtf_stats_lines.append(f'  Avg RTF: {perf_rtf_avg:.3f}')
            rtf_stats_lines.append(f'  Median RTF: {perf_rtf_median:.3f}')
        
        if all_rtf:
            rtf_stats_lines.append(f'\nOverall Min RTF: {min(all_rtf):.3f}')
            rtf_stats_lines.append(f'Overall Max RTF: {max(all_rtf):.3f}')
            # RTF = 1.0 기준선 표시
            ax2.axvline(1.0, color='green', linestyle=':', linewidth=2, alpha=0.7,
                       label='RTF = 1.0 (Real-time)')
        
        ax2.set_xlabel('RTF (Real-Time Factor)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('RTF Histogram (Cold Start vs Performance Test)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        rtf_stats_text = '\n'.join(rtf_stats_lines)
        ax2.text(0.98, 0.98, rtf_stats_text,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 히스토그램이 {filepath}에 저장되었습니다.")
    
    def save_timeline_graph(self, filename: Optional[str] = None):
        """요청 인덱스별 응답 시간 및 RTF 추이 그래프를 저장"""
        # 모든 요청 결과 수집 (cold start + 성능 테스트)
        all_results = self.warmup_results + self.results
        successful_results = [r for r in all_results if r.success]
        
        if not successful_results:
            print("⚠️ 성공한 요청이 없어 타임라인 그래프를 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "response_time_timeline.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        # Font settings
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 그래프 생성 (위아래 서브플롯: 위=응답 시간, 아래=RTF)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 요청 인덱스와 응답 시간, RTF 분리
        request_indices = []
        response_times = []
        rtf_values = []
        is_warmup_list = []
        
        # Warmup 결과 추가
        for idx, result in enumerate(self.warmup_results):
            if result.success:
                request_indices.append(idx + 1)
                response_times.append(result.response_time)
                rtf_values.append(result.rtf if result.rtf is not None else None)
                is_warmup_list.append(True)
        
        # 성능 테스트 결과 추가
        warmup_count = len([r for r in self.warmup_results if r.success])
        for idx, result in enumerate(self.results):
            if result.success:
                request_indices.append(warmup_count + idx + 1)
                response_times.append(result.response_time)
                rtf_values.append(result.rtf if result.rtf is not None else None)
                is_warmup_list.append(False)
        
        # Cold start와 성능 테스트를 색상으로 구분
        warmup_indices = [idx for idx, is_warmup in zip(request_indices, is_warmup_list) if is_warmup]
        warmup_times = [time for time, is_warmup in zip(response_times, is_warmup_list) if is_warmup]
        warmup_rtf = [rtf for rtf, is_warmup in zip(rtf_values, is_warmup_list) if is_warmup and rtf is not None]
        warmup_rtf_indices = [idx for idx, (rtf, is_warmup) in zip(request_indices, zip(rtf_values, is_warmup_list)) if is_warmup and rtf is not None]
        
        perf_indices = [idx for idx, is_warmup in zip(request_indices, is_warmup_list) if not is_warmup]
        perf_times = [time for time, is_warmup in zip(response_times, is_warmup_list) if not is_warmup]
        perf_rtf = [rtf for rtf, is_warmup in zip(rtf_values, is_warmup_list) if not is_warmup and rtf is not None]
        perf_rtf_indices = [idx for idx, (rtf, is_warmup) in zip(request_indices, zip(rtf_values, is_warmup_list)) if not is_warmup and rtf is not None]
        
        # === 위쪽: 응답 시간 타임라인 ===
        # Cold start 플롯
        if warmup_indices:
            ax1.scatter(warmup_indices, warmup_times, 
                      color='orange', alpha=0.6, s=30, 
                      label=f'Cold Start ({len(warmup_indices)} requests)')
            ax1.plot(warmup_indices, warmup_times, 
                   color='orange', alpha=0.3, linewidth=1)
        
        # 성능 테스트 플롯
        if perf_indices:
            ax1.scatter(perf_indices, perf_times, 
                      color='steelblue', alpha=0.6, s=30,
                      label=f'Performance Test ({len(perf_indices)} requests)')
            ax1.plot(perf_indices, perf_times, 
                   color='steelblue', alpha=0.3, linewidth=1)
        
        # 평균선 표시
        if warmup_times:
            warmup_avg = statistics.mean(warmup_times)
            ax1.axhline(warmup_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'Cold Start Avg: {warmup_avg:.3f}s')
        
        if perf_times:
            perf_avg = statistics.mean(perf_times)
            ax1.axhline(perf_avg, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'Performance Test Avg: {perf_avg:.3f}s')
        
        # Cold start와 성능 테스트 경계선 표시
        if warmup_indices and perf_indices:
            boundary = max(warmup_indices)
            ax1.axvline(boundary, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                      label='Warm-up / Performance Test Boundary')
        
        ax1.set_xlabel('Request Index', fontsize=12)
        ax1.set_ylabel('Response Time (seconds)', fontsize=12)
        ax1.set_title('Response Time Timeline (All Requests)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 통계 정보 텍스트 추가
        stats_lines = []
        if warmup_times:
            stats_lines.append(f'Cold Start: {len(warmup_times)} requests')
            stats_lines.append(f'  Avg: {statistics.mean(warmup_times):.3f}s')
            stats_lines.append(f'  Median: {statistics.median(warmup_times):.3f}s')
        
        if perf_times:
            if stats_lines:
                stats_lines.append('')
            stats_lines.append(f'Performance Test: {len(perf_times)} requests')
            stats_lines.append(f'  Avg: {statistics.mean(perf_times):.3f}s')
            stats_lines.append(f'  Median: {statistics.median(perf_times):.3f}s')
        
        if response_times:
            stats_lines.append('')
            stats_lines.append(f'Overall Min: {min(response_times):.3f}s')
            stats_lines.append(f'Overall Max: {max(response_times):.3f}s')
        
        stats_text = '\n'.join(stats_lines)
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # === 아래쪽: RTF 타임라인 ===
        # Cold start RTF 플롯
        if warmup_rtf_indices:
            ax2.scatter(warmup_rtf_indices, warmup_rtf, 
                       color='orange', alpha=0.6, s=30, marker='o',
                       label=f'Cold Start ({len(warmup_rtf)} requests)')
            ax2.plot(warmup_rtf_indices, warmup_rtf, 
                   color='orange', alpha=0.3, linewidth=1)
        
        # 성능 테스트 RTF 플롯
        if perf_rtf_indices:
            ax2.scatter(perf_rtf_indices, perf_rtf, 
                       color='steelblue', alpha=0.6, s=30, marker='o',
                       label=f'Performance Test ({len(perf_rtf)} requests)')
            ax2.plot(perf_rtf_indices, perf_rtf, 
                   color='steelblue', alpha=0.3, linewidth=1)
        
        # RTF 평균선 표시
        if warmup_rtf:
            warmup_rtf_avg = statistics.mean(warmup_rtf)
            ax2.axhline(warmup_rtf_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Cold Start Avg: {warmup_rtf_avg:.3f}')
        
        if perf_rtf:
            perf_rtf_avg = statistics.mean(perf_rtf)
            ax2.axhline(perf_rtf_avg, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Performance Test Avg: {perf_rtf_avg:.3f}')
        
        # RTF = 1.0 기준선 표시
        if warmup_rtf or perf_rtf:
            ax2.axhline(1.0, color='green', linestyle='-.', linewidth=2, alpha=0.8,
                       label='RTF = 1.0 (Real-time)')
        
        # Cold start와 성능 테스트 경계선 표시
        if warmup_rtf_indices and perf_rtf_indices:
            boundary = max(warmup_rtf_indices) if warmup_rtf_indices else 0
            if boundary > 0:
                ax2.axvline(boundary, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                          label='Warm-up / Performance Test Boundary')
        
        ax2.set_xlabel('Request Index', fontsize=12)
        ax2.set_ylabel('RTF (Real-Time Factor)', fontsize=12)
        ax2.set_title('RTF Timeline (All Requests)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # RTF 통계 정보 텍스트 추가
        rtf_stats_lines = []
        if warmup_rtf:
            rtf_stats_lines.append(f'Cold Start: {len(warmup_rtf)} requests')
            rtf_stats_lines.append(f'  Avg RTF: {statistics.mean(warmup_rtf):.3f}')
            rtf_stats_lines.append(f'  Median RTF: {statistics.median(warmup_rtf):.3f}')
        
        if perf_rtf:
            if rtf_stats_lines:
                rtf_stats_lines.append('')
            rtf_stats_lines.append(f'Performance Test: {len(perf_rtf)} requests')
            rtf_stats_lines.append(f'  Avg RTF: {statistics.mean(perf_rtf):.3f}')
            rtf_stats_lines.append(f'  Median RTF: {statistics.median(perf_rtf):.3f}')
        
        if warmup_rtf or perf_rtf:
            all_rtf_vals = warmup_rtf + perf_rtf
            rtf_stats_lines.append('')
            rtf_stats_lines.append(f'Overall Min RTF: {min(all_rtf_vals):.3f}')
            rtf_stats_lines.append(f'Overall Max RTF: {max(all_rtf_vals):.3f}')
        
        rtf_stats_text = '\n'.join(rtf_stats_lines)
        ax2.text(0.02, 0.98, rtf_stats_text,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 타임라인 그래프가 {filepath}에 저장되었습니다.")
    
    def save_results(self, metrics: PerformanceMetrics, filename: Optional[str] = None):
        """결과를 JSON 파일로 저장"""
        self._ensure_result_dir()
        
        if filename is None:
            filename = "stt_load_test_results.json"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "total_requests": self.total_requests,
                "warmup_requests": self.warmup_requests,
                "concurrent_requests": self.concurrent_requests,
                "request_delay": self.request_delay
            },
            "metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "avg_response_time": metrics.avg_response_time,
                "min_response_time": metrics.min_response_time,
                "max_response_time": metrics.max_response_time,
                "median_response_time": metrics.median_response_time,
                "p95_response_time": metrics.p95_response_time,
                "p99_response_time": metrics.p99_response_time,
                "requests_per_second": metrics.requests_per_second
            },
            "detailed_results": [
                {
                    "response_time": r.response_time,
                    "success": r.success,
                    "error": r.error,
                    "text": r.text
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과가 {filepath}에 저장되었습니다.")
        
        # 히스토그램과 타임라인 그래프 저장
        self.save_histogram()
        self.save_timeline_graph()


def generate_speech_like_audio(duration_seconds: float = 10.0, sample_rate: int = 16000) -> io.BytesIO:
    """
    실제 사람 음성과 유사한 오디오 데이터를 생성하여 WAV 파일 형식의 BytesIO 객체로 반환
    
    사람 음성의 특성을 모방:
    - 기본 주파수(F0)와 하모닉 구조
    - 포먼트(Formant) 주파수 (F1, F2, F3)
    - 시간에 따른 진폭 변조 (envelope)
    - 자연스러운 주파수 변조
    
    Args:
        duration_seconds: 오디오 길이 (초) - 기본값: 10.0
        sample_rate: 샘플링 레이트 (Hz) - 기본값: 16000
    
    Returns:
        WAV 형식의 오디오 데이터를 담은 BytesIO 객체
    """
    # 샘플 수 계산
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples)
    
    # 기본 주파수 (F0) - 사람 음성 범위: 남성 85-180Hz, 여성 165-255Hz
    # 랜덤하게 선택하되 자연스러운 범위
    base_f0 = random.uniform(100, 250)  # 일반적인 음성 범위
    
    # 포먼트 주파수 (Formant frequencies) - 사람 음성의 특성 주파수
    # F1: 300-1000Hz, F2: 800-3000Hz, F3: 2000-3500Hz
    formant_f1 = random.uniform(400, 800)
    formant_f2 = random.uniform(1000, 2500)
    formant_f3 = random.uniform(2500, 3500)
    
    # 초기 오디오 데이터
    audio_data = np.zeros(num_samples)
    
    # 기본 주파수와 하모닉 생성 (음성의 하모닉 구조)
    # 기본 주파수와 그 배음들을 생성
    num_harmonics = random.randint(5, 10)
    for h in range(1, num_harmonics + 1):
        harmonic_freq = base_f0 * h
        if harmonic_freq < sample_rate / 2:  # 나이퀴스트 주파수 제한
            # 하모닉의 진폭은 고주파수로 갈수록 감소
            amplitude = 0.3 / h * random.uniform(0.7, 1.3)
            phase = random.uniform(0, 2 * np.pi)
            audio_data += amplitude * np.sin(2 * np.pi * harmonic_freq * t + phase)
    
    # 포먼트 강조 (Formant emphasis)
    # 포먼트 주파수 주변의 주파수를 강조
    for formant_freq in [formant_f1, formant_f2, formant_f3]:
        # 포먼트 주변의 여러 주파수 성분 추가
        for offset in [-50, -25, 0, 25, 50]:
            freq = formant_freq + offset
            if 50 < freq < sample_rate / 2:
                amplitude = random.uniform(0.1, 0.3)
                phase = random.uniform(0, 2 * np.pi)
                audio_data += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # 시간에 따른 진폭 변조 (Envelope) - 음성이 시작되고 끝나는 자연스러운 패턴
    # 여러 "음절" 또는 "단어" 패턴 생성
    num_segments = random.randint(3, 8)
    segment_length = num_samples // num_segments
    
    envelope = np.ones(num_samples)
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, num_samples)
        segment_len = end_idx - start_idx
        
        # 각 세그먼트에 attack-decay-sustain-release (ADSR) envelope 적용
        attack_len = int(segment_len * 0.1)
        decay_len = int(segment_len * 0.1)
        sustain_len = int(segment_len * 0.6)
        release_len = segment_len - attack_len - decay_len - sustain_len
        
        # Attack
        if attack_len > 0:
            envelope[start_idx:start_idx + attack_len] = np.linspace(0, 1, attack_len)
        # Decay
        if decay_len > 0:
            decay_start = start_idx + attack_len
            envelope[decay_start:decay_start + decay_len] = np.linspace(1, 0.7, decay_len)
        # Sustain
        if sustain_len > 0:
            sustain_start = start_idx + attack_len + decay_len
            envelope[sustain_start:sustain_start + sustain_len] = 0.7 + 0.2 * np.random.random(sustain_len)
        # Release
        if release_len > 0:
            release_start = start_idx + attack_len + decay_len + sustain_len
            envelope[release_start:end_idx] = np.linspace(0.7, 0, release_len)
    
    # Envelope 적용
    audio_data *= envelope
    
    # 기본 주파수의 자연스러운 변조 (Vibrato/Tremolo 효과)
    vibrato_rate = random.uniform(4, 7)  # Hz
    vibrato_depth = random.uniform(0.02, 0.05)  # 주파수 변조 깊이
    f0_modulation = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    
    # 주파수 변조를 적용하기 위해 재생성 (간단한 근사)
    modulated_audio = np.zeros(num_samples)
    for h in range(1, min(5, num_harmonics) + 1):
        harmonic_freq = base_f0 * h * f0_modulation
        amplitude = 0.2 / h
        phase = random.uniform(0, 2 * np.pi)
        modulated_audio += amplitude * np.sin(2 * np.pi * harmonic_freq * t + phase)
    
    # 원본과 변조된 신호를 혼합
    audio_data = 0.7 * audio_data + 0.3 * modulated_audio
    
    # 자연스러운 노이즈 추가 (음성에는 항상 약간의 노이즈가 있음)
    noise = np.random.normal(0, 0.05, num_samples)
    audio_data += noise
    
    # 정규화 (-1.0 ~ 1.0 범위로)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val * 0.8  # 클리핑 방지를 위해 0.8로 제한
    
    # 16-bit PCM으로 변환
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # WAV 파일로 변환
    wav_buffer = io.BytesIO()
    try:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 모노
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        # wave.open이 닫힌 후에 seek
        wav_buffer.seek(0)
        
        # WAV 파일이 제대로 생성되었는지 확인
        if wav_buffer.getvalue() == b'':
            raise ValueError("WAV 파일 생성 실패: 빈 파일")
        
        return wav_buffer
    except Exception as e:
        raise ValueError(f"WAV 파일 생성 중 오류: {e}")


def get_audio_duration(audio_data: io.BytesIO, file_path: Optional[str] = None) -> Optional[float]:
    """
    오디오 데이터의 길이를 초 단위로 반환
    
    Args:
        audio_data: 오디오 데이터 (BytesIO)
        file_path: 파일 경로 (선택사항, 파일명에서 확장자 확인용)
    
    Returns:
        오디오 길이 (초), 측정 불가능한 경우 None
    """
    try:
        audio_data.seek(0)
        
        # 파일 경로가 있으면 확장자로 파일 타입 확인
        if file_path:
            file_ext = os.path.splitext(file_path.lower())[1]
            
            # WAV 파일인 경우
            if file_ext == '.wav':
                try:
                    with wave.open(audio_data, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        duration = frames / float(sample_rate)
                        audio_data.seek(0)
                        return duration
                except Exception as e:
                    audio_data.seek(0)
                    print(f"⚠️ WAV 파일 길이 측정 실패: {e}")
                    return None
            
            # MP3 파일인 경우 - mutagen 라이브러리 사용 시도
            elif file_ext == '.mp3':
                try:
                    from mutagen import File
                    # 파일 경로에서 직접 읽기 (BytesIO가 아닌 실제 파일)
                    audio_file = File(file_path)
                    if audio_file is not None and hasattr(audio_file, 'info') and hasattr(audio_file.info, 'length'):
                        duration = audio_file.info.length
                        audio_data.seek(0)
                        return duration
                except ImportError:
                    # mutagen이 설치되지 않은 경우 조용히 None 반환 (경고는 첫 요청에서만)
                    pass
                except Exception:
                    # 오류 발생 시 조용히 None 반환
                    pass
                audio_data.seek(0)
                return None
        
        # BytesIO에서 직접 WAV 파일인지 확인 (파일 경로가 없는 경우)
        audio_data.seek(0)
        header = audio_data.read(4)
        audio_data.seek(0)
        
        if header == b'RIFF':
            # WAV 파일로 시도
            try:
                with wave.open(audio_data, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / float(sample_rate)
                    audio_data.seek(0)
                    return duration
            except Exception as e:
                audio_data.seek(0)
                print(f"⚠️ WAV 파일 길이 측정 실패: {e}")
                return None
        
        return None
    except Exception as e:
        print(f"⚠️ 오디오 길이 측정 중 오류: {e}")
        return None


def load_audio_from_file(file_path: str) -> io.BytesIO:
    """
    파일에서 오디오 데이터를 읽어서 BytesIO 객체로 반환
    
    Args:
        file_path: 오디오 파일 경로
    
    Returns:
        오디오 데이터를 담은 BytesIO 객체
    """
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)
        # 파일 경로 정보 저장 (오디오 길이 측정용)
        audio_buffer.file_path = file_path
        return audio_buffer
    except FileNotFoundError:
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        raise ValueError(f"오디오 파일 읽기 중 오류: {e}")


def get_all_audio_files(folder_path: str) -> List[str]:
    """
    폴더에서 모든 오디오 파일 목록을 가져옴
    
    Args:
        folder_path: 오디오 파일이 있는 폴더 경로
    
    Returns:
        오디오 파일 전체 경로 리스트 (정렬됨)
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
    
    # 지원하는 오디오 파일 확장자
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma']
    
    # 폴더 내의 모든 오디오 파일 찾기
    audio_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file.lower())
            if ext in audio_extensions:
                audio_files.append(file_path)
    
    if not audio_files:
        raise ValueError(f"폴더에 오디오 파일이 없습니다: {folder_path}")
    
    # 파일명으로 정렬하여 일관된 순서 보장
    audio_files.sort()
    return audio_files


# HTTP STT API 호출 함수
async def http_stt_call(audio_data: io.BytesIO, base_url: str, endpoint: str, filename: str = 'audio.wav'):
    """HTTP STT API 호출"""
    import aiohttp
    
    url = f"{base_url}{endpoint}"
    
    # 파일 확장자에 따른 content_type 결정
    _, ext = os.path.splitext(filename.lower())
    content_type_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.wma': 'audio/x-ms-wma'
    }
    content_type = content_type_map.get(ext, 'audio/wav')
    
    async with aiohttp.ClientSession() as session:
        # BytesIO를 바이트 데이터로 읽기
        audio_data.seek(0)  # 파일 포인터를 처음으로
        audio_bytes = audio_data.read()
        
        data = aiohttp.FormData()
        # 바이트 데이터를 파일로 전송
        data.add_field('file', audio_bytes, filename=filename, content_type=content_type)
        data.add_field('model', '1225')
        # data.add_field('model', 'openai/whisper-large-v3')
        # 언어 설정
        data.add_field('language', 'ko')
        # 필요시 추가 필드 (예: model 등)
        # data.add_field('model', 'whisper-1')
        
        try:
            async with session.post(url, data=data) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        return await response.json()
                    except:
                        # JSON이 아닌 경우 텍스트 반환
                        return {"text": response_text}
                else:
                    raise Exception(f"API 호출 실패 (상태 코드: {response.status}): {response_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"네트워크 오류: {e}")
        except Exception as e:
            raise Exception(f"API 호출 중 오류: {e}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    YAML 설정 파일을 읽어옵니다.
    
    Args:
        config_path: 설정 파일 경로 (기본값: config.yaml)
    
    Returns:
        설정 딕셔너리
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


async def main():
    """메인 함수"""
    # 설정 파일 읽기
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print(f"💡 config.yaml 파일을 생성해주세요.")
        return
    except yaml.YAMLError as e:
        print(f"❌ 오류: YAML 파일 파싱 실패: {e}")
        return
    
    # 설정 값 추출 (기본값 포함)
    concurrent_requests = config.get("concurrent_requests", 5)
    request_delay = config.get("request_delay", 0.0)
    use_random_audio = config.get("use_random_audio", True)
    save_audio_samples = config.get("save_audio_samples", False)
    save_path = config.get("save_path", None)
    base_url = config.get("api", {}).get("base_url", "http://192.168.73.172:8000")
    endpoint = config.get("api", {}).get("endpoint", "/v1/audio/transcriptions")
    
    # 랜덤 오디오 설정 (use_random_audio가 true일 때만 사용)
    random_audio_config = config.get("random_audio", {})
    total_requests = random_audio_config.get("total_requests", 100)
    warmup_requests = random_audio_config.get("warmup_requests", 10)
    audio_duration = random_audio_config.get("audio_duration", 10.0)
    sample_rate = random_audio_config.get("sample_rate", 16000)
    
    # Resource 폴더 설정 (use_random_audio가 false일 때 사용)
    resource_config = config.get("resource", {})
    resource_base_path = resource_config.get("base_path", "resource")
    resource_warmup_folder = resource_config.get("warmup_folder", "warm_up")
    resource_test_folder = resource_config.get("test_folder", "test")
    
    print(f"📁 설정 파일: {config_path}")
    
    # 오디오 소스에 따른 설정 출력 및 함수 생성
    if use_random_audio:
        # 랜덤 오디오 생성 모드일 때만 유효성 검사
        if warmup_requests >= total_requests:
            print("❌ 오류: warmup_requests는 total_requests보다 작아야 합니다.")
            return
        print(f"🎵 오디오 설정: 랜덤 생성 모드")
        print(f"   길이 {audio_duration}초, 샘플링 레이트 {sample_rate}Hz")
        print(f"   매 요청마다 새로운 음성과 유사한 오디오 생성 (캐시 방지)")
        print(f"   (포먼트, 하모닉, 진폭 변조 포함)")
        
        # 랜덤 오디오 생성 함수
        def audio_generator(is_warmup: bool = False):
            """음성과 유사한 오디오 생성 함수 (시간 측정 제외)"""
            return generate_speech_like_audio(
                duration_seconds=audio_duration,
                sample_rate=sample_rate
            )
        
        # 오디오 길이 저장 (RTF 계산용) - 나중에 tester에 설정
        tester_audio_duration = audio_duration
    else:
        # Resource 폴더 경로 구성
        warmup_folder_path = os.path.join(resource_base_path, resource_warmup_folder)
        test_folder_path = os.path.join(resource_base_path, resource_test_folder)
        
        print(f"🎵 오디오 설정: Resource 폴더 사용 모드")
        print(f"   Warm-up 폴더: {warmup_folder_path}")
        print(f"   Test 폴더: {test_folder_path}")
        
        # 폴더 존재 확인 및 파일 목록 로드
        warmup_audio_files = []
        test_audio_files = []
        
        if os.path.exists(warmup_folder_path):
            warmup_audio_files = get_all_audio_files(warmup_folder_path)
            print(f"   Warm-up 파일 수: {len(warmup_audio_files)}개")
        else:
            print(f"⚠️ 경고: Warm-up 폴더를 찾을 수 없습니다: {warmup_folder_path}")
        
        if os.path.exists(test_folder_path):
            test_audio_files = get_all_audio_files(test_folder_path)
            print(f"   Test 파일 수: {len(test_audio_files)}개")
        else:
            print(f"⚠️ 경고: Test 폴더를 찾을 수 없습니다: {test_folder_path}")
        
        if not warmup_audio_files and not test_audio_files:
            print("❌ 오류: 사용 가능한 오디오 파일이 없습니다.")
            return
        
        # Resource 폴더 사용 시 파일 개수에 맞춰 요청 수 자동 조정
        if warmup_audio_files:
            actual_warmup_requests = len(warmup_audio_files)
            if warmup_requests != actual_warmup_requests:
                print(f"ℹ️  Warm-up 요청 수를 파일 개수에 맞춰 조정: {warmup_requests} → {actual_warmup_requests}")
                warmup_requests = actual_warmup_requests
        else:
            warmup_requests = 0
            print(f"ℹ️  Warm-up 폴더가 비어있어 Warm-up 요청 수를 0으로 설정")
        
        if test_audio_files:
            actual_test_requests = len(test_audio_files)
            actual_total_requests = warmup_requests + actual_test_requests
            if total_requests != actual_total_requests:
                print(f"ℹ️  총 요청 수를 파일 개수에 맞춰 조정: {total_requests} → {actual_total_requests}")
                total_requests = actual_total_requests
        else:
            print("❌ 오류: Test 폴더에 오디오 파일이 없습니다.")
            return
        
        # 파일 인덱스를 추적하기 위한 변수 (클로저에서 사용)
        warmup_file_index = [0]  # 리스트로 감싸서 참조 전달
        test_file_index = [0]
        
        # Resource 폴더에서 파일 읽기 함수 (순차적으로 모든 파일 사용)
        def audio_generator(is_warmup: bool = False):
            """Resource 폴더에서 오디오 파일을 순차적으로 읽기 (시간 측정 제외)"""
            if is_warmup:
                if not warmup_audio_files:
                    raise ValueError(f"Warm-up 폴더에 오디오 파일이 없습니다: {warmup_folder_path}")
                # 순환하여 사용 (요청 수가 파일 수보다 많을 경우)
                file_path = warmup_audio_files[warmup_file_index[0] % len(warmup_audio_files)]
                warmup_file_index[0] += 1
            else:
                if not test_audio_files:
                    raise ValueError(f"Test 폴더에 오디오 파일이 없습니다: {test_folder_path}")
                # 순환하여 사용 (요청 수가 파일 수보다 많을 경우)
                file_path = test_audio_files[test_file_index[0] % len(test_audio_files)]
                test_file_index[0] += 1
            
            audio_data = load_audio_from_file(file_path)
            # 파일명 정보를 저장 (나중에 API 호출 시 사용)
            audio_data.filename = os.path.basename(file_path)
            # file_path도 명시적으로 저장 (오디오 길이 측정용)
            audio_data.file_path = file_path
            return audio_data
    
    print(f"🌐 API 설정: {base_url}{endpoint}")
    print()
    
    # API 호출 함수 (오디오 데이터를 인자로 받음)
    async def api_call_func(audio_data: io.BytesIO):
        """STT API 호출 함수 (시간 측정에 포함)"""
        # 파일명이 있으면 사용, 없으면 기본값 사용
        filename = getattr(audio_data, 'filename', 'audio.wav')
        return await http_stt_call(audio_data, base_url, endpoint, filename=filename)
    
    # 테스터 생성 및 실행
    tester = STTLoadTester(
        api_call_func=api_call_func,
        audio_generator_func=audio_generator,
        total_requests=total_requests,
        warmup_requests=warmup_requests,
        concurrent_requests=concurrent_requests,
        request_delay=request_delay,
        save_audio_samples=save_audio_samples
    )
    
    # 오디오 길이 설정 (랜덤 오디오 모드일 때만)
    if use_random_audio:
        tester.audio_duration = audio_duration
    
    metrics = await tester.run()
    tester.print_results(metrics)
    
    # 결과 저장
    tester.save_results(metrics, save_path)


if __name__ == "__main__":
    asyncio.run(main())