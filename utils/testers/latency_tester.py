"""
STT Latency Tester 모듈
"""

import asyncio
import time
import statistics
from typing import List, Optional
from datetime import datetime
import json
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용

from ..models import TestResult, PerformanceMetrics
from ..audio_utils import get_audio_duration


class STTLatencyTester:
    """STT 모델 레이턴시 테스터"""
    
    def __init__(
        self,
        api_call_func,
        audio_generator_func,
        total_requests: int,
        warmup_requests: int,
        concurrent_requests: int = 1,
        request_delay: float = 0.0,
        save_audio_samples: bool = False,
        audio_duration: Optional[float] = None
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
            audio_duration: 랜덤 오디오 생성 시 오디오 길이 (초)
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
        self.audio_duration: Optional[float] = audio_duration  # 랜덤 오디오 생성 시 오디오 길이
    
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
        
        print(f"🚀 STT 모델 레이턴시 테스트 시작")
        print(f"   총 요청 수: {self.total_requests}")
        print(f"   Warm-up 요청 수: {self.warmup_requests}")
        print(f"   동시 요청 수: {self.concurrent_requests}")
        print(f"   실제 측정 요청 수: {self.total_requests - self.warmup_requests}")
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

