"""
STT Load Tester 모듈 - 부하 테스트 전용
"""

import asyncio
import time
import statistics
from typing import List, Optional, Callable, Dict
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
from ..vllm_metrics import VLLMMetricsCollector


class STTLoadTester:
    """STT 모델 부하 테스터 - Ramp-up 및 시간 기반 부하 테스트"""
    
    def __init__(
        self,
        api_call_func: Callable[[io.BytesIO], asyncio.Future],
        audio_generator_func: Callable[[bool], io.BytesIO],
        enable_ramp_up: bool = True,  # Ramp-up 테스트 활성화 여부
        enable_hold: bool = True,  # Hold 테스트 활성화 여부
        ramp_up_duration: float = 60.0,  # Ramp-up 시간 (초)
        hold_duration: float = 300.0,  # 최대 부하 유지 시간 (초)
        max_concurrent_users: int = 10,  # 최대 동시 사용자 수
        ramp_up_steps: int = 5,  # Ramp-up 단계 수
        warmup_requests: int = 5,  # Warm-up 요청 수
        request_delay: float = 0.0,  # 요청 간 지연 시간 (초)
        save_audio_samples: bool = False,
        audio_duration: Optional[float] = None
    ):
        """
        Args:
            api_call_func: STT API를 호출하는 비동기 함수 (audio_data: io.BytesIO를 인자로 받음)
            audio_generator_func: 오디오를 생성하는 함수 (io.BytesIO를 반환)
            enable_ramp_up: Ramp-up 테스트 활성화 여부
            enable_hold: Hold 테스트 활성화 여부
            ramp_up_duration: Ramp-up 단계 소요 시간 (초)
            hold_duration: 최대 부하 유지 시간 (초)
            max_concurrent_users: 최대 동시 사용자 수
            ramp_up_steps: Ramp-up 단계 수 (동시 사용자 수를 몇 단계로 나눌지)
            warmup_requests: Warm-up 요청 수
            request_delay: 요청 간 지연 시간 (초)
            save_audio_samples: 오디오 샘플 저장 여부
            audio_duration: 랜덤 오디오 생성 시 오디오 길이 (초)
        """
        self.api_call_func = api_call_func
        self.audio_generator_func = audio_generator_func
        self.enable_ramp_up = enable_ramp_up
        self.enable_hold = enable_hold
        self.ramp_up_duration = ramp_up_duration
        self.hold_duration = hold_duration
        self.max_concurrent_users = max_concurrent_users
        self.ramp_up_steps = ramp_up_steps
        self.warmup_requests = warmup_requests
        self.request_delay = request_delay
        self.results: List[TestResult] = []  # 전체 성능 테스트 결과 (Ramp-up + Hold)
        self.ramp_up_results: List[TestResult] = []  # Ramp-up 단계 결과
        self.hold_results: List[TestResult] = []  # Hold 단계 결과
        self.warmup_results: List[TestResult] = []  # Warm-up 결과
        self.save_audio_samples: bool = save_audio_samples
        self.saved_audio_count: int = 0
        self.result_dir: str = "result"
        self.timestamp_dir: Optional[str] = None
        self.audio_duration: Optional[float] = audio_duration
        self.request_counter: int = 0  # 요청 카운터
        self.start_time: Optional[float] = None  # 테스트 시작 시간
        self.user_count_rtf_data: List[Dict] = []  # 동시 사용자 수별 RTF 데이터 [{user_count, avg_rtf, ...}]
        self.vllm_metrics_collector: Optional[VLLMMetricsCollector] = None  # vLLM 메트릭 수집기
    
    def _save_audio_sample(self, audio_data: io.BytesIO, request_type: str, request_id: int):
        """오디오 샘플을 파일로 저장"""
        if not self.save_audio_samples:
            return
        
        if request_type == "warmup" and request_id == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_sample_warmup_{timestamp}.wav"
            self._write_audio_file(audio_data, filename)
            self.saved_audio_count += 1
        elif request_type == "load" and request_id == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_sample_load_{timestamp}.wav"
            self._write_audio_file(audio_data, filename)
            self.saved_audio_count += 1
    
    def _ensure_result_dir(self):
        """result 폴더와 타임스탬프 하위 폴더가 없으면 생성"""
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
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
        """단일 요청 실행"""
        audio_data = self.audio_generator_func(is_warmup=is_warmup)
        
        # 오디오 길이 측정
        audio_duration = None
        if self.audio_duration is not None:
            audio_duration = self.audio_duration
        else:
            file_path = getattr(audio_data, 'file_path', None)
            if file_path:
                audio_duration = get_audio_duration(audio_data, file_path)
            else:
                filename = getattr(audio_data, 'filename', None)
                if filename:
                    audio_duration = get_audio_duration(audio_data, filename)
                else:
                    audio_duration = get_audio_duration(audio_data)
        
        # 오디오 샘플 저장
        request_type = "warmup" if is_warmup else "load"
        self._save_audio_sample(audio_data, request_type, request_id)
        
        # API 호출을 위해 오디오 데이터 복사
        audio_data.seek(0)
        audio_bytes = audio_data.read()
        audio_data_copy = io.BytesIO(audio_bytes)
        
        # API 호출 시간 측정
        start_time = time.time()
        try:
            response = await self.api_call_func(audio_data_copy)
            response_time = time.time() - start_time
            
            # STT 예측 텍스트 추출
            text = None
            ttft = None
            if isinstance(response, dict):
                text = response.get("text") or response.get("transcription") or response.get("result")
                # 스트리밍인 경우 TTFT 추출
                ttft = response.get("ttft")
            elif isinstance(response, str):
                text = response
            
            # RTF 계산
            rtf = None
            if audio_duration and audio_duration > 0:
                rtf = response_time / audio_duration
            
            return TestResult(
                response_time=response_time,
                success=True,
                text=text,
                audio_duration=audio_duration,
                rtf=rtf,
                ttft=ttft
            )
        except Exception as e:
            response_time = time.time() - start_time
            
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
    
    async def _run_warmup(self):
        """Warm-up 단계 실행"""
        if self.warmup_requests <= 0:
            return
        
        print(f"🔥 Warm-up 단계 ({self.warmup_requests}개 요청)...")
        warmup_start = time.time()
        
        semaphore = asyncio.Semaphore(self.warmup_requests)
        
        async def warmup_request(request_id: int):
            async with semaphore:
                result = await self._make_request(request_id, is_warmup=True)
                self.warmup_results.append(result)
                return result
        
        tasks = [warmup_request(i) for i in range(self.warmup_requests)]
        await asyncio.gather(*tasks)
        
        warmup_time = time.time() - warmup_start
        print(f"   Warm-up 완료 (소요 시간: {warmup_time:.2f}초)")
        print()
    
    async def _run_continuous_load(
        self,
        duration: float,
        concurrent_users: int,
        phase_name: str
    ):
        """지속적인 부하 생성"""
        semaphore = asyncio.Semaphore(concurrent_users)
        end_time = time.time() + duration
        active_tasks = set()
        
        # 현재 단계의 결과를 저장할 리스트
        phase_results: List[TestResult] = []
        
        async def continuous_request():
            """연속적으로 요청을 생성하는 태스크"""
            while time.time() < end_time:
                async with semaphore:
                    request_id = self.request_counter
                    self.request_counter += 1
                    
                    if self.request_delay > 0:
                        await asyncio.sleep(self.request_delay)
                    
                    result = await self._make_request(request_id, is_warmup=False)
                    # 동시 사용자 수 정보 추가
                    result.concurrent_users = concurrent_users
                    self.results.append(result)
                    phase_results.append(result)
        
        # 동시 사용자 수만큼 태스크 생성
        tasks = [continuous_request() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks)
        
        # 단계별 결과 분류 및 동시 사용자 수별 RTF 데이터 수집
        if phase_name.startswith("ramp_up"):
            self.ramp_up_results.extend(phase_results)
            # 동시 사용자 수별 RTF 데이터 수집
            successful_phase_results = [r for r in phase_results if r.success and r.rtf is not None]
            if successful_phase_results:
                rtf_values = [r.rtf for r in successful_phase_results]
                self.user_count_rtf_data.append({
                    "user_count": concurrent_users,
                    "phase": phase_name,
                    "avg_rtf": statistics.mean(rtf_values),
                    "median_rtf": statistics.median(rtf_values),
                    "min_rtf": min(rtf_values),
                    "max_rtf": max(rtf_values),
                    "request_count": len(successful_phase_results)
                })
        elif phase_name == "hold":
            self.hold_results.extend(phase_results)
            # Hold 단계도 동시 사용자 수별 데이터에 추가
            successful_phase_results = [r for r in phase_results if r.success and r.rtf is not None]
            if successful_phase_results:
                rtf_values = [r.rtf for r in successful_phase_results]
                self.user_count_rtf_data.append({
                    "user_count": concurrent_users,
                    "phase": "hold",
                    "avg_rtf": statistics.mean(rtf_values),
                    "median_rtf": statistics.median(rtf_values),
                    "min_rtf": min(rtf_values),
                    "max_rtf": max(rtf_values),
                    "request_count": len(successful_phase_results)
                })
    
    async def run(self, enable_vllm_metrics: bool = False, vllm_base_url: Optional[str] = None, vllm_metrics_interval: float = 1.0) -> PerformanceMetrics:
        """
        부하 테스트 실행
        
        Args:
            enable_vllm_metrics: vLLM 메트릭 수집 활성화 여부
            vllm_base_url: vLLM 서버 기본 URL
            vllm_metrics_interval: 메트릭 수집 간격 (초)
        """
        self._ensure_result_dir()
        self.start_time = time.time()
        
        # vLLM 메트릭 수집 시작
        if enable_vllm_metrics and vllm_base_url:
            self.vllm_metrics_collector = VLLMMetricsCollector(
                base_url=vllm_base_url,
                metrics_endpoint="/metrics",
                collection_interval=vllm_metrics_interval
            )
            await self.vllm_metrics_collector.start_collecting()
            print("📊 vLLM 메트릭 수집 시작")
        
        print(f"🚀 STT 모델 부하 테스트 시작")
        if self.enable_ramp_up:
            print(f"   Ramp-up: 활성화 (시간: {self.ramp_up_duration}초, 단계: {self.ramp_up_steps})")
        else:
            print(f"   Ramp-up: 비활성화")
        if self.enable_hold:
            print(f"   Hold: 활성화 (시간: {self.hold_duration}초)")
        else:
            print(f"   Hold: 비활성화")
        print(f"   최대 동시 사용자 수: {self.max_concurrent_users}")
        print(f"   결과 저장 경로: {self.timestamp_dir}")
        print()
        
        # Ramp-up과 Hold가 모두 비활성화된 경우 확인
        if not self.enable_ramp_up and not self.enable_hold:
            print("❌ 오류: Ramp-up과 Hold 테스트가 모두 비활성화되어 있습니다. 최소 하나는 활성화해야 합니다.")
            raise ValueError("Ramp-up과 Hold 테스트가 모두 비활성화되어 있습니다.")
        
        # Warm-up 단계
        await self._run_warmup()
        
        # Ramp-up 단계
        if self.enable_ramp_up:
            print(f"📈 Ramp-up 단계 시작...")
            ramp_up_start = time.time()
            
            # 각 단계별 동시 사용자 수 계산
            step_duration = self.ramp_up_duration / self.ramp_up_steps
            for step in range(1, self.ramp_up_steps + 1):
                current_users = int((self.max_concurrent_users / self.ramp_up_steps) * step)
                print(f"   단계 {step}/{self.ramp_up_steps}: 동시 사용자 {current_users}명 ({step_duration:.1f}초)")
                
                await self._run_continuous_load(
                    duration=step_duration,
                    concurrent_users=current_users,
                    phase_name=f"ramp_up_step_{step}"
                )
            
            ramp_up_time = time.time() - ramp_up_start
            print(f"   Ramp-up 완료 (소요 시간: {ramp_up_time:.2f}초, 총 요청: {len(self.results)}개)")
            print()
        else:
            print(f"⏭️  Ramp-up 단계 건너뜀 (비활성화됨)")
            print()
        
        # 최대 부하 유지 단계
        if self.enable_hold:
            print(f"🔥 최대 부하 유지 단계 시작 (동시 사용자 {self.max_concurrent_users}명, {self.hold_duration}초)...")
            hold_start = time.time()
            
            await self._run_continuous_load(
                duration=self.hold_duration,
                concurrent_users=self.max_concurrent_users,
                phase_name="hold"
            )
            
            hold_time = time.time() - hold_start
            print(f"   최대 부하 유지 완료 (소요 시간: {hold_time:.2f}초, 총 요청: {len(self.results)}개)")
            print()
        else:
            print(f"⏭️  Hold 단계 건너뜀 (비활성화됨)")
            print()
        
        total_time = time.time() - self.start_time
        print(f"   전체 테스트 시간: {total_time:.2f}초")
        print()
        
        # vLLM 메트릭 수집 중지
        if self.vllm_metrics_collector:
            await self.vllm_metrics_collector.stop_collecting()
            print("📊 vLLM 메트릭 수집 중지")
            print()
        
        # 메트릭 계산
        return self._calculate_metrics(total_time)
    
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
        
        # RTF 계산
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
        print("📈 부하 테스트 결과")
        print("="*60)
        print(f"총 요청 수: {metrics.total_requests}")
        print(f"  - Ramp-up 단계: {len(self.ramp_up_results)}개")
        print(f"  - Hold 단계: {len(self.hold_results)}개")
        print(f"성공한 요청: {metrics.successful_requests} ({metrics.successful_requests/metrics.total_requests*100:.1f}%)")
        print(f"실패한 요청: {metrics.failed_requests} ({metrics.failed_requests/metrics.total_requests*100:.1f}%)")
        print()
        
        # Ramp-up 단계 통계
        if self.ramp_up_results:
            ramp_up_successful = [r for r in self.ramp_up_results if r.success]
            if ramp_up_successful:
                ramp_up_times = [r.response_time for r in ramp_up_successful]
                ramp_up_rtf = [r.rtf for r in ramp_up_successful if r.rtf is not None]
                ramp_up_ttft = [r.ttft for r in ramp_up_successful if r.ttft is not None]
                print("📈 Ramp-up 단계 통계:")
                print(f"  요청 수: {len(self.ramp_up_results)}개 (성공: {len(ramp_up_successful)}개)")
                print(f"  평균 응답 시간: {statistics.mean(ramp_up_times):.3f}초")
                print(f"  중앙값 응답 시간: {statistics.median(ramp_up_times):.3f}초")
                if ramp_up_rtf:
                    print(f"  평균 RTF: {statistics.mean(ramp_up_rtf):.3f}")
                    print(f"  중앙값 RTF: {statistics.median(ramp_up_rtf):.3f}")
                    print(f"  최소 RTF: {min(ramp_up_rtf):.3f}")
                    print(f"  최대 RTF: {max(ramp_up_rtf):.3f}")
                if ramp_up_ttft:
                    print(f"  평균 TTFT: {statistics.mean(ramp_up_ttft):.3f}초")
                    print(f"  중앙값 TTFT: {statistics.median(ramp_up_ttft):.3f}초")
                    print(f"  최소 TTFT: {min(ramp_up_ttft):.3f}초")
                    print(f"  최대 TTFT: {max(ramp_up_ttft):.3f}초")
                print()
        
        # Hold 단계 통계
        if self.hold_results:
            hold_successful = [r for r in self.hold_results if r.success]
            if hold_successful:
                hold_times = [r.response_time for r in hold_successful]
                hold_rtf = [r.rtf for r in hold_successful if r.rtf is not None]
                hold_ttft = [r.ttft for r in hold_successful if r.ttft is not None]
                print("🔥 Hold 단계 통계:")
                print(f"  요청 수: {len(self.hold_results)}개 (성공: {len(hold_successful)}개)")
                print(f"  평균 응답 시간: {statistics.mean(hold_times):.3f}초")
                print(f"  중앙값 응답 시간: {statistics.median(hold_times):.3f}초")
                if hold_rtf:
                    print(f"  평균 RTF: {statistics.mean(hold_rtf):.3f}")
                    print(f"  중앙값 RTF: {statistics.median(hold_rtf):.3f}")
                    print(f"  최소 RTF: {min(hold_rtf):.3f}")
                    print(f"  최대 RTF: {max(hold_rtf):.3f}")
                if hold_ttft:
                    print(f"  평균 TTFT: {statistics.mean(hold_ttft):.3f}초")
                    print(f"  중앙값 TTFT: {statistics.median(hold_ttft):.3f}초")
                    print(f"  최소 TTFT: {min(hold_ttft):.3f}초")
                    print(f"  최대 TTFT: {max(hold_ttft):.3f}초")
                print()
        
        print("전체 통계:")
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
        print()
        
        # TTFT 통계 (스트리밍인 경우)
        all_ttft = [r.ttft for r in self.results if r.success and r.ttft is not None]
        if all_ttft:
            sorted_ttft = sorted(all_ttft)
            n_ttft = len(sorted_ttft)
            print("TTFT (Time to First Token) 통계:")
            print(f"  평균: {statistics.mean(all_ttft):.3f}초")
            print(f"  중앙값: {statistics.median(sorted_ttft):.3f}초")
            print(f"  최소: {min(all_ttft):.3f}초")
            print(f"  최대: {max(all_ttft):.3f}초")
            print(f"  P95: {sorted_ttft[int(n_ttft * 0.95)] if n_ttft > 0 else 0.0:.3f}초")
            print(f"  P99: {sorted_ttft[int(n_ttft * 0.99)] if n_ttft > 0 else 0.0:.3f}초")
            print()
        
        print(f"처리량: {metrics.requests_per_second:.2f} 요청/초")
        print("="*60)
        
        if metrics.failed_requests > 0:
            print("\n❌ 실패한 요청 상세 (최대 10개):")
            failed_count = 0
            for result in self.results:
                if not result.success and failed_count < 10:
                    print(f"  요청 #{failed_count+1}: {result.error}")
                    failed_count += 1
    
    def save_results(self, metrics: PerformanceMetrics, filename: Optional[str] = None):
        """결과를 JSON 파일로 저장"""
        self._ensure_result_dir()
        
        if filename is None:
            filename = "stt_load_test_results.json"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "enable_ramp_up": self.enable_ramp_up,
                "enable_hold": self.enable_hold,
                "ramp_up_duration": self.ramp_up_duration,
                "hold_duration": self.hold_duration,
                "max_concurrent_users": self.max_concurrent_users,
                "ramp_up_steps": self.ramp_up_steps,
                "warmup_requests": self.warmup_requests,
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
                "requests_per_second": metrics.requests_per_second,
                "avg_rtf": metrics.avg_rtf,
                "min_rtf": metrics.min_rtf,
                "max_rtf": metrics.max_rtf,
                "median_rtf": metrics.median_rtf,
                "p95_rtf": metrics.p95_rtf,
                "p99_rtf": metrics.p99_rtf
            },
            "ramp_up_results": [
                {
                    "response_time": r.response_time,
                    "success": r.success,
                    "error": r.error,
                    "text": r.text,
                    "audio_duration": r.audio_duration,
                    "rtf": r.rtf,
                    "ttft": r.ttft,
                    "concurrent_users": r.concurrent_users
                }
                for r in self.ramp_up_results
            ],
            "hold_results": [
                {
                    "response_time": r.response_time,
                    "success": r.success,
                    "error": r.error,
                    "text": r.text,
                    "audio_duration": r.audio_duration,
                    "rtf": r.rtf,
                    "ttft": r.ttft,
                    "concurrent_users": r.concurrent_users
                }
                for r in self.hold_results
            ],
            "detailed_results": [
                {
                    "response_time": r.response_time,
                    "success": r.success,
                    "error": r.error,
                    "text": r.text,
                    "audio_duration": r.audio_duration,
                    "rtf": r.rtf,
                    "ttft": r.ttft,
                    "concurrent_users": r.concurrent_users
                }
                for r in self.results
            ],
            "user_count_rtf_data": self.user_count_rtf_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과가 {filepath}에 저장되었습니다.")
        
        # 그래프 저장
        self.save_timeline_graph()
        self.save_histogram()
        
        # 동시 사용자 수별 RTF 및 TTFT 추이 그래프 저장 (하나의 figure로 통합)
        has_ttft = any(r.ttft is not None for r in self.results if r.success)
        if self.user_count_rtf_data or has_ttft:
            self.save_user_count_combined_graph(has_ttft=has_ttft)
        
        # vLLM 메트릭 그래프 저장
        if self.vllm_metrics_collector and self.vllm_metrics_collector.metrics_history:
            self.save_vllm_metrics_graph()
    
    def save_timeline_graph(self, filename: Optional[str] = None):
        """시간대별 응답 시간 및 RTF 추이 그래프 저장"""
        if not self.results:
            print("⚠️ 성공한 요청이 없어 타임라인 그래프를 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "load_test_timeline.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # TTFT 데이터가 있는지 확인
        has_ttft = any(r.ttft is not None for r in self.results if r.success)
        
        # TTFT가 있으면 3개 서브플롯, 없으면 2개 서브플롯
        if has_ttft:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 시간대별 데이터 수집 (Ramp-up과 Hold 구분)
        ramp_up_successful = [r for r in self.ramp_up_results if r.success]
        hold_successful = [r for r in self.hold_results if r.success]
        
        if not ramp_up_successful and not hold_successful:
            return
        
        # Ramp-up 단계 데이터
        ramp_up_elapsed_times = []
        ramp_up_response_times = []
        ramp_up_rtf_values = []
        ramp_up_ttft_values = []
        
        ramp_up_start_offset = 0  # Ramp-up은 0부터 시작
        for i, result in enumerate(ramp_up_successful):
            elapsed_time = (i / len(ramp_up_successful)) * self.ramp_up_duration if len(ramp_up_successful) > 0 else 0
            ramp_up_elapsed_times.append(elapsed_time)
            ramp_up_response_times.append(result.response_time)
            if result.rtf is not None:
                ramp_up_rtf_values.append((elapsed_time, result.rtf))
            if result.ttft is not None:
                ramp_up_ttft_values.append((elapsed_time, result.ttft))
        
        # Hold 단계 데이터
        hold_elapsed_times = []
        hold_response_times = []
        hold_rtf_values = []
        hold_ttft_values = []
        
        hold_start_offset = self.ramp_up_duration  # Hold는 Ramp-up 이후 시작
        for i, result in enumerate(hold_successful):
            elapsed_time = hold_start_offset + (i / len(hold_successful)) * self.hold_duration if len(hold_successful) > 0 else hold_start_offset
            hold_elapsed_times.append(elapsed_time)
            hold_response_times.append(result.response_time)
            if result.rtf is not None:
                hold_rtf_values.append((elapsed_time, result.rtf))
            if result.ttft is not None:
                hold_ttft_values.append((elapsed_time, result.ttft))
        
        # === 위쪽: 응답 시간 타임라인 ===
        # Ramp-up 단계 플롯
        if ramp_up_elapsed_times:
            ax1.scatter(ramp_up_elapsed_times, ramp_up_response_times, alpha=0.5, s=10, color='orange', label=f'Ramp-up ({len(ramp_up_successful)} requests)')
        
        # Hold 단계 플롯
        if hold_elapsed_times:
            ax1.scatter(hold_elapsed_times, hold_response_times, alpha=0.5, s=10, color='steelblue', label=f'Hold ({len(hold_successful)} requests)')
        
        # Ramp-up 이동 평균
        if len(ramp_up_elapsed_times) > 10:
            window_size = max(10, len(ramp_up_elapsed_times) // 20)
            moving_avg = []
            moving_avg_times = []
            for i in range(window_size, len(ramp_up_elapsed_times)):
                window_times = ramp_up_elapsed_times[i-window_size:i]
                window_values = ramp_up_response_times[i-window_size:i]
                moving_avg.append(statistics.mean(window_values))
                moving_avg_times.append(statistics.mean(window_times))
            
            ax1.plot(moving_avg_times, moving_avg, color='red', linewidth=2, alpha=0.7, label='Ramp-up Moving Avg')
        
        # Hold 이동 평균
        if len(hold_elapsed_times) > 10:
            window_size = max(10, len(hold_elapsed_times) // 20)
            moving_avg = []
            moving_avg_times = []
            for i in range(window_size, len(hold_elapsed_times)):
                window_times = hold_elapsed_times[i-window_size:i]
                window_values = hold_response_times[i-window_size:i]
                moving_avg.append(statistics.mean(window_values))
                moving_avg_times.append(statistics.mean(window_times))
            
            ax1.plot(moving_avg_times, moving_avg, color='blue', linewidth=2, alpha=0.7, label='Hold Moving Avg')
        
        # Ramp-up 응답 시간 평균선
        if ramp_up_response_times:
            ramp_up_avg_response = statistics.mean(ramp_up_response_times)
            ax1.axhline(ramp_up_avg_response, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Ramp-up Avg: {ramp_up_avg_response:.3f}s')
        
        # Hold 응답 시간 평균선
        if hold_response_times:
            hold_avg_response = statistics.mean(hold_response_times)
            ax1.axhline(hold_avg_response, color='steelblue', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Hold Avg: {hold_avg_response:.3f}s')
        
        # 전체 평균선
        all_response_times = ramp_up_response_times + hold_response_times
        if all_response_times:
            avg_response = statistics.mean(all_response_times)
            ax1.axhline(avg_response, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'Overall Average: {avg_response:.3f}s')
        
        # Ramp-up과 Hold 단계 경계선
        if self.enable_ramp_up and self.enable_hold:
            ax1.axvline(self.ramp_up_duration, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                       label='Ramp-up / Hold Boundary')
        
        ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12)
        ax1.set_ylabel('Response Time (seconds)', fontsize=12)
        ax1.set_title('Response Time Timeline (Load Test)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # y축 범위를 데이터에 맞게 동적으로 조정
        if all_response_times:
            min_time = min(all_response_times)
            max_time = max(all_response_times)
            # 여유 공간을 위해 10% 마진 추가
            time_range = max_time - min_time
            if time_range > 0:
                y_margin = time_range * 0.1
                ax1.set_ylim(max(0, min_time - y_margin), max_time + y_margin)
            else:
                # 범위가 0인 경우 (모든 값이 같음) 약간의 여유 공간 추가
                ax1.set_ylim(max(0, min_time - 0.1), max_time + 0.1)
        
        # === 아래쪽: RTF 타임라인 ===
        # Ramp-up RTF 플롯
        if ramp_up_rtf_values:
            ramp_up_rtf_times, ramp_up_rtf_vals = zip(*ramp_up_rtf_values)
            ax2.scatter(ramp_up_rtf_times, ramp_up_rtf_vals, alpha=0.5, s=10, color='orange', label=f'Ramp-up ({len(ramp_up_rtf_values)} requests)')
            
            # Ramp-up RTF 이동 평균
            if len(ramp_up_rtf_values) > 10:
                window_size = max(10, len(ramp_up_rtf_values) // 20)
                moving_avg_rtf = []
                moving_avg_rtf_times = []
                rtf_list = list(ramp_up_rtf_values)
                for i in range(window_size, len(rtf_list)):
                    window_times = [t for t, _ in rtf_list[i-window_size:i]]
                    window_values = [v for _, v in rtf_list[i-window_size:i]]
                    moving_avg_rtf.append(statistics.mean(window_values))
                    moving_avg_rtf_times.append(statistics.mean(window_times))
                
                ax2.plot(moving_avg_rtf_times, moving_avg_rtf, color='red', linewidth=2, alpha=0.7, label='Ramp-up RTF Moving Avg')
            
            # Ramp-up RTF 평균선
            ramp_up_avg_rtf = statistics.mean(ramp_up_rtf_vals)
            ax2.axhline(ramp_up_avg_rtf, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Ramp-up Avg RTF: {ramp_up_avg_rtf:.3f}')
        
        # Hold RTF 플롯
        if hold_rtf_values:
            hold_rtf_times, hold_rtf_vals = zip(*hold_rtf_values)
            ax2.scatter(hold_rtf_times, hold_rtf_vals, alpha=0.5, s=10, color='steelblue', label=f'Hold ({len(hold_rtf_values)} requests)')
            
            # Hold RTF 이동 평균
            if len(hold_rtf_values) > 10:
                window_size = max(10, len(hold_rtf_values) // 20)
                moving_avg_rtf = []
                moving_avg_rtf_times = []
                rtf_list = list(hold_rtf_values)
                for i in range(window_size, len(rtf_list)):
                    window_times = [t for t, _ in rtf_list[i-window_size:i]]
                    window_values = [v for _, v in rtf_list[i-window_size:i]]
                    moving_avg_rtf.append(statistics.mean(window_values))
                    moving_avg_rtf_times.append(statistics.mean(window_times))
                
                ax2.plot(moving_avg_rtf_times, moving_avg_rtf, color='blue', linewidth=2, alpha=0.7, label='Hold RTF Moving Avg')
            
            # Hold RTF 평균선
            hold_avg_rtf = statistics.mean(hold_rtf_vals)
            ax2.axhline(hold_avg_rtf, color='steelblue', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Hold Avg RTF: {hold_avg_rtf:.3f}')
        
        # 전체 RTF 평균선
        all_rtf_vals = []
        if ramp_up_rtf_values:
            all_rtf_vals.extend([v for _, v in ramp_up_rtf_values])
        if hold_rtf_values:
            all_rtf_vals.extend([v for _, v in hold_rtf_values])
        
        if all_rtf_vals:
            avg_rtf = statistics.mean(all_rtf_vals)
            ax2.axhline(avg_rtf, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'Overall Average RTF: {avg_rtf:.3f}')
        
        # Ramp-up과 Hold 단계 경계선
        if self.enable_ramp_up and self.enable_hold:
            ax2.axvline(self.ramp_up_duration, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                       label='Ramp-up / Hold Boundary')
        
        ax2.set_xlabel('Elapsed Time (seconds)', fontsize=12)
        ax2.set_ylabel('RTF (Real-Time Factor)', fontsize=12)
        ax2.set_title('RTF Timeline (Load Test)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # y축 범위를 데이터에 맞게 동적으로 조정
        if all_rtf_vals:
            min_rtf = min(all_rtf_vals)
            max_rtf = max(all_rtf_vals)
            # 여유 공간을 위해 10% 마진 추가
            rtf_range = max_rtf - min_rtf
            if rtf_range > 0:
                y_margin = rtf_range * 0.1
                ax2.set_ylim(min_rtf - y_margin, max_rtf + y_margin)
            else:
                # 범위가 0인 경우 (모든 값이 같음) 약간의 여유 공간 추가
                ax2.set_ylim(min_rtf - 0.1, max_rtf + 0.1)
        
        # === TTFT 타임라인 (TTFT 데이터가 있는 경우) ===
        if has_ttft:
            # Ramp-up TTFT 플롯
            if ramp_up_ttft_values:
                ramp_up_ttft_times, ramp_up_ttft_vals = zip(*ramp_up_ttft_values)
                ax3.scatter(ramp_up_ttft_times, ramp_up_ttft_vals, alpha=0.5, s=10, color='orange', label=f'Ramp-up ({len(ramp_up_ttft_values)} requests)')
                
                # Ramp-up TTFT 이동 평균
                if len(ramp_up_ttft_values) > 10:
                    window_size = max(10, len(ramp_up_ttft_values) // 20)
                    moving_avg_ttft = []
                    moving_avg_ttft_times = []
                    ttft_list = list(ramp_up_ttft_values)
                    for i in range(window_size, len(ttft_list)):
                        window_times = [t for t, _ in ttft_list[i-window_size:i]]
                        window_values = [v for _, v in ttft_list[i-window_size:i]]
                        moving_avg_ttft.append(statistics.mean(window_values))
                        moving_avg_ttft_times.append(statistics.mean(window_times))
                    
                    ax3.plot(moving_avg_ttft_times, moving_avg_ttft, color='red', linewidth=2, alpha=0.7, label='Ramp-up TTFT Moving Avg')
                
                # Ramp-up TTFT 평균선
                ramp_up_avg_ttft = statistics.mean(ramp_up_ttft_vals)
                ax3.axhline(ramp_up_avg_ttft, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                           label=f'Ramp-up Avg TTFT: {ramp_up_avg_ttft:.3f}s')
            
            # Hold TTFT 플롯
            if hold_ttft_values:
                hold_ttft_times, hold_ttft_vals = zip(*hold_ttft_values)
                ax3.scatter(hold_ttft_times, hold_ttft_vals, alpha=0.5, s=10, color='steelblue', label=f'Hold ({len(hold_ttft_values)} requests)')
                
                # Hold TTFT 이동 평균
                if len(hold_ttft_values) > 10:
                    window_size = max(10, len(hold_ttft_values) // 20)
                    moving_avg_ttft = []
                    moving_avg_ttft_times = []
                    ttft_list = list(hold_ttft_values)
                    for i in range(window_size, len(ttft_list)):
                        window_times = [t for t, _ in ttft_list[i-window_size:i]]
                        window_values = [v for _, v in ttft_list[i-window_size:i]]
                        moving_avg_ttft.append(statistics.mean(window_values))
                        moving_avg_ttft_times.append(statistics.mean(window_times))
                    
                    ax3.plot(moving_avg_ttft_times, moving_avg_ttft, color='blue', linewidth=2, alpha=0.7, label='Hold TTFT Moving Avg')
                
                # Hold TTFT 평균선
                hold_avg_ttft = statistics.mean(hold_ttft_vals)
                ax3.axhline(hold_avg_ttft, color='steelblue', linestyle='--', linewidth=2, alpha=0.8,
                           label=f'Hold Avg TTFT: {hold_avg_ttft:.3f}s')
            
            # 전체 TTFT 평균선
            all_ttft_vals = []
            if ramp_up_ttft_values:
                all_ttft_vals.extend([v for _, v in ramp_up_ttft_values])
            if hold_ttft_values:
                all_ttft_vals.extend([v for _, v in hold_ttft_values])
            
            if all_ttft_vals:
                avg_ttft = statistics.mean(all_ttft_vals)
                ax3.axhline(avg_ttft, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                           label=f'Overall Average TTFT: {avg_ttft:.3f}s')
            
            # Ramp-up과 Hold 단계 경계선
            if self.enable_ramp_up and self.enable_hold:
                ax3.axvline(self.ramp_up_duration, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                           label='Ramp-up / Hold Boundary')
            
            ax3.set_xlabel('Elapsed Time (seconds)', fontsize=12)
            ax3.set_ylabel('TTFT (Time to First Token) (seconds)', fontsize=12)
            ax3.set_title('TTFT Timeline (Load Test)', fontsize=13, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # y축 범위를 데이터에 맞게 동적으로 조정
            if all_ttft_vals:
                min_ttft = min(all_ttft_vals)
                max_ttft = max(all_ttft_vals)
                ttft_range = max_ttft - min_ttft
                if ttft_range > 0:
                    y_margin = ttft_range * 0.1
                    ax3.set_ylim(max(0, min_ttft - y_margin), max_ttft + y_margin)
                else:
                    ax3.set_ylim(max(0, min_ttft - 0.1), max_ttft + 0.1)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 타임라인 그래프가 {filepath}에 저장되었습니다.")
    
    def save_histogram(self, filename: Optional[str] = None):
        """응답 시간 및 RTF 히스토그램 저장 (동시 사용자 수별 색상 구분)"""
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("⚠️ 성공한 요청이 없어 히스토그램을 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "load_test_histogram.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # 동시 사용자 수별로 그룹화
        user_groups: Dict[int, List[TestResult]] = {}
        for result in successful_results:
            user_count = result.concurrent_users if result.concurrent_users is not None else 0
            if user_count not in user_groups:
                user_groups[user_count] = []
            user_groups[user_count].append(result)
        
        # 동시 사용자 수별 색상 맵 생성 (더 많은 사용자일수록 진한 색)
        sorted_user_counts = sorted(user_groups.keys())
        if sorted_user_counts:
            max_users = max(sorted_user_counts)
            min_users = min(sorted_user_counts)
            user_range = max_users - min_users if max_users != min_users else 1
            
            # 색상 팔레트 (파란색 계열에서 빨간색 계열로)
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_user_counts)))
        
        # === 위쪽: 응답 시간 히스토그램 ===
        all_response_times = [r.response_time for r in successful_results]
        if all_response_times:
            # 전체 범위로 bins 설정
            min_time = min(all_response_times)
            max_time = max(all_response_times)
            bins = np.linspace(min_time, max_time, 50)
            
            # 각 동시 사용자 수별로 히스토그램 그리기
            for idx, user_count in enumerate(sorted_user_counts):
                group_results = user_groups[user_count]
                group_times = [r.response_time for r in group_results]
                if group_times:
                    color = colors[idx] if len(sorted_user_counts) > 1 else 'steelblue'
                    ax1.hist(
                        group_times,
                        bins=bins,
                        edgecolor='black',
                        alpha=0.6,
                        color=color,
                        label=f'{user_count} users ({len(group_results)} requests)',
                        histtype='stepfilled'
                    )
            
            # 전체 평균 및 중앙값 선
            avg_time = statistics.mean(all_response_times)
            median_time = statistics.median(all_response_times)
            ax1.axvline(avg_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Overall Avg: {avg_time:.3f}s')
            ax1.axvline(median_time, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Overall Median: {median_time:.3f}s')
        
        ax1.set_xlabel('Response Time (seconds)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Response Time Histogram by Concurrent Users (Load Test)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # === 아래쪽: RTF 히스토그램 ===
        rtf_results = [r for r in successful_results if r.rtf is not None]
        if rtf_results:
            all_rtf_values = [r.rtf for r in rtf_results]
            min_rtf = min(all_rtf_values)
            max_rtf = max(all_rtf_values)
            rtf_range = max_rtf - min_rtf
            bins_rtf = np.linspace(min_rtf, max_rtf, 50) if rtf_range > 0 else 50
            
            # 각 동시 사용자 수별로 RTF 히스토그램 그리기
            for idx, user_count in enumerate(sorted_user_counts):
                group_results = [r for r in user_groups[user_count] if r.rtf is not None]
                group_rtf = [r.rtf for r in group_results]
                if group_rtf:
                    color = colors[idx] if len(sorted_user_counts) > 1 else 'steelblue'
                    ax2.hist(
                        group_rtf,
                        bins=bins_rtf,
                        edgecolor='black',
                        alpha=0.6,
                        color=color,
                        label=f'{user_count} users ({len(group_results)} requests)',
                        histtype='stepfilled'
                    )
            
            # 전체 평균 및 중앙값 선
            avg_rtf = statistics.mean(all_rtf_values)
            median_rtf = statistics.median(all_rtf_values)
            ax2.axvline(avg_rtf, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Overall Avg RTF: {avg_rtf:.3f}')
            ax2.axvline(median_rtf, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Overall Median RTF: {median_rtf:.3f}')
            
            # x축 범위를 데이터에 맞게 동적으로 조정
            if rtf_range > 0:
                x_margin = rtf_range * 0.1
                ax2.set_xlim(max(0, min_rtf - x_margin), max_rtf + x_margin)
            else:
                ax2.set_xlim(max(0, min_rtf - 0.1), max_rtf + 0.1)
        
        ax2.set_xlabel('RTF (Real-Time Factor)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('RTF Histogram by Concurrent Users (Load Test)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 히스토그램이 {filepath}에 저장되었습니다.")
    
    def save_user_count_rtf_graph(self, filename: Optional[str] = None):
        """동시 사용자 수별 RTF 추이 그래프 저장"""
        if not self.user_count_rtf_data:
            print("⚠️ 동시 사용자 수별 RTF 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "user_count_rtf_trend.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 데이터 준비
        user_counts = [d["user_count"] for d in self.user_count_rtf_data]
        avg_rtfs = [d["avg_rtf"] for d in self.user_count_rtf_data]
        median_rtfs = [d["median_rtf"] for d in self.user_count_rtf_data]
        min_rtfs = [d["min_rtf"] for d in self.user_count_rtf_data]
        max_rtfs = [d["max_rtf"] for d in self.user_count_rtf_data]
        phases = [d["phase"] for d in self.user_count_rtf_data]
        
        # Ramp-up과 Hold 구분
        ramp_up_indices = [i for i, phase in enumerate(phases) if phase.startswith("ramp_up")]
        hold_indices = [i for i, phase in enumerate(phases) if phase == "hold"]
        
        # 그래프 생성
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Ramp-up 단계 플롯
        if ramp_up_indices:
            ramp_up_users = [user_counts[i] for i in ramp_up_indices]
            ramp_up_avg_rtfs = [avg_rtfs[i] for i in ramp_up_indices]
            ramp_up_median_rtfs = [median_rtfs[i] for i in ramp_up_indices]
            ramp_up_min_rtfs = [min_rtfs[i] for i in ramp_up_indices]
            ramp_up_max_rtfs = [max_rtfs[i] for i in ramp_up_indices]
            
            # 평균 RTF 선
            ax.plot(ramp_up_users, ramp_up_avg_rtfs, 'o-', color='orange', linewidth=2.5, 
                   markersize=10, label='Ramp-up Average RTF', alpha=0.8)
            # 중앙값 RTF 선
            ax.plot(ramp_up_users, ramp_up_median_rtfs, 's--', color='orange', linewidth=2, 
                   markersize=8, label='Ramp-up Median RTF', alpha=0.7)
            # Min-Max 범위
            ax.fill_between(ramp_up_users, ramp_up_min_rtfs, ramp_up_max_rtfs, 
                           alpha=0.2, color='orange', label='Ramp-up Min-Max Range')
        
        # Hold 단계 플롯
        if hold_indices:
            hold_users = [user_counts[i] for i in hold_indices]
            hold_avg_rtfs = [avg_rtfs[i] for i in hold_indices]
            hold_median_rtfs = [median_rtfs[i] for i in hold_indices]
            hold_min_rtfs = [min_rtfs[i] for i in hold_indices]
            hold_max_rtfs = [max_rtfs[i] for i in hold_indices]
            
            # 평균 RTF 선
            ax.plot(hold_users, hold_avg_rtfs, 'o-', color='steelblue', linewidth=2.5, 
                   markersize=10, label='Hold Average RTF', alpha=0.8)
            # 중앙값 RTF 선
            ax.plot(hold_users, hold_median_rtfs, 's--', color='steelblue', linewidth=2, 
                   markersize=8, label='Hold Median RTF', alpha=0.7)
            # Min-Max 범위
            ax.fill_between(hold_users, hold_min_rtfs, hold_max_rtfs, 
                           alpha=0.2, color='steelblue', label='Hold Min-Max Range')
        
        # 전체 평균선
        if avg_rtfs:
            overall_avg = statistics.mean(avg_rtfs)
            ax.axhline(overall_avg, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=f'Overall Average RTF: {overall_avg:.3f}')
        
        # RTF = 1.0 기준선 (실시간 처리 기준)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                  label='RTF = 1.0 (Real-time)')
        
        ax.set_xlabel('Concurrent Users', fontsize=12)
        ax.set_ylabel('RTF (Real-Time Factor)', fontsize=12)
        ax.set_title('RTF Trend by Concurrent Users', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # y축 범위를 데이터에 맞게 동적으로 조정
        if avg_rtfs:
            min_rtf = min(min_rtfs)
            max_rtf = max(max_rtfs)
            rtf_range = max_rtf - min_rtf
            if rtf_range > 0:
                y_margin = rtf_range * 0.1
                ax.set_ylim(max(0, min_rtf - y_margin), max_rtf + y_margin)
            else:
                ax.set_ylim(max(0, min_rtf - 0.1), max_rtf + 0.1)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 동시 사용자 수별 RTF 추이 그래프가 {filepath}에 저장되었습니다.")
    
    def save_user_count_combined_graph(self, has_ttft: bool = False, filename: Optional[str] = None):
        """동시 사용자 수별 RTF 및 TTFT 추이 그래프 저장 (하나의 figure로 통합)"""
        if not self.user_count_rtf_data and not has_ttft:
            print("⚠️ RTF 및 TTFT 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "user_count_rtf_ttft_trend.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 서브플롯 개수 결정 (TTFT가 있으면 2개, 없으면 1개)
        if has_ttft:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        
        # === 위쪽: RTF 그래프 ===
        if self.user_count_rtf_data:
            # 데이터 준비
            user_counts = [d["user_count"] for d in self.user_count_rtf_data]
            avg_rtfs = [d["avg_rtf"] for d in self.user_count_rtf_data]
            median_rtfs = [d["median_rtf"] for d in self.user_count_rtf_data]
            min_rtfs = [d["min_rtf"] for d in self.user_count_rtf_data]
            max_rtfs = [d["max_rtf"] for d in self.user_count_rtf_data]
            phases = [d["phase"] for d in self.user_count_rtf_data]
            
            # Ramp-up과 Hold 구분
            ramp_up_indices = [i for i, phase in enumerate(phases) if phase.startswith("ramp_up")]
            hold_indices = [i for i, phase in enumerate(phases) if phase == "hold"]
            
            # Ramp-up 단계 플롯
            if ramp_up_indices:
                ramp_up_users = [user_counts[i] for i in ramp_up_indices]
                ramp_up_avg_rtfs = [avg_rtfs[i] for i in ramp_up_indices]
                ramp_up_median_rtfs = [median_rtfs[i] for i in ramp_up_indices]
                ramp_up_min_rtfs = [min_rtfs[i] for i in ramp_up_indices]
                ramp_up_max_rtfs = [max_rtfs[i] for i in ramp_up_indices]
                
                # 평균 RTF 선
                ax1.plot(ramp_up_users, ramp_up_avg_rtfs, 'o-', color='orange', linewidth=2.5, 
                       markersize=10, label='Ramp-up Average RTF', alpha=0.8)
                # 중앙값 RTF 선
                ax1.plot(ramp_up_users, ramp_up_median_rtfs, 's--', color='orange', linewidth=2, 
                       markersize=8, label='Ramp-up Median RTF', alpha=0.7)
                # Min-Max 범위
                ax1.fill_between(ramp_up_users, ramp_up_min_rtfs, ramp_up_max_rtfs, 
                               alpha=0.2, color='orange', label='Ramp-up Min-Max Range')
            
            # Hold 단계 플롯
            if hold_indices:
                hold_users = [user_counts[i] for i in hold_indices]
                hold_avg_rtfs = [avg_rtfs[i] for i in hold_indices]
                hold_median_rtfs = [median_rtfs[i] for i in hold_indices]
                hold_min_rtfs = [min_rtfs[i] for i in hold_indices]
                hold_max_rtfs = [max_rtfs[i] for i in hold_indices]
                
                # 평균 RTF 선
                ax1.plot(hold_users, hold_avg_rtfs, 'o-', color='steelblue', linewidth=2.5, 
                       markersize=10, label='Hold Average RTF', alpha=0.8)
                # 중앙값 RTF 선
                ax1.plot(hold_users, hold_median_rtfs, 's--', color='steelblue', linewidth=2, 
                       markersize=8, label='Hold Median RTF', alpha=0.7)
                # Min-Max 범위
                ax1.fill_between(hold_users, hold_min_rtfs, hold_max_rtfs, 
                               alpha=0.2, color='steelblue', label='Hold Min-Max Range')
            
            # 전체 평균선
            if avg_rtfs:
                overall_avg = statistics.mean(avg_rtfs)
                ax1.axhline(overall_avg, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                          label=f'Overall Average RTF: {overall_avg:.3f}')
            
            # RTF = 1.0 기준선 (실시간 처리 기준)
            ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                      label='RTF = 1.0 (Real-time)')
            
            ax1.set_xlabel('Concurrent Users', fontsize=12)
            ax1.set_ylabel('RTF (Real-Time Factor)', fontsize=12)
            ax1.set_title('RTF Trend by Concurrent Users', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)
            
            # y축 범위를 데이터에 맞게 동적으로 조정
            if avg_rtfs:
                min_rtf = min(min_rtfs)
                max_rtf = max(max_rtfs)
                rtf_range = max_rtf - min_rtf
                if rtf_range > 0:
                    y_margin = rtf_range * 0.1
                    ax1.set_ylim(max(0, min_rtf - y_margin), max_rtf + y_margin)
                else:
                    ax1.set_ylim(max(0, min_rtf - 0.1), max_rtf + 0.1)
        
        # === 아래쪽: TTFT 그래프 (TTFT 데이터가 있는 경우) ===
        if has_ttft:
            # 동시 사용자 수별로 그룹화 (Ramp-up과 Hold 구분)
            ramp_up_ttft_data: Dict[int, List[float]] = {}
            hold_ttft_data: Dict[int, List[float]] = {}
            
            # Ramp-up 결과에서 TTFT 수집
            for result in self.ramp_up_results:
                if result.success and result.ttft is not None:
                    user_count = result.concurrent_users if result.concurrent_users is not None else 0
                    if user_count not in ramp_up_ttft_data:
                        ramp_up_ttft_data[user_count] = []
                    ramp_up_ttft_data[user_count].append(result.ttft)
            
            # Hold 결과에서 TTFT 수집
            for result in self.hold_results:
                if result.success and result.ttft is not None:
                    user_count = result.concurrent_users if result.concurrent_users is not None else 0
                    if user_count not in hold_ttft_data:
                        hold_ttft_data[user_count] = []
                    hold_ttft_data[user_count].append(result.ttft)
            
            # Ramp-up 데이터 정리
            ramp_up_sorted_users = sorted(ramp_up_ttft_data.keys()) if ramp_up_ttft_data else []
            ramp_up_avg_ttfts = []
            ramp_up_median_ttfts = []
            ramp_up_min_ttfts = []
            ramp_up_max_ttfts = []
            
            for user_count in ramp_up_sorted_users:
                ttft_values = ramp_up_ttft_data[user_count]
                if ttft_values:
                    ramp_up_avg_ttfts.append(statistics.mean(ttft_values))
                    ramp_up_median_ttfts.append(statistics.median(ttft_values))
                    ramp_up_min_ttfts.append(min(ttft_values))
                    ramp_up_max_ttfts.append(max(ttft_values))
            
            # Hold 데이터 정리
            hold_sorted_users = sorted(hold_ttft_data.keys()) if hold_ttft_data else []
            hold_avg_ttfts = []
            hold_median_ttfts = []
            hold_min_ttfts = []
            hold_max_ttfts = []
            
            for user_count in hold_sorted_users:
                ttft_values = hold_ttft_data[user_count]
                if ttft_values:
                    hold_avg_ttfts.append(statistics.mean(ttft_values))
                    hold_median_ttfts.append(statistics.median(ttft_values))
                    hold_min_ttfts.append(min(ttft_values))
                    hold_max_ttfts.append(max(ttft_values))
            
            # Ramp-up 단계 플롯
            if ramp_up_sorted_users:
                # 평균 TTFT 선
                ax2.plot(ramp_up_sorted_users, ramp_up_avg_ttfts, 'o-', color='orange', linewidth=2.5, 
                       markersize=10, label='Ramp-up Average TTFT', alpha=0.8)
                # 중앙값 TTFT 선
                ax2.plot(ramp_up_sorted_users, ramp_up_median_ttfts, 's--', color='orange', linewidth=2, 
                       markersize=8, label='Ramp-up Median TTFT', alpha=0.7)
                # Min-Max 범위
                ax2.fill_between(ramp_up_sorted_users, ramp_up_min_ttfts, ramp_up_max_ttfts, 
                               alpha=0.2, color='orange', label='Ramp-up Min-Max Range')
            
            # Hold 단계 플롯
            if hold_sorted_users:
                # 평균 TTFT 선
                ax2.plot(hold_sorted_users, hold_avg_ttfts, 'o-', color='steelblue', linewidth=2.5, 
                       markersize=10, label='Hold Average TTFT', alpha=0.8)
                # 중앙값 TTFT 선
                ax2.plot(hold_sorted_users, hold_median_ttfts, 's--', color='steelblue', linewidth=2, 
                       markersize=8, label='Hold Median TTFT', alpha=0.7)
                # Min-Max 범위
                ax2.fill_between(hold_sorted_users, hold_min_ttfts, hold_max_ttfts, 
                               alpha=0.2, color='steelblue', label='Hold Min-Max Range')
            
            # 전체 평균선
            all_ttft_values = []
            if ramp_up_avg_ttfts:
                all_ttft_values.extend(ramp_up_avg_ttfts)
            if hold_avg_ttfts:
                all_ttft_values.extend(hold_avg_ttfts)
            
            if all_ttft_values:
                overall_avg = statistics.mean(all_ttft_values)
                ax2.axhline(overall_avg, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                          label=f'Overall Average TTFT: {overall_avg:.3f}s')
            
            ax2.set_xlabel('Concurrent Users', fontsize=12)
            ax2.set_ylabel('TTFT (Time to First Token) (seconds)', fontsize=12)
            ax2.set_title('TTFT Trend by Concurrent Users', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)
            
            # y축 범위를 데이터에 맞게 동적으로 조정
            all_min_ttfts = ramp_up_min_ttfts + hold_min_ttfts
            all_max_ttfts = ramp_up_max_ttfts + hold_max_ttfts
            if all_min_ttfts and all_max_ttfts:
                min_ttft = min(all_min_ttfts)
                max_ttft = max(all_max_ttfts)
                ttft_range = max_ttft - min_ttft
                if ttft_range > 0:
                    y_margin = ttft_range * 0.1
                    ax2.set_ylim(max(0, min_ttft - y_margin), max_ttft + y_margin)
                else:
                    ax2.set_ylim(max(0, min_ttft - 0.1), max_ttft + 0.1)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 동시 사용자 수별 RTF 및 TTFT 추이 그래프가 {filepath}에 저장되었습니다.")
    
    def save_user_count_ttft_graph(self, filename: Optional[str] = None):
        """동시 사용자 수별 TTFT 추이 그래프 저장"""
        successful_results = [r for r in self.results if r.success and r.ttft is not None]
        if not successful_results:
            print("⚠️ TTFT 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "user_count_ttft_trend.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 동시 사용자 수별로 그룹화
        user_ttft_data: Dict[int, List[float]] = {}
        for result in successful_results:
            user_count = result.concurrent_users if result.concurrent_users is not None else 0
            if user_count not in user_ttft_data:
                user_ttft_data[user_count] = []
            if result.ttft is not None:
                user_ttft_data[user_count].append(result.ttft)
        
        if not user_ttft_data:
            print("⚠️ 동시 사용자 수별 TTFT 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        # 데이터 정리
        sorted_users = sorted(user_ttft_data.keys())
        avg_ttfts = []
        median_ttfts = []
        min_ttfts = []
        max_ttfts = []
        
        for user_count in sorted_users:
            ttft_values = user_ttft_data[user_count]
            if ttft_values:
                avg_ttfts.append(statistics.mean(ttft_values))
                median_ttfts.append(statistics.median(ttft_values))
                min_ttfts.append(min(ttft_values))
                max_ttfts.append(max(ttft_values))
            else:
                sorted_users.remove(user_count)
        
        if not sorted_users:
            print("⚠️ 유효한 TTFT 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        # 그래프 생성
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Ramp-up과 Hold 단계 구분
        ramp_up_indices = []
        hold_indices = []
        
        for i, user_count in enumerate(sorted_users):
            # Ramp-up 단계인지 확인 (user_count가 max_concurrent_users보다 작거나 같고, ramp_up 단계에 해당)
            if self.enable_ramp_up and user_count <= self.max_concurrent_users:
                # Ramp-up 단계의 사용자 수인지 확인
                step_size = self.max_concurrent_users / self.ramp_up_steps
                if user_count <= self.max_concurrent_users and (user_count % int(step_size) == 0 or user_count == self.max_concurrent_users):
                    ramp_up_indices.append(i)
                else:
                    hold_indices.append(i)
            else:
                hold_indices.append(i)
        
        # Ramp-up 단계 데이터
        if ramp_up_indices:
            ramp_up_users = [sorted_users[i] for i in ramp_up_indices]
            ramp_up_avg_ttfts = [avg_ttfts[i] for i in ramp_up_indices]
            ramp_up_median_ttfts = [median_ttfts[i] for i in ramp_up_indices]
            ramp_up_min_ttfts = [min_ttfts[i] for i in ramp_up_indices]
            ramp_up_max_ttfts = [max_ttfts[i] for i in ramp_up_indices]
            
            # 평균 TTFT 선
            ax.plot(ramp_up_users, ramp_up_avg_ttfts, 'o-', color='orange', linewidth=2.5, 
                   markersize=10, label='Ramp-up Average TTFT', alpha=0.8)
            # 중앙값 TTFT 선
            ax.plot(ramp_up_users, ramp_up_median_ttfts, 's--', color='orange', linewidth=2, 
                   markersize=8, label='Ramp-up Median TTFT', alpha=0.7)
            # Min-Max 범위
            ax.fill_between(ramp_up_users, ramp_up_min_ttfts, ramp_up_max_ttfts, 
                           alpha=0.2, color='orange', label='Ramp-up Min-Max Range')
        
        # Hold 단계 데이터
        if hold_indices:
            hold_users = [sorted_users[i] for i in hold_indices]
            hold_avg_ttfts = [avg_ttfts[i] for i in hold_indices]
            hold_median_ttfts = [median_ttfts[i] for i in hold_indices]
            hold_min_ttfts = [min_ttfts[i] for i in hold_indices]
            hold_max_ttfts = [max_ttfts[i] for i in hold_indices]
            
            # 평균 TTFT 선
            ax.plot(hold_users, hold_avg_ttfts, 'o-', color='steelblue', linewidth=2.5, 
                   markersize=10, label='Hold Average TTFT', alpha=0.8)
            # 중앙값 TTFT 선
            ax.plot(hold_users, hold_median_ttfts, 's--', color='steelblue', linewidth=2, 
                   markersize=8, label='Hold Median TTFT', alpha=0.7)
            # Min-Max 범위
            ax.fill_between(hold_users, hold_min_ttfts, hold_max_ttfts, 
                           alpha=0.2, color='steelblue', label='Hold Min-Max Range')
        
        # 전체 평균선
        if avg_ttfts:
            overall_avg = statistics.mean(avg_ttfts)
            ax.axhline(overall_avg, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=f'Overall Average TTFT: {overall_avg:.3f}s')
        
        ax.set_xlabel('Concurrent Users', fontsize=12)
        ax.set_ylabel('TTFT (Time to First Token) (seconds)', fontsize=12)
        ax.set_title('TTFT Trend by Concurrent Users', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # y축 범위를 데이터에 맞게 동적으로 조정
        if avg_ttfts:
            min_ttft = min(min_ttfts)
            max_ttft = max(max_ttfts)
            ttft_range = max_ttft - min_ttft
            if ttft_range > 0:
                y_margin = ttft_range * 0.1
                ax.set_ylim(max(0, min_ttft - y_margin), max_ttft + y_margin)
            else:
                ax.set_ylim(max(0, min_ttft - 0.1), max_ttft + 0.1)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 동시 사용자 수별 TTFT 추이 그래프가 {filepath}에 저장되었습니다.")
    
    def save_vllm_metrics_graph(self, filename: Optional[str] = None):
        """vLLM 메트릭 그래프 저장 (Running Requests, Waiting Requests)"""
        if not self.vllm_metrics_collector or not self.vllm_metrics_collector.metrics_history:
            print("⚠️ vLLM 메트릭 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        self._ensure_result_dir()
        
        if filename is None:
            filename = "vllm_metrics_requests.png"
        
        filepath = os.path.join(self.timestamp_dir, filename)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 데이터 준비
        timestamps = []
        num_running = []
        num_waiting = []
        
        start_time = self.start_time if self.start_time else 0
        
        for snapshot in self.vllm_metrics_collector.metrics_history:
            elapsed_time = snapshot.timestamp - start_time
            timestamps.append(elapsed_time)
            
            num_running.append(snapshot.num_requests_running if snapshot.num_requests_running is not None else 0)
            num_waiting.append(snapshot.num_requests_waiting if snapshot.num_requests_waiting is not None else 0)
        
        if not timestamps:
            print("⚠️ vLLM 메트릭 타임스탬프가 없어 그래프를 생성할 수 없습니다.")
            return
        
        # 그래프 생성 (2개 서브플롯: 위=Running, 아래=Waiting)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # === 위쪽: Running Requests ===
        ax1.plot(timestamps, num_running, color='blue', linewidth=2, alpha=0.8, label='Running Requests')
        ax1.fill_between(timestamps, num_running, alpha=0.3, color='blue')
        if num_running:
            avg_running = statistics.mean(num_running)
            ax1.axhline(avg_running, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Average: {avg_running:.1f}')
        
        # Ramp-up과 Hold 경계선 표시
        if self.enable_ramp_up and self.enable_hold:
            ax1.axvline(self.ramp_up_duration, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                       label='Ramp-up / Hold Boundary')
        
        ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12)
        ax1.set_ylabel('Number of Running Requests', fontsize=12)
        ax1.set_title('Running Requests Over Time (Batching)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # === 아래쪽: Waiting Requests ===
        ax2.plot(timestamps, num_waiting, color='orange', linewidth=2, alpha=0.8, label='Waiting Requests')
        ax2.fill_between(timestamps, num_waiting, alpha=0.3, color='orange')
        if num_waiting:
            avg_waiting = statistics.mean(num_waiting)
            ax2.axhline(avg_waiting, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Average: {avg_waiting:.1f}')
        
        # Ramp-up과 Hold 경계선 표시
        if self.enable_ramp_up and self.enable_hold:
            ax2.axvline(self.ramp_up_duration, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                       label='Ramp-up / Hold Boundary')
        
        ax2.set_xlabel('Elapsed Time (seconds)', fontsize=12)
        ax2.set_ylabel('Number of Waiting Requests', fontsize=12)
        ax2.set_title('Waiting Requests Over Time', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 vLLM 메트릭 그래프가 {filepath}에 저장되었습니다.")

