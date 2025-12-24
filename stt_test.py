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


class STTLoadTester:
    """STT 모델 로드 테스터"""
    
    def __init__(
        self,
        api_call_func,
        audio_generator_func,
        total_requests: int,
        warmup_requests: int,
        concurrent_requests: int = 1,
        request_delay: float = 0.0
    ):
        """
        Args:
            api_call_func: STT API를 호출하는 비동기 함수 (audio_data: io.BytesIO를 인자로 받음)
            audio_generator_func: 오디오를 생성하는 함수 (io.BytesIO를 반환)
            total_requests: 총 요청 수 (N)
            warmup_requests: 버릴 warm-up 요청 수 (M)
            concurrent_requests: 동시 요청 수
            request_delay: 요청 간 지연 시간 (초)
        """
        self.api_call_func = api_call_func
        self.audio_generator_func = audio_generator_func
        self.total_requests = total_requests
        self.warmup_requests = warmup_requests
        self.concurrent_requests = concurrent_requests
        self.request_delay = request_delay
        self.results: List[TestResult] = []  # 성능 테스트 결과
        self.warmup_results: List[TestResult] = []  # Cold start (warmup) 결과
    
    async def _make_request(self, request_id: int) -> TestResult:
        """단일 요청 실행 (오디오 생성 시간 제외)"""
        # 오디오 생성 (시간 측정 제외)
        audio_data = self.audio_generator_func()
        
        # API 호출만 시간 측정에 포함
        start_time = time.time()
        try:
            await self.api_call_func(audio_data)
            response_time = time.time() - start_time
            return TestResult(response_time=response_time, success=True)
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def _run_requests(self, num_requests: int, is_warmup: bool = False) -> List[TestResult]:
        """요청 배치 실행"""
        results = []
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        async def bounded_request(request_id: int):
            async with semaphore:
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                result = await self._make_request(request_id)
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
        print(f"🚀 STT 모델 로드 테스트 시작")
        print(f"   총 요청 수: {self.total_requests}")
        print(f"   Warm-up 요청 수: {self.warmup_requests}")
        print(f"   동시 요청 수: {self.concurrent_requests}")
        print(f"   실제 측정 요청 수: {self.total_requests - self.warmup_requests}")
        print(f"   매 요청마다 새로운 랜덤 오디오 생성")
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
            requests_per_second=len(self.results) / total_time if total_time > 0 else 0
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
        print(f"처리량: {metrics.requests_per_second:.2f} 요청/초")
        print("="*60)
        
        # 실패한 요청 상세 정보
        if metrics.failed_requests > 0:
            print("\n❌ 실패한 요청 상세:")
            for i, result in enumerate(self.results):
                if not result.success:
                    print(f"  요청 #{i+1}: {result.error}")
    
    def save_histogram(self, filename: Optional[str] = None):
        """응답 시간 도수분포표(히스토그램)를 저장 (Cold start와 성능 테스트 구분)"""
        # Cold start (warmup)와 성능 테스트 결과 수집
        warmup_response_times = [r.response_time for r in self.warmup_results if r.success]
        performance_response_times = [r.response_time for r in self.results if r.success]
        
        if not warmup_response_times and not performance_response_times:
            print("⚠️ 성공한 요청이 없어 히스토그램을 생성할 수 없습니다.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_time_histogram_{timestamp}.png"
        
        # Font settings
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 히스토그램 생성
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 모든 응답 시간을 합쳐서 bins 범위 결정
        all_times = warmup_response_times + performance_response_times
        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
            bins = np.linspace(min_time, max_time, 30)
        else:
            bins = 30
        
        # Cold start (warmup) histogram
        if warmup_response_times:
            ax.hist(
                warmup_response_times,
                bins=bins,
                edgecolor='black',
                alpha=0.6,
                color='orange',
                label=f'Cold Start (Warm-up) ({len(warmup_response_times)} requests)'
            )
        
        # Performance test histogram
        if performance_response_times:
            ax.hist(
                performance_response_times,
                bins=bins,
                edgecolor='black',
                alpha=0.6,
                color='steelblue',
                label=f'Performance Test ({len(performance_response_times)} requests)'
            )
        
        # Calculate and display statistics
        stats_lines = []
        
        if warmup_response_times:
            warmup_avg = statistics.mean(warmup_response_times)
            warmup_median = statistics.median(warmup_response_times)
            ax.axvline(warmup_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                     label=f'Cold Start Avg: {warmup_avg:.3f}s')
            stats_lines.append(f'Cold Start: {len(warmup_response_times)} requests')
            stats_lines.append(f'  Avg: {warmup_avg:.3f}s')
            stats_lines.append(f'  Median: {warmup_median:.3f}s')
        
        if performance_response_times:
            perf_avg = statistics.mean(performance_response_times)
            perf_median = statistics.median(performance_response_times)
            ax.axvline(perf_avg, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                     label=f'Performance Test Avg: {perf_avg:.3f}s')
            if not stats_lines:
                stats_lines.append('Performance Test:')
            stats_lines.append(f'  {len(performance_response_times)} requests')
            stats_lines.append(f'  Avg: {perf_avg:.3f}s')
            stats_lines.append(f'  Median: {perf_median:.3f}s')
        
        if all_times:
            stats_lines.append(f'\nOverall Min: {min(all_times):.3f}s')
            stats_lines.append(f'Overall Max: {max(all_times):.3f}s')
        
        ax.set_xlabel('Response Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('STT API Response Time Histogram (Cold Start vs Performance Test)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 텍스트 추가
        stats_text = '\n'.join(stats_lines)
        
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 히스토그램이 {filename}에 저장되었습니다.")
    
    def save_timeline_graph(self, filename: Optional[str] = None):
        """요청 인덱스별 응답 시간 추이 그래프를 저장"""
        # 모든 요청 결과 수집 (cold start + 성능 테스트)
        all_results = self.warmup_results + self.results
        successful_results = [r for r in all_results if r.success]
        
        if not successful_results:
            print("⚠️ 성공한 요청이 없어 타임라인 그래프를 생성할 수 없습니다.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_time_timeline_{timestamp}.png"
        
        # Font settings
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 요청 인덱스와 응답 시간 분리
        request_indices = []
        response_times = []
        is_warmup_list = []
        
        # Warmup 결과 추가
        for idx, result in enumerate(self.warmup_results):
            if result.success:
                request_indices.append(idx + 1)
                response_times.append(result.response_time)
                is_warmup_list.append(True)
        
        # 성능 테스트 결과 추가
        warmup_count = len([r for r in self.warmup_results if r.success])
        for idx, result in enumerate(self.results):
            if result.success:
                request_indices.append(warmup_count + idx + 1)
                response_times.append(result.response_time)
                is_warmup_list.append(False)
        
        # Cold start와 성능 테스트를 색상으로 구분
        warmup_indices = [idx for idx, is_warmup in zip(request_indices, is_warmup_list) if is_warmup]
        warmup_times = [time for time, is_warmup in zip(response_times, is_warmup_list) if is_warmup]
        perf_indices = [idx for idx, is_warmup in zip(request_indices, is_warmup_list) if not is_warmup]
        perf_times = [time for time, is_warmup in zip(response_times, is_warmup_list) if not is_warmup]
        
        # Cold start 플롯
        if warmup_indices:
            ax.scatter(warmup_indices, warmup_times, 
                      color='orange', alpha=0.6, s=30, 
                      label=f'Cold Start (Warm-up) ({len(warmup_indices)} requests)')
            ax.plot(warmup_indices, warmup_times, 
                   color='orange', alpha=0.3, linewidth=1)
        
        # 성능 테스트 플롯
        if perf_indices:
            ax.scatter(perf_indices, perf_times, 
                      color='steelblue', alpha=0.6, s=30,
                      label=f'Performance Test ({len(perf_indices)} requests)')
            ax.plot(perf_indices, perf_times, 
                   color='steelblue', alpha=0.3, linewidth=1)
        
        # 평균선 표시
        if warmup_times:
            warmup_avg = statistics.mean(warmup_times)
            ax.axhline(warmup_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'Cold Start Avg: {warmup_avg:.3f}s')
        
        if perf_times:
            perf_avg = statistics.mean(perf_times)
            ax.axhline(perf_avg, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'Performance Test Avg: {perf_avg:.3f}s')
        
        # Cold start와 성능 테스트 경계선 표시
        if warmup_indices and perf_indices:
            boundary = max(warmup_indices)
            ax.axvline(boundary, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                      label='Warm-up / Performance Test Boundary')
        
        ax.set_xlabel('Request Index', fontsize=12)
        ax.set_ylabel('Response Time (seconds)', fontsize=12)
        ax.set_title('STT API Response Time Timeline (All Requests)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
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
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 타임라인 그래프가 {filename}에 저장되었습니다.")
    
    def save_results(self, metrics: PerformanceMetrics, filename: Optional[str] = None):
        """결과를 JSON 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stt_load_test_results_{timestamp}.json"
        
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
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과가 {filename}에 저장되었습니다.")
        
        # 히스토그램과 타임라인 그래프 저장
        self.save_histogram()
        self.save_timeline_graph()


def generate_random_audio(duration_seconds: float = 10.0, sample_rate: int = 16000) -> io.BytesIO:
    """
    랜덤 오디오 데이터를 생성하여 WAV 파일 형식의 BytesIO 객체로 반환
    
    Args:
        duration_seconds: 오디오 길이 (초) - 기본값: 10.0
        sample_rate: 샘플링 레이트 (Hz) - 기본값: 16000
    
    Returns:
        WAV 형식의 오디오 데이터를 담은 BytesIO 객체
    """
    # 샘플 수 계산
    num_samples = int(duration_seconds * sample_rate)
    
    # 랜덤 오디오 데이터 생성 (화이트 노이즈 + 여러 주파수 조합)
    # 다양한 주파수 성분을 추가하여 더 현실적인 오디오 생성
    t = np.linspace(0, duration_seconds, num_samples)
    
    # 랜덤 주파수와 진폭으로 여러 사인파 생성
    audio_data = np.zeros(num_samples)
    num_components = random.randint(3, 8)
    
    for _ in range(num_components):
        frequency = random.uniform(100, 2000)  # 100Hz ~ 2000Hz
        amplitude = random.uniform(0.1, 0.5)
        phase = random.uniform(0, 2 * np.pi)
        audio_data += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # 화이트 노이즈 추가
    noise = np.random.normal(0, 0.1, num_samples)
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

# HTTP STT API 호출 함수
async def http_stt_call(audio_data: io.BytesIO, base_url: str, endpoint: str):
    """HTTP STT API 호출"""
    import aiohttp
    
    url = f"{base_url}{endpoint}"
    
    async with aiohttp.ClientSession() as session:
        # BytesIO를 바이트 데이터로 읽기
        audio_data.seek(0)  # 파일 포인터를 처음으로
        audio_bytes = audio_data.read()
        
        data = aiohttp.FormData()
        # 바이트 데이터를 파일로 전송
        data.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
        # 필요시 추가 필드 (예: model, language 등)
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
    total_requests = config.get("total_requests", 100)
    warmup_requests = config.get("warmup_requests", 10)
    concurrent_requests = config.get("concurrent_requests", 5)
    request_delay = config.get("request_delay", 0.0)
    audio_duration = config.get("audio_duration", 10.0)
    sample_rate = config.get("sample_rate", 16000)
    save_path = config.get("save_path", None)
    base_url = config.get("api", {}).get("base_url", "http://192.168.73.172:8000")
    endpoint = config.get("api", {}).get("endpoint", "/v1/audio/transcriptions")
    
    # 유효성 검사
    if warmup_requests >= total_requests:
        print("❌ 오류: warmup_requests는 total_requests보다 작아야 합니다.")
        return
    
    print(f"📁 설정 파일: {config_path}")
    print(f"🎵 오디오 설정: 길이 {audio_duration}초, 샘플링 레이트 {sample_rate}Hz")
    print(f"   매 요청마다 새로운 랜덤 오디오 생성 (캐시 방지)")
    print(f"🌐 API 설정: {base_url}{endpoint}")
    print()
    
    # 오디오 생성 함수
    def audio_generator():
        """랜덤 오디오 생성 함수 (시간 측정 제외)"""
        return generate_random_audio(
            duration_seconds=audio_duration,
            sample_rate=sample_rate
        )
    
    # API 호출 함수 (오디오 데이터를 인자로 받음)
    async def api_call_func(audio_data: io.BytesIO):
        """STT API 호출 함수 (시간 측정에 포함)"""
        return await http_stt_call(audio_data, base_url, endpoint)
    
    # 테스터 생성 및 실행
    tester = STTLoadTester(
        api_call_func=api_call_func,
        audio_generator_func=audio_generator,
        total_requests=total_requests,
        warmup_requests=warmup_requests,
        concurrent_requests=concurrent_requests,
        request_delay=request_delay
    )
    
    metrics = await tester.run()
    tester.print_results(metrics)
    
    # 결과 저장
    tester.save_results(metrics, save_path)


if __name__ == "__main__":
    asyncio.run(main())