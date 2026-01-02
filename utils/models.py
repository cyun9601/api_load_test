"""
공통 데이터 모델
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestResult:
    """단일 테스트 결과"""
    response_time: float
    success: bool
    error: Optional[str] = None
    text: Optional[str] = None  # STT 예측 텍스트
    audio_duration: Optional[float] = None  # 오디오 길이 (초)
    rtf: Optional[float] = None  # Real-Time Factor (처리 시간 / 오디오 길이)
    concurrent_users: Optional[int] = None  # 동시 사용자 수 (load test에서 사용)
    ttft: Optional[float] = None  # Time to First Token (초, 스트리밍인 경우 클라이언트 측정)


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

