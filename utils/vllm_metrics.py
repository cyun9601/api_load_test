"""
vLLM Metrics 수집 모듈
"""

import asyncio
import aiohttp
import time
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class VLLMMetricsSnapshot:
    """vLLM 메트릭 스냅샷"""
    timestamp: float
    num_requests_running: Optional[int] = None
    num_requests_waiting: Optional[int] = None
    queue_size: Optional[int] = None


def parse_prometheus_metrics(metrics_text: str) -> Dict[str, float]:
    """
    Prometheus 형식의 메트릭 텍스트를 파싱하여 딕셔너리로 반환
    
    Args:
        metrics_text: Prometheus 형식의 메트릭 텍스트
    
    Returns:
        메트릭 이름을 키로, 값을 값으로 하는 딕셔너리
    """
    metrics_dict = {}
    
    for line in metrics_text.split('\n'):
        line = line.strip()
        # 주석이나 빈 줄 건너뛰기
        if not line or line.startswith('#'):
            continue
        
        # 메트릭 형식: metric_name{labels} value 또는 metric_name value
        if '{' in line:
            # 라벨이 있는 경우
            parts = line.split('}')
            if len(parts) == 2:
                metric_part = parts[0] + '}'
                value_part = parts[1].strip()
                
                # 메트릭 이름과 라벨 분리
                metric_name = metric_part.split('{')[0]
                value = value_part.split()[0] if value_part.split() else None
                
                if value:
                    try:
                        metrics_dict[metric_name] = float(value)
                    except ValueError:
                        pass
        else:
            # 라벨이 없는 경우
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[0]
                value = parts[1]
                try:
                    metrics_dict[metric_name] = float(value)
                except ValueError:
                    pass
    
    return metrics_dict


async def collect_vllm_metrics(base_url: str, metrics_endpoint: str = "/metrics") -> Optional[VLLMMetricsSnapshot]:
    """
    vLLM metrics 엔드포인트에서 메트릭 수집
    
    Args:
        base_url: vLLM 서버 기본 URL
        metrics_endpoint: 메트릭 엔드포인트 경로
    
    Returns:
        VLLMMetricsSnapshot 또는 None (실패 시)
    """
    url = f"{base_url}{metrics_endpoint}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    metrics_dict = parse_prometheus_metrics(metrics_text)
                    
                    snapshot = VLLMMetricsSnapshot(timestamp=time.time())
                    
                    # Running requests 추출
                    snapshot.num_requests_running = int(metrics_dict.get('vllm:num_requests_running', 0) or 
                                                       metrics_dict.get('num_requests_running', 0) or 0)
                    
                    # Waiting requests 추출
                    snapshot.num_requests_waiting = int(metrics_dict.get('vllm:num_requests_waiting', 0) or 
                                                      metrics_dict.get('num_requests_waiting', 0) or 0)
                    
                    # Queue size 추출
                    snapshot.queue_size = int(metrics_dict.get('vllm:queue_size', 0) or 
                                            metrics_dict.get('queue_size', 0) or 0)
                    
                    return snapshot
    except Exception as e:
        # 조용히 실패 (메트릭 수집 실패는 치명적이지 않음)
        return None


class VLLMMetricsCollector:
    """vLLM 메트릭 수집기"""
    
    def __init__(self, base_url: str, metrics_endpoint: str = "/metrics", collection_interval: float = 1.0):
        """
        Args:
            base_url: vLLM 서버 기본 URL
            metrics_endpoint: 메트릭 엔드포인트 경로
            collection_interval: 메트릭 수집 간격 (초)
        """
        self.base_url = base_url
        self.metrics_endpoint = metrics_endpoint
        self.collection_interval = collection_interval
        self.metrics_history: List[VLLMMetricsSnapshot] = []
        self._collecting = False
        self._task: Optional[asyncio.Task] = None
    
    async def _collect_loop(self):
        """메트릭 수집 루프"""
        while self._collecting:
            snapshot = await collect_vllm_metrics(self.base_url, self.metrics_endpoint)
            if snapshot:
                self.metrics_history.append(snapshot)
            await asyncio.sleep(self.collection_interval)
    
    async def start_collecting(self):
        """메트릭 수집 시작"""
        if not self._collecting:
            self._collecting = True
            self.metrics_history = []
            self._task = asyncio.create_task(self._collect_loop())
    
    async def stop_collecting(self):
        """메트릭 수집 중지"""
        self._collecting = False
        if self._task:
            await self._task
            self._task = None

