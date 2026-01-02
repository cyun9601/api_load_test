"""
설정 파일 로드 유틸리티 모듈
"""

import os
import yaml
from typing import Dict


def load_config(config_path: str = "config/latency_test_config.yaml") -> Dict:
    """
    YAML 설정 파일을 읽어옵니다.
    
    Args:
        config_path: 설정 파일 경로 (기본값: config/latency_test_config.yaml)
    
    Returns:
        설정 딕셔너리
    
    Raises:
        FileNotFoundError: 설정 파일을 찾을 수 없을 때
        yaml.YAMLError: YAML 파싱 오류가 발생할 때
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

