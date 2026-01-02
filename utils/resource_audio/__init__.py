"""
Resource 폴더에서 오디오 파일을 읽는 유틸리티 모듈
"""

from .loader import create_resource_audio_generator, get_all_audio_files

__all__ = ['create_resource_audio_generator', 'get_all_audio_files']

