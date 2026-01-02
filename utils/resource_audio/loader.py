"""
Resource 폴더에서 오디오 파일을 순차적으로 읽는 모듈
"""

import io
import os
from typing import List, Callable


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


def load_audio_from_file(file_path: str) -> io.BytesIO:
    """
    파일 경로에서 오디오 파일을 읽어 BytesIO 객체로 반환
    
    Args:
        file_path: 오디오 파일 경로
    
    Returns:
        오디오 데이터를 담은 BytesIO 객체 (file_path 속성 포함)
    """
    try:
        with open(file_path, 'rb') as f:
            audio_buffer = io.BytesIO(f.read())
        # 파일 경로를 속성으로 저장 (나중에 오디오 길이 측정용)
        audio_buffer.file_path = file_path
        return audio_buffer
    except FileNotFoundError:
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        raise ValueError(f"오디오 파일 읽기 중 오류: {e}")


def create_resource_audio_generator(
    warmup_audio_files: List[str],
    test_audio_files: List[str],
    warmup_folder_path: str,
    test_folder_path: str
) -> Callable[[bool], io.BytesIO]:
    """
    Resource 폴더에서 오디오 파일을 순차적으로 읽는 함수를 생성하는 팩토리 함수
    
    Args:
        warmup_audio_files: Warm-up용 오디오 파일 경로 리스트
        test_audio_files: Test용 오디오 파일 경로 리스트
        warmup_folder_path: Warm-up 폴더 경로 (에러 메시지용)
        test_folder_path: Test 폴더 경로 (에러 메시지용)
    
    Returns:
        audio_generator 함수 (is_warmup: bool -> io.BytesIO)
    """
    # 파일 인덱스를 추적하기 위한 변수 (클로저에서 사용)
    warmup_file_index = [0]  # 리스트로 감싸서 참조 전달
    test_file_index = [0]
    
    def audio_generator(is_warmup: bool = False) -> io.BytesIO:
        """
        Resource 폴더에서 오디오 파일을 순차적으로 읽기 (시간 측정 제외)
        
        Args:
            is_warmup: Warm-up 요청 여부
        
        Returns:
            오디오 데이터를 담은 BytesIO 객체 (filename, file_path 속성 포함)
        """
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
    
    return audio_generator

