"""
Resource 폴더에서 오디오 파일을 순차적으로 읽는 모듈
"""

import io
import os
import random
import numpy as np
import wave
from typing import List, Callable, Optional


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


def add_noise_to_wav(audio_data: io.BytesIO, noise_level: float = 0.01) -> io.BytesIO:
    """
    WAV 오디오 데이터에 작은 노이즈를 추가하여 캐시 우회
    
    Args:
        audio_data: WAV 오디오 데이터 (BytesIO)
        noise_level: 노이즈 레벨 (0.0 ~ 1.0, 기본값 0.01 = 1%)
    
    Returns:
        노이즈가 추가된 오디오 데이터 (BytesIO)
    """
    try:
        audio_data.seek(0)
        with wave.open(audio_data, 'rb') as wav_file:
            # WAV 파일 정보 읽기
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            
            # 오디오 데이터 읽기
            audio_bytes = wav_file.readframes(frames)
        
        # 바이트를 numpy 배열로 변환
        if sample_width == 1:
            dtype = np.uint8
            audio_array = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
            audio_array = (audio_array - 128) / 128.0  # -1.0 ~ 1.0 범위로 정규화
        elif sample_width == 2:
            dtype = np.int16
            audio_array = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
            audio_array = audio_array / 32768.0  # -1.0 ~ 1.0 범위로 정규화
        elif sample_width == 4:
            dtype = np.int32
            audio_array = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
            audio_array = audio_array / 2147483648.0  # -1.0 ~ 1.0 범위로 정규화
        else:
            # 지원하지 않는 샘플 폭인 경우 원본 반환
            audio_data.seek(0)
            return audio_data
        
        # 노이즈 생성 (작은 랜덤 노이즈)
        noise = np.random.normal(0, noise_level, audio_array.shape).astype(np.float32)
        audio_array_with_noise = audio_array + noise
        
        # 클리핑 (-1.0 ~ 1.0 범위로 제한)
        audio_array_with_noise = np.clip(audio_array_with_noise, -1.0, 1.0)
        
        # 다시 원래 포맷으로 변환
        if sample_width == 1:
            audio_array_with_noise = (audio_array_with_noise * 128 + 128).astype(np.uint8)
        elif sample_width == 2:
            audio_array_with_noise = (audio_array_with_noise * 32768).astype(np.int16)
        elif sample_width == 4:
            audio_array_with_noise = (audio_array_with_noise * 2147483648).astype(np.int32)
        
        # WAV 파일로 다시 인코딩
        output_buffer = io.BytesIO()
        with wave.open(output_buffer, 'wb') as wav_out:
            wav_out.setnchannels(channels)
            wav_out.setsampwidth(sample_width)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(audio_array_with_noise.tobytes())
        
        output_buffer.seek(0)
        return output_buffer
        
    except Exception as e:
        # 노이즈 추가 실패 시 원본 반환
        audio_data.seek(0)
        return audio_data


def load_audio_from_file(file_path: str, add_noise: bool = False, noise_level: float = 0.01) -> io.BytesIO:
    """
    파일 경로에서 오디오 파일을 읽어 BytesIO 객체로 반환
    
    Args:
        file_path: 오디오 파일 경로
        add_noise: 노이즈 추가 여부 (캐시 우회용)
        noise_level: 노이즈 레벨 (0.0 ~ 1.0, 기본값 0.01 = 1%)
    
    Returns:
        오디오 데이터를 담은 BytesIO 객체 (file_path 속성 포함)
    """
    try:
        with open(file_path, 'rb') as f:
            audio_buffer = io.BytesIO(f.read())
        
        # WAV 파일이고 노이즈 추가가 활성화된 경우
        if add_noise:
            file_ext = os.path.splitext(file_path.lower())[1]
            if file_ext == '.wav':
                audio_buffer = add_noise_to_wav(audio_buffer, noise_level)
        
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
    test_folder_path: str,
    add_noise: bool = False,
    noise_level: float = 0.01
) -> Callable[[bool], io.BytesIO]:
    """
    Resource 폴더에서 오디오 파일을 순차적으로 읽는 함수를 생성하는 팩토리 함수
    
    Args:
        warmup_audio_files: Warm-up용 오디오 파일 경로 리스트
        test_audio_files: Test용 오디오 파일 경로 리스트
        warmup_folder_path: Warm-up 폴더 경로 (에러 메시지용)
        test_folder_path: Test 폴더 경로 (에러 메시지용)
        add_noise: 노이즈 추가 여부 (캐시 우회용, 기본값 False)
        noise_level: 노이즈 레벨 (0.0 ~ 1.0, 기본값 0.01 = 1%)
    
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
        
        audio_data = load_audio_from_file(file_path, add_noise=add_noise, noise_level=noise_level)
        # 파일명 정보를 저장 (나중에 API 호출 시 사용)
        audio_data.filename = os.path.basename(file_path)
        # file_path도 명시적으로 저장 (오디오 길이 측정용)
        audio_data.file_path = file_path
        return audio_data
    
    return audio_generator

