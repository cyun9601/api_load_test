"""
실제 사람 음성과 유사한 랜덤 오디오 생성 모듈
"""

import io
import random
import numpy as np
import wave
from typing import Callable


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


def create_audio_generator(duration_seconds: float = 10.0, sample_rate: int = 16000) -> Callable[[bool], io.BytesIO]:
    """
    랜덤 오디오 생성 함수를 반환하는 팩토리 함수
    
    Args:
        duration_seconds: 오디오 길이 (초) - 기본값: 10.0
        sample_rate: 샘플링 레이트 (Hz) - 기본값: 16000
    
    Returns:
        audio_generator 함수 (is_warmup: bool -> io.BytesIO)
    """
    def audio_generator(is_warmup: bool = False) -> io.BytesIO:
        """
        음성과 유사한 오디오 생성 함수 (시간 측정 제외)
        
        Args:
            is_warmup: Warm-up 요청 여부 (현재는 사용되지 않지만 인터페이스 일관성을 위해 유지)
        
        Returns:
            WAV 형식의 오디오 데이터를 담은 BytesIO 객체
        """
        return generate_speech_like_audio(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate
        )
    
    return audio_generator

