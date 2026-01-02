"""
오디오 관련 유틸리티 함수
"""

import io
import os
import wave
from typing import Optional


def get_audio_duration(audio_data: io.BytesIO, file_path: Optional[str] = None) -> Optional[float]:
    """
    오디오 데이터의 길이를 초 단위로 반환
    
    Args:
        audio_data: 오디오 데이터 (BytesIO)
        file_path: 파일 경로 (선택사항, 파일명에서 확장자 확인용)
    
    Returns:
        오디오 길이 (초), 측정 불가능한 경우 None
    """
    try:
        audio_data.seek(0)
        
        # 파일 경로가 있으면 확장자로 파일 타입 확인
        if file_path:
            file_ext = os.path.splitext(file_path.lower())[1]
            
            # WAV 파일인 경우
            if file_ext == '.wav':
                try:
                    with wave.open(audio_data, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        duration = frames / float(sample_rate)
                        audio_data.seek(0)
                        return duration
                except Exception as e:
                    audio_data.seek(0)
                    print(f"⚠️ WAV 파일 길이 측정 실패: {e}")
                    return None
            
            # MP3 파일인 경우 - mutagen 라이브러리 사용 시도
            elif file_ext == '.mp3':
                try:
                    from mutagen import File
                    # 파일 경로에서 직접 읽기 (BytesIO가 아닌 실제 파일)
                    audio_file = File(file_path)
                    if audio_file is not None and hasattr(audio_file, 'info') and hasattr(audio_file.info, 'length'):
                        duration = audio_file.info.length
                        audio_data.seek(0)
                        return duration
                except ImportError:
                    # mutagen이 설치되지 않은 경우 조용히 None 반환 (경고는 첫 요청에서만)
                    pass
                except Exception:
                    # 오류 발생 시 조용히 None 반환
                    pass
                audio_data.seek(0)
                return None
        
        # BytesIO에서 직접 WAV 파일인지 확인 (파일 경로가 없는 경우)
        audio_data.seek(0)
        header = audio_data.read(4)
        audio_data.seek(0)
        
        if header == b'RIFF':
            # WAV 파일로 시도
            try:
                with wave.open(audio_data, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / float(sample_rate)
                    audio_data.seek(0)
                    return duration
            except Exception as e:
                audio_data.seek(0)
                print(f"⚠️ WAV 파일 길이 측정 실패: {e}")
                return None
        
        return None
    except Exception as e:
        print(f"⚠️ 오디오 길이 측정 중 오류: {e}")
        return None

