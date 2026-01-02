import asyncio
import io
import os
from typing import Callable, Optional

from utils.random_audio import create_audio_generator
from utils.resource_audio import create_resource_audio_generator, get_all_audio_files
from utils.testers import STTLatencyTester
from utils.config import load_config


# HTTP STT API 호출 함수
async def http_stt_call(audio_data: io.BytesIO, base_url: str, endpoint: str, filename: str = 'audio.wav'):
    """HTTP STT API 호출"""
    import aiohttp
    
    url = f"{base_url}{endpoint}"
    
    # 파일 확장자에 따른 content_type 결정
    _, ext = os.path.splitext(filename.lower())
    content_type_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.wma': 'audio/x-ms-wma'
    }
    content_type = content_type_map.get(ext, 'audio/wav')
    
    async with aiohttp.ClientSession() as session:
        # BytesIO를 바이트 데이터로 읽기
        audio_data.seek(0)  # 파일 포인터를 처음으로
        audio_bytes = audio_data.read()
        
        data = aiohttp.FormData()
        # 바이트 데이터를 파일로 전송
        data.add_field('file', audio_bytes, filename=filename, content_type=content_type)
        data.add_field('model', '1225')
        # data.add_field('model', 'openai/whisper-large-v3')
        # 언어 설정
        data.add_field('language', 'ko')
        # 필요시 추가 필드 (예: model 등)
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


async def main():
    """메인 함수"""
    # 설정 파일 읽기
    config_path = os.getenv("CONFIG_PATH", "config/latency_test_config.yaml")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print(f"💡 config/latency_test_config.yaml 파일을 생성해주세요.")
        return
    except Exception as e:
        print(f"❌ 오류: 설정 파일 로드 중 문제가 발생했습니다: {e}")
        return
    
    # 설정 값 추출 (기본값 포함)
    concurrent_requests = config.get("concurrent_requests", 5)
    request_delay = config.get("request_delay", 0.0)
    use_random_audio = config.get("use_random_audio", True)
    save_audio_samples = config.get("save_audio_samples", False)
    save_path = config.get("save_path", None)
    base_url = config.get("api", {}).get("base_url", "http://192.168.73.172:8000")
    endpoint = config.get("api", {}).get("endpoint", "/v1/audio/transcriptions")
    
    # 랜덤 오디오 설정 (use_random_audio가 true일 때만 사용)
    random_audio_config = config.get("random_audio", {})
    total_requests = random_audio_config.get("total_requests", 100)
    warmup_requests = random_audio_config.get("warmup_requests", 10)
    audio_duration = random_audio_config.get("audio_duration", 10.0)
    sample_rate = random_audio_config.get("sample_rate", 16000)
    
    # Resource 폴더 설정 (use_random_audio가 false일 때 사용)
    resource_config = config.get("resource", {})
    resource_base_path = resource_config.get("base_path", "resource")
    resource_warmup_folder = resource_config.get("warmup_folder", "warm_up")
    resource_test_folder = resource_config.get("test_folder", "test")
    
    print(f"📁 설정 파일: {config_path}")
    
    # 오디오 소스에 따른 설정 출력 및 함수 생성
    if use_random_audio:
        # 랜덤 오디오 생성 모드일 때만 유효성 검사
        if warmup_requests >= total_requests:
            print("❌ 오류: warmup_requests는 total_requests보다 작아야 합니다.")
            return
        print(f"🎵 오디오 설정: 랜덤 생성 모드")
        print(f"   길이 {audio_duration}초, 샘플링 레이트 {sample_rate}Hz")
        print(f"   매 요청마다 새로운 음성과 유사한 오디오 생성 (캐시 방지)")
        print(f"   (포먼트, 하모닉, 진폭 변조 포함)")
        
        # 랜덤 오디오 생성 함수
        audio_generator = create_audio_generator(
            duration_seconds=audio_duration,
            sample_rate=sample_rate
        )
        
    else:
        # Resource 폴더 경로 구성
        warmup_folder_path = os.path.join(resource_base_path, resource_warmup_folder)
        test_folder_path = os.path.join(resource_base_path, resource_test_folder)
        
        print(f"🎵 오디오 설정: Resource 폴더 사용 모드")
        print(f"   Warm-up 폴더: {warmup_folder_path}")
        print(f"   Test 폴더: {test_folder_path}")
        
        # 폴더 존재 확인 및 파일 목록 로드
        warmup_audio_files = []
        test_audio_files = []
        
        if os.path.exists(warmup_folder_path):
            warmup_audio_files = get_all_audio_files(warmup_folder_path)
            print(f"   Warm-up 파일 수: {len(warmup_audio_files)}개")
        else:
            print(f"⚠️ 경고: Warm-up 폴더를 찾을 수 없습니다: {warmup_folder_path}")
        
        if os.path.exists(test_folder_path):
            test_audio_files = get_all_audio_files(test_folder_path)
            print(f"   Test 파일 수: {len(test_audio_files)}개")
        else:
            print(f"⚠️ 경고: Test 폴더를 찾을 수 없습니다: {test_folder_path}")
        
        if not warmup_audio_files and not test_audio_files:
            print("❌ 오류: 사용 가능한 오디오 파일이 없습니다.")
            return
        
        # Resource 폴더 사용 시 파일 개수에 맞춰 요청 수 자동 조정
        if warmup_audio_files:
            actual_warmup_requests = len(warmup_audio_files)
            if warmup_requests != actual_warmup_requests:
                print(f"ℹ️  Warm-up 요청 수를 파일 개수에 맞춰 조정: {warmup_requests} → {actual_warmup_requests}")
                warmup_requests = actual_warmup_requests
        else:
            warmup_requests = 0
            print(f"ℹ️  Warm-up 폴더가 비어있어 Warm-up 요청 수를 0으로 설정")
        
        if test_audio_files:
            actual_test_requests = len(test_audio_files)
            actual_total_requests = warmup_requests + actual_test_requests
            if total_requests != actual_total_requests:
                print(f"ℹ️  총 요청 수를 파일 개수에 맞춰 조정: {total_requests} → {actual_total_requests}")
                total_requests = actual_total_requests
        else:
            print("❌ 오류: Test 폴더에 오디오 파일이 없습니다.")
            return
        
        # Resource 폴더 노이즈 설정
        add_noise = resource_config.get("add_noise", False)
        noise_level = resource_config.get("noise_level", 0.01)
        
        # Resource 폴더에서 파일 읽기 함수 생성
        audio_generator = create_resource_audio_generator(
            warmup_audio_files=warmup_audio_files,
            test_audio_files=test_audio_files,
            warmup_folder_path=warmup_folder_path,
            test_folder_path=test_folder_path,
            add_noise=add_noise,
            noise_level=noise_level
        )
    
    print(f"🌐 API 설정: {base_url}{endpoint}")
    print()
    
    # API 호출 함수 (오디오 데이터를 인자로 받음)
    async def api_call_func(audio_data: io.BytesIO):
        """STT API 호출 함수 (시간 측정에 포함)"""
        # 파일명이 있으면 사용, 없으면 기본값 사용
        filename = getattr(audio_data, 'filename', 'audio.wav')
        return await http_stt_call(audio_data, base_url, endpoint, filename=filename)
    
    # 테스터 생성 및 실행
    tester = STTLatencyTester(
        api_call_func=api_call_func,
        audio_generator_func=audio_generator,
        total_requests=total_requests,
        warmup_requests=warmup_requests,
        concurrent_requests=concurrent_requests,
        request_delay=request_delay,
        save_audio_samples=save_audio_samples,
        audio_duration=audio_duration if use_random_audio else None
    )
    
    metrics = await tester.run()
    tester.print_results(metrics)
    
    # 결과 저장
    tester.save_results(metrics, save_path)


if __name__ == "__main__":
    asyncio.run(main())