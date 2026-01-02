import asyncio
import io
import os
from typing import Callable, Optional

from utils.random_audio import create_audio_generator
from utils.resource_audio import create_resource_audio_generator, get_all_audio_files
from utils.testers import STTLoadTester
from utils.config import load_config


# HTTP STT API 호출 함수
async def http_stt_call(audio_data: io.BytesIO, base_url: str, endpoint: str, filename: str = 'audio.wav', model: str = '1225', streaming: bool = False):
    """
    HTTP STT API 호출
    
    Args:
        audio_data: 오디오 데이터 (BytesIO)
        base_url: API 기본 URL
        endpoint: API 엔드포인트
        filename: 파일명
        model: STT 모델 이름
        streaming: 스트리밍 요청 여부
    
    Returns:
        dict: {"text": str, "ttft": float} 형태 (streaming인 경우 ttft 포함)
    """
    import aiohttp
    import time
    import json
    
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
        data.add_field('model', model)
        # 언어 설정
        data.add_field('language', 'ko')
        
        # 스트리밍 요청인 경우 stream 파라미터 추가
        if streaming:
            data.add_field('stream', 'true')
        
        try:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    response_text = await response.text()
                    raise Exception(f"API 호출 실패 (상태 코드: {response.status}): {response_text}")
                
                # 스트리밍 응답 처리
                if streaming:
                    ttft = None
                    first_token_time = None
                    full_text = ""
                    
                    # 첫 번째 데이터 도착 시간 측정
                    request_start_time = time.time()
                    
                    async for line in response.content:
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = first_token_time - request_start_time
                        
                        # SSE 형식 파싱 (data: prefix 제거)
                        try:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                line_str = line_str[6:]  # 'data: ' 제거
                            
                            if line_str and line_str != '[DONE]':
                                try:
                                    chunk_data = json.loads(line_str)
                                    # 텍스트 추출 (다양한 형식 지원)
                                    chunk_text = chunk_data.get("text") or chunk_data.get("delta", {}).get("text", "") or chunk_data.get("content", "")
                                    if chunk_text:
                                        full_text += chunk_text
                                except json.JSONDecodeError:
                                    # JSON이 아닌 경우 텍스트로 처리
                                    if line_str:
                                        full_text += line_str
                        except UnicodeDecodeError:
                            # 바이너리 데이터인 경우 건너뛰기
                            continue
                    
                    result = {"text": full_text}
                    if ttft is not None:
                        result["ttft"] = ttft
                    return result
                
                # 일반 요청 처리
                else:
                    response_text = await response.text()
                    try:
                        return await response.json()
                    except:
                        # JSON이 아닌 경우 텍스트 반환
                        return {"text": response_text}
        except aiohttp.ClientError as e:
            raise Exception(f"네트워크 오류: {e}")
        except Exception as e:
            raise Exception(f"API 호출 중 오류: {e}")


async def main():
    """메인 함수"""
    # 설정 파일 읽기
    config_path = os.getenv("CONFIG_PATH", "config/load_test_config.yaml")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print(f"💡 config/load_test_config.yaml 파일을 생성해주세요.")
        return
    except Exception as e:
        print(f"❌ 오류: 설정 파일 로드 중 문제가 발생했습니다: {e}")
        return
    
    # 부하 테스트 설정 값 추출
    enable_ramp_up = config.get("enable_ramp_up", True)
    enable_hold = config.get("enable_hold", True)
    ramp_up_duration = config.get("ramp_up_duration", 60.0)
    hold_duration = config.get("hold_duration", 300.0)
    max_concurrent_users = config.get("max_concurrent_users", 10)
    ramp_up_steps = config.get("ramp_up_steps", 5)
    warmup_requests = config.get("warmup_requests", 5)
    request_delay = config.get("request_delay", 0.0)
    save_audio_samples = config.get("save_audio_samples", False)
    
    use_random_audio = config.get("use_random_audio", False)
    
    # API 설정
    api_config = config.get("api", {})
    base_url = api_config.get("base_url")
    endpoint = api_config.get("endpoint")
    model = api_config.get("model", "1225")  # 기본값: 1225
    streaming = api_config.get("streaming", False)  # 기본값: False
    
    if not base_url or not endpoint:
        print("❌ 오류: API 설정(base_url, endpoint)이 누락되었습니다. config/load_test_config.yaml을 확인해주세요.")
        return
    
    print(f"📁 설정 파일: {config_path}")
    
    audio_generator_func: Callable[[bool], io.BytesIO]
    tester_audio_duration: Optional[float] = None
    
    # 오디오 소스에 따른 설정 출력 및 함수 생성
    if use_random_audio:
        audio_duration = config.get("random_audio", {}).get("audio_duration", 10.0)
        sample_rate = config.get("random_audio", {}).get("sample_rate", 16000)
        
        print(f"🎵 오디오 설정: 랜덤 생성 모드")
        print(f"   길이 {audio_duration}초, 샘플링 레이트 {sample_rate}Hz")
        
        audio_generator_func = create_audio_generator(audio_duration, sample_rate)
        tester_audio_duration = audio_duration
    else:
        # Resource 폴더 경로 구성
        resource_config = config.get("resource", {})
        resource_base_path = resource_config.get("base_path", "resource")
        resource_warmup_folder = resource_config.get("warmup_folder", "warm_up")
        resource_test_folder = resource_config.get("test_folder", "test")
        
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
        
        # Resource 폴더 노이즈 설정
        add_noise = resource_config.get("add_noise", False)
        noise_level = resource_config.get("noise_level", 0.01)
        
        audio_generator_func = create_resource_audio_generator(
            warmup_audio_files,
            test_audio_files,
            warmup_folder_path,
            test_folder_path,
            add_noise=add_noise,
            noise_level=noise_level
        )
        tester_audio_duration = None
    
    print(f"🌐 API 설정: {base_url}{endpoint}")
    print(f"   모델: {model}")
    print(f"   스트리밍 모드: {'활성화' if streaming else '비활성화'}")
    if streaming:
        print(f"   ⚠️ 스트리밍 모드에서는 클라이언트에서 TTFT를 직접 측정합니다.")
    print()
    
    # API 호출 함수
    async def api_call_func(audio_data: io.BytesIO):
        """STT API 호출 함수"""
        filename = getattr(audio_data, 'filename', 'audio.wav')
        return await http_stt_call(audio_data, base_url, endpoint, filename=filename, model=model, streaming=streaming)
    
    # 부하 테스터 생성 및 실행
    tester = STTLoadTester(
        api_call_func=api_call_func,
        audio_generator_func=audio_generator_func,
        enable_ramp_up=enable_ramp_up,
        enable_hold=enable_hold,
        ramp_up_duration=ramp_up_duration,
        hold_duration=hold_duration,
        max_concurrent_users=max_concurrent_users,
        ramp_up_steps=ramp_up_steps,
        warmup_requests=warmup_requests,
        request_delay=request_delay,
        save_audio_samples=save_audio_samples,
        audio_duration=tester_audio_duration
    )
    
    # vLLM 메트릭 수집 설정 (api.base_url 사용)
    vllm_metrics_config = config.get("vllm_metrics", {})
    enable_vllm_metrics = vllm_metrics_config.get("enable", False)
    vllm_base_url = base_url if enable_vllm_metrics else None  # API base_url 사용
    vllm_metrics_interval = vllm_metrics_config.get("collection_interval", 1.0)
    
    metrics = await tester.run(
        enable_vllm_metrics=enable_vllm_metrics,
        vllm_base_url=vllm_base_url,
        vllm_metrics_interval=vllm_metrics_interval
    )
    tester.print_results(metrics)
    tester.save_results(metrics)


if __name__ == "__main__":
    asyncio.run(main())

