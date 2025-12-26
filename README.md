# STT 모델 로드 테스트 도구

STT(Speech-to-Text) 모델의 성능을 테스트하기 위한 Python 도구입니다. Cold start를 방지하기 위한 warm-up 기능을 포함하고 있습니다.

## 기능

- 🔥 **Warm-up 요청**: Cold start를 방지하기 위해 초기 M개의 요청을 버리고 실제 성능 측정에 사용하지 않습니다
- 📊 **성능 메트릭**: 평균, 중앙값, P95, P99 응답 시간 및 처리량 측정
- ⚡ **동시 요청 지원**: 여러 요청을 동시에 처리하여 실제 부하 상황을 시뮬레이션
- 🎵 **오디오 소스 선택**: 랜덤 오디오 생성 또는 resource 폴더의 실제 오디오 파일 사용
- 📈 **시각화**: 응답 시간 히스토그램 및 타임라인 그래프 자동 생성
- 💾 **결과 저장**: JSON 형식으로 상세한 테스트 결과 저장 (result 폴더에 자동 저장)

## 설치

```bash
pip install -e .
```

또는 필요한 패키지를 직접 설치:

```bash
pip install aiohttp numpy pyyaml matplotlib
```

## 사용 방법

### 설정 파일

프로그램은 `config.yaml` 파일에서 설정을 읽어옵니다. 기본적으로 프로젝트 루트의 `config.yaml` 파일을 사용하며, `CONFIG_PATH` 환경 변수로 다른 경로를 지정할 수 있습니다.

### 기본 사용

```bash
python stt_test.py
```

### 설정 파일 예제

`config.yaml` 파일을 생성하고 다음과 같이 설정하세요:

#### 랜덤 오디오 생성 모드

```yaml
# 테스트 설정
concurrent_requests: 5      # 동시 요청 수
request_delay: 0.0          # 요청 간 지연 시간 (초)

# 오디오 설정
use_random_audio: true      # 랜덤으로 오디오 생성
save_audio_samples: false   # 랜덤으로 생성된 음성 파일 저장 여부

# 랜덤 오디오 생성 설정
random_audio:
  total_requests: 100        # 총 요청 수 (N)
  warmup_requests: 10       # Warm-up 요청 수 (M) - 이 개수만큼 버림
  audio_duration: 10.0      # 생성할 오디오 길이 (초)
  sample_rate: 16000        # 오디오 샘플링 레이트 (Hz)

# API 설정
api:
  base_url: "http://192.168.73.172:8000"
  endpoint: "/v1/audio/transcriptions"

# 결과 저장 설정
save_path: null             # 결과 저장 경로 (null이면 자동 생성, result 폴더에 저장)
```

#### Resource 폴더 사용 모드

```yaml
# 테스트 설정
concurrent_requests: 1      # 동시 요청 수
request_delay: 1.0          # 요청 간 지연 시간 (초)

# 오디오 설정
use_random_audio: false     # resource 폴더의 파일 사용
save_audio_samples: false   # 오디오 샘플 저장 여부

# Resource 폴더 설정 (파일 개수에 맞춰 자동 조정)
resource:
  base_path: "resource"     # resource 폴더 경로
  warmup_folder: "warm_up"  # warm-up용 오디오 파일 폴더
  test_folder: "test"       # 성능 테스트용 오디오 파일 폴더

# API 설정
api:
  base_url: "http://192.168.73.172:8000"
  endpoint: "/v1/audio/transcriptions"

# 결과 저장 설정
save_path: null             # 결과 저장 경로 (null이면 자동 생성, result 폴더에 저장)
```

**참고사항**:
- **랜덤 오디오 모드** (`use_random_audio: true`): 매 요청마다 새로운 랜덤 오디오가 생성되므로 캐시 효과 없이 정확한 성능 측정이 가능합니다. `random_audio` 섹션의 설정이 사용됩니다.
- **Resource 폴더 모드** (`use_random_audio: false`): 지정된 폴더의 오디오 파일을 순차적으로 사용합니다. 요청 수는 파일 개수에 맞춰 자동으로 조정됩니다.
- 모든 결과 파일(JSON, 히스토그램, 타임라인 그래프)은 `result` 폴더에 자동으로 저장됩니다.

### 다른 설정 파일 사용

```bash
# 환경 변수로 설정 파일 경로 지정
CONFIG_PATH=my_config.yaml python stt_test.py
```

## 오디오 소스 모드

### 랜덤 오디오 생성 모드

`use_random_audio: true`로 설정하면 매 요청마다 새로운 랜덤 오디오가 생성됩니다. 이 모드에서는:
- `random_audio` 섹션의 `total_requests`와 `warmup_requests` 설정이 사용됩니다
- 실제 사람 음성과 유사한 오디오를 생성합니다 (포먼트, 하모닉, 진폭 변조 포함)
- 캐시 효과 없이 정확한 성능 측정이 가능합니다

### Resource 폴더 사용 모드

`use_random_audio: false`로 설정하면 지정된 폴더의 오디오 파일을 사용합니다. 이 모드에서는:
- `resource` 섹션에서 폴더 경로를 지정합니다
- Warm-up과 성능 테스트에 각각 다른 폴더를 사용할 수 있습니다
- 폴더 내의 모든 파일을 순차적으로 한 번씩 테스트합니다
- 요청 수는 파일 개수에 맞춰 자동으로 조정됩니다
- 지원 형식: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.wma`

## 결과 파일

모든 결과 파일은 `result` 폴더에 저장됩니다:

- **JSON 결과 파일**: `stt_load_test_results_YYYYMMDD_HHMMSS.json`
  - 테스트 설정, 성능 메트릭, 각 요청의 상세 결과 포함
- **히스토그램**: `response_time_histogram_YYYYMMDD_HHMMSS.png`
  - Cold start와 성능 테스트의 응답 시간 분포 비교
- **타임라인 그래프**: `response_time_timeline_YYYYMMDD_HHMMSS.png`
  - 요청 인덱스별 응답 시간 추이

## 출력 예시

### 랜덤 오디오 생성 모드

```
📁 설정 파일: config.yaml
🎵 오디오 설정: 랜덤 생성 모드
   길이 10.0초, 샘플링 레이트 16000Hz
   매 요청마다 새로운 음성과 유사한 오디오 생성 (캐시 방지)
   (포먼트, 하모닉, 진폭 변조 포함)
🌐 API 설정: http://192.168.73.172:8000/v1/audio/transcriptions

🚀 STT 모델 로드 테스트 시작
   총 요청 수: 100
   Warm-up 요청 수: 10
   동시 요청 수: 5
   실제 측정 요청 수: 90
   매 요청마다 새로운 음성과 유사한 오디오 생성

🔥 Warm-up 단계 (10개 요청)...
   Warm-up 완료 (소요 시간: 0.50초)

📊 성능 측정 단계 (90개 요청)...
============================================================
📈 성능 테스트 결과
============================================================
총 요청 수: 90
성공한 요청: 90 (100.0%)
실패한 요청: 0 (0.0%)

응답 시간 통계:
  평균: 0.505초
  중앙값: 0.500초
  최소: 0.498초
  최대: 0.520초
  P95: 0.515초
  P99: 0.518초

처리량: 180.00 요청/초
============================================================

💾 결과가 result/stt_load_test_results_20240101_120000.json에 저장되었습니다.
📊 히스토그램이 result/response_time_histogram_20240101_120000.png에 저장되었습니다.
📈 타임라인 그래프가 result/response_time_timeline_20240101_120000.png에 저장되었습니다.
```

### Resource 폴더 사용 모드

```
📁 설정 파일: config.yaml
🎵 오디오 설정: Resource 폴더 사용 모드
   Warm-up 폴더: resource/warm_up
   Test 폴더: resource/test
   Warm-up 파일 수: 3개
   Test 파일 수: 30개
ℹ️  Warm-up 요청 수를 파일 개수에 맞춰 조정: 5 → 3
ℹ️  총 요청 수를 파일 개수에 맞춰 조정: 50 → 33
🌐 API 설정: http://192.168.73.172:8000/v1/audio/transcriptions

🚀 STT 모델 로드 테스트 시작
   총 요청 수: 33
   Warm-up 요청 수: 3
   동시 요청 수: 1
   실제 측정 요청 수: 30
   매 요청마다 새로운 음성과 유사한 오디오 생성

🔥 Warm-up 단계 (3개 요청)...
   Warm-up 완료 (소요 시간: 0.15초)

📊 성능 측정 단계 (30개 요청)...
...
```

## 결과 파일 형식

### JSON 결과 파일

JSON 형식으로 저장되며 다음 정보를 포함합니다:

- **테스트 설정**: 총 요청 수, warm-up 요청 수, 동시 요청 수, 요청 지연 시간
- **성능 메트릭**: 평균, 중앙값, 최소, 최대, P95, P99 응답 시간 및 처리량
- **상세 결과**: 각 요청의 응답 시간, 성공/실패 여부, 에러 메시지

### 시각화 파일

- **히스토그램**: Cold start(warm-up)와 성능 테스트의 응답 시간 분포를 비교
- **타임라인 그래프**: 요청 인덱스별 응답 시간 추이를 시각화

## 주의사항

- Resource 폴더 모드에서는 `random_audio` 섹션의 설정이 무시되고, 파일 개수에 맞춰 요청 수가 자동 조정됩니다
- 랜덤 오디오 생성 모드에서는 `resource` 섹션의 설정이 무시됩니다
- 모든 결과 파일은 `result` 폴더에 저장되며, 폴더가 없으면 자동으로 생성됩니다
