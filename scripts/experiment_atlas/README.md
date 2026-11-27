# Experiment settings atlas

Qwen3.6 isolated Epoch 1 / Epoch 2 continuation 실험을 탐색하는 읽기 전용 HTML 서버입니다.
실험을 실행하거나 모델에 요청하지 않고, 저장된 설정·frozen source·벤치마크 파일을 읽습니다.

대상 cohort:
`outputs/qwen_epoch2_continuation_four_baselines_20261109`

## 생성과 실행

저장된 데이터와 benchmark 설치가 있는 Exgentic checkout에서 실행합니다.

```bash
uv run python -m scripts.experiment_atlas.build
uv run python -m scripts.experiment_atlas.server --host 0.0.0.0 --port 8880
```

기본 snapshot 위치는 `outputs/experiment_settings_atlas_20261111/`입니다.
서버의 기본 bind 주소는 `127.0.0.1`이며 위 명령은 외부 접속을 허용합니다.
실험 기록장 서버의 8878 포트와 독립적으로 동작합니다.

페이지에는 다음 자료가 있습니다.

- ACE AUD, PEACE AUD w/o alignment, GEPA, SkillOpt의 학습 흐름과 하이퍼파라미터
- 36개 benchmark·method·seed 실행의 Epoch 1 / 2 config, command, 실제 task 순서
- 원본 706개 task와 실험에 사용한 389개 task의 instruction, 환경·도구 정보
- BFCL의 후속 user turn을 포함한 instruction 검색과 전체 CSV 다운로드
- 고정 membership, seed별 shuffle, AppWorld scenario/variant 구조와 실제 split 분포
- 모델 launcher, parser bridge, 호출 인자, generation defaults, actor token/turn 예산
- 프롬프트 전문, 환경 policy hash, frozen source와 SHA-256
- 기존 토큰 감사의 학습·평가 구분 표
- 새 공통 reflector cohort의 실제 LLM 반환값: reasoning·최종 출력·tool call·호출별 토큰

설정과 instruction은 snapshot입니다. 변경 후 `build`를 다시 실행하고 이 서버를 재시작합니다.
재시작 전에는 해당 서버의 process identity와 command를 확인합니다.
코드와 실험의 FD 한도를 자동으로 바꾸거나 학습 프로세스를 재시작하지 않습니다.
생성 시 원본 instruction 존재 여부, task 수, ordered manifest hash와 Epoch 2 순서를 검증합니다.

실행 상태와 모델 endpoint 상태는 `/api/live`에서 읽으며 15초 동안 캐시합니다.
화면은 30초마다 상태를 확인합니다. 모델의 현재 가용성과 마지막 launcher의 설정을 구분합니다.
실행 중 JSON 파일이 잠시 불완전하면 `unavailable`로 표시합니다.

## 데이터 노출 범위

임의 파일 조회·실험 실행·설정 변경 API는 없습니다.
출처 조회는 builder가 등록한 파일만 허용하며 SSH Docker 접속 환경과 인증 값은 제외합니다.
LLM 조회는 새 cohort의 manifest에 등록된 실행 아래의 model trace·PEACE llm_calls·optimizer analysis만 허용합니다.
요청의 HTTP 인증 필드와 환경 설정은 제공하지 않으며, 공개 tool parameter schema는 그대로 보존합니다.
벤치마크의 공개 tool schema는 parameter 이름을 포함해 보존합니다.
Terminal-Bench의 숨겨진 test/solution 파일과 AppWorld 정답 코드는 수집하지 않습니다.
이 페이지는 연구 서버의 실험 설정과 공개 task instruction을 제공하므로 접속 가능한 사용자에게 보입니다.

주요 API:

```text
GET /api/health
GET /api/settings
GET /api/live
GET /api/tasks?benchmark=bfcl&split=train&selected=selected&q=instruction&page=1
GET /api/task/{benchmark}/{task_id}
GET /api/source/{registered_id}
GET /download/settings.json
GET /download/tasks.csv
GET /api/llm/jobs
GET /api/llm/sources?job=ace_aud_bfcl_s42&kind=analysis&split=train&page=1
GET /api/llm/records?job=ace_aud_bfcl_s42&source=REGISTERED_SOURCE_ID
GET /api/llm/call?job=ace_aud_bfcl_s42&source=REGISTERED_SOURCE_ID&line=0
```

`http://166.104.110.49:8880/#llm-calls`에서 현재 Fresh 1 epoch cohort의 36개 실행을 선택합니다.
방법론·benchmark·seed·역할·Train/Val/Test로 기록을 좁힌 뒤 개별 호출을 읽습니다.
Reasoning과 최종 출력을 나란히 표시하고, 요청·원본 응답·호출 정보 탭 및 JSON 다운로드를 제공합니다.
호출 링크를 복사하면 해당 기록과 호출을 직접 열 수 있습니다. 최신 로그는 목록 새로고침으로 확인합니다.

LiteLLM의 문자열화된 SDK 객체는 제한된 AST를 데이터로 복원하며 실행하지 않습니다.
응답 ID로 bridge 계측을 연결해 실제 temperature·thinking·effort·추론/비추론 토큰을 표시합니다.
미기록 필드는 추정하지 않습니다. 반환값을 확인하기 위해 새 모델 요청을 생성하지 않습니다.
큰 실행 전체 대신 선택한 trace를 읽고, JSONL의 미완성 마지막 줄은 다음 갱신에서 다시 읽습니다.

## 확인

```bash
uv run pytest tests/scripts/test_experiment_atlas.py -q
uv run pytest tests/scripts/test_llm_call_browser.py -q
uv run ruff check scripts/experiment_atlas tests/scripts/test_experiment_atlas.py
```

검색이 후속 user turn까지 포함하는지, 필터가 교집합으로 적용되는지,
출처 allowlist와 파일 경로 차단, 읽기 전용 API, 부분 JSON write 처리를 검사합니다.
로컬 브라우저 검증 결과와 화면 캡처는 생성된 snapshot 폴더에 보관합니다.
