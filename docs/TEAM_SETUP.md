# 팀원용 환경 설정·실험 가이드

이 문서는 Linux/Docker 환경에서 이 연구 포크를 받아 Qwen ACE 실험을
재현하는 절차다. 기준 벤치마크는 BFCL `multi_turn_base`, AppWorld
`test_challenge`, SWE-bench Verified이며 **sequential 순서는
BFCL → AppWorld → SWE-bench**다. 변경 배경과 이미 수행한 검증은
[2026-09-24 변경 기록](CHANGELOG_2026-09-24.md)을 참고한다.

## 1. 준비물 및 기본 설치

- Python 3.11 이상(현재 검증 환경은 3.12), `uv`, `git`, `git-lfs`,
  `curl`, SSH client. SWE-bench에는 Docker CLI와 접근 가능한 Docker
  daemon이 필요하다. GPU/vLLM 서버는 별도로 준비한다.
- AppWorld 설치 스크립트는 Git LFS로 데이터 파일을 받고, BFCL은
  Gorilla 저장소의 고정 commit을 설치한다. 설치·데이터 다운로드에는
  네트워크와 충분한 디스크 공간이 필요하다.

```bash
git clone https://github.com/NamyeongSo/agent_continual_learning.git
cd agent_continual_learning
uv sync --frozen
source .venv/bin/activate
exgentic list benchmarks
```

`uv.lock`이 포함돼 있다. 가상환경은 이 clone 안의 `.venv/`에 생긴다.
가상환경·벤치마크 데이터·출력은 Git에 올리지 않는다. 이미 설치된
시스템 Docker나 GPU 소프트웨어는 `uv sync`가 대신 설치하지 않는다.

## 2. 벤치마크 의존성/데이터 설치

다음 명령은 각 benchmark 전용 격리 venv를 만들고 패키지와 제공된
`setup.sh`를 실행한다. 기본 위치는 사용자 홈의
`.exgentic/benchmarks/<benchmark>/venv/`다.

```bash
exgentic install --benchmark bfcl
exgentic install --benchmark appworld
exgentic install --benchmark swebench
```

| 벤치마크 | 설치 내용 및 확인할 점 |
| --- | --- |
| BFCL | `src/exgentic/benchmarks/bfcl/setup.sh`가 Gorilla/BFCL 고정 commit을 받고 `bfcl_eval`과 추가 의존성을 설치한다. 데이터/코드는 benchmark 환경 안에 설치된다. |
| AppWorld | `src/exgentic/benchmarks/appworld/setup.sh`가 Git LFS가 있는 AppWorld 고정 commit을 설치하고 데이터를 다운로드한다. LFS가 없으면 실패한다. |
| SWE-bench | `requirements.txt`가 SWE-bench harness v4.1.0과 mini-swe-agent v1.17.0을 설치한다. 별도 `setup.sh`는 **없다**. 데이터셋은 Hugging Face에서 읽고 harness는 Docker 이미지를 사용하므로 네트워크·Docker 접근이 필요하다. |

설치가 끝나면 다음으로 task 목록이 읽히는지 확인하고 공통 split을
재현할 수 있다.

```bash
python scripts/utils/create_task_splits.py \
  --seed 42 --train-count 50 --val-count 50 \
  --output scripts/utils/task_splits/seed42_train50_val50.json
```

같은 manifest라면 `Unchanged`를 출력한다. 다르게 재생성되면 덮어쓰지
않고 실패한다. 이는 벤치마크 데이터 버전이나 설치 상태가 달라졌다는
신호이므로 기존 ID 목록을 임의로 바꾸지 말고 원인을 확인한다.

## 3. Qwen vLLM 연결

Qwen용 스크립트는 기본적으로 이 실행 환경에서
`http://127.0.0.1:8006/v1`을 호출한다. vLLM 서버가 다른 머신에
있으면 `OPENAI_API_BASE`를 그 머신에서 **실행 컨테이너가 접근 가능한**
주소로 설정한다. 컨테이너 안의 `127.0.0.1`은 컨테이너 자신이다.

서버에는 `Qwen/Qwen3.6-35B-A3B` 모델과 자동 tool-call 기능이 필요하다.
현재 Qwen 스크립트가 검사하는 기능은 `--enable-auto-tool-choice`,
`--tool-call-parser qwen3_coder`이며, 현재 서버는 reasoning parser도
사용한다. vLLM 버전에 맞는 chat template/parser 조합을 먼저 확인한다.
스크립트는 모델 호출에 `enable_thinking=false`를 전달한다.

```bash
export OPENAI_API_BASE=http://127.0.0.1:8006/v1
export OPENAI_API_KEY=EMPTY
curl -fsS "${OPENAI_API_BASE}/models"
```

API 키가 필요한 서버라면 각자 환경변수로 지정한다. 키·SSH 개인키·
비밀번호를 저장소나 실행 로그에 넣지 않는다.

## 4. Docker/SWE-bench 연결

SWE-bench harness는 Docker daemon을 사용한다. 실행 환경 안에서
`docker info`가 성공해야 한다. 아래 둘 중 하나를 선택한다.

1. 실행 컨테이너에 Docker socket을 마운트하고 Docker CLI를 설치한다.
2. SSH로 접근 가능한 별도 Docker daemon을 사용한다. SSH 키/권한을
   개인 환경에 설정하고 `DOCKER_HOST=ssh://<ssh-alias-or-user@host>:<port>`
   또는 `ACE_DOCKER_HOST`를 export한다.

```bash
docker info
```

현재 연구 환경에서는 두 번째 방식을 사용했지만, 원격 호스트 주소와
인증정보는 공개 저장소에 포함하지 않는다. 팀 내부에서 별도로 전달받아
설정한다. SSH Docker 연결에서 mini-swe-agent가 `paramiko` 누락을
보고하면 benchmark venv에 설치한다.

```bash
uv pip install --python "$HOME/.exgentic/benchmarks/swebench/venv/bin/python" paramiko
```

`src/exgentic/benchmarks/swebench/readme.md`의 오래된 `setup.sh`
안내와 달리 현재 소스에는 SWE-bench `setup.sh`가 없다. 위의
`exgentic install --benchmark swebench`를 사용한다.

## 5. ACE 실험 실행

다음은 세 벤치마크를 sequential로 실행한다. 기본 순서는
BFCL → AppWorld → SWE-bench다. 기본 최대 step은 각각 50/50/100이다.

```bash
# trajectory만으로 reflection하는 기존 ACE
NUM_TASKS=5 MAX_TOKENS=4096 scripts/ace/run_experiment_qwen.sh

# 공식 채점 결과를 종료 후 reflection에 사용하는 training-time ACE
NUM_TASKS=5 scripts/ace/run_experiment_qwen_training_time.sh
```

`NUM_TASKS`는 벤치마크당 task 수다. `MODE=isolated` 또는
`MODE=interleaved`, `ACE_BENCHMARKS=bfcl,appworld,swebench`,
`SEED=42`, `MAX_TOKENS=4096`, `OUTPUT_BASE=...`를 환경변수로 줄 수 있다.
`ACE_APPWORLD_MAX_TURNS`, `ACE_BFCL_MAX_TURNS`,
`ACE_SWEBENCH_MAX_TURNS`로 상한을 개별 조정한다. 예를 들어
`ACE_BENCHMARKS=bfcl NUM_TASKS=1`은 BFCL smoke run이다.

두 실행 스크립트는 현재 **legacy ACE task selector**를 사용한다.
아래 고정 train/val manifest의 train ID와 동일하다고 가정하지 말 것.
GEPA/SkillOpt를 추가할 때는 manifest를 읽어 `train` ID를 학습하고
checkpoint를 고정한 뒤 `val` ID에서 점수만 측정하도록 구현해야 한다.
Validation score를 memory/playbook/skill 업데이트에 넣지 않는다.

## 6. 결과와 점검

- `scripts/ace/outputs/<run-tag>/experiment_config.json`:
  실행 설정과 task order.
- 같은 위치의 `online_metrics.jsonl`: task별 score, token, step.
- `<run-id>/results.json`: 집계 결과. `sessions/<session-id>/results.json`:
  세션별 공식 점수와 세부 메타데이터.
- `sessions/<session-id>/agent/playbook_checkpoint.json`: 매 세션의
  ACE playbook 상태.
- `sessions/<session-id>/benchmark/`: 벤치마크별 채점 파일과 로그.
- `<run-id>/run/run.log`: 긴 실행의 실제 진행 확인에 유용하다. 외부
  `tee` 콘솔 로그가 멈춰도 이 로그와 세션 결과를 확인한다.

Qwen 스크립트는 `EXGENTIC_LLM_TRACE_FORMAT=dedup_v1`을 켠다.
요청 history를 한 turn마다 중복 저장하지 않는 형식이며,
`src/exgentic/integrations/litellm/trace_logger.py`의
`iter_expanded_trace_rows()`로 완전한 요청을 복원할 수 있다.

코드 검증:

```bash
python -m pytest -q tests/scripts/test_task_splits.py \
  tests/agents/ace/test_training_time.py \
  tests/agents/ace/test_turn_limits.py \
  tests/core/orchestrator/test_training_feedback.py
```

## 7. 재현 시 주의할 점

- `training_time` 점수와 no-GT ACE 점수는 서로 다른 학습 조건이다.
- AppWorld `score`는 부분 통과율일 수 있으며 `success=false`여도
  0보다 큰 점수가 나온다. SWE-bench의 세션 완료와 issue resolved도
  같은 뜻이 아니다. 보고서에는 공식 `score`를 기준으로 쓴다.
- 과거 `online_metrics.jsonl`의 playbook bullet 수는 한
  `evaluate()` 그룹이 끝난 뒤 기록되어, 각 세션 중간의 실제 bullet
  수와 다를 수 있다. 세션별 checkpoint를 사용한다.
- 이 저장소에 출력·데이터·Docker 이미지·가상환경·개인 인증정보는
  포함하지 않는다. 팀원이 실제 결과를 재현하려면 별도 다운로드와
  실험 실행이 필요하다.
