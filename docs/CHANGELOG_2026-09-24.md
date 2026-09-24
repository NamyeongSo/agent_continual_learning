# 2026-09-24 연구 환경 변경 기록

이 문서는 2026-09-24 기준 이 저장소에서 진행한 ACE/Qwen 실험 준비와 검증을
팀원에게 인계하기 위한 기록이다. 출발점은 Microsoft Sico의
`labs/AgentStream/exgentic` 스냅샷이며, 이 포크의 설정 방법은
[TEAM_SETUP.md](TEAM_SETUP.md)를 참고한다. 아래 점수는 작은 smoke/탐색 실험의
결과이며 통계적으로 의미 있는 성능 추정치가 아니다.

## 1. 실험 범위와 모델

- 현재 집중하는 벤치마크는 AppWorld `test_challenge`, BFCL
  `multi_turn_base`, SWE-bench Verified 세 가지다.
- 모델은 로컬 vLLM의 `Qwen/Qwen3.6-35B-A3B`를 OpenAI 호환 API
  `http://127.0.0.1:8006/v1`에서 사용한다. Qwen용 스크립트는
  `enable_thinking=false`를 전달하고 시작 전에 tool-call API를 점검한다.
- `scripts/ace/run_experiment_qwen.sh`는 기본 no-GT ACE 실행,
  `scripts/ace/run_experiment_qwen_training_time.sh`는 공식 채점 피드백을
  사용하는 ACE 실행이다. 후자는 기본적으로 벤치마크당 5개 태스크와
  `MAX_TOKENS=4096`을 사용하며 환경변수로 변경할 수 있다.
- 최대 agent/environment step은 AppWorld 50, BFCL 50, SWE-bench 100이다.
  `ACE_APPWORLD_MAX_TURNS`, `ACE_BFCL_MAX_TURNS`,
  `ACE_SWEBENCH_MAX_TURNS`로 개별 조정하고 `MAX_TURNS`는 공통 fallback이다.
- **Sequential 실행 순서는 BFCL → AppWorld → SWE-bench**다.
  `scripts/utils/task_ordering.py`에 순서를 명시했다. 아래의 과거 5개씩
  실행은 변경 전 순서였으므로 현재 순서로 재실행한 결과와 구별한다.

## 2. ACE 학습 경로

- 기존 ACE reflector는 action/observation trajectory만 받았고 공식
  채점 결과는 받지 않았다. AppWorld의 최종 unit-test, BFCL checker,
  SWE-bench harness 점수는 별도로 계산·저장됐다.
- opt-in `training_time`을 추가했다. 이제 세션의 `score()`가 끝난 뒤
  `AgentInstance.receive_training_feedback()`으로 공식 `SessionScore`를
  ACE에 보내며, ACE reflector가 점수와 벤치마크별 세부 결과를 참고해
  reflection을 만든다. 다음 태스크에서 사용할 playbook은 그 뒤 갱신된다.
- 일반 ACE는 여전히 no-GT 프롬프트를 사용한다. A-Mem, AutoSkill,
  Harness, ReasoningBank 등 다른 baseline의 학습 동작은 변경하지 않았다.
- `training_time`은 작업 중 추가 테스트를 돌리는 옵션이 아니라
  **작업 종료 후 공식 채점 피드백을 학습에 쓰는 옵션**이다. 검증용
  태스크에는 이 업데이트를 적용하면 안 된다.

## 3. 실행·기록·안정성

- LiteLLM trace의 `dedup_v1` 형식은 매 turn 입력에 전체 history를
  반복 저장하는 대신 message/tool 정의를 해시로 한 번 기록하고
  call row는 참조만 저장한다. `iter_expanded_trace_rows()`로 복원할 수
  있다. 이전 측정에서 AppWorld 30-step trace는 약 2.48 MB에서
  0.63 MB로 줄었다(약 75%).
- AppWorld venv runner가 대량의 import 경고를 pipe에 쓰다가 기동이
  막히지 않도록 출력 임시 파일과 120초 health timeout을 적용했다.
- SWE-bench는 Docker daemon에 접근해야 한다. 현재 실험 컨테이너는
  Docker CLI를 사용해 별도의 원격 daemon과 SSH로 연결할 수 있다.
  Docker daemon의 data-root는 변경하지 않았다. 호스트 주소·SSH 키·
  비밀번호는 이 저장소에 넣지 않는다.
- ACE는 매 세션 `close()`에서 reflection/curator를 실행하고 세션별
  `playbook_checkpoint.json`을 저장한다. 과거 `online_metrics.jsonl`의
  bullet 수가 5개씩 같아 보인 것은 각 벤치마크의 5개 `evaluate()`가
  끝난 뒤 같은 최종 store 상태로 메트릭을 작성한 탓이다. 이는
  playbook이 5개 단위로만 갱신된다는 뜻이 아니다.
- 긴 실행의 `tee` 콘솔 로그는 호출 연결이 끊기면 갱신이 멈출 수 있다.
  실제 진행 확인에는 run의 `run/run.log`, 세션별 `results.json`, 최종
  `results.json`을 우선 사용한다. 콘솔 로그가 멈춘 것만으로 작업 종료를
  단정하지 않는다.

## 4. 고정 train/validation 태스크 ID

- 세 벤치마크 각각에서 전체 task ID를 정렬하고 벤치마크별 파생 seed로
  한 번 셔플한다. 앞 50개는 train, 다음 50개는 val이다.
- 실제 seed 42 ID 목록은
  [`scripts/utils/task_splits/seed42_train50_val50.json`](../scripts/utils/task_splits/seed42_train50_val50.json)에
  저장했다. 각 벤치마크의 train/val 교집합은 0개이며 재생성 결과가
  동일함을 확인했다. 선정 기준과 재생성 방법은
  [`scripts/utils/TASK_SPLITS.md`](../scripts/utils/TASK_SPLITS.md)에 있다.
- 이 split은 향후 GEPA/SkillOpt 공통 기준이다. 기존 ACE 및 baseline
  스크립트는 아직 이 manifest를 읽지 않으므로, 기존 결과의 태스크
  집합이 새 train 집합과 같다고 가정하면 안 된다.
- validation에서는 학습된 상태를 고정하고 점수만 측정한다. 공식
  채점 결과를 reflection·memory·skill 업데이트에 다시 넣으면 누수다.

## 5. 현재까지 확인한 실행 결과

| 실행 | 태스크 | 결과 | 비고 |
| --- | --- | --- | --- |
| 기존 no-GT sequential | 각 벤치마크 5개, 30 step | AppWorld 1/5, BFCL 3/5, SWE-bench 0/5 | 변경 전 benchmark 순서 |
| 기존 no-GT isolated SWE-bench | 5개, 100 step | benchmark score 0.2 | `results.json` 기준; 콘솔 `tee` 로그는 중간에 멈춤 |
| 새 training-time BFCL smoke | 1개, 50 step | score 0 | 채점 뒤 ACE reflection·curator 수행 |
| 새 training-time AppWorld smoke | 1개, 50 step | score 0.429 | 3/7 unit tests 통과, step limit 도달; 공식 피드백 전달 확인 |

실험 산출물은 `scripts/ace/outputs/` 아래에 있으며 Git에는 포함하지
않는다. 필요한 결과·checkpoint는 별도로 공유해야 한다. 코드 검증으로
train/val split, 순서, feedback 전달, ACE prompt, turn-limit 관련 테스트를
통과시켰다. 이 문서 시점에 **새 순서의 전체 50+50+50 실험**과
**training-time SWE-bench 전체 실험**은 수행하지 않았다.

## 6. 변경 파일 길잡이

- 실험 진입점: `scripts/ace/run_experiment.py`, `run_experiment_qwen*.sh`
- task 선정: `scripts/utils/task_ordering.py`, `create_task_splits.py`
- ACE feedback: `src/exgentic/core/orchestrator/session.py`,
  `src/exgentic/core/agent_instance.py`, `src/exgentic/agents/ace/`
- trace: `src/exgentic/integrations/litellm/trace_logger.py`
- AppWorld 기동: `src/exgentic/adapters/runners/venv.py`,
  `src/exgentic/benchmarks/appworld/appworld_benchmark.py`
