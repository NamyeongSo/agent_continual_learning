# 네 방법의 공통 초기 actor context

새 ACE, PEACE, GEPA, SkillOpt 학습 CLI의 기본값은 `--actor-context shared-empty-v1`입니다.
같은 benchmark와 datapoint의 학습 전 actor 입력을 맞추고 초기 지침의 차이를 통제합니다.
기존 학습의 frozen snapshot과 결과는 해당 실행 당시의 context를 보존합니다.

## 입력 구성

```text
공통 actor 지침
+ benchmark별 공통 지침
+ 학습 정책 영역 (초기에는 비어 있음)
+ datapoint의 instruction와 환경 context
+ 환경이 제공한 public tool schema
```

공통 renderer는 `src/exgentic/actor_context.py`입니다.
ACE와 PEACE는 빈 native playbook의 section 구조를 내부에 유지하고, actor에게는 빈 정책으로 표시합니다.
GEPA의 seed candidate와 SkillOpt의 초기 skill 파일은 빈 문자열입니다.
빈 상태의 actor 문구는 네 방법 모두 `(No learned guidance yet.)`입니다.

학습 후에는 각 방법의 정책 원문을 같은 영역에 삽입합니다. GEPA/SkillOpt의 자유 형식 정책도
bullet ID 유무와 관계없이 표시합니다. ACE/PEACE의 bullet ID와 사용 통계는 학습 이후 원문에 남습니다.
각 방법의 reflector, curator, mutation, optimizer prompt와 정책 선택 알고리즘은 유지합니다.

공통 모드는 tool shortlisting을 끄고 모든 public tool schema를 사용합니다.
`enable_thinking=false`를 사용하며, 명시적인 thinking 활성화와 별도 system prompt override를 거부합니다.
Actor temperature, 모델, retry, token/turn 예산과 optimizer 하이퍼파라미터는 별도의 실행 설정입니다.
비교 실험에서는 이 값들과 benchmark 환경 policy·split·seed도 맞춰야 합니다.

AppWorld 공통 지침은 제공된 환경 policy의 실행 인터페이스를 따르도록 합니다.
GEPA의 과거 전용 prompt에 있던 Python REPL 사용 불가 문구와 task별 예제 전략은 공통 입력에 포함하지 않습니다.
AppWorld policy variant는 네 방법 모두 같은 `--appworld-policy` 파일로 지정합니다.

Terminal-Bench는 native Harbor/Terminus의 system/command protocol을 유지합니다.
네 방법이 native agent에 전달하는 task instruction 뒤에 같은 공통 지침과 학습 정책을 붙입니다.
외부 proxy의 system message가 native 모델의 system message를 대체한다고 해석하면 안 됩니다.

## 새 실험 실행

아래 예시는 공통 context를 명시하는 방법입니다. 현재 코드는 같은 값을 기본값으로 사용합니다.
출력 경로는 새로운 디렉토리를 사용합니다. 동일 비교군의 나머지 실행 인자도 별도로 맞춥니다.

```bash
uv run python scripts/ace/run_experiment.py \
  --mode isolated --seed 42 --benchmarks bfcl --all-train-tasks \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json \
  --output-dir outputs/shared_context_v1/ace/bfcl/s42 \
  --model openai/Qwen/Qwen3.6-35B-A3B --temperature 0.7 \
  --actor-context shared-empty-v1 --training-time --playbook-edit-mode add_update_delete

uv run python scripts/premise/run_experiment.py \
  --mode isolated --seed 42 --benchmarks bfcl --all-train-tasks \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json \
  --output-dir outputs/shared_context_v1/peace/bfcl/s42 \
  --model openai/Qwen/Qwen3.6-35B-A3B --base-url http://127.0.0.1:18006/v1 --temperature 0.7 \
  --actor-context shared-empty-v1 --past 3 --disable-alignment --playbook-edit-mode add_update_delete

uv run python scripts/baselines/run_experiment.py \
  --method gepa --benchmark bfcl --seed 42 \
  --manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json \
  --output outputs/shared_context_v1/gepa/bfcl/s42 --actor-context shared-empty-v1

uv run python scripts/baselines/run_experiment.py \
  --method skillopt --benchmark bfcl --seed 42 \
  --manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json \
  --output outputs/shared_context_v1/skillopt/bfcl/s42 --actor-context shared-empty-v1
```

ACE의 endpoint는 `OPENAI_API_BASE=http://127.0.0.1:18006/v1`로 맞춥니다.
네 entrypoint에 `--dry-run`을 추가하면 모델·환경을 호출하지 않고 설정과 초기 상태를 생성합니다.
위 예시는 context 인자를 설명하기 위한 것이며 전체 실험 큐를 생성하거나 자동 실행하지 않습니다.

## 재개와 고정 평가

실행 config에 context version, renderer SHA-256, benchmark별 초기 actor prompt SHA-256,
빈 정책 SHA-256, tool 선택 방식과 thinking 설정을 저장합니다.
Context가 다르거나 renderer hash가 달라진 checkpoint는 재개를 거부합니다.
기존 Epoch 1의 state에 새 context를 붙여 Epoch 2로 이어가면 초기 조건을 통제할 수 없으므로
공통 context 비교는 fresh Epoch 1부터 시작합니다.

새 profile은 학습, PEACE replay와 span policy-shift 평가, 최종 Test 및 checkpoint 평가에 전달됩니다.
Checkpoint stream 및 CP0 재사용 fingerprint에도 context metadata를 포함합니다.
ACE 고정 평가는 학습 때 기록한 benchmark 설정과 환경 policy를 사용합니다.

과거 prompt 조립 동작이 필요하면 `--actor-context legacy`를 명시합니다.
Framework의 `ACEAgent`/`PremiseAgent`와 `PremisePipeline` 기본값 및 context 필드가 없는
과거 config는 `legacy`로 해석합니다. 역사적 실험의 완전한 재현에는 그 실험의 frozen snapshot을 사용합니다.
Team prompt 파일은 원본 byte와 provenance 검증을 보존하며, 공통 renderer는 별도 모듈입니다.

## 검증

```bash
uv run pytest tests/agents/test_shared_actor_context.py tests/scripts/test_baseline_runtime.py -q
```

BFCL/AppWorld는 실제 네 actor의 첫 completion 직전 `messages`, `tools`, `tool_choice`를 비교합니다.
일반 observation과 image observation, 학습 후 자유 형식 정책, Terminal-Bench의 실제 actor action에서
조립한 native instruction, context 변경 거부를 검사합니다.
GEPA·SkillOpt의 설치된 공식 optimizer에서는 scripted transport로 초기 상태, 학습, Val 선택,
최종 Test 및 cache 기반 복원을 검사합니다. 이 검증에는 모델 응답이나 benchmark 환경 실행이 없습니다.
