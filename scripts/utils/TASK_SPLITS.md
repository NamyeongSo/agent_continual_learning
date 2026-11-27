# Train / Val / Test 분할 및 셔플 규칙

## 1. 벤치마크별 구간

아래 번호는 **전체 task ID를 셔플한 뒤의 1부터 시작하는 순번**이다.
원래 task ID에 들어 있는 숫자나 데이터셋의 원래 나열 순서를 의미하지 않는다.

| 벤치마크 | 대상 subset | Train | Val | Test | 개수 |
|---|---|---|---|---|---|
| BFCL | `multi_turn_base` | 1–50 | 51–100 | 101–150 | 50 / 50 / 50 |
| AppWorld | `test_challenge` | 1–50 | 51–100 | 101–150 | 50 / 50 / 50 |
| SWE-bench | `princeton-nlp/SWE-bench_Verified` | 1–50 | 51–100 | 101–150 | 50 / 50 / 50 |
| Terminal-Bench 2.0 | pinned `2.0` release, 89 tasks | 1–30 | 31–60 | 61–89 | 30 / 30 / 29 |

세 큰 벤치마크에서 151번째 이후의 task는 이번 실험에서 사용하지 않는다.
Terminal-Bench는 89개 전체를 사용한다. 이 분할은 위 subset 내부에서 정의한
연구용 분할이며, 벤치마크가 원래 제공하는 공식 train/test 구분과는 별개다.

## 2. 재현 가능한 셔플

각 벤치마크와 split seed 조합에 다음 절차를 적용한다.

1. 해당 subset의 전체 task ID를 문자열로 읽고, 중복 ID가 없는지 확인한다.
2. ID를 문자열 기준 오름차순으로 정렬한다. 숫자 부분을 따로 정렬하지 않는다.
3. 벤치마크 이름과 seed로 독립적인 난수 seed를 만든다.
4. 정렬한 전체 목록을 한 번 셔플하고, 위 구간으로 train → val → test를 자른다.
5. 정확한 ID 목록을 JSON에 저장하고 모든 방법이 같은 JSON을 사용한다.

```python
import hashlib
import random

benchmark_seed = int(
    hashlib.md5(f"{seed}_{benchmark_slug}".encode()).hexdigest(), 16
) % (2**31)
shuffled = sorted(task_ids)
random.Random(benchmark_seed).shuffle(shuffled)

# BFCL / AppWorld / SWE-bench
train, val, test = shuffled[:50], shuffled[50:100], shuffled[100:150]

# Terminal-Bench 2.0
train, val, test = shuffled[:30], shuffled[30:60], shuffled[60:89]
```

Train/val/test는 별도로 섞어서 뽑지 않고 **한 셔플 목록의 겹치지 않는 구간**에서 뽑는다.
같은 벤치마크와 seed에서는 세 split이 서로 겹치지 않는다. Task ID 단위 분할이며
시나리오, 템플릿, 저장소를 그룹으로 묶는 분할은 적용하지 않는다.

## 3. 요청한 9개 benchmark/seed key

파일: [bfcl_appworld_terminalbench2_seed42_44.json](task_splits/bfcl_appworld_terminalbench2_seed42_44.json)

이 파일은 **BFCL, AppWorld, Terminal-Bench 2.0 × seed 42, 43, 44** 조합만 담는다.
SWE-bench는 위 분할 규칙에는 포함되지만 이 9개 key 파일에는 포함되지 않는다.

| 벤치마크 | Key |
|---|---|
| BFCL | `bfcl_42`, `bfcl_43`, `bfcl_44` |
| AppWorld | `appworld_42`, `appworld_43`, `appworld_44` |
| Terminal-Bench 2.0 | `terminalbench2_42`, `terminalbench2_43`, `terminalbench2_44` |

각 key의 값은 `train`, `val`, `test` 세 배열이다. 각 배열에는 실제 task ID가 들어 있다.
**Seed마다 전체 목록을 다시 셔플하므로 split 구성원도 달라진다.** 서로 다른 seed 사이에는
task가 겹칠 수 있다. 각 seed의 학습 상태를 따로 초기화하고, 같은 seed에서는 ACE,
PREMiSE 및 다른 방법이 동일한 ID를 사용한다.

```bash
.venv/bin/python scripts/utils/create_task_splits.py \
  --seeds 42 43 44 --benchmarks bfcl,appworld,terminalbench2 \
  --output scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json
```

```python
import json
from pathlib import Path

path = Path("scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json")
splits = json.loads(path.read_text())
train_ids = splits["terminalbench2_43"]["train"]
val_ids = splits["terminalbench2_43"]["val"]
test_ids = splits["terminalbench2_43"]["test"]
```

## 4. 실행용 manifest와 실행 순서

[seed42_train_val_test.json](task_splits/seed42_train_val_test.json)은 네 벤치마크의
seed-42 분할에 subset과 split 개수까지 담은 실행용 manifest다. 생성 명령은 다음과 같다.

```bash
.venv/bin/python scripts/utils/create_task_splits.py \
  --seed 42 --output scripts/utils/task_splits/seed42_train_val_test.json
```

ACE의 `--task-split-manifest`에는 실행용 형식 또는 9개 key JSON을 전달한다.
9개 key JSON에서는 실행의 `--seed`와 일치하는 `benchmark_seed` key를 읽는다.
PREMiSE에는 실행용 형식을 전달한다. 모든 train ID를 실행하려면
`--all-train-tasks`를 사용한다. 세 큰 벤치마크는 50개씩, Terminal-Bench는 30개가 선택된다.
작은 pilot에는 `--num-tasks 5`처럼 실행할 train 개수를 명시한다.

Split seed와 실행 순서 seed는 구분한다. 고정된 manifest를 불러온 뒤 실행 순서 seed를
바꾸면 split 구성원은 바뀌지 않고 train 내부 순서가 바뀐다. Sequential은 벤치마크별로
묶고, interleaved는 같은 벤치마크 내부 순서를 유지하며 벤치마크 사이를 섞는다.
Train 일부만 실행하는 경우 `--num-tasks`는 순서 셔플 후 앞 N개를 선택한다.

## 5. Train / Val / Test 사용 원칙

- **Train:** playbook, memory, skill 업데이트와 학습용 replay에 사용한다.
- **Val:** 학습 상태를 고정한 채 평가한다. 설정이나 checkpoint 선택에 사용할 수 있지만
  val task의 정답, trajectory, 점수를 학습 업데이트나 replay 데이터로 넣지 않는다.
- **Test:** 최종 설정과 checkpoint를 결정한 뒤 평가한다. 학습이나 설정 선택에 사용하지 않는다.
- **분할 고정:** 성능이나 오류 여부를 보고 다른 task로 교체하지 않는다. 실행 오류는 해당 ID의
  오류로 기록하고, 나머지 task를 계속 실행한다.

## 6. 이전 실험과의 관계

기존 `seed42_train50_val50.json`, `seed42_train50_val39_terminalbench2.json`,
`seed42_terminalbench2_pilot5.json`은 과거 실험 재현용으로 보존한다.
Seed 42에서 BFCL/AppWorld/SWE-bench의 기존 train 50개와 val 50개는 새 분할에서도 유지된다.
Terminal-Bench의 이전 50/39 분할은 현재 30/30/29 분할과 다르므로 이전 점수를 새 프로토콜의
점수로 간주하지 않는다.

새 5개 pilot manifest는
[seed42_terminalbench2_train30_pilot5.json](task_splits/seed42_terminalbench2_train30_pilot5.json)이다.
기존 pilot의 5개 ID는 모두 새 train 30개에 속하며, val/test는 새 전체 분할의 ID를 유지한다.
이전 ACE 실험은 split seed 42를 고정하고 실행 순서만 42–44로 바꿨으므로,
이번 9개 key 파일의 seed별 독립 분할과 구분해서 보고한다.

## 7. ACE 6개 실험 동시 실행

기존 vLLM 서버를 사용해 sequential/interleaved × seed 42/43/44를 동시에 실행한다.
각 실행은 독립 프로세스, playbook, 출력 디렉터리, 1000개 포트 구간을 사용한다.
Sequential 순서는 BFCL → AppWorld → Terminal-Bench이며, interleaved는 같은 seed의
벤치마크 내부 순서를 유지하면서 벤치마크 사이를 섞는다.

```bash
# 실제 실행 전 6개 작업의 train 개수와 포트 구간 확인
.venv/bin/python scripts/ace/run_parallel_bfcl_appworld_terminalbench2.py --dry-run

# Docker 연결과 기존 모델 서버 확인 후 6개 백그라운드 프로세스 시작
DOCKER_HOST=ssh://your-docker-host \
  .venv/bin/python scripts/ace/run_parallel_bfcl_appworld_terminalbench2.py
```

기본값은 Qwen 서버 `http://127.0.0.1:8006/v1`, thinking 비활성화, training-time ACE,
출력 토큰 8192, BFCL/AppWorld 50 step, Terminal-Bench Terminus-2 100 turn이다.
각 실험은 train 130개, 전체 6개 실험은 train 실행 780개다. 학습이 모두 끝나면 아래
checkpoint test 평가가 자동으로 시작된다. Val은 이 실행에서 사용하지 않는다.
RPC에 사용하는 Pydantic, pydantic-core, cloudpickle 버전은 호스트와 세 벤치마크 환경이
같아야 하며, launcher가 이를 확인한다. Launcher는 vLLM 서버를 시작하거나 종료하지 않는다.

결과는 `scripts/ace/outputs/ace_bfcl_appworld_tb2_seed42_44_<timestamp>/`에 저장한다.
`launch_manifest.json`에 6개 PID, task 순서, 원본 split SHA-256, 로그 위치가 기록된다.
각 `<mode>_s<seed>/experiment.log`와 session 결과 파일로 진행 상황을 확인할 수 있다.

## 8. 학습 후 checkpoint test 평가

두 모드 모두 학습 task 50·100·130개를 처리한 시점의 playbook을 고정한다.
Sequential에서는 각각 BFCL·AppWorld·Terminal-Bench 학습 완료 시점이며,
interleaved에서는 같은 task 개수 시점이다. 각 checkpoint에서 해당 seed의 BFCL test 50개,
AppWorld test 50개, Terminal-Bench test 29개를 모두 평가한다.
총 6실험 × 3checkpoint × 3벤치마크 = 54개 평가 묶음, test task 실행 2,322개다.

각 세션의 완료된 `agent/playbook_checkpoint.json`에서 checkpoint를 복사하므로
나중에 덮어쓰이는 최종 playbook을 중간 시점 평가에 사용하지 않는다.
Test는 `evaluation_mode=True`, `training_time=False`로 실행하여 reflection·curation 없이
playbook을 고정한다. 각 seed의 train/val/test ID와 split SHA-256을 검증한다.

Generator·reflector·curator에 실제 context length 오류가 발생하면 현재 task를 건너뛰고
다음 task로 진행한다. 해당 task의 playbook 변경은 되돌리고 `agent/task_skip.json`에 기록한다.
Terminal-Bench trajectory는 턴별·전체 잘림이나 고정 100k cutoff 없이 전달한다.

`test_evaluation/status.json`에서 대기/평가/완료 상태를 확인할 수 있고,
각 `<mode>_s<seed>/checkpoint_<050|100|130>/<benchmark>/summary.json`에 결과가 저장된다.
학습 중단 후 `run_experiment.py --resume-sessions`로 다시 실행하면 마지막으로 완료된
세션 checkpoint와 task prefix를 복구하고 진행 중이던 task부터 다시 실행한다.

기존 6실험에 평가 controller를 연결하는 명령은 다음과 같다.

```bash
DOCKER_HOST=ssh://your-docker-host \
  .venv/bin/python scripts/ace/evaluate_test_checkpoints.py \
  --launch-manifest <실험_디렉터리>/launch_manifest.json
```
