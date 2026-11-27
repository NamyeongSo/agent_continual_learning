# ACE AppWorld Context Proposal Bottleneck: 준비 설정과 진행 로그

2026-11-16 사용자의 실행 승인 후 실험을 시작했다. 공통 설정은
`scripts/ace/configs/appworld_context_proposal_bottleneck.json`에 있다.
실행기는 `scripts/ace/run_context_proposal_bottleneck.py`이며 `--prepare`로 source/policy/split/GT를 고정하고,
출력 디렉터리의 frozen 실행기에 `--run`을 전달해 실행 또는 재개한다.
실제 진행 상태는 출력 디렉터리의 `status.json`과 `progress.log`를 확인한다.

현재 출력 디렉터리:
`outputs/ace_appworld_context_proposal_bottleneck_teammate_qwen38_s42_turncap_pct_v2_20261116`.

사용자의 후속 지시에 따라 40턴 종료 시 강제 0점 대신 그 시점의 공식 개별 테스트 PCT를 사용한다.
이 예외는 이 실험의 초기/후속 평가에 동일하게 적용하며 입력/출력 예산 등 다른 실패 규칙은 유지한다.
기존 forced-zero 실행과 그림은 이전 디렉터리에 보존했다. 초기 Trajectory 250개를 재생성하지 않고
저장된 40턴 상태 20개만 공식 평가기로 재채점했다. 입력이 바뀌지 않은 완료 Context 187개를 재사용하며,
변경된 20개 Trajectory에 대응하는 두 설정의 40개 제안은 새 공식 피드백을 사용한다.
재채점과 재사용 근거는 `turn_cap_rescoring/`, `migration_manifest.json`에 기록했다.

## 실험 설정

- 고정 AppWorld train 50개만 사용한다. Val/Test는 Reflection 생성이나 Context Update에 사용하지 않는다.
- 모델은 `openai/Qwen/Qwen3.8-27B-FP8`, endpoint는 `http://127.0.0.1:8006/v1`이다.
- Actor/Reflector/Curator 모두 최근 ACE의 호출 설정을 사용한다:
  temperature 1, reasoning xhigh, thinking/preserve-thinking 활성화, 출력 무제한.
  나머지 sampling/timeout/retry 설정도 준비 JSON에 원본 그대로 보관했다.
- 물리 GPU 1만 사용한다. 실행 전에 endpoint의 실제 장치를 확인한다.
- Actor는 턴당 입력 최대 100,000 tokens, Trajectory당 최대 40턴이다.
- AppWorld는 GT reference 검증 환경과 맞춘 SQLModel 0.0.24를 실험 전용 dependency overlay로 사용한다.
  공용 설치의 0.0.47은 바꾸지 않으며 overlay와 source 파일의 해시를 기록한다.
- 환경 policy는 `scripts/premise/policies/appworld_teammate_completion.txt`이며,
  초기 평가와 두 설정의 재평가에서 같은 원문을 사용한다. 원문과 SHA-256을 JSON에 고정했다.
- 지표는 PCT: 공식 개별 테스트의 통과 비율이다. Task별 초기/후속 성능은 Rollout 5개의 PCT 평균이다.

## 단계와 Queue

| 단계 | 작업 | Queue 수 | 동시성 |
| --- | --- | ---: | ---: |
| 1 | Task 50개 × 초기 Rollout 5개, 두 설정이 공유 | 250 | 50 |
| 2 | 초기 Trajectory별 Reflection → Update × w/o GT, w/ GT | 500 | 50 |
| 3 | Updated Context별 해당 Task 재추론 5회 | 2,500 | 50 |

각 단계 전체가 끝나야 다음 단계로 이동한다. 2단계의 작업 하나는 Reflection과 Update를 순차 수행하는 체인이다.
각 Context는 동일한 초기 Context에서 독립적으로 생성하며 후보 사이에 지식을 누적하지 않는다.
1단계 250회와 3단계 2,500회를 합쳐 Actor Trajectory는 총 2,750개다.

w/ GT는 `REFLECTOR_PROMPT_WITH_GT`, w/o GT는 기존 `REFLECTOR_PROMPT_TRAINING_TIME`을 사용한다.
w/ GT에는 같은 Task의 검증된 authored reference trajectory를 추가한다. 자료는
`outputs/appworld_train50_reference_solutions_20261112/solutions.jsonl`에 있다.

## 사람이 보는 로그

콘솔과 실험 출력 디렉터리의 `progress.log`에 같은 줄을 출력한다.
30초마다 출력하고, 단계 시작/완료 및 인프라 오류 발생 시에는 즉시 출력한다.
평가 하나가 끝날 때마다 줄을 추가할 필요는 없다.

아래는 실제 실행 결과가 아닌 표시 예시다.

```text
[10:00:00] START stage=1/3 initial_rollouts total=250 concurrency=50
[10:05:00] stage=1/3 initial_rollouts done=100/250 (40.0%) running=50 pending=100 errors=0 elapsed=00:05:00
[10:12:00] DONE stage=1/3 initial_rollouts done=250/250 (100.0%)
[10:12:00] START stage=2/3 reflection_update total=500 concurrency=50
[10:15:00] stage=2/3 reflection_update done=200/500 (40.0%) running=50 pending=250 errors=0 elapsed=00:03:00
           w/o_gt=100/250 w/gt=100/250
[10:20:00] DONE stage=2/3 reflection_update done=500/500 (100.0%)
[10:20:00] START stage=3/3 updated_context_rollouts total=2500 concurrency=50
[10:25:00] stage=3/3 updated_context_rollouts done=500/2500 (20.0%) running=50 pending=1950 errors=0 elapsed=00:05:00
           w/o_gt=250/1250 w/gt=250/1250
```

- `done`: 유효한 결과를 저장한 작업 수다. 공식 평가 PCT가 0이어도 정상 완료로 센다.
  2단계는 Reflection과 Updated Context가 모두 저장돼야 작업 하나가 완료된다.
- `running`: 현재 실행 중인 작업 수다. 2단계는 체인 수이며 모델 호출 수가 아니다.
- `pending`: 아직 실행되지 않았거나 재시도를 기다리는 작업 수다.
- `errors`: 인프라 오류로 유효한 결과를 확보하지 못한 작업 수다. PCT 실패와 구분한다.
  오류 줄에는 Task/설정/반복 식별자와 짧은 원인을 남긴다.
- `elapsed`: 현재 단계의 경과 시간이다. 저장 시각은 ISO 8601 offset을 함께 기록한다.
  현재 실행 환경의 timezone 데이터가 반환하는 offset은 UTC(`+00:00`)이다.
- 2·3단계는 설정별 완료 개수도 함께 출력한다.

같은 내용을 `status.json`에도 최신 상태 하나로 원자적으로 갱신한다.
여기에는 갱신 시각, 실행 상태, 현재 단계 및 단계별 집계를 보관한다.
장시간 작업 중에도 heartbeat가 갱신되므로 진행률 변화가 없어도 로그 갱신 여부를 볼 수 있다.
인프라 오류로 필요한 결과가 누락되면 해당 단계를 완료로 표시하거나 다음 단계로 넘어가지 않는다.
중단/오류 시에도 마지막 진행 상태를 보존한다.

다음 명령으로 현재 진행 상태를 확인한다.

```bash
tail -f outputs/ace_appworld_context_proposal_bottleneck_teammate_qwen38_s42_turncap_pct_v2_20261116/progress.log
```

## 완료 결과 분석

`scripts/ace/analyze_context_proposal_bottleneck.py`는 완료된 train50/G5/후보당 3회 bank만 읽는다.
원본의 정책 해시·task·반복 개수·공식 테스트 분수를 검증하고 새 inference 없이 표와 그림을 생성한다.

```bash
uv run --no-sync python -m scripts.ace.analyze_context_proposal_bottleneck \
  --run <완료된-run> --output <완료된-run>/analysis/reflection_final
```

후보 효과는 재평가 3회 평균에서 해당 task의 초기 5회 평균을 뺀 값이다. task별 후보 5개를 평균한 뒤
50개 task를 같은 비중으로 집계하며, GT 조건은 같은 task를 대응시킨다. Bootstrap은 후보나 rollout이 아닌
task를 10,000회 재표집한다. 관측 best/worst는 선택과 평가에 같은 bank를 사용하므로 별도 일반화 추정치가 아니다.
후보 간 분산에는 3회 재평가의 잡음이 포함되며, 초기 성능과 변화량의 상관은 상한효과와 수학적 결합의 영향을 받는다.
40턴 종료는 실제 PCT, 다른 예산 종료는 승인된 0점 규칙으로 유지하고 인프라 오류와 구분한다.
생성되는 CSV·JSON·그림에는 전체 cohort를 사용하며, turn-cap 제외 진단 집단은 학습·최종 후보 집계에 쓰지 않는다.
