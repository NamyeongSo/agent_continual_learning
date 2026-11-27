# GEPA / SkillOpt 연속 학습 프로토콜과 남은 선택

2026-10-19 사용자 답변을 반영했다. 이전 [결정 목록](gepa_skillopt_baselines.md)의 제안과 충돌하면 이 문서의 **확정 사항**을 따른다.
2026-10-21 추가 확정: GEPA·SkillOpt의 **전체 학습 예산에는 상한을 두지 않는다.** PREMiSE 비용 집계는 비교·계측용이며 실행 제한값이 아니다.
2026-10-21 추가 확정: GEPA Pareto/완벽 batch 생략과 SkillOpt batch·minibatch 5, accumulation 1, patch, cosine 편집 수 4→2를 적용한다.
**Isolated 실행기를 구현했다.** Sequential/interleaved 지속 전이 실행기는 아직 구현하지 않았다. 아래 비용 예상은 실행 결과와 구분한다.

## Isolated 실행

```bash
.venv/bin/python scripts/baselines/run_experiment.py \
  --method gepa --benchmark bfcl --seed 42 --workers 50 --output outputs/gepa_bfcl_s42
.venv/bin/python scripts/baselines/run_experiment.py \
  --method skillopt --benchmark appworld --seed 43 --workers 50 --output outputs/skillopt_appworld_s43
```

`--dry-run`은 설정만 저장하고, `--pilot`은 별도 진단용 5 train / 2 val / 2 test를 사용한다.
Terminal-Bench 2.1은 두 방법 모두 `--benchmark terminalbench2 --terminalbench2-version 2.1`로 선택한다.
train·validation·test와 동결 checkpoint 평가는 저장된 `benchmark_kwargs.subset`을 사용한다.
2.0 결과 디렉터리를 재사용하면 immutable 설정 검사에서 거부한다.
새 `scripts/baselines/run_queue.py` 큐에도 `--terminalbench2-version 2.1`을 지정할 수 있다.
기존 큐는 생성 당시 버전으로 유지한다. [2.1 설치·실행 예시](terminalbench2.md)를 참고한다.
`--scripted`는 LLM·환경 호출 없이 공식 학습 루프를 검증한다. 본 실험에는 두 옵션을 사용하지 않는다.
누적 예산을 받는 옵션은 없다. BFCL/AppWorld/Terminal은 각각 기존 seed별 train 50/50/30,
validation 50/50/30, test 50/50/29를 사용한다. seed는 42/43/44다.

GEPA는 고정한 공식 `gepa.optimize` 엔진의 외부 시스템 adapter·sampler·stop callback API를 사용한다.
DSPy `GEPA.compile` 래퍼는 metric 예산을 반드시 요구하므로 무제한 예산을 지원하는 이 공식 API를 사용한다.
Pareto parent, minibatch 엄격 개선, full validation, 집계 best 선택은 공식 엔진이 수행한다.
SkillOpt는 공식 `ReflACTTrainer.train`을 수정 없이 실행하고 EnvAdapter/dataloader, Qwen 호출,
제공된 optimizer prompt 로딩만 연결한다. train 단계에서만 reflection용 trajectory를 전달한다.

Actor는 고정 팀 benchmark base + 학습된 지침을 기존 PREMiSE actor에 주입한다.
BFCL/AppWorld의 tool calling·shortlisting과 Terminal의 native Terminus-2 및 공통 token/turn 제한을 공유한다.
이 actor 재사용은 PREMiSE 학습·replay 알고리즘을 실행한다는 뜻이 아니다.
성공 지표는 공식 `SessionResults.success`이며 AppWorld 부분 점수는 `soft` 진단에만 사용한다.
actor/분석 모두 Qwen3.6-35B-A3B, temperature 0.7, non-thinking이다. 분석은 출력 8192,
입력은 모델 context 범위이며 actor의 100k 상한을 적용하지 않는다.

재개는 초기 optimizer 상태에서 저장된 actor/teacher 결정을 재생해 archive·RNG·rejected edits·LR을 복구한다.
SkillOpt 기본 checkpoint만으로 rejected buffer가 복원되지 않는 문제를 피한다.
재생은 이미 끝난 환경/분석 호출을 다시 청구하지 않으며 최초 노출 기록이 달라지면 중단한다.
실제 새 요청·실패·재시도는 `attempts.jsonl`, `analysis_calls.jsonl`, `analysis_errors.jsonl`과 각 session cost report에 남는다.
인프라 오류로 full validation이 불완전하면 점수를 계산하거나 후보를 수락하지 않고 해당 run을 중단한다.

`scripts/baselines/run_queue.py`는 코드 snapshot을 만들고 선행 Qwen 큐 완료 및 GPU lease 해제를 기다린다.
GEPA/SkillOpt × benchmark 3개 × seed 3개 = 18개를 최대 방법별 6개씩, 총 12개 실행한다.
각 run 내부 병렬성은 50이다. 최종 validation best를 먼저 동결하고 test를 평가한다.
사용자가 기존 실험과 동시 실행을 명시하면 queue 생성 시 `--join-output <현재 Qwen 실험 경로>`와
`--max-running-per-method 3`으로 GEPA 3 + SkillOpt 3개를 즉시 시작할 수 있다.
이 경우 기존 소유 서버/bridge를 검증해 공유하고 원래 실행기가 GPU lease와 서버 수명 관리를 유지한다.

## 확정 사항

| 항목 | 적용할 계약 |
| --- | --- |
| Optimizer / 순서 | 지속 optimizer + PREMiSE `manifest_order`. 전체 train pool을 optimizer에 전달하지 않고 현재 batch만 순서대로 공개한다. train 첫 노출은 한 번뿐이다. |
| Isolated | benchmark별 지침, archive, rejected edits, memory, LR, RNG, cursor를 모두 새로 시작한다. |
| Sequential | BFCL → AppWorld → Terminal. GEPA 후보 archive·Pareto 상태와 SkillOpt rejected-edit buffer·LR 진행·optimizer memory를 전달한다. |
| Interleaved validation | seed별 고정 50개: BFCL 17 / AppWorld 17 / Terminal 16. 두 방법이 동일한 ID를 사용하고 학습 중 다시 뽑지 않는다. |
| Sequential / isolated validation | 현재 benchmark의 기존 val 전체: BFCL 50 / AppWorld 50 / Terminal 30. |
| Validation 용도 | GEPA instance별 성능/Pareto와 집계 점수, SkillOpt 일반 step gate 및 best 선택. val trajectory·정답을 train reflection 입력으로 제공하지 않는다. |
| Reflection | 각 공식 방법의 train trajectory reflection 흐름을 따른다. 팀 프롬프트 원문을 보존한다. |
| Actor / 모델 | actor·reflector·optimizer 모두 같은 모델, non-thinking. benchmark별 actor·환경 설정은 선택한 PREMiSE reference와 공유한다. |
| 학습 대상 | 하나의 shared instruction / skill. 팀의 benchmark base는 고정 입력, shared/seed 파일은 학습 초기값으로 사용한다. |
| GEPA | reflection minibatch 5, merge 끔, format failure를 feedback으로 제공, 최종 후보는 공식 validation 집계 best. |
| SkillOpt | epoch 1, 성공·실패 모두 분석, 일반 step gate만 사용, 최종 평가는 best skill. slow update는 끈다. |
| 목적 함수 | 공식 task 완전 성공 여부 0/1. AppWorld 부분 점수를 성공으로 간주하지 않는다. 부분 점수는 진단용으로 보존할 수 있다. |
| 전체 학습 예산 | 누적 rollout·metric call·token·비용·wall-clock 예산 상한 없음. train 한 번 순회는 유지하고, train/validation/실패/재시도 비용은 모두 기록한다. actor의 요청별 token·trajectory turn 제한과 benchmark deadline은 별도로 유지한다. |
| 기록 | unique encounter 기준 provisional 50/100/130에서 validation best 저장. online은 최초 수행의 업데이트 전 점수, offline은 동결한 best checkpoint의 test 추론 점수. |
| 병렬성 | 방법별 최대 50. validation task 실행과 독립 reflection 작업에 적용한다. 다음 업데이트가 이전 업데이트 결과를 쓰는 의존성은 유지한다. |
| 오류 | PREMiSE의 유한 재시도, context/출력 길이 오류, 요청 재시도 소진, 인프라 오류 구분을 재사용한다. |

`scripts/baselines/protocol.toml`에 확정값을 기록한다. 50/100/130과 과거 benchmark 재평가 범위는
연속 전이의 후속 조정 대상으로 둔다. Isolated는 각 benchmark train 종료 시 best를 동결한다.

## PREMiSE에서 실제로 확인한 예산

현재 기본 `past_count=3`은 충분한 이력이 있고 후보 4개가 유효하면:

```text
현재 task 부모 실행 2회
+ 후보 4개 × (현재 task 1회 + 과거 task 3회)
= encounter당 환경 세션 18회
```

초기 이력이 0 → 1 → 2 → 3으로 늘고 매번 정상 업데이트된다면 130개에서 `6 + 10 + 14 + 127×18 = 2,316`회다.
이는 이론값이다. prefix replay도 환경 실행 1회로 세지만 fresh rollout과 LLM 연산량이 같다는 뜻은 아니다.

실제 집계는 dashboard manifest가 지정한 Qwen3.6-35B-A3B의 t3 완료 run 6개를 사용했다.
재개된 run의 iteration이 참조하는 이전 경로까지 따라가고, 복사된 세션은 run ID로 중복 제거했다.
test/checkpoint 평가 세션은 제외한다.

| 모드 | Seed | Unique encounters | Iteration 결과의 실행/재시도 하한 | 세션 경로까지 복원한 하한 |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 42 | 130 | 2,238 | **2,268** |
| Sequential | 43 | 130 | 2,115 | **2,131** |
| Sequential | 44 | 130 | 2,074 | **2,076** |
| Interleaved | 42 | 130 | 1,755 | **1,781** |
| Interleaved | 43 | 130 | 2,198 | **2,226** |
| Interleaved | 44 | 130 | 2,183 | **2,251** |

모든 최종 결과가 참조하는 세션 디렉터리는 복원했다. 다만 실패한 update는 세션 참조 자체를 생략할 수 있다.
따라서 이 표는 **관측된 실행 하한이며 전체 비용의 정확한 총합이 아니다.** 포기한 실행 namespace의 세션도 별도 집계했지만 위 예산에 섞지 않았다.
특히 interleaved seed 42는 업데이트 생략이 많아 예산이 작다. 이를 다른 seed에 공통인 정상 비용으로 사용하지 않는다.

원본 결과를 수정하지 않는 집계 도구:

```bash
.venv/bin/python scripts/baselines/audit_premise_budget.py \
  --archive-root outputs_arr_oct \
  --run outputs_arr_oct/premise_completion_contract_restart_20261003_0557/s42_interleaved_t3 \
  --output .exgentic/baselines/reports/new_budget_report.json
```

`--run`은 반복 지정할 수 있다. report는 state/iteration 내용 해시, 생략된 update 수, 원본 경로, benchmark별 실행 수와 누락 범위를 담는다.
이번 6-run 상세 보고서는 `.exgentic/baselines/reports/premise_budget_20261019.json`에 저장했다.
Isolated는 benchmark별 상태 초기화로 이력 비용이 달라지므로 이 130개짜리 예산을 그대로 쓰지 않고 해당 reference를 별도로 집계해야 한다.

## GEPA에서 설명이 필요한 항목

### Pareto parent 선택

Parent는 다음 수정의 출발점이다. **최종 저장할 best 후보와 다음에 수정할 parent는 서로 다른 선택이다.**
예를 들어 평균 성공률은 A가 높아도, B만 해결하는 Terminal task가 있으면 B를 출발점으로 삼을 가치가 있다.
공식 `pareto`는 val task마다 최고 성능을 낸 후보들의 관계를 보존하고, 중복/지배 후보를 정리한 뒤 담당하는 최고 task 수에 따라 parent를 확률적으로 선택한다.
매번 평균 best만 고르는 `current_best`보다 후보의 강점을 유지한다. **공식 기본값 `pareto` 유지로 확정했다.**

최종 export는 기본적으로 validation 집계 점수 best다. 현재 고정 val 50개의 단순 평균은 benchmark 가중치 17:17:16이며 정확한 1:1:1 macro 평균과 약간 다르다.
benchmark별 점수도 함께 저장한다.

### Perfect-score trajectory

준비한 공식 엔진의 `skip_perfect_score=True`는 **현재 parent가 minibatch 전체에서 perfect이면 그 묶음의 reflection/수정을 생략**한다.
binary success이므로 5개 모두 성공하면 생략하고 다음 묶음으로 간다. 4개 성공·1개 실패면 성공 사례도 reflection 자료에 포함된다.
성공 trajectory 하나하나를 항상 제거하는 옵션이 아니다. 부모의 5회 실행은 이미 발생했으므로 예산에 포함한다.
**공식 기본값 True 유지로 확정했다.**

### Optimizer budget

`max_metric_calls`는 GEPA가 평가한 task 수다. 초기 validation, parent minibatch, 수정 후보 minibatch, full validation 등이 포함된다.
한 task 안의 LLM 요청 횟수·reflection 요청 수·token 수와 다르다. 기본 stopping policy는 iteration 경계에서 검사하므로 엄격한 실물 예산 상한으로 단독 사용하면 안 된다.

Interleaved에서 130개를 크기 5로 나누면 최대 26 proposal 기회가 있다. merge를 끄고 매 proposal이 full val까지 진입하는 경우:

```text
초기 val 50 + parent train 130 + 후보 train 130 + 후보 val 26×50 = 1,610 task 평가
```

perfect batch 생략, minibatch 탈락, 동일 후보 cache 등이 있으면 줄어든다. 재시도·오류 비용은 별도다.
online은 그 묶음에 처음 적용한 parent의 업데이트 전 점수로 기록하며, parent ID와 지침 해시를 남긴다.
별도의 best-prompt online rollout을 추가하는 구성은 위 계산에 포함하지 않는다.

공식 근거: [parent 선택](https://github.com/gepa-ai/gepa/blob/f1cfdee2856e9dc4b5b4dc5cc2e391d0aa20d819/src/gepa/strategies/candidate_selector.py),
[perfect-batch 처리](https://github.com/gepa-ai/gepa/blob/f1cfdee2856e9dc4b5b4dc5cc2e391d0aa20d819/src/gepa/proposer/reflective_mutation/reflective_mutation.py),
[DSPy 설정](https://github.com/stanfordnlp/dspy/blob/1a4f8372eb47e28d975cebf5d09ca6d7f863d32b/dspy/teleprompt/gepa/gepa.py).

## SkillOpt의 batch / minibatch / accumulation / 편집

| 개념 | 의미 | 제안값과 예시 |
| --- | --- | --- |
| Train batch | 동일한 current skill로 먼저 수행할 새 task 묶음 | **5**: 새 task 5개 수행 후 skill 수정·val gate 1회. |
| Reflection minibatch | 이미 생성된 성공/실패 trajectory를 한 analyst 요청에 묶는 단위 | **5**: task를 추가 실행하는 값이 아니다. 성공/실패 그룹은 따로 나누므로 각 요청의 실제 수는 더 작을 수 있다. |
| Accumulation | 여러 train batch의 분석/편집 제안을 모은 뒤 한 번만 업데이트 | **1**: batch 5 × accumulation 1 = 최대 5개 최초 encounter마다 업데이트. 2라면 skill을 고정한 채 10개를 수행하고 한 번 수정한다. 수치 gradient를 누적하는 것이 아니다. |
| Patch | 기존 skill에 append / insert_after / replace / delete 적용 | **공식 기본이며 제안값.** 기존 문서의 일부만 수정한다. |
| Rewrite | 제안으로 skill 전체를 다시 쓰거나 minibatch별 전체 rewrite 생성 | 공식 대안 모드. 기본값이 아니다. |
| Edit budget | 순위화한 편집 중 한 step에 선택할 최대 개수 | **공식 기본 최대 4 / 최소 2 제안.** 4회 reflection, 4개 task, 4 token이라는 뜻이 아니다. 실제 적용 수는 작을 수 있고 gate가 거절하면 적용 0개다. |
| LR schedule | step이 진행될수록 위 편집 개수 상한을 바꾸는 규칙 | **공식 기본 cosine, 4→2 제안.** 일정한 4는 constant, 일정 비율로 줄이면 linear다. |

26번 모두 patch 선택 단계까지 진행되면 공식 cosine의 정수 반올림 결과는 처음 8 step 최대 4개, 다음 9 step 최대 3개, 마지막 9 step 최대 2개다.
편집 제안이 없어서 rank/select 이전에 생략되면 scheduler 진행도 다를 수 있으므로 실제 scheduler cursor를 저장한다.
Sequential 경계에서 LR을 다시 4로 올리지 않는다. 전체 26-step 계획을 공유하고 실제 진행 상태를 전달한다.

일반 gate는 후보 성공률이 current 성공률보다 **엄격하게 높을 때만** 수락한다. 동점이면 거절한다.
전체 best skill은 validation에서 고르고 test는 선택에 사용하지 않는다.
batch 5 / accumulation 1 / 26 step이면 interleaved 비용 상한의 기본 계산은:

```text
초기 val 50 + train 130 + 후보 val 26×50 = 1,480 task 평가
```

cache hit나 편집 없음은 비용을 낮춘다. 위 계산에는 재시도, stage 전환 재평가, 별도 epoch-end 기능을 포함하지 않았다.
성공/실패 analyst 작업은 병렬화하되, patch 취합·적용 후 gate가 끝나야 다음 skill로 넘어간다.
SkillOpt의 patch 취합은 고유 학습 과정이다. 사용자가 금지한 **GEPA의 후보 merge**와 다르므로 그대로 유지한다.

Single pass에서 optimizer memory를 해석할 때 주의할 실제 구현 사항:

- rejected edits와 실패 패턴은 step buffer에 쌓인다. benchmark 경계에서 epoch를 새로 만들면 이 buffer가 사라지므로 하나의 지속 epoch로 처리한다.
- 별도 `meta_skill`은 공식 구현에서 첫 epoch에 생성하지 않는다. epoch 1 조건에서는 이 memory가 비어 있을 수 있다. 존재하는 memory는 전달하지만 생성됐다고 기록하지 않는다.
- slow update는 일반 step 밖의 추가 skill 수정이므로 끈다. meta-skill 플래그는 유지하되 단일 첫 epoch의 기본 생략을 따른다.
- 공식 runtime checkpoint만으로 rejected buffer 전체의 복원이 보장되지 않는다. 연결 실행기에서 buffer·LR cursor·memory·RNG를 명시적으로 저장/복원해야 한다.

공식 근거: [기본 설정](https://github.com/microsoft/SkillOpt/blob/1a44050fc53aa4938a7a3927b6cca550491bfe51/configs/_base_/default.yaml),
[trainer](https://github.com/microsoft/SkillOpt/blob/1a44050fc53aa4938a7a3927b6cca550491bfe51/skillopt/engine/trainer.py),
[edit-budget scheduler](https://github.com/microsoft/SkillOpt/blob/1a44050fc53aa4938a7a3927b6cca550491bfe51/skillopt/optimizer/scheduler.py).

## Sequential의 archive 유지와 예산

BFCL val의 80%와 AppWorld val의 80%는 동일한 평가가 아니다. validation pool ID/해시를 점수·cache에 포함한다.

GEPA는 archive 내용을 보존하되 새 benchmark val로 후보를 재평가하고 그 pool의 instance별 best/Pareto를 다시 구성해야 한다.
이전 benchmark의 점수 이력은 보관하지만 새 pool 점수와 직접 비교하지 않는다. GEPA에는 도메인 전환이라는 기본 API가 없으므로 이 부분은 연결 계층의 책임이다.
이후 parent는 새 pool에서 공식 parent selector가 선택하고, export는 해당 pool의 집계 best를 따른다.

최대 10 / 10 / 6 proposal이 모두 archive에 들어간다는 보수적 계산:

```text
BFCL: 초기 후보 1×50 + train 평가 100 + 후보 val 10×50 = 650
AppWorld: 기존 archive 최대 11×50 + train 평가 100 + 후보 val 10×50 = 1,150
Terminal: 기존 archive 최대 21×30 + train 평가 60 + 후보 val 6×30 = 870
합계 최대 2,670 (재시도 제외)
```

따라서 interleaved의 1,610을 sequential에 그대로 적용할 수 없다. archive가 작거나 후보가 탈락하면 실제 비용은 줄어든다.
SkillOpt도 새 pool에서 current/best의 기준 점수를 다시 측정해야 하며, 이전 pool의 gate cache·best score를 그대로 쓰면 안 된다.
candidate archive와 LR/memory 초기화로 비용을 줄이는 방식은 사용자가 요청한 전이와 맞지 않는다.

**2026-10-21 확정: 전체 학습 예산 상한 없음.** 위 PREMiSE 관측 하한이나 GEPA/SkillOpt 비용 계산을
rollout·metric call 상한으로 사용하지 않는다. 비용 때문에 새 proposal이나 stage 재평가를 생략하는
예약/예산 소진 정책도 적용하지 않는다. 공식 방법의 업데이트와 validation을 수행하고, train을 한 번
순회해 마지막 batch를 처리하면 종료한다. 무제한 예산이 train 반복 노출이나 무한 탐색을 뜻하지 않는다.

train·validation·stage 재평가·실패·재시도는 모두 기록하고, reflection 등의 LLM 호출·token·시간 비용도
따로 남긴다. 고정 validation, 유한 요청 재시도, benchmark deadline, 공통 actor 입력100k·출력2048 및
BFCL30/AppWorld·Terminal40턴 제한은 유지한다. 앞의 비용 계산은 실행 비용을 설명하는 참고값이다.

## 병렬성과 오류 처리 연결 기준

Validation 50개는 worker 최대 50으로 평가한다. BFCL/AppWorld/Terminal에 서로 다른 실행 namespace를 부여하고,
동일 task의 후보 평가도 환경·파일·Docker project가 충돌하지 않게 PREMiSE 실행 경로를 사용한다.
Sequential에서 다음 stage를 미리 학습시키거나, gate 결과 전에 다음 skill을 적용하는 병렬화는 하지 않는다.
프로세스 내부에 남아 있는 PREMiSE 기본 LLM semaphore 32를 그대로 두면 worker만 50이어도 요청 동시성은 32가 된다.
연결 시 task worker와 LLM 제한을 모두 50에 맞추고, 내부 Terminal 프로세스도 포함한 실행 범위를 계측한다.
여러 seed/방법의 run을 동시에 실행할 때의 서버 전체 동시성은 별도 launcher quota다.

Agent의 실제 format/tool-call 실패는 공개 feedback을 주고 실패 점수로 기록한다.
요청 장애나 환경 시작 실패는 모델 오답으로 바꾸지 않는다. 재시도 소진 시 후보 gate를 불완전한 val 평균으로 수락하지 않고 update를 보존/생략한다.
요청 재시도 소진 시 현재 task의 실패 종류와 점수 유무를 남기고 cursor를 진행한다.
tool parser 누락처럼 task 교체로 해결되지 않는 설정 오류는 PREMiSE 정책에 따라 명시적으로 중단한다.
bounded retry의 모든 실제 실행은 비용 ledger에 남긴다.

## 이번에 생성한 재현 도구

```bash
.venv/bin/python -m scripts.baselines.protocol \
  --seed 42 --mode interleaved \
  --output .exgentic/baselines/plans/new_s42_interleaved.json
```

plan은 원래 train/val/test 분리 검사를 재사용하고, task ID의 seed별 SHA-256 순위로 17/17/16을 선택한다.
manifest list의 표시 순서가 바뀌어도 선택 ID는 바뀌지 않는다. source manifest와 validation 내용 해시를 저장한다.
순서는 PREMiSE 함수를 직접 호출해 얻는다. 출력은 **controller 전용 전체 계획**이며 optimizer trainset으로 통째로 넘겨서는 안 된다.
이 metadata 계획 자체는 실행기가 아니므로 `runnable=false`다. Isolated 학습은 위 `run_experiment.py`로 실행한다.
기존 계획 파일 덮어쓰기는 거부한다.

GEPA parent/perfect-batch 기본값과 SkillOpt batch/minibatch/accumulation·patch·편집 schedule 및
전체 학습 예산 상한 없음은 모두 확정했다. 남은 구현은 sequential/interleaved의 상태 전이다.
