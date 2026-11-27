# GEPA / SkillOpt 공식 구현 준비와 실험 설계 결정 사항

작성 기준: 2026-10-19. 아래의 **확인된 구현**, **제안**, **미결정 사항**을 구분한다.

후속 사용자 답변을 반영한 **[확정 프로토콜·PREMiSE 실측 예산·용어 설명](gepa_skillopt_protocol.md)**을 먼저 참고한다.
아래 결정 표는 최초 검토 기록이며, 후속 확정 사항과 충돌하는 제안은 적용하지 않는다.

현재 완료 범위는 공식 소스 확보, 커밋/의존성 고정, 독립 환경 설치, 팀 프롬프트 보존, 오프라인 검증이다.
BFCL·AppWorld·Terminal-Bench를 실제 학습시키는 연결 어댑터와 전체 실행기는 아직 구현하지 않았다.
아래 선택에 따라 학습 의미가 달라지므로, 준비 코드를 완성된 학습 baseline으로 부르지 않는다.

## 1. 가져온 기본 구현과 사용 방법

| 구성 | 공식 소스 | 확인 버전 | 고정 커밋 |
| --- | --- | --- | --- |
| GEPA 진입점 | `stanfordnlp/dspy`, `dspy.GEPA` | 3.4.1 | `1a4f8372eb47e28d975cebf5d09ca6d7f863d32b` |
| DSPy가 사용하는 GEPA 엔진 | `gepa-ai/gepa` | 0.1.4 | `f1cfdee2856e9dc4b5b4dc5cc2e391d0aa20d819` |
| SkillOpt | `microsoft/SkillOpt`, `ReflACTTrainer` | 0.2.0 | `1a44050fc53aa4938a7a3927b6cca550491bfe51` |

버전 문자열이 같아도 release tag와 최신 소스 커밋이 같은 것은 아니다. 이번 준비는 **위 커밋의 소스**를 기준으로 한다.
논문의 특정 release를 재현하려면 버전 선택을 별도로 고정해야 한다.
세 저장소의 MIT 라이선스와 원본 코드는 각 checkout에 그대로 유지했다.

```text
scripts/baselines/
  upstreams.lock.toml                # URL, SHA, 환경, 원본 ZIP/프롬프트 해시
  requirements-gepa.lock.txt         # 설치한 GEPA 환경의 의존성 snapshot
  requirements-skillopt.lock.txt     # 설치한 SkillOpt 환경의 의존성 snapshot
  prepare.py                        # 소스 확보/설치/상태 확인/검증
  smoke.py                          # 모델 호출 없는 공식 코드 검증
  prompts/                          # ZIP의 텍스트 25개, 원문 byte 보존
.exgentic/baselines/
  upstream/{dspy,gepa,skillopt}/      # Git에서 제외되는 공식 checkout
  venvs/{gepa,skillopt}/              # 별도 Python 3.12 환경
```

기존 core 환경과 모델 서버를 건드리지 않고 optimizer별 환경을 따로 설치했다.
lock 파일은 이번 Linux/Python 3.12 설치의 재현을 위한 snapshot이다. 다른 플랫폼까지 검증한 범용 lock은 아니다.

Exgentic 디렉터리에서:

```bash
.venv/bin/python scripts/baselines/prepare.py status
.venv/bin/python scripts/baselines/prepare.py fetch
.venv/bin/python scripts/baselines/prepare.py install
.venv/bin/python scripts/baselines/prepare.py smoke

# 한 baseline만 다룰 때
.venv/bin/python scripts/baselines/prepare.py smoke --method gepa
```

`fetch`는 기존 checkout의 수정 사항이나 다른 origin을 발견하면 보존하고 실패한다.
`install`은 위 전용 환경에서만 실행한다. `smoke`는 원본 코드에서 import되었는지도 확인한다.

검증 범위:

- GEPA: 공식 `dspy.GEPA.compile()`을 로컬 DummyLM과 Python proposer로 실행하여 평가, metric budget, 결과 선택을 검증했다.
- SkillOpt: 공식 trainer import, config load, replace/append 편집, 보호 구역, strict-improvement gate를 검증했다.
- SkillOpt의 실제 teacher reflection 전체 학습과 세 benchmark 연결은 검증 범위에 포함하지 않는다.
- 모델 API 요청, GPU 학습, benchmark Docker task 실행은 수행하지 않았다.

## 2. ZIP에서 확인한 사항

원본: `team_prompt/gepa_skillopt_prompts.zip`.
SHA-256: `20967851eb5a3da7221430b6076ad5d1424d9ebcc178dc383ea57e9d4693c0c0`.

텍스트 25개를 `scripts/baselines/prompts/`에 원문 그대로 추출했다.
`.DS_Store`와 `__MACOSX`는 제외했고, 각 파일의 byte 수와 SHA-256을 lock에 기록했다.

| 발견 | 결정에 미치는 영향 |
| --- | --- |
| SkillOpt optimizer 프롬프트 8개가 이번 공식 checkout의 공통 프롬프트와 byte 단위로 일치 | 이 부분은 팀 전용 알고리즘 수정으로 볼 필요가 없다. base/user 메시지 구성과 실제 호출 경로는 별도로 확인해야 한다. |
| GEPA reflection 파일은 `<curr_instructions>`, `<inputs_outputs_feedback>`와 코드 펜스 출력을 기대 | 이번 DSPy의 기본 `InstructionProposer`는 구조화 입력과 `new_instruction` 출력을 사용한다. 원문을 단순 치환하면 입력/출력 계약이 어긋날 수 있다. |
| GEPA AppWorld base에 bill splitting, 여행지별 파일 분류, playlist, alarm 등의 구체적인 전략이 있음 | 공통 seed보다 강한 도메인 사전 지식일 수 있다. 제공된 사실만으로 데이터 유출이라고 단정할 수는 없지만, 출처와 초기 조건의 공정성을 결정해야 한다. |
| GEPA에는 shared base와 benchmark별 base가, SkillOpt에는 seed skill과 benchmark별 base가 공존 | 어떤 부분이 고정 환경 지침이고 어떤 부분이 학습 대상인지 정해야 한다. 모두 단순 연결하면 전략이 중복되거나 benchmark 전환 시 이전 지식이 사라질 수 있다. |
| legacy 학습 history와 exported prompt 평가 도구가 이미 있음 | 과거 산출물 재평가는 재사용할 수 있다. 그러나 이번 공식 소스의 실행 어댑터가 존재하거나 과거 코드와 동일하다는 증거는 아니다. |

원본 프롬프트를 변경할 때는 원본 사본과 변경본을 구분하고 두 해시를 실행 manifest에 저장하는 것이 좋다.

## 3. 현재 저장소에서 재사용할 계약

| 항목 | 현재 코드/정의 | 연결 시 주의점 |
| --- | --- | --- |
| Task manifest | `scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json` | seed별 정확한 ID를 사용한다. 별도 split sampler로 다시 나누지 않는다. |
| 학습 순서 | `src/exgentic/agents/premise/pipeline.py::manifest_order` | benchmark 내부 shuffle와 interleave를 공유한다. legacy `get_unified_task_order`는 task selection 규칙이 다르다. |
| Benchmark 실행 | `exgentic.execute` / `RunConfig`와 기존 benchmark adapter | optimizer 환경에 benchmark 전체를 재설치하기보다 core 실행 경로를 공유할 수 있다. |
| Actor | BFCL/AppWorld의 tool-calling, Terminal-Bench의 Harbor Terminus-2 | DSPy ReAct나 SkillOpt 자체 chat actor로 바꾸면 optimizer와 actor가 동시에 바뀐다. |
| 학습 상태 | isolated는 benchmark별 독립, sequential/interleaved는 공유 가능 | 실험 모드의 isolated와 runner의 프로세스/venv 격리는 다른 개념이다. |
| 공식 score | 특히 AppWorld `SessionScore.score`는 pass percentage / 100, `success`는 완전 성공 여부 | partial score와 binary success가 다르다. SkillOpt `hard`/`soft` 매핑을 명시해야 한다. |
| 공유 LLM 제한 | PREMiSE의 `shared_llm_slots()`는 프로세스 내부 semaphore | 여러 baseline/seed 프로세스 전체의 호출 수를 제한하는 전역 quota는 아니다. |
| Frozen checkpoint 평가 | 기존 checkpoint queue, prompt hash, task split hash, test IDs 검사 | 학습 상태를 동결한 뒤 평가한다. optimizer resume 파일을 그대로 평가 상태로 사용하지 않는다. |
| Terminal prompt | `run_terminus2`의 `playbook`이 내부 Terminus-2 지침으로 전달됨 | 바깥 agent의 system prompt만 바꾸면 내부 actor에 최적화 결과가 전달되지 않을 수 있다. |

현재 분할:

| Benchmark | Train | Val | Test |
| --- | ---: | ---: | ---: |
| BFCL `multi_turn_base` | 50 | 50 | 50 |
| AppWorld `test_challenge` | 50 | 50 | 50 |
| Terminal-Bench 2.0 | 30 | 30 | 29 |
| 합계 / seed | 130 | 130 | 129 |

Seed 42/43/44는 순서만 바뀌는 것이 아니라 split 구성원도 달라진다.
현재 manifest의 ID를 모든 방법에서 공유해야 paired 비교가 가능하다.
자세한 분할 규칙은 `scripts/utils/TASK_SPLITS.md`를 따른다.

| 모드 | baseline당 seed 3개의 training stream | 기본 학습량과 상태 |
| --- | ---: | --- |
| isolated | 9 | benchmark별 50 / 50 / 30, 서로 학습 상태를 공유하지 않음 |
| sequential | 3 | BFCL 50 → AppWorld 50 → Terminal 30, 총 130 |
| interleaved | 3 | 같은 train ID와 benchmark 내부 순서를 유지하며 총 130을 혼합 |

모두 지원하면 두 baseline에서 합계 30개의 training stream이다. 이를 동시에 30개 실행한다는 뜻은 아니다.
Sequential의 50/100/130은 도메인 경계와 일치하지만 interleaved의 같은 숫자는 도메인별 노출량이 다르다.
Isolated에는 전체 stream의 100/130 checkpoint 개념을 그대로 적용하지 않고 benchmark별 기준을 정해야 한다.

## 4. 먼저 결정해야 하는 연구 설계

아래 권장안은 제안이며, 실험 설정에 적용하거나 실제 실험을 시작하지 않았다.
`필수`는 연구 결론을 바꾸는 항목, `위임 가능`은 권장 설정으로 시작할 수 있는 항목이다.
표의 상세 항목은 총 88개이며, 최초 구현 전에 답이 필요한 최소 12개 묶음은 12절에 모았다.

| ID | 결정 | 선택지와 영향 | 제안 / 우선순위 |
| --- | --- | --- | --- |
| A01 | 주장의 중심 | 공식 optimizer의 원래 성능 / PREMiSE와 같은 stream에서의 적응 성능 / 두 가지를 별도 보고 | **필수.** 연속 학습이 논문의 중심이면 stream 비교를 주실험으로, native optimizer 결과를 별도 보고하는 구성이 해석에 유리하다. |
| A02 | 순서의 엄격성 | train pool 전체를 미리 허용 / 반영된 prefix만 접근 / 현재 task만 접근 | **필수.** 엄격한 online 주장에는 미래 train task·trajectory 접근을 막고 실제 접근 기록이 필요하다. |
| A03 | 외부 validation 허용 | 공식 held-out val gate 허용 / train의 seen-history만 사용 / 두 프로토콜 비교 | **필수.** PREMiSE의 current/past train replay와 val gate는 관측 정보가 다르다. 같다고 취급하지 않는다. |
| A04 | 공정성의 기준 | unique train task 수 / 전체 environment rollout 수 / LLM token·비용 / wall-clock | **필수.** unique task와 총 rollout을 모두 기록하고 하나를 주 예산 축으로 지정한다. |
| A05 | 초기 지침 공정성 | 팀 프롬프트 그대로 / 공통 seed / 도메인 지침을 고정하고 학습 부분만 공통 seed | **필수.** 주 비교에는 동일한 고정 환경 정책과 통제된 초기 지식이 가장 해석하기 쉽다. 원본 팀 프롬프트 재현은 별도 조건으로 유지할 수 있다. |
| A06 | Actor 공정성 | 현재 PREMiSE와 동일 actor / DSPy·SkillOpt 자체 actor까지 사용 | **필수.** optimizer 차이를 보려면 현재 actor를 공유한다. 자체 actor 결과는 별도 실험으로 구분한다. |
| A07 | 업데이트 단위 | task 1개 / 같은 benchmark의 batch / 혼합 benchmark batch | **필수.** batch 단위에서는 batch 안의 후속 task가 앞선 task의 업데이트를 받지 않는다는 점을 명시한다. |
| A08 | 온라인 점수 | 업데이트 전 수행 점수 / 업데이트 후 재실행 점수 / validation 점수 | **필수.** prequential online 성능은 업데이트 전 점수로 기록하고 나머지는 별도 필드로 둔다. |

### GEPA의 순서 지원은 특히 별도 설계가 필요하다

공식 `dspy.GEPA.compile(student, trainset=..., valset=...)`는 일반적인 task별 `observe/update` API가 아니다.
기본 엔진 sampler는 epoch shuffle을 사용하므로 입력 list를 interleave했다고 반영 순서가 고정되지 않는다.
현재 DSPy는 `gepa_kwargs`로 core의 `batch_sampler`, callback, stopping policy 등을 전달할 수 있다.
그러나 이것만으로 모든 train 접근·merge·평가까지 causal prefix 조건을 만족한다고 단정할 수 없다.

| 접근 | 공식 구현을 사용하는 방식 | 보존되는 것 / 달라지는 것 |
| --- | --- | --- |
| Native stage optimization | isolated에서 benchmark별 compile, sequential에서 stage별 compile 후 지침 전달, mixed 설정에서는 pooled compile | 공식 학습 루프를 가장 그대로 사용한다. pooled compile은 엄격한 online interleaving과 구별해야 한다. |
| 지속 optimizer + ordered sampler | 공식 compile/core를 유지하고 sampler·budget·callback을 연결 | Pareto/archive 유지에 유리하다. 미래 task 접근과 validation scope를 별도로 검증해야 한다. |
| Window별 compile | 현재 window/prefix만 제공하고, 이전 선택 지침으로 새 compile을 시작 | 데이터 경계를 명확히 만들기 쉽다. 매 window마다 optimizer의 Pareto/archive 상태가 재시작되므로 원래 GEPA와 달라진다. |

**단순히 raw `gepa.optimize()`로 교체하는 것은 사용자가 요청한 DSPy 진입점과 다르다.**
이 작업은 `dspy.GEPA`를 진입점으로 유지한다는 조건에서 설계한다.

## 5. 데이터·모드·전이 상태 결정

| ID | 결정 | 선택지 / 구체적인 질문 | 제안 |
| --- | --- | --- | --- |
| B01 | split와 seed | 기존 9-key manifest / 새 manifest, seeds 42–44 유지 여부 | 기존 manifest를 공통 사용. 변경하면 PREMiSE도 같은 분할로 재평가해야 비교가 맞는다. |
| B02 | isolated의 상태 | benchmark마다 seed prompt·skill·optimizer memory 전부 초기화 / 일부 공유 | 전부 초기화. 이 모드의 목적을 상태 전이 없는 대조군으로 분명히 한다. |
| B03 | sequential benchmark 순서 | BFCL → AppWorld → Terminal / 역순 / 여러 permutation | 기존 순서를 기본으로 하고 순서 효과가 핵심이면 추가 permutation을 별도 예산으로 둔다. |
| B04 | interleaving 규칙 | 기존 `manifest_order` / round-robin / 전체 pooled shuffle | 기존 benchmark 내부 순서를 보존하는 manifest 순서. optimizer 내부 sampler가 다시 섞지 않도록 검증한다. |
| B05 | 전이 대상 | 최종 prompt/skill만 / Pareto archive, rejected edits, optimizer memory, LR state까지 | 주장을 먼저 결정한다. stage별 native 실행은 prompt/skill-only warm start가 가장 명확하지만 full-state continual optimization과 다른 조건이다. |
| B06 | benchmark 전환 시 base | 새 benchmark base로 교체 / 공유 학습 지침에 고정 환경 policy만 변경 | 공유 학습 지침은 유지하고 환경 계약만 바꾼다. base 교체로 앞선 지식이 지워지는 것을 피한다. |
| B07 | val 접근 범위 | current benchmark / seen benchmarks / 처음부터 all benchmarks | 엄격한 continual 주장이면 current 또는 seen을 우선 검토. all은 미래 도메인 validation 사용을 명시한다. |
| B08 | selection pool 변경 | 이전 score 유지 / incumbent와 비교 후보를 새 pool에서 다시 평가 | 새 pool에서 재평가해야 한다. 다른 pool의 score를 바로 비교하거나 기존 Pareto 점수를 그대로 쓰면 의미가 달라진다. |

Val trajectory를 reflection에 직접 넣는 것과, val score로 후보를 선택하는 것은 구별한다.
저장소 split 규칙은 val을 설정/후보 선택에 사용하는 것은 허용하지만, val의 정답·trajectory를 train reflection 자료로 넣는 것은 허용하지 않는다.
선택 score 자체도 적응에 정보를 주므로 논문의 정보 접근 조건에는 포함한다.

## 6. Actor·프롬프트·feedback 결정

| ID | 결정 | 선택지 / 영향 | 제안 |
| --- | --- | --- | --- |
| C01 | Student 모델 | 모델·revision·serving endpoint, 방법마다 같은지 | 실험에 쓰는 실제 모델을 지정. 기존 모델명을 과거 config에서 추정해 확정하지 않는다. |
| C02 | Optimizer/reflector 모델 | student와 동일 / 강한 별도 teacher | **필수.** 다른 teacher를 쓰면 비용과 성능 요인이 달라진다. 방법별 조건을 맞추거나 별도 보고한다. |
| C03 | 생성 설정 | acting/reflection별 temperature, reasoning/thinking, max tokens | 두 역할을 분리해 명시. 방법 간 actor 설정은 통일하고 teacher 차이는 기록한다. |
| C04 | AppWorld 환경 정책 | 현재 공통 정책 / 팀 base의 구체 전략 포함 | A05와 함께 결정. 앱 허용 목록, 공개 schema, finish 계약은 모든 방법에서 동일해야 한다. |
| C05 | AppWorld tool shortlist | 기존 30개 shortlist와 history window / 전체 tools | 현재 actor와 동일 설정. 도구 선택기를 optimizer 개선 효과와 섞지 않는다. |
| C06 | GEPA reflection 프롬프트 | 현재 공식 `InstructionProposer()` / 팀 원문을 호환 proposer로 연결 | 원형 비교는 공식 기본 proposer. 팀 프롬프트 사용 시 JSON output 계약과 placeholder 변환을 명시적으로 테스트한다. |
| C07 | SkillOpt prompt 구성 | optimizer 원문, user wrapper, seed skill, benchmark base의 연결 순서 | optimizer 8개는 이미 공식과 동일. user 메시지와 seed/base의 역할을 정하고 실제 rendered prompt 해시를 저장한다. |
| C08 | Terminal-Bench 주입 위치 | outer agent system / 내부 Terminus-2의 learned instruction 영역 | 내부 actor에 직접 전달. native Terminus-2 환경·도구 계약과 completion 동작은 공통으로 유지한다. |
| C09 | Reflection에 보이는 정보 | 공식 scalar score만 / 공개 grader feedback / 전체 private grader assertion | 공개 계약과 정제된 공식 feedback을 기본으로 한다. private expected value·비밀번호 등이 일반 지침으로 학습되지 않게 현재 PREMiSE의 정제 계약을 재사용한다. |
| C10 | 긴 trajectory | 전체 / 구조화 요약 / token 예산에 맞춘 truncation | 실제 Terminal 내부 episode를 포함한다. 잘라낸 범위와 원본 경로를 남기며, 조용히 끝부분을 버리지 않는다. |
| C11 | 지침 길이 | method별 자연 길이 / 공통 token cap / 길이별 ablation | token 수를 먼저 기록하고, cap을 쓴다면 truncation보다 명시적 제약·실패 처리를 택한다. cap은 알고리즘에 영향을 준다. |

DSPy GEPA에는 실제로 최적화할 predictor가 보여야 한다. 단순한 `Module.forward()`에서 외부 `execute()`를 호출하는 것만으로는
지침이 최적화 대상 predictor로 등록되거나 trajectory가 trace에 들어간다고 보장할 수 없다.
연결 어댑터는 **후보 지침 → actor에 전달된 실제 지침 → trajectory → metric/feedback**를 끝까지 검증해야 한다.
이것은 구현자가 책임질 연결 계약이며, 사용자가 low-level trace API를 직접 선택할 필요는 없다.

## 7. GEPA 설정 결정

아래의 공식 기본값은 준비한 DSPy 커밋의 값이다. 최종 실험값으로 확정한 것은 아니다.

| ID | 항목 | 공식 동작 / 선택지 | 제안 |
| --- | --- | --- | --- |
| G01 | 최적화 component | 하나의 shared instruction / benchmark별 component / actor 전체 ReAct program | 전이 비교에는 shared instruction 하나가 해석하기 쉽다. benchmark별 독립 component는 전이 성격이 바뀐다. |
| G02 | 예산 지정 | `auto` / `max_full_evals` / `max_metric_calls` 중 하나 | 명시적인 `max_metric_calls`로 시작. `auto=heavy`는 재현 가능한 총비용 통제와 다르다. 숫자는 A04 이후 확정한다. |
| G03 | stage별 예산 | sequential 3등분 / train 수 비례 50:50:30 / 전체 run 공용 | 비교 목적에 맞게 명시. 300/300/300과 전체 900은 같은 학습 기회를 보장하지 않는다. legacy 숫자를 그대로 새 기본값으로 쓰지 않는다. |
| G04 | reflection minibatch | 기본 3 / 1 / 다른 크기 | task별 적응을 주장하면 1 또는 window 경계를 논의해야 한다. native 재현은 기본 3부터 시작할 수 있다. |
| G05 | Parent 선택 | `pareto` 기본 / `current_best` | 공식 `pareto` 유지. 단순 best-only로 바꾸면 중요한 알고리즘 요소가 달라진다. |
| G06 | 성공 trajectory 반영 | `skip_perfect_score=True` 기본 / false | 기본 유지 여부를 명시. 부분 점수와 perfect score의 정의를 맞춘다. |
| G07 | Merge | `use_merge=True`, 최대 5회 기본 / 비활성화 | native 주 비교는 기본 유지. stream 조건에서 미래 데이터 접근·추가 rollout 비용을 검증한다. |
| G08 | Format failure | `add_format_failure_as_feedback=False` 기본 / true | format 오류와 인프라 오류를 구분한 뒤 결정. 잘못된 출력 자체를 학습 신호로 쓸지 명시한다. |
| G09 | 평가 수 / 병렬 | `num_threads`와 core 평가 정책 | 총 resource quota 아래에서 병렬화한다. method 내부 thread 수를 실험 수와 혼동하지 않는다. |
| G10 | 최종/전이 후보 | validation aggregate best / Pareto frontier의 다른 후보 / 마지막 후보 | aggregate best와 당시 acting 후보를 따로 저장. 최종 후보 선택을 test 성능으로 바꾸지 않는다. |
| G11 | 재현 상태 | seed, `log_dir`, detailed results, sampler/callback state | `track_stats=True`, 독립 log directory, seed/후보 lineage 보존. 데이터·지침 정의가 바뀐 상태에 기존 `log_dir`를 덮어 resume하지 않는다. |

`max_metric_calls`는 optimizer가 관리하는 rollout/metric 호출 예산이다.
하나의 multi-turn task 안에서 호출되는 LLM 횟수나 reflection LM 비용과 같지 않다.
실제 episode, LM 요청, token, wall time을 별도로 집계해야 한다.

## 8. SkillOpt 설정 결정

| ID | 항목 | 공식 기본/선택지 | 제안 |
| --- | --- | --- | --- |
| S01 | Epoch 수 | 공통 base config는 4 / single pass 1 / 기타 | **필수.** stream을 몇 번 재방문하는지 정한다. epoch 4는 train 130개를 한 번 본 조건과 다르다. |
| S02 | Batch와 accumulation | 기본 batch 40, accumulation 1 / 작은 batch | **필수.** 50/50/30 또는 혼합 130의 구성과 업데이트 빈도에 맞춘다. batch 10이면 shared stream에서 13 step이지만 benchmark 경계를 넘을 수 있다. |
| S03 | 마지막 작은 batch | 남은 task 전체 사용 / drop / padding·반복 | drop 없이 실제 task 수를 기록. padding으로 반복한 rollout은 별도 비용으로 센다. |
| S04 | Reflection minibatch | 기본 8 | 전체 train batch와 다른 개념이다. 실패/성공 trajectory를 한 번에 몇 개 teacher에 줄지 결정한다. |
| S05 | 성공/실패 분석 | `failure_only=False` 기본 / 실패만 | 원형 비교에는 성공·실패 모두 유지. partial success 분류를 E02와 맞춘다. |
| S06 | 편집 방식 | `patch` 기본 / suggestions 기반 rewrite / minibatch 전체 rewrite | 주 비교는 `patch`. rewrite는 별도 알고리즘 변형으로 구분한다. |
| S07 | Edit budget | learning rate 4, minimum 2 기본 | 여기서 LR은 가중치 학습률이 아니라 선택할 편집 수다. token 길이 제한과도 다르다. |
| S08 | LR schedule | `cosine` 기본 / constant / linear / autonomous | 짧은 stream에서는 실제 step별 edit budget을 확인한 뒤 선택한다. epoch 경계에서 reset할지도 기록한다. |
| S09 | 일반 step gate | `use_gate=True` 기본, improvement가 strict `>` | 유지 권장. 끄더라도 현재 trainer는 validation rollout을 계속하므로 “val 비용 제거”와 같지 않다. |
| S10 | Gate scalar | `hard` 기본 / `soft` / mixed | **필수.** 현재 PREMiSE 공식 score와 비교하려면 E02의 hard/soft 매핑부터 결정한다. 동점 수락 여부 변경은 별도 변형이다. |
| S11 | Slow update | 기본 활성화, sampled training tasks 기본 20 | 첫 epoch은 빈 보호 영역만 삽입하고 실질적인 비교·지침 생성은 epoch 2 말부터다. 추가 재실행·미래 train 접근을 계수하고 pre/post snapshot을 남긴다. |
| S12 | Slow-update gate | 준비한 config의 `slow_update_gate_with_selection=False` | **필수.** 일반 gate가 켜져도 slow update는 기본 force-accept 경로다. strict gate를 여기에도 적용하려면 이 옵션을 true로 지정하고 변경 사실을 기록한다. |
| S13 | Meta skill | 기본 활성화, epoch 1에서는 skip, epoch 2 말에 생성 후 다음 epoch에서 사용 | 1 epoch에서는 실효가 없고, 2 epoch만 실행하면 생성된 memory가 후속 epoch에 쓰이지 않는다. 기능 비교에는 실제 사용 시점까지 보아야 한다. |
| S14 | 추가 확장 | meta-reflect, deep reflection, skill-aware appendix, semantic-density/multi-objective 보너스 등 | 요청한 baseline의 기본 patch/gate 비교에서는 추가 확장을 명시적으로 끄고, 필요한 항목만 별도 ablation으로 켠다. |
| S15 | 전이/평가 artifact | current skill / global best skill / slow-update 이후 skill | current와 best, epoch 말 전후 snapshot을 모두 남기고 주 평가 대상을 지정한다. 특히 ungated slow update 뒤 current와 best가 다를 수 있다. |

`use_slow_update=True`, `use_meta_skill=True`라고 표시하는 것만으로 모든 epoch-level 기능이 실제로 사용되었다고 볼 수 없다.
기능별 실행 횟수, 입력 task IDs, 생성물 해시와 다음 단계에서 읽혔는지를 남겨야 한다.

## 9. Score·검증·checkpoint 결정

| ID | 결정 | 선택지 / 영향 | 제안 |
| --- | --- | --- | --- |
| E01 | 최적화 scalar | native official continuous score / 완전 성공률 / benchmark별 다르게 | **필수.** main score를 고르고 나머지도 같이 저장한다. 서로 다른 scalar의 결과를 같은 수치로 취급하지 않는다. |
| E02 | SkillOpt hard/soft | `hard=완전 성공`, `soft=공식 부분 점수` / 둘 다 공식 score | hard는 analyst의 성공/실패 분류에도 쓰인다. 부분 점수 0.3을 hard에 그대로 넣으면 truthy로 성공 분류될 수 있다. 분류에는 binary success, PREMiSE score와의 정렬에는 soft gate를 검토한다. |
| E03 | Benchmark 합산 | task micro mean / benchmark macro mean / 벤치마크별 Pareto | **필수.** 50/50/30이면 micro는 BFCL/AppWorld에 각각 50/130, Terminal에 30/130의 가중치. macro는 각각 1/3이다. raw score와 aggregate 정의를 같이 저장한다. |
| E04 | Val 크기 | 전체 50/50/30 / 고정 subset / 매번 resampling | 작은 pilot은 고정 subset이 비용 예측에 좋다. 바뀌는 subset의 raw score를 그대로 gate 비교하지 않는다. full-val 최종 평가를 별도로 둔다. |
| E05 | 평가 noise | temperature 0 단회 / seed 반복 / paired repeats | 본실험 단회/반복 횟수를 정하고 같은 조건에서 incumbent/candidate를 비교한다. cache는 stochastic 반복의 대체가 아니다. |
| E06 | Checkpoint 축 | shared stream의 unique encounters 50/100/130 / optimizer step / rollout budget | **필수.** batch가 경계를 넘을 때 정확한 snapshot 시점을 정한다. Isolated는 benchmark별 50/50/30과 중간 지점을 별도 정의한다. 같은 이름에 다른 지식량을 담지 않는다. |
| E07 | Test 일정 | initial+50/100/130 / final only / validation-selected best | test는 선택에 사용하지 않는다. 비용이 큰 모든 checkpoint 평가는 비동기 frozen queue로 분리할 수 있다. |
| E08 | 과거 도메인 평가 | checkpoint마다 모든 benchmark / 현재 benchmark만 | catastrophic forgetting을 보려면 전체 3개 benchmark가 필요하다. 현재 benchmark만 보면 이전 능력 저하를 놓친다. |
| E09 | 최종 표 | seed별 score, mean±sample SD / pooled task mean / paired comparison | 기존 seed별 split을 유지하고 seed별 결과와 유효 task 수를 함께 보고한다. 부족한 seed를 0으로 채우지 않는다. |

BFCL/AppWorld/Terminal의 score를 모두 0–1로 표현할 수 있어도 동일한 의미·난이도·분산을 갖는 것은 아니다.
raw benchmark score를 보존한 뒤 사전에 정한 aggregate만 사용한다.

## 10. 병렬성·자원·오류 정책 결정

| ID | 결정 | 선택지 / 영향 | 제안 |
| --- | --- | --- | --- |
| P01 | 동시에 실행할 run 수 | baseline × mode × seed의 동시 수 | **자원 입력 필요.** isolated 3 benchmark×3 seed=9, sequential 3, interleaved 3으로 baseline당 15, 두 baseline 합계 30개의 training stream이다. |
| P02 | 독립 병렬 층 | run / rollout / validation / reflection / checkpoint test | 각 층에 별도 cap을 두고 전체 LLM·Docker cap 아래에서 공유한다. thread 하나의 수치로 모두 대체하지 않는다. |
| P03 | 상태 변경 병렬 | 같은 skill/prompt의 동시 업데이트 / version별 병렬 평가 후 순차 commit | 업데이트는 단일 소유자. 평가 job에는 불변 후보 hash와 task IDs를 고정한다. 완료 순서로 stream 순서를 바꾸지 않는다. |
| P04 | 후보·task 병렬 평가 | GEPA num_threads, SkillOpt rollout workers/analyst_workers | 각 환경 세션이 독립적인지 먼저 확인. AppWorld directory, BFCL state, Docker compose/network 이름이 겹치지 않아야 한다. |
| P05 | Endpoint quota | 모델 endpoint별 요청 수·token 제한, GPU/메모리, teacher/actor 분리 여부 | **자원 입력 필요.** 기존 프로세스와 합산해 정한다. 원격 모델의 단순 URL만으로 안전한 동시성 수치를 추정하지 않는다. |
| P06 | Terminal 자원 | 동시 container 수, image build 수, 원격 Docker/SSH session 수 | **자원 입력 필요.** 느린 Terminal을 별도 queue로 두고 BFCL/AppWorld가 대기하지 않도록 한다. |
| P07 | Train vs test 우선순위 | 학습 우선 / 평가 우선 / 고정 resource partition | 이미 돌고 있는 실험의 자원을 침범하지 않는 정책을 정하고 frozen test가 학습 상태를 잠그지 않게 한다. |
| P08 | Async 학습 | 아직 끝나지 않은 task를 두고 다음 후보를 학습 / batch barrier | 처음에는 batch barrier와 ordered commit. async는 stale-policy 효과가 있으므로 별도 변형이다. |
| P09 | 자원 계측 | 모델 요청, token, verifier, replay, Docker build, queue wait | 최소 actual rollout/LLM/token/wall time을 단계별 기록. 병렬 wall time만으로 sample efficiency를 비교하지 않는다. |
| F01 | 인프라 오류 | 0으로 학습 / 제외·재시도 / 실험 중단 | 실제 환경·전송 오류를 일반 실패 점수로 넣지 않는다. 모델의 잘못된 답/도구 사용 실패와 구분한다. |
| F02 | 재시도 횟수 | endpoint·task·optimizer 단계별 bounded retry | 횟수와 backoff를 명시하고 원본 실패를 보존한다. deterministic schema/config 오류를 무한 재시도하지 않는다. |
| F03 | 재시도 예산 | 성공 요청만 계수 / 실제 소비를 모두 계수 | 실제 rollout/LLM 비용은 실패·재시도도 센다. 유효 점수와 비용 집계를 분리한다. |
| F04 | Gate 평가 일부 실패 | 누락 task로 비교 / 후보 보류 후 복구 / 실험 중단 | 동일 task IDs의 유효 평가가 갖춰질 때까지 후보 선택을 보류하는 쪽이 해석에 유리하다. |
| F05 | Timeouts / step 한계 | 기존 PREMiSE와 동일 / 공식 benchmark deadline / 추가 운영 상한 | **필수.** Terminal의 agent/verifier/RPC timeout과 action/token 한계를 구분해 동일 정책으로 맞춘다. |
| F06 | 비정상 optimizer 출력 | 공식 parse/repair / 재요청 / 해당 update skip | 공식 처리 경로를 우선 사용하고 raw response와 적용/거절 이유를 보존한다. parse 복구 성공을 새 연구 성능으로 혼동하지 않는다. |

## 11. Resume·산출물·기록 결정

| ID | 결정 | 선택지 / 영향 | 제안 |
| --- | --- | --- | --- |
| R01 | Resume 단위 | rollout / completed batch / accepted update / stage | 완료 batch·update 단위 atomic state를 기본으로 하고 기존 유효 rollout은 hash 확인 후 재사용한다. |
| R02 | Resume 동일성 | 모델/프롬프트/split/순서/환경 변경 허용 | model revision, rendered prompt, split IDs/hash, order/hash, budget, upstream SHA, runtime config를 identity에 포함한다. 다르면 forked run으로 둔다. |
| R03 | Optimizer 상태 보존 | GEPA archive/sampler/RNG, SkillOpt current/best/rejected buffer/LR/meta | prompt 텍스트만 있는 resume를 full-state resume라고 부르지 않는다. 공식 상태 파일과 dataloader state를 함께 저장한다. |
| R04 | Cache 범위 | model response / trajectory / score / candidate-val result | task+benchmark environment+prompt/model/config+split identity가 같을 때만 재사용. 일부 설정만 같은 캐시는 제외한다. |
| R05 | 부분 batch 중단 | 끝난 task만 반영 / batch 재구성 / 처음부터 재실행 | 동일 batch와 후보로 남은 task를 복구하고, 이미 적용한 update가 중복 적용되지 않게 한다. |
| R06 | Frozen artifact | prompt/skill만 / optimizer state만 / 평가 가능 bundle | 평가용 rendered prompt/skill, 고정 policy, 모델 설정, task IDs/hash, source SHA를 함께 저장한다. |
| R07 | 저장량 | 전체 trajectory/LLM raw / 요약만 / 민감정보 정제본 | 연구 산출물에는 점수와 근거 경로·해시를 남기고, 대용량 raw는 ignored output에 보관한다. 실패와 거절 후보도 재현 가능한 범위에서 유지한다. |
| O01 | 공통 결과 포맷 | upstream 전용 로그만 / 공통 JSONL+upstream 원본 | 기존 plotting/checkpoint queue가 읽을 공통 index와 upstream 상세 로그를 같이 둔다. 실제 알고리즘 기록을 요약으로 대체하지 않는다. |
| O02 | 실험 기록장 연결 | 수동 / launch·완료 milestone 자동 등록 | 사용자가 원하면 `$experiment-journal` 흐름으로 기록. 사용자 자유 노트는 보존하고 agent 설명은 진행 기록에 추가한다. |
| O03 | Paper provenance | 방법 이름만 / upstream commit+wrapper diff+prompt hash+protocol | 후자가 필요하다. official-native, ordered sampler 변형, window reset 변형을 구분해 이름 붙인다. |
| O04 | Legacy 결과 연결 | 이전 결과와 합산 / 별도 label / 동일 조건 확인 후 재사용 | 현재 source/runtime/split/prompt identity가 확인되지 않으면 별도 label. 이미 export된 prompt의 frozen 평가는 기존 queue를 활용할 수 있다. |

## 12. 최초 구현 전에 답이 필요한 최소 항목

나머지는 위 표의 제안으로 시작할 수 있지만, 아래는 실험 의미나 실제 자원에 관한 입력이 필요하다.

1. **A01/A02:** native optimizer 충실도와 causal stream 일치 중 어느 것을 주 비교로 둘지. GEPA ordered-sampler 또는 window 방식 중 허용 범위.
2. **A03/B07:** held-out val 사용 허용 여부와 current/seen/all 중 범위.
3. **A04/G02/G03:** 예산의 기준과 실제 숫자, stage별 배분.
4. **A05/C04/C06/C07:** 팀 프롬프트 그대로인지, 공통 seed/고정 환경 정책으로 정렬할지, GEPA reflection 원문까지 재현할지.
5. **C01/C02/C03:** actor/teacher 모델, endpoint, thinking/reasoning, token 설정.
6. **A07/S01/S02:** task별/배치별 업데이트와 epoch/batch 수.
7. **B05:** benchmark 전환 시 학습 지침만 전이할지 optimizer state까지 이어갈지.
8. **E01/E02/E03:** 공식 부분 점수 vs binary success, SkillOpt gate metric, benchmark aggregate.
9. **S11/S12/S13:** slow update, 해당 gate, meta skill의 활성화와 실제 비교 방식.
10. **E06/E07/E08:** checkpoint 의미, test 일정, 이전 benchmark forgetting 평가 범위.
11. **P01/P05/P06/P07:** 가용 endpoint/GPU/Docker 자원과 동시 run·Terminal session 한도, 기존 실험과의 우선순위.
12. **F01/F04/F05:** 오류 처리·부분 평가의 수락 정책과 benchmark/actor 제한.

예를 들어 “기존 seed/split/actor를 유지하고 shared instruction을 학습한다”는 선택까지는 명확해도,
GEPA의 900 metric calls와 SkillOpt의 130 unique tasks는 같은 비용이나 같은 업데이트 기회가 아니다.
이 숫자들이 정해지기 전에 전체 matrix를 실행하면 나중에 비교 조건 때문에 재실행할 가능성이 크다.

비용 예시로, SkillOpt에서 130 task를 batch 10으로 한 번 순회하고 매 step마다 새 후보를 130개 val 전체에서
평가한다면, 후보 13회의 validation만으로 1,690 rollout이다. 초기 val 130과 train 130을 더하면
test·추가 재평가 전에도 1,950 rollout이 된다. 실제 비용은 후보 생성/거절·cache·평가 scope에 따라 달라지며,
이 숫자는 설정 권장값이 아니라 **unique train 수만으로 비용을 맞출 수 없는 이유**를 보여 주는 계산이다.

## 13. 결정 이후의 구현 순서와 완료 기준

1. 기존 `execute`를 호출하는 공통 frozen-instruction actor/benchmark bridge를 만든다.
2. BFCL 하나에서 DSPy predictor trace와 SkillOpt EnvAdapter/dataloader를 연결하고 실제 prompt 주입·점수·feedback·예산을 검증한다.
3. AppWorld의 공통 policy/shortlisting/finish 계약과 Terminal-Bench의 내부 Terminus-2 주입을 같은 bridge로 연결한다.
4. isolated의 완전 초기화, sequential 전이 경계, interleaved manifest 순서와 sampler를 구현한다.
5. run/rollout/reflection/test의 계층별 quota, 환경 namespace, 복구와 frozen checkpoint queue를 붙인다.
6. 작은 pilot으로 actual task 접근 순서, 정보 분리, resume 동일성, 비용 계수를 확인한 뒤 전체 matrix를 실행한다.

완료 판정에는 단순 CLI import 외에도 다음이 필요하다: 세 benchmark의 실제 수행, 모든 mode의 상태 계약,
seed 간 격리, parallel/serial 동등성의 허용 범위, 중단 후 중복 update 없는 resume,
test를 학습에 사용하지 않는 frozen 평가, legacy 결과와 구분되는 provenance.

## 공식 소스 근거

- [DSPy GEPA 진입점, 설정, compile](https://github.com/stanfordnlp/dspy/blob/1a4f8372eb47e28d975cebf5d09ca6d7f863d32b/dspy/teleprompt/gepa/gepa.py)
- [DSPy instruction proposer와 입출력 계약](https://github.com/stanfordnlp/dspy/blob/1a4f8372eb47e28d975cebf5d09ca6d7f863d32b/dspy/teleprompt/gepa/instruction_proposal.py)
- [GEPA core sampler, callback, 예산 API](https://github.com/gepa-ai/gepa/blob/f1cfdee2856e9dc4b5b4dc5cc2e391d0aa20d819/src/gepa/api.py)
- [SkillOpt benchmark extension 계약](https://github.com/microsoft/SkillOpt/blob/1a44050fc53aa4938a7a3927b6cca550491bfe51/docs/guide/new-benchmark.md)
- [SkillOpt 기본 config](https://github.com/microsoft/SkillOpt/blob/1a44050fc53aa4938a7a3927b6cca550491bfe51/configs/_base_/default.yaml)
- [SkillOpt trainer: 일반 gate, slow gate, meta epoch 처리](https://github.com/microsoft/SkillOpt/blob/1a44050fc53aa4938a7a3927b6cca550491bfe51/skillopt/engine/trainer.py)
- [SkillOpt gate score와 strict improvement](https://github.com/microsoft/SkillOpt/blob/1a44050fc53aa4938a7a3927b6cca550491bfe51/skillopt/evaluation/gate.py)
