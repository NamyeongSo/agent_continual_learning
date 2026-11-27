# 공통 reflector · 역할별 temperature · medium 실험

2026-11-11 cohort: `outputs/qwen_shared_reflection_medium_fresh1_four_baselines_20261111`.
기존 epoch 2 continuation과 분리하여 fresh 1 epoch로 실행한다.

- BFCL → AppWorld → Terminal-Bench 2.0 순서. 각 benchmark에서 네 방법론 × seeds 42–44, 총 36개.
- ACE AUD와 PEACE AUD w/o alignment는 `exgentic.reflection`의 동일한 System과 instruction-aware 규칙을 사용한다.
  최초 원인 결정 또는 결정적 성공 행동을 선택하고 기존 소비 로직에 연결할 때만 span으로 인코딩한다.
  후속 사용자 지시가 공개된 시점을 보존한다. ACE는 entry assessment를 bullet counter로 변환하고 reflection 전체를
  curator에 전달한다. PEACE는 해당 지점부터 실제 환경 replay를 수행한다.
- GEPA·SkillOpt의 optimizer prompt와 알고리즘은 기존 구현을 유지한다. 네 방법론 모두 공통 초기 actor context
  `shared-empty-v1`와 빈 learned policy에서 시작한다.
- 모든 actor(학습·replay·고정 평가)는 temperature 0.0. Reflector·curator·mutator·optimizer·judge는 0.7.
  전체 요청에 `reasoning_effort=medium`과 `enable_thinking=true`를 적용한다.
- Qwen template은 low/medium/high에 고유한 예산을 배정하지 않는다. medium 필드를 전달하는 것과 별도로,
  실제 측정에 기반한 reasoning 상한 16,384/request를 적용한다. 비추론 completion 예산은 actor 2,048/환경 turn,
  분석 8,192/request로 유지한다. 새 opt-in actor budget 버전은 `actor-turn-v2-nonreasoning`이다.
  실제 비용의 completion usage에서는 reasoning을 차감하지 않는다.
- 비추론 길이와 반복은 생성 후 action 실행 전에 검사하며 초과 응답을 조용히 자르지 않는다.
  Actor 길이 초과는 trajectory 실패 종료, 분석 응답의 잘림은 유한 재시도 대상이다.
  Actor input 100,000/turn, BFCL 30 turns, AppWorld/Terminal-Bench 40 turns는 유지한다.
- PEACE: AUD, alignment off, beam 2, candidates 4, current 1, past 3, replay workers 28, LLM slots 32.
- GEPA·SkillOpt: initial full Val, 단일 train stream, 누적 학습 예산 무제한, optimizer seed 42.
  Data-order seed는 42–44. PEACE의 algorithm RNG는 해당 run seed를 사용한다.
- Train/Val/Test: BFCL 50/50/50, AppWorld 50/50/50, Terminal-Bench 30/30/29.
  Seed별 membership은 동일하고 split 내부 순서만 달라진다. AppWorld는 official binary success(TGC)로 집계하며
  original environment policy bytes/SHA를 train/replay/frozen evaluation에 고정한다.
- 전체 최대 12개 실험. 최초 실행은 ACE·PEACE의 6개와 GEPA·SkillOpt seed42를 시작했다.
  2026-11-11 사용자 변경에 따라 **빈 자리 즉시 투입하는 rolling queue**로 전환했다.
  하나의 학습·최종 Test가 끝나면 다른 seed의 완료를 기다리지 않고 BFCL → AppWorld → Terminal-Bench 우선순위로
  실행 가능한 다음 실험을 투입한다. GEPA·SkillOpt는 각 방법론·benchmark의 seed42 실제 시작을 기준으로
  seed43 +15분, seed44 +30분 간격을 유지한다. 겹쳐 실행되는 benchmark가 같은 포트를 쓰지 않도록
  12개 포트 블록을 현재 실행 중인 작업에 독점 배정하고, 완료 후 재사용한다.
- 새 실행 프로세스의 NOFILE soft limit은 4,096이다. 기존 vLLM 서버를 유지한다.

## 검증 근거와 한계

`outputs/shared_reflection_medium_calibration_20261111`에 실제 inference/training 진단과 prompt 비교를 보존한다.
네 방법론의 소규모 학습, AppWorld와 native Terminal-Bench actor 추론, 최종 prompt로 ACE·PEACE 각 1개 추가 학습을
실행했다. 진단용 val/test 라벨의 task도 canonical train pool에서만 가져왔다.

최종 prompt 비교는 기존 train trajectory 12개(3 benchmarks × 성공/실패 × 개발/holdout)에서 수행했다.
동일 모델을 이용한 blind rubric은 20점 만점이다. 개발 점수는 기존 PEACE 13.17, first-decision 16.50,
instruction-aware 16.50. 후속 지시 causal regression을 통과한 instruction-aware를 개발 자료만으로 선택했다.
Holdout에서는 기존 19.17, first-decision 15.00, 선택안 18.00으로 **선택안의 우월성을 입증하지 못했다**.
실제 task accuracy 개선 여부는 본 실험으로 측정한다. Prompt 변경 뒤 사용한 holdout case ID는 이전 비교의
holdout case와 겹치지 않도록 교체했다. 이는 단일 사전 등록된 독립 검증을 대신하지는 않는다.

Calibration 1,182개 응답 중 actor 774개가 8,192 reasoning cap을 사용했고 그중 2개가 경계에 도달했다.
Actor 205개와 분석 23개를 16,384 cap으로 추가 실행했으며 해당 집단에서는 경계 도달·반복·길이 종료가 없었다.
8,192 cap 분석 180개에서는 반복 응답 1개를 거부했다. 이 관측으로 16,384의 유한 reasoning cap을 선택했다.
짧은 진단은 장기 학습의 모든 정상 길이나 실패 유형을 보장하지 않는다.

## 실행과 관측

`scripts/utils/run_shared_reflection_matrix.py`는 코드 snapshot과 source SHA 목록, immutable split/profile 및
AppWorld policy를 고정한다. Benchmark installation과 venv는 기존 격리 환경을 참조한다. 준비 시 36개 config를
dry-run하고 benchmark·seed별 네 방법론의 공통 초기 context metadata가 일치하는지 확인했다.

`scripts/utils/serve_reasoning_bridge.py`는 vLLM token IDs로 reasoning/visible usage를 분리한다.
원래 usage와 profile SHA, 실제 temperature/effort, 반복·길이 종료 정보를 기록한다.
비공개 Docker 연결 환경은 별도 ignored 0600 파일에 두고 공개 manifest/HTML에서 제외한다.

실시간 화면: `http://166.104.110.49:8880/#shared-reflection`.
실험 기록장: `http://166.104.110.49:8878/#experiment/exp-20261111-8497707d`.
실제 실행 상태는 `progress.json`, `status.json`, 각 run의 `job_status.json` 및 `job_complete.json`을 따른다.
인프라 오류를 공식 task 실패로 바꾸지 않으며 유한 복구 후에도 누락 결과가 있으면 benchmark barrier를 해제하지 않는다.
