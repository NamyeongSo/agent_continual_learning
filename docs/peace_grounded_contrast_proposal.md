# PEACE: 검증된 대조 사례로 조건부 규칙 만들기

상태: **선택적으로 활성화하는 구현 완료**. `--grounded-proposals --grounded-retrieval ab`로 사용한다.
기존 기본 PEACE·policy-shift 12개와 별도 cohort에서 실행하며 기존 기본값은 유지한다.

## 연구 질문과 핵심 가설

목표는 여러 trajectory에서 재사용할 수 있는 playbook 규칙을 만드는 것이다.
과거 사례를 추가하는 데서 끝내지 않고, **관찰로 뒷받침되는 주장 → 사례 간 비교 → 적용 조건이 있는 규칙**을
proposal의 필수 산출물로 만든다.

검증할 가설은 다음과 같다.

1. Reflection에서 관찰과 추측을 구분하면 잘못된 원인 설명이 규칙으로 굳어드는 일이 줄어든다.
2. 유사 trajectory 조건을 보장하면 A가 무관한 사례를 고르는 문제가 줄어든다.
3. 사례의 공통점·차이·예외를 명시하게 하면 현재 task의 절차를 늘어놓는 proposal보다 적용 범위가 명확해진다.

효과는 아직 입증되지 않았다. 아래 검증 단계와 비교 실험은 이 가설을 시험하기 위한 것이다.

## 1. Reflection을 검증 가능한 주장으로 바꾼다

입력은 현재 parent trajectory, 실제 actor history, 공개 tool contract, 기존 playbook, 공식 결과다.
기존처럼 parent당 두 reflection을 만들 수 있지만, 각 reflection은 아래 항목을 분리해야 한다.

| 항목 | 내용 |
|---|---|
| 관찰된 현상 | 어느 action·observation·최종 상태에 무엇이 기록돼 있는가 |
| 결정 당시 정보 | 해당 action의 요청 history에 어떤 지시와 값이 이미 있었는가 |
| 제안한 설명 | 관찰을 설명할 수 있는 가설은 무엇인가 |
| 근거와 반대 근거 | 설명을 지지하는 기록과 회복·반례가 되는 후속 기록 |
| 수정할 결정 | 실행한 action과 무엇이 달라져야 하는가 |
| 적용 조건 | 어떤 공개 지시·상태·도구 조건에서 이 수정이 필요한가 |
| 불확실성 | 기록만으로 확인할 수 없는 것은 무엇인가 |

Action index·turn·history 위치를 근거 ID로 붙인다. 학습할 규칙에는 이 감사용 ID를 넣지 않는다.
구현에서는 모델이 인용문을 다시 쓰지 않고 source ID를 선택하며, 프로그램이 해당 기록의 원문을 붙인다.
자료는 action과 observation이 번갈아 나오는 시간순 timeline으로 제공한다.
Draft·review·비교·operation 출력에는 JSON schema를 적용하고, 시간·근거·API 검사는 별도로 수행한다.

### 검증 절차

프로그램 검사를 먼저 하고, 이어 별도의 evidence review 요청이 원문 근거와 주장을 대조한다.
Review는 제안자의 확신이나 문장 유사도가 아니라 각 주장에 대해 `supported / uncertain / contradicted`를 반환한다.
이는 확률 보정된 신뢰 점수도, 인과관계의 증명도 아니다.

- **시간 순서:** action i의 수정은 실행 전 history H_i에 있던 정보만 사용해야 한다.
  BFCL `finish` 뒤에 공개된 다음 지시를 이전 turn에서 수행했어야 한다고 처방하지 않는다.
- **회복 여부:** 중간 오류가 이후 복구됐는지 확인한다. 복구됐다는 이유만으로 모든 영향이 사라졌다고 단정하지도 않는다.
  최종적으로 남은 문제와 그 근거를 찾는다.
- **수정의 일관성:** 잘못됐다고 지목한 action과 expected action이 실질적으로 같은지 검사한다.
  달라지는 argument·순서·조건을 구체적으로 적어야 한다.
- **정보 출처:** 공식 checker는 실패 현상을 찾는 데 쓸 수 있지만, checker에만 있는 이름·토큰·정답 문자열을
  actor가 알고 있던 값처럼 처방하지 않는다.
- **공개 API:** 도구 이름, 인수, 기본값, 종료 방식은 공개 contract와 일치해야 한다.
- **설명 범위:** 전체 task의 성공/실패만으로 특정 action이 원인이라고 확정하지 않는다.

실제 actor 요청 history가 가장 확실한 시간 근거다. 없는 과거 기록은 순서가 보존된 지시·observation으로 복원하고,
공개 시점을 확정할 수 없는 주장은 uncertain으로 남긴다.

명확한 오류는 근거와 함께 한 번 수정 요청한다. 여전히 불확실한 원인 설명은 검색용 근거나 policy-shift의
실패 span으로 승격하지 않는다. 실패 현상 자체가 명확하다면 현상 기록은 보존하되, 원인은 미상으로 남길 수 있다.

### 성공 사례도 조건을 추출한다

"모든 요구를 성공적으로 수행했다"는 요약만으로는 충분하지 않다.
성공 사례에서는 공개된 전제조건, 이를 충족한 행동, 확인된 결과를 연결한다.
이 연결을 뒷받침할 기록이 없으면 일반 규칙의 지지 사례로 쓰지 않는다.

## 2. 검증된 기록만 검색 archive에 넣는다

Archive 항목은 다음을 가진다.

```text
benchmark, task_id, iteration, parent_rank
full_recorded_actions, original_actor_history_reference
supported_observation, supported_correction, applicability_conditions
evidence_ids, recovery_or_counterevidence, unresolved_uncertainty
official_outcome, review_result, method_version
```

현재 iteration 이전의 같은 benchmark 학습 기록만 사용한다. 현재 task와 val/test는 제외한다.
서로 다른 task를 두 사례로 세고, 같은 task의 중복 reflection은 대표 기록으로 묶는다.
선택되지 않은 parent의 기록도 근거 검사를 통과했다면 포함할 수 있다.

Archive는 현재 학습의 신뢰할 수 있는 사실을 저장하는 장소가 아니라, 검토된 관찰과 제한된 주장을 저장하는 장소다.
같은 모델이 만든 두 reflection이 일치한다는 이유로 검증 완료로 취급하지 않는다.

## 3. A/B 검색의 우선순위를 명확히 고정한다

Trajectory 유사도 F는 기존 **전체 action multiset strict F1**을 그대로 쓴다.
Tool name과 canonical arguments, 중복 count를 유지하고 순서·call ID·validation metadata는 제외한다.
점수가 낮은 경우 이를 높은 의미적 유사도로 설명하지 않는다.

Reflection 유사도 R은 검증된 **적용 조건·관찰된 문제·수정 내용**으로 계산한다.
단순 성공 선언, task 고유 정답 값, 감사용 ID는 embedding 텍스트에 넣지 않는다.
MiniLM의 모델·revision은 고정한다. 필드명을 포함한 검증 내용을 직렬화하고, 256 wordpiece를 넘으면
token 단위로 분할해 각 chunk를 독립 인코딩한 뒤 평균·L2 정규화한다. Chunk 길이는 special token을 포함해
256 이하로 제한하고 버전을 기록한다. 뒷부분을 조용히 버리지 않는다.

### A: 행동이 유사하고 설명·적용 조건에 의미 있는 차이가 있는 사례

1. F > 0인 과거 task만 후보로 둔다.
2. Task별 대표 기록과 task 순서는 **F 내림차순**으로 정한다.
3. F가 같은 경우에만 R 오름차순, stable ID 순으로 정한다.
4. 이 순서대로 최대 8개 task의 근거 카드를 검사한다. 8은 초기 설계값이며 최적값이라는 주장이 아니다.
5. 비교 가능한 결정인데 수정·적용 조건이 실제로 다른지 확인한다. 단순히 주제가 다른 사례는 제외한다.
6. 통과한 기록 중 순위가 가장 높은 서로 다른 task를 최대 요청 개수까지 선택한다.

즉, **역할 조건을 충족한 집합에서 F가 우선**이다. R을 낮추기 위해 더 낮은 F를 우선하지 않는다.
상위 절반이나 경계 동점 확장 규칙은 사용하지 않는다. 모든 F가 0이면 유사 사례 없음으로 처리한다.
F > 0은 필요한 조건일 뿐 의미적 관련성의 충분조건은 아니므로 5번 검사가 필요하다.

### B: 행동 맥락이 다르지만 같은 수정 원리를 지지하는 사례

1. R 내림차순으로 최대 8개 과거 task를 검사한다. R 동점에서는 F 오름차순, stable ID 순이다.
2. 동일한 문제·수정 원리가 관찰 근거로 지지되는지 검사한다.
3. 전체 action이 동일한 사례는 제외한다. 개체 이름만 다른 복제 사례를 맥락 다양성의 근거로 과장하지 않는다.
4. 서로 다른 행동 맥락에서도 현재 규칙의 조건이 성립하는지 확인한 기록을 최대 요청 개수까지 선택한다.

예를 들어 파일 생성과 트윗이 모두 문자열을 다룬다는 것만으로 B가 되지 않는다.
두 사례 모두에서 **사용자가 원문 보존을 요구했고 추가 가공이 그 요구를 위반했다**는 근거가 있어야 한다.
Checker에만 있는 특정 문자열 형식은 이 공통 조건의 근거가 될 수 없다.

### 개수와 fallback

기본안은 A 한 개와 B 한 개, 총 최대 두 개다. A-only/B-only/Random은 비교용 변형으로 남긴다.
A+B는 A를 먼저 고르고 같은 task를 B에서 제외한다.

유효한 사례가 0개나 1개라면 그 수만 사용한다. Random으로 몰래 채워 A/B라고 부르지 않는다.
사례가 없어도 검증된 현재 기록만으로 기존 PEACE 방식의 proposal을 만들 수 있다.
각 요청에 요청 수·실제 수·부족 사유를 기록한다.

## 4. 비교 결과를 먼저 쓰고 proposal을 만든다

Mutator에는 현재 근거 카드와 최대 두 과거 근거 카드를 제공한다.
각 카드에는 해당 시점의 지시, 문제 action, 관련 observation, 회복 구간, 공개 API의 필요한 정의를 인용한다.
짧은 카드만으로 판단이 부족하면 원래 trajectory의 해당 구간을 추가로 읽고, 근거를 확보할 수 없으면 abstain한다.
긴 원문을 통째로 주는 방식을 기본으로 삼지 않는다.
현재 구현의 archive 카드에는 검토를 통과한 literal 인용을 제공한다. 추가 원문이 필요한 상태라면
새 환경 상호작용을 요청하는 대신 그 관계를 승인하지 않고 current-only로 fallback한다.

출력은 비교 결과와 playbook operation을 함께 포함한다. 이는 한 요청의 구조화된 산출물로 구현할 수 있다.

```json
{
  "comparison": {
    "shared_observation": "공통으로 확인된 사실과 evidence IDs",
    "relevant_difference": "결과나 수정 방향을 바꾸는 공개 조건",
    "supported_invariant": "조건이 성립할 때 재사용할 수 있는 절차",
    "counterevidence": "반대 기록 또는 아직 확인하지 못한 예외",
    "current_applicability": "현재 결정 시점에 조건이 성립하는 근거"
  },
  "rule": {
    "condition": "actor가 당시 관찰할 수 있는 적용 조건",
    "procedure": "조건부 행동 지침",
    "check": "수행 결과를 확인하는 관찰",
    "exception": "근거가 있는 예외 또는 명시적 적용 범위 제한",
    "support_ids": ["current", "past"],
    "evidence_scope": "current_only 또는 cross_task_supported"
  },
  "operations": []
}
```

`operations`는 기존 ADD/UPDATE/DELETE schema를 따른다. 위 JSON은 설계 설명용 골격이며 완성된 API schema는 아니다.
각 operation은 rule과 support IDs에 연결한다. Playbook에는 benchmark 범위와 조건부 지침을 간결히 쓰고,
감사용 비교 자료·task ID·hidden answer는 iteration artifact에만 둔다.

두 개 이상의 서로 다른 task에서 조건과 절차가 지지될 때 cross_task_supported로 기록한다.
이는 일반화가 증명됐다는 뜻이 아니다. 현재 한 사례만 있으면 current_only 범위로 남긴다.
예외가 관찰되지 않았다면 "없음"으로 확정하지 않고 아직 확인하지 못했다고 기록한다.

## 5. 근거를 넘는 수정은 적용하지 않는다

최종 operation 검사에서 다음을 확인한다.

- 모든 명령적 주장이 현재 또는 과거의 공개 근거에 연결되는가.
- 미래 지시나 checker의 비공개 정답을 미리 요구하지 않는가.
- 기존 bullet을 UPDATE하면서 새 근거와 충돌하는 오래된 문장을 그대로 남기지 않았는가.
- 여러 규칙을 묶어 replay가 좋아졌다는 이유로 근거 없는 문장을 함께 승인하지 않았는가.
- 같은 기능의 기존 규칙을 중복 ADD하지 않고 적절히 UPDATE하는가.

범위가 너무 넓으면 조건을 좁히고, 지지되지 않는 문장은 제거한다. 근거 있는 수정이 남지 않으면 operations는 빈 배열이다.
서로 다른 task를 억지로 하나의 규칙으로 묶지 않는다.

이 검사는 proposal의 문장과 근거를 점검한다. 기존 w/o alignment 후보 선정에 alignment 점수나 새 Pareto 축을 추가하지 않는다.

## 6. 후보 선정은 기존 방식을 유지한다

Parent 두 개, 최대 네 개 proposal, 현재·과거 Pareto라는 골격은 유지한다.
Reflection·operation 검사를 통과한 후보만 기존 evaluator에 전달한다.
두 개의 유효한 후보를 확보하지 못하면 기존 beam을 보존하고 `no_update_insufficient_grounded_candidates`로 기록한다.
이를 공식 task 실패나 infrastructure 실패와 혼합하지 않는다.

- **실제 replay 버전:** 현재1·과거 p의 환경 replay 공식 점수로 기존 Pareto를 계산한다.
- **Policy-shift 버전:** 검증된 span의 저장 history prefix에서 action만 생성한다.
  저장된 학습 action을 기준으로 span 내 action별 strict F1을 평균한다.
  성공 source는 F1, 실패 source는 1−F1을 평가값으로 쓰고 기존 현재·과거 Pareto를 적용한다.
  실패 원인·수정 결정이 불명확한 span은 이 방식으로 정책 이탈을 보상하지 않는다.

Policy shift는 실제 task 개선이나 설명의 인과성을 직접 측정하지 않는다.
두 evaluator는 별도 변형으로 비교하고, 기본 방법론 검증 중에 선택 기준까지 함께 바꾸지 않는다.
Past 수·memory 선정 방식·task order도 같은 비교에서는 맞춘다.

기존 actor policy bytes, 모델, split, algorithm seed와 actor-turn-v1 예산을 유지한다.
BFCL actor는 입력100,000·출력2,048 tokens/turn, 최대30turns이며 replay prefix와 측정된 retry를 포함한다.
Reflection·review·mutation은 별도 분석 예산과 유한한 재시도 횟수를 명시하고 실제 비용을 기록한다.
기본 actor temperature0, reflection/proposal0.7을 사용한다. 추가 review의 sampling과 예산도 실행 전에 고정한다.
구현의 review temperature는0.7이며, 각 reflection은 최대 두 번 작성·검토한다.
후보마다 최종 operation review를 한 번 수행한다. A/B 관계 검토는 역할별 최대8개 task를 한 요청에서 비교한다.
검색용 RNG는 기존 past sampling RNG를 소비하지 않는다.

등록된 실행은 BFCL A+B, seeds42–44의3개, fresh2epochs, 실제 replay current1/past3, actor/action workers28이다.
기존 role-temperature 기본 PEACE 세seed가 학습·최종Test를 모두 마친 뒤 시작한다.
그 이후 기존 queue에 남는 policy-shift lane은3개이므로 새3개와 합쳐 동시성6이하를 유지한다.

## 사례: 잘못된 조기 종료 설명을 어떻게 막는가

기존 A/s42/iter1의 실제 순서는 다음과 같았다.

1. 파일을 찾는다.
2. `finish`가 다음 turn의 복사 지시를 반환한다.
3. 잘못된 위치에 `backup_tests`를 만든다.
4. 경로 오류를 회복하고 올바른 위치에 복사는 성공한다.
5. 잘못 만든 추가 디렉터리가 남아 공식 상태가 불일치한다.

새 reflection은 2번을 미래 작업 미수행으로 비난할 수 없다.
남은 상태 차이와 3번의 side effect를 연결하는 관찰을 제시해야 한다.
같은 공개 경로 조건이 있는 과거 사례로 보강할 수 있다면 다음 형태의 규칙을 제안한다.

> BFCL GorillaFileSystem에서 파일을 특정 폴더로 복사할 때 현재 위치와 대상 폴더의 부모 위치를 먼저 확인한다.
> 대상 위치를 공개 API가 지원하는 방식으로 찾아가고, 필요한 폴더가 그 위치에 없을 때만 생성한다.
> 복사 결과와 의도하지 않은 추가 폴더 생성 여부를 확인한다.

이 규칙은 "항상 finish를 늦춰라"나 "mkdir를 쓰지 마라"로 확장하지 않는다.
과거 사례가 이 조건을 지지하지 않으면 현재 사례에서 지지되는 범위로만 기록한다.

## 검증 계획

### 먼저 저장된 입력에서 proposal의 기여를 분리한다

같은 trajectory·기존 playbook·검증된 reflection을 고정하고 다음을 비교한다.

| 비교군 | Reflection 검증 | 수정한 검색 | 명시적 사례 비교 |
|---|---|---|---|
| 기존 PEACE | 없음 | 사례 없음 | 없음 |
| 검증만 적용 | 있음 | 사례 없음 | 없음 |
| 검증 + 사례 제공 | 있음 | 있음 | 없음 |
| 전체 방법 | 있음 | 있음 | 있음 |
| Random 대조 | 있음 | 같은 적격 archive에서 무작위 | 있음 |

첫 두 비교군의 raw/검증 reflection은 각각 보존한다. 사례의 효과를 볼 때는 검증된 현재 reflection을 같게 고정한다.
Random은 전체 방법이 실제로 받은 사례 수와 길이 예산을 맞춘다. 가능하면 요구된 A/B 관계가 성립하는
후보 집합 내에서 무작위 선택하는 대조도 추가해, 관계의 존재와 순위의 기여를 구분한다.
0개 fallback의 효과와 2개를 실제로 제공한 요청의 효과를 나눠 보고한다.

검사할 것은 다음과 같다.

- A의 역할 검사를 통과한 집합에서 더 높은 F를 버리고 낮은 F를 고른 위반 건수, F=0 처리, 실제 검색 coverage.
- Reflection의 시간 순서 오류, 회복 무시, 비공개 정보 의존, 수정의 모순.
- 규칙의 사실 근거, 현재 적용 가능성, 사례 고유 값 복사, 과도한 일반화.
- Cross-task support라고 표기한 규칙이 실제 두 task에서 지지되는지.
- 후보 중복, 무수정 빈도, 입력량·추가 호출·시간·비용.

방법 내부 review의 통과율을 품질의 정답률로 쓰지 않는다.
고정된 표본을 실패/성공·API별로 나눠 원래 기록과 함께 독립 검토하고, 검토자에게 variant를 가린다.

### 그다음 학습 효과를 비교한다

위 단계에서 의도한 입력 조건과 품질 개선이 확인되면 matched split·seeds·epochs·past 수·evaluator로
온라인 학습을 별도로 실행한다. Online과 최종 고정 정책의 공식 Train/Val/Test를 보고한다.
후보 생성량과 분석 예산 상한을 맞추고 실제 비용도 비교한다.
중단·인프라 누락은 측정 실패와 구분하며, 실험별 다른 진행 구간을 최종 성능 차이로 해석하지 않는다.

## 남는 한계

- Strict F1은 서로 다른 인수·개체 때문에 여전히 희소하다. Coverage 부족을 숨기지 않으며,
  정규화한 구조 유사도를 쓰려면 별도 변형으로 명시해야 한다.
- Evidence review도 모델 오류를 낼 수 있다. 근거 인용과 독립 표본 검토가 필요하다.
- 관찰상 지지되는 설명은 통제된 인과 실험과 다르다. Replay 점수도 묶인 규칙 각각의 정당성을 보장하지 않는다.
- 신뢰할 수 있는 사례가 적은 초기에는 current-only proposal이 많아진다.
- 추가 검토와 비교의 비용이 있다. 품질 개선과 최종 성능·비용을 함께 측정해야 한다.

진단 근거: 중단된 retrieval cohort의 `quality_audit/assessment.md`, `action_retrieval_audit.json`.
기존 프로토콜은 [PEACE Proposal Retrieval](./peace_proposal_retrieval.md)을 참고한다.
