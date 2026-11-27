**공통 fixed balanced split v1**

분할 구성 seed=42로 task ID 소속을 고정했습니다. 실험 seeds42/43/44는 각 벤치마크의 train/val/test 내부 순서만 바꾸며, 모든 방법은 같은 seed에서 같은 순서를 사용합니다. seed가 바뀌어도 task가 다른 split으로 이동하거나 재선정되지 않습니다.

| 벤치마크 | Train | Val | Test | 미사용 |
| --- | --- | --- | --- | --- |
| bfcl | 50 | 50 | 50 | 50 |
| appworld | 50 | 50 | 50 | 267 |
| terminalbench2 | 30 | 30 | 29 | 0 |

**BFCL: multi_turn_base**

주요 도메인별 무작위 추출 후12/13개 할당. 세 split의 task ID는 서로 겹치지 않습니다.

| 도메인 | Train | Val | Test | 미사용 |
| --- | --- | --- | --- | --- |
| 파일시스템 | 13 | 12 | 12 | 13 |
| 차량제어 | 13 | 13 | 12 | 12 |
| 주식거래 | 12 | 13 | 13 | 12 |
| 여행예약 | 12 | 12 | 13 | 13 |

**AppWorld: test_challenge**

139개 원본 시나리오에서 난이도 비율에 맞춰50개를 선정했습니다. 시나리오마다 변형1/2/3을 서로 다른 split에 하나씩 배정했습니다. 각 split은 같은50개 시나리오를 포함하며 task ID는 서로 겹치지 않습니다. task ID suffix는 난이도가 아니라 변형 번호입니다.

| 난이도 | 원본 시나리오 | 선정 시나리오 | Train | Val | Test |
| --- | --- | --- | --- | --- | --- |
| 1 | 24 | 9 | 9 | 9 | 9 |
| 2 | 50 | 18 | 18 | 18 | 18 |
| 3 | 65 | 23 | 23 | 23 | 23 |

| 변형 번호 | Train | Val | Test |
| --- | --- | --- | --- |
| 1 | 17 | 17 | 16 |
| 2 | 16 | 17 | 17 |
| 3 | 17 | 16 | 17 |

**Terminal-Bench2.0**

89개 task 전체를30/30/29개로 배정했습니다. 모든16개 category의 split 간 task 개수 차이는 최대1입니다. task가1개뿐인5개 category도2/2/1개씩 분산해 category 다양성을 맞췄습니다. 아래 난이도 목표를 동시에 만족시키며 category×difficulty 조합의 비율 편차도 최소화했습니다. 난이도는 데이터셋의 작성자 메타데이터 기준입니다.

| 난이도 | 원본 개수/비율 | Train | Val | Test |
| --- | --- | --- | --- | --- |
| easy | 4 (4.49%) | 1 (3.33%) | 2 (6.67%) | 1 (3.45%) |
| medium | 55 (61.80%) | 19 (63.33%) | 18 (60.00%) | 18 (62.07%) |
| hard | 30 (33.71%) | 10 (33.33%) | 10 (33.33%) | 10 (34.48%) |

| Category | Train | Val | Test | 총계 |
| --- | --- | --- | --- | --- |
| data-processing | 2 | 1 | 1 | 4 |
| data-querying | 1 | 0 | 0 | 1 |
| data-science | 2 | 3 | 3 | 8 |
| debugging | 1 | 2 | 2 | 5 |
| file-operations | 2 | 1 | 2 | 5 |
| games | 1 | 0 | 0 | 1 |
| machine-learning | 1 | 1 | 1 | 3 |
| mathematics | 1 | 1 | 2 | 4 |
| model-training | 1 | 1 | 2 | 4 |
| optimization | 0 | 0 | 1 | 1 |
| personal-assistant | 0 | 1 | 0 | 1 |
| scientific-computing | 3 | 3 | 2 | 8 |
| security | 3 | 3 | 2 | 8 |
| software-engineering | 9 | 9 | 8 | 26 |
| system-administration | 3 | 3 | 3 | 9 |
| video-processing | 0 | 1 | 0 | 1 |

**재현 및 검증**

현재 실험의 성공률/완료 여부는 분할에 사용하지 않았습니다. 원본 membership manifest는 고정된 기준 목록입니다. 실제 실험에는 benchmark_seed 항목으로 구성된 ordered manifest를 사용합니다. 기존 select_task_split_sections와 validate_task_split에서 42/43/44의 ID집합은 모두 같고 순서는 서로 다름을 검증했습니다. 원본 fingerprint, 정책, 선택 시나리오, 검증 hash를 함께 저장했습니다. train-as-val 실험은 이 원본 분할을 유지한 채 학습 중 검증 입력만train으로 연결하면 됩니다.

**사용 파일 및 seed 의미**

- 기준 구성: `bfcl_appworld_terminalbench2_fixed_balanced_seed42_v1.json`
- 실행용 seed별 순서: `bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json`
- 실험 seed는 data-order seed로만 사용합니다. 분할 구성 seed와 알고리즘/모델의 별도 난수 설정은 분리합니다. 후속 학습에서 실험 seed를 데이터 재선정이나 분할에 사용하지 않습니다.
