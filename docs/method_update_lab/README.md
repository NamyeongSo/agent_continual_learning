# One Update Lab

A-MEM, AutoSkill, Harness, ReasoningBank의 한 번의 업데이트를 실제 Exgentic 구현과 함께 탐색하는 한국어 HTML 가이드입니다.

`dist/index.html`을 브라우저로 열거나, **`dist/standalone.html` 한 파일만 복사해서 열면 됩니다.**
외부 JavaScript·CSS·폰트·API가 필요하지 않습니다. `a_mem.html`, `autoskill.html`, `harness.html`,
`reasoning_bank.html`은 방법별로 바로 열 수 있는 페이지입니다.

- 단계 이동, 진행 슬라이더, 자동 재생, 방향키 탐색
- 완료한 실행과 누락된 실행을 같은 목록 API 예시로 비교
- A-MEM의 4가지 진화 분기와 100번째 진화 시 검색 인덱스 재구축
- AutoSkill의 add/merge/discard, 낮은 유사도에서 merge → add, 검색 혼합 가중치·threshold 실습
- Harness의 프롬프트·메모리·스킬 연산 선택, no-op, 외부 예외 rollback
- ReasoningBank의 성공/실패 평가, 빈 줄 파서, 임베딩 누락 시 skip
- 단계별 입력·출력·영구 저장 상태, 필드 단위 전후 비교, 현재 단계 JSON 내보내기
- 실제 함수·프롬프트 57개 발췌본, 원본 경로·행 번호·파일 SHA-256

실행 데이터와 LLM 출력·검색 점수는 교육용 고정 예시입니다. 실제 모델 또는 임베딩 모델을 호출하지 않습니다.
상태 객체는 설명에 필요한 필드만 담은 축약 표현이며, 원본 체크포인트 파일 형식과 동일하지 않습니다.
Harness 예시는 기존 스냅샷 버전 0에서 시작합니다. 모델의 수치 성능을 비교하는 페이지가 아닙니다.

## 소스 갱신과 검증

저장소 루트에서 실행합니다.

```bash
python docs/method_update_lab/capture_sources.py
python docs/method_update_lab/build_pages.py
python docs/method_update_lab/verify_lab.py
```

검증에는 Python Playwright와 Chromium이 필요합니다. 페이지를 사용하는 데는 필요하지 않습니다.
`verify_lab.py`는 직접 파일을 열어 분기별 저장 상태·탐색·내보내기·모바일 레이아웃을 검사합니다.

`source-data.js`는 캡처 시점의 **작업 트리 파일**을 사용합니다. 저장소 HEAD와 별도로 각 파일의 해시를
포함하므로 커밋하지 않은 소스 변경도 구분할 수 있습니다. 캡처 후에는 `build_pages.py`로 단일 HTML도
다시 생성해야 합니다.

발췌 소스는 Apache-2.0 라이선스를 따르며 ReasoningBank의 Google 원저작권 표기를 포함합니다.
