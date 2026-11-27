"use strict";

window.LiveExperiments = (() => {
  const names = {ace: "ACE", premise: "PREMiSE", ps_st_pareto: "ps_st_pareto", gepa: "GEPA", skillopt: "SkillOpt"};
  const benchmarks = {bfcl: "BFCL", appworld: "AppWorld", terminalbench2: "Terminal-Bench"};
  const phases = {training: "학습 중", validation: "Offline-val 중", updating: "업데이트 중", test: "Test 중",
    pending: "대기", paused: "일시중지", test_pending: "Test 대기", completed: "완료", stopped: "중단", error: "오류 확인"};
  const esc = (text) => String(text ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[c]);
  const number = value => Number(value || 0).toLocaleString("ko-KR");
  const ratio = p => p.total ? (100 * p.done / p.total).toFixed(1) + "%" : "예정";
  const kst = value => new Date(value).toLocaleString("ko-KR", {
    timeZone: "Asia/Seoul", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false
  }) + " KST";
  let active = null;
  const filters = {method: "all", variant: "all", benchmark: "all", state: "all", seed: "all"};
  const hiddenStorage = "exgentic.live.hidden.v1";
  const hiddenKinds = {batch: "실험 묶음", run: "개별 실행", table: "메인 테이블", queue: "CP 평가", shift: "Policy shift", study: "ACE 분석실험", module: "화면 영역"};
  const hiddenKey = (kind, id) => JSON.stringify([kind, String(id)]);
  function readHidden() {
    try {
      const rows = JSON.parse(localStorage.getItem(hiddenStorage) || "[]");
      return new Map((Array.isArray(rows) ? rows : []).filter(row => row && Object.hasOwn(hiddenKinds, row.kind)
        && typeof row.id === "string" && typeof row.label === "string")
        .map(row => [hiddenKey(row.kind, row.id), row]));
    } catch (_) { return new Map(); }
  }
  let hiddenItems = readHidden();
  const isHidden = (kind, id) => hiddenItems.has(hiddenKey(kind, id));
  function saveHidden(view) {
    try {
      localStorage.setItem(hiddenStorage, JSON.stringify([...hiddenItems.values()]));
      view.storageUnavailable = false;
    } catch (_) { view.storageUnavailable = true; }
  }
  function removeButton(view, kind, id, label) {
    return view.preview ? "" : `<button type="button" class="live-remove" data-live-hide-kind="${kind}"
      data-live-hide-id="${esc(id)}" data-live-hide-label="${esc(label)}" aria-label="${esc(label)} 화면에서 삭제"
      title="화면에서 삭제">삭제</button>`;
  }
  function visibleBatches(view) {
    return view.data.batches.filter(batch => (!view.id || batch.journal_id === view.id) && !isHidden("batch", batch.id))
      .map(batch => ({...batch, runs: batch.runs.filter(row => !isHidden("run", `${batch.id}/${row.id}`))}))
      .filter(batch => batch.runs.length);
  }
  function renderHiddenManager(view) {
    const dialog = view.root.querySelector("[data-live-hidden-dialog]");
    if (!dialog) return;
    dialog.querySelector("[data-live-hidden-items]").innerHTML = hiddenItems.size ? [...hiddenItems.entries()]
      .reverse().map(([key, item]) => `<li><div><span>${esc(hiddenKinds[item.kind])}</span><strong>${esc(item.label)}</strong></div>
        <button type="button" class="button small" data-live-restore="${esc(key)}">복원</button></li>`).join("")
      : '<li class="live-hidden-empty">숨긴 항목이 없습니다.</li>';
    dialog.querySelector("[data-live-restore-all]").disabled = !hiddenItems.size;
  }
  function updateVisibilityControls(view) {
    if (view.lastHidden && !isHidden(view.lastHidden.kind, view.lastHidden.id)) view.lastHidden = null;
    const manage = view.root.querySelector("[data-live-hidden-manage]");
    if (manage) manage.textContent = `숨긴 항목 (${hiddenItems.size})`;
    const notice = view.root.querySelector("[data-live-display-notice]");
    if (!notice) return;
    notice.hidden = !view.lastHidden;
    if (view.lastHidden) {
      notice.querySelector("span").textContent = `${view.lastHidden.label} · 화면에서 삭제했습니다.`
        + (view.storageUnavailable ? " 브라우저 저장이 불가능해 이번 화면에서만 유지됩니다." : " 이 브라우저에 저장됩니다.");
    }
  }
  function restoreHidden(view, key) {
    if (key === null) hiddenItems.clear(); else hiddenItems.delete(key);
    if (!view.lastHidden || !hiddenItems.has(hiddenKey(view.lastHidden.kind, view.lastHidden.id))) view.lastHidden = null;
    saveHidden(view);
    render(view);
    renderHiddenManager(view);
  }

  function bar(p, name, kind = "train") {
    const unknown = p.indeterminate || !p.total;
    return `<div class="live-bar ${kind}"><div class="live-bar-label"><span>${esc(name)}</span><strong>${p.indeterminate ? "진행 중" : `${number(p.done)} <span>/ ${number(p.total)}</span> <b>${ratio(p)}</b>`}</strong></div>
      <progress max="${p.total || 1}" ${unknown ? "" : `value="${p.done}"`} aria-label="${esc(name)}${unknown ? ": 진행률 미정" : `: ${p.done}/${p.total}`}"></progress></div>`;
  }

  function countTile(label, count, style = "") {
    return `<div class="live-stat ${style}"><span>${label}</span><strong>${number(count)}</strong></div>`;
  }

  function totals(rows, key) {
    return rows.reduce((sum, row) => ({done: sum.done + row[key].done, total: sum.total + row[key].total,
      errors: sum.errors + (row[key].errors || 0)}), {done: 0, total: 0, errors: 0});
  }

  function retrieval(row, batch) {
    const record = row.proposal_retrieval;
    if (!record) return "";
    const labels = {a: "A only", b: "B only", ab: "A+B", random: "Random"};
    const metric = value => typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
    const source = (id, label, kind, candidate) => `<details class="live-evidence" data-retrieval-key="${esc(`${batch}/${row.id}/${record.iteration}/${candidate}/${kind}/${id}`)}"
      data-retrieval-batch="${esc(batch)}" data-retrieval-run="${esc(row.id)}" data-retrieval-iteration="${record.iteration}"
      data-retrieval-source="${esc(id)}" data-retrieval-kind="${kind}"><summary>${esc(label)}</summary><div class="live-evidence-body"></div></details>`;
    const candidates = record.candidates || [];
    const preview = candidates[0]?.examples?.map(example => `${example.retrieval_role}: ${example.task_id}`).join(" · ");
    return `<section class="live-retrieval"><p><strong>Proposal 검색 · ${esc(labels[record.mode] || record.mode)}</strong></p>
      ${record.waiting_for ? `<p>대응 선행 run의 학습·최종 Test 완료 대기 · ${esc(record.waiting_for)}</p>` : ""}
      ${record.latest_status === "skipped" ? `<p class="live-retrieval-warning">최근 처리 iter ${number(record.processed)}: ${record.latest_reason === "no_update_insufficient_grounded_candidates" ? "근거 부족 · 기존 beam 보존" : "건너뜀"}</p>${record.latest_error ? `<details><summary>처리 오류</summary><pre>${esc(record.latest_error)}</pre></details>` : ""}` : ""}
      ${candidates.length ? `<p>최근 저장 iter ${number(record.iteration)} · ${esc(record.task_id)}</p>
        ${preview ? `<p class="live-retrieval-preview">후보 1 · ${esc(preview)}</p>` : ""}
        <details data-retrieval-key="${esc(`${batch}/${row.id}/retrieval`)}"><summary>후보별 가져온 사례 (${candidates.length}개 proposal)</summary>
        <p class="live-retrieval-help">A: action 유사 · reflection 상이 / B: reflection 유사 · action 상이. 점수는 0–1이며 strict F1은 exact action 일치 기준입니다. 완료된 iteration의 저장 기록을 표시합니다.</p>
        ${candidates.map(candidate => `<section class="live-retrieval-candidate"><strong>${esc(candidate.id)} · parent ${esc(candidate.parent_rank)}${candidate.valid === false ? " · mutation 무효" : ""}</strong>
          <p>${number(candidate.selected)} / ${number(candidate.requested)}개 선택 · 과거 ${number(candidate.eligible_tasks)} tasks${candidate.query_embedding_truncated ? " · 기준 reflection MiniLM 입력 잘림" : ""}</p>
          ${source(candidate.reflection_id, "기준 trajectory · reflection 보기", "query", candidate.id)}
          ${candidate.grounded_proposal ? `<details data-retrieval-key="${esc(`${batch}/${row.id}/${record.iteration}/${candidate.id}/grounded`)}"><summary>조건부 규칙 · 비교 결과 · 근거 검토</summary><pre>${esc(JSON.stringify(candidate.grounded_proposal, null, 2))}</pre></details>` : ""}
          ${candidate.examples.map(example => `<div class="live-retrieval-example"><div><b>${esc(example.retrieval_role)}</b> <code>${esc(example.task_id)}</code></div>
            <p>iter ${number(example.iteration)} · ${esc(example.id)}${example.embedding_truncated ? " · MiniLM 입력 잘림" : ""}</p>
            <dl><dt>Action strict F1</dt><dd>${metric(example.action_f1)}</dd><dt>Reflection cosine</dt><dd>${metric(example.reflection_cosine)}</dd></dl>
            ${source(example.id, "가져온 trajectory · reflection 원문", "source", candidate.id)}</div>`).join("") || '<p>이전의 다른 task 사례가 없어 추가 예시 없이 proposal을 생성했습니다.</p>'}
          </section>`).join("")}</details>` : '<p>첫 proposal 검색 기록 저장 대기</p>'}</section>`;
  }

  function evidenceHTML(source) {
    if (!source?.available) return '<p>해당 원문 기록을 찾을 수 없습니다.</p>';
    const reflection = source.reflection || {};
    return `<p><strong>${esc(source.task_id)}</strong> · 원래 공식 결과: ${source.success === true ? "성공" : source.success === false ? "실패" : "미기록"}</p>
      <h4>Task</h4><pre>${esc(typeof source.task === "string" ? source.task : JSON.stringify(source.task, null, 2))}</pre>
      <h4>Reflection · ${esc(reflection.id)}</h4><p>Span ${esc(reflection.span?.start)} → ${esc(reflection.span?.end)} (${esc(reflection.span?.type)})</p>
      <pre>${esc(reflection.analysis)}</pre><h4>Expected behavior</h4><pre>${esc(reflection.expected_behavior)}</pre>
      ${source.grounding ? `<details><summary>Reflection 근거 카드 · 검토 결과</summary><pre>${esc(JSON.stringify({card: reflection.evidence_card, review: source.grounding}, null, 2))}</pre></details>` : ""}
      <details><summary>전체 trajectory · ${(source.actions || []).length} actions</summary><pre>${esc(JSON.stringify(source.actions, null, 2))}</pre></details>
      <details><summary>Public tool contract</summary><pre>${esc(JSON.stringify(source.public_tool_contract, null, 2))}</pre></details>`;
  }

  function bindRetrieval(view, groups) {
    groups.querySelectorAll("details[data-retrieval-key]").forEach(details => {
      if (details.dataset.retrievalBound) return;
      details.dataset.retrievalBound = "true";
      const key = details.dataset.retrievalKey;
      details.open = view.retrievalOpen.has(key);
      const load = async () => {
        if (details.open) view.retrievalOpen.add(key); else view.retrievalOpen.delete(key);
        if (!details.open || !details.dataset.retrievalSource) return;
        const body = details.querySelector(".live-evidence-body");
        const query = new URLSearchParams({batch: details.dataset.retrievalBatch,
          run: details.dataset.retrievalRun, iteration: details.dataset.retrievalIteration});
        const requestKey = query.toString();
        body.textContent = "원문을 불러오는 중…";
        try {
          if (view.retrievalRequests.size > 48) view.retrievalRequests.delete(view.retrievalRequests.keys().next().value);
          if (!view.retrievalRequests.has(requestKey)) view.retrievalRequests.set(requestKey, (async () => {
            const response = await fetch(`/api/proposal-retrieval?${query}`, {cache: "no-store"});
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            return data;
          })().catch(error => { view.retrievalRequests.delete(requestKey); throw error; }));
          const data = await view.retrievalRequests.get(requestKey);
          const sources = details.dataset.retrievalKind === "query" ? data.queries : data.sources;
          body.innerHTML = evidenceHTML(sources[details.dataset.retrievalSource]);
        } catch (error) {
          body.textContent = `원문을 가져오지 못했습니다 (${error.message}). 접었다 펼치면 다시 조회합니다.`;
        }
      };
      details.addEventListener("toggle", load);
      if (details.open) load();
    });
  }

  function card(row, batch, view) {
    const current = row.current;
    const detail = row.state === "running" ? current
      ? bar(current, current.label, current.phase === "validation" ? "validation" : current.phase === "test" ? "test" : "activity")
      : `<div class="live-working"><span class="live-pulse"></span>${row.phase === "training"
        ? `학습 태스크 ${number(Math.min(row.train.done + 1, row.train.total))} 처리 중`
        : esc(phases[row.phase] || row.phase)}</div>` : "";
    const scores = ["train", "val", "test"].map(split => {
      const score = (row.scores || {})[split] || {};
      const measured = typeof score.value === "number" && Number.isFinite(score.value);
      return `<div class="live-score ${split}"><span>${esc(score.label || `${split === "val" ? "Offline-val" : split === "test" ? "Offline" : "Online"} 점수`)}</span>
        <strong>${measured ? (100 * score.value).toFixed(1) + "<small>%</small>" : "—"}</strong>
        <small>${measured ? `${score.complete ? "전체 완료" : "집계 중"} · ${number(score.count)}/${number(score.total)} task${score.metric === "appworld_tgc_v1" ? ` · 성공 ${number(score.successes)}/${number(score.count)}` : ""}` : "측정값 대기"}${score.errors ? ` · 오류 ${number(score.errors)}개 제외` : ""}</small></div>`;
    }).join("");
    return `<article class="live-run ${row.state}" data-run-id="${esc(row.id)}">
      <header><div><h3>${esc(row.variant_label || names[row.method] || row.method)} <span>· ${esc(benchmarks[row.benchmark] || row.benchmark)}</span></h3>
      <span class="live-seed">seed ${esc(row.seed)}${row.gpu != null ? ` · GPU ${esc(row.gpu)}` : ""}</span></div><div class="live-item-actions"><span class="live-state ${row.state}">${esc(row.phase === "validation" && row.validation_source === "train" ? "Train 선택 평가 중" : phases[row.phase] || row.phase)}</span>
      ${removeButton(view, "run", `${batch}/${row.id}`, `${row.variant_label || names[row.method] || row.method} · ${benchmarks[row.benchmark] || row.benchmark} · seed ${row.seed}`)}</div></header>
      <div class="live-scores">${scores}</div>
      ${bar(row.train, `학습 · ${row.train_unit}`)}${bar(row.test, "Test 처리", "test")}
      <div class="live-activity">${detail}${row.test.errors ? `<span class="live-error-count">Test 실행 오류 ${row.test.errors}개</span>` : ""}</div>
      ${row.role_temperatures ? `<p class="live-retrieval-help">Generator T=${esc(row.role_temperatures.actor)} · Reflection T=${esc(row.role_temperatures.reflection)} · Proposal T=${esc(row.role_temperatures.mutation)}</p>` : ""}
      ${retrieval(row, batch)}
      <footer><span>${row.pid ? `PID ${row.pid}` : esc(row.id)}</span>${row.train.done === row.train.total && row.train.total && !row.train.complete
        ? "<span>최종 업데이트 처리 중</span>" : ""}</footer></article>`;
  }

  function matches(row) {
    return (filters.method === "all" || row.method === filters.method)
      && (filters.variant === "all" || row.variant === filters.variant)
      && (filters.benchmark === "all" || row.benchmark === filters.benchmark)
      && (filters.seed === "all" || String(row.seed) === filters.seed)
      && (filters.state === "all" || row.state === filters.state
        || (filters.state === "attention" && ["error", "stopped"].includes(row.state)));
  }

  function checkpointQueues(view) {
    return (view.data.checkpoint_queues || []).filter(queue => !isHidden("queue", queue.id) && (!view.id || queue.journal_id === view.id
      || (queue.source_journal_ids || []).includes(view.id)));
  }

  function renderEvaluationOverview(view) {
    const target = view.root.querySelector(".live-evaluation-overview");
    const queues = checkpointQueues(view).filter(queue => queue.live_overview);
    target.hidden = !queues.length || isHidden("module", "evaluation-progress");
    target.innerHTML = queues.map(queue => {
      const counts = queue.task_counts || {};
      const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
      const phase = {running: "평가 진행 중", paused: "새 작업 배정 일시중지", completed: "평가 완료",
        stopped: "실행기 확인 필요", error: "오류 확인 필요", watching: "새 체크포인트 대기"}[queue.phase] || queue.phase;
      const canonical = queue.rows.filter(row => !row.alias && row.selected !== false);
      const split = name => canonical.reduce((sum, row) => {
        const value = row.splits[name] || {};
        return {done: sum.done + (value.done || 0), total: sum.total + (value.total || 0)};
      }, {done: 0, total: 0});
      const ports = Object.entries(queue.gpu_queries?.active || {}).map(([port, count]) =>
        `<span><b>포트 ${esc(port)}</b> ${number(count)} / ${number(queue.gpu_queries.per_endpoint_query_limits?.[port] ?? queue.gpu_queries.per_gpu_query_limit)}</span>`).join("");
      return `<section class="live-eval-card" data-evaluation-overview="${esc(queue.id)}">
        <header><div><span class="eyebrow">LIVE CHECKPOINT EVALUATION</span><h3>CP${queue.checkpoint_interval || 10} · Train / Val / Test</h3>
          <p>${esc(phase)}${queue.serving_gpu_indices?.length ? ` · GPU ${queue.serving_gpu_indices.map(esc).join(", ")}` : ""} · 체크포인트 ${number(queue.checkpoints)} / ${number(queue.expected_checkpoints || queue.checkpoints)}개 등록</p></div>
          <div class="live-item-actions"><button type="button" class="button primary" data-live-open-queue="${esc(queue.id)}">CP별 상세 보기 ↓</button>
          ${removeButton(view, "module", "evaluation-progress", "평가 실시간 요약")}</div></header>
        <div class="live-eval-stats">${countTile("완료 · 재사용 포함", counts.completed, "completed")}
          ${countTile("평가 실행", counts.running, "running")}${countTile("평가 대기", counts.pending)}
          ${countTile("실행 오류", counts.error, counts.error ? "error" : "")}</div>
        ${bar({done: counts.completed || 0, total}, "현재 등록된 평가 작업", "test")}
        <div class="live-eval-splits">${bar(split("train"), "Frozen Train")}${bar(split("val"), "Frozen Val", "validation")}${bar(split("test"), "Frozen Test", "test")}</div>
        <div class="live-eval-ports">${ports}<span>공통 호출 큐 대기 ${number(queue.gpu_queries?.waiting)}</span></div>
        <p class="live-eval-note">기존 결과 ${number(queue.initial_reused_outcomes)}건 재사용 · 동일 정책 ${number(queue.skipped)} CP 결과 공유 · ${esc(kst(queue.updated_at))} 갱신 · 5초 자동 갱신</p>
      </section>`;
    }).join("");
  }

  const shiftPhase = phase => ({planned: "CP 평가 종료 후 실행 예약", preparing: "입력 검증 중",
    preparing_histories: "History 준비 중", waiting_checkpoint_queue: "CP 평가 종료 대기",
    probing: "추론 진행 중", waiting_model: "모델 서버 연결 대기", completed: "측정 완료",
    completed_with_errors: "측정 종료 · 누락 확인 필요", failed: "실행 오류 확인 필요",
    stopped: "실행기 중단"}[phase] || phase);

  function policyShifts(view) {
    return (view.data.policy_shifts || []).filter(shift => !isHidden("shift", shift.id) && (!view.id || shift.journal_id === view.id
      || (shift.source_journal_ids || []).includes(view.id)));
  }

  function duration(seconds) {
    if (seconds == null || !Number.isFinite(seconds)) return "집계 중";
    const minutes = Math.ceil(seconds / 60);
    return minutes >= 60 ? `${number(Math.floor(minutes / 60))}시간 ${minutes % 60}분` : `${minutes}분`;
  }

  function renderPolicyShifts(view) {
    const target = view.root.querySelector(".live-policy-shifts");
    const shifts = policyShifts(view);
    target.closest("[data-live-module]").hidden = !shifts.length || isHidden("module", "policy-shifts");
    target.innerHTML = shifts.map(shift => {
      const counts = shift.counts, speed = shift.throughput || {};
      const elapsed = Math.max(1, Math.round((speed.window_seconds || 0) / 60));
      const estimate = speed.eta_seconds == null ? "속도 집계 후 표시" : `${duration(speed.eta_seconds)} · 최근 속도 기준`;
      return `<section class="live-policy-shift" data-policy-shift="${esc(shift.id)}">
        <header><div><span class="eyebrow">POLICY SHIFT · LIVE</span><h3><span class="live-pulse"></span>${esc(shiftPhase(shift.phase))}</h3></div>
          <div class="live-item-actions">${shift.journal_id ? `<a class="button small" href="#experiment/${encodeURIComponent(shift.journal_id)}">측정 기록 ↗</a>` : ""}
          ${removeButton(view, "shift", shift.id, shift.title || `Policy shift · ${shift.id}`)}</div></header>
        ${bar(shift.progress, "전체 추론 처리", "validation")}
        <div class="live-shift-stats">${countTile("완료", counts.completed, "completed")}${countTile("실행 중", counts.running, "running")}
          ${countTile("대기", counts.pending)}${countTile("오류", counts.error, counts.error ? "error" : "")}
          <div class="live-stat"><span>${speed.window_seconds >= 60 ? `최근 ${elapsed}분` : "최근 관측 구간"} 처리 속도</span>
          <strong>${speed.per_minute == null ? "집계 중" : speed.per_minute.toFixed(1)}${speed.per_minute == null ? "" : "<small>건/분</small>"}</strong></div>
          <div class="live-stat"><span>추론 완료까지 예상</span><strong class="live-shift-eta">${esc(estimate)}</strong></div></div>
        <p class="live-shift-meta">baseline별 자체 CP0 history · temperature ${esc(shift.temperature)}
          · ${shift.per_gpu_query_limit ? `GPU당 동시성 ${number(shift.per_gpu_query_limit)} · ` : ""}전체 동시성 ${number(shift.workers)}
          · ${shift.updated_at ? `진행 보고 ${esc(kst(shift.updated_at))}` : "첫 진행 보고 대기"} · 5초마다 자동 조회</p>
        ${shift.gpu_queries?.active ? `<p class="live-shift-meta">현재 GPU query: ${Object.entries(shift.gpu_queries.active).map(([gpu, active]) =>
          `GPU${esc(gpu)} ${number(active)}/${number(shift.gpu_queries.per_gpu_query_limit)}`).join(" · ")}</p>` : ""}
        <p class="live-shift-meta">진행률은 추론 요청 처리 기준입니다. Policy shift 값은 추론 종료 후 ΔCP별로 집계합니다.</p>
      </section>`;
    }).join("");
  }

  function renderResults(view) {
    const target = view.root.querySelector(".live-result-tables");
    if (!target) return;
    const previous = new Map([...target.querySelectorAll("[data-results-scroll]")].map(el => [el.dataset.resultsScroll, el.scrollLeft]));
    const opened = new Map([...target.querySelectorAll("details[data-result-table]")].map(el => [el.dataset.resultTable, el.open]));
    const tables = (view.data.result_tables || []).filter(table => !isHidden("table", table.id));
    target.innerHTML = tables.map(table => {
      const groups = [...new Set(table.columns.map(column => column.benchmark))];
      let isOpen = opened.get(table.id) ?? true;
      if (!opened.has(table.id)) {
        try {
          const saved = localStorage.getItem(`exgentic.live.table.${view.id || "all"}.${table.id}`);
          if (saved !== null) isOpen = saved === "open";
        } catch (_) { /* Native disclosure remains usable without storage. */ }
      }
      const labels = {no_source: "재실험 대기", ambiguous_sources: "중복 소스 확인 필요", incomplete: "집계 중",
        policy_unverified: "최종 정책 확인 대기", completed: "완료",
        training_val_selection_unverified: "학습 Val 선택 설정 확인 필요",
        checkpoint_missing_or_ambiguous: "CP 누락 또는 중복 소스 확인 필요",
        checkpoint_validation_incomplete: "CP Val 평가 완료 대기", selected_test_incomplete: "선택된 CP의 Test 누락"};
      const showStars = table.show_star_markers !== false;
      const earlyLabel = table.offline_splits ? "Test (early)" : "Offline (early)";
      const splitLegend = table.test_only
        ? '<strong>Test</strong> 학습 종료 후 최종 고정 정책으로 평가한 전체 Test split 점수입니다. 각 seed의 Test가 끝나면 자동 반영합니다.'
        : table.offline_splits
        ? '<strong>Offline 평가</strong> · <strong>Train / Val / Test</strong>는 동일한 최종 정책을 checkpoint 평가 큐에서 각 split에 추론한 점수입니다. 학습 중 기록한 점수와 별도로 실행한 과거 Test는 사용하지 않습니다.'
        : '<strong>Online</strong> 학습 과정의 기록 · <strong>Offline</strong> 최종 정책 Test · <strong>Offline-val</strong> 같은 최종 CP의 Val 평가.';
      const partialLegend = !table.partial_seeds ? '새 실험의 각 셀은 세 seed의 해당 측정이 모두 완료되면 자동으로 채워집니다.'
        : (showStars ? '<strong>*</strong> 세 seed 중 일부 완료: 완료된 seed만 평균에 반영하며 세 seed가 완료되면 별표가 사라집니다.'
          : '완료된 seed의 전체 split 결과만 평균에 반영합니다. 누락되거나 오류가 있는 seed는 제외하며 0점으로 채우지 않습니다. seed별 완료 여부는 셀에 마우스를 올려 확인할 수 있습니다.')
          + ' 한 seed만 있으면 표준편차를 표시하지 않습니다.';
      const partialCoverage = showStars ? [] : table.rows.flatMap(row => table.columns
        .filter(column => !column.id.startsWith("avg.") && row.cells[column.id].provisional)
        .map(column => `${row.label} ${row.setting_label || row.setting} · ${column.benchmark} ${column.metric}: n=${row.cells[column.id].ready}/${row.cells[column.id].expected}`));
      const cells = row => table.columns.map(column => {
        const cell = row.cells[column.id];
        const title = cell.historical ? `ReAct Offline 기존 보고값 유지 · ${cell.source}`
          : cell.average ? `같은 seed의 벤치마크 완료 결과로 macro-average · ${cell.ready}/${cell.expected} seeds`
          : `${cell.ready}/${cell.expected} seeds · ` + cell.samples.map(sample => `seed ${sample.seed}: ${labels[sample.reason] || sample.reason} (${sample.count}/${sample.total} task)`
            + (sample.checkpoint === undefined ? "" : ` · CP${sample.checkpoint}`)
            + (sample.val_score == null ? "" : ` · Val${sample.test_result_reused ? "(학습 선택)" : ""} ${(100 * sample.val_score).toFixed(2)}%`)
            + (sample.test_result_reused ? " · 학습 Val-best 최종 Test 재사용" : "")).join(" · ");
        const content = cell.mean === null ? `<span class="result-empty">--</span>${cell.ready ? `<small class="result-pending">${cell.ready}/${cell.expected} seeds</small>` : ""}`
          : `<span class="result-value">${cell.mean.toFixed(2)}${cell.average || cell.std == null ? "" : ` <small>(${cell.std.toFixed(2)})</small>`}</span>${cell.provisional && showStars ? '<sup class="result-provisional" aria-label="세 seed 중 일부 완료">*</sup>' : ""}`;
        return `<td class="rank-${cell.rank}" data-shade="${esc(cell.shade)}" title="${esc(title)}" data-result-column="${esc(column.id)}">${content}</td>`;
      }).join("");
      const rows = table.rows.map(row => `<tr class="${row.method === "premise" ? "ours " : ""}${row.setting === "isolated" ? "method-start" : ""}" data-result-row="${esc(row.id)}" data-result-method="${row.method}" data-result-setting="${row.setting}">
        ${table.row_layout === "variants" ? `<th scope="row">${esc(row.label)}</th>` : row.setting === "isolated" ? `<th rowspan="${row.method === "react" ? 1 : 3}" scope="rowgroup">${esc(row.label)}</th>` : ""}
        <th scope="row">${esc(row.setting_label || row.setting)}</th>${cells(row)}</tr>`).join("");
      return `<details class="live-results" data-result-table="${esc(table.id)}" ${isOpen ? "open" : ""}><summary><div><span class="eyebrow">AUTOMATIC RESULTS TABLE</span><h3><span class="live-table-arrow" aria-hidden="true">›</span>${esc(table.title)}</h3>
        <p>${esc(table.model)} · seeds ${table.seeds.join(", ")} · 평균 (표본 표준편차), 단위 %${table.inference_profile_label ? ` · ${esc(table.inference_profile_label)}` : ""}</p></div>
        <div class="live-item-actions"><a class="button small" href="/api/live-results.tex?id=${encodeURIComponent(table.id)}">LaTeX 다운로드</a>
        ${removeButton(view, "table", table.id, table.title)}</div></summary>
        <div class="live-results-scroll" data-results-scroll="${esc(table.id)}"><table class="live-result-table${groups.length === 1 ? " single-benchmark" : ""}${table.test_only ? " test-only" : ""}">
        <thead><tr><th rowspan="2">Method</th><th rowspan="2">Setting</th>${groups.map(name => `<th colspan="${table.columns.filter(column => column.benchmark === name).length}" scope="colgroup">${esc(name)}</th>`).join("")}</tr>
        <tr>${table.columns.map(column => `<th scope="col">${column.metric}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
        <p class="live-results-legend">${splitLegend}
        ${table.offline_early ? `<strong>${earlyLabel}</strong> ${table.offline_early.reuse_training_val_best_variants?.length ? 'GEPA Val·SkillOpt Val은 학습 중 Val-best로 선택한 최종 정책의 Test를 그대로 재사용합니다. 나머지 세팅은 ' : ''}CP0를 포함한 평가 CP의 재평가 Val 최고점으로 선택합니다. 동점은 가장 이른 CP입니다. Test는 선택에 사용하지 않으며, 선택 CP의 Test가 누락되면 해당 seed를 제외합니다. 셀에 마우스를 올리면 선택 CP와 Val 점수가 보입니다.` : ''}
        AppWorld는 모든 열에서 <strong>TGC</strong>(완전 성공 task / 공식 결과가 기록된 task)로 집계합니다.
        ${partialLegend} <strong>굵게</strong> 최고, <u>밑줄</u> 두 번째로 높은 값, 동점 동일 표시.
        <span class="result-color-key"><i class="result-low"></i>낮음 <i class="result-high"></i>높음</span></p>
        ${partialCoverage.length ? `<p class="live-results-coverage">일부 seed 집계: ${esc(partialCoverage.join("; "))}.</p>` : ""}
        <p class="live-results-source">재수집 데이터 사용${table.environment_policy_label ? ` · 환경 policy: ${esc(table.environment_policy_label)}` : ""}${table.reported_react_offline ? " · ReAct Offline은 기존 보고값 유지" : ""} · PEACE = PREMiSE · ${esc(table.playbook_edit_mode)} 기준 · ${table.completed_cells}개 셀 완료${table.partial_seeds ? ` · ${table.provisional_cells}개 셀 일부 seed 집계${showStars ? "(*)" : ""}` : ""}${table.include_average !== false ? " · Avg.는 같은 seed의 모든 벤치마크 완료 후 표시" : ""}</p></details>`;
    }).join("");
    const module = target.closest("[data-live-module]");
    if (module) module.hidden = !tables.length || isHidden("module", "results");
    target.querySelectorAll("details[data-result-table]").forEach(details => details.addEventListener("toggle", () => {
      try {
        localStorage.setItem(`exgentic.live.table.${view.id || "all"}.${details.dataset.resultTable}`, details.open ? "open" : "closed");
      } catch (_) { /* Optional persistence. */ }
    }));
    target.querySelectorAll("[data-results-scroll]").forEach(el => { el.scrollLeft = previous.get(el.dataset.resultsScroll) || 0; });
  }

  function renderCheckpoints(view) {
    const target = view.root.querySelector(".live-checkpoint-queues");
    if (!target) return;
    const queues = checkpointQueues(view);
    const module = target.closest("[data-live-module]");
    if (module) module.hidden = !queues.length || isHidden("module", "checkpoints");
    const opened = new Map([...target.querySelectorAll("details[data-cp-stream]")].map(el => [el.dataset.cpStream, el.open]));
    const charts = new Map();
    view.cpSeries ||= new Map();
    target.innerHTML = queues.map(queue => {
      const counts = queue.task_counts, rows = queue.rows;
      const followup = queue.policy_shift_followup;
      const queuePhase = {stopped: "평가 실행기 중단", watching: "새 CP 자동 등록 대기",
        paused: "새 평가 제출 일시중지", waiting_model: "모델 서버 연결 대기", running: "평가 진행 중",
        waiting_prerequisites: "PEACE 학습·최종 test 완료 후 재개 예약", waiting_recovery: "재개 조건 확인 필요",
        waiting_cleanup: "Terminal-Bench 네트워크 정리 후 CP 평가 재개 대기",
        completed: "평가 완료", error: "평가 오류 확인 필요"}[queue.phase] || queue.phase;
      const canonical = rows.filter(row => !row.alias && row.selected !== false);
      const prerequisites = queue.resume_schedule?.contexts || [];
      const reservation = prerequisites.length && queue.resume_schedule.phase !== "started"
        ? ` · PEACE 학습 ${prerequisites.filter(row => row.training_complete).length}/6 · 최종 test ${prerequisites.filter(row => row.test_complete).length}/6 완료` : "";
      const splitTotals = split => canonical.reduce((sum, row) => {
        const value = row.splits[split] || {};
        return {done: sum.done + (value.done || 0), total: sum.total + (value.total || 0)};
      }, {done: 0, total: 0});
      const filtered = rows.filter(row => matches(row) && (!view.cpPoint || String(row.point) === view.cpPoint));
      const groups = new Map();
      for (const row of filtered) {
        if (!groups.has(row.stream)) groups.set(row.stream, []);
        groups.get(row.stream).push(row);
      }
      const cells = row => ["train", "val", "test"].map(split => {
        const value = row.splits[split] || {done: 0, total: 0};
        return `<td>${bar(value, split === "val" ? "Offline-val" : split, split === "train" ? "train" : split === "val" ? "validation" : "test")}
          <strong class="live-cp-score">${value.score === null || value.score === undefined ? "점수 대기" : `${(100 * value.score).toFixed(1)}%`}</strong>
          <small>${value.done === value.total && !value.errors ? "전체 완료" : "완료분 평균"}${value.errors ? ` · 실행 오류 ${value.errors}개 제외` : ""}</small></td>`;
      }).join("");
      const tables = [...groups.entries()].map(([id, items]) => {
        items.sort((a, b) => a.point - b.point);
        const first = items[0], isActive = items.some(row => row.state === "running");
        const chartId = `${queue.id}/${id}`;
        charts.set(chartId, items);
        return `<details class="live-cp-stream" data-cp-stream="${esc(chartId)}" ${(opened.has(chartId) ? opened.get(chartId) : isActive) ? "open" : ""}>
          <summary><strong>${esc(first.run_variant?.startsWith("peace") ? "PEACE" : names[first.method] || first.method)} · ${esc(benchmarks[first.benchmark] || first.benchmark)} · seed ${esc(first.seed)}</strong>
          <span>${items.length} CP · 실행 ${items.filter(row => row.state === "running").length} · 동일 ${items.filter(row => row.alias).length}</span></summary>
          <p class="live-cp-source">${esc(first.run_variant || first.variant)} · ${esc(first.variant)}</p><div class="live-cp-chart-mount"></div>
          <div class="live-cp-table-scroll"><table class="live-cp-table">
          <thead><tr><th>CP / 상태</th><th>Train</th><th>Offline-val</th><th>Test</th></tr></thead><tbody>${items.map(row => {
            const state = row.alias ? `CP${String(row.alias).split("_cp").pop()}와 동일 · 평가 생략` : phases[row.state] || row.state;
            return `<tr data-cp-id="${esc(row.id)}"><td><strong>CP ${String(row.point).padStart(3, "0")}</strong><span class="live-cp-status">${esc(state)}</span>
              <small>${esc(row.policy_hash)}${row.selected_val_score === null || row.selected_val_score === undefined ? "" : `<br>선택 점수 ${(100 * row.selected_val_score).toFixed(1)}%`}</small></td>${cells(row)}</tr>`;
          }).join("")}</tbody></table></div></details>`;
      }).join("");
      return `<section class="live-cp-queue" data-checkpoint-queue="${esc(queue.id)}"><header><div><span class="eyebrow">CHECKPOINT INFERENCE</span><h3>${queue.checkpoint_interval || 5} task CP 평가 · 공용 큐</h3>
        <p>${esc(queuePhase)} · 실행 <b>${number(counts.running)} / ${queue.workers}</b> · 대기 ${number(counts.pending)} · 완료 ${number(counts.completed)} · 오류 ${number(counts.error)}</p>
        ${queue.gpu_queries?.active ? `<p>동시 query: ${Object.entries(queue.gpu_queries.active).map(([gpu, active]) => `${/^\d{4,5}$/.test(gpu) ? "포트 " : "GPU"}${esc(gpu)} ${number(active)}/${number(queue.gpu_queries.per_endpoint_query_limits?.[gpu] ?? queue.gpu_queries.per_gpu_query_limit)}`).join(" · ")}${queue.gpu_queries.waiting ? ` · 공통 큐 대기 ${number(queue.gpu_queries.waiting)}` : ""}</p>` : ""}
        ${queue.scope_label || reservation ? `<p>${esc(queue.scope_label || "")}${esc(reservation)}</p>` : ""}
        <p>등록 ${queue.checkpoints} CP · 동일 내용 ${queue.skipped} CP 평가 생략${queue.invalidated_checkpoints ? ` · 이전 보상 오류로 제외 ${queue.invalidated_checkpoints} CP` : ""}${queue.source_issues ? ` · 복원 확인 필요 ${queue.source_issues}건` : ""} · ${esc(kst(queue.updated_at))} 갱신</p></div>
        <div class="live-item-actions">${view.preview ? "" : `<label>CP<select data-cp-point><option value="">전체</option>${[...new Set(rows.map(row => row.point))].sort((a, b) => a - b).map(point => `<option value="${point}" ${view.cpPoint === String(point) ? "selected" : ""}>CP ${point}</option>`).join("")}</select></label>`}
        ${removeButton(view, "queue", queue.id, queue.title || `CP 평가 · ${queue.id}`)}</div></header>
        <div class="live-cp-totals">${bar(splitTotals("train"), "CP Train")}${bar(splitTotals("val"), "CP Val", "validation")}${bar(splitTotals("test"), "CP Test", "test")}</div>
        ${followup ? `<p class="live-cp-source">후속 Policy shift: ${esc(shiftPhase(followup.phase))} · 실시간 진행률은 상단 Policy shift 패널에서 확인할 수 있습니다.</p>` : ""}
        ${view.preview ? "" : tables || '<p class="live-cp-source">선택한 필터에 해당하는 CP가 없습니다.</p>'}
        <p class="live-explainer">학습 task_id ${number(queue.checkpoint_interval || 5)}개마다 정책을 고정합니다. 학습 중 CP를 자동 추가하며, 이미 진행된 시점도 복원합니다. GEPA는 해당 시점까지의 Val 최고 후보를 선택합니다. 이 평가 결과는 학습에 전달하지 않습니다.</p></section>`;
    }).join("");
    target.querySelectorAll("details[data-cp-stream]").forEach(details => {
      const draw = () => {
        if (!details.open || details.closest("[data-live-module]")?.open === false) return;
        const container = details.querySelector(".live-cp-chart-mount");
        if (container.childElementCount) return;
        const id = details.dataset.cpStream;
        if (!view.cpSeries.has(id)) view.cpSeries.set(id, new Set(["train", "val", "test"]));
        window.CheckpointCharts.mount(container, charts.get(id), {
          label: details.querySelector("summary strong").textContent,
          visible: view.cpSeries.get(id)
        });
      };
      details.addEventListener("toggle", draw);
      draw();
    });
    target.querySelectorAll("[data-cp-point]").forEach(select => select.addEventListener("change", () => {
      view.cpPoint = select.value; renderCheckpoints(view);
    }));
  }

  function renderStandaloneEvaluations(view, queues) {
    const phases = {evaluating: "평가 중", completed: "완료", infrastructure_errors: "측정 종료 · 누락 있음",
      planned: "예약", waiting_predecessor: "앞선 평가 종료 대기", waiting_terminalbench: "Terminal-Bench 종료 대기",
      stopped: "실행기 확인 필요"};
    const target = view.root.querySelector(".live-standalone-evaluations");
    target.hidden = !queues.length;
    const priority = phase => phase === "evaluating" ? 0 : ["planned", "waiting_predecessor", "waiting_terminalbench"].includes(phase) ? 1 : 2;
    target.innerHTML = [...queues].sort((a, b) => priority(a.phase) - priority(b.phase) || a.order - b.order).map(queue => `<section class="live-eval-card">
      <header><div><span class="eyebrow">EVALUATION QUEUE · ${number(queue.order)}</span>
        <h3>${esc(queue.label)}</h3><p>${esc(phases[queue.phase] || queue.phase)} · 동시성 ${number(queue.workers)}
        ${queue.after_label ? ` · ${esc(queue.after_label)} 이후` : ""}</p></div></header>
      <p>${esc(queue.model)} · ${esc(queue.protocol)}</p>
      ${queue.active_tasks != null ? `<p class="live-eval-note"><strong>현재 실행 ${number(queue.active_tasks.length)} / 동시성 ${number(queue.workers)}</strong>
        ${queue.active_tasks.length ? ` · ${queue.active_tasks.map(esc).join(", ")}` : ""}</p>` : ""}
      ${queue.rows.map(row => {const all = row.splits.all; return `<div class="live-eval-stats">
        ${countTile(`${benchmarks[row.benchmark] || row.benchmark} 측정 · 예정 ${number(all.total)}`, all.measured)}
        ${countTile("성공", all.successes, "completed")}${countTile("실패 · 예산 종료 포함", all.failed)}
        ${countTile("인프라 오류 · 미측정", all.errors, all.errors ? "error" : "")}
        ${countTile("진행·대기", all.remaining)}${countTile("실패한 시도 · 재시도 포함", row.failed_attempts)}
        </div>${bar({done: all.processed, total: all.total}, `${benchmarks[row.benchmark]} 처리 완료`, "test")}
        <div class="table-wrap"><table><thead><tr><th>Split</th><th>측정 / 예정</th><th>성공</th><th>실패</th><th>인프라 오류</th><th>측정분 성공률</th></tr></thead>
        <tbody>${["test", "val", "train", "all"].map(split => {const s = row.splits[split]; return `<tr>
          <td>${esc(split)}</td><td>${number(s.measured)} / ${number(s.total)}</td><td>${number(s.successes)}</td>
          <td>${number(s.failed)}</td><td>${number(s.errors)}</td><td>${s.score_percent == null ? "—" : s.score_percent.toFixed(1) + "%"}${s.complete ? "" : " · 진행값"}</td></tr>`;}).join("")}</tbody></table></div>`;}).join("")}
      <p class="live-eval-note">인프라 오류는 성공률 분모에서 제외합니다. 전체 예정 수가 측정될 때 최종 점수입니다. 5초 자동 갱신
        ${queue.updated_at ? ` · ${esc(kst(queue.updated_at))}` : ""}
        ${queue.journal_ids.map(id => `<a href="#experiment/${esc(id)}">실험 기록 ↗</a>`).join(" · ")}</p>
      </section>`).join("");
  }

  function renderACEStudies(view, studies) {
    const target = view.root.querySelector(".live-ace-studies");
    target.hidden = !studies.length;
    const labels = {running: "실행 중", completed: "완료", error: "오류 확인", stopped: "중단", planned: "예약"};
    const modes = {thinking: "Thinking", nonthinking: "Non-thinking"};
    const score = value => typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) + "%" : "—";
    const paired = (value, sd) => typeof value === "number" && typeof sd === "number"
      ? `${(value * 100).toFixed(2)} ± ${(sd * 100).toFixed(2)}%` : "—";
    target.innerHTML = studies.map(study => `<section class="live-eval-card">
      <header><div><span class="eyebrow">ACE APPWORLD · TRAIN ONLY</span><h3>${esc(study.label)}</h3>
      <p>${esc(labels[study.state] || study.state)} · 동시성 ${number(study.workers)} · 오류 ${number(study.errors)}</p></div>
      ${removeButton(view, "study", study.id, study.label)}</header>
      <p>${esc(study.model)} · w/o GT · ADD-only · seeds 42, 43, 44</p>
      <div class="live-batch-progress">${study.stages.map(stage => bar(stage,
        `${stage.label} (${stage.unit})${stage.current ? ` · 현재 실행 ${number(stage.running)}` : ""}`,
        stage.current ? "train" : "test")).join("")}</div>
      ${Object.keys(study.by_mode).length && study.state === "running" ? `<p>${Object.entries(study.by_mode)
        .map(([mode, row]) => `${esc(modes[mode] || mode)} ${number(row.completed)}/${number(row.total)}`).join(" · ")}</p>` : ""}
      ${study.initial_scores.length ? `<div class="table-wrap"><table><thead><tr><th>학습 전</th><th>PCT</th><th>TGC</th><th>pass@5</th><th>pass⁵</th></tr></thead>
        <tbody>${study.initial_scores.map(row => `<tr><td>${esc(modes[row.mode] || row.mode)}</td>
          <td>${score(row.pct_percent)}</td><td>${score(row.tgc_percent)}</td><td>${score(row.pass_at_5_percent)}</td>
          <td>${score(row.pass_power_5_percent)}</td></tr>`).join("")}</tbody></table></div>` : ""}
      ${study.final_scores.length ? `<div class="table-wrap"><table><thead><tr><th>최종 Train · 모드</th><th>선정 전략</th><th>PCT</th><th>TGC</th></tr></thead>
        <tbody>${study.final_scores.map(row => `<tr><td>${esc(modes[row.mode] || row.mode)}</td><td>${esc(row.strategy)}</td>
          <td>${paired(row.pct_mean, row.pct_std)}</td><td>${paired(row.tgc_mean, row.tgc_std)}</td></tr>`).join("")}</tbody></table></div>
        <p class="live-eval-note">평균 ± 표준편차는 세 seed의 Train50 점수 기준입니다.</p>` : ""}
      <p class="live-eval-note">BWT·Held-out 평가 제외 · 5초 자동 갱신 ${study.updated_at ? ` · ${esc(kst(study.updated_at))}` : ""}
        ${study.journal_id ? ` · <a href="#experiment/${esc(study.journal_id)}">실험 기록 ↗</a>` : ""}</p>
    </section>`).join("");
  }

  function render(view) {
    if (!view.data || !view.root.isConnected) return;
    const batches = visibleBatches(view);
    updateVisibilityControls(view);
    const rows = batches.flatMap(b => b.runs);
    const shifts = policyShifts(view);
    const activeShifts = shifts.filter(shift => ["probing", "preparing", "preparing_histories", "waiting_model"].includes(shift.phase));
    const activeCheckpoints = checkpointQueues(view).filter(queue => queue.phase === "running");
    const evaluations = (view.data.evaluations || []).filter(queue => !view.id || queue.journal_ids.includes(view.id)
      || (queue.source_journal_ids || []).includes(view.id));
    const studies = (view.data.ace_studies || []).filter(study => (!view.id || study.journal_id === view.id)
      && !isHidden("study", study.id));
    const nav = document.getElementById("live-nav-count");
    if (nav) nav.textContent = visibleBatches({...view, id: null}).flatMap(batch => batch.runs).filter(row => row.state === "running").length
      + (view.data.policy_shifts || []).filter(shift => !isHidden("shift", shift.id)
        && ["probing", "preparing", "preparing_histories", "waiting_model"].includes(shift.phase)).length
      + checkpointQueues({...view, id: null}).filter(queue => queue.phase === "running").length
      + (view.data.evaluations || []).filter(queue => queue.phase === "evaluating").length
      + (view.data.ace_studies || []).filter(study => study.state === "running" && !isHidden("study", study.id)).length;
    if (view.compact && !rows.length && !checkpointQueues(view).length && !shifts.length && !evaluations.length && !studies.length) { view.root.hidden = true; return; }
    view.root.hidden = false;
    const counts = state => rows.filter(r => r.state === state).length;
    const train = totals(rows, "train"), test = totals(rows, "test");
    const running = rows.filter(r => r.state === "running");
    view.root.querySelector(".live-stats").innerHTML = [countTile("전체 실험", rows.length),
      countTile("학습·Test 실행", running.length, "running"), countTile("대기", counts("pending")),
      countTile("일시중지", counts("paused"), "paused"),
      countTile("완료", counts("completed"), "completed"), countTile("확인 필요", counts("error") + counts("stopped"), "error"),
      countTile("Test 실행 오류", test.errors, test.errors ? "error" : "")].join("");
    if (shifts.length) view.root.querySelector(".live-stats").innerHTML += countTile("Policy shift 실행", activeShifts.length, "running");
    if (studies.length) view.root.querySelector(".live-stats").innerHTML += countTile("ACE 분석실험 실행", studies.filter(study => study.state === "running").length, "running");
    if (running.some(row => row.gpu != null)) {
      view.root.querySelector(".live-stats").innerHTML += [0, 1].map(gpu =>
        countTile(`GPU ${gpu} 실행`, running.filter(row => Number(row.gpu) === gpu && row.gpu != null).length, "running")).join("");
    }
    const stage = running.reduce((sum, r) => { sum[r.phase] = (sum[r.phase] || 0) + 1; return sum; }, {});
    view.root.querySelector(".live-totals").innerHTML = bar(train, "전체 학습 태스크 · 예정 포함")
      + bar(test, "전체 Test 처리 · 예정 포함", "test");
    view.root.querySelector(".live-phase-summary").textContent = [...Object.entries(stage)
      .map(([name, n]) => `${phases[name] || name} ${n}`), ...activeCheckpoints.map(queue =>
        `CP 평가 ${number(queue.task_counts.running)}개 실행`), ...activeShifts.map(shift =>
        `Policy shift ${shiftPhase(shift.phase)} · ${number(shift.progress.done)}/${number(shift.progress.total)} 처리`),
      ...evaluations.filter(queue => queue.phase === "evaluating").map(queue => `${queue.label} 평가 중`),
      ...studies.filter(study => study.state === "running").map(study => `${study.label} · ${study.stages.find(stage => stage.current)?.label || "진행 중"}`)].join(" · ")
        || (counts("paused") ? `${counts("paused")}개 실험 일시중지 · 완료 결과 보존` : "현재 실행 중인 실험이 없습니다.");
    renderPolicyShifts(view);
    renderEvaluationOverview(view);
    renderResults(view);
    renderCheckpoints(view);
    renderStandaloneEvaluations(view, evaluations);
    renderACEStudies(view, studies);
    if (view.preview) return;
    const variants = new Map(rows.filter(row => row.variant).map(row => [row.variant, row.variant_label || row.variant]));
    const variantFilter = view.root.querySelector('[data-live-filter="variant"]');
    if (filters.variant !== "all" && !variants.has(filters.variant)) filters.variant = "all";
    variantFilter.innerHTML = '<option value="all">전체 설정</option>' + [...variants].map(([key, label]) =>
      `<option value="${esc(key)}">${esc(label)}</option>`).join("");
    variantFilter.value = filters.variant;
    view.root.querySelector(".live-methods").innerHTML = Object.entries(names).filter(([method]) => rows.some(r => r.method === method))
      .map(([method, name]) => {
        const group = rows.filter(r => r.method === method);
        return `<button class="live-method ${filters.method === method ? "selected" : ""}" data-live-method="${method}" aria-pressed="${filters.method === method}">
          <strong>${name}</strong><span>실행 ${group.filter(r => r.state === "running").length} · 중지 ${group.filter(r => r.state === "paused").length} · 대기 ${group.filter(r => r.state === "pending").length} · 완료 ${group.filter(r => r.state === "completed").length}</span></button>`;
      }).join("");
    const groups = view.root.querySelector(".live-groups");
    const open = new Map([...groups.querySelectorAll("details[data-batch]")].map(el => [el.dataset.batch, el.open]));
    const evidenceNodes = new Map([...groups.querySelectorAll("details.live-evidence")].map(el => [el.dataset.retrievalKey, el]));
    let visible = 0;
    groups.innerHTML = batches.map(batch => {
      const selected = batch.runs.filter(matches);
      visible += selected.length;
      if (!selected.length) return "";
      const batchRunning = batch.runs.filter(r => r.state === "running").length;
      const batchDone = batch.runs.filter(r => r.state === "completed").length;
      const batchPaused = batch.runs.filter(r => r.state === "paused").length;
      const schedule = batch.schedule || {};
      const reservation = schedule.phase === "paused" && schedule.resume_at
        ? `일시중지 · ${kst(schedule.resume_at)} 자동 재개 예약`
        : schedule.phase === "scheduled" && schedule.pause_at && schedule.resume_at
          ? `${kst(schedule.pause_at)} 중지 · ${kst(schedule.resume_at)} 재개 예약` : "";
      const opened = open.has(batch.id) ? open.get(batch.id) : batchRunning > 0 || batchPaused > 0 || batches.length === 1;
      return `<details class="live-batch" data-batch="${esc(batch.id)}" ${opened ? "open" : ""}>
        <summary><div><strong>${esc(batch.title)}</strong><span>${esc(batch.model)}${batch.max_parallel ? ` · 실험 동시성 ${number(batch.max_parallel)}` : ""}${batch.waiting_for ? " · 선행 실험 완료 후 시작" : ""}</span></div>
        <div class="live-batch-count"><b>${batchRunning} 실행${batchPaused ? ` · ${batchPaused} 일시중지` : ""}</b><span>${batchDone} / ${batch.runs.length} 완료</span></div>
        ${removeButton(view, "batch", batch.id, batch.title)}</summary>
        ${reservation ? `<p class="live-reservation">${esc(reservation)}</p>` : ""}
        <div class="live-batch-top"><span>${selected.length}개 표시${batch.waiting_for ? ` · 대기 대상: ${esc(batch.waiting_for)}` : ""}</span>
          ${batch.journal_id ? `<a href="#experiment/${esc(batch.journal_id)}">실험 기록 열기 ↗</a>` : ""}</div>
        ${batch.max_parallel ? `<div class="live-batch-progress">${bar(totals(batch.runs, "train"), "이 실험 큐 · 학습 태스크")}${bar(totals(batch.runs, "test"), "이 실험 큐 · Test 처리", "test")}</div>` : ""}
        <div class="live-run-grid">${selected.map(row => card(row, batch.id, view)).join("")}</div></details>`;
    }).join("") || '<div class="empty"><h2>표시할 실험이 없습니다</h2><p>필터를 바꾸거나 상단의 숨긴 항목에서 복원해 주세요.</p></div>';
    view.root.querySelector(".live-visible").textContent = `${visible} / ${rows.length}개 표시`;
    groups.querySelectorAll("details.live-evidence").forEach(details => {
      const previous = evidenceNodes.get(details.dataset.retrievalKey);
      if (previous) details.replaceWith(previous);
    });
    bindRetrieval(view, groups);
  }

  async function refresh(view) {
    clearTimeout(view.timer);
    if (active !== view || !view.root.isConnected) return;
    if (document.hidden || view.paused) { schedule(view); return; }
    if (view.loading) return;
    view.loading = true;
    const button = view.root.querySelector("[data-live-refresh]");
    if (button) button.disabled = true;
    view.controller = new AbortController();
    const timeout = setTimeout(() => view.controller.abort(), 25000);
    try {
      const response = await fetch("/api/live-progress", {cache: "no-store", signal: view.controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      view.data = await response.json();
      if (active !== view) return;
      render(view);
      const clock = new Date(view.data.generated_at).toLocaleTimeString("ko-KR", {hour12: false, timeZone: "Asia/Seoul"}) + " KST";
      view.root.querySelector(".live-connection").textContent = view.data.initializing ? "● 실험 결과 첫 집계 중 · 자동 갱신"
        : view.paused ? `갱신 일시정지 · ${clock}` : `● 실시간 · ${clock} 갱신 · 5초마다 조회${view.data.refreshing ? " · 다음 집계 중" : ""}`;
      view.root.querySelector(".live-connection").classList.remove("offline");
      const notice = view.root.querySelector(".live-notice");
      notice.hidden = !view.data.read_issues;
      notice.textContent = "일부 진행 파일을 갱신 중입니다. 마지막 정상 데이터를 표시합니다.";
    } catch (error) {
      if (active !== view) return;
      view.root.querySelector(".live-connection").textContent = "● 연결 지연 · 마지막 갱신 값 유지";
      view.root.querySelector(".live-connection").classList.add("offline");
      const notice = view.root.querySelector(".live-notice");
      notice.hidden = false;
      notice.textContent = `진행 현황을 가져오지 못했습니다 (${error.name === "AbortError" ? "응답 시간 초과" : error.message}). 자동으로 재연결합니다.`;
    } finally {
      clearTimeout(timeout);
      view.loading = false;
      if (button) button.disabled = false;
      schedule(view);
    }
  }

  function schedule(view) {
    if (active === view && view.root.isConnected) view.timer = setTimeout(() => refresh(view), 5000);
  }

  function unmount() {
    if (active) { clearTimeout(active.timer); active.controller?.abort(); active = null; }
  }

  function mount(root, {id = null, compact = false, preview = false} = {}) {
    unmount();
    if (!root) return;
    const view = {root, id, compact, preview, data: null, paused: false, loading: false,
      retrievalOpen: new Set(), retrievalRequests: new Map()};
    active = view;
    const module = (name, title, content) => `<details class="live-module" data-live-module="${name}" open>
      <summary><strong>${title}</strong><span class="live-module-state"><span class="when-open">접기</span><span class="when-closed">펼치기</span></span>
      ${removeButton(view, "module", name, `${title} 영역 전체`)}</summary>${content}</details>`;
    const resultsModule = module("results", "메인 테이블", '<div class="live-result-tables"></div>');
    const checkpointsModule = module("checkpoints", "CP 평가", '<div class="live-checkpoint-queues"></div>');
    const shiftsModule = module("policy-shifts", "Policy shift · 실시간 진행률", '<div class="live-policy-shifts"></div>');
    root.className = `live-dashboard ${compact ? "compact" : ""}`;
    root.innerHTML = `<div class="live-heading"><div><span class="eyebrow">LIVE EXPERIMENTS</span><h2>${preview ? "실시간 실행 현황" : "학습·평가·Policy shift, 한눈에"}</h2>
      <p class="live-phase-summary">진행 데이터를 불러오는 중…</p></div><div class="live-controls"><span class="live-connection" role="status">연결 중…</span>
      ${preview ? '<a class="button primary" href="#live">전체 모니터 열기 ↗</a>' : '<button class="button small" data-live-refresh>새로고침</button><button class="button small subtle" data-live-pause>자동 갱신 중</button><button class="button small" data-live-hidden-manage>숨긴 항목</button>'}</div></div>
      ${preview ? "" : `<p class="live-display-help">각 항목의 ‘삭제’로 모니터를 정리하세요. 표시 설정은 이 브라우저에 저장되며 숨긴 항목에서 복원할 수 있습니다. 실험 실행과 원본 기록은 유지됩니다.</p>
        <div class="live-display-notice" data-live-display-notice role="status" hidden><span></span><button type="button" class="button small" data-live-undo>실행 취소</button></div>
        <dialog class="medium-dialog live-hidden-dialog" data-live-hidden-dialog aria-labelledby="live-hidden-title">
          <header class="dialog-header"><h2 id="live-hidden-title">숨긴 항목</h2><button type="button" class="icon-button" data-live-hidden-close aria-label="닫기">×</button></header>
          <div class="dialog-body"><p class="hint">복원하면 모니터에 다시 표시됩니다. 묶음이나 영역을 숨겼다면 해당 묶음·영역도 복원해 주세요.</p>
          <ul class="live-hidden-items" data-live-hidden-items></ul></div>
          <footer class="dialog-footer"><button type="button" class="button" data-live-restore-all>전체 복원</button><button type="button" class="button primary" data-live-hidden-close>닫기</button></footer>
        </dialog>`}
      <div class="live-notice" role="status" hidden></div><div class="live-ace-studies"></div><div class="live-standalone-evaluations"></div><div class="live-evaluation-overview" hidden></div>${shiftsModule}<div class="live-stats"></div><div class="live-totals"></div>
      ${preview ? "" : resultsModule}
      ${preview ? checkpointsModule : ""}
      ${preview ? "" : `<div class="live-methods"></div><div class="live-filters">
        <label>방법<select data-live-filter="method"><option value="all">전체 방법</option>${Object.entries(names).map(([key, name]) => `<option value="${key}">${name}</option>`).join("")}</select></label>
        <label>설정<select data-live-filter="variant"><option value="all">전체 설정</option></select></label>
        <label>벤치마크<select data-live-filter="benchmark"><option value="all">전체 벤치마크</option>${Object.entries(benchmarks).map(([key, name]) => `<option value="${key}">${name}</option>`).join("")}</select></label>
        <label>상태<select data-live-filter="state"><option value="all">전체 상태</option><option value="running">실행 중</option><option value="paused">일시중지</option><option value="pending">대기</option><option value="completed">완료</option><option value="skipped">동일 CP 생략</option><option value="attention">확인 필요</option></select></label>
        <label>Seed<select data-live-filter="seed"><option value="all">전체 Seed</option><option>42</option><option>43</option><option>44</option></select></label>
        <span class="live-visible"></span></div>${checkpointsModule}<div class="live-groups"><div class="loading">진행 데이터를 불러오는 중입니다.</div></div>
        <p class="live-explainer">학습은 각 task의 진행 수입니다. GEPA·SkillOpt는 최초 노출 수를 세며, 마지막 후보 검증·업데이트가 끝날 때까지 실행 중으로 표시합니다. Test는 결과가 기록된 task 수이며 성공률과 별개입니다. 공식 실패 점수 0도 처리 완료에 포함하고, 실행 오류는 따로 표시합니다.</p>`}`;
    root.querySelectorAll("[data-live-module]").forEach(details => {
      const key = `exgentic.live.module.${id || "all"}.${details.dataset.liveModule}`;
      try {
        const saved = localStorage.getItem(key);
        if (saved !== null) details.open = saved === "open";
      } catch (_) { /* Keep native disclosure behavior when storage is unavailable. */ }
      details.addEventListener("toggle", () => {
        try { localStorage.setItem(key, details.open ? "open" : "closed"); } catch (_) { /* Optional persistence. */ }
        if (details.open && details.dataset.liveModule === "checkpoints" && view.data) renderCheckpoints(view);
      });
    });
    root.querySelectorAll("[data-live-filter]").forEach(el => {
      el.value = filters[el.dataset.liveFilter];
      el.addEventListener("change", () => { filters[el.dataset.liveFilter] = el.value; render(view); });
    });
    root.addEventListener("click", event => {
      const queueLink = event.target.closest("[data-live-open-queue]");
      if (queueLink) {
        hiddenItems.delete(hiddenKey("module", "checkpoints"));
        saveHidden(view);
        const module = view.root.querySelector('[data-live-module="checkpoints"]');
        module.hidden = false;
        module.open = true;
        renderCheckpoints(view);
        const queue = [...view.root.querySelectorAll("[data-checkpoint-queue]")]
          .find(el => el.dataset.checkpointQueue === queueLink.dataset.liveOpenQueue);
        queue?.scrollIntoView({behavior: "smooth", block: "start"});
        return;
      }
      const hide = event.target.closest("[data-live-hide-kind]");
      if (hide) {
        event.preventDefault();
        event.stopPropagation();
        const item = {kind: hide.dataset.liveHideKind, id: hide.dataset.liveHideId, label: hide.dataset.liveHideLabel};
        hiddenItems.set(hiddenKey(item.kind, item.id), item);
        view.lastHidden = item;
        saveHidden(view);
        render(view);
        root.querySelector("[data-live-undo]")?.focus({preventScroll: true});
        return;
      }
      if (event.target.closest("[data-live-hidden-manage]")) {
        renderHiddenManager(view);
        root.querySelector("[data-live-hidden-dialog]").showModal();
        return;
      }
      if (event.target.closest("[data-live-hidden-close]")) {
        root.querySelector("[data-live-hidden-dialog]").close();
        return;
      }
      const restore = event.target.closest("[data-live-restore]");
      if (restore) { restoreHidden(view, restore.dataset.liveRestore); return; }
      if (event.target.closest("[data-live-restore-all]")) { restoreHidden(view, null); return; }
      if (event.target.closest("[data-live-undo]")) {
        if (view.lastHidden) restoreHidden(view, hiddenKey(view.lastHidden.kind, view.lastHidden.id));
        return;
      }
      const method = event.target.closest("[data-live-method]");
      if (method) {
        filters.method = filters.method === method.dataset.liveMethod ? "all" : method.dataset.liveMethod;
        root.querySelector('[data-live-filter="method"]').value = filters.method;
        render(view);
      }
      if (event.target.closest("[data-live-refresh]")) {
        view.paused = false;
        root.querySelector("[data-live-pause]").textContent = "자동 갱신 중";
        refresh(view);
      }
      if (event.target.closest("[data-live-pause]")) {
        view.paused = !view.paused;
        event.target.textContent = view.paused ? "자동 갱신 재개" : "자동 갱신 중";
        root.querySelector(".live-connection").textContent = view.paused ? "갱신 일시정지" : "다시 연결 중…";
        if (!view.paused) refresh(view);
      }
    });
    refresh(view);
  }
  window.addEventListener("storage", event => {
    if (event.key !== hiddenStorage && event.key !== null) return;
    hiddenItems = readHidden();
    if (active) { render(active); renderHiddenManager(active); }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && active) refresh(active); });
  return {mount, unmount};
})();
