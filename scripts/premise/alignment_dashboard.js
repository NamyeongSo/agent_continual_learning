"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  catalog: null, run: null, records: [], filtered: [], reviews: {}, active: null,
  detail: null, draft: null, page: 0, pageSize: 30, pending: new Map(), saving: null,
  saveTimer: null, detailRequest: 0, loading: false, draftKey: null, recovered: new Map(),
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pretty(value) {
  if (value === undefined || value === null) return "(기록 없음)";
  if (typeof value === "string") {
    try { return JSON.stringify(JSON.parse(value), null, 2); } catch { return value; }
  }
  return JSON.stringify(value, null, 2);
}

function copy(value) { return JSON.parse(JSON.stringify(value)); }
function emptyReview() {
  return { axes: Object.fromEntries(state.catalog.axes.map((a) => [a.id, null])), checked: false, notes: "" };
}
function complete(review) { return review && state.catalog.axes.every((a) => review.axes[a.id] === 0 || review.axes[a.id] === 1); }
function failed(review) { return review && Object.values(review.axes).includes(0); }
function rate(n, d) { return d ? `${Math.round(n / d * 100)}%` : "—"; }
function shortRun(name) {
  const bits = name.split("/");
  const leaf = bits.at(-1) === "run" ? bits.at(-2) : bits.at(-1);
  return `${leaf} · ${bits[0]}`;
}

async function api(path, options) {
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); } catch { throw new Error("서버 응답을 읽을 수 없습니다."); }
  if (!response.ok) throw new Error(data.error || `서버 오류 (${response.status})`);
  return data;
}

let toastTimer;
function toast(message, error = false) {
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 5000);
}

function saveStatus(text, mode = "saved") {
  $("save-text").textContent = text;
  $("save-indicator").classList.toggle("error", mode === "error");
  $("save-indicator").classList.toggle("saving", mode === "saving");
  $("review-save-state").textContent = text;
  $("review-save-state").parentElement.classList.toggle("error", mode === "error");
  $("retry-save").hidden = mode !== "error";
}

function cacheDrafts() {
  if (!state.draftKey) return;
  try {
    const drafts = new Map([...state.recovered, ...state.pending]);
    if (drafts.size) localStorage.setItem(state.draftKey, JSON.stringify([...drafts]));
    else localStorage.removeItem(state.draftKey);
  } catch {
    // The server remains the canonical store even when browser storage is unavailable.
  }
}

function edited() {
  if (!state.active || !state.draft) return;
  if (!complete(state.draft)) state.draft.checked = false;
  state.reviews[state.active] = copy(state.draft);
  state.pending.set(state.active, { run: state.run, review: copy(state.draft), version: Date.now() + Math.random() });
  cacheDrafts();
  saveStatus("변경 사항 저장 대기", "saving");
  renderEvaluationState();
  filterRows(false);
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(flush, 400);
}

async function flush() {
  clearTimeout(state.saveTimer);
  if (state.saving) return state.saving;
  if (!state.pending.size) return true;
  state.saving = (async () => {
    try {
      while (state.pending.size) {
        const [id, item] = state.pending.entries().next().value;
        saveStatus("로컬 파일에 저장 중…", "saving");
        const data = await api("/api/review", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run: item.run, id, review: item.review }),
        });
        if (state.pending.get(id)?.version === item.version) {
          state.pending.delete(id);
          if (item.run === state.run) state.reviews[id] = data.review;
        }
        cacheDrafts();
      }
      saveStatus(`로컬 저장 완료 · ${new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
      filterRows(false);
      return true;
    } catch (error) {
      saveStatus(`저장 실패 · ${error.message}`, "error");
      cacheDrafts();
      return false;
    } finally { state.saving = null; }
  })();
  return state.saving;
}

function axisRate(rows, key) {
  const reviews = rows.map((r) => state.reviews[r.id]).filter((r) => r && (r.axes[key] === 0 || r.axes[key] === 1));
  return [reviews.filter((r) => r.axes[key] === 1).length, reviews.length];
}

function allPassRate(rows) {
  const reviews = rows.map((r) => state.reviews[r.id]).filter((r) => r?.checked && complete(r));
  return [reviews.filter((r) => !failed(r)).length, reviews.length];
}

function renderMetrics() {
  const rows = state.filtered;
  const checked = rows.filter((r) => state.reviews[r.id]?.checked).length;
  $("stat-total").textContent = rows.length.toLocaleString();
  $("stat-scope").textContent = `${new Set(rows.map((r) => `${r.benchmark}|${r.task_id}`)).size}개 task · 현재 필터 기준`;
  $("stat-reviewed").textContent = checked.toLocaleString();
  $("stat-reviewed-percent").textContent = rate(checked, rows.length);
  $("completion-bar").style.width = `${rows.length ? checked / rows.length * 100 : 0}%`;
  const calibration = axisRate(rows, "score_calibration");
  $("stat-calibration").textContent = rate(...calibration);
  $("stat-calibration-n").textContent = `충족 ${calibration[0]} / 평가 ${calibration[1]}건`;
  const all = allPassRate(rows);
  $("stat-all-pass").textContent = rate(...all);
  $("stat-all-pass-n").textContent = `모두 충족 ${all[0]} / 검증 완료 ${all[1]}건`;
  $("axis-bars").replaceChildren();
  state.catalog.axes.forEach((axis) => {
    const [passed, count] = axisRate(rows, axis.id);
    const line = element("div", "axis-stat");
    const progress = element("div", "progress");
    const fill = element("i"); fill.style.width = `${count ? passed / count * 100 : 0}%`; progress.append(fill);
    line.append(element("span", "", axis.name), progress, element("small", "muted", `${rate(passed, count)} · ${passed}/${count}`));
    $("axis-bars").append(line);
  });
  $("benchmark-stats").replaceChildren();
  [...new Set(rows.map((r) => r.benchmark))].sort().forEach((benchmark) => {
    const subset = rows.filter((r) => r.benchmark === benchmark);
    const count = subset.filter((r) => state.reviews[r.id]?.checked).length;
    const c = axisRate(subset, "score_calibration");
    const a = allPassRate(subset);
    const tr = element("tr");
    [benchmark, `${count} / ${subset.length}`, `${rate(...c)} (${c[0]}/${c[1]})`, `${rate(...a)} (${a[0]}/${a[1]})`]
      .forEach((value) => tr.append(element("td", "", value)));
    $("benchmark-stats").append(tr);
  });
}

function hashOrder(id) {
  let h = 42;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return h >>> 0;
}

function filterRows(resetPage = true) {
  const query = $("search").value.trim().toLowerCase();
  const benchmark = $("benchmark-filter").value, scope = $("scope-filter").value;
  const score = $("score-filter").value, status = $("status-filter").value;
  state.filtered = state.records.filter((row) => {
    const review = state.reviews[row.id];
    return (!benchmark || row.benchmark === benchmark) && (!scope || row.scope === scope)
      && (score === "" || Number(score) === row.judge_score)
      && (!status || (status === "pending" && !review?.checked) || (status === "checked" && review?.checked) || (status === "failed" && failed(review)))
      && (!query || `${row.task_id} ${row.candidate} ${row.iteration} ${row.rationale}`.toLowerCase().includes(query));
  });
  if ($("shuffle").checked) state.filtered.sort((a, b) => hashOrder(a.id) - hashOrder(b.id));
  if (resetPage) state.page = 0;
  renderMetrics(); renderQueue();
}

function renderQueue() {
  const list = $("case-list"); list.replaceChildren();
  const totalPages = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
  state.page = Math.min(state.page, totalPages - 1);
  const rows = state.filtered.slice(state.page * state.pageSize, (state.page + 1) * state.pageSize);
  $("queue-count").textContent = `${state.filtered.length.toLocaleString()}개 판정`;
  if (!rows.length) list.append(element("p", "no-actions", "조건에 맞는 판정이 없습니다."));
  rows.forEach((row) => {
    const review = state.reviews[row.id];
    const button = element("button", `case-item${row.id === state.active ? " active" : ""}`);
    button.type = "button"; button.setAttribute("role", "listitem");
    button.setAttribute("aria-current", row.id === state.active ? "true" : "false");
    const top = element("div", "case-item-top");
    const scoreClass = row.judge_score === 0 ? " zero" : row.judge_score === 0.5 ? " partial" : "";
    top.append(element("span", "bench-tag", row.benchmark), element("span", `score-pill${scoreClass}`, $("blind-toggle").checked ? "—" : `${row.judge_score}`));
    const meta = element("div", "case-item-meta");
    meta.append(element("span", "", `#${String(row.iteration).padStart(3, "0")} · ${row.candidate.replace("candidate_", "C")} · ${row.scope}`));
    const label = review?.checked ? "✓ 완료" : failed(review) ? "0 포함" : review && Object.values(review.axes).some((v) => v !== null) ? "평가 중" : "○";
    meta.append(element("span", `case-item-status${failed(review) ? " failed" : ""}`, label));
    button.append(top, element("div", "case-item-task", row.task_id), meta);
    button.addEventListener("click", () => selectCase(row.id));
    list.append(button);
  });
  $("page-label").textContent = `${state.page + 1} / ${totalPages}`;
  $("page-prev").disabled = state.page === 0; $("page-next").disabled = state.page >= totalPages - 1;
  const index = state.filtered.findIndex((r) => r.id === state.active);
  $("case-prev").disabled = index <= 0;
  $("case-next").disabled = index < 0 || index >= state.filtered.length - 1;
}

function actionNodes(actions, offset = 0, isReplay = false) {
  const fragment = document.createDocumentFragment();
  if (!actions?.length) { fragment.append(element("p", "no-actions", "기록된 행동이 없습니다.")); return fragment; }
  actions.forEach((item, index) => {
    const action = item.action || {};
    const card = element("article", "action");
    const heading = element("div", "action-heading");
    const number = item.episode !== undefined ? `episode[${item.episode}]` : `${isReplay ? "replay" : "action"}[${offset + index}]`;
    heading.append(element("span", "action-number", number), element("span", "action-name", action.name || item.action_class || "action"));
    card.append(heading);
    const args = action.arguments ?? action;
    let native = null;
    if (action.name === "terminus2_episode" && typeof args.response === "string") {
      try { native = JSON.parse(args.response); } catch { /* Raw response remains visible below. */ }
    }
    if (native && typeof native === "object" && !Array.isArray(native)) {
      if (native.analysis) card.append(element("div", "action-label", "NATIVE ANALYSIS"), element("pre", "", pretty(native.analysis)));
      if (Array.isArray(native.commands)) native.commands.forEach((command, i) => {
        card.append(element("div", "action-label", `COMMAND ${i}${command.duration !== undefined ? ` · ${command.duration}s` : ""}`), element("pre", "native-command", pretty(command.keystrokes ?? command)));
      });
      for (const [key, value] of Object.entries(native)) if (!["analysis", "commands"].includes(key)) card.append(element("div", "action-label", key.toUpperCase()), element("pre", "", pretty(value)));
    } else if (args && typeof args === "object" && !Array.isArray(args)) {
      for (const [key, value] of Object.entries(args)) card.append(element("div", "action-label", key.toUpperCase()), element("pre", "", pretty(value)));
    } else card.append(element("pre", "", pretty(args)));
    if (item.message) {
      const detail = element("details"); detail.append(element("summary", "", "원본 message"), element("pre", "", pretty(item.message))); card.append(detail);
    }
    const observation = element("div", "action-observation");
    if (item.observation && typeof item.observation === "object" && !Array.isArray(item.observation)) {
      for (const [key, value] of Object.entries(item.observation)) observation.append(element("div", "action-label", key.toUpperCase()), element("pre", "", pretty(value)));
    } else observation.append(element("pre", "", pretty(item.observation)));
    card.append(element("div", "action-label", "OBSERVATION"), observation);
    const raw = element("details"); raw.append(element("summary", "", "기록된 action JSON 전체"), element("pre", "", JSON.stringify(item, null, 2))); card.append(raw);
    fragment.append(card);
  });
  return fragment;
}

function renderAxes() {
  $("axis-controls").replaceChildren();
  state.catalog.axes.forEach((axis, index) => {
    const row = element("div", "axis-control");
    const description = element("div", "axis-description");
    const name = element("p", "axis-name");
    name.append(element("span", "axis-number", `0${index + 1}`), document.createTextNode(axis.name));
    description.append(name, element("p", "axis-question", axis.question));
    const rubric = element("details"); rubric.append(element("summary", "", "평가 기준 보기"));
    [["1 · 충족", axis.pass], ["0 · 미충족", axis.fail]].forEach(([title, text]) => {
      const p = element("p"); p.append(element("b", "", title), document.createTextNode(` — ${text}`)); rubric.append(p);
    });
    description.append(rubric);
    const options = element("div", "axis-options");
    options.setAttribute("role", "group"); options.setAttribute("aria-label", axis.name);
    [0, 1].forEach((value) => {
      const button = element("button", "axis-score", `${value}`);
      button.type = "button"; button.dataset.axis = axis.id; button.dataset.value = `${value}`;
      button.setAttribute("aria-label", `${axis.name} ${value}: ${value ? "충족" : "미충족"}`);
      button.append(element("span", "", value ? "충족" : "미충족"));
      button.addEventListener("click", () => {
        state.draft.axes[axis.id] = state.draft.axes[axis.id] === value ? null : value;
        edited();
      });
      options.append(button);
    });
    row.append(description, options); $("axis-controls").append(row);
  });
}

function renderEvaluationState() {
  if (!state.draft) return;
  document.querySelectorAll(".axis-score").forEach((button) => {
    const active = state.draft.axes[button.dataset.axis] === Number(button.dataset.value);
    button.classList.toggle("active", active); button.setAttribute("aria-pressed", `${active}`);
  });
  const count = Object.values(state.draft.axes).filter((v) => v === 0 || v === 1).length;
  $("review-progress").textContent = `${count} / 5 평가`;
  $("review-checked").checked = state.draft.checked;
  $("review-checked").disabled = !complete(state.draft);
  $("review-checked").title = complete(state.draft) ? "직접 검증 완료" : "다섯 축을 모두 평가하면 체크할 수 있습니다.";
  $("save-next").disabled = !complete(state.draft);
}

function renderBlind() {
  const hidden = $("blind-toggle").checked;
  $("judge-grade").hidden = hidden; $("judge-rationale").hidden = hidden; $("judge-hidden").hidden = !hidden;
  renderQueue();
}

function renderCase(data) {
  $("empty-state").hidden = true; $("case-content").hidden = false;
  $("case-title").textContent = data.task_id;
  $("case-meta").textContent = `Iteration ${data.iteration} · ${data.candidate} · ${data.artifact}`;
  $("case-badges").replaceChildren(
    element("span", "badge", data.benchmark.toUpperCase()), element("span", "badge kind", data.scope === "current" ? "CURRENT TASK" : `PAST TASK ${data.past_index + 1}`),
  );
  if (data.selected) $("case-badges").append(element("span", "badge selected", "선택된 candidate"));
  if (!data.valid_candidate) $("case-badges").append(element("span", "badge", "무효 candidate의 기록"));
  $("case-warning").textContent = data.warnings.join("\n"); $("case-warning").hidden = !data.warnings.length;
  $("task-text").textContent = pretty(data.source?.task || data.replay.task);
  $("task-context").textContent = pretty(data.source?.context);
  $("tool-contract").textContent = pretty(data.source?.public_tool_contract);
  $("expected-behavior").textContent = data.reflection?.expected_behavior || "당시 reflection을 복원하지 못했습니다.";
  $("reflection-analysis").textContent = data.reflection?.analysis || "(기록 없음)";
  const start = data.span.start, end = data.span.end;
  $("span-label").textContent = start !== undefined ? `SPAN [${start}, ${end}) · ${data.span.type || ""}` : "SPAN 불명";
  $("judge-grade").textContent = `${data.judge.score} · ${data.judge.label || "label 없음"}`;
  $("judge-grade").className = `judge-grade${data.judge.score === 0 ? " zero" : data.judge.score === 0.5 ? " partial" : ""}`;
  $("judge-rationale").textContent = data.judge.rationale || "(설명 없음)";
  $("judge-prompt").textContent = data.reconstructed_prompt || "원본·reflection이 없어 프롬프트를 재구성할 수 없습니다.";
  const original = data.source?.actions || [];
  const span = start !== undefined && end !== undefined ? original.slice(start, end) : [];
  const replay = data.replay.actions || [];
  $("original-count").textContent = `${span.length} actions`; $("replay-count").textContent = `${replay.length} actions`;
  $("original-actions").replaceChildren(actionNodes(span, start || 0));
  $("replay-actions").replaceChildren(actionNodes(replay, 0, true));
  $("original-actions").scrollTop = 0; $("replay-actions").scrollTop = 0;
  $("full-original").replaceChildren();
  document.querySelector(".extra-evidence").open = false;
  $("prefix-note").textContent = `원본 출처: ${data.source_artifact || "복원 불가"}. 원본의 [0, ${start ?? "?"})는 복원 prefix입니다. 비교 패널은 judge에 전달된 replay actions 전체를 보여줍니다. 원본은 원래 action 번호, replay는 전달된 배열 번호를 사용하며 Terminal-Bench는 기록된 episode 번호를 표시합니다.`;
  const { actions: _actions, context: _context, public_tool_contract: _contract, ...metadata } = data.replay;
  $("replay-metadata").textContent = pretty({ ...metadata, official_score: data.official_score, original_score: data.source?.score });
  $("mutation-text").textContent = pretty({ mutation_summary: data.mutation_summary, applied: data.applied });
  state.draft = copy(state.reviews[data.id] || emptyReview());
  $("review-notes").value = state.draft.notes || "";
  renderEvaluationState(); renderBlind();
}

async function selectCase(id) {
  if (id === state.active && state.detail) return;
  if (state.active && !(await flush())) { toast("저장이 완료되어야 다른 판정으로 이동할 수 있습니다.", true); return; }
  const request = ++state.detailRequest;
  state.active = id; state.detail = null;
  $("case-content").hidden = true; $("empty-state").hidden = false;
  $("empty-state").querySelector("h2").textContent = "기록된 근거를 읽고 있습니다…";
  const index = state.filtered.findIndex((r) => r.id === id);
  if (index >= 0) state.page = Math.floor(index / state.pageSize);
  renderQueue();
  try {
    const data = await api(`/api/case?run=${encodeURIComponent(state.run)}&id=${encodeURIComponent(id)}`);
    if (request !== state.detailRequest) return;
    state.detail = data; renderCase(data);
    if (state.pending.size) flush();
    const url = new URL(location.href); url.searchParams.set("run", state.run); url.searchParams.set("case", id);
    history.replaceState(null, "", url);
  } catch (error) {
    if (request !== state.detailRequest) return;
    $("empty-state").querySelector("h2").textContent = "판정을 불러오지 못했습니다";
    $("empty-state").querySelector("p").textContent = error.message; toast(error.message, true);
  }
}

async function loadRun(id, refresh = false) {
  if (state.loading) return;
  if (!(await flush())) { $("run-select").value = state.run; toast("저장 실패를 해결한 뒤 실험을 변경하세요.", true); return; }
  state.loading = true; $("run-select").disabled = true; $("refresh").disabled = true;
  ++state.detailRequest; state.active = null; state.detail = null; state.draft = null;
  $("case-content").hidden = true; $("empty-state").hidden = false;
  $("empty-state").querySelector("h2").textContent = "실험의 판정을 연결하고 있습니다…";
  try {
    const data = await api(`/api/run?run=${encodeURIComponent(id)}${refresh ? "&refresh=1" : ""}`);
    state.run = id; state.records = data.records; state.reviews = data.reviews;
    $("run-select").value = id; $("run-name").textContent = data.run;
    $("benchmark-filter").replaceChildren(new Option("전체", ""));
    [...new Set(state.records.map((r) => r.benchmark))].sort().forEach((b) => $("benchmark-filter").append(new Option(b, b)));
    ["search", "scope-filter", "score-filter", "status-filter"].forEach((key) => { $(key).value = ""; });
    $("global-message").textContent = data.warnings.join("\n"); $("global-message").hidden = !data.warnings.length;
    let restored = 0;
    for (const [caseId, draft] of state.recovered) {
      if (draft.run === id && state.records.some((r) => r.id === caseId)) {
        state.pending.set(caseId, draft); state.recovered.delete(caseId); restored++;
      }
    }
    if (restored) toast(`${restored}개 미저장 평가를 복구하고 있습니다.`);
    for (const [caseId, draft] of state.pending) if (draft.run === id && state.records.some((r) => r.id === caseId)) state.reviews[caseId] = copy(draft.review);
    filterRows();
    const fromUrl = new URL(location.href).searchParams.get("case");
    const preferred = state.records.find((r) => r.id === fromUrl) || state.filtered.find((r) => !state.reviews[r.id]?.checked) || state.filtered[0];
    if (preferred) await selectCase(preferred.id);
    else { $("empty-state").querySelector("h2").textContent = "표시할 alignment 판정이 없습니다"; }
  } catch (error) { toast(error.message, true); $("empty-state").querySelector("h2").textContent = "실험을 불러오지 못했습니다"; }
  finally { state.loading = false; $("run-select").disabled = false; $("refresh").disabled = false; }
}

async function navigate(direction) {
  const index = state.filtered.findIndex((r) => r.id === state.active);
  const row = state.filtered[index + direction];
  if (row) await selectCase(row.id);
}

async function initialize() {
  try {
    state.catalog = await api("/api/catalog");
    state.draftKey = `premise-alignment-drafts-v1:${state.catalog.review_file}`;
    try {
      const restored = JSON.parse(localStorage.getItem(state.draftKey) || "[]");
      for (const [id, draft] of restored) {
        if (typeof id === "string" && draft?.review?.axes && typeof draft.run === "string") state.recovered.set(id, draft);
      }
    } catch { /* Server annotations remain available if a browser draft is invalid. */ }
    $("storage-path").textContent = `자동 저장: ${state.catalog.review_file}`;
    state.catalog.runs.forEach((run) => $("run-select").append(new Option(shortRun(run.name), run.id)));
    renderAxes(); saveStatus("평가·메모·체크가 로컬 파일에 자동 저장됩니다.");
    if (!state.catalog.runs.length) { $("empty-state").querySelector("h2").textContent = "저장된 alignment 판정이 없습니다"; return; }
    const requested = new URL(location.href).searchParams.get("run");
    const id = state.catalog.runs.some((run) => run.id === requested) ? requested : state.catalog.default_run;
    await loadRun(id);
  } catch (error) { saveStatus(error.message, "error"); toast(error.message, true); }
}

$("run-select").addEventListener("change", () => loadRun($("run-select").value));
$("refresh").addEventListener("click", () => loadRun(state.run, true));
["benchmark-filter", "scope-filter", "score-filter", "status-filter", "shuffle"].forEach((id) => $(id).addEventListener("change", filterRows));
$("search").addEventListener("input", filterRows);
$("page-prev").addEventListener("click", () => { state.page--; renderQueue(); });
$("page-next").addEventListener("click", () => { state.page++; renderQueue(); });
$("case-prev").addEventListener("click", () => navigate(-1));
$("case-next").addEventListener("click", () => navigate(1));
$("blind-toggle").addEventListener("change", renderBlind);
$("review-notes").addEventListener("input", () => { state.draft.notes = $("review-notes").value; edited(); });
$("review-checked").addEventListener("change", () => { state.draft.checked = $("review-checked").checked; edited(); });
$("retry-save").addEventListener("click", flush);
$("save-next").addEventListener("click", async () => {
  state.draft.checked = true; edited();
  if (!(await flush())) return;
  filterRows();
  const index = state.filtered.findIndex((r) => r.id === state.active);
  const ordered = [...state.filtered.slice(index + 1), ...state.filtered.slice(0, index + 1)];
  const next = ordered.find((r) => !state.reviews[r.id]?.checked);
  if (next) { await selectCase(next.id); document.querySelector(".review-workspace").scrollIntoView({ behavior: "smooth", block: "start" }); }
  else toast("현재 필터의 모든 판정을 검증했습니다.");
});
document.querySelector(".extra-evidence").addEventListener("toggle", (event) => {
  if (event.target.open && state.detail && !$("full-original").childNodes.length) $("full-original").append(actionNodes(state.detail.source?.actions || []));
});
$("export").addEventListener("click", async () => {
  await flush();
  try {
    const data = await api("/api/export");
    if (state.pending.size || state.recovered.size) data.unsaved_browser_drafts = Object.fromEntries([...state.recovered, ...state.pending]);
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const a = element("a"); a.href = url; a.download = `premise-alignment-reviews-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { toast(error.message, true); }
});
document.addEventListener("keydown", (event) => {
  if (event.altKey && ["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); navigate(event.key === "ArrowLeft" ? -1 : 1); }
});
window.addEventListener("beforeunload", (event) => {
  if (state.pending.size) { cacheDrafts(); event.preventDefault(); event.returnValue = ""; }
});
initialize();
