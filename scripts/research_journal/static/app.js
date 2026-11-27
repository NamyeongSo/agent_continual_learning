"use strict";

const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);
const statusLabels = { planned: "계획", running: "진행 중", completed: "완료", blocked: "보류", archived: "보관" };
const textFields = ["title", "purpose", "hypothesis", "method", "benchmark", "model", "seeds", "split", "budget",
  "git_commit", "command", "run_path", "summary", "limitations", "next_steps", "started_on"];
const conditions = [["method", "방법 / 변형"], ["benchmark", "벤치마크"], ["model", "모델"], ["seeds", "Seed"],
  ["split", "데이터 / split"], ["budget", "예산 / 설정"], ["git_commit", "Git commit"], ["run_path", "실험 경로"]];
const app = { docs: [], info: null, selected: new Set(), status: "all", query: "", method: "", tag: "", sort: "updated",
  currentId: null, editing: null, editorDirty: false, noteDraft: null, logDraft: null,
  lastHash: "", skipHash: false, toastTimer: null };

function dateLabel(value, detailed = false) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat("ko-KR", detailed
    ? { dateStyle: "medium", timeStyle: "short" } : { year: "numeric", month: "2-digit", day: "2-digit" }).format(date);
}

function sizeLabel(value) {
  return value < 1024 ? `${value} B` : value < 1048576 ? `${(value / 1024).toFixed(1)} KiB` : `${(value / 1048576).toFixed(1)} MiB`;
}

function toast(message, error = false) {
  const element = $("#toast");
  clearTimeout(app.toastTimer);
  element.textContent = message;
  element.className = error ? "error" : "";
  element.hidden = false;
  app.toastTimer = setTimeout(() => { element.hidden = true; }, error ? 10000 : 4500);
}

async function api(path, options = {}) {
  const headers = { ...options.headers };
  if (options.method && options.method !== "GET" && app.info?.read_only) {
    throw new Error("외부 접속은 조회 전용입니다. 기록 수정은 로컬 접속에서 가능합니다.");
  }
  if (options.method && options.method !== "GET") headers["X-Journal-Token"] = app.info.csrf_token;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let message = `요청 실패 (${response.status})`;
    try { message = (await response.json()).error || message; } catch { /* Retain the HTTP status. */ }
    throw new Error(message);
  }
  return response.json();
}

async function download(url) {
  toast("내보내기 파일을 준비하고 있습니다.");
  const response = await fetch(url);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "내보내기에 실패했습니다.");
  }
  const disposition = response.headers.get("Content-Disposition") || "";
  const name = disposition.match(/filename\*=UTF-8''(.+)/)?.[1];
  const objectUrl = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = name ? decodeURIComponent(name) : "research-report";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
  toast("다운로드를 준비했습니다.");
}

function upsert(doc) {
  const index = app.docs.findIndex((item) => item.id === doc.id);
  if (index >= 0 && app.docs[index].revision > doc.revision) return;
  if (index < 0) app.docs.push(doc); else app.docs[index] = doc;
  if (app.noteDraft?.id === doc.id && app.noteDraft.base === doc.notes) app.noteDraft.revision = doc.revision;
  updateNavigation();
}

function currentDoc() {
  const doc = app.docs.find((item) => item.id === app.currentId);
  if (!doc) throw new Error("실험 기록을 찾을 수 없습니다.");
  return doc;
}

function updateNavigation() {
  const active = app.docs.filter((doc) => doc.status !== "archived");
  $$("[data-count]").forEach((element) => {
    const status = element.dataset.count;
    element.textContent = status === "all" ? active.length : app.docs.filter((doc) => doc.status === status).length;
  });
  $("#selection-count").textContent = app.selected.size;
  $$("[data-nav]").forEach((element) => {
    element.classList.toggle("active", location.hash === "#live" ? element.dataset.nav === "live"
      : !app.currentId && location.hash.startsWith("#list") && element.dataset.nav === app.status);
    if (element.classList.contains("active")) element.setAttribute("aria-current", "page");
    else element.removeAttribute("aria-current");
  });
}

function exportControl(ids) {
  return `<select class="export-select" aria-label="내보내기 형식" data-export-ids="${escapeHtml(ids.join(","))}">
    <option value="">내보내기…</option><option value="html">공유용 HTML (첨부 포함)</option>
    <option value="markdown">Markdown + 첨부 ZIP</option><option value="csv">결과 지표 CSV</option>
    <option value="latex">논문용 LaTeX 표</option></select>`;
}

function badge(status) { return `<span class="badge ${status}">${statusLabels[status]}</span>`; }

function filteredDocs() {
  let docs = app.docs.filter((doc) => app.status === "all" ? doc.status !== "archived" : doc.status === app.status);
  const words = app.query.toLocaleLowerCase().split(/\s+/).filter(Boolean);
  if (words.length) docs = docs.filter((doc) => {
    const source = [doc.id, ...textFields.map((field) => doc[field]), doc.notes, ...doc.tags,
      ...doc.log.map((entry) => entry.text), ...doc.artifacts.map((item) => item.name + " " + item.caption)].join(" ").toLocaleLowerCase();
    return words.every((word) => source.includes(word));
  });
  if (app.method) docs = docs.filter((doc) => doc.method === app.method);
  if (app.tag) docs = docs.filter((doc) => doc.tags.includes(app.tag));
  return docs.sort((a, b) => app.sort === "title" ? a.title.localeCompare(b.title, "ko") : b.updated_at.localeCompare(a.updated_at));
}

function renderList() {
  const active = app.docs.filter((doc) => doc.status !== "archived");
  const methods = [...new Set(app.docs.map((doc) => doc.method).filter(Boolean))].sort();
  const tags = [...new Set(app.docs.flatMap((doc) => doc.tags))].sort();
  $("#breadcrumb").textContent = "연구 / 실험 목록";
  $("#content").innerHTML = `<div class="intro"><span class="eyebrow">EXPERIMENT INDEX</span><h1>${app.status === "all" ? "실험 목록" : statusLabels[app.status] + " 실험"}</h1><p>무엇을, 왜 실험했는지. 결과와 해석을 함께 남겨 두세요.</p></div>
    <div id="live-preview"></div><div class="stats"><div class="stat"><span class="stat-label">등록된 실험</span><strong>${active.length}</strong></div>
    <div class="stat"><span class="stat-label">진행 중</span><strong>${active.filter((doc) => doc.status === "running").length}</strong></div>
    <div class="stat"><span class="stat-label">완료</span><strong>${active.filter((doc) => doc.status === "completed").length}</strong></div>
    <div class="stat"><span class="stat-label">완료 후 결론 미작성</span><strong>${active.filter((doc) => doc.status === "completed" && !doc.summary).length}</strong></div></div>
    <div class="filters"><input id="search" type="search" aria-label="실험 검색" placeholder="제목, 목적, 경로, 노트, 태그 검색" value="${escapeHtml(app.query)}">
    <select id="method-filter" aria-label="방법 필터"><option value="">모든 방법</option>${methods.map((method) => `<option ${app.method === method ? "selected" : ""} value="${escapeHtml(method)}">${escapeHtml(method)}</option>`).join("")}</select>
    <select id="tag-filter" aria-label="태그 필터"><option value="">모든 태그</option>${tags.map((tag) => `<option ${app.tag === tag ? "selected" : ""} value="${escapeHtml(tag)}">${escapeHtml(tag)}</option>`).join("")}</select>
    <select id="sort" aria-label="정렬"><option value="updated" ${app.sort === "updated" ? "selected" : ""}>최근 수정순</option><option value="title" ${app.sort === "title" ? "selected" : ""}>제목순</option></select>
    <button class="button" data-action="refresh" aria-label="실험 목록 새로고침">새로고침</button></div><div id="list-results"></div>`;
  $("#search").addEventListener("input", (event) => { app.query = event.target.value; renderRows(); });
  for (const [id, field] of [["method-filter", "method"], ["tag-filter", "tag"], ["sort", "sort"]]) {
    $("#" + id).addEventListener("change", (event) => { app[field] = event.target.value; renderRows(); });
  }
  renderRows();
  window.LiveExperiments.mount($("#live-preview"), {compact: true, preview: true});
}

function renderRows() {
  const docs = filteredDocs();
  const container = $("#list-results");
  if (!container) return;
  const selected = app.selected.size;
  const selection = selected ? `<div class="selection-bar"><span><strong>${selected}개 실험 선택됨</strong> · 목록을 이동해도 선택이 유지됩니다.</span><div class="actions"><button class="button small" data-action="compare">비교하기</button>${exportControl([...app.selected])}<button class="button small subtle" data-action="clear-selection">선택 해제</button></div></div>` : "";
  const meta = `<div class="list-meta"><span>${docs.length}개 실험${app.query || app.tag || app.method ? " · 필터 적용 중" : ""}</span><span>체크박스로 선택해 비교 · 내보내기</span></div>`;
  if (!docs.length) {
    container.innerHTML = selection + meta + `<div class="empty"><span class="eyebrow">YOUR RESEARCH, ON RECORD</span><h2>${app.docs.length ? "조건에 맞는 실험이 없습니다" : "첫 실험을 기록해 보세요"}</h2><p>${app.docs.length ? "검색어나 필터를 조정하거나 새 실험을 등록할 수 있습니다." : "제목과 목적부터 시작하면 됩니다. 실행 경로, 결과물, 직접 작성한 노트가 한 실험 안에 모입니다."}</p><button class="button primary" data-action="new">+ 실험 등록</button><div class="steps"><span>01 목적과 경로 등록</span><span>02 노트와 결과물 정리</span><span>03 비교하고 보고서로 공유</span></div></div>`;
    return;
  }
  container.innerHTML = selection + meta + `<div class="table-wrap"><table class="experiment-table"><thead><tr><th class="check-cell"><input type="checkbox" data-select-all aria-label="현재 목록 전체 선택" ${docs.every((doc) => app.selected.has(doc.id)) ? "checked" : ""}></th><th>실험 / 목적 / 경로</th><th>상태</th><th>방법 / 벤치마크</th><th>기록</th><th>최근 수정</th></tr></thead><tbody>${docs.map((doc) => `
    <tr><td class="check-cell"><input type="checkbox" data-select="${doc.id}" aria-label="${escapeHtml(doc.title)} 선택" ${app.selected.has(doc.id) ? "checked" : ""}></td><td class="title-cell"><a class="experiment-link" href="#experiment/${doc.id}">${escapeHtml(doc.title)}</a><p class="cell-purpose">${escapeHtml(doc.purpose)}</p><code class="cell-path" title="${escapeHtml(doc.run_path)}">${escapeHtml(doc.run_path || "실험 경로 미입력")}</code>${doc.tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</td><td>${badge(doc.status)}</td><td>${escapeHtml(doc.method || "—")}<span class="cell-secondary">${escapeHtml(doc.benchmark || "벤치마크 미입력")}</span></td><td>지표 ${doc.metrics.length} · 첨부 ${doc.artifacts.length}<span class="cell-secondary">${doc.notes ? "노트 작성됨" : "노트 미작성"} · 진행 ${doc.log.length}</span></td><td title="${escapeHtml(dateLabel(doc.updated_at, true))}">${escapeHtml(dateLabel(doc.updated_at))}<span class="cell-secondary">${escapeHtml(doc.id.slice(-8))}</span></td></tr>`).join("")}</tbody></table></div>`;
}

function renderDetail() {
  const doc = currentDoc();
  if (!app.logDraft || app.logDraft.id !== doc.id) app.logDraft = { id: doc.id, value: "" };
  if (!app.noteDraft || app.noteDraft.id !== doc.id || !app.noteDraft.dirty) {
    app.noteDraft = { id: doc.id, value: doc.notes, base: doc.notes, revision: doc.revision, dirty: false };
  }
  $("#breadcrumb").textContent = "연구 / 실험 기록";
  $("#content").innerHTML = `<div class="detail-heading"><div><span class="eyebrow">${doc.id}</span><h1>${escapeHtml(doc.title)}</h1><div class="detail-meta">${badge(doc.status)}<span>생성 ${escapeHtml(dateLabel(doc.created_at, true))}</span><span>수정 ${escapeHtml(dateLabel(doc.updated_at, true))}</span></div></div><div class="actions">${exportControl([doc.id])}<button class="button primary" data-action="edit">실험 정보 편집</button></div></div>
    <nav class="detail-tabs" aria-label="실험 내 이동"><a href="#experiment/${doc.id}" data-scroll="overview">개요 · 해석</a><a href="#experiment/${doc.id}" data-scroll="results">결과 · 산출물</a><a href="#experiment/${doc.id}" data-scroll="experiment-notes">실험 노트</a><a href="#experiment/${doc.id}" data-scroll="reproduction">재현 정보</a></nav>
    <div id="live-detail"></div><div class="detail-grid"><div class="detail-primary">
    <section class="panel summary-panel" id="overview"><h2>핵심 결론</h2>${doc.summary ? `<div class="body-text">${escapeHtml(doc.summary)}</div>` : '<p class="hint">아직 결론이 없습니다. 결과를 확인한 뒤 관찰과 해석을 남겨 주세요.</p>'}</section>
    <section class="panel"><div class="field-block"><h3>실험 목적</h3><div class="body-text">${escapeHtml(doc.purpose)}</div></div>${doc.hypothesis ? `<div class="field-block"><h3>가설 / 비교할 변경</h3><div class="body-text">${escapeHtml(doc.hypothesis)}</div></div>` : ""}${doc.limitations ? `<div class="field-block"><h3>한계 / 주의사항</h3><div class="body-text">${escapeHtml(doc.limitations)}</div></div>` : ""}${doc.next_steps ? `<div class="field-block"><h3>다음 할 일</h3><div class="body-text">${escapeHtml(doc.next_steps)}</div></div>` : ""}</section>
    <section class="panel" id="results"><div class="section-heading"><h2>결과 지표 <span class="muted">${doc.metrics.length}</span></h2><button class="button small" data-action="edit-metrics">지표 편집</button></div>${doc.metrics.length ? `<div class="table-wrap"><table class="metrics-table"><thead><tr><th>지표</th><th>값</th><th>단위</th><th>Split</th><th>근거 / 출처</th></tr></thead><tbody>${doc.metrics.map((metric) => `<tr><td>${escapeHtml(metric.name)}</td><td class="metric-value">${escapeHtml(metric.value)}</td><td>${escapeHtml(metric.unit)}</td><td>${escapeHtml(metric.split)}</td><td>${escapeHtml(metric.source)}</td></tr>`).join("")}</tbody></table></div>` : '<p class="hint">비교할 수치를 기록하세요. 원본 CSV나 그래프는 아래에 첨부할 수 있습니다.</p>'}</section>
    <section class="panel" id="artifacts"><div class="section-heading"><h2>산출물 <span class="muted">${doc.artifacts.length}</span></h2><button class="button small" data-action="open-import">서버 경로에서 가져오기</button></div><label class="dropzone" id="dropzone"><strong>파일을 끌어 놓거나 선택하세요</strong>그래프 · CSV / TSV · PDF · 분석 코드 및 문서 / 파일당 64 MiB<input type="file" id="upload-files" multiple aria-label="산출물 업로드"></label><p class="hint" id="upload-status">첨부 시점의 사본과 SHA-256을 보관합니다. 캡션에 그림의 조건과 해석을 남겨 주세요.</p><div class="artifact-list">${doc.artifacts.map((item) => artifactCard(doc.id, item)).join("")}</div></section>
    <section class="panel" id="experiment-notes"><div class="section-heading"><h2>실험 노트</h2><span class="hint">이 실험에 대한 자유로운 기록</span></div><p class="hint">관찰, 실패 원인, 해석, 논문 아이디어를 작성하세요. 저장한 노트는 검색과 보고서에 포함됩니다.</p><form id="note-form"><textarea id="experiment-note" class="note-textarea" aria-label="실험 노트" placeholder="실험을 진행하며 알게 된 점을 자유롭게 남겨 주세요.\n\n관찰한 현상\n가능한 원인과 해석\n확인할 질문 / 논문에 쓸 문장">${escapeHtml(app.noteDraft.value)}</textarea><div class="note-controls"><span class="save-status ${app.noteDraft.dirty ? "dirty" : ""}" id="note-save-status">${app.noteDraft.dirty ? "저장하지 않은 변경 사항이 있습니다" : doc.notes ? "서버에 저장된 노트입니다" : "노트를 작성한 뒤 저장해 주세요"}</span><button class="button primary" type="submit" id="save-note">노트 저장</button></div></form></section>
    <section class="panel" id="progress"><div class="section-heading"><h2>날짜별 진행 기록 <span class="muted">${doc.log.length}</span></h2></div>${doc.log.length ? `<div class="timeline">${[...doc.log].reverse().map((entry) => `<div class="timeline-entry"><time datetime="${escapeHtml(entry.time)}">${escapeHtml(dateLabel(entry.time, true))}</time><div class="body-text">${escapeHtml(entry.text)}</div></div>`).join("")}</div>` : '<p class="hint">실행 시작, 중간 관찰, 오류 해결 등을 시간순으로 쌓아 두세요.</p>'}<form class="log-form" id="log-form"><label class="hint" for="log-text">새 진행 기록</label><textarea id="log-text" required rows="3" placeholder="오늘 진행한 일, 관찰한 점, 다음 확인 사항"></textarea><div class="note-controls"><span class="hint">등록 시각은 자동으로 기록됩니다.</span><button class="button" type="submit">진행 기록 추가</button></div></form></section>
    </div><aside class="detail-aside"><section class="panel" id="reproduction"><h3>경로와 재현 정보</h3><div class="path-box mono">${escapeHtml(doc.run_path || "경로가 입력되지 않았습니다")}</div>${doc.run_path ? '<button class="button small" data-action="copy-path">경로 복사</button>' : ""}<dl class="facts">${conditions.filter(([field]) => field !== "run_path").map(([field, label]) => `<dt>${label}</dt><dd>${escapeHtml(doc[field] || "—")}</dd>`).join("")}<dt>시작일</dt><dd>${escapeHtml(doc.started_on || "—")}</dd></dl>${doc.command ? `<hr><h3>실행 명령</h3><pre class="prewrap">${escapeHtml(doc.command)}</pre><button class="button small" data-action="copy-command">명령 복사</button>` : ""}</section><section class="panel"><h3>기록 관리</h3>${doc.tags.length ? `<div>${doc.tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>` : '<p class="hint">태그가 없습니다.</p>'}<div class="actions"><button class="button small" data-action="select-current">${app.selected.has(doc.id) ? "비교 선택 해제" : "비교 목록에 추가"}</button><button class="button small subtle" data-action="archive">${doc.status === "archived" ? "계획으로 복원" : "보관함으로 이동"}</button></div><p class="hint">보관한 실험도 노트와 첨부 파일을 유지합니다.</p></section></aside></div>`;
  $("#experiment-note").addEventListener("input", (event) => {
    app.noteDraft.value = event.target.value;
    app.noteDraft.dirty = app.noteDraft.value !== app.noteDraft.base;
    $("#note-save-status").textContent = app.noteDraft.dirty ? "저장하지 않은 변경 사항이 있습니다" : "변경 사항이 없습니다";
    $("#note-save-status").classList.toggle("dirty", app.noteDraft.dirty);
  });
  $("#note-form").addEventListener("submit", saveNote);
  $("#log-text").value = app.logDraft.value;
  $("#log-text").addEventListener("input", (event) => { app.logDraft.value = event.target.value; });
  $("#log-form").addEventListener("submit", saveLog);
  $("#upload-files").addEventListener("change", (event) => uploadFiles([...event.target.files]));
  const dropzone = $("#dropzone");
  dropzone.setAttribute("tabindex", "0");
  dropzone.setAttribute("role", "button");
  dropzone.addEventListener("keydown", (event) => {
    if (["Enter", " "].includes(event.key)) { event.preventDefault(); $("#upload-files").click(); }
  });
  dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (event) => { event.preventDefault(); dropzone.classList.remove("dragover"); uploadFiles([...event.dataTransfer.files]); });
  for (const item of doc.artifacts) loadPreview(doc.id, item);
  if (app.info?.read_only) $$("#content textarea").forEach(element => { element.readOnly = true; });
  window.LiveExperiments.mount($("#live-detail"), {id: doc.id, compact: true, preview: true});
}

function artifactUrl(id, artifact, action = "file") { return `/api/experiments/${id}/artifacts/${artifact.id}/${action}`; }

function artifactCard(id, item) {
  const suffix = item.name.split(".").pop().toLowerCase();
  const image = ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(suffix);
  let preview = "";
  if (image) preview = `<a href="${artifactUrl(id, item)}" target="_blank" rel="noopener"><img class="artifact-image" loading="lazy" src="${artifactUrl(id, item)}" alt="${escapeHtml(item.caption || item.name)}"></a>`;
  else if (suffix === "pdf") preview = `<details class="artifact-caption"><summary>PDF 미리보기</summary><iframe loading="lazy" title="${escapeHtml(item.name)}" src="${artifactUrl(id, item)}"></iframe></details>`;
  else if (["csv", "tsv", "txt", "md", "tex", "json", "yaml", "yml", "log"].includes(suffix)) preview = `<div data-preview="${item.id}"><p class="hint">미리보기 불러오는 중…</p></div>`;
  return `<article class="artifact"><header class="artifact-header"><div><strong>${escapeHtml(item.name)}</strong><small>${sizeLabel(item.size)} · ${escapeHtml(dateLabel(item.created_at, true))}</small></div><div class="actions"><a class="button small" href="${artifactUrl(id, item)}?download=1" download="${escapeHtml(item.name)}">다운로드</a><button class="button small subtle" data-action="remove-artifact" data-artifact="${item.id}">첨부 해제</button></div></header><div class="artifact-preview">${preview}</div><div class="artifact-caption"><label class="hint" for="caption-${item.id}">캡션 / 해석</label><textarea id="caption-${item.id}" rows="2" placeholder="무엇을 보여 주는 결과물인지 남겨 주세요.">${escapeHtml(item.caption)}</textarea><div class="note-controls"><span class="hint">${item.source_path ? "서버 파일에서 복사" : "브라우저 업로드"}</span><button class="button small" data-action="save-caption" data-artifact="${item.id}">캡션 저장</button></div><details><summary>원본 경로 · SHA-256</summary><code>${escapeHtml(item.source_path || "브라우저 업로드")}</code><code>${item.sha256}</code></details></div></article>`;
}

async function loadPreview(id, item) {
  const target = $(`[data-preview="${item.id}"]`);
  if (!target) return;
  try {
    const data = await api(artifactUrl(id, item, "preview"));
    if (data.kind === "table") {
      const [head, ...rows] = data.rows;
      target.innerHTML = head ? `<div class="table-wrap"><table><thead><tr>${head.map((cell) => `<th>${escapeHtml(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : '<p class="hint">빈 표입니다.</p>';
    } else target.innerHTML = `<pre>${escapeHtml(data.text)}</pre>`;
    if (data.truncated) target.insertAdjacentHTML("beforeend", '<p class="hint">일부 내용만 표시합니다. 전체 파일은 다운로드해 주세요.</p>');
  } catch (error) { target.textContent = error.message; }
}

async function saveNote(event) {
  event.preventDefault();
  const button = $("#save-note");
  button.disabled = true;
  const value = $("#experiment-note").value;
  const id = app.currentId;
  const textarea = $("#experiment-note");
  try {
    const doc = await api(`/api/experiments/${id}`, { method: "PUT", json: { notes: value, revision: app.noteDraft.revision } });
    upsert(doc);
    if (app.currentId !== id || !textarea.isConnected) { toast("실험 노트를 서버에 저장했습니다."); return; }
    app.noteDraft = { id: doc.id, base: doc.notes, value, revision: doc.revision, dirty: value !== doc.notes };
    // Keep any typing that happened while the save request was in flight.
    const latest = $("#experiment-note").value;
    app.noteDraft.value = latest;
    app.noteDraft.dirty = latest !== doc.notes;
    $("#note-save-status").textContent = app.noteDraft.dirty ? "추가 변경 사항을 저장해 주세요" : `${dateLabel(doc.updated_at, true)} 저장됨`;
    $("#note-save-status").classList.toggle("dirty", app.noteDraft.dirty);
    toast("실험 노트를 서버에 저장했습니다.");
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

async function saveLog(event) {
  event.preventDefault();
  const text = $("#log-text").value;
  const id = app.currentId;
  const button = $("button", event.target);
  button.disabled = true;
  try {
    upsert(await api(`/api/experiments/${id}/notes`, { method: "POST", json: { text } }));
    if (app.currentId !== id) { toast("진행 기록을 추가했습니다."); return; }
    if (app.logDraft?.id === id && app.logDraft.value === text) app.logDraft.value = "";
    renderDetail();
    $("#progress").scrollIntoView({ block: "start" });
    toast("진행 기록을 추가했습니다.");
  } catch (error) { toast(error.message, true); button.disabled = false; }
}

async function uploadFiles(files) {
  if (!files.length) return;
  const id = app.currentId;
  let count = 0;
  try {
    for (const file of files) {
      if (file.size > app.info.max_file_bytes) throw new Error(`${file.name}: 최대 파일 크기는 64 MiB입니다.`);
      if ($("#upload-status")) $("#upload-status").textContent = `${count + 1}/${files.length} 업로드 중: ${file.name}`;
      upsert(await api(`/api/experiments/${id}/artifacts?${new URLSearchParams({ name: file.name })}`, { method: "POST", body: file }));
      count += 1;
    }
    toast(`${count}개 파일을 첨부했습니다.`);
  } catch (error) { toast(`${count ? count + "개 첨부 완료. " : ""}${error.message}`, true); }
  finally { if (app.currentId === id) renderDetail(); }
}

function metricEditor(metric = {}) {
  return `<div class="metric-editor"><div class="metric-grid"><label>지표명<input data-metric="name" value="${escapeHtml(metric.name)}" placeholder="Success rate"></label><label>값<input data-metric="value" value="${escapeHtml(metric.value)}" placeholder="0 또는 72.4 ± 1.2"></label><label>단위<input data-metric="unit" value="${escapeHtml(metric.unit)}" placeholder="%"></label><label>Split<input data-metric="split" value="${escapeHtml(metric.split)}" placeholder="test"></label><button type="button" class="icon-button" data-action="remove-metric" aria-label="지표 삭제">×</button></div><label class="metric-source">근거 / 출처<input data-metric="source" value="${escapeHtml(metric.source)}" placeholder="결과 파일 이름, task 수, 집계 방식"></label></div>`;
}

function openEditor(doc = null, focusMetrics = false) {
  app.editing = doc;
  const form = $("#experiment-form");
  form.reset();
  $("#editor-title").textContent = doc ? "실험 정보 편집" : "새 실험 등록";
  $("#template-row").hidden = Boolean(doc);
  $("#editor-error").textContent = "";
  $("#metric-fields").innerHTML = (doc?.metrics || []).map(metricEditor).join("");
  for (const field of textFields) form.elements[field].value = doc?.[field] || "";
  form.elements.status.value = doc?.status || "planned";
  form.elements.tags.value = (doc?.tags || []).join(", ");
  app.editorDirty = false;
  $("#editor").showModal();
  if (focusMetrics) {
    if (!doc.metrics.length) $("#metric-fields").innerHTML = metricEditor();
    $("#metric-fields").scrollIntoView({ block: "center" });
    $("[data-metric='name']", $("#metric-fields")).focus();
  } else form.elements.title.focus();
}

function closeEditor() {
  if (app.editorDirty && !confirm("저장하지 않은 실험 정보가 있습니다. 편집을 닫을까요?")) return;
  app.editorDirty = false;
  $("#editor").close();
}

function editorPayload() {
  const form = $("#experiment-form");
  const payload = Object.fromEntries(textFields.map((field) => [field, form.elements[field].value]));
  payload.status = form.elements.status.value;
  payload.tags = form.elements.tags.value.split(",").map((tag) => tag.trim()).filter(Boolean);
  payload.metrics = $$(".metric-editor").map((row) => {
    const metric = Object.fromEntries($$("[data-metric]", row).map((input) => [input.dataset.metric, input.value.trim()]));
    if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(metric.value) && Number.isFinite(Number(metric.value))) metric.value = Number(metric.value);
    return metric;
  }).filter((metric) => Object.values(metric).some((value) => value !== ""));
  if (app.editing) payload.revision = app.editing.revision;
  return payload;
}

async function saveExperiment(event) {
  event.preventDefault();
  const button = $("#save-experiment");
  button.disabled = true;
  $("#editor-error").textContent = "";
  try {
    const doc = await api(app.editing ? `/api/experiments/${app.editing.id}` : "/api/experiments", {
      method: app.editing ? "PUT" : "POST", json: editorPayload(),
    });
    upsert(doc);
    app.editorDirty = false;
    $("#editor").close();
    if (location.hash === `#experiment/${doc.id}`) renderDetail();
    else location.hash = `#experiment/${doc.id}`;
    toast("실험 기록을 저장했습니다. 노트와 산출물은 이 페이지에서 추가할 수 있습니다.");
  } catch (error) { $("#editor-error").textContent = error.message; }
  finally { button.disabled = false; }
}

function renderCompare() {
  $("#breadcrumb").textContent = "연구 / 실험 비교";
  const docs = [...app.selected].map((id) => app.docs.find((doc) => doc.id === id)).filter(Boolean);
  const intro = `<div class="detail-heading"><div><span class="eyebrow">EXPERIMENT COMPARISON</span><h1>실험 비교</h1><p class="muted">결과와 재현 조건을 나란히 확인합니다.</p></div><div class="actions"><a class="button" href="#list/all">목록에서 선택</a>${docs.length ? exportControl(docs.map((doc) => doc.id)) : ""}</div></div>`;
  if (!docs.length) {
    $("#content").innerHTML = intro + `<div class="empty"><h2>비교할 실험을 선택하세요</h2><p>실험 목록의 체크박스로 선택하면 결과 지표와 재현 조건을 함께 볼 수 있습니다.</p><a class="button primary" href="#list/all">실험 목록으로 이동</a></div>`;
    return;
  }
  const headers = `<tr><th>항목</th>${docs.map((doc) => `<th><a href="#experiment/${doc.id}">${escapeHtml(doc.title)}</a><br>${badge(doc.status)}</th>`).join("")}</tr>`;
  const keys = [...new Set(docs.flatMap((doc) => doc.metrics.map((metric) => JSON.stringify([metric.name, metric.split, metric.unit]))))].map(JSON.parse);
  const metricRows = keys.map(([name, split, unit]) => `<tr><th>${escapeHtml(name)}<span class="cell-secondary">${escapeHtml([split, unit].filter(Boolean).join(" · "))}</span></th>${docs.map((doc) => {
    const metric = doc.metrics.find((item) => item.name === name && item.split === split && item.unit === unit);
    return `<td>${metric ? `<strong class="metric-value">${escapeHtml(metric.value)}</strong><span class="cell-secondary">${escapeHtml(metric.source)}</span>` : '<span class="muted">—</span>'}</td>`;
  }).join("")}</tr>`).join("");
  const conditionRows = conditions.map(([field, label]) => {
    const different = new Set(docs.map((doc) => doc[field])).size > 1;
    return `<tr class="${different ? "diff-row" : ""}"><th>${label}${different ? '<span class="cell-secondary">조건 차이 있음</span>' : ""}</th>${docs.map((doc) => `<td class="${different ? "different" : ""}">${escapeHtml(doc[field] || "—")}</td>`).join("")}</tr>`;
  }).join("");
  $("#content").innerHTML = intro + `<div class="compare-note">같은 지표명·split·단위를 한 행에 표시합니다. 결과가 없으면 —로 남깁니다. 노란색 셀은 실험 간 조건이 다르다는 뜻이며, 통제 조건이 같은지 함께 확인하세요.</div><h2 class="compare-title">결과 지표</h2><div class="table-wrap"><table class="compare-table"><thead>${headers}</thead><tbody>${metricRows || `<tr><td colspan="${docs.length + 1}">등록된 지표가 없습니다.</td></tr>`}</tbody></table></div><h2 class="compare-title">재현 조건</h2><div class="table-wrap"><table class="compare-table"><thead>${headers}</thead><tbody>${conditionRows}</tbody></table></div><h2 class="compare-title">목적과 해석</h2><div class="table-wrap"><table class="compare-table"><thead>${headers}</thead><tbody>${[["purpose", "목적"], ["summary", "핵심 결론"], ["limitations", "한계 / 주의사항"], ["next_steps", "다음 할 일"]].map(([field, label]) => `<tr><th>${label}</th>${docs.map((doc) => `<td>${escapeHtml(doc[field] || "—")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

async function scanFiles() {
  $("#scan-results").innerHTML = '<p class="hint">파일 목록을 확인하고 있습니다.</p>';
  $("#import-error").textContent = "";
  try {
    const data = await api(`/api/experiments/${app.currentId}/files`);
    $("#scan-results").innerHTML = `<p class="hint mono">${escapeHtml(data.root)}</p><div class="scan-files">${data.files.map((file) => `<div class="scan-file"><div><span class="mono">${escapeHtml(file.relative)}</span><small>${sizeLabel(file.size)}</small></div><button class="button small" data-action="import-file" data-path="${escapeHtml(file.path)}">가져오기</button></div>`).join("") || '<p class="hint">표시할 산출물이 없습니다. 위에서 파일 경로를 직접 입력할 수 있습니다.</p>'}</div>${data.truncated ? '<p class="hint">일부 폴더나 파일은 생략되었습니다. 경로를 직접 입력하면 가져올 수 있습니다.</p>' : ""}`;
  } catch (error) { $("#scan-results").innerHTML = `<p class="form-error">${escapeHtml(error.message)}</p>`; }
}

async function importFile(path, button = null) {
  $("#import-error").textContent = "";
  if (button) button.disabled = true;
  try {
    upsert(await api(`/api/experiments/${app.currentId}/import`, { method: "POST", json: { path, caption: $("#import-caption").value } }));
    renderDetail();
    if (button) button.textContent = "가져옴";
    $("#import-path").value = "";
    toast("산출물 사본을 첨부했습니다.");
  } catch (error) { $("#import-error").textContent = error.message; if (button) button.disabled = false; }
}

function showStorage() {
  $("#storage-content").innerHTML = `<h3>데이터 저장 폴더</h3><div class="path-box mono">${escapeHtml(app.info.data_dir)}</div><p class="hint">브라우저를 닫거나 서버를 재시작해도 기록이 유지됩니다. 데이터베이스와 첨부 사본을 이 폴더에 함께 저장합니다.</p><button class="button primary" data-action="backup">전체 백업 ZIP 다운로드</button><hr><h3>공유하기</h3><p class="hint">실험의 내보내기 메뉴에서 공유용 HTML을 선택하세요. 이미지·표·첨부 파일과 노트를 포함하므로 받는 사람이 서버 없이 열 수 있습니다. Markdown ZIP에는 문서와 첨부 파일이 함께 들어갑니다.</p><h3>백업 복원하기</h3><ol class="compact-list"><li>기록장 서버를 종료합니다.</li><li>백업 ZIP을 새로운 빈 폴더에 압축 해제합니다.</li><li>서버 실행 시 <code>--data-dir /복원한/폴더</code>를 지정합니다.</li></ol><hr><h3>서버 파일을 가져올 수 있는 폴더</h3>${app.info.allowed_roots.map((path) => `<div class="path-box mono">${escapeHtml(path)}</div>`).join("")}<p class="hint">다른 위치의 산출물은 브라우저에서 업로드하거나 서버 시작 시 해당 폴더를 --allow-root로 추가하세요.</p>`;
  $("#storage-dialog").showModal();
}

async function copyText(value) {
  try { await navigator.clipboard.writeText(value); toast("클립보드에 복사했습니다."); }
  catch { toast("자동 복사를 사용할 수 없습니다. 표시된 텍스트를 직접 선택해 복사해 주세요.", true); }
}

async function action(name, button) {
  if (name === "new") openEditor();
  else if (name === "edit") openEditor(currentDoc());
  else if (name === "edit-metrics") openEditor(currentDoc(), true);
  else if (name === "close-editor") closeEditor();
  else if (name === "add-metric") { $("#metric-fields").insertAdjacentHTML("beforeend", metricEditor()); app.editorDirty = true; }
  else if (name === "remove-metric") { button.closest(".metric-editor").remove(); app.editorDirty = true; }
  else if (name === "compare") { if (location.hash === "#compare") renderCompare(); else location.hash = "#compare"; }
  else if (name === "clear-selection") { app.selected.clear(); updateNavigation(); renderRows(); }
  else if (name === "select-current") {
    if (app.selected.has(app.currentId)) app.selected.delete(app.currentId); else app.selected.add(app.currentId);
    updateNavigation(); renderDetail();
  } else if (name === "refresh") {
    app.docs = (await api("/api/experiments")).experiments;
    updateNavigation(); renderList(); toast("최신 기록을 불러왔습니다.");
  } else if (name === "copy-path") await copyText(currentDoc().run_path);
  else if (name === "copy-command") await copyText(currentDoc().command);
  else if (name === "storage") showStorage();
  else if (name === "close-storage") $("#storage-dialog").close();
  else if (name === "backup") await download("/api/backup");
  else if (name === "open-import") {
    $("#import-path").value = ""; $("#import-caption").value = "";
    $("#import-error").textContent = ""; $("#scan-results").innerHTML = "";
    $("#import-dialog").showModal();
  } else if (name === "close-import") $("#import-dialog").close();
  else if (name === "scan-files") await scanFiles();
  else if (name === "import-file") await importFile(button.dataset.path, button);
  else if (name === "save-caption") {
    const artifactId = button.dataset.artifact;
    button.disabled = true;
    try {
      upsert(await api(`/api/experiments/${app.currentId}/artifacts/${artifactId}`, { method: "PUT", json: { caption: $("#caption-" + artifactId).value } }));
      toast("캡션을 저장했습니다.");
    } finally { button.disabled = false; }
  } else if (name === "remove-artifact") {
    if (!confirm("이 실험에서 첨부를 해제할까요? 원본 파일은 유지됩니다.")) return;
    upsert(await api(`/api/experiments/${app.currentId}/artifacts/${button.dataset.artifact}`, { method: "DELETE" }));
    renderDetail(); toast("첨부를 해제했습니다.");
  } else if (name === "archive") {
    const doc = currentDoc();
    const status = doc.status === "archived" ? "planned" : "archived";
    if (!confirm(status === "archived" ? "이 실험을 보관함으로 이동할까요? 노트와 산출물은 유지됩니다." : "이 실험을 계획 상태로 복원할까요?")) return;
    upsert(await api(`/api/experiments/${doc.id}`, { method: "PUT", json: { status, revision: doc.revision } }));
    renderDetail(); toast(status === "archived" ? "보관함으로 이동했습니다." : "실험을 복원했습니다.");
  }
}

function route() {
  if (app.skipHash) { app.skipHash = false; return; }
  const hash = location.hash || "#list/all";
  if (hash !== app.lastHash && (app.noteDraft?.dirty || app.logDraft?.value.trim())) {
    if (!confirm("저장하지 않은 노트 또는 진행 기록이 있습니다. 저장하지 않고 이동할까요?")) {
      app.skipHash = true; location.hash = app.lastHash; return;
    }
    app.noteDraft = null;
    app.logDraft = null;
  }
  const [view, id] = hash.slice(1).split("/");
  app.currentId = view === "experiment" ? id : null;
  if (view === "list") app.status = id in statusLabels || id === "all" ? id : "all";
  app.lastHash = hash;
  window.LiveExperiments.unmount();
  updateNavigation();
  try {
    if (view === "live") {
      $("#breadcrumb").textContent = "연구 / 실시간 모니터";
      $("#content").innerHTML = '<div class="intro"><span class="eyebrow">EXPERIMENT OBSERVATORY</span><h1>실시간 실험 모니터</h1><p>실행 중인 학습, 후보 검증, Test 진행률을 자동으로 갱신합니다.</p></div><div id="live-monitor"></div>';
      window.LiveExperiments.mount($("#live-monitor"));
    }
    else if (view === "experiment") renderDetail();
    else if (view === "compare") renderCompare();
    else renderList();
    document.title = app.currentId ? `${currentDoc().title} · 실험 기록장` : "실험 기록장";
    window.scrollTo(0, 0);
  } catch (error) { $("#content").innerHTML = `<div class="inline-error">${escapeHtml(error.message)} <a href="#list/all">목록으로 돌아가기</a></div>`; }
}

document.addEventListener("click", (event) => {
  const scrollLink = event.target.closest("[data-scroll]");
  if (scrollLink) { event.preventDefault(); $("#" + scrollLink.dataset.scroll)?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  const routeLink = event.target.closest("a[href^='#']");
  if (routeLink && routeLink.hash !== location.hash && (app.noteDraft?.dirty || app.logDraft?.value.trim())) {
    if (!confirm("저장하지 않은 노트 또는 진행 기록이 있습니다. 저장하지 않고 이동할까요?")) {
      event.preventDefault(); return;
    }
    app.noteDraft = null;
    app.logDraft = null;
  }
  const button = event.target.closest("[data-action]");
  if (!button || button.disabled) return;
  event.preventDefault();
  action(button.dataset.action, button).catch((error) => toast(error.message, true));
});

document.addEventListener("change", (event) => {
  const target = event.target;
  if (target.matches("[data-select]")) {
    if (target.checked) app.selected.add(target.dataset.select); else app.selected.delete(target.dataset.select);
    updateNavigation(); renderRows();
  } else if (target.matches("[data-select-all]")) {
    for (const doc of filteredDocs()) { if (target.checked) app.selected.add(doc.id); else app.selected.delete(doc.id); }
    updateNavigation(); renderRows();
  } else if (target.matches("[data-export-ids]") && target.value) {
    const params = new URLSearchParams({ format: target.value, ids: target.dataset.exportIds });
    target.value = "";
    download(`/api/export?${params}`).catch((error) => toast(error.message, true));
  }
});

$("#experiment-form").addEventListener("submit", saveExperiment);
$("#experiment-form").addEventListener("input", () => { app.editorDirty = true; });
$("#experiment-form").addEventListener("change", () => { app.editorDirty = true; });
$("#editor").addEventListener("cancel", (event) => { event.preventDefault(); closeEditor(); });
$("#template").addEventListener("change", (event) => {
  if (event.target.value !== "mutation") return;
  const form = $("#experiment-form");
  form.elements.title.value = "ACE / PREMiSE: ADD-only vs ADD + UPDATE";
  form.elements.purpose.value = "기존 bullet을 수정하는 UPDATE 연산이 별도 교정 bullet을 추가하는 ADD-only 방식보다 playbook 품질과 downstream 성능을 개선하는지 확인한다.";
  form.elements.hypothesis.value = "ADD-only baseline과 ADD + UPDATE를 비교한다. mutation / curator 프롬프트와 operation 적용 코드 모두에서 UPDATE를 지원한다. task split, seed, 모델, 예산과 나머지 설정을 맞춰 UPDATE의 효과를 분리한다.";
  form.elements.method.value = "ACE / PREMiSE · mutation ablation";
  form.elements.tags.value = "ablation, playbook, update";
  app.editorDirty = true;
});
$("#import-form").addEventListener("submit", (event) => { event.preventDefault(); importFile($("#import-path").value, $("button[type=submit]", event.target)).finally(() => { $("button[type=submit]", event.target).disabled = false; }); });
window.addEventListener("hashchange", route);
window.addEventListener("beforeunload", (event) => {
  if (app.noteDraft?.dirty || app.editorDirty || $("#log-text")?.value.trim()) { event.preventDefault(); event.returnValue = ""; }
});

async function start() {
  try {
    const [info, data] = await Promise.all([api("/api/info"), api("/api/experiments")]);
    app.info = info; app.docs = data.experiments;
    document.body.classList.toggle("read-only", Boolean(info.read_only));
    $("#connection-dot").className = "dot connected";
    $("#connection-label").textContent = info.read_only ? "외부 조회 연결됨" : "로컬 서버 연결됨";
    if (!location.hash) history.replaceState(null, "", "#live");
    route();
  } catch (error) {
    $("#connection-dot").className = "dot error";
    $("#connection-label").textContent = "연결 실패";
    $("#content").innerHTML = `<div class="inline-error">${escapeHtml(error.message)} 페이지를 새로고침하거나 서버 상태를 확인해 주세요.</div>`;
  }
}

start();
