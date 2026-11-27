"use strict";

const $ = (id) => document.getElementById(id);
const controls = ["model", "mode", "arm", "seed"];
const modeNames = { thinking: "Thinking", nonthinking: "Non-thinking", high: "High", low: "Low" };
const armNames = { without_gt: "w/o GT", with_gt: "w/ GT" };
const state = { catalog: null, queue: [], index: -1, pair: null, busy: false, finished: false };
let saved = {};
try { saved = JSON.parse(localStorage.getItem("reflection-review-filters") || "{}"); } catch { /* Defaults work without storage. */ }

async function api(path, options) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function message(text = "", error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function contentLabels() {
  const rules = state.catalog.content_kind === "independent_rules";
  document.title = rules ? "Playbook Rule Review" : "Reflection Detail Review";
  $("brand-title").textContent = rules ? "Playbook Rule Review" : "Reflection Review";
  $("review-question").textContent = rules ? "어느 rule 묶음이 더 디테일한가요?" : "어느 Reflection이 더 디테일한가요?";
  $("review-instructions").textContent = rules
    ? "동일 Task에서 비축적 업데이트로 실제 추가된 rule만 비교합니다. 선택하면 자동 저장하고 다음 쌍으로 이동합니다."
    : "내용의 구체성과 설명의 세밀함을 비교해 주세요. 선택하면 자동 저장하고 다음 쌍으로 이동합니다.";
  $("left-title").textContent = rules ? "Rules A" : "Reflection A";
  $("right-title").textContent = rules ? "Rules B" : "Reflection B";
  $("review-footer").textContent = rules ? "ACE · AppWorld · Non-accumulated playbook rules" : "ACE · AppWorld · Reflection content review";
}

function renderContent(side, pair) {
  const element = $(`${side}-text`);
  element.replaceChildren();
  const rules = pair[`${side}_rules`];
  $(`${side}-rule-count`).hidden = !Array.isArray(rules);
  if (Array.isArray(rules)) {
    $(`${side}-rule-count`).textContent = `${rules.length} rules`;
    if (!rules.length) {
      const empty = document.createElement("p");
      empty.className = "no-rules";
      empty.textContent = "추가된 rule이 없습니다.";
      element.append(empty);
    }
    for (const [index, content] of rules.entries()) {
      const block = document.createElement("div");
      block.className = "rule-block";
      const label = document.createElement("span");
      label.className = "rule-number";
      label.textContent = `RULE ${String(index + 1).padStart(2, "0")}`;
      const text = document.createElement("div");
      text.className = "rule-body";
      text.textContent = content;
      block.append(label, text);
      element.append(block);
    }
  } else element.textContent = pair[side];
  element.scrollTop = 0;
}

function busy(value) {
  state.busy = value;
  document.querySelectorAll("select,button[data-choice],#previous,#next,#review-again").forEach((el) => {
    el.disabled = value;
  });
  if (!value) $("previous").disabled = state.index <= 0 || !state.queue.length;
}

function filters() { return Object.fromEntries(controls.map((key) => [key, $(key).value])); }

function fill(id, values, preferred) {
  const el = $(id);
  el.replaceChildren();
  for (const [value, label, disabled = false] of values) {
    const option = new Option(label, value);
    option.disabled = disabled;
    el.add(option);
  }
  el.value = values.some((v) => v[0] === String(preferred) && !v[2]) ? preferred : values.find((v) => !v[2])?.[0] || "";
}

function populate(from = "model") {
  const model = state.catalog.models.find((m) => m.id === $("model").value);
  if (from === "model") {
    const modes = [...new Set(model.groups.map((g) => g.mode))];
    modes.sort((a, b) => ["thinking", "high", "nonthinking", "low"].indexOf(a) - ["thinking", "high", "nonthinking", "low"].indexOf(b));
    fill("mode", modes.map((m) => [m, modeNames[m]]), saved.mode || $("mode").value);
  }
  if (["model", "mode"].includes(from)) {
    const available = model.groups.filter((g) => g.mode === $("mode").value);
    fill("arm", Object.entries(armNames).map(([key, label]) => [key, label, !available.some((g) => g.arm === key)]), saved.arm || $("arm").value);
  }
  if (["model", "mode", "arm"].includes(from)) {
    const seeds = [...new Set(model.groups.filter((g) => g.mode === $("mode").value && g.arm === $("arm").value).map((g) => g.seed))].sort();
    fill("seed", [["all", "모든 seed"], ...seeds.map((s) => [String(s), `Seed ${s}`])], saved.seed || $("seed").value || "42");
  }
}

function updateProgress() {
  const done = state.queue.filter((p) => p.choice).length;
  $("scope-progress").textContent = `${done} / ${state.queue.length} 쌍 평가 완료`;
  $("progress-fill").style.width = `${state.queue.length ? 100 * done / state.queue.length : 0}%`;
  $("scope-caption").textContent = `${state.queue.length - done}쌍 남음`;
  $("global-progress").textContent = `전체 ${state.catalog.reviewed.toLocaleString()} / ${state.catalog.total.toLocaleString()}`;
}

function empty(title, copy) {
  state.pair = null;
  $("comparison").hidden = true;
  $("empty").hidden = false;
  $("empty-title").textContent = title;
  $("empty-copy").textContent = copy;
  $("review-again").hidden = !state.queue.length;
}

async function showPair(index) {
  state.index = index;
  if (index < 0 || !state.queue[index]) {
    state.finished = true;
    empty(state.queue.length ? "이 조건의 평가를 모두 마쳤어요." : "아직 비교할 쌍이 없어요.",
      state.queue.length ? "다른 모델이나 조건을 선택하거나, 저장한 평가를 다시 확인할 수 있어요." : "실제 선정된 Reflection 쌍이 준비되면 표시됩니다.");
    return;
  }
  state.finished = false;
  const pair = await api(`/api/pairs/${state.queue[index].id}`);
  state.pair = pair;
  $("empty").hidden = true;
  $("comparison").hidden = false;
  $("pair-position").textContent = `PAIR ${index + 1} / ${state.queue.length}`;
  $("task-id").textContent = `Task ${pair.task_id}`;
  $("pair-seed").textContent = `Seed ${pair.seed}`;
  $("task-text").textContent = pair.task;
  $("task-text").parentElement.open = false;
  renderContent("left", pair);
  renderContent("right", pair);
  $("note").value = pair.vote?.note || "";
  $("rated-badge").hidden = !pair.vote;
  for (const side of ["left", "right"]) {
    const selected = pair.vote?.choice === side;
    $(`card-${side}`).classList.toggle("selected", selected);
    $(`card-${side}`).querySelector(".choice-tag").hidden = !selected;
  }
  document.querySelector('[data-choice="tie"]').classList.toggle("selected", pair.vote?.choice === "tie");
}

function nextUnrated(after = -1) {
  for (let step = 1; step <= state.queue.length; step++) {
    const index = (after + step) % state.queue.length;
    if (!state.queue[index].choice) return index;
  }
  return -1;
}

async function loadQueue() {
  busy(true);
  message();
  try {
    const result = await api(`/api/queue?${new URLSearchParams(filters())}`);
    state.queue = result.pairs;
    updateProgress();
    await showPair(nextUnrated());
    try { localStorage.setItem("reflection-review-filters", JSON.stringify(filters())); } catch { /* Votes live on server. */ }
  } catch (error) { message(error.message, true); }
  finally { busy(false); }
}

async function vote(choice) {
  if (state.busy || !state.pair) return;
  busy(true);
  message();
  try {
    const result = await api("/api/vote", { method: "POST", headers: { "Content-Type": "application/json", "X-Review-Token": state.catalog.csrf_token },
      body: JSON.stringify({ pair_id: state.pair.id, choice, note: $("note").value.trim() }) });
    state.catalog.reviewed = result.reviewed;
    state.queue[state.index].choice = choice;
    $("save-status").textContent = `✓ 저장됨 · ${new Date().toLocaleTimeString("ko-KR")}`;
    updateProgress();
    await showPair(nextUnrated(state.index));
  } catch (error) { message(`저장/이동 실패: ${error.message} 새로고침하면 저장 상태를 다시 확인합니다.`, true); }
  finally { busy(false); }
}

async function navigate(index) {
  if (state.busy) return;
  busy(true);
  try { await showPair(index); message(); }
  catch (error) { message(error.message, true); }
  finally { busy(false); }
}

for (const key of controls) $(key).addEventListener("change", async () => {
  saved = {};
  populate(key);
  await loadQueue();
});
document.querySelectorAll("[data-choice]").forEach((button) => button.addEventListener("click", () => vote(button.dataset.choice)));
$("previous").addEventListener("click", () => navigate(state.index - 1));
$("next").addEventListener("click", () => navigate(nextUnrated(state.index)));
$("review-again").addEventListener("click", () => navigate(0));
document.addEventListener("keydown", (event) => {
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey || ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  const choice = { "1": "left", "2": "right", "0": "tie" }[event.key];
  if (choice) { event.preventDefault(); vote(choice); }
});

(async () => {
  busy(true);
  try {
    state.catalog = await api("/api/catalog");
    contentLabels();
    fill("model", state.catalog.models.map((m) => [m.id, `${m.label}${m.groups.length ? "" : " · 준비 중"}`, !m.groups.length]), saved.model);
    populate();
    await loadQueue();
  } catch (error) { message(`불러오지 못했습니다: ${error.message}`, true); }
  finally { busy(false); }
})();
