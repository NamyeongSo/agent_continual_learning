'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
} [c]));
const fmt = v => typeof v === 'number' ? new Intl.NumberFormat('en-US').format(v) : v === true ? 'true' : v === false ? 'false' : v === null ? 'null' : String(v ?? '—');
const pretty = v => JSON.stringify(v, null, 2);
const names = {
  ace: 'ACE AUD',
  premise: 'PEACE AUD w/o alignment',
  skillopt: 'SkillOpt',
  gepa: 'GEPA'
};
const bnames = {
  bfcl: 'BFCL',
  appworld: 'AppWorld',
  terminalbench2: 'Terminal-Bench 2.0'
};
const bshort = {
  bfcl: 'BFCL',
  appworld: 'AppWorld',
  terminalbench2: 'Terminal-Bench'
};
const snames = {
  train: 'Train',
  val: 'Val',
  test: 'Test',
  unused: '미사용'
};
const navnames = {
  'llm-calls': 'LLM 응답 · Reasoning',
  'shared-reflection': '공통 reflector · Fresh 1 epoch',
  'initial-context': '공통 초기 context',
  overview: '실험 한눈에 보기',
  methods: '베이스라인 4종',
  benchmarks: '벤치마크 구성',
  tasks: 'Datapoint 탐색',
  splits: 'Split · Seed · 순서',
  serving: '모델 서빙 · 호출',
  prompts: '프롬프트 전문',
  runs: '실행별 설정',
  tokens: '학습 · 평가 토큰',
  sources: '재현 · 근거 자료'
};
const state = {
  data: null,
  live: null,
  epoch: 1,
  method: 'ace',
  bench: 'bfcl',
  splitBench: 'bfcl',
  taskBench: '',
  taskSplit: '',
  taskQuery: '',
  selectedOnly: 'all',
  page: 1,
  task: null,
  taskTab: 'instruction',
  seed: 42,
  methodForTask: 'ace',
  run: null,
  promptMethod: 'all',
  sourceQuery: ''
};
let catalogSerial = 0;
let taskSerial = 0;
let checkpointSerial = 0;
const api = async url => {
  const r = await fetch(url);
  if (!r.ok) throw Error(`HTTP ${r.status}`);
  return r.json();
};

function pill(text, color = '') {
  return `<span class="pill ${color}">${esc(text)}</span>`;
}

function splitPill(split) {
  return pill(snames[split] || split, {
    train: 'blue',
    val: 'purple',
    test: 'green',
    unused: ''
  } [split]);
}

function icon(m) {
  return `<span class="method-icon ${m}">${{ace:'AC',premise:'PE',skillopt:'SK',gepa:'GE'}[m]}</span>`;
}

function source(title, label) {
  const id = state.data.source_refs[title] || title;
  return `<button class="evidence" data-source="${esc(id)}">${esc(label||title)}</button>`;
}

function code(value, title = '설정 JSON') {
  const text = typeof value === 'string' ? value : pretty(value);
  return `<div class="code-box"><div class="code-header"><span>${esc(title)}</span><button data-copy>복사</button></div><pre>${esc(text)}</pre></div>`;
}

function jsonDetails(value, title = '원시 설정 전체') {
  return `<details class="disclosure"><summary>${esc(title)}</summary><div class="inner">${code(value,title)}</div></details>`;
}

function scope() {
  return `<div class="scope-switch" aria-label="Epoch 선택"><button data-epoch="1" class="${state.epoch===1?'active':''}">Epoch 1</button><button data-epoch="2" class="${state.epoch===2?'active':''}">Epoch 2 continuation</button></div>`;
}

function heading(kicker, title, description, control = '') {
  return `<div class="page-heading"><div><span class="eyebrow">${esc(kicker)}</span><h1>${esc(title)}</h1><p>${description}</p></div>${control}</div>`;
}

function section(title, content, description = '', action = '') {
  return `<section class="section"><div class="section-header"><div><h2>${esc(title)}</h2>${description?`<p>${description}</p>`:''}</div>${action}</div>${content}</section>`;
}

function table(headers, rows, cls = '') {
  return `<div class="table-wrap"><table class="${cls}"><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c??'—'}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function params(rows) {
  return table(['파라미터', '설정값', '적용 범위 · 근거'], rows.map(([k, v, d]) => [esc(k), esc(typeof v === 'object' ? pretty(v) : fmt(v)), d]), 'param-table');
}

function facts(items) {
  return `<div class="panel facts">${items.map(([k,v,d])=>`<div class="fact"><label>${esc(k)}</label><strong>${esc(v)}</strong><p>${d}</p></div>`).join('')}</div>`;
}

function note(title, text) {
  return `<div class="audit-note"><span class="note-mark">◷</span><div><strong>${esc(title)}</strong><br>${text}</div></div>`;
}

function tabbar(items, selected, attribute) {
  return `<div class="tabs">${items.map(([id,label])=>`<button ${attribute}="${esc(id)}" class="${selected===id?'active':''}">${esc(label)}</button>`).join('')}</div>`;
}

function runFor(method = state.method, bench = state.bench, seed = 42) {
  return state.data.runs.find(r => r.method === method && r.benchmark === bench && r.seed === Number(seed));
}

function configFor(method = state.method, bench = state.bench, seed = 42) {
  const r = runFor(method, bench, seed);
  return r.epochs[String(state.epoch)].config || r.epochs['1'].config;
}

function splitbar(bench) {
  const s = state.data.benchmark_stats[bench];
  return `<div class="split-bar">${['train','val','test','unused'].map(k=>`<span class="${k}" title="${snames[k]} ${s.counts[k]||0}" style="width:${100*(s.counts[k]||0)/s.pool}%"></span>`).join('')}</div><div class="split-legend">${['train','val','test','unused'].filter(k=>s.counts[k]).map(k=>`<span><i class="dot ${k}"></i>${snames[k]} ${s.counts[k]}</span>`).join('')}</div>`;
}

function methodTabs() {
  return tabbar(Object.entries(names), state.method, 'data-method');
}

function benchSelect() {
  return `<div class="inline-controls"><select id="method-benchmark" aria-label="벤치마크">${Object.entries(bshort).map(([k,v])=>`<option value="${k}" ${state.bench===k?'selected':''}>${v}</option>`).join('')}</select></div>`;
}
const methodContent = {
  ace: {
    short: '성공·실패를 반영하며 playbook을 누적 수정',
    unit: 'Playbook bullets',
    validation: '학습 trajectory의 공식 feedback',
    selection: '매 task 후 curator 적용',
    steps: [
      ['Generator', '현재 playbook으로 환경 실행', '현재 task와 환경 policy, 누적 playbook을 actor에게 전달합니다. 이 실행은 benchmark별 환경 turn 제한과 입력·출력 토큰 제한을 따릅니다.', 'ACE actor'],
      ['Reflector', '종료한 trajectory + 공식 결과', 'training_time=true입니다. 실행이 끝난 trajectory와 공식 training feedback으로 실패 원인·성공 패턴을 작성하고 사용한 bullet을 helpful/harmful/neutral로 평가합니다. 반영 전 근거는 task의 실행 결과입니다.', 'ACE actor'],
      ['Curator', 'ADD · UPDATE · DELETE', '매 1개 세션마다 reflector 결과와 기존 playbook을 입력합니다. 새 지식을 추가하고 기존 bullet을 수정하거나 삭제합니다. JSON operations를 실제 playbook 편집 코드에 적용합니다.', 'ACE actor'],
      ['Next task', '수정한 playbook으로 진행', 'isolated 모드이므로 벤치마크·seed별 playbook을 저장합니다. 다음 task에 누적한 playbook을 제공합니다. 최종 평가에서는 이 policy를 고정하고 reflection·curation을 실행하지 않습니다.', 'ACE defaults']
    ]
  },
  premise: {
    short: '두 parent에서 네 후보를 만들고 현재·과거 성능으로 선택',
    unit: 'Beam 2 + replay memory',
    validation: '현재 1개 + 과거 최대 3개 실제 replay',
    selection: '공식 점수의 current/past Pareto',
    steps: [
      ['Parent ×2', '두 playbook으로 현재 task 실행', '서로 독립된 두 beam parent가 현재 task를 각각 수행합니다. 첫 parent의 pre-update trajectory가 Online 보고의 기준입니다. parent worker 수는 2입니다.', 'PEACE pipeline'],
      ['Reflection ×4', 'Parent마다 reflection 2개', 'parent별로 두 개의 reflection을 만들고 성공·실패의 결정적 action span을 선택합니다. reflection worker 수는 4입니다. 각 reflection은 현재 task와 기록된 observations를 근거로 삼습니다.', 'PEACE pipeline'],
      ['Mutation ×4', '부모 policy로부터 후보 4개', '각 reflection으로 parent playbook에 ADD·UPDATE·DELETE를 적용해 후보를 만듭니다. mutation worker 수는 4입니다. alignment judge는 이 variant에서 비활성화되어 있습니다.', 'PEACE pipeline'],
      ['Replay', 'Current 1 + Past ≤3', '모든 후보가 같은 current/past 표본에서 실제 환경 replay를 수행합니다. 선택한 span 이전의 action prefix를 복원한 뒤 해당 후보 policy로 continuation을 생성합니다. 최대 4 × 4 = 16개의 후보 replay 작업이 생기고 replay worker 한도는 28입니다.', 'PEACE pipeline'],
      ['Pareto → Beam 2', '현재·과거 공식 점수 최대화', '두 목적은 current 공식 점수와 past 평균 공식 점수입니다. Pareto front를 순서대로 선택한 뒤 front 안에서는 점수 합, 최소 점수, 후보 번호 순으로 정렬합니다. 유효 후보가 부족할 때는 부모 beam을 보존합니다.', 'PEACE selection']
    ]
  },
  skillopt: {
    short: '실패·성공 패치와 validation gate로 skill을 최적화',
    unit: 'Skill markdown',
    validation: '고정 Val 전체의 hard metric',
    selection: 'Gate + best skill',
    steps: [
      ['Rollout', 'Minibatch 5 · accumulation 1', '공식 ReflACTTrainer와 Exgentic EnvAdapter를 연결합니다. 학습 stream을 task 5개씩 한 번씩 소비합니다. BFCL·AppWorld는 epoch당 10 minibatch, Terminal-Bench는 6 minibatch입니다.', 'SkillOpt runtime'],
      ['Analyst', '실패·성공 trajectory 분석', 'failure_only=false입니다. 성공과 실패 양쪽에서 수정 patch를 제안합니다. analyst worker는 50, analyst minibatch는 5입니다.', 'SkillOpt runtime'],
      ['Merge · Rank', '병합 후 편집 예산 적용', 'merge_batch_size=8로 patch들을 병합하고 ranking을 거쳐 적용할 edit을 고릅니다. 초기 edit_budget=4, min_edit_budget=2이며 cosine schedule을 사용합니다. lr_control_mode는 fixed입니다.', 'SkillOpt runtime'],
      ['Update · Gate', 'Patch 적용과 Val 평가', 'skill_update_mode=patch, use_gate=true, gate_metric=hard입니다. 고정 Val의 score로 gate와 best skill을 선택합니다. 학습 reflection에는 validation trajectory를 전달하지 않습니다.', 'SkillOpt runtime'],
      ['Meta-skill', 'Epoch 경계의 meta update', 'use_meta_skill=true입니다. Epoch 2 continuation에서는 후반 epoch의 train pool에서 sample을 골라 현재·이전 skill rollout을 비교합니다. 최종 Test에는 Val로 고른 best_skill.md를 고정해 사용합니다.', 'Epoch 2 continuation']
    ]
  },
  gepa: {
    short: 'trajectory 기반 지시문 제안과 validation Pareto 선택',
    unit: 'Shared instruction string',
    validation: '고정 Val 전체의 공식 hard score',
    selection: 'Pareto candidate selection',
    steps: [
      ['Current rollout', 'Ordered minibatch 5', '매 단계마다 현재 5개 task만 loader에 공개합니다. capture_traces가 켜진 train rollout을 reflection에 제공합니다. optimizer seed는 42로 고정하고 task 순서는 별도의 data-order seed를 따릅니다.', 'GEPA runtime'],
      ['Instruction proposal', '현재 instruction + trajectory + feedback', '공식 GEPA adapter의 propose_new_texts에서 instruction_proposal prompt를 호출합니다. 같은 Qwen 모델을 쓰며 분석 출력 예산은 8,192 토큰입니다. perfect score=1.0이면 제안을 생략할 수 있습니다.', 'GEPA runtime'],
      ['Candidate rollout', '같은 minibatch에서 후보 실행', '새 instruction과 기존 instruction의 task 결과를 비교합니다. evaluation label과 policy hash를 포함한 캐시를 사용해 optimizer 재시작 시 같은 결과를 재사용합니다.', 'GEPA runtime'],
      ['Val · Pareto', '고정 Val로 후보 population 선택', 'candidate_selection_strategy=pareto, use_merge=false입니다. Val 결과는 점수로만 제공하고 reflection에는 train trajectory를 사용합니다. metric 호출의 누적 cap은 없습니다.', 'GEPA runtime'],
      ['Freeze best', '단일 stream 종료 후 Test', '고정 stream을 모두 소비하면 stop callback이 종료합니다. Val로 고른 best candidate의 instruction을 동결하고 final Test를 실행합니다.', 'GEPA runtime']
    ]
  }
};

function initialContext() {
  const c = state.data.next_actor_context;
  if (!c) return heading('SHARED INITIAL CONTEXT', '공통 초기 context', '자료 생성 후 표시됩니다.');
  return heading('NEW TRAINING PROTOCOL / ' + c.version.toUpperCase(), '같은 초기 context에서 학습을 시작합니다',
    '새 학습 CLI의 기본값입니다. 이 페이지의 기존 36개 실행은 당시 frozen 설정으로 표시하며, 공통 context 비교는 fresh Epoch 1부터 시작합니다.') +
    facts([['대상 방법','ACE · GEPA · SkillOpt · PEACE','네 방법에 같은 actor 입력 구조'],['초기 학습 정책','빈 문자열','ACE·PEACE의 빈 section도 같은 문구로 렌더링'],['Protocol',c.version,'--actor-context shared-empty-v1']]) +
    section('공통 입력의 조립', `<div class="panel panel-body"><div class="numbers"><span class="number-chip">공통 actor 지침</span> → <span class="number-chip">benchmark 공통 지침</span> → <span class="number-chip">학습 정책 영역</span> → <span class="number-chip">task · 환경 context · tools</span></div><p class="caption">학습 정책 영역은 처음에 (No learned guidance yet.)로 동일하게 표시합니다. 학습 이후에는 각 방법이 만든 정책 원문을 같은 영역에 삽입합니다.</p></div>`) +
    section('통일하는 설정', `<div class="panel">${params([
      ['초기 learned policy','empty','GEPA seed candidate와 SkillOpt initial skill은 빈 문자열'],
      ['ACE / PEACE 내부 초기 상태','빈 native playbook','section 구조는 유지하고 actor에는 빈 정책으로 표시'],
      ['actor 지침과 benchmark 지침','공통 renderer','방법별 base prompt와 초기 seed 전략을 공통 입력으로 대체'],
      ['tool shortlisting',false,'같은 task의 모든 public tool schema 사용'],
      ['enable_thinking',false,'공통 모드에서 활성화 요청을 거부'],
      ['optimizer / reflection prompt','각 방법의 구현 유지','정책 업데이트와 후보 선정 절차를 유지'],
      ['고정 평가와 replay','동일 context version 전달','다른 version·renderer의 continuation 및 cache 재사용을 거부']
    ])}</div>`) +
    section('Benchmark별 초기 actor prompt 전문', `<div class="panel">${Object.entries(c.initial_prompts).map(([bench,text])=>`<details class="prompt-item"><summary><strong>${bnames[bench]}</strong><span>네 방법 공통 · 학습 전</span></summary>${code(text,'Initial actor prompt')}<div class="caption-box">SHA-256 ${esc(c.metadata.initial_actor_prompt_sha256[bench])}</div></details>`).join('')}</div>`, 'AppWorld는 제공된 환경 policy의 실행 인터페이스를 따릅니다. Terminal-Bench는 native Terminus protocol에 동일 지침을 붙입니다.') +
    (c.preflight?.validation?.final_core_regression ? section('오프라인 검증 결과', facts([
      ['Core 회귀 테스트',c.preflight.validation.final_core_regression.passed + ' passed',c.preflight.validation.final_core_regression.skipped + ' skipped'],
      ['Native Terminal-Bench',c.preflight.validation.native_terminalbench_venv.passed + ' passed','전용 venv에서 adapter · budget · replay 검증'],
      ['새 실행 설정',c.preflight.configurations + '개 일치','4 methods × 3 benchmarks · 초기 context hash 동일']
    ]), '모델 추론과 실제 benchmark 환경 실행 없이 구현을 검증했습니다.') : '') +
    section('검증과 재현', `<div class="grid-2"><div class="panel panel-body">${code(c.metadata,'공통 context metadata')}${source(c.source,'공통 renderer 원문과 hash')}</div><div class="panel panel-body text-block"><h3>첫 요청을 비교하는 검증</h3><p>실제 네 actor의 completion 직전 messages·tools·tool_choice를 가로채 비교합니다. 일반 observation과 image observation, 학습 후 자유 형식 정책 및 native Terminal-Bench instruction을 검사합니다.</p><p>GEPA·SkillOpt 공식 optimizer에서는 scripted transport로 학습·Val 선택·Test·복원을 검증합니다. 모델 응답과 환경 실행을 사용하는 성능 실험은 별도입니다.</p>${c.preflight?code(c.preflight,'4 methods × 3 benchmarks · dry-run 설정 검증'):''}<h3>Fresh 실행과 호환성</h3><p>기존 checkpoint에 새 context를 붙여 이어 학습하면 초기 조건이 맞지 않습니다. 다른 context와 renderer hash의 재개를 거부합니다. 과거 동작은 --actor-context legacy로 선택할 수 있습니다.</p><p>모델·temperature·budget·split·환경 policy는 실행 설정에서 별도로 맞춥니다.</p></div></div>`);
}

function overview() {
  const d = state.data,
    e = state.epoch;
  return heading('PROTOCOL / QWEN3.6', '실험의 전체 구조', `네 가지 방법론이 같은 task membership과 actor 예산을 공유합니다. 이 페이지는 frozen code와 실행 설정을 연결해, 각 설정이 어디에 적용되는지 보여줍니다.`, scope()) +
    `<div class="metric-grid">${[['4','방법론','ACE · PEACE · SkillOpt · GEPA'],['3','벤치마크','BFCL · AppWorld · Terminal-Bench'],['389','사용 datapoint','원본 706개 중 train / val / test'],['36','독립 실행','4 methods × 3 benchmarks × 3 seeds']].map(([v,l,t])=>`<div class="metric-card"><span class="label">${l}</span><strong>${v}</strong><p>${t}</p></div>`).join('')}</div>` +
    facts([
      ['모델', 'Qwen3.6-35B-A3B', 'Actor와 분석에 같은 모델 사용'],
      ['역할별 temperature', '0.7', '기준 cohort의 전 역할 설정'],
      ['실행 모드', 'Isolated', 'benchmark·seed마다 policy를 독립 저장']
    ]) +
    section('네 방법론의 학습 방식', `<div class="panel">${table(['방법론','누적하는 정책','학습 중 검증','다음 정책 선택'],Object.entries(names).map(([m,n])=>[`<a class="method-title" href="#methods/${m}">${icon(m)}<strong>${n}</strong></a>`,esc(methodContent[m].unit),esc(methodContent[m].validation),esc(methodContent[m].selection)]))}</div>`, '같은 actor 예산 위에서 정책을 업데이트하고 선택하는 방식이 다릅니다.', `<a class="subtle-link" href="#methods">단계별 설정 보기 ↗</a>`) +
    `<div class="grid-2">${section('공유 actor 예산',`<div class="panel">${params([['input / environment turn','100,000 tokens','선택·acting·계측된 재시도 입력의 합'],['output / environment turn','2,048 tokens','선택·acting·계측된 재시도 출력의 합'],['trajectory turns','BFCL 30 / AppWorld·TB 40','Replay prefix도 turn에 포함'],['analysis output','8,192 tokens','Reflection·mutation 등은 독립 예산']])}<div class="caption-box">예산 초과나 출력 truncation은 해당 trajectory의 score 0 / success=false로 기록합니다. ${source('Actor budgets')}</div></div>`)}${section('데이터와 split',`<div class="panel panel-body">${Object.entries(bshort).map(([b,n])=>`<div style="margin-bottom:22px"><div class="benchmark-title"><span class="benchmark-symbol">${{bfcl:'B',appworld:'A',terminalbench2:'T'}[b]}</span>${n}<span class="muted" style="font-size:10px;font-weight:400;margin-left:auto">Pool ${d.benchmark_stats[b].pool}</span></div>${splitbar(b)}</div>`).join('')}<a href="#splits" class="caption">선정 규칙과 seed별 순서 확인 ↗</a></div>`)}</div>` +
    section('동시성과 실행 격리', `<div class="panel">${params([
      ['Epoch 2 experiment scheduler', d.protocol.max_parallel, '한 benchmark에서 최대 12개 독립 실행'],
      ['benchmark barrier', d.protocol.benchmark_barrier, '12개 실행의 학습과 최종 Test 완료 후 다음 benchmark'],
      ['seed stagger', d.protocol.seed_stagger_seconds + ' seconds', '같은 benchmark의 seed 시작 간격'],
      ['최종 Test workers', 50, '독립 실행 하나의 task worker 한도'],
      ['PEACE workers', 'parent 2 · reflection 4 · mutation 4 · replay 28', '각 실행 내부의 단계별 worker 한도'],
      ['PEACE LLM semaphore', 32, '해당 실행 내에서 공유하는 LLM 요청 한도'],
      ['GEPA / SkillOpt workers', 50, '각 optimizer 실행 내부의 rollout / analysis 한도'],
      ['vLLM max_num_seqs', 128, '서버의 sequence scheduling 한도; 실험 수와 별도']
    ])}<div class="caption-box">각 실행은 state·출력 경로를 분리합니다. 여러 실행이 동일한 모델 endpoint를 공유하며, 위 수치들은 서로 다른 계층의 동시성 한도입니다. ${source('Epoch 2 continuation')}</div></div>`) +
    section(`Epoch ${e}의 진행 방식`, e === 1 ? `<div class="panel panel-body text-block"><h3>각 benchmark의 train stream을 한 번씩 학습</h3><p>BFCL 50개, AppWorld 50개, Terminal-Bench 30개를 seed별 순서로 학습합니다. Val을 쓰는 GEPA·SkillOpt는 고정 Val로 정책을 선택하고, 모든 방법이 최종 policy를 고정해 Test를 수행합니다.</p><div class="source-ref-row">${source('Epoch 2 continuation','Epoch 1 출처와 복원 상태')}${source('Split order generator')}</div></div>` : `<div class="panel panel-body text-block"><h3>Epoch 1의 학습 상태를 이어서 한 epoch 추가</h3><p>Epoch 2 순서는 실제 Epoch 1 순서의 복사본에 Random(seed).shuffle을 적용합니다. BFCL·AppWorld는 누적 100 encounters, Terminal-Bench는 누적 60 encounters를 목표로 합니다. 한 benchmark의 12개 실행이 학습·Test를 마치면 다음 benchmark로 이동합니다.</p><span class="formula">BFCL 12 runs → AppWorld 12 runs → Terminal-Bench 12 runs</span><div class="source-ref-row">${source('Epoch 2 continuation')}</div></div>`) +
    note('실행 기록에서 확인된 차이', 'AppWorld ACE의 Epoch 2 고정 Train/Val 요청은 temperature 0.0이 기록됐습니다. 기준 학습 설정은 0.7입니다. 평가 시작 오류와 해당 요청의 근거는 <a href="#serving">모델 호출</a>에서 볼 수 있습니다.');
}

function methodParameters(m) {
  const c = configFor(m),
    bench = state.bench,
    sourceKey = {
      ace: 'ACE defaults',
      premise: 'PEACE pipeline',
      skillopt: 'SkillOpt runtime',
      gepa: 'GEPA runtime'
    } [m];
  const s = source(sourceKey, 'Frozen 구현');
  const common = [
    ['temperature', c.temperature_all_roles ?? c.temperature ?? 0.7, 'Actor·분석 공통'],
    ['actor output / turn', 2048, 'actor-turn-v1 최종 강제 상한'],
    ['actor max turns', bench === 'bfcl' ? 30 : 40, `${bshort[bench]} trajectory`],
    ['analysis max_tokens', 8192, 'Actor와 독립된 출력 예산']
  ];
  const additional = {
    ace: [
      ['playbook_edit_mode', 'add_update_delete', 'ADD / UPDATE / DELETE'],
      ['training_time', true, '학습 종료 feedback을 reflection에 제공'],
      ['evaluation_mode', false, '학습 실행의 설정'],
      ['curator_frequency', 1, '매 세션마다 curation'],
      ['playbook_token_budget', 80000, '누적 playbook 예산'],
      ['use_json_mode', true, 'Reflector·curator JSON 출력'],
      ['use_bulletpoint_analyzer', false, '유사 bullet 자동 병합 비활성'],
      ['actor retry attempts', 3, '1초 / 2초 backoff; context 오류 종료'],
      ['simple analysis attempts', 3, 'Reflection·curation 호출'],
      ['num_retries at budget wrapper', 0, 'SDK의 계측되지 않는 내부 재시도 억제']
    ],
    premise: [
      ['beam size', 2, 'Parent 2개를 보존'],
      ['reflections / parent', 2, '총 reflection 4개'],
      ['candidate count', 4, '총 mutation 후보 4개'],
      ['past_count', c.past_count ?? 3, 'Memory의 과거 task 최대 3개'],
      ['current_count', 1, '현재 task 하나'],
      ['candidate validation', 'environment_replay', '실제 환경과 replay'],
      ['alignment_scoring', false, 'current/past 공식 점수 Pareto'],
      ['parent_workers', 2, '동시 parent 환경 실행'],
      ['reflection_workers', 4, 'Reflection 생성'],
      ['mutation_workers', 4, 'Mutation 생성'],
      ['replay_workers', 28, '후보 replay 동시 실행 한도'],
      ['llm_calls', 32, '실행 내 공유 LLM semaphore'],
      ['replay_retries', c.replay_retries ?? 2, '초기 실행 후 최대 2회 재시도'],
      ['llm timeout', 600, '초'],
      ['llm SDK retries', 1, 'PremiseLLM client 설정'],
      ['algorithm_seed', c.algorithm_seed ?? 42, 'data-order seed와 별도'],
      ['playbook_edit_mode', 'add_update_delete', 'ADD / UPDATE / DELETE']
    ],
    skillopt: [
      ['batch_size', 5, '순서가 고정된 train minibatch'],
      ['accumulation', 1, '매 minibatch 갱신'],
      ['minibatch_size', 5, 'Analyst minibatch'],
      ['analyst_workers', 50, '동시 analyst 호출 한도'],
      ['merge_batch_size', 8, '제안 patch 병합'],
      ['edit_budget', 4, '초기 편집 예산'],
      ['min_edit_budget', 2, '최소 편집 예산'],
      ['lr_scheduler', 'cosine', 'Native scheduler'],
      ['lr_control_mode', 'fixed', '고정 제어 모드'],
      ['skill_update_mode', 'patch', 'Skill patch 적용'],
      ['failure_only', false, '성공·실패 모두 분석'],
      ['use_gate', true, 'Val로 업데이트 gate'],
      ['gate_metric', 'hard', '공식 binary metric'],
      ['use_meta_skill', true, 'Meta skill 활성화'],
      ['use_semantic_density', false, '기능 플래그'],
      ['use_slow_update', false, '기능 플래그'],
      ['optimizer_seed', 42, '모든 data-order seed에서 고정'],
      ['rollout_workers', 50, 'Native Runtime worker 수'],
      ['actor worker attempts', 3, '인프라 오류의 유한 재시도'],
      ['analysis attempts', 3, 'Response가 남으면 usage 집계'],
      ['cumulative training cap', 'unlimited', '고정 stream 소진 시 종료']
    ],
    gepa: [
      ['engine', 'gepa.optimize', '공식 external-system API'],
      ['minibatch_size', 5, 'OrderedSampler'],
      ['candidate_selection_strategy', 'pareto', 'Native candidate population'],
      ['skip_perfect_score', true, 'Perfect-score minibatch 제안 생략'],
      ['perfect_score', 1.0, '공식 hard metric'],
      ['use_merge', false, 'Candidate merge 비활성'],
      ['max_metric_calls', null, '누적 metric 호출 cap 없음'],
      ['cache_evaluation', false, 'GEPA engine flag'],
      ['Runtime rollout cache', true, 'Item + learned policy SHA + label'],
      ['optimizer_seed', 42, '모든 data-order seed에서 고정'],
      ['workers', 50, 'Rollout·teacher 요청 한도'],
      ['actor worker attempts', 3, '인프라 오류 유한 재시도'],
      ['analysis attempts', 3, '분석 응답 재시도'],
      ['stop callback', 'state.i + 1 ≥ ceil(N / 5)', '단일 stream 종료']
    ]
  };
  return common.concat(additional[m]).map((r, i) => [r[0], r[1], r[2] + (i === common.length ? ` · ${s}` : '')]);
}

function methods() {
  const m = state.method,
    c = methodContent[m],
    run = runFor(),
    cfg = configFor();
  return heading('METHODS / IMPLEMENTATION', names[m], c.short, scope()) +
    `<div class="panel">${methodTabs()}<div class="method-summary"><h3>${c.unit}</h3>${esc(c.validation)}를 기준으로 정책을 갱신합니다.<div class="numbers"><span class="number-chip">data-order seeds 42 / 43 / 44</span><span class="number-chip">isolated per benchmark</span><span class="number-chip">${state.epoch===1?'초기 정책부터 학습':'Epoch 1 state에서 continuation'}</span></div></div><div class="pipeline">${c.steps.map(([title,sub],i)=>`${i?'<span class="flow-arrow">→</span>':''}<button class="flow-step ${i===0?'selected':''}" data-step="${i}"><strong>${esc(title)}</strong><span>${esc(sub)}</span></button>`).join('')}</div><div class="explanation" id="flow-explanation"><b>${esc(c.steps[0][0])}</b> ${esc(c.steps[0][2])}<div class="source-ref-row">${source(c.steps[0][3])}</div></div></div>` +
    section('하이퍼파라미터', `<div class="panel">${params(methodParameters(m))}</div>`, '수치가 직접 기록된 config와 frozen 구현의 기본값을 함께 확인합니다.', benchSelect()) +
    (m === 'premise' ? section('Current–Past Pareto와 replay의 의미', `<div class="grid-2"><div class="panel panel-body text-block"><h3>평가 축</h3><span class="formula">candidate → (current official score, mean(past official score))</span><p>Past 표본은 해당 실행의 memory에서 현재 task ID를 제외하고 비복원 추출합니다. memory가 3개보다 적으면 가능한 수만 사용합니다. 같은 표본을 네 후보가 공유합니다.</p><p>같은 front 안에서는 공식 점수 합의 내림차순, 최소 점수 내림차순, 후보 번호 오름차순으로 두 정책을 선택합니다.</p>${source('PEACE selection')}</div><div class="panel panel-body text-block"><h3>span의 앞부분을 복원하는 실제 환경 replay</h3><p>기록한 span.start 앞의 action을 같은 환경에서 다시 실행한 뒤, 후보 playbook으로 후속 action을 생성합니다. 각 replay의 prefix와 continuation은 trajectory turn 예산에 함께 들어갑니다.</p><p>이 cohort는 과거 3개를 쓰는 실제 환경 replay variant입니다. Alignment LLM judge 호출은 생성되지 않습니다.</p>${source('PEACE pipeline')}</div></div>`) : '') +
    (m === 'skillopt' ? section('Native trainer의 전체 설정', `<div class="panel panel-body">${code(state.data.skillopt_native[state.bench][state.epoch===1?'epoch1':'epoch2']||state.data.skillopt_native[state.bench].epoch1,'SkillOpt effective config')}<p class="caption">비활성 backend의 default 필드도 원시 config에 남아 있습니다. 위 하이퍼파라미터 표에는 이 실행에서 사용하는 요소를 표시했습니다.</p>${source(state.data.skillopt_native[state.bench].source,'저장된 native config')}</div>`) : '') +
    section('실행 config와 프롬프트', `<div class="grid-2"><div class="panel panel-body"><h3 class="panel-title">${bshort[state.bench]} · seed 42 · Epoch ${state.epoch}</h3>${code(cfg,'Persisted experiment config')}${run.epochs[String(state.epoch)].source?source(run.epochs[String(state.epoch)].source,'원본 config와 SHA-256'):'<p class="caption">Epoch 2 실행 전입니다. 원본 Epoch 1 config와 계획된 continuation command를 함께 표시합니다.</p>'}</div><div class="panel panel-body text-block"><h3>입력 구성</h3><ol class="step-list"><li><div><strong>환경 policy와 task</strong><br>Benchmark가 현재 instruction과 허용된 action schema를 제공합니다.</div></li><li><div><strong>${esc(c.unit)}</strong><br>이전 train task에서 학습한 정책을 actor prompt에 삽입합니다.</div></li><li><div><strong>실행 history</strong><br>현재 trajectory의 action과 observation을 모델 요청에 포함합니다.</div></li><li><div><strong>학습 후 analysis</strong><br>${esc(c.validation)}에서 얻은 자료로 정책 갱신을 시도합니다.</div></li></ol><a class="button" href="#prompts/${m}">이 방법의 프롬프트 전문 ↗</a></div></div>`) +
    (state.epoch === 2 ? section('Epoch 2의 상태 복원', `<div class="panel panel-body text-block"><h3>${m==='ace'?'Playbook와 online metrics':m==='premise'?'Beam·memory·RNG':m==='gepa'?'Optimizer population·RNG를 cached decisions로 재구성':'Native checkpoint와 step buffer'}를 이어서 사용</h3><p>${{ace:'완료된 task의 playbook checkpoint와 online_metrics를 복원하고 남은 task만 실행합니다. 새 run ID를 사용해 중단된 세션의 업데이트가 잘못 재사용되지 않도록 합니다.',premise:'Epoch 1 state의 beam, memory, RNG를 복원합니다. 학습 순서만 두 epoch의 연속 order로 확장합니다. 같은 benchmark·seed의 state를 이어갑니다.',gepa:'Epoch 1의 rollout·teacher cache를 읽으며 optimizer 상태를 재구성합니다. cache miss가 나면 새로 Epoch 1 모델 추론을 실행하지 않고 복원 오류로 종료합니다. 후보 population과 Val subscores가 원본 checkpoint와 같은지 검사합니다.',skillopt:'Native epoch boundary state의 current_step을 유지하고 총 scheduler horizon을 2 epochs로 확장합니다. Epoch 2 buffer는 새 epoch의 규칙에 따라 시작하고, epoch 중간 재개에서는 digest로 이미 완료한 step buffer를 복원합니다.'}[m]}</p>${source('Epoch 2 continuation')}</div>`) : '');
}

function benchmarkCard(b) {
  const s = state.data.benchmark_stats[b];
  return `<div class="panel panel-body"><div class="benchmark-title"><span class="benchmark-symbol">${b[0].toUpperCase()}</span>${esc(s.name)}<span class="pill" style="margin-left:auto">Pool ${s.pool}</span></div>${splitbar(b)}<p class="caption">${{bfcl:'4개 주요 domain × 각 50개. Multi-turn user instruction과 도구 state를 평가합니다.',appworld:'139개 시나리오 × 3개 변형. 50개 시나리오의 변형을 split별로 하나씩 배정합니다.',terminalbench2:'89개 task 전체. Category와 difficulty를 함께 맞춰 분할합니다.'}[b]}</p><a class="caption" href="#tasks/${b}">Instruction 탐색 ↗</a></div>`;
}

function benchmarks() {
  const vals = state.data.manifest.sources;
  return heading('BENCHMARKS / DATA', '벤치마크와 평가 구성', '각 benchmark의 interaction 방식, 공식 성공 기준과 데이터 구성을 확인할 수 있습니다.') +
    `<div class="grid-3">${Object.keys(bnames).map(benchmarkCard).join('')}</div>` +
    section('환경·action·metric', `<div class="panel">${table(['벤치마크','모델이 수행하는 작업','종료와 공식 metric','데이터 크기'],[
 ['<strong>BFCL · multi_turn_base</strong>','Function call로 시뮬레이션된 API state를 변경합니다. user turn은 finish 이후 순서대로 공개됩니다.','공식 multi-turn checker의 binary success. finish는 현재 user turn을 종료합니다.','Train 50 · Val 50 · Test 50'],
 ['<strong>AppWorld · test_challenge</strong>','Original environment policy + code_generation. 애플리케이션 API를 Python code로 호출합니다.','TGC = official test_tracker.success. 전체 task 성공 1 / 그 외 0. Partial test pass rate는 별도 진단값입니다.','Train 50 · Val 50 · Test 50'],
 ['<strong>Terminal-Bench 2.0</strong>','Native Harbor Terminus 2가 task container의 terminal을 조작합니다.','Official Harbor verifier reward. 정상 verifier 실행 여부와 task reward를 구분해 기록합니다.','Train 30 · Val 30 · Test 29']
 ])}</div>`) +
    section('AppWorld의 policy와 split 구조', `<div class="grid-2"><div class="panel panel-body text-block"><h3>Original environment policy를 고정</h3><p>Training, replay, checkpoint와 최종 평가에 같은 policy bytes와 SHA-256을 사용합니다. 이 환경 policy는 각 방법이 학습하는 playbook·instruction·skill과 별도로 제공됩니다.</p><div class="hash-line">${esc(state.data.policy.source&&state.data.runs.find(r=>r.policy_sha256)?.policy_sha256)}</div><details class="disclosure"><summary>Environment policy 전문</summary><div class="inner">${code(state.data.policy.text,'AppWorld Original policy')}</div></details></div><div class="panel panel-body text-block"><h3>같은 scenario, 서로 다른 variant</h3><p>Train·Val·Test 각각의 task ID는 겹치지 않습니다. 세 split은 같은 50개 scenario를 공유하고, 각 scenario의 variant 1·2·3을 서로 다른 split에 하나씩 배정합니다. task ID의 suffix는 variant 번호입니다.</p><p>난이도 1 / 2 / 3의 scenario 배정은 split마다 9 / 18 / 23개입니다. Variant suffix는 난이도와 별개입니다.</p><a href="#splits" class="caption">구성 비율과 membership 확인 ↗</a></div></div>`) +
    section('Task별 환경 제한', `<div class="panel panel-body text-block"><p>Actor turn 상한은 BFCL 30, AppWorld·Terminal-Bench 40입니다. AppWorld adapter의 기본 max_interactions=200과 별개로 actor-turn-v1의 40 turn 상한이 적용됩니다.</p><p>Terminal-Bench의 agent timeout, verifier timeout, environment build timeout, CPU·memory·storage는 각 task.toml에 정의되어 있습니다. 개별 datapoint의 환경 탭에서 정확한 값을 확인할 수 있습니다.</p><p>AppWorld TGC는 공식 outcome이 기록된 task를 분모로 사용하고, outcome 누락과 인프라 오류는 분리합니다. Budget 초과와 출력 truncation은 해당 trajectory의 공식 실패로 기록합니다.</p>${source('Actor budgets')}<div class="source-ref-row">${source('BFCL environment')}${source('AppWorld environment')}${source('Terminal-Bench environment')}${source('Terminal-Bench budgets')}</div></div>`) +
    section('Pinned 데이터와 upstream', `<div class="panel">${table(['구성요소','버전 또는 고정 revision','출처'],[
 ['BFCL dataset','BFCL_v4_multi_turn_base.json · 200 tasks',`SHA-256 ${esc(vals.bfcl.sha256)}`],
 ['AppWorld','edc960129fa6889c2b381715ecd108982029f6d1','test_challenge · 417 tasks'],
 ['Terminal-Bench 2.0','2fd12b88aafdd04a52c298e3940bcb189f9766d6','89 tasks · 16 categories'],
 ...Object.entries(state.data.upstreams.sources).map(([k,v])=>[esc(k),esc(v.version||'source pin'),`${esc(v.commit)}<small>${esc(v.url)}</small>`])
 ])}</div>`);
}

function distribution(bench, field) {
  const data = state.data.benchmark_stats[bench].distributions[field];
  if (!data) return '';
  const labels = [...new Set(Object.values(data).flatMap(d => Object.keys(d)))].sort();
  return `<div class="panel">${table([field,'Train','Val','Test','미사용'],labels.map(k=>[esc(k),...['train','val','test','unused'].map(s=>fmt(data[s]?.[k]||0))]),'split-matrix')}</div>`;
}

function splits() {
  const b = state.splitBench,
    d = state.data,
    info = d.benchmark_stats[b],
    seed = state.seed,
    m = state.methodForTask,
    run = runFor(m, b, seed);
  return heading('PARTITIONS / ORDER', 'Split은 고정하고, 순서만 바꿉니다', 'Membership seed와 실행 순서를 구분합니다. 같은 task ID 목록을 모든 방법과 seeds 42–44가 공유합니다.') +
    facts([
      ['Membership seed', '42', '고정된 균형 분할 생성'],
      ['Data-order seeds', '42 / 43 / 44', '각 split 안에서 task 순서만 변경'],
      ['Optimizer seed', '42', 'GEPA·SkillOpt 및 PEACE algorithm RNG']
    ]) +
    section('Membership 생성 규칙', `<div class="grid-3"><div class="panel panel-body text-block"><h3>BFCL · Domain balance</h3><p>GorillaFileSystem, VehicleControlAPI, TradingBot, TravelAPI 각 domain의 50개 task에서 train/val/test를 할당합니다.</p><p>Train 13·13·12·12<br>Val 12·13·13·12<br>Test 12·12·13·13</p>${source('Split membership generator')}</div><div class="panel panel-body text-block"><h3>AppWorld · Scenario / Variant</h3><p>139개 scenario에서 난이도 비율을 맞춰 50개를 선정합니다. 동일 scenario의 세 variant를 서로 다른 split에 배정합니다.</p><p>Difficulty 1 / 2 / 3 → 9 / 18 / 23개. Variant 할당은 cyclic counterbalance를 사용합니다.</p>${source('Split membership generator')}</div><div class="panel panel-body text-block"><h3>Terminal-Bench · MILP</h3><p>HiGHS MILP로 89개 task를 30 / 30 / 29로 배정합니다. Category별 개수 차이는 최대 1입니다.</p><p>Easy·Medium·Hard 목표<br>Train 1 / 19 / 10<br>Val 2 / 18 / 10<br>Test 1 / 18 / 10</p>${source('Split membership generator')}</div></div>`) +
    section('실제 분포', `<div class="panel">${tabbar(Object.entries(bshort),b,'data-split-bench')}<div class="panel-body">${splitbar(b)}<div class="grid-2">${Object.keys(info.distributions).filter(k=>['primary_domain','difficulty','category','variant'].includes(k)).map(k=>`<div><h3 class="panel-title">${esc(k)}</h3>${distribution(b,k)}</div>`).join('')}</div></div></div>`) +
    section('Seed와 epoch별 실제 task 순서', `<div class="filters"><select id="order-method" aria-label="방법론">${Object.entries(names).map(([k,v])=>`<option value="${k}" ${m===k?'selected':''}>${esc(v)}</option>`).join('')}</select><select id="order-seed" aria-label="Seed">${[42,43,44].map(s=>`<option ${seed===s?'selected':''}>${s}</option>`).join('')}</select></div><div class="grid-2">${[1,2].map(e=>`<div class="panel panel-body"><h3 class="panel-title">Epoch ${e} · ${run['epoch'+e+'_order'].length} encounters</h3><div class="order-list">${run['epoch'+e+'_order'].map((id,i)=>`<div class="order-item"><span>${String(i+1).padStart(2,'0')}</span><button data-task-jump="${b}/${esc(id)}">${esc(id)}</button></div>`).join('')}</div></div>`).join('')}</div>`, 'Task ID를 누르면 해당 instruction으로 이동합니다.') +
    section('순서 생성과 Epoch 2 재셔플', `<div class="panel panel-body text-block"><span class="formula">derived = int(SHA256("order-v1:{seed}:{benchmark}:{split}")[:16], 16)<br>Random(derived).shuffle(canonical_split_ids)</span><p>이는 ordered manifest를 만드는 규칙입니다. Epoch 1 실행에서는 각 방법이 실제로 소비한 순서를 config에서 확인하고, 이 순서를 Epoch 2의 출발점으로 사용합니다.</p><span class="formula">epoch2 = epoch1_actual_order.copy()<br>Random(seed).shuffle(epoch2)</span><div class="source-ref-row">${source('Split order generator')}${source('Epoch 2 continuation')}</div></div>`) +
    section('검증과 hash', `<div class="panel panel-body"><p class="caption">선정 task ID의 중복과 split 간 overlap, 도메인·난이도·category 분포 및 membership hash가 기록되어 있습니다.</p><div class="hash-line">ordered manifest SHA-256<br>${esc(d.manifest_sha256)}</div>${jsonDetails(info.validation,'선택한 benchmark의 split 검증 원본')}</div>`);
}

function tasks() {
  const bench = state.taskBench;
  return heading('DATA / INSTRUCTIONS', 'Datapoint 탐색', '전체 instruction을 검색하고, 실제 split·seed별 순서와 task별 환경 설정을 확인하세요. BFCL은 모든 user turn을 시간 순서로 보여줍니다.', `<a class="button small" href="/download/tasks.csv">전체 instruction CSV ↓</a>`) +
    `<div class="filters"><label class="search-label"><input class="input" id="task-search" type="search" aria-label="Instruction 및 task ID 검색" placeholder="Task ID, instruction, domain, category 검색" value="${esc(state.taskQuery)}"></label><select id="task-benchmark" aria-label="벤치마크 필터"><option value="">모든 벤치마크</option>${Object.entries(bshort).map(([k,v])=>`<option value="${k}" ${bench===k?'selected':''}>${v}</option>`).join('')}</select><select id="task-split" aria-label="Split 필터"><option value="">모든 split</option>${Object.entries(snames).map(([k,v])=>`<option value="${k}" ${state.taskSplit===k?'selected':''}>${v}</option>`).join('')}</select><select id="task-selected" aria-label="사용 데이터 필터"><option value="all">원본 전체 706개</option><option value="selected" ${state.selectedOnly==='selected'?'selected':''}>실험 사용 389개</option></select></div>` +
    `<div class="catalog-layout"><div class="panel"><div class="task-count" id="task-count">목록을 읽는 중</div><div class="catalog-list" id="task-list"></div><div class="pager"><button data-page="-1" id="page-prev">← 이전</button><span id="page-label"></span><button data-page="1" id="page-next">다음 →</button></div></div><div class="panel task-detail" id="task-detail"><div class="empty">목록에서 datapoint를 선택하세요.</div></div></div>`;
}
async function loadCatalog() {
  const serial = ++catalogSerial;
  const query = new URLSearchParams({
    benchmark: state.taskBench,
    split: state.taskSplit,
    q: state.taskQuery,
    selected: state.selectedOnly,
    page: state.page
  });
  const result = await api('/api/tasks?' + query);
  if (serial !== catalogSerial || !$('#task-list')) return;
  $('#task-count').textContent = `${fmt(result.total)}개 datapoint · ${state.taskSplit?snames[state.taskSplit]:'모든 split'}`;
  $('#task-list').innerHTML = result.items.map(t => `<button class="task-row ${state.task?.id===t.id&&state.task?.benchmark===t.benchmark?'selected':''}" data-task="${esc(t.benchmark)}/${esc(t.id)}"><div class="row-meta">${splitPill(t.split)}<span>${bshort[t.benchmark]}</span>${t.turn_count?`<span>${t.turn_count} user turns</span>`:''}</div><strong>${esc(t.id)}</strong><p class="preview">${esc(t.preview.slice(0,150))}${t.preview.length>150?'…':''}</p><div class="row-meta"><span>${esc(t.meta.primary_domain||t.meta.category||(t.meta.scenario_id?'Scenario '+t.meta.scenario_id:''))}</span></div></button>`).join('') || '<div class="empty">조건에 맞는 instruction이 없습니다.</div>';
  $('#page-label').textContent = `${state.page} / ${Math.max(1,Math.ceil(result.total/result.page_size))}`;
  $('#page-prev').disabled = state.page <= 1;
  $('#page-next').disabled = state.page * result.page_size >= result.total;
  if (!result.items.length) {
    taskSerial++;
    state.task = null;
    $('#task-detail').innerHTML = '<div class="empty">조건에 맞는 datapoint가 없습니다.</div>';
  } else if (!state.task || !result.items.some(t => t.id === state.task.id && t.benchmark === state.task.benchmark)) {
    await selectTask(result.items[0].benchmark, result.items[0].id);
  } else renderTask();
}
async function selectTask(bench, id) {
  const serial = ++taskSerial;
  const value = await api(`/api/task/${encodeURIComponent(bench)}/${encodeURIComponent(id)}`);
  if (serial !== taskSerial || !$('#task-detail')) return;
  state.task = value;
  state.taskTab = 'instruction';
  renderTask();
  $$('.task-row').forEach(x => x.classList.toggle('selected', x.dataset.task === `${bench}/${id}`));
}

function taskInfo(item) {
  const meta = item.meta;
  return `<div class="task-meta-grid">${Object.entries(meta).filter(([k])=>!['involved_classes','tags'].includes(k)).map(([k,v])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('')}</div>`;
}

function renderTask() {
  const t = state.task;
  if (!t || !$('#task-detail')) return;
  const tabs = [
    ['instruction', 'Instruction'],
    ['environment', '도구 · 환경'],
    ['position', 'Split · 실행 순서'],
    ['metadata', '원본 · 메타데이터']
  ];
  let content = '';
  if (state.taskTab === 'instruction') content = (t.turns.length ? `<div class="turn-hint">모델에게는 각 user turn이 순서대로 공개됩니다. 이전 turn을 finish한 뒤 다음 instruction이 도착합니다.</div>${t.turns.map(turn=>`<div class="turn"><label>USER TURN ${turn.turn}</label><p>${esc(turn.text)}</p></div>`).join('')}` : `<div class="instruction">${esc(t.instruction)}</div>`);
  if (state.taskTab === 'environment') content = t.benchmark === 'bfcl' ? `<p class="caption">${esc((t.meta.involved_classes||[]).join(' · '))} · ${t.tools.length} public tool definitions. Task의 excluded functions를 반영했습니다.</p>${t.tools.map(tool=>`<details class="disclosure"><summary>${esc(tool.name)}<span class="muted" style="font-weight:400"> · ${esc((tool.description||'').slice(0,100))}</span></summary><div class="inner">${code(tool,'Public function schema')}</div></details>`).join('')}${jsonDetails(t.context,'초기 환경 설정')}` : t.benchmark === 'appworld' ? `<h3 class="panel-title">Original policy · code_generation</h3><p class="caption">현재 task datetime을 기준으로 날짜 표현을 해석하고, 환경의 애플리케이션 API를 code로 호출합니다. 같은 environment policy가 모든 방법과 split에 적용됩니다.</p>${code(t.context,'Task context')}${code(state.data.policy.text,'Environment policy')}` : `<h3 class="panel-title">Task container · Harbor Terminus 2</h3><p class="caption">Agent, verifier, environment마다 task.toml의 별도 timeout과 resource 제한을 사용합니다.</p>${Object.entries(t.context).map(([k,v])=>jsonDetails(v,k)).join('')}`;
  if (state.taskTab === 'position') {
    const order = t.orders[state.methodForTask][String(state.seed)];
    content = `<div class="filters"><select id="task-order-method" aria-label="방법별 순서">${Object.entries(names).map(([k,v])=>`<option value="${k}" ${state.methodForTask===k?'selected':''}>${esc(v)}</option>`).join('')}</select><select id="task-order-seed" aria-label="Seed별 순서">${[42,43,44].map(s=>`<option ${state.seed===s?'selected':''}>${s}</option>`).join('')}</select></div>${facts([['Task split',snames[t.split],t.split==='unused'?'실험에 사용하지 않은 원본 task':'Membership seed 42로 고정'],['Epoch 1 순서',order.epoch1||'—','실제로 학습한 order의 1-based 위치'],['Epoch 2 순서',order.epoch2||'—','Epoch 1 실제 순서를 다시 셔플']])}<div class="section">${table(['Seed','Epoch 1','Epoch 2','Val','Test'],[42,43,44].map(s=>[s,...['epoch1','epoch2','val','test'].map(k=>t.orders[state.methodForTask][s][k]??'—')]))}</div>`;
  }
  if (state.taskTab === 'metadata') content = taskInfo(t) + jsonDetails(t.meta, '데이터셋 메타데이터') + `<h3 class="panel-title">Instruction 출처</h3><p class="source-path">${esc(t.source.path)}</p><div class="hash-line">SHA-256 ${esc(t.source.sha256)}</div>`;
  $('#task-detail').innerHTML = `<div class="detail-header"><span class="small-caps">${esc(bnames[t.benchmark])}</span><h2>${esc(t.id)}</h2><div class="row-meta">${splitPill(t.split)}${t.meta.difficulty?pill('Difficulty '+t.meta.difficulty):''}${t.meta.variant?pill('Variant '+t.meta.variant):''}${t.meta.user_turns?pill(t.meta.user_turns+' user turns'):''}</div></div>${tabbar(tabs,state.taskTab,'data-task-tab')}<div class="detail-content">${content}</div>`;
}

function serving() {
  const d = state.data;
  return heading('INFERENCE / REQUEST PATH', '모델이 서빙되고 호출되는 방식', '서버 설정, parser bridge, 역할별 호출 인자, actor-turn 예산의 적용 순서를 연결해 볼 수 있습니다.') +
    `<div class="call-flow"><div class="call-node"><span class="small-caps">CLIENT</span><strong>Actor · Reflection · Mutation</strong><div class="port">OpenAI-compatible</div><p>기준 모델 Qwen3.6-35B-A3B<br>enable_thinking = false</p></div><span class="call-connector">→</span><div class="call-node"><span class="small-caps">TOOL PARSER BRIDGE</span><strong>qwen3_coder</strong><div class="port">:18006 /v1</div><p>tool_choice=auto 요청 해석<br>모델 출력의 tool call을 복원</p></div><span class="call-connector">→</span><div class="call-node"><span class="small-caps">INFERENCE SERVER</span><strong>vLLM · 현재 launcher의 GPU 1</strong><div class="port">:8006</div><p>Qwen3.6-35B-A3B<br>최대 context 262,144 tokens</p></div></div>` +
    `<div id="runtime-status" class="caption">${serviceSummary()}</div>` +
    historicalServing() +
    section('현재 남아 있는 launcher의 vLLM 설정', `<div class="panel">${params([
 ['model','Qwen/Qwen3.6-35B-A3B','Actor와 분석 역할에 공통'],['CUDA_VISIBLE_DEVICES','1','GPU index; tensor-parallel-size 1'],['gpu_memory_utilization',0.95,'KV cache 포함 GPU 메모리 목표'],['max_model_len',262144,'모델 서버의 전체 context 한도'],['max_num_batched_tokens',16384,'한 scheduling iteration의 token batch'],['max_num_seqs',128,'서버 scheduling sequence 한도'],['kv_cache_dtype','fp8','KV cache precision'],['reasoning_parser','qwen3','서빙 parser'],['language_model_only',true,'텍스트 언어 모델'],['enable_prefix_caching',true,'Prefill prefix 캐시'],['chat_template','qwen3_6-27B.jinja','별도 template 파일'],['native auto-tool flags','launcher에서 비활성','qwen3_coder bridge가 auto tools 처리']
 ])}<div class="caption-box">${source('Model launcher')}&nbsp;&nbsp;${source('Chat template')}&nbsp;&nbsp;${source('Parser bridge')}</div></div>`) +
    section('역할별 요청 인자', `<div class="panel">${table(['역할','temperature','최대 출력','timeout · retries','출력 형식 / thinking'],[
 ['ACE actor','0.7','설정 8,192 → turn budget이 최대 2,048로 제한','Actor 최대 3시도. 계측 wrapper 내부 SDK retry=0','tool calls · enable_thinking=false'],
 ['ACE reflector / curator','0.7','8,192','최대 3시도, exponential backoff','JSON mode · enable_thinking=false'],
 ['PEACE parent / replay actor','0.7','최대 2,048 / environment turn','모델 설정 timeout 600s, num_retries 1. Turn budget이 모든 계측된 재시도 포함','tool calls · enable_thinking=false'],
 ['PEACE reflection / mutation','0.7','8,192','PremiseLLM timeout 600s / SDK retries 1','JSON parsing · enable_thinking=false'],
 ['GEPA / SkillOpt actor','0.7','최대 2,048 / environment turn','Runtime actor subprocess 최대 3시도','tool calls · enable_thinking=false'],
 ['GEPA / SkillOpt optimizer','0.7','8,192','OpenAI client timeout 600s, max_retries 0; Runtime 최대 3시도','텍스트 또는 stage별 JSON · enable_thinking=false'],
 ['Terminal-Bench actor','0.7','최대 2,048 / environment turn','Native Terminus 2 / 각 task의 agent deadline','Harbor native actor · recorded budget policy']
 ])}</div>`, 'Terminal-Bench는 native Terminus 2의 JSON command protocol을 사용합니다. 공통 예시의 tools/tool_choice는 BFCL·AppWorld의 tool actor 경로에 해당합니다. 아래 세션 config와 프롬프트 탭에서 benchmark별 차이를 확인할 수 있습니다.') +
    section('요청 예시와 생략된 인자', `<div class="grid-2"><div class="panel panel-body"><h3 class="panel-title">Actor의 요청 구성 예시</h3>${code({model:'Qwen/Qwen3.6-35B-A3B',messages:[{role:'system',content:'<method actor system prompt + learned policy>'},{role:'user',content:'<task instruction and environment context>'}],tools:'<public action schemas>',tool_choice:'auto',temperature:0.7,max_tokens:2048,num_retries:0,max_retries:0,extra_body:{chat_template_kwargs:{enable_thinking:false}}},'구성 예시 · 실제 task에 따라 messages와 tools가 달라집니다')}<p class="caption">클라이언트 공통 설정은 temperature=0.7을 명시하며 top_p·top_k를 덮어쓰지 않습니다. 아래 서버 기본값과 합쳐 적용됩니다. frequency_penalty·presence_penalty와 모델 sampling seed도 공통 요청에서 별도로 고정하지 않습니다. 데이터 순서·optimizer seed와는 다릅니다.</p></div><div class="panel panel-body text-block"><h3>Turn 예산의 적용 순서</h3><ol class="step-list"><li><div><strong>Role별 모델 설정</strong><br>temperature, timeout, 초기 max_tokens를 구성합니다.</div></li><li><div><strong>남은 turn 예산으로 clamp</strong><br>요청의 max_tokens를 min(requested, remaining output)으로 제한합니다.</div></li><li><div><strong>실제 usage 계측</strong><br>모든 선택·acting·계측된 retry의 prompt/completion tokens를 같은 environment turn에 더합니다.</div></li><li><div><strong>실패 처리</strong><br>Budget 초과 또는 finish_reason=length면 trajectory를 공식 score 0으로 종료합니다.</div></li></ol>${source('Actor budgets')}<div class="section">${source('Baseline LLM runtime')}<br>${source('Baseline actor')}</div></div></div>`) +
    section('서버가 보충하는 sampling 기본값', `<div class="grid-2"><div class="panel panel-body">
      <h3 class="panel-title">요청값과 모델 기본값의 합성</h3>
      ${params([['temperature', 0.7, '클라이언트 명시값이 모델 default 1.0을 덮어씀'], ['top_p', 0.95, '모델 generation_config.json · 요청에서 별도 지정 없음'], ['top_k', 20, '모델 generation_config.json · 요청에서 별도 지정 없음']])}
      <p class="caption">현재 launcher에는 --generation-config vllm 옵션이 없습니다. 마지막 서버 시작 로그에서도 모델 generation_config의 sampling 값을 읽었다는 메시지를 확인했습니다.</p>
      ${code(d.serving_evidence.last_server_log_excerpt || '마지막 서버 로그 근거 없음', '마지막 서버 시작 로그 발췌')}
      </div><div class="panel panel-body"><h3 class="panel-title">설치 버전과 model revision</h3>
      ${code({installed_vllm_version: d.serving_evidence.installed_vllm_version, model_revision_pinned_in_launcher: false, gpu: 'NVIDIA H200 NVL · CUDA_VISIBLE_DEVICES=1 · TP=1'}, '현재 설치 및 launcher 근거')}
      <p class="caption">launcher는 모델 이름을 지정하고 Hugging Face revision은 고정하지 않았습니다. 현재 캐시에서 확인한 아래 revision들은 같은 sampling 설정을 갖습니다. 이 목록이 과거 요청의 모델 revision을 증명하지는 않습니다.</p>
      ${d.serving_evidence.generation_configs.map(g => `<details class="disclosure"><summary>${esc(g.revision)}</summary><div class="inner">${code(g.config)}${source(g.source, 'generation_config 원본')}${jsonDetails(g.architecture, '모델 architecture 전체 config')}${source(g.architecture_source, '모델 config 원본')}</div></details>`).join('')}
      </div></div>`) +
    section('기준과 달랐던 실제 평가', `<div class="panel panel-body text-block"><h3>AppWorld ACE의 Epoch 2 고정 Train/Val</h3><p>학습은 temperature 0.7입니다. 토큰 감사에서 고정 Train/Val의 실제 요청 temperature 0.0을 확인했습니다. 같은 실행 묶음의 config와 평가 helper 상수가 달라 생긴 차이로 기록되어 있습니다.</p><h3>평가 시작 전 중단</h3><p>ACE의 일부 최종 Test는 temperature/seed 검증 또는 session_count 검증에서 중단됐습니다. GEPA·SkillOpt의 고정 Train/Val 500건은 temperature 키 누락으로 actor 호출 전에 실패했습니다. 실행 상태는 실제 source status에서 조회합니다.</p><a class="caption" href="#runs">실행별 상태와 config ↗</a></div>`) +
    section('실제 저장된 세션 config', `<div class="panel">${d.examples.map(e=>`<details class="prompt-item"><summary><strong>${names[e.method]} · ${bshort[e.benchmark]}</strong><span>실행 세션에서 추출</span></summary>${code(e,'RunConfig / agent_kwargs / benchmark_kwargs')}</details>`).join('')}</div>`);
}

function historicalServing() {
  const d = state.data;
  const flag = (command, name) => {
    const index = (command || []).indexOf(name);
    return index >= 0 ? command[index + 1] : '해당 plan에 기록 없음';
  };
  const rows = d.epoch1_protocols.map(p => [
    esc(p.path.split('/').slice(-2,-1)[0]), p.run_ids.length,
    p.config.max_parallel, p.config.server_max_num_seqs,
    esc(flag(p.config.server_command, '--max-model-len')),
    esc(flag(p.config.server_command, '--revision'))
  ]);
  return section('시점별 서버 설정과 근거 범위', `<div class="panel">
    ${table(['Epoch 1 원본 실행 계획','이 페이지의 관련 실행','저장된 scheduler 상한','max_num_seqs','max_model_len','model revision'], rows)}
    <div class="caption-box">위 값은 저장된 Epoch 1 원본 plan에서 읽었습니다. 원본 plan에는 추가 baseline이 포함되고, 스케줄링 정책이 도중에 갱신된 기록도 있어 이 표의 scheduler 상한을 모든 시점에 적용된 값으로 해석하면 안 됩니다. AppWorld plan에 server_command가 없으면 context·revision을 추정하지 않습니다.</div>
    </div><div class="grid-2 section"><div class="panel panel-body text-block">
      <h3>현재 launcher와 학습 당시의 설정</h3><p>원본 server_command에는 context 131,072·max_num_seqs 256·revision 995ad96eacd98c81ed38be0c5b274b04031597b0이 기록돼 있습니다. 현재 남아 있는 launcher에는 context 262,144·max_num_seqs 128이 들어 있고 revision 옵션이 없습니다.</p>
      <p>현재 launcher 파일이 과거 Epoch 2 전체 요청에 그대로 적용됐는지는 이 파일만으로 확인할 수 없습니다. 아래 launcher 표는 현재 파일의 내용이며, 호출 예산은 실행 config와 actor wrapper를 기준으로 확인합니다.</p>
    </div><div class="panel panel-body"><h3 class="panel-title">Epoch 2 시작 전 parser 검증</h3>
      ${code(d.historical_serving_preflight,'serving_preflight.json')}
      <p class="caption">auto/none tokenization이 같고 add tool call을 복원하는지 검사한 결과입니다. 이 검증 파일에는 전체 vLLM launch command나 model revision이 기록돼 있지 않습니다.</p>
    </div></div>`) + section('Epoch 1 원본 계획의 세부 설정', `<div class="panel">${d.epoch1_protocols.map(p=>`<details class="prompt-item"><summary><strong>${esc(p.path.split('/').slice(-2,-1)[0])}</strong><span>${p.run_ids.length} related runs</span></summary>${code(p.config)}${jsonDetails(p.run_ids,'이 페이지에서 참조하는 실행 ID')}${source(p.source,'원본 계획의 출처와 hash')}</details>`).join('')}</div>`);
}

function serviceSummary() {
  const l = state.live;
  if (!l) return '현재 서비스 상태를 확인하는 중입니다.';
  return `<span class="pill ${l.services.vllm?.online?'green':'amber'}">vLLM ${l.services.vllm?.online?'연결됨':'현재 연결되지 않음'}</span> <span class="pill ${l.services.parser?.online?'green':'amber'}">Parser ${l.services.parser?.online?'연결됨':'현재 연결되지 않음'}</span> <span class="snapshot">확인 ${esc(new Date(l.checked_at).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'}))} KST · 원본 계획과 현재 launcher를 아래에서 시점별로 구분합니다</span>`;
}

function prompts() {
  const items = state.data.prompts.filter(p => state.promptMethod === 'all' || p.methods.includes(state.promptMethod));
  return heading('PROMPTS / FULL TEXT', '모델에 전달하는 프롬프트', 'Frozen source에서 읽은 템플릿과 정책 원문입니다. 실제 호출에는 현재 task, trajectory, policy와 공식 feedback이 삽입됩니다.') +
    `<div class="panel">${tabbar([['all','전체'],...Object.entries(names)],state.promptMethod,'data-prompt-method')}${items.map(p=>`<details class="prompt-item"><summary><strong>${esc(p.name)}</strong><span>${esc(p.methods.map(m=>names[m]).join(' · '))}</span></summary><pre>${esc(p.text)}</pre><div class="source-ref-row">${source(p.source,'Frozen 원본과 SHA-256')}</div></details>`).join('')}</div>` +
    section('PEACE reflection·mutation의 실제 생성 코드', `<div class="panel panel-body text-block"><p>F-string으로 task·trajectory를 조립하는 부분까지 전체 source에서 확인할 수 있습니다. Alignment prompt는 source에 남아 있지만, 이 cohort의 alignment_scoring=false 설정에서는 해당 호출을 수행하지 않습니다.</p><div class="source-ref-row">${source(state.data.prompts.find(p=>p.name==='EVIDENCE_RULES')?.source||state.data.source_refs['PEACE pipeline'],'PEACE prompt 생성 원본')}${source('ACE actor','ACE prompt 삽입 방식')}${source('Baseline LLM runtime','GEPA·SkillOpt 메시지 구성')}</div></div>`);
}

function runStatus(run) {
  const p = state.live?.cohort?.progress?.jobs?.[run.id]?.status;
  return p || 'unknown';
}

function runs() {
  const selected = state.run ? state.data.runs.find(r => r.id === state.run) : null;
  const labels = {
    completed: '학습·Test 완료',
    error: '오류',
    pending: '대기',
    paused: '중지',
    running: '실행 중'
  };
  const colors = {
    completed: 'green',
    error: 'red',
    running: 'blue',
    pending: '',
    paused: 'amber'
  };
  const rows = state.data.runs.map(r => [
    `<span class="method-title">${icon(r.method)}<strong>${names[r.method]}</strong></span>`,
    bshort[r.benchmark], r.seed, `${r.epoch1_count} → ${r.train_count}`,
    `<span data-run-status="${esc(r.id)}">${pill(labels[runStatus(r)] || runStatus(r), colors[runStatus(r)])}</span>`,
    `<button class="evidence" data-run="${esc(r.id)}">설정 열기</button>`
  ]);
  return heading('RUNS / CONFIGURATION', '36개 실행의 실제 설정',
      '방법론, benchmark, seed마다 독립된 state와 policy를 저장합니다. 실행을 선택하면 Epoch 1·2의 config와 command를 비교할 수 있습니다.', scope()) +
    `<div class="panel">${table(['방법','벤치마크','Seed','학습 encounters','현재 Epoch 2 상태','상세'], rows)}</div>` +
    (selected ? runDetail(selected) : '<p class="caption" style="margin-top:20px">상세를 선택하면 실행의 전체 config와 task order가 아래에 표시됩니다.</p>');
}

function runDetail(r) {
  const e = r.epochs[String(state.epoch)];
  const policy = r.method === 'premise' ? e.config?.benchmark_configs?.[r.benchmark]?.policy_override : r.benchmark_kwargs?.policy_override;
  const title = `${names[r.method]} · ${bshort[r.benchmark]} · seed ${r.seed}`;
  return section(title, `<div class="panel panel-body">
    <div class="source-path">${esc(r.id)}</div>
    <div class="grid-2"><div><h3 class="panel-title">Epoch ${state.epoch} config</h3>
      ${e.config ? code(e.config) : '<p class="caption">이 실행의 Epoch 2 config는 아직 생성되지 않았습니다.</p>'}
      ${e.source ? source(e.source, '저장된 config') : ''}
    </div><div><h3 class="panel-title">실행 command</h3>
      ${code(e.command_shell, 'Recorded command · shell-quoted argv')}
      <h3 class="panel-title">현재 상태</h3>${code(state.live?.runs?.[r.id] || {}, 'job_status.json')}
      <h3 class="panel-title">Actor budget</h3>${code(r.actor_limits)}
    </div></div>
    <h3 class="panel-title">Frozen 코드 snapshot</h3><p class="source-path">${esc(r.source_snapshot)}</p>
    ${jsonDetails(r.benchmark_kwargs, 'Benchmark kwargs')}
    ${jsonDetails(r.source_spec, '원본 실행 spec 전체')}
    <p class="source-path">Epoch 1 source plan: ${esc(r.source_plan)}</p>
    ${policy ? jsonDetails(policy, '환경 policy') : ''}
    ${jsonDetails({epoch1: r.epoch1_order, epoch2: r.epoch2_order, val: r.val_ids, test: r.test_ids}, '모든 task ID와 순서')}
  </div>`);
}

function tokens() {
  const d = state.data,
    t = d.token_tables;
  return heading('USAGE / ACCOUNTING', '학습과 평가의 토큰 비용', '실제 usage 로그에서 입력·출력 토큰을 집계했습니다. Epoch 1 학습과 학습 후 원본 Test, 별도의 고정 정책 평가를 확인할 수 있습니다.') +
    section('Epoch 1 · 학습과 원본 Test', `<div class="panel">${table(t['e1-01-train-vs-test.csv'][0],t['e1-01-train-vs-test.csv'].slice(1).map(r=>r.map(esc))) }<div class="caption-box">학습 비용에는 actor, reflection·curation·mutation, replay, 학습 중 초기·후보 Val이 포함됩니다. 원본 Test는 방법마다 387건이며 BFCL 150, AppWorld 150, Terminal-Bench 87건입니다.</div></div>`) +
    section('단계별 소비', `<div class="panel">${table(t['e1-03-stages.csv'][0],t['e1-03-stages.csv'].slice(1).map(r=>r.map(esc)))}</div>`, 'Cached whole responses는 새 모델 소비에 합산하지 않았습니다.') +
    section('추가 고정 정책 평가', `<div class="panel">${table(t['e1-04-frozen-checkpoint-evaluation.csv'][0],t['e1-04-frozen-checkpoint-evaluation.csv'].slice(1).map(r=>r.map(esc))) }<div class="caption-box">최종 policy 36개가 실제 평가 32개를 참조합니다. 같은 policy의 평가를 재사용한 경우, 실행된 비용을 한 번 계산했습니다. 원본 Test와 별도로 실행한 평가 큐의 관측값입니다.</div></div>`) +
    section('Epoch 1 → Epoch 2 전체 요약', `<div class="panel">${table(t['01-method-token-summary.csv'][0],t['01-method-token-summary.csv'].slice(1).map(r=>r.map(esc)))}</div>`) +
    note('누락과 평가 조건', 'Epoch 1의 usage 미기록 호출 68개와 별도 평가 시도 5개는 미확인입니다. AppWorld ACE의 Epoch 2 고정 Train/Val은 실제 temperature 0.0이었습니다. 자세한 기록과 CSV는 <a href="http://166.104.110.49:8878/#experiment/exp-20261109-68f88079" target="_blank" rel="noopener">실험 기록장</a>에 있습니다.');
}

function sources() {
  const d = state.data,
    hashes = [
      ['Ordered split', d.manifest_sha256],
      ['AppWorld environment policy', d.runs.find(r => r.policy_sha256)?.policy_sha256],
      ['초기 continuation manifest pin', d.protocol.continuation_script_sha256]
    ];
  return heading('PROVENANCE / REPRODUCIBILITY', '설정의 근거와 재현 자료', '설정 화면에서 사용하는 근거를 파일 경로, SHA-256, 원문과 함께 확인할 수 있습니다. 내려받는 설정 JSON에도 동일한 근거 정보가 포함됩니다.') +
    `<div class="provenance-grid">${hashes.map(([k,v])=>`<div class="panel panel-body"><span class="small-caps">${esc(k)}</span><div class="hash-line" style="margin-top:14px">${esc(v)}</div></div>`).join('')}</div>` +
    section('핵심 구현과 버전', `<div class="panel">${Object.entries(d.source_refs).map(([title,id])=>{const s=d.sources.find(s=>s.id===id);return `<div class="source-card"><div><strong>${esc(title)}</strong><p>${esc(s.path)}</p><code>${esc(s.sha256)}</code></div><div>${source(id,'원문 보기')}</div></div>`;}).join('')}</div>`) +
    section('Continuation runner의 기록된 revision', `<div class="panel">${table(['Revision','기록 hash와 일치','원문'],d.runner_versions.map(r=>[r.revision,pill(r.matches_recorded_hash?'일치':'변경 확인',r.matches_recorded_hash?'green':'amber'),source(r.source)]))}<div class="caption-box">실행별 command가 가리키는 파일을 확인하세요. 초기 manifest pin과 이후 revision을 함께 보존했습니다.</div></div>`) +
    section('전체 설정 근거', `<div class="panel">${d.sources.filter(s=>!Object.values(d.source_refs).includes(s.id)).map(s=>`<div class="source-card"><div><strong>${esc(s.title)}</strong><p>${esc(s.path)}</p><code>${esc(s.sha256)}</code></div><div>${source(s.id,s.text?'원문 보기':'출처 확인')}</div></div>`).join('')}</div>`) +
    section('재현에서 중요한 구분', `<div class="panel panel-body text-block"><h3>원본 snapshot과 작업 디렉토리의 코드</h3><p>모델과 학습 loop는 각 실행이 가리키는 frozen snapshot에서 읽습니다. 이 페이지의 방법론과 config는 해당 snapshot 및 기록된 실행 파일을 기준으로 합니다.</p><h3>설정값과 관측값</h3><p>기준 manifest의 temperature와 실제 요청 로그가 다른 평가가 있습니다. 요청 payload의 일부 인자가 trace에서 생략된 경우에는 세션 config와 budget wrapper 코드를 함께 제공합니다.</p><h3>벤치마크별 격리 환경</h3><p>BFCL, AppWorld, Terminal-Bench의 dependency는 별도 benchmark venv에 설치되어 있습니다. GEPA와 SkillOpt도 별도 optimizer venv를 사용하며 core actor subprocess와 연결합니다.</p><h3>고정 policy 평가</h3><p>ACE·PEACE는 최종 playbook, GEPA·SkillOpt는 Val로 고른 instruction·best skill을 고정해 평가합니다. AppWorld는 environment policy SHA까지 일치시킵니다.</p></div>`);
}

async function refreshCheckpointEvaluation() {
  const host = $('#checkpoint-evaluation-content');
  if (!host) return;
  const serial = ++checkpointSerial;
  const d = await api('/api/checkpoint-evaluation');
  if (serial !== checkpointSerial || host !== $('#checkpoint-evaluation-content')) return;
  if (!d.available) { host.textContent = 'CP 평가 진행 기록을 기다리고 있습니다.'; return; }
  const opened = host.querySelector('details')?.open || false;
  const counts = d.counts;
  const phase = {running:'평가 진행 중',paused:'새 작업 배정 일시중지',completed:'평가 완료',error:'오류 확인 필요'}[d.phase] || d.phase;
  const ports = Object.entries(d.query_limits).map(([port, limit]) =>
    `포트 ${port}: ${fmt(d.gpu_queries.active?.[port] || 0)} / ${fmt(limit)}`).join(' · ');
  const progress = value => `<div class="progress-label"><span>${fmt(value.done)} / ${fmt(value.total)}</span><b>${value.total ? (100 * value.done / value.total).toFixed(1) : '0.0'}%</b></div><div class="tiny-progress"><div style="width:${value.total ? Math.min(100,100 * value.done / value.total) : 0}%"></div></div>`;
  host.classList.remove('loading');
  host.innerHTML = heading('공통 REFLECTOR · FRESH 1 EPOCH / CP10', `CP10 평가 · GPU ${d.gpu_indices.join(', ')}`,
    `${esc(phase)} · CP ${fmt(d.checkpoints)} / ${fmt(d.expected_checkpoints)} 등록 · ${esc(ports)} · 공통 호출 큐 대기 ${fmt(d.gpu_queries.waiting || 0)}`,
    `<a class="button" href="${esc(d.live_monitor)}" target="_blank" rel="noopener">실시간 기록장 ↗</a>`) +
    facts([['완료',fmt(counts.completed || 0),'기존 결과 재사용 포함'],['실행',fmt(counts.running || 0),'평가 환경 수'],
      ['대기',fmt(counts.pending || 0),'공통 작업 큐'],['오류',fmt(counts.error || 0),'공식 점수 없이 오류로 분리']]) +
    section('Frozen Train · Val · Test', `<div class="panel">${table(['Split','평가 진행률'],
      Object.entries(d.splits).map(([split,value])=>[esc(snames[split]),progress(value)]))}
      <div class="caption-box">기존 결과 ${fmt(d.reused_outcomes)}건 재사용 · 동일 정책 ${fmt(d.aliases)} CP 결과 공유 · ${esc(new Date(d.updated_at).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'}))} KST · 5초 자동 갱신</div></div>`) +
    `<details class="disclosure" ${opened?'open':''}><summary>방법론 · Benchmark · Seed · CP별 진행과 점수 (${d.rows.length}개)</summary><div class="inner">${table(
      ['방법','Benchmark','Seed','CP','상태','Train','Val','Test'],d.rows.map(row=>[
        esc(names[row.method] || row.method),esc(bshort[row.benchmark] || row.benchmark),esc(row.seed),esc(row.point),esc(row.alias?'동일 정책 공유':row.state),
        ...['train','val','test'].map(split=>{const v=row.splits[split] || {};return `${fmt(v.done || 0)}/${fmt(v.total || 0)} · ${v.score==null?'점수 대기':(100*v.score).toFixed(1)+'%'}`;})]))}</div></details>`;
}

async function refreshSharedReflection() {
  const host = $('#shared-reflection-content');
  if (!host) return;
  const d = await api('/api/shared-reflection');
  if (!d.available) { host.textContent = '새 실험 설정이 아직 준비되지 않았습니다.'; return; }
  const p = d.plan;
  const when = seconds => seconds ? new Date(seconds * 1000).toLocaleString('ko-KR', {timeZone:'Asia/Seoul',hour12:false}) + ' KST' : 'benchmark 진입 후';
  const statusNames = {running:'실행 중',completed:'학습·Test 완료',pending:'예약 대기',error:'오류'};
  const statusColors = {running:'blue',completed:'green',pending:'amber',error:'red'};
  const score = x => Number(x).toFixed(2);
  const validation = d.validation.scores_out_of_20;
  const groups = d.calibration.groups;
  host.classList.remove('loading');
  host.innerHTML = heading('NEW COHORT / FRESH 1 EPOCH', '공통 reflector · medium · 36개 실험',
    'BFCL → AppWorld → Terminal-Bench. 동일한 공통 초기 actor context에서 새로 학습합니다. 기존 Epoch 2 continuation 결과와 분리한 실험입니다.',
    `<a class="button" href="${esc(d.journal)}" target="_blank" rel="noopener">실험 기록장 ↗</a>`) +
    facts([
      ['실행 중',fmt(d.counts.running || 0),'최대 12개 실험'],
      ['완료',`${d.counts.completed || 0} / 36`,'학습과 최종 Test 모두 완료한 수'],
      ['예약 대기',fmt(d.counts.pending || 0),'seed 시차 또는 benchmark 순서 대기'],
      ['오류',fmt(d.counts.error || 0),`마지막 확인 ${esc(new Date(d.checked_at).toLocaleTimeString('ko-KR',{timeZone:'Asia/Seoul'}))} KST · 30초 자동 갱신`]
    ]) +
    section('실제 요청에 적용하는 공통 설정', `<div class="panel">${params([
      ['모델',p.model,'기존 vLLM 서버 사용 · 요청/usage 보존 bridge 경유'],
      ['temperature','actor 0.0 / reflector·mutator·optimizer·judge 0.7','학습 acting, replay와 고정 평가 actor에 동일 적용'],
      ['reasoning','medium + thinking=true · 상한 16,384 / 요청','Qwen은 medium별 고유 예산을 제공하지 않아 명시적 상한을 별도로 적용'],
      ['비추론 completion','actor 2,048 / 환경 turn · 분석 8,192 / 요청','추론 토큰을 제외한 기존 길이 유지. 생성 후 action 실행 전에 검사'],
      ['actor 제한','input 100,000 / turn · BFCL 30 turns · AppWorld/TB 40 turns','초과·잘린 출력은 실패 종료. 실제 비용에는 reasoning도 모두 포함'],
      ['PEACE','AUD · alignment off · beam 2 · candidates 4 · current 1 · past 3','실제 환경 replay · replay workers 28 · LLM slots 32'],
      ['GEPA / SkillOpt','최초 full Val: BFCL 50 · AppWorld 50 · TB 30','각 방법론·benchmark의 seed42 실제 시작 후 seed43 +15분·seed44 +30분 · 누적 학습 예산 무제한'],
      ['초기 context','shared-empty-v1 · empty learned policy','benchmark·seed별 네 방법론의 초기 prompt 및 profile SHA 일치 확인'],
      ['파일 열기 한도',4096,'실행 프로세스의 soft limit'],
    ])}</div>`) +
    section('예약과 현재 진행', `<div class="panel">${table(
      ['Benchmark','방법론','Seed','상태','학습','초기 Val','투입 시각 / 시차'],
      p.jobs.map(j => [esc(bshort[j.benchmark]),esc(names[j.method]),j.seed,
        pill(statusNames[j.runtime.status] || j.runtime.status,statusColors[j.runtime.status]),
        `${j.trained} / ${j.train_count}` + (j.detail.phase==='test' ? '<br>최종 Test 평가 중' : ''),
        j.initial_validation ? `${j.initial_validation.completed} / ${j.initial_validation.total}` : '—',
        j.release_at ? esc(when(j.release_at)) : j.release_delay_seconds ? `해당 방법론 seed42 시작 +${j.release_delay_seconds/60}분` : '빈 자리 발생 시 투입']))}</div>`,
      '전체 최대 12개를 유지하며, 학습·최종 Test가 끝난 자리에 다음 실행 가능한 실험을 바로 투입합니다. 대기 우선순위는 BFCL → AppWorld → Terminal-Bench이며 benchmark 간 동시 실행이 가능합니다. GEPA·SkillOpt의 seed 시작 간격은 각 방법론·benchmark의 seed42 실제 시작을 기준으로 유지합니다.') +
    section('Prompt 검증: 기존 train trajectory 12개', `<div class="panel">${table(
      ['프롬프트','개발 6개 · /20','별도 holdout 6개 · /20'],
      [['기존 PEACE',score(validation.development.legacy_peace),score(validation.holdout.legacy_peace)],
       ['first decision',score(validation.development.first_decision),score(validation.holdout.first_decision)],
       ['선택: instruction aware',score(validation.development.instruction_aware),score(validation.holdout.instruction_aware)]]
    )}<div class="caption-box">선정에는 개발 사례와 후속 지시의 원인 지점 regression만 사용했습니다. Holdout은 선정에 사용하지 않았습니다. 동일 모델의 블라인드 rubric 점수이며, holdout에서 기존 PEACE보다 높지 않았습니다. 실제 학습 성능의 개선 여부는 이번 본 실험으로 측정합니다. 최종 prompt로 ACE·PEACE 각 1개 추가 학습까지 완료했습니다.</div></div>`) +
    section('추론 길이 실측과 상한 선정', `<div class="panel">${table(
      ['사전 calibration','호출 수','추론 P50 / P95 / P99','최대','상한 도달','반복 / 길이 종료'],
      Object.entries(groups).map(([key,g])=>[esc(key),fmt(g.reasoning.count),`${fmt(g.reasoning.p50)} / ${fmt(g.reasoning.p95)} / ${fmt(g.reasoning.p99)}`,fmt(g.reasoning.max),g.cap_hits,`${g.repetitions} / ${g.length_finishes}`])
    )}<div class="caption-box">8,192 상한에서 actor 2건이 경계에 도달해 16,384로 여유를 뒀습니다. 분석 1건의 반복 출력은 거부되었습니다. 최종 16,384 설정의 샘플에서는 반복·길이 종료가 없었습니다. Calibration의 진단용 val/test 라벨도 원래 train pool에서만 추출했으며, 본 Test는 사용하지 않았습니다.</div></div>`) +
    section('이번 cohort의 실제 요청 관측', `<div class="panel">${table(
      ['역할','완료 요청','실제 T','effort','input','output 전체','추론 / 비추론','profile 일치'],
      Object.entries(d.usage).map(([role,u])=>[esc(role),fmt(u.requests),esc(u.temperature.join(', ') || '아직 없음'),esc(u.reasoning_effort.join(', ') || '아직 없음'),fmt(u.prompt_tokens),fmt(u.completion_tokens),`${fmt(u.reasoning_tokens)} / ${fmt(u.visible_tokens)}`,u.requests ? pill(u.profile_matches?'일치':'불일치',u.profile_matches?'green':'red'):'—'])
    )}<div class="caption-box">새 cohort bridge에서 응답이 완료된 요청만 집계합니다. actor에는 training·replay·고정 평가 acting이 포함되며 analysis에는 모든 분석 모듈이 포함됩니다. 사전 calibration 비용은 제외합니다. <a href="#llm-calls">실제 reasoning·최종 출력 보기 →</a></div></div>`) +
    section('ACE · PEACE 공통 reflector 규칙', jsonDetails(d.prompt.system,'System prompt 전문') + jsonDetails(d.prompt.rules,'공통 지시문 전문') +
      `<div class="panel panel-body text-block"><p>전체 기록과 후속 지시의 공개 시점을 읽고 최초의 원인 결정 또는 결정적 성공 행동을 고릅니다. 기존 소비 로직과 replay 연결을 위해 결과를 span.start/end로 인코딩합니다. ACE는 entry_assessments를 bullet counter로 변환하고 전체 reflection JSON을 curator에 전달합니다. PEACE는 해당 지점과 expected_behavior로 후보를 만들고 실제 환경 replay로 평가합니다.</p></div>`) +
    section('Split과 재현 자료', `<div class="panel">${table(['Benchmark','Train','Val','Test'],['bfcl','appworld','terminalbench2'].map(b=>{const j=p.jobs.find(j=>j.benchmark===b);return [esc(bnames[b]),j.train_ids.length,j.val_ids.length,j.test_ids.length];}))}</div>` +
      jsonDetails({output:p.output,manifest_sha256:p.manifest_sha256,inference_profile_sha256:p.inference_profile_sha256,reflection_template_sha256:d.prompt.source_sha256,appworld_policy_sha256:p.appworld_policy_sha256,profile:p.profile},'고정한 설정과 SHA-256') +
      `<a class="button small" href="/api/shared-reflection" target="_blank">36개 command · 원시 설정 JSON ↗</a>`);
}

const views = {
  'llm-calls': () => '<div id="llm-calls-content" class="loading">저장된 모델 응답을 읽고 있습니다.</div>',
  'shared-reflection': () => '<div id="checkpoint-evaluation-content" class="loading">CP 평가 실시간 진행률을 읽고 있습니다.</div><div id="shared-reflection-content" class="loading">새 cohort의 설정과 실행 상태를 읽고 있습니다.</div>',
  'initial-context': initialContext,
  overview,
  methods,
  benchmarks,
  tasks,
  splits,
  serving,
  prompts,
  runs,
  tokens,
  sources
};

function route() {
  const [view0, ...parts] = location.hash.slice(1).split('/');
  const view = views[view0] ? view0 : 'overview';
  if (view === 'methods' && names[parts[0]]) state.method = parts[0];
  if (view === 'prompts' && names[parts[0]]) state.promptMethod = parts[0];
  if (view === 'tasks' && parts[0] && bnames[parts[0]]) state.taskBench = parts[0];
  if (view === 'runs' && parts[0]) state.run = decodeURIComponent(parts[0]);
  $('#main').innerHTML = (view === 'overview' ? '<div id="checkpoint-evaluation-content" class="loading">현재 CP 평가 진행률을 읽고 있습니다.</div>' : '') + views[view]();
  $('#current-location').textContent = navnames[view];
  const sharedView = view === 'shared-reflection' || view === 'llm-calls';
  $('.cohort-note strong').textContent = sharedView ? 'Qwen3.6 · 공통 reflector' : 'Qwen3.6 · Isolated';
  $('.cohort-note span:last-child').textContent = sharedView ? 'Fresh 1 epoch · medium · 36 runs' : '원본 Epoch 1 → Epoch 2';
  $('.sidebar-bottom a').href = sharedView ? 'http://166.104.110.49:8878/#experiment/exp-20261111-8497707d' : 'http://166.104.110.49:8878/#experiment/exp-20261109-68f88079';
  $('.top-actions a').href = sharedView ? '/api/shared-reflection' : '/download/settings.json';
  $$('#navigation a').forEach(a => a.classList.toggle('active', a.dataset.view === view));
  $('#sidebar').classList.remove('open');
  window.scrollTo({
    top: 0,
    behavior: 'instant'
  });
  if (view === 'tasks') {
    if (parts[1]) {
      state.task = null;
      loadCatalog().then(() => selectTask(parts[0], decodeURIComponent(parts.slice(1).join('/')))).catch(showError);
    } else loadCatalog().catch(showError);
  }
  if (view === 'shared-reflection') {
    refreshCheckpointEvaluation().catch(showError);
    refreshSharedReflection().catch(showError);
  }
  if (view === 'overview') refreshCheckpointEvaluation().catch(showError);
  if (view === 'llm-calls') window.renderLLMCalls().catch(showError);
}

function showError(e) {
  console.error(e);
  toast('자료를 불러오지 못했습니다. ' + e.message);
}

function toast(t) {
  $('#toast').textContent = t;
  $('#toast').classList.add('visible');
  setTimeout(() => $('#toast').classList.remove('visible'), 2500);
}
async function showSource(id) {
  const s = await api('/api/source/' + encodeURIComponent(id));
  $('#source-title').textContent = s.title;
  $('#source-path').textContent = s.path;
  $('#source-sha').textContent = 'SHA-256 ' + s.sha256;
  $('#source-text').textContent = s.text || '이 자료는 실행 manifest 또는 config입니다. 해당 실행의 설정 화면과 설정 JSON에서 값을 확인할 수 있습니다.';
  $('#source-dialog').showModal();
}
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    toast('복사했습니다.');
  } catch {
    toast('텍스트를 선택해 복사하세요.');
  }
}
document.addEventListener('click', e => {
  const b = e.target.closest('button,a');
  if (!b) return;
  if (b.dataset.source) showSource(b.dataset.source).catch(showError);
  if (b.hasAttribute('data-copy')) copyText(b.closest('.code-box').querySelector('pre').textContent);
  if (b.dataset.epoch) {
    state.epoch = Number(b.dataset.epoch);
    route();
  }
  if (b.dataset.method) {
    state.method = b.dataset.method;
    location.hash = 'methods/' + state.method;
  }
  if (b.dataset.step) {
    const i = Number(b.dataset.step),
      s = methodContent[state.method].steps[i];
    $('#flow-explanation').innerHTML = `<b>${esc(s[0])}</b> ${esc(s[2])}<div class="source-ref-row">${source(s[3])}</div>`;
    $$('.flow-step').forEach((x, n) => x.classList.toggle('selected', n === i));
  }
  if (b.dataset.splitBench) {
    state.splitBench = b.dataset.splitBench;
    route();
  }
  if (b.dataset.task) {
    const [bench, id] = b.dataset.task.split('/');
    selectTask(bench, id).catch(showError);
  }
  if (b.dataset.taskTab) {
    state.taskTab = b.dataset.taskTab;
    renderTask();
  }
  if (b.dataset.page) {
    state.page += Number(b.dataset.page);
    loadCatalog().catch(showError);
  }
  if (b.dataset.taskJump) {
    state.task = null;
    location.hash = 'tasks/' + b.dataset.taskJump;
  }
  if (b.dataset.promptMethod) {
    state.promptMethod = b.dataset.promptMethod;
    route();
  }
  if (b.dataset.run) {
    state.run = b.dataset.run;
    location.hash = 'runs/' + encodeURIComponent(state.run);
  }
});
let searchTimer;
document.addEventListener('input', e => {
  if (e.target.id === 'task-search') {
    state.taskQuery = e.target.value;
    state.page = 1;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadCatalog().catch(showError), 220);
  }
});
document.addEventListener('change', e => {
  const id = e.target.id,
    v = e.target.value;
  if (id === 'method-benchmark') {
    state.bench = v;
    route();
  }
  if (id === 'order-method') {
    state.methodForTask = v;
    route();
  }
  if (id === 'order-seed') {
    state.seed = Number(v);
    route();
  }
  if (id === 'task-benchmark') {
    state.taskBench = v;
    state.page = 1;
    loadCatalog().catch(showError);
  }
  if (id === 'task-split') {
    state.taskSplit = v;
    state.page = 1;
    loadCatalog().catch(showError);
  }
  if (id === 'task-selected') {
    state.selectedOnly = v;
    state.page = 1;
    loadCatalog().catch(showError);
  }
  if (id === 'task-order-method') {
    state.methodForTask = v;
    renderTask();
  }
  if (id === 'task-order-seed') {
    state.seed = Number(v);
    renderTask();
  }
});
$('#menu-button').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
$('#source-close').addEventListener('click', () => $('#source-dialog').close());
$('#source-dialog').addEventListener('click', e => {
  if (e.target === $('#source-dialog')) {
    const r = e.target.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) e.target.close();
  }
});
async function refreshLive() {
    try {
      state.live = await api('/api/live');
      if ($('#shared-reflection-content')) await refreshSharedReflection();
      const online = state.live.services.vllm?.online && (state.live.services.parser?.online || state.live.services.reasoning_bridge?.online);
      $('#service-status').classList.toggle('online', online);
      $('#service-status').innerHTML = `<i></i>${online?'모델 연결됨':'모델 현재 오프라인'}`;
      if ($('#runtime-status')) $('#runtime-status').innerHTML = serviceSummary();
    $$('[data-run-status]').forEach(node => {
      const run = state.data.runs.find(r => r.id === node.dataset.runStatus);
      const status = runStatus(run);
      node.innerHTML = pill({completed:'학습·Test 완료',error:'오류',pending:'대기',paused:'중지',running:'실행 중'}[status] || status,
        {completed:'green',error:'red',running:'blue',paused:'amber'}[status]);
    });
    } catch (e) {
      console.error(e);
    }
  }
  (async () => {
    try {
      state.data = await api('/api/settings');
      $('#snapshot-date').textContent = '자료 생성 ' + new Date(state.data.generated_at).toLocaleString('ko-KR', {
        timeZone: 'Asia/Seoul'
      }) + ' KST';
      await refreshLive();
      route();
      window.addEventListener('hashchange', route);
      setInterval(refreshLive, 30000);
      setInterval(() => refreshCheckpointEvaluation().catch(showError), 5000);
    } catch (e) {
      $('#main').innerHTML = '<div class="empty">설정 자료를 불러오지 못했습니다. ' + esc(e.message) + '</div>';
      console.error(e);
    }
  })();
