'use strict';
(() => {
  const view = {job: 'ace_aud_bfcl_s42', kind: '', split: '', q: '', page: 1, source: null, line: null,
    jobs: [], records: [], call: null, tab: 'reading'};
  const modules = {actor: 'Actor', reflection: 'Reflector', mutation: 'Mutator / Curator',
    instruction_proposal: 'GEPA · Proposal', analyst: 'SkillOpt · Analyst', rewriter: 'SkillOpt · Rewrite',
    ranking: 'SkillOpt · Ranking', analysis: 'Analysis', judge: 'Judge'};
  let catalogSerial = 0, recordSerial = 0, detailSerial = 0, searchTimer;
  const present = value => value === null || value === undefined ? '미기록' : fmt(value);
  const when = value => value ? new Date(typeof value === 'number' ? value * 1000 : value)
    .toLocaleString('ko-KR', {timeZone: 'Asia/Seoul', hour12: false}) : '시각 미기록';
  const url = (endpoint, params = {}) => '/api/llm/' + endpoint + '?' + new URLSearchParams(params);
  const selectedJob = () => view.jobs.find(j => j.id === view.job);
  const stage = split => ({train: 'Train · Replay', val: 'Validation', test: '최종 Test'}[split] || split);

  function filters() {
    const job = selectedJob();
    const options = (values, selected) => values.map(([key, label]) =>
      `<option value="${esc(key)}" ${String(selected) === String(key) ? 'selected' : ''}>${esc(label)}</option>`).join('');
    return `<div class="llm-filters panel">
      <label>방법론<select id="llm-method">${options(Object.entries(names), job.method)}</select></label>
      <label>Benchmark<select id="llm-benchmark">${options(Object.entries(bshort), job.benchmark)}</select></label>
      <label>Seed<select id="llm-seed">${options([42,43,44].map(s => [s,s]), job.seed)}</select></label>
      <label>역할<select id="llm-kind">${options([['','모두'],['actor','Actor'],['analysis','Reflection · Mutation · Optimizer']], view.kind)}</select></label>
      <label>단계<select id="llm-split">${options([['','모두'],['train','Train · Replay'],['val','Validation'],['test','최종 Test']], view.split)}</select></label>
      <button class="button" id="llm-refresh">목록 새로고침 ↻</button>
    </div>`;
  }

  window.renderLLMCalls = async () => {
    const host = $('#llm-calls-content');
    if (!host) return;
    const data = await api('/api/llm/jobs');
    if (!$('#llm-calls-content')) return;
    view.jobs = data.jobs;
    if (!view.jobs.length) { host.textContent = '실험 기록이 아직 없습니다.'; return; }
    const parts = location.hash.slice(1).split('/');
    const linked = parts[1] && view.jobs.find(j => j.id === parts[1]);
    if (linked) { view.job = linked.id; view.source = parts[2] || null; view.line = parts[3] ? Number(parts[3]) : null; }
    if (!selectedJob()) view.job = view.jobs[0].id;
    host.classList.remove('loading');
    host.innerHTML = heading('RECORDED MODEL RETURNS / FRESH 1 EPOCH', 'LLM 응답 · Reasoning',
      '실험에서 Qwen이 실제 반환한 추론, 행동, reflection과 수정안을 읽습니다. 요청·응답 원문과 토큰 계측을 같은 호출에서 확인할 수 있습니다.',
      '<a class="button" href="#shared-reflection">실험 현황 →</a>') + filters() +
      `<div class="llm-layout"><aside class="llm-library panel"><div class="llm-library-heading"><span class="small-caps">기록 선택</span><h2>Trajectory · 분석 호출</h2><p id="llm-source-count">불러오는 중</p></div>
        <div class="llm-search"><input id="llm-search" type="search" placeholder="Task ID · replay · 모듈 검색" aria-label="기록 검색" value="${esc(view.q)}"></div>
        <div id="llm-source-list" aria-live="polite"></div><div id="llm-pagination"></div></aside>
        <div class="llm-workspace"><div class="llm-call-strip panel" id="llm-records">왼쪽에서 기록을 선택하세요.</div>
        <div id="llm-detail" aria-live="polite" class="llm-detail"><div class="panel llm-empty">저장된 호출을 선택하면 reasoning과 최종 출력이 나란히 나타납니다.</div></div></div></div>`;
    await loadSources(view.source);
  };

  async function loadSources(preferred) {
    ++recordSerial;
    ++detailSerial;
    const serial = ++catalogSerial, job = view.job;
    const data = await api(url('sources', {job, kind: view.kind, split: view.split, q: view.q, page: view.page}));
    if (serial !== catalogSerial || !$('#llm-source-list')) return;
    const start = (data.page - 1) * data.page_size;
    $('#llm-source-count').textContent = `${fmt(data.total)}개 기록 묶음 · ${data.total ? start + 1 : 0}–${Math.min(start + data.page_size, data.total)}`;
    $('#llm-source-list').innerHTML = data.items.length ? data.items.map(source =>
      `<button class="llm-source ${source.id === view.source ? 'selected' : ''}" data-llm-source="${source.id}">
        <span class="llm-source-badges">${pill(source.kind === 'actor' ? 'ACTOR' : (modules[source.stage] || 'ANALYSIS'), source.kind === 'actor' ? 'blue' : 'purple')}${pill(stage(source.split))}</span>
        <strong>${esc(source.task_id || source.label || '모델 호출')}</strong><span class="llm-source-label">${esc(source.label || source.path.split('/').slice(-5,-2).join(' / '))}</span>
        <time>${esc(when(source.modified_at))}</time></button>`).join('') : '<div class="llm-empty">조건에 맞는 기록이 없습니다.</div>';
    $('#llm-pagination').innerHTML = `<button class="button small" data-llm-page="-1" ${data.page === 1 ? 'disabled' : ''}>← 이전</button><span>${data.page} / ${Math.max(1, Math.ceil(data.total / data.page_size))}</span><button class="button small" data-llm-page="1" ${start + data.page_size >= data.total ? 'disabled' : ''}>다음 →</button>`;
    const choice = preferred || (data.items.some(s => s.id === view.source) ? view.source : data.items[0]?.id);
    if (choice) await loadRecords(choice);
    else {
      view.source = null;
      $('#llm-records').textContent = '기록이 없습니다.';
      $('#llm-detail').innerHTML = '';
    }
  }

  async function loadRecords(source) {
    const serial = ++recordSerial, job = view.job;
    const keep = source === view.source ? view.line : null;
    view.source = source;
    $$('.llm-source').forEach(b => b.classList.toggle('selected', b.dataset.llmSource === source));
    $('#llm-records').innerHTML = '<div class="llm-empty">실제 호출과 토큰 계측을 읽고 있습니다…</div>';
    const data = await api(url('records', {job, source}));
    if (serial !== recordSerial || job !== view.job || !$('#llm-records')) return;
    view.records = data.items;
    $('#llm-records').innerHTML = `<div class="llm-strip-heading"><strong>${esc(data.source.task_id || data.source.label || '모델 호출')}</strong><span>${stage(data.source.split)} · 호출 ${data.items.length}개</span></div><div class="llm-call-buttons">${data.items.map((call, i) =>
      `<button class="llm-call-button" data-llm-line="${call.line}"><span>#${i + 1} ${esc(modules[call.module] || call.module)}</span><small>추론 ${esc(present(call.reasoning_tokens))} · ${esc(call.finish_reason || call.status)}</small></button>`).join('')}</div>`;
    const chosen = data.items.find(v => v.line === keep) || data.items[0];
    if (chosen) await loadCall(chosen.line);
    else $('#llm-detail').innerHTML = '<div class="panel llm-empty">아직 응답이 저장되지 않았습니다. 목록을 새로고침해 다시 확인하세요.</div>';
  }

  async function loadCall(line) {
    const serial = ++detailSerial, job = view.job, source = view.source;
    view.line = line;
    const call = await api(url('call', {job, source, line}));
    if (serial !== detailSerial || job !== view.job || source !== view.source || !$('#llm-detail')) return;
    view.call = call;
    $$('.llm-call-button').forEach(b => b.classList.toggle('selected', Number(b.dataset.llmLine) === line));
    renderCall();
  }

  function renderCall() {
    const call = view.call, job = selectedJob(), observed = call.observed, message = call.messages[0] || {};
    const reason = call.messages.map(m => m.reasoning).filter(Boolean).join('\n\n');
    const download = url('call', {job: view.job, source: view.source, line: view.line, download: 1});
    const total = (call.reasoning_tokens || 0) + (call.visible_tokens || 0);
    const ratio = total ? Math.min(100, call.reasoning_tokens / total * 100) : 0;
    let body;
    if (view.tab === 'request') {
      body = `<div class="panel panel-body">${code(call.request, '저장된 요청 · messages / tools / parameters')}<p class="caption">호출 추적기에 없는 인자는 미기록입니다. 아래 호출 정보의 bridge 관측값과 함께 확인하세요.</p></div>`;
    } else if (view.tab === 'response') {
      body = `<div class="panel panel-body">${code(call.response, '저장된 응답 원문')}<p class="caption">일부 LiteLLM 기록은 choices를 SDK 객체의 문자열로 저장합니다. 읽기 화면은 이를 데이터로 복원해 표시합니다.</p></div>`;
    } else if (view.tab === 'metadata') {
      body = `<div class="panel panel-body">${params([
        ['temperature', observed.temperature ?? call.request.temperature, 'bridge 관측값 우선'],
        ['reasoning_effort', observed.reasoning_effort ?? call.request.reasoning_effort ?? '미기록', '요청에 전달한 effort'],
        ['enable_thinking', observed.enable_thinking ?? call.request.extra_body?.chat_template_kwargs?.enable_thinking ?? '미기록', '실제 thinking 설정'],
        ['추론 상한', observed.reasoning_limit ?? '미기록', '요청당 reasoning token 한도'],
        ['비추론 상한', observed.visible_limit ?? '미기록', 'actor 2,048 · 분석 8,192'],
        ['응답 ID', call.request_id || '미기록', '응답 원문과 bridge 계측의 연결 키']
      ])}${code({observed: call.observed, usage: call.usage, source: call.provenance}, '호출 계측과 원본 위치')}</div>`;
    } else {
      const outputs = call.messages.map(m => {
        let content = m.content;
        if (typeof content === 'string') { try { content = pretty(JSON.parse(content)); } catch {} }
        const tools = m.tool_calls.map(tool => {
          const f = tool.function || {}; let args = f.arguments;
          if (typeof args === 'string') { try { args = JSON.parse(args); } catch {} }
          return `<div class="llm-tool"><span class="small-caps">TOOL CALL</span><strong>${esc(f.name || tool.type)}</strong><pre>${esc(typeof args === 'string' ? args : pretty(args))}</pre></div>`;
        }).join('');
        return `${tools}${content ? `<pre class="llm-text">${esc(typeof content === 'string' ? content : pretty(content))}</pre>` : ''}${m.function_call ? code(m.function_call, 'Function call') : ''}`;
      }).join('');
      body = `<div class="llm-reading-grid"><section class="llm-reading llm-reasoning panel"><div class="llm-reading-title"><span class="llm-dot"></span><h2>Reasoning</h2><span>${esc(present(call.reasoning_tokens))} tokens</span></div><pre class="llm-text">${esc(reason || '이 기록에는 reasoning 본문이 없습니다.')}</pre></section>
        <section class="llm-reading llm-output panel"><div class="llm-reading-title"><span class="llm-dot"></span><h2>최종 출력</h2><span>${esc(present(call.visible_tokens))} tokens</span></div>${outputs || '<div class="llm-empty">최종 출력이 없거나 응답 대기 중입니다.</div>'}</section></div>`;
    }
    $('#llm-detail').innerHTML = `<div class="panel llm-call-heading"><div><div class="llm-source-badges">${pill(names[job.method], 'blue')}${pill(bshort[job.benchmark])}${pill('seed ' + job.seed)}${pill(stage(call.source.split))}</div><h2>${esc(modules[call.module] || call.module)}</h2><p>${esc(when(call.timestamp))} · ${esc(call.source.task_id || call.source.label || '')}</p></div><div class="llm-call-actions"><button class="button small" id="llm-copy-link">호출 링크 복사</button><a class="button small" href="${esc(download)}">JSON ↓</a></div></div>
      <div class="llm-metrics panel"><div><label>Input</label><strong>${esc(present(call.usage.prompt_tokens))}</strong></div><div><label>Reasoning</label><strong>${esc(present(call.reasoning_tokens))}</strong></div><div><label>최종 출력</label><strong>${esc(present(call.visible_tokens))}</strong></div><div><label>Finish reason</label><strong>${esc(message.finish_reason || call.status)}</strong></div><div><label>응답 시간</label><strong>${observed.seconds != null ? observed.seconds.toFixed(1) + 's' : call.elapsed_seconds != null ? call.elapsed_seconds.toFixed(1) + 's' : '미기록'}</strong></div></div>
      ${total ? `<div class="llm-token-ratio" aria-label="Output 중 reasoning ${ratio.toFixed(1)}퍼센트"><span style="width:${ratio}%"></span></div>` : ''}
      <div class="llm-tabs" role="group" aria-label="호출 내용">${[['reading','Reasoning + 최종 출력'],['request','요청'],['response','원본 응답'],['metadata','호출 정보']].map(([tab,label]) => `<button data-llm-tab="${tab}" class="${view.tab === tab ? 'active' : ''}">${label}</button>`).join('')}</div>
      ${call.error ? `<div class="audit-note"><strong>기록된 오류</strong><span>${esc(call.error)}</span></div>` : ''}${body}<p class="llm-provenance">원본: ${esc(call.source.path)} · line ${call.line + 1}</p>`;
  }

  document.addEventListener('click', event => {
    if (!$('#llm-calls-content')) return;
    const b = event.target.closest('button');
    if (!b || b.disabled) return;
    if (b.dataset.llmSource) { view.line = null; loadRecords(b.dataset.llmSource).catch(showError); }
    if (b.dataset.llmLine !== undefined) loadCall(Number(b.dataset.llmLine)).catch(showError);
    if (b.dataset.llmTab) { view.tab = b.dataset.llmTab; renderCall(); }
    if (b.dataset.llmPage) { view.page += Number(b.dataset.llmPage); loadSources().catch(showError); }
    if (b.id === 'llm-refresh') loadSources(view.source).catch(showError);
    if (b.id === 'llm-copy-link') copyText(location.origin + location.pathname + '#llm-calls/' +
      [view.job, view.source, view.line].map(encodeURIComponent).join('/'));
  });
  document.addEventListener('change', event => {
    if (!$('#llm-calls-content') || !event.target.id.startsWith('llm-')) return;
    const job = selectedJob();
    if (['llm-method','llm-benchmark','llm-seed'].includes(event.target.id)) {
      const method = $('#llm-method').value, benchmark = $('#llm-benchmark').value, seed = Number($('#llm-seed').value);
      view.job = view.jobs.find(j => j.method === method && j.benchmark === benchmark && j.seed === seed)?.id || job.id;
    }
    view.kind = $('#llm-kind').value; view.split = $('#llm-split').value;
    view.source = null; view.line = null; view.page = 1;
    loadSources().catch(showError);
  });
  document.addEventListener('input', event => {
    if (event.target.id !== 'llm-search') return;
    view.q = event.target.value; view.page = 1; view.source = null; view.line = null;
    clearTimeout(searchTimer); searchTimer = setTimeout(() => loadSources().catch(showError), 250);
  });
})();
