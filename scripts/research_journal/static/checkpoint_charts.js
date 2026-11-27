"use strict";

window.CheckpointCharts = (() => {
  const splits = [["train", "Train (CP 평가)"], ["val", "Offline-val"], ["test", "Test (CP 평가)"]];
  const esc = text => String(text ?? "").replace(/[&<>"']/g,
    c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
  const width = 900, height = 320, left = 62, right = 24, top = 32, bottom = 58;
  const floor = height - bottom;

  function measured(value) {
    return value && typeof value.score === "number" && Number.isFinite(value.score)
      && value.score >= 0 && value.score <= 1 && value.done > 0 && value.total >= value.done;
  }

  function chart(rows, visible, label) {
    const points = [...rows].filter(row => Number.isFinite(row.point)).sort((a, b) => a.point - b.point);
    if (!points.length) return '<p class="live-cp-chart-empty">등록된 checkpoint가 없습니다.</p>';
    const min = points[0].point, max = points[points.length - 1].point;
    const x = point => min === max ? (left + width - right) / 2
      : left + (point - min) / (max - min) * (width - left - right);
    const y = score => floor - score * (floor - top);
    const ticks = [0, 25, 50, 75, 100].map(value => `<g class="live-cp-chart-grid">
      <line x1="${left}" x2="${width - right}" y1="${y(value / 100)}" y2="${y(value / 100)}" />
      <text x="${left - 12}" y="${y(value / 100) + 4}" text-anchor="end">${value}</text></g>`).join("");
    const stride = Math.max(1, Math.ceil(points.length / 12));
    const labels = points.filter((row, index) => index % stride === 0 || index === points.length - 1)
      .map(row => `<text class="live-cp-chart-tick" x="${x(row.point)}" y="${floor + 24}" text-anchor="middle">${row.point}</text>`).join("");
    let count = 0;
    const series = splits.filter(([split]) => visible.has(split)).map(([split, title]) => {
      let path = "", connected = false;
      const markers = [];
      for (const row of points) {
        const value = row.splits?.[split];
        if (!measured(value)) { connected = false; continue; }
        count++;
        const px = x(row.point), py = y(value.score);
        path += `${connected ? "L" : "M"}${px},${py} `;
        connected = true;
        const complete = value.done === value.total && !value.errors;
        const alias = row.alias ? ` · 동일 정책 ${String(row.alias).split("_cp").pop()}의 평가 재사용` : "";
        const message = `CP ${row.point} · ${title}: ${(100 * value.score).toFixed(2)}% · 측정 ${value.done}/${value.total} task`
          + ` · ${complete ? "전체 완료" : "집계 중: 완료분 평균"}`
          + (value.errors ? ` · 실행 오류 ${value.errors}개 제외` : "") + alias;
        const shape = row.alias
          ? `<path d="M${px},${py - 6} l6,6 l-6,6 l-6,-6 Z" />`
          : `<circle cx="${px}" cy="${py}" r="5" />`;
        markers.push(`<g class="live-cp-chart-point ${complete ? "complete" : "partial"}${row.alias ? " alias" : ""}"
          data-chart-point="${row.point}" data-chart-split="${split}" tabindex="0" role="img" aria-label="${esc(message)}">
          <title>${esc(message)}</title>${shape}</g>`);
      }
      return `<g class="live-cp-chart-series ${split}" data-chart-series="${split}">
        <path class="live-cp-chart-line" d="${path.trim()}" />${markers.join("")}</g>`;
    }).join("");
    return `<div class="live-cp-chart-scroll"><svg class="live-cp-chart-svg" viewBox="0 0 ${width} ${height}"
      role="group" aria-label="${esc(label)} · checkpoint별 split 점수 그래프">
      <text class="live-cp-chart-axis-title" x="${left}" y="18">점수 (%)</text>
      ${ticks}<line class="live-cp-chart-axis" x1="${left}" x2="${width - right}" y1="${floor}" y2="${floor}" />${labels}
      <text class="live-cp-chart-axis-title" x="${(left + width - right) / 2}" y="${height - 8}" text-anchor="middle">Checkpoint (학습 task 수)</text>
      ${series}${count ? "" : `<text class="live-cp-chart-empty" x="${(left + width - right) / 2}" y="${height / 2}"
        text-anchor="middle">${visible.size ? "아직 측정된 점수가 없습니다" : "표시할 split을 선택해 주세요"}</text>`}</svg></div>`;
  }

  function mount(target, rows, {label = "CP 평가", visible = new Set(splits.map(([split]) => split))} = {}) {
    target.innerHTML = `<figure class="live-cp-chart"><figcaption>Checkpoint별 평가 점수</figcaption>
      <div class="live-cp-chart-legend">${splits.map(([split, title]) => `<label class="${split}">
        <input type="checkbox" data-chart-toggle="${split}" ${visible.has(split) ? "checked" : ""}>
        <span class="live-cp-chart-swatch" aria-hidden="true"></span>${title}</label>`).join("")}</div>
      <div data-chart-plot>${chart(rows, visible, label)}</div>
      <p class="live-cp-chart-readout" role="status">점에 마우스를 올리거나 키보드로 선택하면 점수와 측정 건수를 볼 수 있습니다.</p>
      <p class="live-cp-chart-note">● 전체 완료 · ○ 집계 중 · ◇ 동일 정책 재사용. 미측정 값은 0점으로 채우지 않습니다.
      각 점은 현재 측정된 task의 평균이며 CP마다 측정 건수가 다를 수 있습니다.
      Train은 고정된 CP의 재평가 점수입니다.${rows.some(row => row.benchmark === "appworld") ? " AppWorld는 모든 split에서 TGC를 사용합니다." : ""}</p></figure>`;
    const plot = target.querySelector("[data-chart-plot]");
    const readout = target.querySelector(".live-cp-chart-readout");
    const inspect = event => {
      const point = event.target.closest("[data-chart-point]");
      if (point) readout.textContent = point.getAttribute("aria-label");
    };
    plot.addEventListener("pointerover", inspect);
    plot.addEventListener("focusin", inspect);
    target.querySelectorAll("[data-chart-toggle]").forEach(input => input.addEventListener("change", () => {
      if (input.checked) visible.add(input.dataset.chartToggle);
      else visible.delete(input.dataset.chartToggle);
      plot.innerHTML = chart(rows, visible, label);
      readout.textContent = "점에 마우스를 올리거나 키보드로 선택하면 점수와 측정 건수를 볼 수 있습니다.";
    }));
  }

  return {mount};
})();
