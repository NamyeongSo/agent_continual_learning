# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Optional browser regressions for partial checkpoint score reporting.

Run with Playwright installed and its Chromium available. A custom browser can
be selected with PLAYWRIGHT_CHROMIUM_EXECUTABLE.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
STATIC = Path(__file__).resolve().parents[2] / "scripts/research_journal/static"


@pytest.fixture
def page():
    with playwright.sync_playwright() as driver:
        options = {"headless": True}
        if executable := os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"):
            options["executable_path"] = executable
        browser = driver.chromium.launch(**options)
        page = browser.new_page()
        page.set_content('<div id="chart"></div>')
        page.add_script_tag(path=str(STATIC / "checkpoint_charts.js"))
        yield page
        browser.close()


def row(point, score=None, *, done=0, total=50, errors=0, alias=None, stream="stream"):
    return {"id": f"{stream}_cp{point:03d}", "stream": stream, "method": "premise", "benchmark": "appworld",
            "seed": 42, "point": point, "variant": "cohort", "state": "pending", "alias": alias,
            "splits": {"train": {"score": score, "done": done, "total": total, "errors": errors},
                       "val": {"score": None, "done": 0, "total": 50, "errors": 0},
                       "test": {"score": None, "done": 0, "total": 50, "errors": 0}}}


def mount(page, rows):
    page.evaluate("rows => CheckpointCharts.mount(document.getElementById('chart'), rows)", rows)


def test_partial_zero_is_visible_missing_is_absent_and_counts_exclude_errors(page):
    mount(page, [row(5, 0, done=10, errors=2), row(10), row(15, 1, done=50)])
    points = page.locator("[data-chart-point]")
    assert points.count() == 2
    zero = page.locator('[data-chart-point="5"]')
    assert "partial" in zero.get_attribute("class")
    zero.focus()
    text = page.locator(".live-cp-chart-readout").inner_text()
    assert "0.00%" in text and "10/50" in text and "오류 2개 제외" in text
    assert "complete" in page.locator('[data-chart-point="15"]').get_attribute("class")
    # A missing checkpoint must break the line, rather than imply an observed zero or interpolate it.
    path = page.locator('[data-chart-series="train"] .live-cp-chart-line').get_attribute("d")
    assert path.count("M") == 2 and "L" not in path
    assert "TGC" in page.locator(".live-cp-chart-note").inner_text()


def test_unsorted_and_alias_checkpoints_keep_numeric_axis_and_reuse_provenance(page):
    mount(page, [row(15, 0.5, done=50, alias="stream_cp005"), row(5, 0.5, done=50)])
    assert page.locator(".live-cp-chart-tick").all_text_contents() == ["5", "15"]
    point = page.locator('[data-chart-point="15"]')
    assert point.locator("path").count() == 1
    point.focus()
    assert "동일 정책 005의 평가 재사용" in page.locator(".live-cp-chart-readout").inner_text()


def test_no_results_or_hidden_splits_do_not_invent_scores(page):
    mount(page, [row(5)])
    assert page.locator("[data-chart-point]").count() == 0
    assert "아직 측정된 점수가 없습니다" in page.locator("svg").text_content()
    for split in ["train", "val", "test"]:
        page.locator(f'[data-chart-toggle="{split}"]').uncheck()
    assert "표시할 split을 선택" in page.locator("svg").text_content()


def test_live_checkpoint_overview_refreshes_and_opens_hidden_details_on_mobile(page, tmp_path):
    queue = {"id": "current", "live_overview": True, "phase": "running", "workers": 208,
             "checkpoint_interval": 10, "checkpoints": 153, "expected_checkpoints": 156,
             "initial_reused_outcomes": 3454, "skipped": 48,
             "task_counts": {"completed": 4000, "pending": 10000, "running": 208, "error": 0},
             "rows": [row(10, .5, done=25)], "updated_at": "2026-11-12T07:30:00+00:00",
             "gpu_queries": {"active": {"8007": 128, "8006": 80}, "waiting": 5,
                             "per_endpoint_query_limits": {"8007": 128, "8006": 80}}}
    data = {"generated_at": queue["updated_at"], "summary": {}, "batches": [], "checkpoint_queues": [queue]}
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/live-progress", lambda route: route.fulfill(json=data))
    page.route("http://journal.test/", lambda route: route.fulfill(body='<div id="live"></div>'))
    page.goto("http://journal.test/")
    page.add_style_tag(path=str(STATIC / "style.css"))
    page.add_style_tag(path=str(STATIC / "live.css"))
    page.evaluate("""localStorage.setItem('exgentic.live.hidden.v1', JSON.stringify([
        {kind:'module', id:'checkpoints', label:'CP 평가'}]))""")
    page.add_script_tag(path=str(STATIC / "checkpoint_charts.js"))
    page.add_script_tag(path=str(STATIC / "live.js"))
    page.evaluate("LiveExperiments.mount(document.getElementById('live'))")
    card = page.locator("[data-evaluation-overview]")
    playwright.expect(card).to_be_visible()
    for text in ["153 / 156", "3,454", "포트 8007", "128 / 128", "80 / 80"]:
        playwright.expect(card).to_contain_text(text)
    assert "CP 평가 208개 실행" in page.locator(".live-phase-summary").inner_text()
    assert page.locator('[data-live-module="checkpoints"]').is_hidden()
    card.screenshot(path=str(tmp_path / "overview-desktop.png"))
    card.locator("[data-live-open-queue]").click()
    playwright.expect(page.locator('[data-live-module="checkpoints"]')).to_be_visible()
    queue["task_counts"]["completed"] = 4017
    page.locator("[data-live-refresh]").click()
    playwright.expect(card.locator(".live-eval-stats .live-stat").first).to_contain_text("4,017")
    page.set_viewport_size({"width": 390, "height": 844})
    card.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    card.screenshot(path=str(tmp_path / "overview-mobile.png"))
    assert not errors


def test_live_refresh_updates_partial_scores_and_preserves_open_state_and_split_visibility(page):
    rows = [row(5, 0, done=10), row(10, 0.4, done=25)]
    data = {"generated_at": "2026-10-30T00:00:00+00:00", "summary": {"running": 0}, "batches": [],
            "checkpoint_queues": [{"id": "queue", "rows": rows, "task_counts": {}, "phase": "stopped",
                                   "workers": 100, "checkpoints": 2, "skipped": 0,
                                   "updated_at": "2026-10-30T00:00:00+00:00"}]}
    page.route("**/api/live-progress", lambda route: route.fulfill(json=data))
    # Use a same-origin document without a real server or changes to any experiment.
    page.route("http://journal.test/", lambda route: route.fulfill(body='<div id="live"></div>'))
    page.goto("http://journal.test/")
    page.add_script_tag(path=str(STATIC / "checkpoint_charts.js"))
    page.add_script_tag(path=str(STATIC / "live.js"))
    page.evaluate("LiveExperiments.mount(document.getElementById('live'))")
    details = page.locator("details[data-cp-stream]")
    playwright.expect(details).to_have_count(1)
    assert page.locator(".live-cp-chart-svg").count() == 0
    details.locator("summary").click()
    playwright.expect(page.locator(".live-cp-chart-svg")).to_be_visible()
    page.locator('[data-chart-toggle="test"]').uncheck()
    rows[0]["splits"]["train"].update(score=0.6, done=50)
    page.locator("[data-live-refresh]").click()
    point = page.locator('[data-chart-point="5"]')
    playwright.expect(point).to_have_class("live-cp-chart-point complete")
    assert details.get_attribute("open") is not None
    assert not page.locator('[data-chart-toggle="test"]').is_checked()
    point.focus()
    assert "60.00%" in page.locator(".live-cp-chart-readout").inner_text()
    details.locator("summary").click()
    page.locator("[data-live-refresh]").click()
    playwright.expect(details).not_to_have_attribute("open", "")
