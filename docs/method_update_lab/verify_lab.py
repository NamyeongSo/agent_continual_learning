# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Exercise the actual HTML in Chromium, including update branches and offline use."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

PROJECT = Path(__file__).resolve().parent
DIST = PROJECT / "dist"
ARTIFACTS = PROJECT / "qa-artifacts"


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    checks = []
    errors = []
    network = []

    def passed(label: str) -> None:
        checks.append(label)
        print(f"PASS {label}", flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: network.append(request.url) if request.url.startswith("http") else None)

        def open_method(method: str) -> None:
            page.goto((DIST / f"{method}.html").as_uri())
            expect(page.locator("h1")).to_be_visible()

        def finish() -> dict:
            page.locator("#jump-last").click()
            return page.evaluate("MemoryLabUI.getState().model.final")

        page.goto((DIST / "index.html").as_uri())
        expect(page.locator(".method-card")).to_have_count(4)
        page.screenshot(path=str(ARTIFACTS / "overview-desktop.png"), full_page=True)
        page.locator('.method-card[href="a_mem.html"]').click()
        expect(page.locator("h1")).to_have_text("A-MEM")
        passed("overview links open the four method pages")

        for branch in ["STRENGTHEN_AND_UPDATE", "STRENGTHEN", "UPDATE_NEIGHBOR", "NO_EVOLUTION"]:
            open_method("a_mem")
            page.locator("#decision").select_option(branch)
            final = finish()
            assert len(final["notes"]) == 3
            assert len(final["notes"][2]["links"]) == (2 if "STRENGTHEN" in branch else 0)
            baseline = page.evaluate("MemoryLabUI.getState().model.before")
            assert final["notes"][0]["content"] == baseline["notes"][0]["content"]
            assert (final["notes"][0]["context"] != baseline["notes"][0]["context"]) == (
                branch in ["STRENGTHEN_AND_UPDATE", "UPDATE_NEIGHBOR"]
            )
            passed(f"A-MEM {branch}: note addition and the correct mutation scope")
        page.locator("#decision").select_option("STRENGTHEN_AND_UPDATE")
        page.locator("#consolidate").check()
        assert finish()["index_rebuilt"]
        page.locator("#only-changes").check()
        expect(page.locator("#after-json")).to_contain_text("evo_count")
        passed("A-MEM 100th evolution rebuild and field-only diff")
        page.locator('[data-inspect="E0"]').click()
        expect(page.locator("#state-inspect")).to_contain_text("E0 · 현재 값")
        page.screenshot(path=str(ARTIFACTS / "a-mem-desktop.png"), full_page=True)
        passed("graph node inspection")

        for action in ["merge", "add", "discard"]:
            open_method("autoskill")
            page.locator("#decision").select_option(action)
            final = finish()
            assert len(final["skills"]) == (2 if action == "add" else 1)
            assert final["skills"][0]["version"] == ("0.1.1" if action == "merge" else "0.1.0")
            passed(f"AutoSkill {action}: skill count and version")
        page.locator("#decision").select_option("merge")
        page.locator("#similarity").select_option("0.2")
        assert len(finish()["skills"]) == 2
        passed("AutoSkill merge without a comparison target becomes add")
        page.locator("#bm25-weight").fill("1")
        page.locator("#retrieval-threshold").fill("0.9")
        expect(page.locator("#score-results")).to_contain_text("1 / 2개")
        passed("hybrid search weighting changes rank and threshold admission")

        open_method("harness")
        final = finish()
        assert final["version"] == 1 and len(final["ops"]) == 5
        assert {s["name"] for s in final["skills"]} == {"목록 집계", "집계 범위 검증"}
        passed("Harness edits prompt, memory, and all three skill operations")
        page.locator("#rollback").check()
        final = finish()
        before = page.evaluate("MemoryLabUI.getState().model.before")
        assert final == before
        passed("Harness outer exception restores the full baseline")
        page.locator("#rollback").uncheck()
        for control in ["prompt", "memory", "skills"]:
            page.locator(f"#{control}").uncheck()
        assert finish() == before
        passed("Harness no-op does not commit a new version")

        for verdict in ["success", "failure"]:
            open_method("reasoning_bank")
            page.locator("#verdict").select_option(verdict)
            final = finish()
            assert len(final["entries"]) == 2
            assert final["entries"][1]["status"] == ("success" if verdict == "success" else "fail")
            assert len(final["entries"][1]["memory_items"]) == 1
            passed(f"ReasoningBank {verdict}: append entry with the selected lesson branch")
        page.locator("#spacing").check()
        assert len(finish()["entries"][1]["memory_items"]) == 4
        passed("ReasoningBank blank-line parser exposes four strings from one logical item")
        page.locator("#embedding").uncheck()
        assert len(finish()["entries"]) == 1
        passed("ReasoningBank missing query embedding skips learning")

        open_method("a_mem")
        page.locator("#next").click()
        expect(page.locator("#step-counter")).to_have_text("02 / 06")
        page.locator("#reset").click()
        page.locator("#main").focus()
        page.keyboard.press("ArrowRight")
        expect(page.locator("#step-counter")).to_have_text("02 / 06")
        page.locator("#scrubber").fill("4")
        expect(page.locator("#step-counter")).to_have_text("05 / 06")
        page.locator("#prev").click()
        expect(page.locator("#step-counter")).to_have_text("04 / 06")
        page.locator("#reset").click()
        page.locator("#play").click()
        page.wait_for_function("MemoryLabUI.getState().step > 0")
        page.locator("#play").click()
        expect(page.locator("#play")).to_have_attribute("aria-pressed", "false")
        passed("previous, next, reset, scrubber, keyboard and autoplay controls")

        page.locator("#source-select").select_option("EVOLUTION_DECISION_PROMPT")
        expect(page.locator("#source-code")).to_contain_text("STRENGTHEN_AND_UPDATE")
        expect(page.locator("#source-path")).to_contain_text("a_mem/prompts.py")
        with page.expect_download() as download_info:
            page.locator("#export").click()
        saved = ARTIFACTS / "export.json"
        download_info.value.save_as(saved)
        exported = json.loads(saved.read_text())
        assert exported["method"] == "a_mem" and exported["model_called"] is False
        passed("exact source viewer and stage JSON download")

        page.goto((DIST / "standalone.html").as_uri())
        page.locator('.method-card[href="#harness"]').click()
        expect(page.locator("h1")).to_have_text("Harness")
        page.locator('.text-button[href="#sources"]').click()
        expect(page.locator("h1")).to_have_text("Harness")
        page.locator('.nav a[href="#reasoning_bank"]').click()
        expect(page.locator("h1")).to_have_text("ReasoningBank")
        page.locator("#jump-last").click()
        expect(page.locator("#context-content pre")).to_be_visible()
        passed("standalone HTML works offline, including section anchors and method routing")

        page.set_viewport_size({"width": 390, "height": 844})
        for name in ["index", "a_mem", "autoskill", "harness", "reasoning_bank"]:
            page.goto((DIST / f"{name}.html").as_uri())
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"), name
            if name != "index":
                page.locator("#jump-last").click()
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"), name
                assert page.evaluate("""() => {
                    const active = document.querySelector('.step-item.active').getBoundingClientRect();
                    const strip = document.querySelector('#stepper').getBoundingClientRect();
                    return active.right <= strip.right + 1 && active.left >= strip.left - 1;
                }"""), f"{name}: active step is outside the mobile timeline"
            page.screenshot(path=str(ARTIFACTS / f"{name}-mobile.png"), full_page=True)
        passed("all five pages fit a 390px mobile viewport before and after the update")
        assert not errors, errors
        assert not network, network
        passed("no browser JavaScript errors and no external network requests")
        browser.close()
    result = {"passed": len(checks), "checks": checks, "browser_errors": errors, "external_requests": network}
    (ARTIFACTS / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Verified {len(checks)} interaction and state checks. Artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
