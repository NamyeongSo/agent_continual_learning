# SPDX-License-Identifier: Apache-2.0
"""Check actual selection pairing, blind responses, and durable human votes without model calls."""

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.ace.reflection_review import ReviewServer, ReviewStore, added_rules, build_pairs, digest


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def selected(tmp_path):
    jobs = []
    for mode in ("thinking", "nonthinking"):
        reflection_root = tmp_path / "outputs/reflection" / mode
        initial = "## RULES\n\n## OTHERS"
        write(reflection_root / "protocol.json", {"initial_playbook": initial, "playbook_edit_mode": "add_only"})
        for arm in ("without_gt", "with_gt"):
            directory = tmp_path / f"outputs/run/{mode}/{arm}"
            write(directory / "protocol.json", {"playbook_edit_mode": "add_only",
                                                "source_reflection_run": str(reflection_root), "task_orders": {
                str(seed): {"train": ["train_task"], "val": ["val_task"], "test": ["test_task"]}
                for seed in (42, 43)}})
            for seed in (42, 43):
                for strategy, advantage in (("best_advantage", 1.2), ("worst_advantage", -0.8)):
                    condition = {"id": f"s{seed}_{arm}_{strategy}", "mode": strategy, "seed": seed, "arm": arm}
                    text = f"{mode}/{arm}/{seed}: <script>alert('unsafe')</script> plain Reflection {advantage}"
                    source = reflection_root / "stage2" / f"{arm}_{seed}_{strategy}" / "reflection.json"
                    capture = source.parent / "capture.json"
                    write(capture, {"playbook": initial})
                    write(source, {"reflection_for_curator": text, "playbook_after_tags": initial,
                                   "source_capture_sha256": digest(capture.read_text())})
                    rule = f"Independent applied rule {advantage}\n```python\nif ready:\n    run()\n```"
                    updated = source.parent / "updated_context.txt"
                    updated.write_text(f"## RULES\n\n[r-00001] helpful=0 harmful=0 :: {rule}\n## OTHERS")
                    write(source.parent / "result.json", {"stage": 2, "task_id": "train_task", "source_repeat": 1,
                          "arm": arm, "reflection_path": str(source), "source_capture_path": str(capture),
                          "updated_context_path": str(updated), "updated_context_sha256": digest(updated.read_text()),
                          "curator_parsed": {"operations": [{"type": "ADD", "content": "NOT ACTUALLY ADDED"}]}})
                    candidate = {"task_id": "train_task", "reflection": text, "reflection_sha256": digest(text),
                                 "source_reflection_path": str(source), "source_repeat": 1, "source_id": "train_r1",
                                 "mean_pct": 0.7, "task": "The original task"}
                    write(directory / "conditions" / condition["id"] / "selections.json",
                          [{"candidate": candidate, "advantage": advantage}])
                    jobs.append({"path": str(directory), "condition": condition, "steps": 1})
    plan = tmp_path / "outputs/run/plan.json"
    write(plan, jobs)
    return tmp_path, (("qwen38", "Qwen", "run/plan.json"),), plan


@pytest.fixture
def store(tmp_path, selected):
    root, models, _ = selected
    pairs, _ = build_pairs(root, models)
    result = ReviewStore(tmp_path / "votes")
    result.sync(pairs)
    return result


def test_pairs_use_actual_selected_reflections_and_match_task_mode_gt_seed(selected):
    root, models, _ = selected
    pairs, hashes = build_pairs(root, models)
    assert len(pairs) == 8 and len({p["id"] for p in pairs}) == 8
    assert hashes
    for pair in pairs:
        assert set(pair["candidates"]) == {"best_advantage", "worst_advantage"}
        assert pair["task_id"] == "train_task"
        assert all(c["task"] == pair["task"] for c in pair["candidates"].values())


def test_loader_rejects_changed_original_reflection(selected):
    root, models, plan = selected
    job = json.loads(plan.read_text())[0]
    selection = json.loads((Path(job["path"]) / "conditions" /
                            job["condition"]["id"] / "selections.json").read_text())[0]
    write(Path(selection["candidate"]["source_reflection_path"]),
          {"reflection_for_curator": "changed"})
    with pytest.raises(ValueError, match="differs"):
        build_pairs(root, models)


def test_blinding_survives_restart_and_hides_all_selection_metadata(store):
    before = [store.pair(p["id"]) for p in store.queue({})]
    after = ReviewStore(store.directory)
    assert before == [after.pair(p["id"]) for p in after.queue({})]
    pair = before[0]
    assert set(pair) == {"id", "model", "mode", "arm", "seed", "task_id", "task", "left", "right", "vote"}
    assert "best_advantage" not in json.dumps(pair) and "worst_advantage" not in json.dumps(pair)
    assert "source_path" not in json.dumps(pair) and "mean_pct" not in json.dumps(pair)
    assert "<script>" in pair["left"]  # The browser must render this with textContent, not HTML.


def test_vote_is_durable_idempotent_and_can_be_corrected(store):
    pair_id = store.queue({})[0]["id"]
    store.vote(pair_id, "left", "specific steps")
    store.vote(pair_id, "left", "specific steps")
    resumed = ReviewStore(store.directory)
    assert resumed.pair(pair_id)["vote"]["choice"] == "left"
    assert resumed.catalog()["reviewed"] == 1
    store.vote(pair_id, "right", "revised judgment")
    assert store.pair(pair_id)["vote"]["choice"] == "right"
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM votes").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
        chosen = conn.execute("SELECT selected_strategy FROM votes").fetchone()[0]
        left = conn.execute("SELECT left_strategy FROM pairs WHERE id=?", (pair_id,)).fetchone()[0]
        assert chosen != left


def test_filtered_progress_and_independent_seed_votes(store):
    filters = {"model": "qwen38", "mode": "nonthinking", "arm": "with_gt", "seed": "43"}
    queue = store.queue(filters)
    assert len(queue) == 1
    store.vote(queue[0]["id"], "tie")
    assert store.queue(filters)[0]["choice"] == "tie"
    assert store.queue({**filters, "seed": "42"})[0]["choice"] is None
    assert store.catalog()["reviewed"] == 1
    with store.connect() as conn:
        assert conn.execute("SELECT selected_strategy FROM votes").fetchone()[0] is None


def test_simultaneous_duplicate_submission_is_counted_once(store):
    pair_id = store.queue({})[0]["id"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: store.vote(pair_id, "left"), range(8)))
    assert store.catalog()["reviewed"] == 1
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_archive_preserves_votes_if_a_source_pair_changes(store, selected):
    pair_id = store.queue({})[0]["id"]
    store.vote(pair_id, "right")
    root, models, _ = selected
    pairs, _ = build_pairs(root, models)
    store.sync([p for p in pairs if p["id"] != pair_id])
    assert store.catalog()["reviewed"] == 0
    assert pair_id in store.export_csv().decode("utf-8-sig")
    with pytest.raises(KeyError):
        store.pair(pair_id)
    store.sync(pairs)
    assert store.pair(pair_id)["vote"]["choice"] == "right"


@pytest.fixture
def http_server(store):
    server = ReviewServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def request(url, data=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return urllib.request.urlopen(req)


def test_http_vote_requires_same_origin_token_and_never_returns_strategy(http_server, store):
    catalog = json.load(request(http_server + "/api/catalog"))
    pair_id = store.queue({})[0]["id"]
    data = {"pair_id": pair_id, "choice": "left", "note": "more detail"}
    for headers in ({}, {"X-Review-Token": catalog["csrf_token"], "Origin": "http://other-site.test"}):
        with pytest.raises(urllib.error.HTTPError) as caught:
            request(http_server + "/api/vote", data, headers)
        assert caught.value.code == 403
    result = json.load(request(http_server + "/api/vote", data,
                               {"X-Review-Token": catalog["csrf_token"], "Origin": http_server}))
    assert result["saved"] and result["reviewed"] == 1
    pair = json.load(request(http_server + "/api/pairs/" + pair_id))
    assert pair["vote"]["choice"] == "left" and "selected_strategy" not in json.dumps(pair)
    assert "best_advantage" not in json.dumps(pair)


@pytest.mark.parametrize("choice,note", [("best_advantage", ""), (None, ""), ("left", 1), ("left", "x" * 2001)])
def test_bad_vote_is_not_saved(store, choice, note):
    with pytest.raises(ValueError):
        store.vote(store.queue({})[0]["id"], choice, note)
    assert store.catalog()["reviewed"] == 0


def test_csv_is_blind_and_neutralizes_spreadsheet_formulas(store):
    store.vote(store.queue({})[0]["id"], "tie", "=1+1")
    content = store.export_csv().decode("utf-8-sig")
    assert "'=1+1" in content
    assert "best_advantage" not in content and "selected_strategy" not in content


def test_http_does_not_serve_source_files_or_database(http_server):
    for path in ("/../reviews.sqlite3", "/source_manifest.json", "/api/pairs/unknown"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            request(http_server + path)
        assert caught.value.code == 404
    assert request(http_server + "/").status == 200


def test_independent_rules_use_actual_saved_update_not_reflection_or_proposed_operations(selected, tmp_path):
    root, models, _ = selected
    pairs, _ = build_pairs(root, models, content_kind="independent_rules")
    reflections, _ = build_pairs(root, models)
    assert len(pairs) == len(reflections) == 8
    assert {p["id"] for p in pairs}.isdisjoint({p["id"] for p in reflections})
    for pair in pairs:
        assert pair["task_id"] == "train_task" and pair["content_kind"] == "independent_rules"
        for c in pair["candidates"].values():
            assert "Independent applied rule" in c["text"]
            assert "if ready:\n    run()" in c["text"]
            assert "plain Reflection" not in c["text"] and "NOT ACTUALLY ADDED" not in c["text"]
            assert "helpful=" not in c["text"] and "## OTHERS" not in c["text"]
    store = ReviewStore(tmp_path / "rule_votes", "independent_rules")
    store.sync(pairs)
    pair = store.pair(pairs[0]["id"])
    assert pair["content_kind"] == "independent_rules"
    assert pair["left_rules"] and pair["right_rules"]
    assert "source_path" not in json.dumps(pair) and "best_advantage" not in json.dumps(pair)
    assert "rule set" in store.catalog()["criterion"]


@pytest.mark.parametrize("field,value", [("task_id", "wrong_task"), ("source_repeat", 4), ("arm", "other_arm")])
def test_independent_rule_pair_rejects_different_task_or_candidate_or_gt(selected, field, value):
    root, models, plan = selected
    job = json.loads(plan.read_text())[0]
    choice = json.loads((Path(job["path"]) / "conditions" / job["condition"]["id"] /
                         "selections.json").read_text())[0]
    path = Path(choice["candidate"]["source_reflection_path"]).parent / "result.json"
    result = json.loads(path.read_text())
    result[field] = value
    write(path, result)
    with pytest.raises(ValueError, match="does not match"):
        build_pairs(root, models, content_kind="independent_rules")


def test_rule_extractor_preserves_multiline_code_and_only_removes_playbook_metadata():
    initial = "## RULES\n\n## OTHERS"
    first = "Apply conditions carefully.\n```python\n## OTHERS\nif ok:\n    run()\n```"
    second = "A second applied rule."
    updated = (f"## RULES\n\n[x-00001] helpful=0 harmful=0 :: {first}\n\n"
               f"## OTHERS\n[x-00002] helpful=0 harmful=0 :: {second}\n")
    assert [r["content"] for r in added_rules(initial, updated)] == [first, second]
    assert added_rules(initial, initial) == []
    with pytest.raises(ValueError, match="empty learned"):
        added_rules(updated, updated)


def test_rule_and_reflection_votes_cannot_be_mixed(selected, store):
    root, models, _ = selected
    pairs, _ = build_pairs(root, models, content_kind="independent_rules")
    store.vote(store.queue({})[0]["id"], "left")
    with pytest.raises(ValueError, match="Cannot mix"):
        store.sync(pairs)
    with pytest.raises(ValueError, match="separate database"):
        ReviewStore(store.directory, "independent_rules")
    assert store.catalog()["reviewed"] == 1
