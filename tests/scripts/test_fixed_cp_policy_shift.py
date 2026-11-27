# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
import sqlite3

from scripts.utils import run_checkpoint_policy_shift as common
from scripts.utils import run_fixed_cp_policy_shift as study
from scripts.utils import run_pairwise_policy_shift as pairs
from scripts.utils.policy_shift_metrics import digest


def test_drained_gate_preserves_reviewed_timeouts_and_rejects_active_or_changed_results(tmp_path):
    db = sqlite3.connect(tmp_path / "queue.sqlite3")
    db.executescript("CREATE TABLE streams(id TEXT); CREATE TABLE tasks(id INTEGER,status TEXT,result TEXT);"
                     "CREATE TABLE stream_invalidations(stream TEXT);")
    db.execute("INSERT INTO streams VALUES('source')")
    error = json.dumps({"error": "TimeoutError"})
    db.executemany("INSERT INTO tasks VALUES(?,?,?)", [(1, "completed", '{}'), (2, "error", error)])
    db.commit()
    plan = {"queue_output": str(tmp_path), "selected_stream_ids": ["source"],
            "terminal_timeout_exclusions": {"2": digest(error)}}
    assert study.gate(plan)["ready"]
    db.execute("UPDATE tasks SET status='running' WHERE id=1")
    db.commit()
    assert not study.gate(plan)["ready"]
    db.execute("UPDATE tasks SET status='completed' WHERE id=1")
    db.execute("UPDATE tasks SET result='changed' WHERE id=2")
    db.commit()
    assert not study.gate(plan)["ready"]


def test_history_banks_are_separate_and_identical_policies_reuse_requests_with_repeat_control(tmp_path):
    db = common.own_db(tmp_path)
    db.execute("CREATE TABLE anchor_owners(anchor TEXT,method TEXT,PRIMARY KEY(anchor,method))")
    for anchor, owner, seed in [("ace", "ace_aud", 42), ("gepa", "gepa_val", 42),
                                ("gepa43", "gepa_val", 43)]:
        db.execute("INSERT INTO anchors VALUES(?,'bfcl',?,'task','unused')", (anchor, seed))
        db.execute("INSERT INTO anchor_owners VALUES(?,?)", (anchor, owner))
    db.execute("INSERT INTO anchor_owners VALUES('gepa','gepa_train_as_val')")
    for stream, method in [("v", "gepa_val"), ("t", "gepa_train_as_val")]:
        for point, prompt in [(0, "root"), (10, "root"), (20, "changed"), (30, "changed")]:
            db.execute("INSERT INTO policies VALUES(?,?,'bfcl',?,42,?)", (stream, point, method, prompt))
    db.commit()
    study.build_jobs(db, {"model": "Qwen", "temperature": 0, "input_tokens": 100000, "output_tokens": 2048})
    assert {r[0] for r in db.execute("SELECT DISTINCT anchor FROM bindings")} == {"gepa"}
    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 6
    initial = db.execute("SELECT job,reference FROM bindings WHERE stream='v' AND point=0").fetchone()
    alias = db.execute("SELECT job FROM bindings WHERE stream='v' AND point=10").fetchone()
    assert initial["job"] != initial["reference"]
    assert alias["job"] == initial["reference"]
    changed = db.execute("SELECT job FROM bindings WHERE stream='v' AND point IN (20,30)").fetchall()
    assert changed[0][0] == changed[1][0]


def test_session_tool_recovery_preserves_schema_order_and_message_exclusion():
    tools = study.session_tools({"actions": [
        {"name": "z", "description": "Z", "arguments_schema": {"maximum": 10**30}},
        {"name": "message", "is_message": True},
        {"name": "a", "description": "A", "arguments_schema": {"type": "object"}},
    ]})
    assert [tool["function"]["name"] for tool in tools] == ["a", "z"]
    assert tools[1]["function"]["parameters"]["maximum"] == 2147483647


def test_frozen_renderer_uses_the_training_helpers(tmp_path):
    folder = tmp_path / "src/exgentic/agents/ace"
    (folder / "prompts").mkdir(parents=True)
    (folder / "playbook_utils.py").write_text("def get_playbook_stats(text): return {'n': len(text)}\n")
    (folder / "prompts/generator.py").write_text(
        "from ..playbook_utils import get_playbook_stats\n"
        "def build_generator_system_prompt(text): return str(get_playbook_stats(text)['n']) + ':' + text\n")
    checks = {}
    render = study.frozen_renderer(tmp_path, checks)
    assert render("ABC") == "3:ABC"
    assert len(checks) == 2


def test_pairwise_analysis_keeps_baseline_variations_separate(tmp_path):
    db = common.own_db(tmp_path)
    db.execute("CREATE TABLE anchor_owners(anchor TEXT,method TEXT,PRIMARY KEY(anchor,method))")
    db.execute("INSERT INTO anchors VALUES('history','bfcl',42,'task','unused')")
    for stream, method, final in [("v", "gepa_val", "changed"), ("t", "gepa_train_as_val", "root")]:
        db.execute("INSERT INTO anchor_owners VALUES('history',?)", (method,))
        for point, prompt in [(0, "root"), (10, final)]:
            db.execute("INSERT INTO policies VALUES(?,?,'bfcl',?,42,?)", (stream, point, method, prompt))
    db.commit()
    plan = {"model": "Qwen", "temperature": 0, "input_tokens": 100000, "output_tokens": 2048,
            "workers": 128, "comparison": "all_within_seed_checkpoint_pairs_by_distance"}
    study.build_jobs(db, plan)
    for row in db.execute("SELECT * FROM jobs").fetchall():
        action = json.dumps([{"name": row["prompt"], "arguments": {}}])
        db.execute("UPDATE jobs SET status='completed',action=? WHERE id=?", (action, row["id"]))
    db.commit()
    result = pairs.analyze(tmp_path, plan)
    scores = {r["method"]: r["mean_shift"] for r in result["scores"] if r["metric"] == "strict_accuracy"}
    assert scores == {"gepa_val": 1, "gepa_train_as_val": 0}
