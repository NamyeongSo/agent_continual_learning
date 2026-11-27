# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
import sqlite3

from scripts.utils.plot_pairwise_policy_shift_interim import analyze_partial


def test_all_checkpoint_filter_preserves_primary_cp0_and_excludes_incomplete_histories(tmp_path):
    path = tmp_path / "study.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE policies(stream,point,benchmark,method,seed);
            CREATE TABLE anchors(id,task); CREATE TABLE bindings(stream,point,anchor,job,reference);
            CREATE TABLE jobs(id,status,action,error);""")
        for point in (0, 10, 20):
            db.execute("INSERT INTO policies VALUES('s',?,'bfcl','ace_aud',42)", (point,))
        for anchor, partial in (("common", False), ("partial", True)):
            db.execute("INSERT INTO anchors VALUES(?,?)", (anchor, anchor))
            primary, repeat = anchor + "_primary", anchor + "_repeat"
            action = json.dumps([{"name": "f", "arguments": {}}])
            db.execute("INSERT INTO jobs VALUES(?,'completed',?,NULL)", (primary, action))
            db.execute("INSERT INTO jobs VALUES(?,'error',NULL,'repeat missing')", (repeat,))
            for point in (0, 10, 20):
                job = repeat if point == 0 else anchor + str(point)
                if point:
                    status = "pending" if partial and point == 20 else "completed"
                    db.execute("INSERT INTO jobs VALUES(?,?,?,NULL)", (job, status, action))
                db.execute("INSERT INTO bindings VALUES('s',?,?,?,?)", (point, anchor, job, primary))
    before = path.read_bytes()
    summary, pairs, tasks = analyze_partial(path, require_all_checkpoints=True)
    assert summary["coverage"][0]["all_checkpoint_common_histories"] == 1
    assert summary["coverage"][0]["compared_tasks"] == 1
    assert len(pairs) == 3 and {row["task_id"] for row in tasks} == {"common"}
    assert all(row["mean_shift"] == 0 for row in summary["scores"])
    assert path.read_bytes() == before
    _, _, available_tasks = analyze_partial(path)
    assert {row["task_id"] for row in available_tasks} == {"common", "partial"}
