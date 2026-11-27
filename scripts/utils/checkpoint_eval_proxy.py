# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Let existing final-test consumers wait for the shared checkpoint inference queue."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


def wait_for_result(marker, task_id, policy_hash, context_profile=None):
    output = Path(marker["output"])
    while True:
        heartbeat = output / "heartbeat.json"
        if heartbeat.exists() and time.time() - heartbeat.stat().st_mtime > 180:
            raise RuntimeError("Shared checkpoint queue heartbeat is stale; no duplicate inference was launched")
        database = output / "queue.sqlite3"
        if database.exists():
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30) as db:
                db.row_factory = sqlite3.Row
                checkpoint = db.execute("SELECT * FROM checkpoints WHERE stream=? AND point=?",
                                        (marker["stream"], marker["point"])).fetchone()
                if checkpoint:
                    if checkpoint["sha"] != policy_hash:
                        raise ValueError("Final-test policy differs from the representative checkpoint")
                    if context_profile is not None:
                        payload = json.loads(checkpoint["payload"])
                        stream = payload["stream"]
                        if any(stream.get(key) != value for key, value in context_profile.items()):
                            raise ValueError("Final-test actor context differs from the representative checkpoint")
                    row = db.execute("SELECT * FROM tasks WHERE cp=? AND split='test' AND task_id=?",
                                     (checkpoint["canonical"], str(task_id))).fetchone()
                    if row and row["status"] == "completed":
                        return json.loads(row["result"])
                    if row and row["status"] == "error":
                        raise RuntimeError("Shared checkpoint evaluation exhausted infrastructure retries: "
                                           + str(json.loads(row["result"]).get("error")))
        time.sleep(2)


def evaluate_one(job):
    marker = json.loads((Path(job["output"]) / "checkpoint_queue.json").read_text())
    payload = json.loads(Path(job["frozen"]).read_text())
    sha = hashlib.sha256(payload["playbook"].encode()).hexdigest()
    profile = {key: job[key] for key in ("actor_context", "actor_context_metadata")} \
        if job.get("actor_context", "legacy") != "legacy" else None
    result = wait_for_result(marker, job["task_id"], sha, profile) if profile else \
        wait_for_result(marker, job["task_id"], sha)
    return {**job, **result, "external_checkpoint_queue": marker["output"]}


def baseline_final(request):
    marker = json.loads((Path(request["output"]) / "checkpoint_queue.json").read_text())
    profile = {key: request[key] for key in ("actor_context", "actor_context_metadata")} \
        if request.get("actor_context", "legacy") != "legacy" else None
    result = wait_for_result(marker, request["item"]["task_id"], request["prompt_sha256"], profile) if profile else \
        wait_for_result(marker, request["item"]["task_id"], request["prompt_sha256"])
    return {"id": request["item"]["task_id"], "hard": int(result["score"]),
            "soft": result.get("partial_score", result["score"]),
            "prompt_sha256": hashlib.sha256(request["system_prompt"].encode()).hexdigest(),
            "trace": {"external_checkpoint_queue": marker["output"],
                      "session_dir": result.get("session_output"), "actions": []}}
