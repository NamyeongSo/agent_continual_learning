# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Execute one immutable checkpoint/task pair; never reflect or update a policy."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_OUTPUT = None
_BASE_URL = None


def initialize(output, base_url):
    global _OUTPUT, _BASE_URL
    import psutil

    from scripts.utils.process_identity import stable_identity
    from scripts.utils.run_qwen_isolated_when_free import experiment_environment, write

    _OUTPUT = Path(output)
    _BASE_URL = base_url
    os.environ.update(experiment_environment())
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.pop("EXGENTIC_PORT_BLOCK_START", None)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    write(_OUTPUT / "workers" / f"{os.getpid()}.json", {
        "pid": os.getpid(), "created": psutil.Process().create_time(), "phase": "idle", **stable_identity(os.getpid()),
    })


def evaluate(task):
    from scripts.utils.checkpoint_sources import digest
    from scripts.utils.run_qwen_isolated_when_free import evaluate_one, read, write

    checkpoint = Path(task["checkpoint_file"])
    started = time.time()
    worker = _OUTPUT / "workers" / f"{os.getpid()}.json"
    identity = read(worker)
    write(worker, {**identity, "phase": "running", "task": task["id"], "at": started})
    folder = _OUTPUT / "sessions" / f"task_{task['id']:07d}" / f"attempt_{task['attempt']}"
    try:
        payload = read(checkpoint)
        if digest(payload["content"]) != task["content_sha256"]:
            raise ValueError("Frozen checkpoint policy changed")
        spec = payload["stream"]
        actor_context = spec.get("actor_context", "legacy")
        if actor_context != "legacy":
            from exgentic.actor_context import require_matching_actor_context

            benchmarks = list(spec["actor_context_metadata"]["initial_actor_prompt_sha256"])
            if spec["benchmark"] not in benchmarks:
                raise ValueError("Frozen actor context does not cover this benchmark")
            require_matching_actor_context(spec, actor_context, benchmarks)
        if spec.get("appworld_policy_override") is not None:
            if digest(spec["appworld_policy_override"]) != spec.get("appworld_policy_sha256"):
                raise ValueError("Frozen checkpoint AppWorld policy changed")
            if spec["method"] in {"gepa", "skillopt"} and (
                spec["benchmark_kwargs"].get("policy_override") != spec["appworld_policy_override"]
            ):
                raise ValueError("Optimizer and checkpoint AppWorld policies differ")
        if spec["method"] in {"ace", "premise"}:
            job = {"method": spec["method"], "benchmark": spec["benchmark"], "seed": spec["seed"],
                   "task_id": task["task_id"], "index": task["id"], "output": str(folder),
                   "frozen": str(checkpoint), "frozen_sha256": digest(checkpoint.read_bytes()),
                   "base_url": _BASE_URL, "benchmark_kwargs": spec.get("benchmark_kwargs", {}),
                   "playbook_edit_mode": spec.get("playbook_edit_mode", "add_only")}
            if actor_context != "legacy":
                job.update(actor_context=actor_context, actor_context_metadata=spec["actor_context_metadata"])
            if spec.get("appworld_policy_override") is not None:
                job.update(appworld_policy_override=spec["appworld_policy_override"],
                           appworld_policy_sha256=spec["appworld_policy_sha256"])
            result = evaluate_one(job)
        else:
            from scripts.baselines.actor import run

            extension = "txt" if spec["method"] == "gepa" else "md"
            base = Path(spec["snapshot"]) / f"scripts/baselines/prompts/{spec['method']}/base"
            if actor_context != "legacy":
                from exgentic.actor_context import shared_actor_prompt

                system = shared_actor_prompt(spec["benchmark"], payload["content"])
            else:
                system = (base / f"{spec['benchmark']}.{extension}").read_text() + "\n\n" + payload["content"]
            raw = run({"method": spec["method"], "benchmark": spec["benchmark"],
                       "model": "openai/" + spec["model"].removeprefix("openai/"),
                       "benchmark_kwargs": spec["benchmark_kwargs"], "system_prompt": system,
                       **({"actor_context": actor_context, "learned_policy": payload["content"]}
                          if actor_context != "legacy" else {}),
                       "item": {"task_id": task["task_id"], "split": task["split"]},
                       "session_output": str(folder)})
            result = {"status": "completed", "score": raw["hard"], "partial_score": raw["soft"],
                      "run_id": raw["run_id"],
                      **({"score_metric": "appworld_tgc_v1", "tgc": raw["hard"]}
                         if spec["benchmark"] == "appworld" else {})}
        compact = {key: value for key, value in result.items() if key in {
            "status", "score", "partial_score", "error", "session_id", "session_status",
            "budget_failure", "started_at", "ended_at", "run_id",
            "score_metric", "tgc",
            "appworld_policy_sha256",
        }}
        compact.update(session_output=str(folder), worker_pid=os.getpid(), elapsed_seconds=time.time() - started)
        files = list(folder.glob("*/sessions/*/results.json"))
        if files:
            session = json.loads(files[0].read_text())
            compact["cost_reports"] = session.get("cost_reports", {})
            compact["steps"] = session.get("steps")
            compact["raw_results"] = str(files[0])
        return compact
    except Exception as exc:
        return {"status": "error", "score": None, "error": f"{type(exc).__name__}: {exc}",
                "session_output": str(folder), "worker_pid": os.getpid(), "elapsed_seconds": time.time() - started}
    finally:
        write(worker, {**identity, "phase": "idle", "at": time.time()})
