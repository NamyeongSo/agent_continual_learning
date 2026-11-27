# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Measure missing public argument schemas without changing trajectories, labels or rubrics."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise import reward_data, rubric_prompts
from exgentic.agents.premise.reward_data import BENCHMARKS, digest, write_json
from exgentic.agents.premise.reward_pipeline import RewardLLM
from scripts.premise.analyze_judge_specificity import SCOPED, Analysis, decision_metrics, paired_difference


def restore_public_schemas(visible, session):
    """Restore only schemas of already selected public APIs, excluding hidden tool variants."""
    tools = {tool["name"]: tool for tool in session["actions"] if not tool.get("is_hidden")}
    result, changed = copy.deepcopy(visible), 0
    for tool in result["public_tools"]:
        source = tools[tool["name"]]
        schema = source.get("arguments_schema")
        if not isinstance(schema, dict):
            raise ValueError("Source public tool has no argument schema")
        if "arguments_schema" in tool:
            if tool["arguments_schema"] != schema:
                raise ValueError("Existing/source public schemas disagree")
        else:
            tool["arguments_schema"] = copy.deepcopy(schema)
            changed += 1
    return result, changed


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    specific = args.specificity_source.resolve()
    source_config = json.loads((specific / "config.json").read_text())
    dataset = Path(source_config["source_roots"]["rubric"]) / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    identity = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_implementation_sha256": hashlib.sha256(Path(reward_data.__file__).read_bytes()).hexdigest(),
        "specificity_config_sha256": digest(source_config),
        "baseline_source": str(specific),
        "workers": args.workers,
        "feedback_policy": "unchanged frozen 500-token observations",
        "intervention": "Restore arguments_schema for observed public APIs only",
        "arms": ["no_rubric", "expanded_scoped"],
        "split": "validation; development diagnostic only",
    }
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Different contract diagnostic configuration")
    write_json(path, identity)
    shutil.copy2(Path(__file__), root / "source_snapshot.py")

    def status(**kwargs):
        write_json(root / "status.json", {"pid": os.getpid(), "updated_at": datetime.now(UTC).isoformat(), **kwargs})

    status(status="running", stage="wait_for_specificity_queue", processed=0)
    while True:
        state = json.loads((specific / "status.json").read_text())
        if state["status"].startswith("completed"):
            break
        if state["status"] == "error":
            raise RuntimeError("Specificity queue failed before contract comparison")
        time.sleep(10)
    selections = json.loads((specific / "rubric_selection.json").read_text())
    baselines = [
        r
        for r in json.loads((specific / "results.json").read_text())
        if r["experiment"] == "rubric" and r["arm"] in identity["arms"]
    ]
    record_map = {r["id"]: r for r in manifest["records"]}
    visible, affected, contracts = {}, set(), []
    for record in manifest["records"]:
        if record["split"] != "val":
            continue
        value = json.loads((dataset / "pairs" / f"{record['id']}.json").read_text())
        if digest(value) != record["input_sha256"]:
            raise ValueError("Frozen input hash mismatch")
        session = json.loads((Path(record["provenance"]["original_session_dir"]) / "session.json").read_text())
        restored, count = restore_public_schemas(value, session)
        visible[record["id"]] = restored
        if count:
            affected.add(record["id"])
        contracts.append(
            {
                "record_id": record["id"],
                "benchmark": record["benchmark"],
                "restored_schemas": count,
                "old_input_sha256": digest(value),
                "new_input_sha256": digest(restored),
            }
        )
        write_json(root / "pairs" / f"{record['id']}.json", restored)
    write_json(root / "schema_changes.json", contracts)
    identity["restored_input_sha256"] = digest(contracts)
    caller = SimpleNamespace(
        root=root,
        identity=identity,
        lock=threading.Lock(),
        request_locks={},
        args=SimpleNamespace(retry_errors=args.retry_errors),
        llm=RewardLLM(
            source_config["base_url"],
            source_config["model"],
            root / "llm_calls",
            threading.BoundedSemaphore(args.workers),
        ),
    )

    def judge(base):
        result = {**base, "input_variant": "restored_schema"}
        pair = copy.deepcopy(visible[base["record_id"]])
        if base["swapped"]:
            pair["trajectory_a"], pair["trajectory_b"] = pair["trajectory_b"], pair["trajectory_a"]
        rules = [] if base["arm"] == "no_rubric" else selections["expanded"]
        public_rules = [{k: r[k] for k in ("id", "name", "description")} for r in rules]
        system = rubric_prompts.JUDGE + (SCOPED if base["arm"] == "expanded_scoped" else "")
        messages = rubric_prompts.messages(system, {"comparison": pair, "criteria": public_rules})
        try:
            raw = Analysis.call(caller, messages, "public_contract_judge", Analysis.parse_choice)
            canonical = ("B" if raw == "A" else "A") if base["swapped"] else raw
            result.update(
                status="completed",
                choice=canonical,
                raw_choice=raw,
                reward=int(canonical == record_map[base["record_id"]]["label"]),
            )
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__, error_message=str(exc))
        return result

    queue = sorted((r for r in baselines if r["record_id"] in affected), key=lambda r: digest([42, r]))
    results = [{**r, "input_variant": "unchanged_no_tool_schema"} for r in baselines if r["record_id"] not in affected]
    status(status="running", stage="schema_judgments", processed=0, total=len(queue))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(judge, row) for row in queue]
        for processed, future in enumerate(as_completed(jobs), 1):
            results.append(future.result())
            status(status="running", stage="schema_judgments", processed=processed, total=len(queue))
    write_json(root / "results.json", results)
    contrasts, metrics = {}, {}
    for arm in identity["arms"]:
        contrasts[arm], metrics[arm] = {}, {}
        for bench in BENCHMARKS:
            old = [r for r in baselines if r["arm"] == arm and r["benchmark"] == bench]
            new = [r for r in results if r["arm"] == arm and r["benchmark"] == bench]
            contrasts[arm][bench] = paired_difference(old, new)
            metrics[arm][bench] = {"old": decision_metrics(old), "restored": decision_metrics(new)}
    summary = {
        "metrics": metrics,
        "paired_contrasts": contrasts,
        "requested_new_decisions": len(queue),
        "restored_pairs": len(affected),
        "errors": sum(r["status"] != "completed" for r in results),
        "limitations": "Development validation reused. Schema restoration changes input length as well as information; "
        "only 5/5/4 independent task groups. No retraining performed.",
    }
    write_json(root / "summary.json", summary)
    status(
        status="completed_with_errors" if summary["errors"] else "completed",
        stage="finished",
        processed=len(queue),
        total=len(queue),
        errors=summary["errors"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specificity-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except BaseException as exc:
            write_json(
                args.output / "status.json",
                {"pid": os.getpid(), "status": "error", "error_type": type(exc).__name__, "error_message": str(exc)},
            )
            raise


if __name__ == "__main__":
    main()
