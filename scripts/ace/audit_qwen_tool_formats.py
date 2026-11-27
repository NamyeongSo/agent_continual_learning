# SPDX-License-Identifier: Apache-2.0
"""Read-only audit of saved Qwen AppWorld actions rejected by code-generation mode.

Inspect the recorded outcome's final attempt only, across initial/candidate/final
Train phases. This never calls a model, reruns an environment, or edits scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from scripts.ace.gemma_python_format_repair import extract_python

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/ace_appworld_results_summary_20261123"
MODELS = ("qwen38", "qwen36", "qwen35")
MODES = ("thinking", "nonthinking")
ARMS = ("without_gt", "with_gt")
STRATEGIES = ("best", "random", "worst", "best_advantage", "worst_advantage")
REJECTION = "Code-generation mode accepts exactly one execute_python code block per agent turn."


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_bytes())


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def catalog():
    path = SUMMARY / "model_strategy_bars_20261124/plot_model_strategy_bars.py"
    spec = importlib.util.spec_from_file_location("saved_qwen_format_audit_sources", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    entries = []
    for model in MODELS:
        for mode in MODES:
            accumulation = helper.DATA.MODEL_RUNS[model][mode]
            protocol = helper.DATA.read(accumulation / "protocol.json")
            tasks = set(protocol["task_orders"]["42"]["train"])
            if len(tasks) != 50:
                raise ValueError("Expected pinned original Train50")
            reflection = Path(protocol["source_reflection_run"])
            for stage, filename, expected in (("initial", "stage1_results.json", 250),
                                               ("candidate", "stage3_results.json", 1500)):
                rows = helper.DATA.read(reflection / filename)
                helper.DATA.validate(rows)
                if len(rows) != expected or {r["task_id"] for r in rows} != tasks:
                    raise ValueError("Initial/candidate Train coverage differs")
                entries.extend({"model": model, "mode": mode, "phase": stage, "strategy": "",
                                "seed": "", "arm": row.get("arm", "shared_initial"), "row": row} for row in rows)
            for arm in ARMS:
                for strategy in STRATEGIES:
                    cohort, advantage = helper.accumulation_source(model, mode, arm, strategy)
                    for seed in (42, 43, 44):
                        condition = f"s{seed}_{arm}_{strategy}"
                        rows = helper.rows_for_context(cohort, advantage, mode, seed, condition)
                        if {r["task_id"] for r in rows} != tasks:
                            raise ValueError("Final Train coverage differs")
                        entries.extend({"model": model, "mode": mode, "phase": "accumulated_train",
                                        "strategy": strategy, "seed": seed, "arm": arm, "row": row} for row in rows)
    if len(entries) != 19500:
        raise ValueError("Expected 1500 initial + 9000 candidate + 9000 accumulated Train rollouts")
    identities = {(e["model"], e["mode"], e["phase"], e["row"]["id"]) for e in entries}
    if len(identities) != len(entries):
        raise ValueError("Duplicate rollout identity")
    return entries, helper.DATA.SOURCES


def inspect(entry):
    row = entry["row"]
    path = Path(row["capture_path"])
    meta = {key: value for key, value in entry.items() if key != "row"}
    meta.update(task_id=row["task_id"], id=row["id"], pct=row["pct"], success=row["success"],
                condition_id=row.get("condition_id", ""), capture_path=str(path),
                turn_limit_reached=bool(row.get("turn_limit_reached")))
    try:
        payload = path.read_bytes()
        capture = json.loads(payload)
        actions, observations = capture["actions"], capture["observations"]
    except (OSError, ValueError, KeyError) as exc:
        return {**meta, "coverage_error": repr(exc)}
    if digest_bytes(capture["playbook"].encode()) != row["initial_context_sha256"]:
        return {**meta, "coverage_error": "Captured playbook SHA differs from official result context SHA"}
    by_step = defaultdict(list)
    for action in actions:
        by_step[action["step"]].append(action)
    invalid = [step for step, group in by_step.items()
               if len(group) != 1 or group[0]["action"] != "execute_python"]
    kinds = Counter(a["action"] for step in invalid for a in by_step[step])
    messages = [a for a in actions if a["action"] == "message"]
    fenced = [a for a in messages if "```" in a.get("content", "")]
    repairable = [a for a in messages if extract_python(a.get("content", ""))[0] is not None]
    rejections = [o for o in observations if REJECTION in str(o.get("content", ""))]
    return {**meta, "capture_sha256": digest_bytes(payload), "actions": len(actions),
            "format_error": bool(invalid or rejections), "invalid_action_turns": len(invalid),
            "invalid_action_steps": invalid, "invalid_action_types": dict(kinds),
            "parallel_action_turns": sum(len(g) > 1 for g in by_step.values()),
            "observed_rejection_turns": len(rejections), "ordinary_reply_turns": len(messages),
            "ordinary_reply_fenced_turns": len(fenced), "ordinary_reply_repairable_python_turns": len(repairable),
            "ordinary_reply_steps": [a["step"] for a in messages],
            "fenced_reply_steps": [a["step"] for a in fenced],
            "undeclared_tool_names": sorted(k for k in kinds if k not in {"message", "execute_python"})}


def aggregate(rows):
    valid = [r for r in rows if "coverage_error" not in r]
    bad = [r for r in valid if r["format_error"]]
    return {"expected_rollouts": len(rows), "inspected_rollouts": len(valid),
            "missing_or_invalid_captures": len(rows) - len(valid), "format_error_rollouts": len(bad),
            "format_error_percent": 100 * len(bad) / len(valid) if valid else None,
            "ordinary_reply_rollouts": sum(r["ordinary_reply_turns"] > 0 for r in valid),
            "ordinary_reply_fenced_rollouts": sum(r["ordinary_reply_fenced_turns"] > 0 for r in valid),
            "ordinary_reply_repairable_python_rollouts": sum(r["ordinary_reply_repairable_python_turns"] > 0
                                                              for r in valid),
            "undeclared_tool_rollouts": sum(bool(r["undeclared_tool_names"]) for r in valid),
            "parallel_action_rollouts": sum(r["parallel_action_turns"] > 0 for r in valid),
            "format_error_turns": sum(r["invalid_action_turns"] for r in bad),
            "observed_rejection_turns": sum(r["observed_rejection_turns"] for r in valid),
            "format_error_rollouts_succeeded": sum(r["success"] for r in bad),
            "format_error_rollouts_failed": sum(not r["success"] for r in bad),
            "format_error_rollouts_turn_limited": sum(r["turn_limit_reached"] for r in bad)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUMMARY / "qwen_tool_format_audit_20261127")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries, sources = catalog()
    results, last = [], 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(inspect, e) for e in entries]
        for future in as_completed(futures):
            results.append(future.result())
            if time.monotonic() - last >= 15 or len(results) == len(entries):
                status = {"updated_at": now(), "state": "running", "completed": len(results), "total": len(entries),
                          "format_error_cases_so_far": sum(r.get("format_error", False) for r in results),
                          "coverage_errors_so_far": sum("coverage_error" in r for r in results)}
                write(output / "status.json", status)
                print(json.dumps(status), flush=True)
                last = time.monotonic()
    results.sort(key=lambda r: (r["model"], r["mode"], r["phase"], r["arm"], str(r["seed"]), r["id"]))
    by_phase = []
    by_setting = []
    by_condition = []
    for fields, destination in [(("model", "mode", "phase"), by_phase),
                                (("model", "mode", "phase", "arm"), by_setting),
                                (("model", "mode", "phase", "arm", "seed", "strategy"), by_condition)]:
        groups = defaultdict(list)
        for row in results:
            groups[tuple(row[field] for field in fields)].append(row)
        for key, rows in sorted(groups.items(), key=lambda pair: str(pair[0])):
            destination.append({**dict(zip(fields, key, strict=True)), **aggregate(rows)})
    summary = {"completed_at": now(), "scope": "Initial G5, independent candidate G5x3 per GT arm, "
               "and final-context Train50 across both modes, both GT arms, seeds42/43/44 and all five strategies",
               "overall": aggregate(results), "by_model_mode_phase": by_phase,
               "by_model": {m: aggregate([r for r in results if r["model"] == m]) for m in MODELS},
               "new_inference_calls": 0, "saved_outcomes_modified": False,
               "limitations": "Final recorded attempts only; does not audit discarded infrastructure attempts, "
               "BWT, Heldout, AUD variants or earlier invalid cohorts. Format-error association is not causation."}
    write(output / "summary.json", summary)
    write(output / "cases.json", [r for r in results if r.get("format_error")])
    write(output / "coverage_errors.json", [r for r in results if "coverage_error" in r])
    write(output / "source_hashes.json", {"result_collections": sources,
                                          "captures": {r["capture_path"]: r["capture_sha256"] for r in results
                                                       if "capture_sha256" in r},
                                          "script_sha256": digest_bytes(Path(__file__).read_bytes())})
    for name, rows in (("by_model_mode_phase", by_phase), ("by_setting", by_setting), ("by_condition", by_condition)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(output / "status.json", {"updated_at": now(), "state": "completed", "completed": len(results),
                                    "total": len(entries), **summary["overall"]})
    print(json.dumps({"output": str(output), "overall": summary["overall"], "by_model": summary["by_model"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
