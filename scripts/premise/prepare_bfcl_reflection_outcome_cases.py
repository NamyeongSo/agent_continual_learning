# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Extract outcome extremes and their frozen reflection/update/behavior evidence offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_action(action: dict):
    if "actions" in action:
        return [compact_action(a) for a in action["actions"]]
    return {"name": action.get("name"), "arguments": action.get("arguments")}


def behavior_evidence(path: Path) -> dict:
    result = read(path)
    root = path.parents[4]
    run = f"alignment_{result['checkpoint_key']}_{result['context']}_{result['partition']}_{result['index']:03d}"
    suffix = (Path("sessions") / result["checkpoint_key"] / result["context"] / result["partition"]
              / run / "sessions" / result["session_id"] / "trajectory.jsonl")
    possibilities = [root / suffix] + [p / suffix for p in (root / "contributors").glob("*") if p.is_dir()]
    found = [p for p in possibilities if p.is_file()]
    if len({digest(p) for p in found}) > 1:
        raise ValueError("Multiple different traces for the recorded evaluation session")
    output = {"result_path": str(path.resolve()), "result_sha256": digest(path),
              "task_id": result["task_id"], "score": result["score"], "steps": result["steps"],
              "session_status": result.get("session_status"), "trajectory_available": bool(found)}
    if found:
        trace = found[0]
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        score_path = trace.parent / "benchmark" / "bfcl_score.json"
        score = read(score_path) if score_path.exists() else {}
        checker = score.get("session_metadata", {}).get("checker_result", {})
        output.update({"trajectory_path": str(trace.resolve()), "trajectory_sha256": digest(trace),
                       "actions": [{"step": e["step"], "action": compact_action(e["action"])}
                                   for e in events if e["event"] == "action"],
                       "score_path": str(score_path.resolve()) if score_path.exists() else None,
                       "score_sha256": digest(score_path) if score_path.exists() else None,
                       "checker_valid": checker.get("valid"), "error_type": checker.get("error_type"),
                       "error_message": checker.get("error_message"), "checker_details": checker.get("details")})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = read(args.input)
    sources = [Path(r["path"]) for r in summary["source_files"]]
    if any(digest(p) != r["sha256"] for p, r in zip(sources, summary["source_files"], strict=True)):
        raise ValueError("Frozen selection/outcome source changed")
    old, cache = (read(p) for p in sources)
    candidates = read(args.candidates)
    points = {(m["label"], p["key"]): p for m in cache["models"] for p in m["points"]}
    rows = [r for r in summary["checkpoint_rows"]
            if r["case"] == "current_past" and r["weight"] == "task_mean" and r["n"] == 10]
    chosen = defaultdict(list)
    for model in summary["models"]:
        subset = [r for r in rows if r["model"] == model]
        for metric in ("gain_past", "lift_validation40"):
            for direction in ("high", "low"):
                row = sorted(subset, key=lambda r: (r[metric], r["key"]), reverse=direction == "high")[0]
                chosen[model, row["key"]].append(f"{direction}_{metric}")
    chosen["Qwen", "s44_cp035"].append("illustrative_negative_past_positive_validation")
    chosen["Gemma", "s44_cp015"].append("illustrative_positive_past_negative_validation")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = args.output / "cases"
    cases.mkdir(exist_ok=True)
    catalog, behavior = [], []
    for (model, key), reasons in chosen.items():
        selected = next(r for r in rows if r["model"] == model and r["key"] == key)
        point = points[model, key]
        prob = next(r["choice_probabilities"] for r in old["checkpoint_rows"]
                    if r["model"] == model and r["key"] == key and r["case"] == "current_past"
                    and r["weight"] == "task_mean" and r["n"] == 10 and r["method"] == "success_mean")
        group = [r for r in candidates if r["model"] == model and r["key"] == key]
        trace_path = Path(group[0]["source"]).parent / "proposal_trace.json"
        trace = read(trace_path)
        case = {"model": model, "key": key, "selection_criteria": reasons, "selected_expectation": selected,
                "choice_probabilities": prob, "trace_path": str(trace_path.resolve()),
                "trace_sha256": digest(trace_path), "task_id": trace["task_id"], "task": trace["task"],
                "policy": trace.get("context"), "score": trace["score"], "feedback": trace["details"],
                "actions": trace["actions"], "public_tool_contract": trace.get("public_tool_contract"),
                "candidates": []}
        lines = [f"=== {model} {key} ({', '.join(reasons)}) ===",
                 f"Selected expectation: past={100 * selected['gain_past']:+.3f}pp, "
                 f"val={100 * selected['lift_validation40']:+.3f}pp", f"Task: {trace['task_id']} {trace['task']}"]
        for row in group:
            path, i = Path(row["source"]), row["candidate"]
            candidate = read(path)
            if digest(path) != point["provenance"]["context_sha256"][i]:
                raise ValueError("Candidate differs from the evaluated context")
            record = {**row, "selection_probability": prob[i - 1], "reflection": candidate["reflection"],
                      "applied": candidate["applied"], "mutation_summary": candidate["mutation_summary"],
                      "source_sha256": digest(path)}
            case["candidates"].append(record)
            catalog.append({**row, "selection_probability": prob[i - 1], "review_reason": ";".join(reasons)})
            result_path = Path(point["provenance"]["result_paths"][i][point["current_index"]])
            if digest(result_path) != point["provenance"]["result_sha256"][i][point["current_index"]]:
                raise ValueError("Archived current-task result changed")
            behavior.append({"model": model, "key": key, "candidate": i, **behavior_evidence(result_path)})
            lines += [f"\nCANDIDATE {i}: p(select)={prob[i - 1]:.4f} "
                      f"past={100 * row['gain_past']:+.3f}pp val={100 * row['validation40_lift']:+.3f}pp",
                      "REFLECTION: " + json.dumps(candidate["reflection"], ensure_ascii=False),
                      "ACTUAL UPDATE: " + json.dumps(candidate["applied"], ensure_ascii=False)]
        for metric, original in (("gain_past", "gain_past"), ("validation40_lift", "lift_validation40")):
            if not np.isclose(np.array(prob) @ [r[metric] for r in group], selected[original], rtol=0, atol=1e-12):
                raise ValueError("Five-candidate outcomes no longer reproduce the selected CP point")
        (cases / f"{model.lower()}_{key}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
        (cases / f"{model.lower()}_{key}_review.txt").write_text("\n".join(lines) + "\n")
    (args.output / "case_catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    (args.output / "behavior_trace_summary.json").write_text(json.dumps(behavior, indent=2) + "\n")
    print(f"Prepared {len(chosen)} CP, {len(catalog)} candidates, "
          f"{sum(r['trajectory_available'] for r in behavior)} current-task behavior traces; no model calls.")


if __name__ == "__main__":
    main()
