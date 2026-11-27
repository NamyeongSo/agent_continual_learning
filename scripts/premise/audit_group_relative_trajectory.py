# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Reconstruct complete action sequences on every original acting prefix, without replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows  # noqa: E402
from scripts.premise.audit_group_relative_f1 import (  # noqa: E402
    SpanPolicyShiftEvaluator,
    canonical,
    configure_llm_slots,
    normalize_action,
    replace_span,
    run_draw,
    sha,
    terminal_parser,
    write_json,
)
from scripts.premise.audit_group_relative_update_distance import context_name  # noqa: E402
from scripts.utils.policy_shift_metrics import response_message  # noqa: E402


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def actor_request(prepared, case, point):
    anchor = json.loads(Path(case["source"]).read_text())
    if sha(Path(case["source"])) != case["source_sha256"]:
        raise ValueError("Frozen prefix changed")
    policy = prepared["policies"][case["benchmark"]][str(point)]
    request = replace_span(anchor, policy["content"])
    request["generation_settings"] = copy.deepcopy(prepared["generation_settings"])
    return request


def prepare_temperature_ablation(paired_run, temperature):
    """Freeze the exact existing corpus and change both probe arms' temperature only."""
    path = paired_run / "prepared.json"
    source = json.loads(path.read_text())
    if source["protocol"] != "full_trajectory_regenerated_contexts_cp10_t1_v1":
        raise ValueError("Temperature ablation requires the frozen regenerated-context T1 cohort")
    if temperature not in (0, 1):
        raise ValueError("Only the explicit T0/T1 sampling ablation is supported")
    config = json.loads(Path(source["model_call_source"]["path"]).read_text())
    if config["model"] != source["model"] or config["model_settings"] != source["model_call_settings"]:
        raise ValueError("A live setting other than the explicit temperature override changed")
    prepared = copy.deepcopy(source)
    prepared["protocol"] = f"full_trajectory_regenerated_contexts_cp10_t{temperature:g}_v1"
    for settings in (prepared["generation_settings"], prepared["model_call_settings"]):
        settings["temperature"] = temperature
    prepared["temperature"] = temperature
    prepared["temperature_ablation"] = {
        "paired_run": str(paired_run), "prepared_sha256": sha(path),
        "source_temperature": source["temperature"], "probe_temperature": temperature,
        "changed_generation_parameters": ["temperature"],
        "rollout_sampling_changed": False,
        "interpretation": "Both evaluation arms regenerate with the same chosen temperature. T0 compares "
                          "greedy conditional actions, not the importance ratio of the T1 rollout policy.",
    }
    for case in prepared["cases"]:
        if sha(Path(case["source"])) != case["source_sha256"]:
            raise ValueError("Paired corpus history changed")
    return prepared


def prepare(baseline, output, repeats, comparison="regenerated_contexts"):
    original = json.loads((baseline / "prepared.json").read_text())
    source = Path(original["source"])
    config = json.loads(Path(original["model_call_source"]["path"]).read_text())
    if config["model"] != original["model"] or config["model_settings"] != original["model_call_settings"]:
        raise ValueError("Live model-call configuration changed")
    db = sqlite3.connect(f"file:{source / 'policy_shift.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    queue_path = Path(json.loads((source / "plan.json").read_text())["queue_output"]) / "queue.sqlite3"
    queue = sqlite3.connect(f"file:{queue_path}?mode=ro", uri=True)
    queue.row_factory = sqlite3.Row
    cases, trajectories, policies = [], [], {}
    settings = next(iter(original["cases"][0]["requests"].values()))["generation_settings"]
    for benchmark, points in original["checkpoints"].items():
        bank = db.execute(
            "SELECT p.stream,p.point,t.content FROM policies p JOIN prompts t ON t.sha=p.prompt "
            "WHERE p.benchmark=? AND p.method=? AND p.seed=? ORDER BY p.point",
            (benchmark, original["method"], original["seed"]),
        ).fetchall()
        if [r["point"] for r in bank] != points:
            raise ValueError("Checkpoint grid changed")
        policies[benchmark] = {str(r["point"]): {"content": r["content"],
                               "sha256": hashlib.sha256(r["content"].encode()).hexdigest()} for r in bank}
        cp = queue.execute("SELECT * FROM checkpoints WHERE stream=? AND point=0", (bank[0]["stream"],)).fetchone()
        selected = [c for c in original["cases"] if c["benchmark"] == benchmark]
        for task in sorted({c["task"] for c in selected}):
            prior = next(c for c in selected if c["task"] == task)
            template = json.loads(Path(prior["source"]).read_text())
            recorded = db.execute(
                "SELECT a.* FROM anchors a JOIN anchor_owners o ON o.anchor=a.id "
                "WHERE a.benchmark=? AND a.task=? AND a.seed=? AND o.method=?",
                (benchmark, task, original["seed"], original["method"]),
            ).fetchall()
            existing = {digest(json.loads(Path(r["file"]).read_text())["messages"]): dict(r) for r in recorded}
            task_row = queue.execute("SELECT * FROM tasks WHERE cp=? AND split='train' AND task_id=?",
                                     (cp["canonical"], task)).fetchone()
            if task_row["status"] != "completed":
                raise ValueError("Missing original trajectory")
            folder = Path(json.loads(task_row["result"])["session_output"])
            traces = list(folder.glob("**/run/litellm/trace.jsonl")) if benchmark == "terminalbench2" else list(
                folder.glob("**/agent/litellm/trace.jsonl")
            )
            if len(traces) != 1:
                raise ValueError("Ambiguous original trace")
            parser = terminal_parser(**prior["parser"]) if "parser" in prior else None
            trace, seen, steps, skipped = traces[0], set(), [], []
            for row in iter_expanded_trace_rows(trace):
                if row.get("status") != "success":
                    continue
                response = row["response"]
                identity = response.get("id") or digest(response)
                if identity in seen:
                    continue
                request = row["request"]
                anchor = {**copy.deepcopy(template), "messages": request["messages"],
                          "tools": request.get("tools", template["tools"])}
                try:
                    replace_span(anchor, policies[benchmark]["0"]["content"])
                except (KeyError, ValueError, IndexError, TypeError):
                    continue  # Reflection, selection or other calls do not have the exact actor policy span.
                seen.add(identity)
                index = sum(m["role"] == "assistant" for m in anchor["messages"])
                try:
                    message = response_message(response)
                    actions = normalize_action(message, benchmark, parser)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    skipped.append({"action_index": index, "response_id": identity, "error": str(exc)[:500]})
                    continue  # No valid action was executed from this failed/truncated response.
                maximum = 30 if benchmark == "bfcl" else 40
                if index >= maximum:
                    raise ValueError(f"Original trajectory exceeds the retained turn limit: {benchmark}/{task}")
                old = existing.get(digest(anchor["messages"]))
                if old is not None:
                    source_path, identifier = Path(old["file"]), old["id"]
                    anchor = json.loads(source_path.read_text())
                else:
                    identifier = digest(anchor)
                    source_path = output / "histories" / f"{identifier}.json"
                    write_json(source_path, anchor)
                case = {"id": identifier, "benchmark": benchmark, "task": task, "action_index": index,
                        "source": str(source_path), "source_sha256": sha(source_path),
                        **({"parser": prior["parser"]} if "parser" in prior else {})}
                cases.append(case)
                steps.append({"case": identifier, "action_index": index, "original_actions": actions,
                              "source_response_id": identity,
                              "source_model": request.get("model"),
                              "source_temperature": request.get("temperature")})
            steps.sort(key=lambda r: r["action_index"])
            indices = [r["action_index"] for r in steps]
            if indices != list(range(len(steps))):
                raise ValueError(f"Original action sequence has a gap or duplicate: {benchmark}/{task}: {indices}")
            if not steps:
                raise ValueError("Original trajectory has no valid acting steps")
            trajectories.append({"benchmark": benchmark, "task": task, "steps": steps,
                                 "original_actions": [a for r in steps for a in r["original_actions"]],
                                 "source_trace": str(trace), "source_trace_sha256": sha(trace),
                                 "unexecuted_invalid_responses": skipped})
    db.close()
    queue.close()
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate original acting prefixes")
    regenerated = comparison == "regenerated_contexts"
    return {"protocol": ("full_trajectory_regenerated_contexts_cp10_t1_v1" if regenerated else
                         "full_trajectory_original_vs_reconstructed_cp10_t1_v1"),
            "baseline": {"path": str(baseline), "prepared_sha256": sha(baseline / "prepared.json")},
            "model": original["model"], "model_call_settings": original["model_call_settings"],
            "model_call_source": original["model_call_source"], "generation_settings": settings,
            "temperature": 1, "repeats": repeats, "seed": original["seed"], "method": original["method"],
            "checkpoints": original["checkpoints"], "policies": policies, "cases": cases,
            "trajectories": trajectories,
            "reference": ("Independent Qwen3.8 regeneration under the trajectory's original collection context CP0. "
                          "Historical actions provide provenance and prefix layout only; never use them as scores' "
                          "reference." if regenerated else
                          "Original valid recorded CP0 acting actions from the historical Qwen3.6 trajectory. "
                          "Also collect one independent Qwen3.8 CP0 reconstruction as a matched reference control."),
            "comparison": ("Independently regenerate both sides under their respective contexts using identical "
                           "model/sampling settings and original prefixes; concatenate all generated actions in "
                           "prefix order and score the two complete sequences once. Never insert historical actions "
                           "or shared history into either side. " if regenerated else
                           "Concatenate all newly predicted actions in original prefix order, then score once "
                           "against the whole original trajectory. Never copy original actions into predictions. ")
                          + "Candidate actions are not executed or used to modify later original prefixes.",
            "distance": "Stored checkpoint update index minus original CP0 index (0,10,...). "
                        "CP-pair analysis and independent same-checkpoint controls remain available separately."}


def all_jobs(prepared):
    for case in prepared["cases"]:
        yield case, 0, -1
        for point in prepared["checkpoints"][case["benchmark"]]:
            for repeat in range(prepared["repeats"]):
                yield case, point, repeat


def label(case, point, repeat):
    return f"{case['benchmark']}_{case['id']}_{context_name(point)}_{repeat:02d}"


def copy_reusable(prepared, output):
    baseline = Path(prepared["baseline"]["path"])
    old = json.loads((baseline / "prepared.json").read_text())
    if prepared["temperature"] != old["temperature"]:
        return 0  # A T1 response cannot provide independent T0 evidence.
    lookup = {c["id"]: c for c in old["cases"]}
    endpoint = Path(old["baseline"]["path"])
    reused = 0
    for case, point, repeat in all_jobs(prepared):
        prior = lookup.get(case["id"])
        if prior is None:
            continue
        request = actor_request(prepared, case, point)
        if request != prior["requests"][context_name(point)]:
            raise ValueError("Reused probe request differs")
        source = (baseline / "draws" / f"{label(case, point, repeat)}.json") if repeat >= 0 else (
            endpoint / "draws" / f"{case['benchmark']}_{case['id']}_initial_-1.json"
        )
        if not source.exists():
            continue
        row = json.loads(source.read_text())
        if row["status"] != "completed":
            continue
        row.update(context=context_name(point), reused_from=str(source), reused_sha256=sha(source))
        target = output / "draws" / f"{label(case, point, repeat)}.json"
        if target.exists() and json.loads(target.read_text()) != row:
            raise ValueError("Cached reused draw changed")
        write_json(target, row)
        reused += 1
    return reused


def copy_additional(prepared, output, sources):
    """Reuse interrupted runs only when every request field and draw identity match."""
    copied = 0
    for folder in sources:
        previous = json.loads((folder / "prepared.json").read_text())
        by_id = {c["id"]: c for c in previous["cases"]}
        for case, point, repeat in all_jobs(prepared):
            name = f"{label(case, point, repeat)}.json"
            source, target = folder / "draws" / name, output / "draws" / name
            if target.exists() or not source.exists():
                continue
            old = by_id.get(case["id"])
            if old is None or actor_request(previous, old, point) != actor_request(prepared, case, point):
                raise ValueError("Interrupted-run request differs")
            row = json.loads(source.read_text())
            if (row["case"], row["context"], row["repeat"]) != (case["id"], context_name(point), repeat):
                raise ValueError("Interrupted-run draw identity differs")
            if row["status"] != "completed":
                continue
            row.update(reused_from=str(source), reused_sha256=sha(source))
            write_json(target, row)
            copied += 1
    return copied


def run_one(evaluator, output, prepared, case, point, repeat):
    request_case = {**case, "requests": {context_name(point): actor_request(prepared, case, point)}}
    return run_draw(evaluator, output, request_case, context_name(point), repeat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--comparison", choices=("recorded_original", "regenerated_contexts"),
                        default="regenerated_contexts")
    parser.add_argument("--reuse-run", type=Path, action="append", default=[])
    parser.add_argument("--paired-run", type=Path,
                        help="Use the frozen complete T1 corpus for a separate temperature-only ablation")
    parser.add_argument("--temperature", type=float, choices=(0, 1))
    args = parser.parse_args()
    if args.repeats < 2 or args.workers < 1:
        parser.error("At least two full-trajectory repeats and one worker are required")
    if (args.paired_run is None) != (args.temperature is None):
        parser.error("--paired-run and --temperature must be supplied together")
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "prepared.json"
    prepared = (prepare_temperature_ablation(args.paired_run.resolve(), args.temperature) if args.paired_run else
                prepare(args.baseline, args.output, args.repeats, args.comparison))
    if prepared["repeats"] != args.repeats:
        raise ValueError("A paired ablation must retain the original repetition count")
    if path.exists() and json.loads(path.read_text()) != prepared:
        raise ValueError("Frozen protocol changed")
    write_json(path, prepared)
    copy_reusable(prepared, args.output)
    copy_additional(prepared, args.output, [p.resolve() for p in args.reuse_run])
    jobs = list(all_jobs(prepared))
    missing = [j for j in jobs if not (args.output / "draws" / f"{label(j[0], j[1], j[2])}.json").exists()]
    reused = sum("reused_from" in json.loads(p.read_text()) for p in (args.output / "draws").glob("*.json"))
    print(json.dumps({"trajectories": len(prepared["trajectories"]), "prefixes": len(prepared["cases"]),
                      "total_draws": len(jobs), "reused": reused, "missing": len(missing)}), flush=True)
    if args.prepare_only:
        return
    os.environ["EXGENTIC_ACTOR_OUTPUT_UNLIMITED"] = "1"
    configure_llm_slots(args.workers)
    settings = prepared["model_call_settings"]
    evaluator = SpanPolicyShiftEvaluator(
        model=prepared["model"], base_url=args.base_url, output=args.output, temperature=prepared["temperature"],
        max_tokens=None, timeout=settings["timeout"], num_retries=settings["num_retries"], match_source_settings=True,
    )
    (args.output / "calls").mkdir(exist_ok=True)
    write_json(args.output / "launch.json", {"pid": os.getpid(), "command": sys.argv,
               "workers": args.workers, "runner_sha256": sha(Path(__file__)), "started_unix": time.time()})
    random.Random(prepared["seed"]).shuffle(missing)
    finished, errors = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, evaluator, args.output, prepared, *job) for job in missing]
        for future in as_completed(futures):
            row = future.result()
            finished += 1
            errors += row["status"] == "error"
            if finished % 10 == 0 or finished == len(missing):
                status = {"new_finished": finished, "new_total": len(missing), "errors": errors,
                          "reused": reused, "expected_total_draws": len(jobs), "updated_unix": time.time()}
                write_json(args.output / "status.json", status)
                print(json.dumps(status), flush=True)
    write_json(args.output / "collection_complete.json", {"expected_draws": len(jobs),
               "new_finished": finished, "errors": errors, "completed_unix": time.time()})


if __name__ == "__main__":
    main()
