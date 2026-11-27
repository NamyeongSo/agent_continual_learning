# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Measure conditional action agreement at every ten-update checkpoint distance.

Reuse a completed matched-generation audit's endpoints; sample intermediate
checkpoints independently, then compare all checkpoint pairs on common prefixes.
No benchmark environment actions are executed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
import random
import sqlite3
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.audit_group_relative_f1 import (  # noqa: E402
    SpanPolicyShiftEvaluator,
    action_prf,
    canonical,
    configure_llm_slots,
    content_signature,
    equal_task_mean,
    replace_span,
    run_draw,
    sha,
    write_json,
)

METRICS = ("precision", "recall", "f1")
MODES = ("strict", "loose")


def context_name(point):
    return f"cp{point:03d}"


def draw_path(output, case, point, repeat):
    return output / "draws" / f"{case['benchmark']}_{case['id']}_{context_name(point)}_{repeat:02d}.json"


def prepare(baseline):
    original = json.loads((baseline / "prepared.json").read_text())
    summary = json.loads((baseline / "summary.json").read_text())
    valid = {c["id"] for c in summary["cases"]}
    source = Path(original["source"])
    config = json.loads(Path(original["model_call_source"]["path"]).read_text())
    if config["model"] != original["model"] or config["model_settings"] != original["model_call_settings"]:
        raise ValueError("Live configuration differs from the endpoint audit; use a separate matched cohort")
    if original["temperature"] != 1 or original["repeats"] != 10:
        raise ValueError("This protocol requires temperature 1 and ten independent draws per checkpoint")
    cases, checkpoints = [], {}
    db = sqlite3.connect(f"file:{source / 'policy_shift.sqlite3'}?mode=ro", uri=True)
    for case in original["cases"]:
        if case["id"] not in valid:
            continue
        anchor_path = Path(case["source"])
        if sha(anchor_path) != case["source_sha256"]:
            raise ValueError("Original history bytes changed")
        if "parser" in case and sha(Path(case["parser"]["path"])) != case["parser"]["sha256"]:
            raise ValueError("Native action parser changed")
        policies = db.execute(
            "SELECT p.point,t.content FROM policies p JOIN prompts t ON t.sha=p.prompt "
            "WHERE p.benchmark=? AND p.method=? AND p.seed=? ORDER BY p.point",
            (case["benchmark"], original["method"], original["seed"]),
        ).fetchall()
        points = [p for p, _ in policies]
        if points != list(range(0, points[-1] + 1, 10)):
            raise ValueError("Expected exactly one checkpoint at every ten updates, starting at zero")
        checkpoints[case["benchmark"]] = points
        new = {k: copy.deepcopy(v) for k, v in case.items()
               if k not in ("requests", "context_sha256", "context_points")}
        new.update(points=points, requests={}, context_sha256={}, reuse={})
        anchor = json.loads(anchor_path.read_text())
        for point, content in policies:
            key = context_name(point)
            request = replace_span(anchor, content)
            request["generation_settings"] = copy.deepcopy(case["requests"]["initial"]["generation_settings"])
            new["requests"][key] = request
            new["context_sha256"][key] = hashlib.sha256(content.encode()).hexdigest()
            if point in (points[0], points[-1]):
                old_context = "initial" if point == points[0] else "learned"
                if request != case["requests"][old_context]:
                    raise ValueError("Reused endpoint request differs from the reconstructed checkpoint request")
                if new["context_sha256"][key] != case["context_sha256"][old_context]:
                    raise ValueError("Endpoint policy hash differs")
                new["reuse"][key] = old_context
        cases.append(new)
    db.close()
    return {
        "protocol": "all_checkpoint_pairs_distance10_matched_t1_v1",
        "model": original["model"], "temperature": 1, "repeats": 10, "seed": original["seed"],
        "method": original["method"], "source": original["source"],
        "model_call_source": original["model_call_source"], "model_call_settings": original["model_call_settings"],
        "baseline": {"path": str(baseline), "prepared_sha256": sha(baseline / "prepared.json"),
                     "summary_sha256": sha(baseline / "summary.json")},
        "excluded_baseline_cases": [c["id"] for c in original["cases"] if c["id"] not in valid],
        "checkpoints": checkpoints, "cases": cases,
        "distance_definition": "Absolute difference of stored training/update step indices, not accepted mutations.",
        "aggregation": "Actions -> prefix mean -> equal prefix weights within task -> equal task weights "
                       "within checkpoint pair -> equal checkpoint pair weights within distance; common complete "
                       "prefix intersection across all checkpoints. Later checkpoint is prediction. Distance zero "
                       "uses ordered distinct draws, never a draw compared with itself.",
    }


def reuse_endpoints(prepared, output):
    baseline, rows = Path(prepared["baseline"]["path"]), []
    for case in prepared["cases"]:
        for point in case["points"]:
            context = context_name(point)
            if context not in case["reuse"]:
                continue
            for repeat in range(prepared["repeats"]):
                source = baseline / "draws" / (
                    f"{case['benchmark']}_{case['id']}_{case['reuse'][context]}_{repeat:02d}.json"
                )
                row = json.loads(source.read_text())
                if (row["status"] != "completed" or row["case"] != case["id"] or row["repeat"] != repeat
                        or row["context"] != case["reuse"][context]):
                    raise ValueError("Endpoint draw identity/validity mismatch")
                row.update(context=context, reused_from=str(source), reused_sha256=sha(source))
                path = draw_path(output, case, point, repeat)
                if path.exists() and json.loads(path.read_text()) != row:
                    raise ValueError("Cached endpoint differs from the pinned original")
                write_json(path, row)
                rows.append(row)
    return rows


def pair_scores(earlier, later, mode, same_checkpoint=False):
    pairs = [(b, a) for i, a in enumerate(earlier) for j, b in enumerate(later)
             if not same_checkpoint or i != j]
    scores = [action_prf(b, a, mode) for b, a in pairs]
    result = {metric: fmean(s[metric] for s in scores) for metric in METRICS}
    result.update(f1_pair_sd=pstdev(s["f1"] for s in scores),
                  zero_f1_fraction=fmean(s["f1"] == 0 for s in scores),
                  exact_action_match=fmean(canonical(b) == canonical(a) for b, a in pairs))
    return result


def analyze(prepared, draws, require_complete=True):
    index = {}
    for row in draws:
        key = (row["case"], row["context"], row["repeat"])
        if key in index:
            raise ValueError(f"Duplicate draw: {key}")
        index[key] = row
    pair_rows, prefix_rows, coverage = [], [], {}
    repeats = prepared["repeats"]
    for benchmark, points in prepared["checkpoints"].items():
        cases = [c for c in prepared["cases"] if c["benchmark"] == benchmark]
        valid, missing, errors = [], 0, []
        by_point = Counter()
        for case in cases:
            complete = True
            for point in points:
                for repeat in range(repeats):
                    row = index.get((case["id"], context_name(point), repeat))
                    if row is None:
                        complete, missing = False, missing + 1
                    elif row["status"] != "completed":
                        complete = False
                        by_point[point] += 1
                        errors.append({k: row[k] for k in ("case", "context", "repeat", "error_type", "error")})
            if complete:
                valid.append(case)
        if require_complete and missing:
            raise ValueError(f"Incomplete {benchmark}: {missing} missing draws")
        coverage[benchmark] = {"planned_prefixes": len(cases), "complete_prefixes": len(valid),
                               "tasks": len({c["task"] for c in valid}), "missing_draws": missing,
                               "error_draws": len(errors), "errors_by_checkpoint": dict(by_point),
                               "errors": errors, "included_cases": [c["id"] for c in valid]}
        for case in valid:
            banks = {p: [index[(case["id"], context_name(p), k)]["actions"] for k in range(repeats)]
                     for p in points}
            # Diagnose exact code/command changes separately from formatting or command duration.
            signatures = {p: [canonical(content_signature(a, benchmark)) for a in bank] for p, bank in banks.items()}
            for mode in MODES:
                selves = {p: pair_scores(banks[p], banks[p], mode, True) for p in points}
                for earlier, later in itertools.combinations_with_replacement(points, 2):
                    score = selves[earlier] if earlier == later else pair_scores(banks[earlier], banks[later], mode)
                    prefix_rows.append({
                        "benchmark": benchmark, "task": case["task"], "case": case["id"], "mode": mode,
                        "earlier": earlier, "later": later, "distance": later - earlier, **score,
                        "self_baseline_f1": (selves[earlier]["f1"] + selves[later]["f1"]) / 2,
                        "sampling_adjusted_gap": (selves[earlier]["f1"] + selves[later]["f1"]) / 2 - score["f1"],
                        "content_action_match": fmean(
                            b == a for i, a in enumerate(signatures[earlier])
                            for j, b in enumerate(signatures[later]) if earlier != later or i != j
                        ),
                        "identical_context": int(case["context_sha256"][context_name(earlier)]
                                                 == case["context_sha256"][context_name(later)]),
                    })
        for mode in MODES:
            for earlier, later in itertools.combinations_with_replacement(points, 2):
                rows = [r for r in prefix_rows if r["benchmark"] == benchmark and r["mode"] == mode
                        and r["earlier"] == earlier and r["later"] == later]
                if not rows:
                    continue
                keys = (*METRICS, "f1_pair_sd", "zero_f1_fraction", "exact_action_match", "self_baseline_f1",
                        "sampling_adjusted_gap", "identical_context", "content_action_match")
                pair_rows.append({"benchmark": benchmark, "mode": mode, "earlier": earlier, "later": later,
                                  "distance": later - earlier, "prefixes": len(rows),
                                  "tasks": len({r["task"] for r in rows}),
                                  **{k: equal_task_mean(rows, k) for k in keys}})
    distance_rows = []
    groups = defaultdict(list)
    for row in pair_rows:
        groups[(row["benchmark"], row["mode"], row["distance"])].append(row)
    for (benchmark, mode, distance), rows in sorted(groups.items()):
        keys = (*METRICS, "f1_pair_sd", "zero_f1_fraction", "exact_action_match", "self_baseline_f1",
                "sampling_adjusted_gap", "identical_context", "content_action_match")
        distance_rows.append({"benchmark": benchmark, "mode": mode, "distance": distance,
                              "checkpoint_pairs": len(rows), "prefixes": rows[0]["prefixes"],
                              "tasks": rows[0]["tasks"], **{k: fmean(r[k] for r in rows) for k in keys}})
    return {"protocol": prepared["protocol"], "model": prepared["model"], "temperature": prepared["temperature"],
            "analysis_status": "interim" if any(c["missing_draws"] for c in coverage.values()) else "complete",
            "repeats": repeats, "distance_definition": prepared["distance_definition"],
            "aggregation": prepared["aggregation"], "coverage": coverage,
            "distances": distance_rows, "checkpoint_pairs": pair_rows, "prefix_pairs": prefix_rows,
            "draw_counts": dict(Counter("reused" if "reused_from" in r else "new" for r in draws)),
            "analysis_source_sha256": sha(Path(__file__))}


def request_audit(prepared, output):
    settings = prepared["model_call_settings"]
    expected = {k: v for k, v in settings.items() if k not in ("timeout", "num_retries", "max_tokens")}
    expected["model"] = prepared["model"]
    calls, errors, output_tokens, input_tokens, mismatches = 0, 0, [], [], []
    for path in sorted((output / "calls").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            calls += 1
            errors += int("error_type" in row)
            request = row["request"]
            for k, v in expected.items():
                actual = request.get(k)
                matches = (str(actual).removeprefix("openai/") == str(v).removeprefix("openai/")) if k == "model" else (
                    actual == v
                )
                if not matches:
                    mismatches.append({"file": path.name, "field": k})
            if "max_tokens" in request or "max_completion_tokens" in request:
                mismatches.append({"file": path.name, "field": "output_cap_present"})
            usage = (row.get("response") or {}).get("usage") or {}
            if usage.get("completion_tokens") is not None:
                output_tokens.append(usage["completion_tokens"])
            if usage.get("prompt_tokens") is not None:
                input_tokens.append(usage["prompt_tokens"])
    histories_ok = all(sha(Path(c["source"])) == c["source_sha256"] for c in prepared["cases"])
    return {"new_physical_calls": calls, "new_error_attempts": errors, "settings_mismatches": mismatches,
            "measured_input_tokens": sum(input_tokens), "measured_output_tokens": sum(output_tokens),
            "max_output_tokens_observed": max(output_tokens, default=None),
            "source_histories_unchanged": histories_ok, "all_requested_settings_match": not mismatches}


def write_report(output, summary):
    for name, rows in (("distance_summary", summary["distances"]),
                       ("checkpoint_pairs", summary["checkpoint_pairs"]), ("prefix_pairs", summary["prefix_pairs"])):
        with (output / f"{name}.csv").open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = ["# Qwen3.8 temperature 1: agreement versus update distance", "",
             f"Analysis status: **{summary['analysis_status']}**.", "",
             f"Protocol: `{summary['protocol']}`. Ten independent draws per checkpoint and prefix.", "",
             summary["distance_definition"], "", summary["aggregation"], "",
             "Distance zero measures independent sampling agreement, with no self-draw comparisons. "
             "Precision uses the later checkpoint as prediction and the earlier as reference. "
             "Loose overlap compares action/function names with multiplicities, "
             "ignoring arguments and command/code text. "
             "P/R/F1 are independently averaged; reported F1 is not the harmonic mean of averaged P and R.", "",
             "| Benchmark | mode | distance | CP pairs | tasks/prefixes | precision | recall "
             "| F1 | self F1 baseline | gap |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["distances"]:
        lines.append(f"| {r['benchmark']} | {r['mode']} | {r['distance']} | {r['checkpoint_pairs']} | "
                     f"{r['tasks']}/{r['prefixes']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                     f"{r['self_baseline_f1']:.4f} | {r['sampling_adjusted_gap']:.4f} |")
    lines += ["", "Gap = mean of the two checkpoints' independent self F1 minus their cross-checkpoint F1. "
              "It is a descriptive sampling control, not a validated policy ratio or optimization objective.", "",
              "Coverage uses one complete-prefix intersection across every checkpoint, preserving identical weights "
              "at all distances. Errors are excluded uniformly, never scored zero. Endpoint banks were reused from "
              "the prior matched Qwen3.8 audit; intermediate banks are newly sampled. Historical prefixes and learned "
              "contexts originated with Qwen3.6, so this is a conditional agreement pilot, not a Qwen3.8 training "
              "performance or semantic correctness evaluation. Four tasks per benchmark; no environment replay. "
              "Distance is correlated with training position and context length; "
              "pair-level tables retain this information. Output is unlimited, input is capped at 100k "
              "and prefix turn limits remain BFCL30/AppWorld40/Terminal40.", "",
              "`f1_pair_sd` in CSV is average within-prefix pair-score SD, a descriptive variability measure, "
              "not a standard error: all-pair comparisons share generated samples.", "",
              "`content_action_match` in CSV ignores only Python formatting/comments (AST equality) in AppWorld "
              "and command duration in Terminal-Bench. It still compares actual code/commands, unlike loose name-only "
              "overlap; it does not establish semantic equivalence or execution correctness.", "",
              "Coverage: `" + json.dumps(summary["coverage"], ensure_ascii=False) + "`", ""]
    (output / "report.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmarks = sorted(summary["coverage"])
    fig, axes = plt.subplots(len(benchmarks), 3, figsize=(12, 9), squeeze=False, sharey=True)
    for i, benchmark in enumerate(benchmarks):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for mode, color in (("strict", "#235789"), ("loose", "#c55a11")):
                rows = [r for r in summary["distances"] if r["benchmark"] == benchmark and r["mode"] == mode]
                ax.plot([r["distance"] for r in rows], [r[metric] for r in rows], "o-", color=color, label=mode)
                if metric == "f1":
                    ax.plot([r["distance"] for r in rows], [r["self_baseline_f1"] for r in rows],
                            ":", color=color, alpha=0.7, label=f"{mode} self baseline")
            ax.set(title=f"{benchmark}: {metric}", ylim=(-0.03, 1.03), xlabel="Update distance")
            distances = [r["distance"] for r in summary["distances"] if r["benchmark"] == benchmark]
            if distances:
                ax.set_xticks(list(range(0, max(distances) + 1, 10)))
            else:
                ax.text(.5, .5, "No complete prefixes yet", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel("Mean agreement")
            if i == 0 and j in (0, 2):
                ax.legend(fontsize=8)
    fig.suptitle("Conditional action agreement at T=1 | independent repeats at distance 0")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"agreement_by_update_distance.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(benchmarks), 2, figsize=(10, 9), squeeze=False)
    for i, benchmark in enumerate(benchmarks):
        strict = [r for r in summary["distances"] if r["benchmark"] == benchmark and r["mode"] == "strict"]
        distances = [r["distance"] for r in strict]
        axes[i, 0].plot(distances, [r["f1"] for r in strict], "o-", color="#235789", label="Cross-checkpoint F1")
        axes[i, 0].plot(distances, [r["self_baseline_f1"] for r in strict], "o:", color="#777777",
                        label="Mean self F1 of the two checkpoints")
        for mode, color in (("strict", "#235789"), ("loose", "#c55a11")):
            rows = [r for r in summary["distances"] if r["benchmark"] == benchmark and r["mode"] == mode]
            axes[i, 1].plot([r["distance"] for r in rows], [r["sampling_adjusted_gap"] for r in rows],
                            "o-", color=color, label=mode)
        axes[i, 1].axhline(0, color="#777777", linewidth=.8)
        for j in (0, 1):
            axes[i, j].set(title=f"{benchmark}: " + ("strict F1" if j == 0 else "self F1 minus cross F1"),
                           xlabel="Update distance", xticks=distances)
            axes[i, j].grid(alpha=.2)
            if i == 0:
                axes[i, j].legend(fontsize=8)
    fig.suptitle("Sampling controls at T=1 | descriptive means, independently scaled panels")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"strict_f1_sampling_control.{suffix}", dpi=180)
    plt.close(fig)


def finalize(prepared, output, require_complete=True):
    draws = [json.loads(p.read_text()) for p in (output / "draws").glob("*.json")]
    summary = analyze(prepared, draws, require_complete)
    summary["request_audit"] = request_audit(prepared, output)
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Explicitly labeled interim common-prefix analysis")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "prepared.json"
    prepared = json.loads(frozen.read_text()) if args.analyze_only else prepare(args.baseline)
    if frozen.exists() and json.loads(frozen.read_text()) != prepared:
        raise ValueError("Prepared configuration differs; use a separate output directory")
    write_json(frozen, prepared)
    if args.analyze_only:
        summary = finalize(prepared, args.output, not args.allow_incomplete)
        print(json.dumps({"coverage": summary["coverage"], "requests": summary["request_audit"]}), flush=True)
        return
    reused = reuse_endpoints(prepared, args.output)
    jobs = [(case, context_name(point), repeat) for case in prepared["cases"] for point in case["points"]
            if context_name(point) not in case["reuse"] for repeat in range(prepared["repeats"])]
    print(f"Prepared {len(prepared['cases'])} prefixes; {len(reused)} reused draws, {len(jobs)} new draws", flush=True)
    if args.prepare_only:
        return
    configure_llm_slots(args.workers)
    os.environ["EXGENTIC_ACTOR_OUTPUT_UNLIMITED"] = "1"
    settings = prepared["model_call_settings"]
    evaluator = SpanPolicyShiftEvaluator(
        model=prepared["model"], base_url=args.base_url, output=args.output,
        temperature=prepared["temperature"], match_source_settings=True, max_tokens=None,
        timeout=settings["timeout"], num_retries=settings["num_retries"],
    )
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=10) as response:
        serving = json.load(response)
    write_json(args.output / "launch.json", {
        "pid": os.getpid(), "command": sys.argv, "workers": args.workers, "serving_models": serving,
        "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_checkout": True, "runner_sha256": sha(Path(__file__)),
    })
    random.Random(prepared["seed"]).shuffle(jobs)
    finished, errors = 0, 0
    (args.output / "calls").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_draw, evaluator, args.output, *job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            finished += 1
            errors += row["status"] == "error"
            if finished % 10 == 0 or finished == len(jobs):
                write_json(args.output / "status.json", {
                    "new_finished": finished, "new_total": len(jobs), "reused": len(reused), "new_errors": errors,
                })
                print(f"new draws {finished}/{len(jobs)}; errors {errors}", flush=True)
    summary = finalize(prepared, args.output)
    print(json.dumps({"coverage": summary["coverage"], "requests": summary["request_audit"]}), flush=True)


if __name__ == "__main__":
    main()
