# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Compare complete concatenations regenerated under two contexts, never original actions."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.premise.audit_group_relative_f1 import action_prf, canonical, sha, write_json  # noqa: E402
from scripts.premise.audit_group_relative_update_distance import context_name, request_audit  # noqa: E402
from scripts.premise.trajectory_text_metrics import LexicalScorer, definitions  # noqa: E402

METRICS = ("precision", "recall", "f1", "bleu1", "bleu4", "meteor")
SCORER = None


def concatenate(steps):
    """Concatenate new action atoms only; step order and multiplicities are preserved."""
    if not steps or any(step is None for step in steps):
        raise ValueError("A full reconstruction requires every original acting prefix")
    return [action for step in steps for action in step]


def compare_steps(predicted_steps, reference_steps, mode, scorer):
    if len(predicted_steps) != len(reference_steps):
        raise ValueError("Reconstructed sequences must cover identical original prefixes")
    predicted, reference = concatenate(predicted_steps), concatenate(reference_steps)
    prefix = [action_prf(p, r, mode) for p, r in zip(predicted_steps, reference_steps, strict=True)]
    return {**action_prf(predicted, reference, mode), **scorer.score(predicted, reference, mode),
            **{f"prefix_{m}": fmean(s[m] for s in prefix) for m in ("precision", "recall", "f1")},
            "exact_action_sequence": float(canonical(predicted) == canonical(reference)),
            "predicted_actions": len(predicted), "reference_actions": len(reference)}


def build_payloads(prepared, draws, require_complete=True):
    if prepared["protocol"] not in {"full_trajectory_regenerated_contexts_cp10_t1_v1",
                                    "full_trajectory_regenerated_contexts_cp10_t0_v1"}:
        raise ValueError("This analysis exclusively compares regenerated contexts")
    expected = {(c["id"], context_name(p), k) for c in prepared["cases"]
                for p in prepared["checkpoints"][c["benchmark"]] for k in range(prepared["repeats"])}
    expected |= {(c["id"], context_name(0), -1) for c in prepared["cases"]}
    index = {}
    for row in draws:
        key = row["case"], row["context"], row["repeat"]
        if key not in expected or key in index:
            raise ValueError("Unexpected or duplicate draw identity")
        index[key] = row
    missing = len(expected - set(index))
    if require_complete and missing:
        raise ValueError(f"Incomplete bank: {missing} missing draws")
    payloads, coverage = [], {}
    for benchmark, points in prepared["checkpoints"].items():
        trajectories = [t for t in prepared["trajectories"] if t["benchmark"] == benchmark]
        included, excluded = [], []
        for trajectory in trajectories:
            identifiers = [s["case"] for s in trajectory["steps"]]
            if [s["action_index"] for s in trajectory["steps"]] != list(range(len(identifiers))):
                raise ValueError("Original prefix order is incomplete or duplicated")
            keys = [(identifier, context_name(p), k) for identifier in identifiers
                    for p in points for k in range(prepared["repeats"])]
            keys += [(identifier, context_name(0), -1) for identifier in identifiers]
            failures = []
            for key in keys:
                row = index.get(key)
                if row is None or row["status"] != "completed":
                    failures.append({"case": key[0], "context": key[1], "repeat": key[2],
                                     "error_type": row.get("error_type", "unknown") if row else "missing"})
            if failures:
                excluded.append({"task": trajectory["task"], "failures": failures})
                continue
            included.append(trajectory["task"])
            banks = {p: [[index[(i, context_name(p), k)]["actions"] for i in identifiers]
                         for k in range(prepared["repeats"])] for p in points}
            fixed = [index[(i, context_name(0), -1)]["actions"] for i in identifiers]
            common = {"benchmark": benchmark, "task": trajectory["task"], "prefixes": len(identifiers)}
            for earlier, later in itertools.combinations_with_replacement(points, 2):
                payloads.append({**common, "kind": "pairwise", "earlier": earlier, "later": later,
                                 "reference_banks": banks[earlier], "prediction_banks": banks[later]})
            for point in points:
                payloads.append({**common, "kind": "cp0_reference", "earlier": 0, "later": point,
                                 "reference_banks": [fixed], "prediction_banks": banks[point]})
        coverage[benchmark] = {"planned_tasks": len(trajectories), "complete_tasks": len(included),
                               "included_tasks": included, "excluded_tasks": excluded,
                               "planned_prefixes": sum(len(t["steps"]) for t in trajectories),
                               "complete_prefixes": sum(len(t["steps"]) for t in trajectories
                                                        if t["task"] in included)}
    return payloads, coverage, {"expected": len(expected), "present": len(index), "missing": missing,
                                "errors": sum(r["status"] != "completed" for r in draws),
                                "reused": sum("reused_from" in r for r in draws)}


def build_complete_sequence_payloads(prepared, draws):
    """Secondary CP0 comparison: retain valid whole sequences, never partial prefixes.

    Keep tasks with at least two complete independent CP0 sequences and at least
    one complete sequence at every later checkpoint. Task membership is constant
    across distances; counts of valid sample pairs may differ and are reported.
    """
    index = {(r["case"], r["context"], r["repeat"]): r for r in draws}
    payloads, coverage = [], {}
    for benchmark, points in prepared["checkpoints"].items():
        trajectories = [t for t in prepared["trajectories"] if t["benchmark"] == benchmark]
        included, excluded, sample_counts = [], [], []
        for trajectory in trajectories:
            ids = [s["case"] for s in trajectory["steps"]]

            def complete(point, repeat, identifiers=ids):
                rows = [index.get((i, context_name(point), repeat)) for i in identifiers]
                return [r["actions"] for r in rows] if all(
                    r is not None and r["status"] == "completed" for r in rows
                ) else None

            banks = {p: {k: sequence for k in range(prepared["repeats"])
                         if (sequence := complete(p, k)) is not None} for p in points}
            if (reference := complete(0, -1)) is not None:
                banks[0][-1] = reference
            if len(banks[0]) < 2 or any(not banks[p] for p in points):
                excluded.append({"task": trajectory["task"], "valid_sequences_per_checkpoint": {
                    str(p): len(banks[p]) for p in points}})
                continue
            included.append(trajectory["task"])
            root_ids = sorted(banks[0])
            for point in points:
                prediction_ids = sorted(banks[point])
                payloads.append({"benchmark": benchmark, "task": trajectory["task"], "prefixes": len(ids),
                                 "kind": "cp0_complete_sequences", "earlier": 0, "later": point,
                                 "reference_banks": [banks[0][k] for k in root_ids],
                                 "reference_repeat_ids": root_ids,
                                 "prediction_banks": [banks[point][k] for k in prediction_ids],
                                 "prediction_repeat_ids": prediction_ids})
                sample_counts.append({"task": trajectory["task"], "point": point,
                                      "reference_sequences": len(root_ids),
                                      "candidate_sequences": len(prediction_ids)})
        coverage[benchmark] = {"planned_tasks": len(trajectories), "complete_tasks": len(included),
                               "included_tasks": included, "excluded_tasks": excluded,
                               "planned_prefixes": sum(len(t["steps"]) for t in trajectories),
                               "complete_prefixes": sum(len(t["steps"]) for t in trajectories
                                                        if t["task"] in included),
                               "sample_counts": sample_counts}
    return payloads, coverage


def initialize_worker(dependencies, nltk_data):
    global SCORER
    SCORER = LexicalScorer(Path(dependencies), Path(nltk_data))


def score_payload(payload, scorer=None):
    scorer = scorer or SCORER
    if scorer is None:
        raise ValueError("Metric worker is not initialized")
    rows = []
    for mode in ("strict", "loose"):
        scores = []
        for i, reference in enumerate(payload["reference_banks"]):
            for j, prediction in enumerate(payload["prediction_banks"]):
                ref_id = payload.get("reference_repeat_ids", list(range(len(payload["reference_banks"]))))[i]
                pred_id = payload.get("prediction_repeat_ids", list(range(len(payload["prediction_banks"]))))[j]
                self_control = payload["kind"] in ("pairwise", "cp0_complete_sequences")
                if self_control and payload["earlier"] == payload["later"] and ref_id == pred_id:
                    continue  # Never compare a generated sequence with itself.
                scores.append({"reference_repeat": ref_id, "prediction_repeat": pred_id,
                               **compare_steps(prediction, reference, mode, scorer)})
        keys = [k for k in scores[0] if k not in ("reference_repeat", "prediction_repeat")]
        rows.append({k: payload[k] for k in ("benchmark", "task", "prefixes", "kind", "earlier", "later")}
                    | {"mode": mode, "distance": payload["later"] - payload["earlier"],
                       "sample_pairs": len(scores), **{k: fmean(s[k] for s in scores) for k in keys},
                       "repeats": scores})
    return rows


def score_payload_cached(payload, folder, signature):
    """Cache deterministic metrics by complete sequence contents and scorer version."""
    folder = Path(folder)
    payload_hash = hashlib.sha256(canonical(payload).encode()).hexdigest()
    key = hashlib.sha256(canonical([signature, payload_hash]).encode()).hexdigest()
    path = folder / f"{key}.json"
    with (folder / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            cached = json.loads(path.read_text())
            valid = (cached["signature"] == signature and cached["payload_sha256"] == payload_hash
                     and cached["result_sha256"] == hashlib.sha256(canonical(cached["rows"]).encode()).hexdigest())
            if not valid:
                raise ValueError("Deterministic score cache checksum mismatch")
            return cached["rows"]
        rows = score_payload(payload)
        write_json(path, {"signature": signature, "payload_sha256": payload_hash, "rows": rows,
                          "result_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest()})
        return rows


def aggregate(task_rows, coverage, prepared):
    selves = {(r["benchmark"], r["task"], r["mode"], r["earlier"]): r for r in task_rows
              if r["kind"] == "pairwise" and r["earlier"] == r["later"]}
    root_controls = {(r["benchmark"], r["task"], r["mode"]): r for r in task_rows
                     if r["kind"] == "cp0_complete_sequences" and r["later"] == 0}
    groups = defaultdict(list)
    base_keys = (*METRICS, "prefix_precision", "prefix_recall", "prefix_f1", "exact_action_sequence",
                 "predicted_actions", "reference_actions")
    for row in task_rows:
        for metric in METRICS:
            if row["kind"] == "cp0_complete_sequences":
                a = b = root_controls[(row["benchmark"], row["task"], row["mode"])][metric]
            else:
                a = selves[(row["benchmark"], row["task"], row["mode"], row["earlier"])][metric]
                b = selves[(row["benchmark"], row["task"], row["mode"], row["later"])][metric]
            row[f"self_{metric}"] = (a + b) / 2
            row[f"gap_{metric}"] = (a + b) / 2 - row[metric]
        row["concat_gain_f1"] = row["f1"] - row["prefix_f1"]
        groups[(row["benchmark"], row["kind"], row["mode"], row["earlier"], row["later"])].append(row)
    keys = (*base_keys, "concat_gain_f1", *(f"self_{m}" for m in METRICS), *(f"gap_{m}" for m in METRICS))
    pairs, fixed, complete_sequences = [], [], []
    for (benchmark, kind, mode, earlier, later), rows in sorted(groups.items()):
        result = {"benchmark": benchmark, "kind": kind, "mode": mode, "earlier": earlier, "later": later,
                  "distance": later - earlier, "tasks": len(rows), "prefixes": sum(r["prefixes"] for r in rows),
                  "sample_pairs_per_task": rows[0]["sample_pairs"],
                  "min_sample_pairs_per_task": min(r["sample_pairs"] for r in rows),
                  "max_sample_pairs_per_task": max(r["sample_pairs"] for r in rows),
                  **{k: fmean(r[k] for r in rows) for k in keys}}
        if len({r["sample_pairs"] for r in rows}) > 1:
            result["sample_pairs_per_task"] = None
        if kind == "cp0_reference":
            for metric in METRICS:
                estimates = [fmean(next(s[metric] for s in r["repeats"] if s["prediction_repeat"] == k)
                                   for r in rows) for k in range(prepared["repeats"])]
                result[f"repeat_sd_{metric}"] = pstdev(estimates)
            fixed.append(result)
        elif kind == "cp0_complete_sequences":
            complete_sequences.append(result)
        else:
            pairs.append(result)
    bins = defaultdict(list)
    for row in pairs:
        bins[(row["benchmark"], row["mode"], row["distance"])].append(row)
    distances = []
    for (benchmark, mode, distance), rows in sorted(bins.items()):
        distances.append({"benchmark": benchmark, "mode": mode, "distance": distance,
                          "checkpoint_pairs": len(rows), "tasks": rows[0]["tasks"],
                          "prefixes": rows[0]["prefixes"], **{k: fmean(r[k] for r in rows) for k in keys}})
    return {"protocol": prepared["protocol"], "coverage": coverage, "model": prepared["model"],
            "temperature": prepared["temperature"], "repeats": prepared["repeats"],
            "metric_definitions": definitions(), "distances": distances, "cp0_reference": fixed,
            "cp0_complete_sequences": complete_sequences,
            "checkpoint_pairs": pairs, "task_comparisons": task_rows,
            "historical_actions_used_for_scoring": False,
            "aggregation": "Score each concatenated regenerated sequence pair once; equal task mean within "
                           "checkpoint pair, then equal CP-pair mean within distance. One complete-trajectory "
                           "intersection across every CP, repeat and independent CP0 reference. Same-checkpoint "
                           "comparisons use distinct draws in both directions. Positive distances use later "
                           "context as prediction and earlier as reference. Prefix PRF is an aligned-step "
                           "diagnostic on the same generated samples, not the main scoring unit."}


def save_tables(output, summary):
    for name, key in (("trajectory_distance_summary", "distances"), ("trajectory_cp0_reference", "cp0_reference"),
                      ("trajectory_checkpoint_pairs", "checkpoint_pairs"),
                      ("trajectory_cp0_complete_sequences", "cp0_complete_sequences")):
        rows = summary[key]
        with (output / f"{name}.csv").open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-subdir", default="analysis")
    args = parser.parse_args()
    args.run = args.run.resolve()
    prepared = json.loads((args.run / "prepared.json").read_text())
    draws = [json.loads(p.read_text()) for p in (args.run / "draws").glob("*.json")]
    payloads, coverage, counts = build_payloads(prepared, draws, not args.allow_incomplete)
    complete_payloads, complete_coverage = build_complete_sequence_payloads(prepared, draws)
    payloads += complete_payloads
    if not payloads:
        raise ValueError("No complete trajectories yet")
    output = args.run / args.output_subdir
    output.mkdir(exist_ok=True)
    cache = args.run / "metric_cache"
    cache.mkdir(exist_ok=True)
    signature = hashlib.sha256(canonical({
        "analysis": sha(Path(__file__)), "lexical": sha(Path(__file__).with_name("trajectory_text_metrics.py")),
        "wordnet_zip": sha(args.nltk_data.resolve() / "corpora/wordnet.zip"),
    }).encode()).hexdigest()
    task_rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(args.dependencies.resolve()), str(args.nltk_data.resolve()))) as pool:
        futures = [pool.submit(score_payload_cached, p, str(cache), signature) for p in payloads]
        for i, future in enumerate(as_completed(futures), 1):
            task_rows.extend(future.result())
            if i % 10 == 0 or i == len(futures):
                write_json(output / "status.json", {"finished_comparisons": i, "total": len(futures)})
                print(f"trajectory comparisons {i}/{len(futures)}", flush=True)
    task_rows.sort(key=lambda r: (r["benchmark"], r["task"], r["kind"], r["mode"], r["earlier"], r["later"]))
    summary = aggregate(task_rows, coverage, prepared)
    summary["complete_sequence_coverage"] = complete_coverage
    summary["complete_sequence_definition"] = (
        "Secondary view uses only valid full sequences; no partial-prefix filling. Keep a constant task "
        "intersection with >=2 independently generated CP0 sequences (repeat IDs -1,0,1) and >=1 complete "
        "candidate sequence at every checkpoint. Compare all valid CP0/candidate pairs; at distance0 exclude "
        "identical repeat IDs in both directions. Sample pair counts may vary; report min/max and condition "
        "interpretation on successful reconstruction. Dotted baseline is CP0 independent self agreement, "
        "not a balanced two-context self control. Strict all-repeats-common results are preserved separately."
    )
    summary.update(draw_counts=counts, analysis_status="interim" if counts["missing"] else "complete",
                   request_audit=request_audit(prepared, args.run),
                   analysis_sha256=sha(Path(__file__)), lexical_source_sha256=sha(Path(__file__).with_name(
                       "trajectory_text_metrics.py")))
    summary["score_cache_signature"] = signature
    write_json(output / "summary.json", summary)
    save_tables(output, summary)
    print(json.dumps({"coverage": coverage, "draw_counts": counts}), flush=True)


if __name__ == "__main__":
    main()
