# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Score saved regenerated actions at identical prefixes, without trajectory concatenation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.premise.audit_group_relative_f1 import (  # noqa: E402
    action_prf,
    canonical,
    content_signature,
    sha,
    write_json,
)
from scripts.premise.compare_regenerated_temperatures import ordered_similarity  # noqa: E402
from scripts.premise.trajectory_text_metrics import LexicalScorer, definitions  # noqa: E402

METRICS = ("precision", "recall", "f1", "bleu1", "bleu4", "meteor", "content_precision", "content_recall",
           "content_f1", "ordered_action_similarity", "content_ordered_action_similarity", "exact_action_list",
           "zero_f1_fraction", "one_f1_fraction", "repeat_pair_sd_f1")
SCORER = None


def score_prefix_pair(prediction, reference, benchmark, mode, scorer):
    """One saved prefix's response is the unit, including any calls within that response."""
    cp, cr = content_signature(prediction, benchmark), content_signature(reference, benchmark)
    prf = action_prf(prediction, reference, mode)
    exact_p = prediction if mode == "strict" else [a["name"] for a in prediction]
    exact_r = reference if mode == "strict" else [a["name"] for a in reference]
    return {**prf, **scorer.score(prediction, reference, mode),
            **{f"content_{k}": v for k, v in action_prf(cp, cr, mode).items()},
            "ordered_action_similarity": ordered_similarity(prediction, reference, mode),
            "content_ordered_action_similarity": ordered_similarity(cp, cr, mode),
            "exact_action_list": float(canonical(exact_p) == canonical(exact_r))}


def independent_pairs(earlier, later, bank=False, reference=False):
    refs = [-1] if reference else [-1, 0, 1] if bank else [0, 1]
    preds = [-1, 0, 1] if bank and later == 0 else [0, 1]
    return [(i, j) for i in refs for j in preds if reference or earlier != later or i != j]


def valid_prefixes(prepared, indexes):
    included, excluded = [], []
    for case in prepared["cases"]:
        keys = [(case["id"], p, k) for p in prepared["checkpoints"][case["benchmark"]]
                for k in ([-1, 0, 1] if p == 0 else [0, 1])]
        failures = [{"temperature": temp, "checkpoint": p, "repeat": k,
                     "error_type": index.get((identifier, p, k), {}).get("error_type", "missing")}
                    for temp, index in enumerate(indexes) for identifier, p, k in keys
                    if index.get((identifier, p, k), {}).get("status") != "completed"]
        if failures:
            excluded.append({"case": case["id"], "benchmark": case["benchmark"], "task": case["task"],
                             "failures": failures})
        else:
            included.append(case)
    return included, excluded


def build_specs(prepared, matched, valid):
    tasks = {(t["benchmark"], t["task"]): t for t in prepared["trajectories"]}
    cases = {c["id"]: c for c in prepared["cases"]}
    specs = []

    def add(case, cohort, view, earlier, later, pairs):
        specs.append({"case": case["id"], "benchmark": case["benchmark"], "task": case["task"],
                      "cohort": cohort, "view": view, "earlier": earlier, "later": later,
                      "pairs": sorted(set(pairs))})

    views = {"pairwise": "all_pairs", "cp0_reference": "cp0_reference", "cp0_complete_sequences": "cp0_bank"}
    for row in matched["task_comparisons"]:
        if row["mode"] != "strict":
            continue
        pairs = [(-1 if row["kind"] == "cp0_reference" else p["reference_repeat"], p["prediction_repeat"])
                 for p in row["repeats"]]
        for step in tasks[(row["benchmark"], row["task"])]["steps"]:
            add(cases[step["case"]], "matched_sequences", views[row["kind"]], row["earlier"], row["later"], pairs)
    for case in valid:
        points = prepared["checkpoints"][case["benchmark"]]
        for a, b in itertools.combinations_with_replacement(points, 2):
            add(case, "valid_prefixes", "all_pairs", a, b, independent_pairs(a, b))
        for point in points:
            add(case, "valid_prefixes", "cp0_bank", 0, point, independent_pairs(0, point, bank=True))
            add(case, "valid_prefixes", "cp0_reference", 0, point, independent_pairs(0, point, reference=True))
    return specs


def initialize(dependencies, nltk_data):
    global SCORER
    SCORER = LexicalScorer(Path(dependencies), Path(nltk_data))


def score_case(job):
    case, comparisons, banks = job
    rows = []
    for temp, bank in enumerate(banks):
        for (earlier, later), pairs in sorted(comparisons.items()):
            for i, j in sorted(pairs):
                reference, prediction = bank[(earlier, i)], bank[(later, j)]
                if reference["status"] != "completed" or prediction["status"] != "completed":
                    raise ValueError("A failed draw entered a prefix pair")
                for mode in ("strict", "loose"):
                    rows.append({"temperature": temp, "case": case["id"], "benchmark": case["benchmark"],
                                 "task": case["task"], "earlier": earlier, "later": later,
                                 "reference_repeat": i, "prediction_repeat": j, "mode": mode,
                                 **score_prefix_pair(prediction["actions"], reference["actions"],
                                                     case["benchmark"], mode, SCORER)})
    return rows


def row_identity(row):
    return tuple(row[k] for k in ("temperature", "case", "earlier", "later", "reference_repeat",
                                 "prediction_repeat", "mode"))


def prefix_means(specs, scored):
    rows = []
    for spec in specs:
        for temp in (0, 1):
            for mode in ("strict", "loose"):
                pairs = [scored[(temp, spec["case"], spec["earlier"], spec["later"], i, j, mode)]
                         for i, j in spec["pairs"]]
                keys = set(pairs[0]) - {"temperature", "case", "benchmark", "task", "earlier", "later",
                                       "reference_repeat", "prediction_repeat", "mode"}
                row = {k: v for k, v in spec.items() if k != "pairs"}
                row.update(temperature=temp, mode=mode, distance=spec["later"] - spec["earlier"],
                           sample_pairs=len(pairs), **{k: fmean(p[k] for p in pairs) for k in keys})
                row.update(zero_f1_fraction=fmean(p["f1"] == 0 for p in pairs),
                           one_f1_fraction=fmean(p["f1"] == 1 for p in pairs),
                           repeat_pair_sd_f1=pstdev(p["f1"] for p in pairs))
                rows.append(row)
    controls = {(r["temperature"], r["cohort"], r["case"], r["mode"], r["view"], r["later"]): r
                for r in rows if r["earlier"] == r["later"]}
    for r in rows:
        common = r["temperature"], r["cohort"], r["case"], r["mode"]
        if r["view"] == "cp0_bank":
            a = b = controls[(*common, "cp0_bank", 0)]
        else:
            a, b = (controls[(*common, "all_pairs", point)] for point in (r["earlier"], r["later"]))
        for metric in METRICS[:11]:
            r[f"self_{metric}"] = (a[metric] + b[metric]) / 2
            r[f"gap_{metric}"] = r[f"self_{metric}"] - r[metric]
    return rows


def mean_groups(rows, group_keys, score_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    return [{**dict(zip(group_keys, key, strict=True)),
             **{k: fmean(r[k] for r in values) for k in score_keys}, "members": len(values)}
            for key, values in sorted(groups.items())]


def aggregate_prefixes(rows):
    scores = (*METRICS, *(f"self_{k}" for k in METRICS[:11]), *(f"gap_{k}" for k in METRICS[:11]))
    keys = ("temperature", "cohort", "view", "benchmark", "task", "mode", "earlier", "later", "distance")
    tasks = mean_groups(rows, keys, scores)
    bench_keys = tuple(k for k in keys if k != "task")
    benches = mean_groups(tasks, bench_keys, scores)
    for bench in benches:
        selected = [t for t in tasks if all(t[k] == bench[k] for k in bench_keys)]
        bench["tasks"] = bench.pop("members")
        bench["prefixes"] = sum(t["members"] for t in selected)
    distance_keys = tuple(k for k in bench_keys if k not in ("earlier", "later"))
    distances = mean_groups([b for b in benches if b["view"] == "all_pairs"], distance_keys, scores)
    for row in distances:
        example = next(b for b in benches if all(b[k] == row[k] for k in distance_keys))
        row.update(tasks=example["tasks"], prefixes=example["prefixes"], checkpoint_pairs=row.pop("members"))
    curves = [*distances, *(b for b in benches if b["view"] != "all_pairs")]
    return tasks, benches, curves


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_old_prefix_prf(tasks, paired):
    aliases = {"pairwise": "all_pairs", "cp0_reference": "cp0_reference", "cp0_complete_sequences": "cp0_bank"}
    index = {(r["temperature"], r["view"], r["benchmark"], r["task"], r["mode"], r["earlier"], r["later"]): r
             for r in tasks if r["cohort"] == "matched_sequences"}
    checked = 0
    for source in paired["summaries"]:
        for row in source["task_comparisons"]:
            new = index[(source["temperature"], aliases[row["kind"]], row["benchmark"], row["task"], row["mode"],
                         row["earlier"], row["later"])]
            for metric in ("precision", "recall", "f1"):
                if abs(new[metric] - row[f"prefix_{metric}"]) > 1e-12:
                    raise ValueError("Aligned-prefix PRF disagrees with the earlier same-sample diagnostic")
                checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-zero", type=Path, required=True)
    parser.add_argument("--run-one", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [args.run_zero.resolve(), args.run_one.resolve()]
    prepared = [json.loads((r / "prepared.json").read_text()) for r in runs]
    paired_path = runs[0] / "comparison/summary.json"
    paired = json.loads(paired_path.read_text())
    for key in ("cases", "trajectories", "policies", "checkpoints", "repeats", "model"):
        if prepared[0][key] != prepared[1][key]:
            raise ValueError(f"Temperature cohorts differ in {key}")
    if [p["temperature"] for p in prepared] != [0, 1] or prepared[0]["repeats"] != 2:
        raise ValueError("Expected the frozen two-repeat T0/T1 study")
    for key in ("generation_settings", "model_call_settings"):
        settings = [{k: v for k, v in p[key].items() if k != "temperature"} for p in prepared]
        if settings[0] != settings[1]:
            raise ValueError("A non-temperature call setting differs")
    if list(map(str, runs)) != paired["source_runs"] or [sha(r / "analysis/summary.json") for r in runs] != (
        paired["source_summary_sha256"]
    ):
        raise ValueError("The paired comparison does not match these source runs")
    expected = {(c["id"], p, k) for c in prepared[0]["cases"]
                for p in prepared[0]["checkpoints"][c["benchmark"]]
                for k in ([-1, 0, 1] if p == 0 else [0, 1])}
    indexes, hashes = [], []
    for run in runs:
        source = [json.loads(f.read_text()) for f in sorted((run / "draws").glob("*.json"))]
        index = {(r["case"], int(r["context"].removeprefix("cp")), r["repeat"]): r for r in source}
        if len(index) != len(source) or set(index) != expected:
            raise ValueError("Incomplete or duplicate source bank")
        indexes.append(index)
        hashes.append({str(f.relative_to(run)): sha(f) for f in sorted((run / "draws").glob("*.json"))})
    for case in prepared[0]["cases"]:
        if sha(Path(case["source"])) != case["source_sha256"]:
            raise ValueError("An original prefix changed")
    valid, excluded = valid_prefixes(prepared[0], indexes)
    specs = build_specs(prepared[0], paired["summaries"][0], valid)
    by_case = defaultdict(lambda: defaultdict(set))
    for spec in specs:
        by_case[spec["case"]][(spec["earlier"], spec["later"])].update(spec["pairs"])
    cases = {c["id"]: c for c in prepared[0]["cases"]}
    jobs = [(cases[identifier], comparisons,
             [{(p, k): row for (i, p, k), row in index.items() if i == identifier} for index in indexes])
            for identifier, comparisons in sorted(by_case.items())]
    write_json(args.output / "prepared.json", {
        "protocol": "aligned_prefix_regenerated_contexts_t0_t1_cp10_v1", "source_runs": list(map(str, runs)),
        "source_prepared_sha256": [sha(r / "prepared.json") for r in runs],
        "paired_summary_sha256": sha(paired_path), "new_model_calls": 0, "environment_execution": False,
        "unit": "One response action list at the identical original history prefix. Never join different prefixes.",
        "cohorts": {"matched_sequences": "Exact same task/prefix/repeat-ID pairs as prior whole-concat comparison.",
                    "valid_prefixes": "Keep each prefix valid at all checkpoints/repeats/references at BOTH "
                                      "temperatures; fixed prefix/task set, without requiring a complete trajectory."},
        "valid_prefixes": valid, "excluded_prefixes": excluded, "checkpoints": prepared[0]["checkpoints"],
        "generation_settings": [p["generation_settings"] for p in prepared], "model": prepared[0]["model"],
        "metric_definitions": definitions(), "scoring_source_sha256": sha(Path(__file__))})
    write_json(args.output / "source_draw_hashes.json", hashes)
    scored = {}
    with (args.output / "pair_scores.jsonl").open("w") as handle, ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize,
        initargs=(str(args.dependencies.resolve()), str(args.nltk_data.resolve())),
    ) as pool:
        futures = [pool.submit(score_case, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            for row in future.result():
                identity = row_identity(row)
                if identity in scored:
                    raise ValueError("Duplicate scored pair")
                scored[identity] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 10 == 0 or n == len(futures):
                write_json(args.output / "status.json", {"cases_scored": n, "total_cases": len(futures)})
                print(f"prefix cases {n}/{len(futures)}", flush=True)
    prefixes = prefix_means(specs, scored)
    tasks, benches, curves = aggregate_prefixes(prefixes)
    checked = validate_old_prefix_prf(tasks, paired)
    with (args.output / "prefix_means.jsonl").open("w") as handle:
        for row in prefixes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.output / "aligned_prefix_curves.csv", curves)
    write_csv(args.output / "aligned_prefix_checkpoint_pairs.csv", benches)
    write_csv(args.output / "aligned_prefix_tasks.csv", tasks)
    write_json(args.output / "summary.json", {
        "protocol": "aligned_prefix_regenerated_contexts_t0_t1_cp10_v1", "analysis_status": "complete",
        "new_model_calls": 0, "model": prepared[0]["model"], "prefix_pair_mode_scores": len(scored),
        "prior_prefix_prf_values_verified": checked, "cohort_validity": {
            "valid_prefixes": {b: {"prefixes": sum(c["benchmark"] == b for c in valid),
                                    "tasks": len({c["task"] for c in valid if c["benchmark"] == b})}
                               for b in prepared[0]["checkpoints"]}},
        "aggregation": "Score each prefix/draw pair separately; mean independent repeat pairs at the prefix; "
                       "equal prefix mean within task; equal task mean within checkpoint pair; equal CP-pair "
                       "mean within distance. No pooling TP/FP/FN across prefixes and no missing-as-zero.",
        "self_controls": "CP0-bank view: CP0 independent self control. All-pairs/reference views: average "
                         "independent earlier/later context controls. Gaps are descriptive and can be negative.",
        "metric_definitions": definitions(), "scoring_source_sha256": sha(Path(__file__)),
        "curves": curves, "checkpoint_pairs": benches, "tasks": tasks})
    print(json.dumps({"prefix_pair_mode_scores": len(scored), "prior_prf_values_verified": checked,
                      "valid_prefixes": len(valid), "excluded_prefixes": len(excluded)}))


if __name__ == "__main__":
    main()
