# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Compare saved T0/T1 reconstructions on shared tasks AND independent sequence IDs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.premise.analyze_regenerated_trajectories import aggregate, concatenate  # noqa: E402
from scripts.premise.audit_group_relative_f1 import (  # noqa: E402
    action_prf,
    canonical,
    content_signature,
    sha,
    write_json,
)

EXTRAS = ("content_f1", "ordered_action_similarity", "content_ordered_action_similarity",
          "prefix_exact_fraction", "content_prefix_exact_fraction")
VIEWS = {"distances": "pairwise", "cp0_reference": "cp0_reference",
         "cp0_complete_sequences": "cp0_complete_sequences"}


def row_key(row):
    return tuple(row[k] for k in ("benchmark", "task", "kind", "mode", "earlier", "later"))


def pair_key(row):
    return row["reference_repeat"], row["prediction_repeat"]


def shared_rows(zero, one):
    """Never compare different task cohorts or different available sequence draws.

    Strict views retain tasks valid at every CP/repeat in BOTH runs. The secondary
    view additionally intersects complete CP0 and candidate repeat IDs, retaining
    a constant task set over the entire checkpoint grid. No partial filling.
    """
    indexes = [{row_key(r): r for r in s["task_comparisons"]} for s in (zero, one)]
    shared_keys = set(indexes[0]) & set(indexes[1])
    accepted = set()
    for key in shared_keys:
        benchmark, task, kind, mode, _, _ = key
        keys = {k for k in shared_keys if k[:4] == (benchmark, task, kind, mode)}
        all_keys = {k for k in indexes[0] if k[:4] == (benchmark, task, kind, mode)}
        all_keys |= {k for k in indexes[1] if k[:4] == (benchmark, task, kind, mode)}
        if keys != all_keys:
            continue
        if kind == "cp0_complete_sequences":
            common = [set(map(pair_key, indexes[0][k]["repeats"])) &
                      set(map(pair_key, indexes[1][k]["repeats"])) for k in keys]
            if any(not pairs for pairs in common):
                continue
        accepted.update(keys)
    paired = [[], []]
    for key in sorted(accepted):
        ids = set(map(pair_key, indexes[0][key]["repeats"])) & set(map(pair_key, indexes[1][key]["repeats"]))
        for arm, index in zip(paired, indexes, strict=True):
            row = copy.deepcopy(index[key])
            row["repeats"] = [r for r in row["repeats"] if pair_key(r) in ids]
            score_keys = set(row["repeats"][0]) - {"reference_repeat", "prediction_repeat"}
            row.update({k: fmean(r[k] for r in row["repeats"]) for k in score_keys})
            row["sample_pairs"] = len(row["repeats"])
            arm.append(row)
    return paired


def ordered_similarity(predicted, reference, mode="strict"):
    """1 - unit-cost Levenshtein distance / max action count; no transpositions."""
    atom = canonical if mode == "strict" else lambda a: a["name"]
    a, b = [atom(v) for v in predicted], [atom(v) for v in reference]
    if not a and not b:
        return 1.0
    prior = list(range(len(b) + 1))
    for i, value in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, prior[j] + 1, prior[j - 1] + (value != other)))
        prior = current
    return 1 - prior[-1] / max(len(a), len(b))


def diagnostics(predicted_steps, reference_steps, benchmark, mode):
    p, r = concatenate(predicted_steps), concatenate(reference_steps)
    cp, cr = content_signature(p, benchmark), content_signature(r, benchmark)
    normalize = (lambda actions: actions) if mode == "strict" else (lambda actions: [a["name"] for a in actions])
    return {"content_f1": action_prf(cp, cr, mode)["f1"],
            "ordered_action_similarity": ordered_similarity(p, r, mode),
            "content_ordered_action_similarity": ordered_similarity(cp, cr, mode),
            "prefix_exact_fraction": fmean(canonical(normalize(a)) == canonical(normalize(b))
                                           for a, b in zip(predicted_steps, reference_steps, strict=True)),
            "content_prefix_exact_fraction": fmean(
                canonical(normalize(content_signature(a, benchmark))) ==
                canonical(normalize(content_signature(b, benchmark)))
                for a, b in zip(predicted_steps, reference_steps, strict=True))}


def add_diagnostics(rows, prepared, draws):
    index = {(r["case"], r["context"], r["repeat"]): r for r in draws}
    tasks = {(t["benchmark"], t["task"]): t for t in prepared["trajectories"]}
    checked = 0
    for row in rows:
        trajectory = tasks[(row["benchmark"], row["task"])]

        def sequence(point, repeat, steps=trajectory["steps"]):
            records = [index[(s["case"], f"cp{point:03d}", repeat)] for s in steps]
            if any(r["status"] != "completed" for r in records):
                raise ValueError("A matched sequence contains an unsuccessful prefix")
            return [r["actions"] for r in records]

        for pair in row["repeats"]:
            ref_id = -1 if row["kind"] == "cp0_reference" else pair["reference_repeat"]
            reference = sequence(row["earlier"], ref_id)
            prediction = sequence(row["later"], pair["prediction_repeat"])
            actual = action_prf(concatenate(prediction), concatenate(reference), row["mode"])
            if any(abs(actual[k] - pair[k]) > 1e-12 for k in actual):
                raise ValueError("Saved pair PRF disagrees with raw regenerated actions")
            pair.update(diagnostics(prediction, reference, row["benchmark"], row["mode"]))
            checked += 1
        row.update({k: fmean(p[k] for p in row["repeats"]) for k in EXTRAS})
    return checked


def matched_summary(source, rows, prepared):
    coverage = {}
    for benchmark, old in source["coverage"].items():
        tasks = sorted({r["task"] for r in rows if r["benchmark"] == benchmark and r["kind"] == "pairwise"})
        coverage[benchmark] = {**old, "included_tasks": tasks, "complete_tasks": len(tasks),
                               "complete_prefixes": sum(len(t["steps"]) for t in prepared["trajectories"]
                                                        if t["benchmark"] == benchmark and t["task"] in tasks),
                               "excluded_tasks": [{"task": t["task"], "reason": "Not complete at both temperatures"}
                                                  for t in prepared["trajectories"]
                                                  if t["benchmark"] == benchmark and t["task"] not in tasks]}
    summary = aggregate(rows, coverage, prepared)
    summary["complete_sequence_coverage"] = {}
    for benchmark in coverage:
        tasks = sorted({r["task"] for r in rows if r["benchmark"] == benchmark and
                        r["kind"] == "cp0_complete_sequences"})
        summary["complete_sequence_coverage"][benchmark] = {
            "included_tasks": tasks, "complete_tasks": len(tasks),
            "planned_tasks": coverage[benchmark]["planned_tasks"]}
    for view, kind in VIEWS.items():
        for result in summary[view]:
            selected = [r for r in rows if r["kind"] == kind and r["benchmark"] == result["benchmark"] and
                        r["mode"] == result["mode"] and r["distance"] == result["distance"]]
            groups = defaultdict(list)
            for row in selected:
                groups[(row["earlier"], row["later"])].append(row)
            result.update({k: fmean(fmean(r[k] for r in group) for group in groups.values()) for k in EXTRAS})
    return summary


def plot_comparison(summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    groups = {"prf": ("precision", "recall", "f1"), "bleu_meteor": ("bleu1", "bleu4", "meteor"),
              "action_diagnostics": ("content_f1", "ordered_action_similarity", "prefix_exact_fraction")}
    titles = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "Terminal-Bench"}
    for mode in ("strict", "loose"):
        for group, metrics in groups.items():
            fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5))
            for i, (benchmark, title) in enumerate(titles.items()):
                n = summaries[0]["complete_sequence_coverage"][benchmark]["complete_tasks"]
                for j, metric in enumerate(metrics):
                    ax = axes[i, j]
                    for summary, color in zip(summaries, ("#235789", "#c55a11"), strict=True):
                        rows = sorted((r for r in summary["cp0_complete_sequences"] if
                                       r["benchmark"] == benchmark and r["mode"] == mode), key=lambda r: r["distance"])
                        ax.plot([r["distance"] for r in rows], [r[metric] for r in rows], "o-", color=color,
                                label=f"T={summary['temperature']:g}", markersize=4)
                        if rows:
                            ax.axhline(rows[0][metric], color=color, linestyle=":", alpha=.5)
                    ax.set(title=f"{title} (shared tasks={n}) | {metric}", xlabel="Update distance",
                           ylim=(-.03, 1.03))
                    ax.set_xticks(range(0, 31 if benchmark == "terminalbench2" else 51, 10))
                    ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
            fig.suptitle(f"{mode.capitalize()} regenerated whole trajectories | shared tasks and sequence IDs",
                         fontsize=14)
            fig.text(.5, .02, "Original history prefixes fixed; both arms regenerate. "
                     "Dotted lines: independent CP0 self agreement. No confidence intervals.",
                     ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .055, 1, .9))
            name = f"paired_t0_t1_{mode}_{group}"
            for suffix in ("png", "pdf"):
                path = output / f"{name}.{suffix}"
                fig.savefig(path, dpi=200)
                files.append(path.name)
            plt.close(fig)
    return files


def plot_self_distribution(summaries, output):
    import matplotlib.pyplot as plt

    values = []
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for i, benchmark in enumerate(("bfcl", "appworld", "terminalbench2")):
        tasks = summaries[0]["complete_sequence_coverage"][benchmark]["included_tasks"]
        for j, mode in enumerate(("strict", "loose")):
            ax = axes[i, j]
            for arm, summary in enumerate(summaries):
                for t, task in enumerate(tasks):
                    row = next(r for r in summary["task_comparisons"] if r["benchmark"] == benchmark and
                               r["kind"] == "cp0_complete_sequences" and r["mode"] == mode and
                               r["task"] == task and r["distance"] == 0)
                    pairs = [p for p in row["repeats"] if p["reference_repeat"] < p["prediction_repeat"]]
                    for k, pair in enumerate(pairs):
                        x = arm + .055 * (t - (len(tasks) - 1) / 2) + .01 * (k - 1)
                        ax.scatter(x, pair["f1"], s=30, color=f"C{t}", alpha=.75)
                        values.append({"benchmark": benchmark, "task": task, "mode": mode,
                                       "temperature": summary["temperature"], **pair})
            ax.set(xticks=[0, 1], xticklabels=["T=0", "T=1"], ylim=(-.03, 1.03), xlim=(-.4, 1.4),
                   title=f"{benchmark} | {mode} | {len(tasks)} shared tasks", ylabel="CP0 independent-pair F1")
            ax.grid(alpha=.2)
    fig.suptitle("Same-context F1 distributions on identical valid task/repeat IDs", fontsize=14)
    fig.text(.5, .02, "One point per unordered CP0 repeat pair. Colors identify tasks. "
             "Pairs share generated sequences; points are not independent samples.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .95))
    files = []
    for suffix in ("png", "pdf"):
        path = output / f"paired_cp0_f1_distribution.{suffix}"
        fig.savefig(path, dpi=200)
        files.append(path.name)
    plt.close(fig)
    with (output / "paired_cp0_pair_distribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-zero", type=Path, required=True)
    parser.add_argument("--run-one", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [args.run_zero.resolve(), args.run_one.resolve()]
    prepared = [json.loads((p / "prepared.json").read_text()) for p in runs]
    sources = [json.loads((p / "analysis/summary.json").read_text()) for p in runs]
    if [p["temperature"] for p in prepared] != [0, 1]:
        raise ValueError("Expected evaluation temperatures 0 and 1")
    for key in ("cases", "trajectories", "policies", "checkpoints", "repeats", "model"):
        if prepared[0][key] != prepared[1][key]:
            raise ValueError(f"The temperature cohorts differ in {key}")
    for key in ("generation_settings", "model_call_settings"):
        settings = [{k: v for k, v in p[key].items() if k != "temperature"} for p in prepared]
        if settings[0] != settings[1]:
            raise ValueError("A non-temperature model-call setting differs")
    if any(s["analysis_status"] != "complete" for s in sources):
        raise ValueError("Complete collection and analysis before paired comparison")
    rows = shared_rows(*sources)
    summaries, checks = [], []
    for p, source, selected, run in zip(prepared, sources, rows, runs, strict=True):
        draws = [json.loads(f.read_text()) for f in (run / "draws").glob("*.json")]
        checks.append(add_diagnostics(selected, p, draws))
        summaries.append(matched_summary(source, selected, p))
    if checks[0] != checks[1]:
        raise ValueError("Unmatched pair counts")
    args.output.mkdir(exist_ok=True, parents=True)
    flat = [{"temperature": s["temperature"], "view": view, **r} for s in summaries for view in VIEWS
            for r in s[view]]
    fields = list(dict.fromkeys(k for r in flat for k in r))
    with (args.output / "paired_temperature_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    result = {"source_runs": list(map(str, runs)),
              "source_summary_sha256": [sha(p / "analysis/summary.json") for p in runs],
              "analysis_source_sha256": sha(Path(__file__)), "verified_raw_prf_pairs_per_temperature": checks[0],
              "matching_rule": "Shared task set over all checkpoint distances, with identical valid reference and "
                               "prediction repeat-ID pairs at both temperatures. Strict all-repeats and secondary "
                               "complete-sequence views retain their distinct validity rules. Errors are never zeros.",
              "diagnostic_definitions": {
                  "content_f1": "Whole-concat action F1 after Python AST formatting normalization for AppWorld "
                                "or removing terminal_command.duration for Terminal; BFCL unchanged. "
                                "Not semantic equivalence.",
                  "ordered_action_similarity": "1 - unit-cost action-atom Levenshtein / max action count; strict uses "
                                               "canonical full actions, loose uses action names; empty/empty=1.",
                  "content_ordered_action_similarity": "Same ordered action similarity after content normalization.",
                  "prefix_exact_fraction": "Fraction of original prefixes whose regenerated ordered action lists "
                                           "are identical under strict/loose representation; diagnostic only.",
                  "content_prefix_exact_fraction": "Same prefix exact fraction after content normalization."},
              "summaries": summaries}
    result["figures"] = plot_comparison(summaries, args.output) + plot_self_distribution(summaries, args.output)
    write_json(args.output / "summary.json", result)
    print(json.dumps({"verified_raw_prf_pairs_per_temperature": checks[0],
                      "strict_shared_tasks": summaries[0]["coverage"],
                      "secondary_shared_tasks": summaries[0]["complete_sequence_coverage"]}))


if __name__ == "__main__":
    main()
