# SPDX-License-Identifier: Apache-2.0
"""Render the saved accumulation chart layout with seed-selected independent PCT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (ROOT / "outputs/ace_appworld_results_summary_20261123"
                  "/qwen_gemma26b_31b_figures_20261127")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def collect(bars, initial):
    data, base = bars.DATA, bars.BASE
    records, selected_rows, sources = [], [], {}
    for model in bars.MODELS:
        for mode in bars.MODES:
            run = data.MODEL_RUNS[model][mode]
            protocol, candidates = data.load_reflection(model, mode, run)
            tasks = set(data.INITIAL[model, mode])
            if len(tasks) != 50:
                raise ValueError("Expected all 50 original Train tasks")
            trials = [r for group in data.INITIAL[model, mode].values() for r in group]
            if abs(data.metrics(trials)["pct"] - initial[model, mode]["pct"]) > 1e-12:
                raise ValueError("The independent bank does not match the published initial PCT")
            for arm in bars.ARMS:
                if model in bars.GEMMAS and arm == "with_gt":
                    continue
                for strategy in bars.STRATEGIES:
                    cohort, _ = base.accumulation_source(model, mode, arm, strategy)
                    cp = data.read(cohort / "protocol.json")
                    for key in ("model", "model_settings", "appworld_policy_sha256", "initial_playbook",
                                "playbook_edit_mode", "turn_limit_scoring"):
                        if cp[key] != protocol[key]:
                            raise ValueError(f"Candidate/selection cohort mismatch: {cohort}/{key}")
                    for seed in bars.SEEDS:
                        cid = f"s{seed}_{arm}_{strategy}"
                        selection_path = cohort / "conditions" / cid / "selections.json"
                        selected = data.read(selection_path)
                        sources[str(selection_path)] = digest(selection_path)
                        if [r["candidate"]["task_id"] for r in selected] != cp["task_orders"][str(seed)]["train"]:
                            raise ValueError("Saved selections differ from the pinned training order")
                        if not {r["candidate"]["task_id"] for r in selected} <= tasks:
                            raise ValueError("Selection contains tasks outside the initial Train50")
                        chosen = {}
                        for item in selected:
                            candidate = item["candidate"]
                            key = arm, candidate["task_id"], candidate["source_repeat"]
                            evaluations = candidates[key]
                            if set(evaluations) != {0, 1, 2}:
                                raise ValueError("A selected candidate must have all three official evaluations")
                            score = sum((data.exact_reward(r) for r in evaluations.values()), Fraction()) / 3
                            if score != Fraction(candidate["score_numerator"], candidate["score_denominator"]):
                                raise ValueError("The selected candidate's exact score differs from its saved bank")
                            chosen[candidate["task_id"]] = candidate["source_repeat"]
                        record = data.independent(model, mode, seed, arm, strategy, selected, candidates)
                        records.append({**record, "train_tasks": 50})
                        for task in sorted(tasks):
                            repeat = chosen.get(task)
                            rows = (list(candidates[arm, task, repeat].values()) if task in chosen
                                    else data.INITIAL[model, mode][task])
                            selected_rows.append({"model": model, "mode": mode, "arm": arm,
                                                  "strategy": strategy, "seed": seed, "task_id": task,
                                                  "candidate_source_repeat": repeat if task in chosen else "",
                                                  "skipped_zero_signal_task": task not in chosen,
                                                  "evaluations": len(rows), **data.metrics(rows)})
    if len(records) != 240 or len(selected_rows) != 12000:
        raise ValueError("Expected 80 measured conditions x 3 seeds x 50 Train task scores")
    summary = {}
    for row in records:
        key = tuple(row[k] for k in ("model", "mode", "arm", "strategy"))
        if key in summary:
            continue
        group = [r for r in records if tuple(r[k] for k in ("model", "mode", "arm", "strategy")) == key]
        if len(group) != 3 or {r["seed"] for r in group} != {42, 43, 44}:
            raise ValueError("Every strategy must have all three selection seeds")
        summary[key] = {k: row[k] for k in ("model", "mode", "arm", "strategy", "learned_tasks", "train_tasks")}
        summary[key]["seeds"] = "42,43,44"
        for metric in ("pct", "tgc"):
            values = [r[f"independent_{metric}"] for r in group]
            summary[key][f"independent_{metric}_mean"] = mean(values)
            summary[key][f"independent_{metric}_std"] = stdev(values)
    return records, selected_rows, summary, {**data.SOURCES, **sources}


def render_source(path):
    source = path.read_text()
    # Keep the proven layout and drawing routines, without its accumulation-data collectors/main.
    source = (source[:source.index("\ndef collect():")]
              + source[source.index("\ndef bar("):source.index("\ndef main():")])
    source = source.replace("accumulated", "independent")
    source = source.replace("Accumulated: final Context", "Independent: single-reflection Context")
    source = source.replace("before learning and after Context accumulation",
                            "before learning and after one Reflection update")
    source = source.replace("seed-level Train means (42, 43, 44)", "seed-selected candidate-bank means (42, 43, 44)")
    start, end = source.index("    fig.text(0.5, 0.032,"), source.index("    fig.subplots_adjust(")
    note = ("Selected tasks: saved 3-rollout candidate mean; skipped zero-signal tasks: initial 5-rollout mean. "
            "Selection and plotted scores share the bank. Gemma w/ GT unmeasured.")
    source = source[:start] + f'    fig.text(0.5, 0.032, {note!r},\n' \
        '             ha="center", fontsize=11.5, color="#536176")\n' + source[end:]
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    folder = bundle / "data/independent"
    folder.mkdir(exist_ok=True)
    original_figures = {p.name: digest(p) for p in bundle.iterdir() if p.suffix in {".png", ".pdf", ".svg"}
                        and "_independent_" not in p.name}
    original_source = bundle / "source_snapshot/plot_accumulated_diagnostic.py"
    plot_path = bundle / "source_snapshot/plot_independent.py"
    plot_path.write_text(render_source(original_source))
    bars = module("single_reflection_qwen_gemma_layout", plot_path)
    bars.OUT = folder
    initial_csv = bundle / "data/accumulated/initial_metrics.csv"
    initial = {(r["model"], r["mode"]): {k: v if k in {"model", "mode"} else float(v)
                                         for k, v in r.items()} for r in csv_read(initial_csv)}
    records, task_scores, summary, sources = collect(bars, initial)
    cached_path = bundle.parent / "model_strategy_bars_20261124/metrics_by_seed.csv"
    cached = {(r["model"], r["mode"], r["arm"], r["strategy"], int(r["seed"])): r
              for r in csv_read(cached_path) if r["model"].startswith("qwen")}
    checked = 0
    for row in records:
        if row["model"].startswith("qwen"):
            prior = cached[row["model"], row["mode"], row["arm"], row["strategy"], row["seed"]]
            for metric in ("pct", "tgc"):
                if abs(row[f"independent_{metric}"] - float(prior[f"independent_{metric}"])) > 1e-12:
                    raise ValueError("Previously reported Qwen independent expectation changed")
            checked += 1
    if checked != 180:
        raise ValueError("Expected all 180 Qwen seed-condition means to match previous reporting")
    csv_write(folder / "metrics_by_seed.csv", records)
    csv_write(folder / "metrics_mean_std.csv", list(summary.values()))
    csv_write(folder / "task_scores_by_seed.csv", task_scores)
    shutil.copy2(initial_csv, folder / "initial_metrics.csv")
    bars.draw(initial, summary, "pct")
    original_bars = csv_read(bundle / "data/accumulated/plotted_bars_pct.csv")
    new_bars = csv_read(folder / "plotted_bars_pct.csv")
    for old, new in zip(original_bars, new_bars, strict=True):
        if any(old[k] != new[k] for k in ("model", "mode", "arm", "category", "x")):
            raise ValueError("Model/panel/bar layout changed")
        if old["category"] == "initial" and old["mean_percent"] != new["mean_percent"]:
            raise ValueError("Initial PCT bar changed")
    reference_csv = bundle / "data/accumulated/reference_lines_pct.csv"
    if digest(reference_csv) != digest(folder / "reference_lines_pct.csv"):
        raise ValueError("The original five-rollout maximum-PCT reference lines changed")
    copied = []
    for file in folder.iterdir():
        if file.suffix in {".png", ".pdf", ".svg"}:
            destination = bundle / file.name
            shutil.copy2(file, destination)
            copied.append(destination)
    if any(digest(bundle / name) != sha for name, sha in original_figures.items()):
        raise ValueError("An existing figure was modified")
    shutil.copy2(__file__, bundle / "source_snapshot" / Path(__file__).name)
    provenance = {"generated_at": datetime.now(UTC).isoformat(), "figures": [str(p) for p in copied],
                  "measure": "Saved seed-selected single-reflection candidate-bank PCT; original selection fixed",
                  "initial_and_reference_lines": "Exact values and positions copied from the accumulated PCT figure",
                  "reference_line_definition": "Task mean of the maximum PCT over the task's initial five runs",
                  "official_tgc_pass_at_5": "Retained in initial_metrics.csv; distinct from maximum-PCT reference line",
                  "std": "Sample SD across seed42/43/44 means over all 50 Train tasks, not SD of the 3 repeat scores",
                  "skipped_zero_signal_tasks": "Use the task's original initial-five mean in independent expectation",
                  "gemma_gt": "Both Gemma models have no w/ GT strategy scores; keep them unmeasured",
                  "selection_bias": "Candidate selection and plotted scores use the same saved three-repeat bank",
                  "qwen_seed_means_cross_checked": checked, "new_model_calls": 0,
                  "source_sha256": {**sources, str(initial_csv): digest(initial_csv),
                                    str(reference_csv): digest(reference_csv), str(cached_path): digest(cached_path)},
                  "script_sha256": digest(Path(__file__)), "plot_source_sha256": digest(plot_path),
                  "original_figure_sha256": original_figures}
    (folder / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": provenance["figures"], "seed_conditions": len(records),
                      "existing_initial_and_reference_lines_preserved": True, "verified": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
