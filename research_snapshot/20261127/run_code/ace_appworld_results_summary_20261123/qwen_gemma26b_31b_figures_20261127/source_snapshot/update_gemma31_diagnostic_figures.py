# SPDX-License-Identifier: Apache-2.0
"""Update the saved five-model figure bundle with all 52 selected Gemma31 reruns.

Retain all unselected original outcomes, including rerun regressions. Back up the
published bundle first. Candidate distributions and overlaps remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/ace_appworld_results_summary_20261123"
BUNDLE = SUMMARY / "qwen_gemma26b_31b_figures_20261127"
DIAGNOSTIC = ROOT / "outputs/ace_appworld_gemma_format_repair_diagnostic_20261127"
FIELDS = ("pct", "tgc")
BAR_NOTE = (
    "Gemma31 Thinking: 52 prior-format-error cases replaced by regex-fallback reruns; other outcomes retained. "
    "Selected-case diagnostic, not an unbiased full rerun. Gemma w/ GT unmeasured."
)
SCATTER_NOTE = (
    "Gemma31 Thinking includes 52 selected-case reruns; other outcomes retained. "
    "Selection bias applies; correlations are descriptive."
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact(row):
    if type(row["success"]) is not bool:
        raise ValueError("TGC must be a recorded official boolean")
    if row.get("budget_failure"):
        if row["pct"] != 0 or row["success"]:
            raise ValueError("Invalid terminal budget result")
        return Fraction(0)
    passed, total = row["passed_tests"], row["total_tests"]
    if not 0 <= passed <= total or total <= 0 or abs(row["pct"] - passed / total) > 1e-12:
        raise ValueError("Invalid official PCT result")
    return Fraction(passed, total)


def row_key(mode, row):
    return mode, row["condition_id"], row["task_id"]


def load_overlay(diagnostic, cohort="31b"):
    expected = {"31b": 52, "26b": 115}[cohort]
    status = read(diagnostic / cohort / "status.json")
    queue = read(diagnostic / cohort / "queue.json")
    if status["state"] != "completed" or status["completed"] != expected or len(queue) != expected:
        raise RuntimeError(f"{cohort} completion barrier: all {expected} selected reruns must be valid and saved")
    overlay, pins = {}, {}
    for item in queue:
        path = Path(item["root"]) / "stage3" / item["job"]["id"] / "result.json"
        row, old = read(path), read(item["baseline_result_path"])
        if any(row.get(k) != v for k, v in item["job"].items()):
            raise ValueError("Rerun job metadata differs")
        for key in ("initial_context_sha256", "appworld_policy_sha256"):
            if row[key] != old[key]:
                raise ValueError(f"Rerun changed {key}")
        exact(row)
        key = row_key(item["sampler"], row)
        modes = {"thinking"} if cohort == "31b" else {"thinking", "nonthinking"}
        if key in overlay or key[0] not in modes or row["arm"] != "without_gt":
            raise ValueError("Unexpected or duplicated diagnostic case")
        overlay[key] = row
        pins[str(path)] = digest(path)
    return overlay, pins


def calculate(diagnostic, baseline_seed_rows, cohort="31b"):
    overlay, pins = load_overlay(diagnostic, cohort)
    expected = {"31b": 52, "26b": 115}[cohort]
    model_key = {"31b": "gemma31", "26b": "gemma26"}[cohort]
    original = Path(next(m["original"] for m in read(diagnostic / "manifest.json")["models"]
                         if m["name"] == cohort))
    queue_path = original / "control/evaluation_queue.json"
    pins[str(queue_path)] = digest(queue_path)
    queue = read(queue_path)
    if len(queue) != 1500:
        raise ValueError("Expected the original 30 x 50 final-Train outcomes")
    groups, replaced, comparisons = defaultdict(lambda: {"old": [], "new": []}), set(), []
    for item in queue:
        path = Path(item["root"]) / "stage3" / item["job"]["id"] / "result.json"
        old = read(path)
        pins[str(path)] = digest(path)
        key = row_key(item["sampler"], old)
        new = overlay.get(key, old)
        strategy = old["condition_id"].removeprefix(f"s{old['seed']}_without_gt_")
        group = groups[item["sampler"], old["seed"], strategy]
        group["old"].append(old)
        group["new"].append(new)
        if key in overlay:
            replaced.add(key)
            comparisons.append({"mode": key[0], "condition_id": key[1], "task_id": key[2],
                                "original_pct": float(exact(old)), "rerun_pct": float(exact(new)),
                                "original_tgc": old["success"], "rerun_tgc": new["success"]})
    if len(replaced) != expected or len(groups) != 30:
        raise ValueError(f"Diagnostic did not join all and only the selected {expected} cases")
    indexed = {(r["mode"], int(r["seed"]), r["strategy"]): r for r in baseline_seed_rows
               if r["model"] == model_key and r["arm"] == "without_gt"}
    new_seed, changes = {}, []
    for key, group in groups.items():
        if len(group["old"]) != 50 or len({r["task_id"] for r in group["new"]}) != 50:
            raise ValueError("A final Context must retain all 50 Train tasks")
        scores = {}
        for variant in ("old", "new"):
            rows = group[variant]
            scores[variant] = {"pct": float(sum((exact(r) for r in rows), Fraction()) / 50),
                               "tgc": sum(r["success"] for r in rows) / 50}
        for metric in FIELDS:
            if abs(scores["old"][metric] - float(indexed[key][f"accumulated_{metric}"])) > 1e-12:
                raise ValueError(f"Original baseline CSV is not reproduced: {key}/{metric}")
        new_seed[key] = scores["new"]
        changes.append({"mode": key[0], "seed": key[1], "strategy": key[2],
                        **{f"{v}_{m}": scores[v][m] for v in ("old", "new") for m in FIELDS}})
    return new_seed, changes, comparisons, pins


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def annotate_source(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Saved plotting source changed; annotation anchor is not unique")
    return source.replace(old, new)


def render_bars(staging, by_seed, initial_rows, summary_rows):
    summary = {}
    for row in summary_rows:
        typed = {k: float(v) if k.startswith("accumulated_") else v for k, v in row.items()}
        group = [r for r in by_seed if all(r[k] == row[k] for k in ("model", "mode", "arm", "strategy"))]
        if len(group) != 3:
            raise ValueError("Seed group is incomplete")
        for metric in FIELDS:
            values = [float(r[f"accumulated_{metric}"]) for r in group]
            typed[f"accumulated_{metric}_mean"] = mean(values)
            typed[f"accumulated_{metric}_std"] = stdev(values)
        summary[tuple(row[k] for k in ("model", "mode", "arm", "strategy"))] = typed
    folder = staging / "data/accumulated"
    csv_write(folder / "metrics_by_seed.csv", by_seed)
    csv_write(folder / "metrics_mean_std.csv", list(summary.values()))
    initial = {(r["model"], r["mode"]): {
        k: float(v) if k not in {"model", "mode"} else v for k, v in r.items()
    } for r in initial_rows}
    source = (staging / "source_snapshot/plot_accumulated.py").read_text()
    old_note = ('"Both Gemma models have w/o GT results only; '
                'their shared initial baselines are shown in w/ GT, "\n'
                '             "with unrun accumulated conditions left empty. adv. = advantage."')
    source = annotate_source(source, old_note, repr(BAR_NOTE))
    path = staging / "source_snapshot/plot_accumulated_diagnostic.py"
    path.write_text(source)
    bars = import_module("gemma31_diagnostic_bars", path)
    bars.OUT = folder
    for metric in FIELDS:
        bars.draw(initial, summary, metric)
    for file in folder.iterdir():
        if file.suffix in {".png", ".pdf", ".svg"}:
            shutil.copy2(file, staging / file.name)
    return summary


def render_scatters(staging, updated_models=()):
    source = (ROOT / "scripts/ace/extend_qwen_gemma31_scatter_and_gain.py").read_text()
    source = annotate_source(source, "ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path({str(ROOT)!r})")
    source = annotate_source(source,
                             '"Descriptive correlations: two modes share each model; "\n'
                             '                 "these are not ten independent models."', repr(SCATTER_NOTE))
    if updated_models:
        source = annotate_source(source, '    for old in read_csv(old_csv):\n',
                                 '    for old in read_csv(old_csv):\n'
                                 f'        if old["model"] in {tuple(updated_models)!r}:\n'
                                 '            continue  # Updated cohorts are checked against new outcomes.\n')
    path = staging / "source_snapshot/scatter_diagnostic.py"
    path.write_text(source)
    plots = import_module("gemma31_diagnostic_scatter", path)
    plots.OUT = staging
    plots.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                              "svg.fonttype": "none", "pdf.fonttype": 42})
    reference = plots.module("gemma31_parent_scatter_reference", plots.OLD / "plot_scatter.py")
    pct_x = {(r["model"], r["mode"], r["metric"]): float(r["overlap_of_random_percent"])
             for r in csv_read(staging / "data/selected_reflection/overlap_and_means.csv")}
    tgc_x = {(r["model"], r["mode"], r["metric"]): float(r["overlap_of_random_percent"])
             for r in csv_read(staging / "data/overlap_scatter/tgc_overlap_and_means.csv")}
    pct_y, tgc_y = plots.gaps("pct"), plots.gaps("tgc")
    labels = {**reference.LABELS, "gemma31": "Gemma4-31B"}
    families = [
        (pct_x, pct_y, "PCT", "PCT", "pct_overlap_vs_accumulation_pct_gap", plots.OLD / "scatter_points.csv"),
        (pct_x, tgc_y, "PCT", "TGC", "pct_overlap_vs_accumulation_tgc_gap", plots.OLD / "tgc/scatter_points.csv"),
        (tgc_x, tgc_y, "TGC", "TGC", "tgc_overlap_vs_accumulation_tgc_gap",
         plots.OLD / "tgc_reward_overlap/scatter_points.csv"),
    ]
    correlations = {}
    for overlaps, gaps, x_reward, y_metric, name, old in families:
        correlations[name] = plots.scatter(reference, labels, overlaps, gaps, x_reward, y_metric, name, old)
    return correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC)
    args = parser.parse_args()
    bundle, diagnostic = args.bundle.resolve(), args.diagnostic.resolve()
    update = bundle / "updates/gemma31_format_repair_20261127"
    if (update / "receipt.json").exists():
        print("Already published: " + str(update / "receipt.json"), flush=True)
        return
    update.mkdir(parents=True, exist_ok=True)
    backup, staging = update / "before", update / "staging"
    if not (update / "backup_manifest.json").exists():
        hashes = {}
        for path in bundle.iterdir():
            if path.name == "updates":
                continue
            destination = backup / path.name
            if path.is_dir():
                shutil.copytree(path, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        for path in backup.rglob("*"):
            if path.is_file():
                hashes[str(path.relative_to(backup))] = digest(path)
        write(update / "backup_manifest.json", hashes)
    shutil.copytree(backup, staging, dirs_exist_ok=True)
    old_seeds = csv_read(backup / "data/accumulated/metrics_by_seed.csv")
    new_seed, changes, comparisons, pins = calculate(diagnostic, old_seeds)
    by_seed = [dict(row) for row in old_seeds]
    for row in by_seed:
        if row["model"] == "gemma31" and row["arm"] == "without_gt":
            scores = new_seed[row["mode"], int(row["seed"]), row["strategy"]]
            for metric in FIELDS:
                row[f"accumulated_{metric}"] = scores[metric]
    render_bars(staging, by_seed, csv_read(backup / "data/accumulated/initial_metrics.csv"),
                csv_read(backup / "data/accumulated/metrics_mean_std.csv"))
    correlations = render_scatters(staging)
    # Check every non-Gemma31-Thinking point and every overlap is unchanged.
    for path in (staging / "data/overlap_scatter").glob("*_points.csv"):
        old_rows, new_rows = csv_read(backup / path.relative_to(staging)), csv_read(path)
        for old, new in zip(old_rows, new_rows, strict=True):
            for field in ("model", "mode", "arm", "overlap_metric", "overlap_percent"):
                if old[field] != new[field]:
                    raise ValueError(f"Unexpected point/overlap change: {field}")
            if old["model"] != "gemma31" or old["mode"] != "thinking":
                for field in ("gap_mean_pp", "gap_std_pp"):
                    if abs(float(old[field]) - float(new[field])) > 1e-12:
                        raise ValueError("Unrelated scatter point changed")
    unchanged = [path for path in backup.iterdir() if path.suffix in {".png", ".pdf", ".svg"}
                 and "accumulated" not in path.name and "overlap_vs" not in path.name]
    if not all(digest(path) == digest(staging / path.name) for path in unchanged):
        raise ValueError("Candidate violin files changed unexpectedly")
    receipt = {"generated_at": datetime.now(UTC).isoformat(), "variant": "gemma31_selected52_regex_fallback_rerun",
               "original_final_train_rollouts": 1500, "replaced_rollouts": 52, "original_rollouts_retained": 1448,
               "selection": "All 52 saved format-error cases, including successes and rerun regressions",
               "mean_std": "Mean and sample SD across seed42,43,44 Train50 scores",
               "candidate_violins_and_overlap": "Unchanged; candidate evaluations and selection were not rerun",
               "limitations": BAR_NOTE + " Changes also include stochastic rerun variation, not a pure parser effect.",
               "backup": str(backup), "source_result_sha256": pins, "script_sha256": digest(Path(__file__)),
               "correlations_descriptive": correlations}
    write(update / "context_scores_before_after.json", changes)
    write(update / "selected_cases_before_after.json", comparisons)
    shutil.copy2(Path(__file__), staging / "source_snapshot" / Path(__file__).name)
    with (staging / "README.txt").open("a") as stream:
        stream.write("\nUPDATED with Gemma31 selected52 format-repair reruns. " + receipt["limitations"] +
                     "\nOriginal bundle backup: " + str(backup) + "\n" +
                     "Use update_gemma31_diagnostic_figures.py for this version; original build scripts restore "
                     "the uncorrected baseline. See updates/gemma31_format_repair_20261127/receipt.json.\n")
    for relative in ("verification.json", "scatter_and_gain_verification.json", "data/accumulated/provenance.json"):
        record = read(staging / relative)
        record["diagnostic_update"] = {key: value for key, value in receipt.items()
                                       if key != "source_result_sha256"}
        write(staging / relative, record)
    # Publish only after all figures and numerical validations have succeeded.
    published = {}
    for path in staging.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(staging)
        if (backup / relative).exists() and digest(backup / relative) == digest(path):
            continue
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        published[str(relative)] = digest(target)
    receipt["published_sha256"] = published
    receipt["unchanged_candidate_figure_files"] = len(unchanged)
    write(update / "receipt.json", receipt)
    print(json.dumps({"bundle": str(bundle), "published_files": len(published), "replacement_cases": 52,
                      "backup": str(backup), "verified": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
