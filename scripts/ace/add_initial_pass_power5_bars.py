# SPDX-License-Identifier: Apache-2.0
"""Add empirical initial pass^5 beside Initial PCT/TGC, preserving saved scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (ROOT / "outputs/ace_appworld_results_summary_20261123"
                  "/qwen_gemma26b_31b_figures_20261127")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def csv_read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Plot source anchor changed: {old[:80]}")
    return source.replace(old, new)


def plotting_source(source):
    source = replace(source, 'OFFSETS = (0.0, 1.35, 2.50, 3.65, 4.80, 5.95)',
                     'OFFSETS = (0.0, 1.10, 2.55, 3.90, 5.25, 6.60, 7.95)')
    source = replace(source, 'STRIDE = 7.45', 'STRIDE = 9.45')
    source = replace(source, 'COLORS = BASE.COLORS', 'COLORS = {**BASE.COLORS, "pass_power_5": "#b69742"}')
    source = replace(source, 'figsize=(40, 16)', 'figsize=(46, 16)')
    source = replace(source, 'left, right = start - 0.60, start + 6.55',
                     'left, right = start - 0.60, start + 8.55')
    anchor = '                missing = model in GEMMAS and arm == "with_gt"'
    insertion = '''                all_five = baseline["pass_power_5"] * 100
                pass_x = start + OFFSETS[1]
                bar(ax, pass_x, all_five, COLORS["pass_power_5"])
                plotted.append({"metric": metric, "model": model, "mode": mode, "arm": arm,
                                "category": "pass_power_5", "variant": "baseline", "x": pass_x,
                                "mean_percent": all_five, "std_percent": "", "n_training_seeds": ""})
'''
    source = replace(source, anchor, insertion + anchor)
    source = replace(source, 'enumerate(STRATEGIES, start=1)', 'enumerate(STRATEGIES, start=2)')
    source = replace(source, 'ax.axvspan(start + 0.80, right,', 'ax.axvspan(start + 1.95, right,')
    source = replace(source, 'ax.text(start + 3.65, 53,', 'ax.text(start + 5.15, 53,')
    source = replace(source, 'tick_labels.extend((f"Initial\\n{metric.upper()}", "Best", "Random", "Worst",',
                     'tick_labels.extend((f"Initial\\n{metric.upper()}", "Pass$^5$", "Best", "Random", "Worst",')
    source = replace(source, 'ax.text(start + 3.0, -0.18,', 'ax.text(start + 4.0, -0.18,')
    source = replace(source, '(len(MODELS) - 1) * STRIDE + 6.60', '(len(MODELS) - 1) * STRIDE + 8.60')
    source = replace(source, 'handles = [Patch(facecolor=COLORS["initial"], label=f"Initial {metric.upper()}"),',
                     'handles = [Patch(facecolor=COLORS["initial"], label=f"Initial {metric.upper()}"),\n'
                     '               Patch(facecolor=COLORS["pass_power_5"],\n'
                     '                     label="Initial pass$^5$ (all 5 official successes)"),')
    source = replace(source, 'Initial mean and five accumulated selection strategies per model',
                     'Initial mean, initial pass$^5$, and five accumulated selection strategies per model')
    source = replace(source, 'ncol=3,\n               frameon=False', 'ncol=4,\n               frameon=False')
    source = replace(source, 'Initial bars: mean over 50 tasks x 5 runs.',
                     'Initial: mean over 50 tasks x 5 runs. '
                     'pass$^5$: fraction of tasks with all five official successes.')
    source = replace(source, 'assert len(plotted) == 100 and len(lines) == 20',
                     'assert len(plotted) == 120 and len(lines) == 20')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    update = bundle / "updates/initial_pass_power5_20261127"
    if (update / "receipt.json").exists():
        print("Already updated: " + str(update / "receipt.json"))
        return
    backup, staging = update / "before", update / "staging"
    update.mkdir(parents=True, exist_ok=True)
    source_path = bundle / "source_snapshot/plot_accumulated_diagnostic.py"
    if not (update / "backup_manifest.json").exists():
        candidates = [p for p in bundle.iterdir() if p.is_file() and "accumulated" in p.name]
        candidates += list((bundle / "data/accumulated").iterdir())
        candidates += [source_path, bundle / "README.txt", bundle / "verification.json"]
        hashes = {}
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(bundle)
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            hashes[str(relative)] = digest(path)
        write(update / "backup_manifest.json", hashes)
    originals = {str(p.relative_to(bundle)): digest(p) for p in bundle.iterdir()
                 if p.suffix in {".png", ".pdf", ".svg"} and "accumulated" not in p.name}
    source = plotting_source((backup / "source_snapshot/plot_accumulated_diagnostic.py").read_text())
    new_source = update / "plot_accumulated_with_pass_power5.py"
    new_source.write_text(source)
    bars = module("saved_accumulation_with_pass_power5", new_source)
    folder = staging / "data/accumulated"
    folder.mkdir(parents=True, exist_ok=True)
    bars.OUT = folder
    initial_rows = csv_read(backup / "data/accumulated/initial_metrics.csv")
    task_rows = csv_read(backup / "data/accumulated/initial_task_metrics.csv")
    tasks_by_key = {(r["model"], r["mode"], r["task_id"]): r for r in task_rows}
    pins, measures = {}, {}
    for row in initial_rows:
        model, mode = row["model"], row["mode"]
        protocol_path = bars.DATA.MODEL_RUNS[model][mode] / "protocol.json"
        protocol = read(protocol_path)
        path = Path(protocol["source_reflection_run"]) / "stage1_results.json"
        pins[str(protocol_path)], pins[str(path)] = digest(protocol_path), digest(path)
        results = read(path)
        groups = defaultdict(list)
        for result in results:
            if type(result["success"]) is not bool:
                raise ValueError("Official boolean success is required for pass^5")
            groups[result["task_id"]].append(result)
        if len(results) != 250 or len(groups) != 50 or set(groups) != set(protocol["task_orders"]["42"]["train"]):
            raise ValueError("Initial bank must contain the original matched Train50/G5")
        passed_all = 0
        for task, trials in groups.items():
            if len(trials) != 5 or {r["source_repeat"] for r in trials} != set(range(5)):
                raise ValueError("Every task must have exactly five independent initial outcomes")
            success_count = sum(r["success"] for r in trials)
            passed_all += success_count == 5
            baseline = tasks_by_key[model, mode, task]
            if abs(success_count / 5 - float(baseline["initial_mean_tgc"])) > 1e-12:
                raise ValueError("Initial TGC bank differs from the published baseline")
            baseline["initial_pass_power_5"] = int(success_count == 5)
        row["pass_power_5"] = passed_all / 50
        if not row["pass_power_5"] <= float(row["tgc"]) <= float(row["pass_at_5"]) + 1e-12:
            raise ValueError("Initial pass^5, TGC, pass@5 ordering is invalid")
        measures[model, mode] = {"all_five_success_tasks": passed_all, "tasks": 50,
                                 "pass_power_5": passed_all / 50}
    initial = {(r["model"], r["mode"]): {k: v if k in {"model", "mode"} else float(v)
                                         for k, v in r.items()} for r in initial_rows}
    summary_rows = csv_read(backup / "data/accumulated/metrics_mean_std.csv")
    summary = {tuple(r[k] for k in ("model", "mode", "arm", "strategy")): {
        k: float(v) if k.startswith("accumulated_") else v for k, v in r.items()
    } for r in summary_rows}
    csv_write(folder / "initial_metrics.csv", initial_rows)
    csv_write(folder / "initial_task_metrics.csv", task_rows)
    for metric in ("pct", "tgc"):
        bars.draw(initial, summary, metric)
        old = csv_read(backup / "data/accumulated" / f"plotted_bars_{metric}.csv")
        new = csv_read(folder / f"plotted_bars_{metric}.csv")
        index = {tuple(r[k] for k in ("model", "mode", "arm", "category")): r for r in new}
        for original in old:
            key = tuple(original[k] for k in ("model", "mode", "arm", "category"))
            for field in ("mean_percent", "std_percent", "n_training_seeds"):
                if original[field] != index[key][field]:
                    raise ValueError("An existing bar's score or uncertainty changed")
        added = [r for r in new if r["category"] == "pass_power_5"]
        if len(added) != 20 or any(abs(float(r["mean_percent"]) -
                                     100 * measures[r["model"], r["mode"]]["pass_power_5"]) > 1e-12 for r in added):
            raise ValueError("Expected 20 identical-across-GT empirical pass^5 baseline bars")
        old_lines = csv_read(backup / "data/accumulated" / f"reference_lines_{metric}.csv")
        new_lines = csv_read(folder / f"reference_lines_{metric}.csv")
        if any(a["mean_percent"] != b["mean_percent"] for a, b in zip(old_lines, new_lines, strict=True)):
            raise ValueError("An initial reference-line score changed")
    provenance = read(backup / "data/accumulated/provenance.json")
    provenance["initial_pass_power_5"] = {
        "definition": "Fraction of the 50 Train tasks with all five official initial success flags true",
        "not_estimated_from_mean_tgc_power": True, "bars_per_figure": 20, "std": "None; one initial G5 bank",
        "data_sha256": pins,
    }
    provenance["bars_per_figure"] = 120
    write(folder / "provenance.json", provenance)
    published = {}
    for path in folder.iterdir():
        targets = [bundle / "data/accumulated" / path.name]
        if path.suffix in {".png", ".pdf", ".svg"}:
            targets.append(bundle / path.name)
        for target in targets:
            shutil.copy2(path, target)
            published[str(target.relative_to(bundle))] = digest(target)
    snapshot = bundle / "source_snapshot/plot_accumulated_with_pass_power5.py"
    shutil.copy2(new_source, snapshot)
    shutil.copy2(Path(__file__), bundle / "source_snapshot" / Path(__file__).name)
    if any(digest(bundle / relative) != sha for relative, sha in originals.items()):
        raise ValueError("An unrelated figure changed")
    receipt = {"updated_at": datetime.now(UTC).isoformat(), "published_sha256": published,
               "pass_power_5": [{"model": key[0], "mode": key[1], **value} for key, value in measures.items()],
               "initial_bank_source_sha256": pins, "script_sha256": digest(Path(__file__)),
               "plot_source_sha256": digest(snapshot), "backup": str(backup),
               "existing_bar_values_and_std_unchanged": True, "existing_reference_line_values_unchanged": True,
               "unrelated_figure_files_unchanged": len(originals), "new_inference_calls": 0}
    with (bundle / "README.txt").open("a") as stream:
        stream.write("\nAdded initial pass^5 beside Initial PCT/TGC in both accumulated bar charts. "
                     "pass^5 = Train tasks with all five official initial successes / 50 (not mean TGC raised to 5). "
                     "Initial pass@5 / max-PCT reference lines and all existing score/SD values retained. "
                     "Source: source_snapshot/plot_accumulated_with_pass_power5.py. "
                     "Receipt/backup: updates/initial_pass_power5_20261127/.\n")
    verification = read(bundle / "verification.json")
    verification["initial_pass_power_5_update"] = {k: v for k, v in receipt.items()
                                                  if k not in {"published_sha256", "initial_bank_source_sha256"}}
    write(bundle / "verification.json", verification)
    write(update / "receipt.json", receipt)
    print(json.dumps({"bundle": str(bundle), "pass_power_5": receipt["pass_power_5"], "verified": True}, indent=2))


if __name__ == "__main__":
    main()
