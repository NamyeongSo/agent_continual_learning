# SPDX-License-Identifier: Apache-2.0
"""Extend three saved Qwen/Gemma figure families with a completed Gemma31 cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/ace_appworld_results_summary_20261123"
GAIN = SUMMARY / "reflection_gain_with_oss120b_20261123/qwen_gemma26b_wo_gt_advantage_gain_20261125"
ALL = SUMMARY / "reflection_advantage_violin_20261124/qwen_gemma26b_20261126"
BARS = SUMMARY / "qwen_gemma_accumulated_20261125"
G26 = ROOT / "outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"
G31 = ROOT / "outputs/ace_appworld_gemma4_31b_it_wo_gt_train_only_20261124"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"Plot source anchor changed ({source.count(old)} != {count}): {old[:100]}")
    return source.replace(old, new)


def literal(path):
    return f"Path({str(path)!r})"


def models(source):
    source = replace(source, 'MODELS = ("qwen38", "qwen36", "qwen35", "gemma26")',
                     'MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")')
    return replace(source, '"gemma26": "Gemma4\\n26B-A4B"}',
                   '"gemma26": "Gemma4\\n26B-A4B", "gemma31": "Gemma4\\n31B"}')


def selected_source(output):
    source = (GAIN / "plot_qwen_gemma.py").read_text()
    source = models(source)
    source = replace(source, 'OUT = Path(__file__).resolve().parent', f"OUT = {literal(output)}")
    source = replace(source, 'SOURCE = OUT.parent', f"SOURCE = {literal(GAIN.parent)}")
    source = replace(source, 'ROOT = OUT.parents[3]', f"ROOT = {literal(ROOT)}")
    source = replace(source, 'GEMMA = ROOT / "outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"',
                     f'GEMMAS = {{"gemma26": {literal(G26)}, "gemma31": {literal(G31)}}}')
    source = replace(source, '"gemma26": "++"}', '"gemma26": "++", "gemma31": "xx"}')
    source = replace(source, 'def load_gemma(mode):', 'def load_gemma(model, mode):')
    source = replace(source, 'folder = GEMMA / mode / "reflection"', 'folder = GEMMAS[model] / mode / "reflection"')
    source = replace(source, 'assert protocol["model"] == "openai/google/gemma-4-26B-A4B-it"',
                     'assert protocol["model"] == {"gemma26": "openai/google/gemma-4-26B-A4B-it", '
                     '"gemma31": "openai/google/gemma-4-31B-it"}[model]')
    source = replace(source, 'REF.INITIAL["gemma26", mode], REF.BANKS["gemma26", mode]',
                     'REF.INITIAL[model, mode], REF.BANKS[model, mode]')
    source = replace(source, 'if model == "gemma26":\n                load_gemma(mode)',
                     'if model in GEMMAS:\n                load_gemma(model, mode)')
    source = replace(source, 'read(GEMMA / "advantage"', 'read(GEMMAS[model] / "advantage"')
    source = replace(source, 'if model != "gemma26":', 'if model.startswith("qwen"):')
    source = replace(source, 'len(candidates_out) == 2000', 'len(candidates_out) == 2500')
    source = replace(source, 'figsize=(17.2, 12.4)', 'figsize=(21.5, 12.4)')
    source = replace(source, 'xticks=range(4), xticklabels=ticks, xlim=(-0.65, 3.65)',
                     'xticks=range(len(MODELS)), xticklabels=ticks, xlim=(-0.65, len(MODELS) - 0.35)')
    source = replace(source, '"groups": 16', '"groups": 20')
    source = replace(source, '"candidate_reflections": 2000', '"candidate_reflections": 2500')
    source = replace(source, '"gemma_training_advantages_verified": 100', '"gemma_training_advantages_verified": 200')
    return source.replace('qwen_gemma26b_wo_gt_grpo_advantage_and_pct_gain_4panel',
                          'qwen_gemma26b_31b_wo_gt_grpo_advantage_and_pct_gain_4panel')


def all_source():
    source = (ROOT / "scripts/ace/plot_generated_reflection_advantage_violin.py").read_text()
    source = models(source)
    source = replace(source, 'ROOT = Path(__file__).resolve().parents[2]', f"ROOT = {literal(ROOT)}")
    source = replace(source, 'def collect(original, gemma):', 'def collect(original, gemma, gemma31):')
    start = source.index('    for mode in MODES:\n        folder = gemma')
    end = source.index('    panels = defaultdict(list)', start)
    block = source[start:end].replace('folder = gemma /', 'folder = cohort /')
    block = block.replace('data(gemma / "advantage"', 'data(cohort / "advantage"')
    block = block.replace('"model": "gemma26"', '"model": model')
    source = source[:start] + '    for model, cohort in (("gemma26", gemma), ("gemma31", gemma31)):\n' + \
        ''.join('    ' + line if line.strip() else line for line in block.splitlines(keepends=True)) + source[end:]
    source = replace(source, 'len(panels) == 14 and len(task_groups) == 700 and len(candidates) == 3500',
                     'len(panels) == 16 and len(task_groups) == 800 and len(candidates) == 4000')
    source = replace(source, 'assert model == "gemma26" and arm == "with_gt"',
                     'assert model in ("gemma26", "gemma31") and arm == "with_gt"')
    source = replace(source, 'figsize=(17, 6.5)', 'figsize=(21, 6.5)')
    source = replace(source, 'plt.subplots(2, 2, figsize=(18, 11.5)', 'plt.subplots(2, 2, figsize=(22, 11.5)')
    source = replace(source, '"Gemma26 has w/o GT data only."', '"Both Gemma models have w/o GT data only."')
    source = replace(source, 'parser.add_argument("--gemma", type=Path, required=True)',
                     'parser.add_argument("--gemma", type=Path, required=True)\n'
                     '    parser.add_argument("--gemma31", type=Path, required=True)')
    source = replace(source, 'collect(args.original, args.gemma)', 'collect(args.original, args.gemma, args.gemma31)')
    source = replace(source, 'assert len(old) == 12', 'assert len(old) == 14')
    source = source.replace('reflection_advantage_all_candidates_qwen_gemma26b_',
                            'reflection_advantage_all_candidates_qwen_gemma26b_31b_')
    return source.replace('Gemma26 w/ GT shown as not measured', 'Both Gemma w/ GT cohorts shown as not measured')


def bars_source(output):
    source = (BARS / "plot_qwen_gemma_accumulated.py").read_text()
    source = replace(source, 'OUT = Path(__file__).resolve().parent', f"OUT = {literal(output)}")
    source = replace(source, 'REFERENCE = OUT.parent / "model_strategy_bars_20261124/plot_model_strategy_bars.py"',
                     f'REFERENCE = {literal(SUMMARY / "model_strategy_bars_20261124/plot_model_strategy_bars.py")}')
    start, end = source.index('GEMMA = DATA.BASE'), source.index('MODES = ("thinking", "nonthinking")')
    source = source[:start] + f'''GEMMAS = {{"gemma26": {literal(G26)}, "gemma31": {literal(G31)}}}
for model, cohort in GEMMAS.items():
    DATA.MODEL_RUNS[model] = {{m: cohort / m / "accumulation" for m in ("thinking", "nonthinking")}}
    DATA.ADV_RUNS[model] = cohort / "advantage"
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")
''' + source[end:]
    source = replace(source, '"gemma26": "Gemma4-26B-A4B"}',
                     '"gemma26": "Gemma4-26B-A4B", "gemma31": "Gemma4-31B"}')
    source = source.replace('model == "gemma26" and arm == "with_gt"', 'model in GEMMAS and arm == "with_gt"')
    source = replace(source, 'len(initial) == 8 and len(by_seed) == 210 and len(summary) == 70',
                     'len(initial) == 10 and len(by_seed) == 240 and len(summary) == 80')
    source = replace(source, 'len(task_baselines) == 400', 'len(task_baselines) == 500')
    start, end = source.index('def verify_cached('), source.index('def bar(')
    source = source[:start] + f'''def verify_cached(initial, by_seed, summary):
    with {literal(BARS / "metrics_by_seed.csv")}.open(newline="") as stream:
        old = {{(r["model"], r["mode"], r["arm"], r["strategy"], int(r["seed"])): r
               for r in csv.DictReader(stream)}}
    indexed = {{(r["model"], r["mode"], r["arm"], r["strategy"], r["seed"]): r for r in by_seed}}
    for key, row in old.items():
        for metric in ("pct", "tgc"):
            assert abs(indexed[key][f"accumulated_{{metric}}"] - float(row[f"accumulated_{{metric}}"])) < 1e-12
    for model, cohort in GEMMAS.items():
        for row in DATA.read(cohort / "train_final_by_seed.json"):
            actual = indexed[model, row["mode"], "without_gt", row["strategy"], row["seed"]]
            for metric in ("pct", "tgc"):
                assert abs(actual[f"accumulated_{{metric}}"] - row[metric]) < 1e-12
        for row in DATA.read(cohort / "train_final_mean_std.json"):
            actual = summary[model, row["mode"], "without_gt", row["strategy"]]
            for metric in ("pct", "tgc"):
                for stat in ("mean", "std"):
                    assert abs(actual[f"accumulated_{{metric}}_{{stat}}"] - row[f"{{metric}}_{{stat}}"] ) < 1e-12
    with {literal(BARS / "initial_metrics.csv")}.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for metric in ("pct", "tgc", "pass_at_5", "mean_task_max_pct_at_5"):
                assert abs(initial[row["model"], row["mode"]][metric] - float(row[metric])) < 1e-12


''' + source[end:]
    source = replace(source, 'figsize=(32, 16)', 'figsize=(40, 16)')
    source = replace(source, 'Gemma4-26B-A4B has w/o GT results only;', 'Both Gemma models have w/o GT results only;')
    source = replace(source, 'its shared initial baseline is shown in w/ GT,',
                     'their shared initial baselines are shown in w/ GT,')
    source = replace(source, 'len(plotted) == 86 and len(lines) == 16', 'len(plotted) == 100 and len(lines) == 20')
    source = replace(source, 'for r in plotted) == 70', 'for r in plotted) == 80')
    source = source.replace('qwen_gemma_accumulated_{metric}_4panel', 'qwen_gemma26b_31b_accumulated_{metric}_4panel')
    for old, new in [('"initial_banks": 8', '"initial_banks": 10'),
                     ('"initial_rollouts": 2000', '"initial_rollouts": 2500'),
                     ('"complete_seed_contexts": 210', '"complete_seed_contexts": 240'),
                     ('"train_rollouts": 10500', '"train_rollouts": 12000'),
                     ('"strategy_groups": 70', '"strategy_groups": 80'),
                     ('"bars_per_figure": 86', '"bars_per_figure": 100'),
                     ('"model_local_reference_lines_per_figure": 16', '"model_local_reference_lines_per_figure": 20')]:
        source = replace(source, old, new, count=2 if old == '"bars_per_figure": 86' else 1)
    source = source.replace('Gemma26 w/ GT was not run.', 'Both Gemma w/ GT cohorts were not run.')
    source = source.replace('All 210 seed scores', 'All 240 seed scores')
    source = replace(source, '(OUT.parent / "build_report.py")', literal(SUMMARY / "build_report.py"))
    return source


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUMMARY / "qwen_gemma26b_31b_figures_20261127")
    args = parser.parse_args()
    output = args.output.resolve()
    status = json.loads((G31 / "status.json").read_text())
    receipt = json.loads((G31 / "completed.json").read_text())
    assert status["state"] == "completed" and receipt["contexts"] == 30 and receipt["train_evaluations"] == 1500
    originals = {str(p): digest(p) for directory in (GAIN, ALL, BARS)
                 for p in directory.iterdir() if p.is_file()}
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    folders = {key: output / "data" / key for key in ("selected_reflection", "all_candidates", "accumulated")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    scripts = {"selected_reflection": selected_source(folders["selected_reflection"]),
               "all_candidates": all_source(), "accumulated": bars_source(folders["accumulated"])}
    for name, source in scripts.items():
        path = snapshot / f"plot_{name}.py"
        path.write_text(source)
        compile(source, str(path), "exec")
        command = [sys.executable, str(path)]
        if name == "all_candidates":
            command += ["--original", str(ALL), "--gemma", str(G26), "--gemma31", str(G31),
                        "--output", str(folders[name])]
        with (output / f"build_{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env={**os.environ, "MPLBACKEND": "Agg",
                           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Rendered {name}", flush=True)
    # Confirm all old distributions and overlapping means remain numerically identical.
    with np.load(GAIN / "source_densities.npz") as old, \
            np.load(folders["selected_reflection"] / "source_densities.npz") as new:
        for key in old.files:
            assert np.allclose(old[key], new[key], atol=1e-12, rtol=0), key
        old_density_checks = len(old.files) - 2
    gain_candidates = read_csv(folders["selected_reflection"] / "candidate_scores_and_advantages.csv")
    all_candidates = read_csv(folders["all_candidates"] / "reflection_advantages.csv")
    indexed = {(r["mode"], r["task_id"], r["source_repeat"]): r for r in all_candidates if r["model"] == "gemma31"}
    for row in gain_candidates:
        if row["model"] == "gemma31":
            other = indexed[row["mode"], row["task_id"], row["source_repeat"]]
            assert abs(float(row["advantage"]) - float(other["advantage"])) < 1e-12
    assert len(indexed) == 500
    figures = []
    for key, directory in folders.items():
        for path in sorted(directory.iterdir()):
            if path.suffix not in (".png", ".pdf", ".svg"):
                continue
            if key == "all_candidates" and "wo_gt_2panel" not in path.name:
                continue
            destination = output / path.name
            shutil.copy2(path, destination)
            figures.append(destination.name)
    assert len(figures) == 16
    assert all(digest(Path(p)) == expected for p, expected in originals.items())
    manifest = {"created_at": datetime.now(UTC).isoformat(),
                "models": ["qwen38", "qwen36", "qwen35", "gemma26", "gemma31"],
                "new_inference_calls": 0, "gemma31_completion_receipt": receipt,
                "figures": figures, "old_selected_density_arrays_verified": old_density_checks,
                "gemma31_candidate_advantages_cross_checked": len(indexed),
                "original_files_preserved": originals,
                "original_plot_scripts": {str(p): digest(p) for p in
                    (GAIN / "plot_qwen_gemma.py", ROOT / "scripts/ace/plot_generated_reflection_advantage_violin.py",
                     BARS / "plot_qwen_gemma_accumulated.py")},
                "generated_plot_scripts": {str(p): digest(p) for p in snapshot.glob("*.py")},
                "coordinator_sha256": digest(Path(__file__)),
                "selection_metric": "PCT; official TGC reported separately without changing saved selections",
                "std": "Sample SD across seed-level Train50 means for seeds42,43,44; no task-level error bars",
                "missing_conditions": "Gemma26/31 w/ GT unmeasured, never filled with zero"}
    (output / "verification.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "figures": figures, "verification": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
