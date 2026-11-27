# SPDX-License-Identifier: Apache-2.0
"""Extend the existing saved single-Reflection gain plots with completed GPT-OSS-120B banks."""

import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "build_report.py"
spec = importlib.util.spec_from_file_location("ace_saved_results_report", SOURCE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
report.OUT = OUT


def main():
    for model, modes in report.MODEL_RUNS.items():
        for mode, run in modes.items():
            report.load_reflection(model, mode, run)
            assert len(report.INITIAL[model, mode]) == 50
            assert len(report.BANKS[model, mode]) == 100
    task_sets = {tuple(sorted(tasks)) for tasks in report.INITIAL.values()}
    assert len(task_sets) == 1
    summaries = []
    for filtered, advantage, name in [
        (False, False, "independent_pct_gain_all_tasks_split_violin_n_over_50_with_random"),
        (True, False, "independent_pct_gain_nonzero_advantage_split_violin_n_over_50_with_random"),
        (False, True, "independent_advantage_sign_pct_gain_split_violin_n_over_50_with_random"),
    ]:
        for model, modes in report.MODEL_RUNS.items():
            for mode in modes:
                for arm in report.ARMS:
                    points = report.gain_points(model, mode, arm, filtered=filtered, advantage=advantage)
                    for strategy, distribution in points.items():
                        values = np.array([v for v, _ in distribution])
                        weights = np.array([w for _, w in distribution])
                        n = round(float(weights.sum()))
                        assert 0 < n <= 50
                        if not filtered and not advantage:
                            assert n == 50
                        density = report.density(distribution, np.linspace(-100, 100, 801))
                        assert np.isclose(np.trapz(density, np.linspace(-100, 100, 801)), n / 50, atol=1e-12)
                        center = float(np.average(values, weights=weights))
                        spread = float(np.sqrt(np.average((values - center) ** 2, weights=weights)))
                        summaries.append({"plot": name, "model": model, "mode": mode, "arm": arm,
                                          "strategy": strategy, "tasks": n, "density_area": n / 50,
                                          "mean_gain_pp": center, "distribution_std_pp": spread})
        report.plot_gain(name, filtered=filtered, advantage=advantage)
    report.write_csv("gain_distributions.csv", summaries)
    report.write_csv("evaluation_coverage.csv", report.COVERAGE)
    metadata = {
        "created_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
        "models": report.LABELS,
        "panel_mapping": {"top": "Qwen Thinking / GPT-OSS High", "bottom": "Qwen Non-thinking / GPT-OSS Low"},
        "metric": "candidate 3-repeat mean PCT minus task-specific initial 5-repeat mean PCT, percentage points",
        "selection": "best/worst exact mean PCT extrema; uniform ties; random uniform over 5 candidates",
        "advantage_selection": "uniform over nonnegative/nonpositive candidates; skip all-zero groups",
        "density": "Gaussian KDE with 7 pp bandwidth; half-violin area proportional to n/50; common width scale per plot",
        "mean_markers": "horizontal bars",
        "source_sha256": report.SOURCES,
        "plot_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([r for r in summaries if r["model"] == "oss120"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
