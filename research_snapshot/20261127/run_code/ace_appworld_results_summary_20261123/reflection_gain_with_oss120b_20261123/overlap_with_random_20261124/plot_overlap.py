# SPDX-License-Identifier: Apache-2.0
"""Annotate the original split violins with their geometric overlap against Random."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent
HELPER = OUT.parents[1] / "build_report.py"
SPEC = importlib.util.spec_from_file_location("violin_overlap_source", HELPER)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
REPORT.OUT = OUT
NAME = "independent_pct_gain_all_tasks_split_violin_n_over_50_with_random"
GRID = np.linspace(-100, 100, 801)


def area(values, grid=GRID):
    return float(np.trapz(values, grid))


def intersection(first, second, grid=GRID):
    """Exact intersection area of the piecewise-linear density curves, including crossings."""
    difference = first - second
    indices = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    if not len(indices):
        return area(np.minimum(first, second), grid)
    fractions = -difference[indices] / (difference[indices + 1] - difference[indices])
    crossings = grid[indices] + fractions * (grid[indices + 1] - grid[indices])
    refined = np.sort(np.concatenate([grid, crossings]))
    return area(np.minimum(np.interp(refined, grid, first), np.interp(refined, grid, second)), refined)


def export_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    original = json.loads((ORIGINAL / "provenance.json").read_text())
    for model, modes in REPORT.MODEL_RUNS.items():
        for mode, run in modes.items():
            REPORT.load_reflection(model, mode, run)
            assert len(REPORT.INITIAL[model, mode]) == 50 and len(REPORT.BANKS[model, mode]) == 100
    for path, sha in REPORT.SOURCES.items():
        if path.endswith(("stage1_results.json", "stage3_results.json", "protocol.json")):
            assert original["source_sha256"].get(path) == sha, f"Original graph source changed: {path}"
    records, cached, arrays = [], {}, {"gain_grid_pp": GRID}
    for model, modes in REPORT.MODEL_RUNS.items():
        for mode in modes:
            for arm in REPORT.ARMS:
                points = REPORT.gain_points(model, mode, arm)
                densities = {strategy: REPORT.density(values, GRID) for strategy, values in points.items()}
                assert all(abs(sum(w for _, w in p) - 50) < 1e-9 for p in points.values())
                best, worst, random = (densities[s] for s in ("best", "worst", "random"))
                assert all(np.isclose(area(d), 1, atol=1e-12) for d in (best, worst, random))
                left = intersection(best, random)
                right = intersection(worst, random)
                colored_area, random_area = area(best) + area(worst), 2 * area(random)
                overlap = left + right
                fraction = overlap / random_area
                union = colored_area + random_area - overlap
                mixture = (best + worst) / 2
                mixed_overlap = intersection(mixture, random) / area(random)
                assert 0 <= fraction <= mixed_overlap + 1e-12 <= 1 + 1e-12
                record = {"model": model, "mode": mode, "arm": arm, "tasks": 50,
                          "best_random_left_overlap_percent": 100 * left / area(random),
                          "worst_random_right_overlap_percent": 100 * right / area(random),
                          "colored_shape_area": colored_area, "random_shape_area": random_area,
                          "intersection_area": overlap, "overlap_of_random_percent": 100 * fraction,
                          "geometric_iou_percent": 100 * overlap / union,
                          "alternative_equal_mixture_density_overlap_percent": 100 * mixed_overlap,
                          "kde_bandwidth_pp": 7, "grid_step_pp": 0.25}
                records.append(record)
                cached[model, mode, arm] = record
                for strategy, density in densities.items():
                    arrays[f"{model}_{mode}_{arm}_{strategy}"] = density
    assert len(records) == 20
    export_csv("overlap_by_model_mode_gt.csv", records)
    np.savez_compressed(OUT / "source_densities.npz", **arrays)
    return records, cached


def draw(cached):
    close = REPORT.plt.close
    REPORT.plt.close = lambda *args, **kwargs: None
    try:
        REPORT.plot_gain("temporary_original_geometry")
        fig = REPORT.plt.gcf()
    finally:
        REPORT.plt.close = close
    fig.set_size_inches(18, 12.2)
    for row in range(2):
        for col, arm in enumerate(REPORT.ARMS):
            ax = fig.axes[row * 2 + col]
            labels = []
            for model in REPORT.LABELS:
                mode = tuple(REPORT.MODEL_RUNS[model])[row]
                overlap = cached[model, mode, arm]["overlap_of_random_percent"]
                labels.append(REPORT.SHORT[model] + f"\nn=50/50\nOverlap {overlap:.1f}%")
            ax.set_xticks(range(5), labels, fontsize=10.5)
            ax.tick_params(axis="x", pad=7)
    for legend in fig.legends:
        legend.set_bbox_to_anchor((0.5, 0.066))
    for text in fig.texts:
        if text.get_text().startswith("All train tasks"):
            text.set_position((0.5, 0.037))
    fig.text(0.5, 0.017,
             "Overlap = common area of the best-left + worst-right shape and Random / total Random area. "
             "Computed from the displayed KDEs; bandwidth 7 pp.",
             ha="center", fontsize=9.5, color="#44566d")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.20, hspace=0.53, wspace=0.10)
    name = NAME + "_overlap_annotated"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=190, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=100, facecolor="white")
    REPORT.plt.close(fig)
    (OUT / "temporary_original_geometry.png").unlink()


def main():
    OUT.mkdir(exist_ok=True)
    # A crossing between sampling grid points must not be discarded.
    assert np.isclose(intersection(np.array([1., 0.]), np.array([0., 1.]), np.array([0., 1.])), 0.25)
    assert np.isclose(intersection(np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([-1., 1.])), 1)
    original = ORIGINAL / f"{NAME}.png"
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    records, cached = collect()
    draw(cached)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_sha
    manifest = {"generated_at": datetime.now(UTC).isoformat(), "original_image": str(original),
                "original_image_sha256_unchanged": original_sha, "groups": len(records),
                "displayed_metric": "(integral min(best, random) + integral min(worst, random)) "
                                    "/ (2 * integral random), expressed as a percentage",
                "meaning": "Intersection of the displayed colored left+right shape and symmetric gray Random "
                           "shape, divided by the gray shape area. Both original shapes have equal total area.",
                "secondary_csv_metrics": "Separate left/right overlap, geometric IoU, and equal-weight best/worst "
                                         "mixture-density overlap. The latter is not the split-shape overlap.",
                "integration": "Piecewise-linear density curves on the original 0.25 pp grid; crossings inserted",
                "density": "Original Gaussian KDE, 7 pp bandwidth, per-kernel 4-SD truncation and boundary "
                           "renormalization; original n/50 task weighting and uniform candidate selection pools",
                "model_order": list(REPORT.LABELS), "new_inference_calls": 0,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "report_helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                "source_sha256": REPORT.SOURCES}
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for record in records:
        print(record["model"], record["mode"], record["arm"],
              f"overlap={record['overlap_of_random_percent']:.2f}%",
              f"mixture={record['alternative_equal_mixture_density_overlap_percent']:.2f}%")
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
