# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Render the four offline split measurements as a publication table without online scores or stars."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research_journal.result_tables import rounded  # noqa: E402
from scripts.utils.plot_checkpoint_learning_curves import BENCHMARKS, write_csv  # noqa: E402

METRICS = {"train": "Train", "val": "Val", "test": "Test", "test_early": "Test (early)"}
CAPTION = (
    "Offline performance (%). Benchmark cells show seed mean (sample SD); Avg. is the mean of within-seed "
    "benchmark macro-averages. Train, Val and Test use final CP50 for BFCL/AppWorld and CP30 for Terminal-Bench. "
    "For GEPA Val and SkillOpt Val, Test (early) reuses Test of the training Val-best final policy. "
    "Other settings use each seed's best re-evaluated Val CP, including CP0; exact ties choose the earliest CP. "
    "Only complete seed/split evaluations are included. Terminal Train uses n=2 for ACE, SkillOpt Val, "
    "SkillOpt Train-as-val and PEACE; ACE Test (early) uses n=1 (seed43). All other benchmark cells use n=3. "
    "Avg. uses the same complete seeds across all benchmarks; a single seed has no SD. "
    "AppWorld uses Original policy + code generation and official TGC. "
    "Test is the final-CP queue reevaluation, independent of the earlier main-table final test run. "
    "Top distinct means are bold and second-highest are underlined; ties share formatting. No star markers."
)


def summarize(rows, average=False):
    valid = [r for r in rows if r["score"] is not None]
    values = [100 * r["score"] for r in valid]
    mean = float(rounded(statistics.mean(values))) if values else None
    sd = float(rounded(statistics.stdev(values))) if len(values) > 1 else None
    return {"mean": mean, "std": sd, "n": len(valid), "seeds": [r["seed"] for r in valid],
            "display": "--" if mean is None else f"{mean:.2f}" +
            (f" ({sd:.2f})" if sd is not None and not average else ""),
            "rank": 0, "color": "#ffffff", "shade": "white", "average": average}


def build_table(source, config):
    with (source / "offline_split_seed_scores.csv").open() as stream:
        samples = [{**row, "seed": int(row["seed"]), "score": float(row["score"]) if row["score"] else None}
                   for row in csv.DictReader(stream)]
    identity = {(r["variant"], r["benchmark"], r["seed"], r["metric"]) for r in samples}
    expected = {(row["variant"], benchmark, seed, metric) for row in config["rows"]
                for benchmark in BENCHMARKS for seed in config["seeds"] for metric in METRICS}
    if identity != expected or len(samples) != len(expected):
        raise ValueError("Source figure does not contain the expected six-setting cohort")
    groups = {**BENCHMARKS, "avg": "Avg."}
    rows, macro_samples = [], []
    for spec in config["rows"]:
        cells = {}
        for metric in METRICS:
            for benchmark in BENCHMARKS:
                selected = [s for s in samples if (s["variant"], s["benchmark"], s["metric"]) ==
                            (spec["variant"], benchmark, metric)]
                cells[f"{benchmark}.{metric}"] = summarize(selected)
            macros = []
            for seed in config["seeds"]:
                selected = [s for s in samples if (s["variant"], s["seed"], s["metric"]) ==
                            (spec["variant"], seed, metric)]
                value = statistics.mean(s["score"] for s in selected) if all(
                    s["score"] is not None for s in selected) else None
                macros.append({"variant": spec["variant"], "benchmark": "avg", "seed": seed,
                               "metric": metric, "score": value})
            macro_samples.extend(macros)
            cells[f"avg.{metric}"] = summarize(macros, average=True)
        rows.append({"id": spec["id"], "label": spec["label"], "setting": spec["setting_label"],
                     "method": spec["method"], "cells": cells})
    columns = [{"id": f"{benchmark}.{metric}", "benchmark": title, "metric": label}
               for benchmark, title in groups.items() for metric, label in METRICS.items()]
    for column in columns:
        values = sorted({row["cells"][column["id"]]["mean"] for row in rows
                         if row["cells"][column["id"]]["mean"] is not None}, reverse=True)
        for row in rows:
            cell = row["cells"][column["id"]]
            if cell["mean"] is None:
                continue
            rank = values.index(cell["mean"]) + 1
            cell["rank"] = rank if rank <= 2 else 0
            if len(values) > 1:
                normalized = (cell["mean"] - values[-1]) / (values[0] - values[-1])
                amount = round(35 * abs(normalized - .5) * 2)
                name, rgb = ("scorered", (183, 62, 41)) if normalized < .5 else ("scoreblue", (7, 92, 173))
                cell["shade"] = f"{name}!{amount}!white"
                cell["color"] = "#" + "".join(f"{round(255 + (c - 255) * amount / 100):02x}" for c in rgb)
    return {"title": "Offline main table · Original AppWorld policy · PEACE w/o alignment",
            "seeds": config["seeds"], "columns": columns, "rows": rows, "caption": CAPTION}, macro_samples


def latex(table):
    def esc(text):
        return text.replace("&", r"\&").replace("%", r"\%").replace(" · ", " / ")

    lines = [r"\documentclass{article}", r"\usepackage[margin=1cm,landscape,a3paper]{geometry}",
             r"\usepackage[table]{xcolor}", r"\usepackage{booktabs,adjustbox}",
             r"\definecolor{oursbg}{RGB}{241,246,252}", r"\definecolor{scorered}{HTML}{B73E29}",
             r"\definecolor{scoreblue}{HTML}{075CAD}", r"\begin{document}\thispagestyle{empty}",
             r"\begin{table}[ht]\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.2}", r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{@{}ll*{16}{c}@{}}\toprule",
             r"\textbf{Method} & \textbf{Setting} & " + " & ".join(
                 r"\multicolumn{4}{c}{\textbf{" + name + "}}" for name in (*BENCHMARKS.values(), "Avg.")) + r" \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}\cmidrule(lr){15-18}",
             " & & " + " & ".join(col["metric"] for col in table["columns"]) + r" \\\midrule"]
    for row in table["rows"]:
        rendered = []
        for col in table["columns"]:
            cell = row["cells"][col["id"]]
            text = "--" if cell["mean"] is None else f"{cell['mean']:.2f}" + (
                rf"\,{{\scriptsize ({cell['std']:.2f})}}" if cell["std"] is not None and not cell["average"] else "")
            if cell["rank"] == 1:
                text = r"{\bfseries\boldmath " + text + "}"
            elif cell["rank"] == 2:
                text = r"\underline{" + text + "}"
            rendered.append(r"\cellcolor{" + cell["shade"] + "}" + text)
        prefix = r"\rowcolor{oursbg}" if row["method"] == "premise" else ""
        lines.append(prefix + r"\textbf{" + row["label"] + "} & " + esc(row["setting"]) + " & " +
                     " & ".join(rendered) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}", r"\caption{" + esc(CAPTION) + "}",
              r"\label{tab:offline_main}\end{table}\end{document}"]
    return "\n".join(lines) + "\n"


def draw(table, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    widths = [1.3, 2.9] + [1.55] * 16
    xs = [0]
    for width in widths:
        xs.append(xs[-1] + width)
    fig, ax = plt.subplots(figsize=(xs[-1], 6.4))
    ax.set(xlim=(0, xs[-1]), ylim=(-2.15, 5.25))
    ax.axis("off")
    ax.text(0, 5.1, "Qwen3.6 · Offline main table", fontsize=20, weight="bold", va="top")
    ax.text(0, 4.6, "Original AppWorld policy · PEACE AUD w/o alignment · seeds 42-44 · scores (%)",
            fontsize=12, color="#526271", va="top")
    top, height = 3.9, .59
    for group, start in zip((*BENCHMARKS.values(), "Avg."), (2, 6, 10, 14), strict=True):
        left, right = xs[start], xs[start + 4]
        ax.text((left + right) / 2, top, group, fontsize=13, weight="bold", ha="center", va="center")
        ax.plot([left + .04, right - .04], [top - .24, top - .24], color="#9fa9b2", lw=.8)
    for index, label in enumerate(["Method", "Setting"] + [c["metric"] for c in table["columns"]]):
        ax.text((xs[index] + xs[index + 1]) / 2, top - .53, label, fontsize=10.5,
                weight="bold", ha="center", va="center")
    ax.plot([0, xs[-1]], [top + .3, top + .3], color="#536271", lw=1.2)
    ax.plot([0, xs[-1]], [top - .81, top - .81], color="#536271", lw=1)
    underlines = []
    for index, row in enumerate(table["rows"]):
        y = top - .81 - (index + 1) * height
        bg = "#f1f6fc" if row["method"] == "premise" else "white"
        ax.add_patch(Rectangle((0, y), xs[-1], height, facecolor=bg, edgecolor="none"))
        ax.text(.08, y + height / 2, row["label"], fontsize=11, weight="bold", va="center")
        ax.text(xs[1] + .08, y + height / 2, row["setting"].replace(" · ", " / "), fontsize=10.3, va="center")
        for ci, col in enumerate(table["columns"]):
            cell = row["cells"][col["id"]]
            left, width = xs[ci + 2], widths[ci + 2]
            ax.add_patch(Rectangle((left, y), width, height, facecolor=cell["color"], edgecolor="none"))
            text = ax.text(left + width / 2, y + height / 2, cell["display"], fontsize=10.1,
                           weight="bold" if cell["rank"] == 1 else "normal", ha="center", va="center")
            if cell["rank"] == 2:
                underlines.append((text, y + .13))
        ax.plot([0, xs[-1]], [y, y], color="#e0e5ea", lw=.45)
    fig.canvas.draw()
    for text, y in underlines:
        box = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        ax.plot([box.x0, box.x1], [y, y], color="#344454", lw=.7)
    bottom = top - .81 - len(table["rows"]) * height
    ax.plot([0, xs[-1]], [bottom, bottom], color="#536271", lw=1.1)
    notes = [
        "Train / Val / Test: final-CP queue evaluation (BFCL & AppWorld CP50; Terminal-Bench CP30). "
        "GEPA Val / SkillOpt Val: Test (early) reuses Test of the training Val-best final policy.",
        "Other settings: Test (early) uses per-seed best re-evaluated Val CP, CP0 included; earliest exact tie. "
        "No checkpoint is selected using Test scores.",
        "Benchmark cells = mean (sample SD). Avg. = within-seed macro-average, then seed mean. "
        "Incomplete seed/split measurements are excluded; one seed has no SD. No star markers.",
        "Terminal Train: n=2 for ACE, SkillOpt Val, SkillOpt Train-as-val and PEACE. "
        "ACE Test (early): n=1 (seed43). Other benchmark cells: n=3. Avg. uses the same complete seeds.",
        "AppWorld uses official TGC. Test is a CP reevaluation, independent of the earlier final test run. "
        "Bold: highest; underline: second distinct mean. Colors are normalized within columns.",
    ]
    for i, note in enumerate(notes):
        ax.text(0, bottom - .30 - i * .27, note, fontsize=9.5, color="#526271", va="top")
    fig.subplots_adjust(left=.014, right=.995, bottom=.025, top=.99)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"offline_main_table.{suffix}", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Offline split figure directory")
    parser.add_argument("--table-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.table_manifest.read_text())
    provenance = json.loads((args.source / "figure_manifest.json").read_text())
    for name in ("offline_split_seed_scores.csv", "offline_split_summary.csv"):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != provenance["files"][name]:
            raise ValueError("Source figure data changed")
    table, macro = build_table(args.source, config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "offline_main_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    (args.output / "offline_main_table.tex").write_text(latex(table))
    body = latex(table).split(r"\begin{table}[ht]", 1)[1].split(r"\end{table}", 1)[0]
    (args.output / "offline_main_table_body.tex").write_text(r"\begin{table*}[t]" + body + r"\end{table*}" + "\n")
    headings = ["Method", "Setting"] + [c["benchmark"] + " " + c["metric"] for c in table["columns"]]
    with (args.output / "offline_main_table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        for row in table["rows"]:
            writer.writerow([row["label"], row["setting"]] + [row["cells"][c["id"]]["display"]
                                                                  for c in table["columns"]])
    lines = ["# Offline main table", "", "| " + " | ".join(headings) + " |",
             "|" + "|".join(["---", "---"] + ["---:"] * 16) + "|"]
    for row in table["rows"]:
        lines.append("| " + " | ".join([row["label"], row["setting"]] +
                     [row["cells"][c["id"]]["display"] for c in table["columns"]]) + " |")
    (args.output / "offline_main_table.md").write_text("\n".join(lines) + "\n\n" + CAPTION + "\n")
    write_csv(args.output / "macro_seed_scores.csv", macro)
    draw(table, args.output)
    manifest = {"created_at": datetime.now(UTC).isoformat(), "source": str(args.source.resolve()),
                "journal_id": config["journal_id"], "source_figure_manifest": provenance,
                "columns": 16, "rows": 6, "online": False, "star_markers": False,
                "macro_average": "Within-seed mean only when all three benchmarks have that seed's complete value",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                          if p.is_file() and p.name != "manifest.json"}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "rows": 6, "columns": 16}, ensure_ascii=False))


if __name__ == "__main__":
    main()
