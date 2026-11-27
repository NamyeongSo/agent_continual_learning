# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Export six recorded PREMiSE trees on one page, collapsing omitted ancestry exactly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.premise.plot_search_tree import COLORS, build_graph, node_color  # noqa: E402

DEFAULT_STEPS = (1, 2, 3, 50, 51, 100, 101, 129, 130)
CANDIDATE_X = {1: 3.1, 2: 4.45, 3: 5.8, 4: 7.15}
PARENT_X = (3.775, 6.475)
BENCHMARK_LABELS = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "TB2"}


def panel_node_color(node: dict) -> str:
    """Use neutral outlines for white failed candidates and carried beam nodes."""
    if node["kind"] == "carry" or (node["kind"] == "candidate" and not node["valid"]):
        return COLORS["pruned"]
    return node_color(node)


def contract_graph(graph: dict, steps: tuple[int, ...]) -> dict:
    """Connect displayed nodes to their real ancestors, rather than matching beam ranks."""
    if not steps or tuple(sorted(set(steps))) != steps or steps[0] < 1 or steps[-1] > graph["steps"]:
        raise ValueError("Displayed steps must be unique, increasing, and within the recorded run")
    by_id = {node["id"]: node for node in graph["nodes"]}
    displayed, edges, omitted = [], [], []
    layout = {0: 0}
    previous, y = 0, 0
    for step in steps:
        if step - previous > 1:
            omitted.append({"start": previous + 1, "end": step - 1, "y": y + 1})
            y += 1
        y += 1
        layout[step] = y
        row = graph["rows"][step - 1]
        if row["skipped"] and any(by_id[nid]["kind"] == "candidate" for nid in row["nodes"]):
            y += 0.65
        previous = step
    for node in graph["nodes"]:
        if node["iteration"] not in layout:
            continue
        row = graph["rows"][node["iteration"] - 1] if node["iteration"] else None
        x = CANDIDATE_X[node["number"]] if node["kind"] == "candidate" else PARENT_X[node["rank"] - 1]
        plot_y = layout[node["iteration"]]
        # Retain recorded attempts at skipped steps. Carry nodes sit below them,
        # connected to the previous beam rather than to an unselected attempt.
        if node["kind"] == "carry" and row and any(by_id[nid]["kind"] == "candidate" for nid in row["nodes"]):
            plot_y += 0.65
        displayed.append({**node, "plot_x": x, "plot_y": plot_y})
    visible_ids = {node["id"] for node in displayed}
    previous_step = dict(zip(steps, (0, *steps[:-1]), strict=True))
    for node in displayed:
        if node["kind"] == "root":
            continue
        ancestor = node["parent"]
        path = [node["id"], ancestor]
        while ancestor not in visible_ids:
            ancestor = by_id[ancestor]["parent"]
            path.append(ancestor)
        if by_id[ancestor]["iteration"] != previous_step[node["iteration"]]:
            raise ValueError(f"Ancestry bypasses the preceding displayed beam: {node['id']}")
        edges.append({
            "source": ancestor,
            "target": node["id"],
            "collapsed": len(path) > 2,
            "omitted_node_count": len(path) - 2,
            "full_path": list(reversed(path)),
        })
    if len(edges) != len(displayed) - 2:
        raise ValueError("Every displayed non-root node must have exactly one ancestor edge")
    return {"nodes": displayed, "edges": edges, "omitted": omitted, "layout": layout}


def draw_panel(ax, graph: dict, compressed: dict) -> None:
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MplPath

    nodes = {node["id"]: node for node in compressed["nodes"]}
    seed, mode = graph["config"]["seed"], graph["config"]["mode"]
    stats = graph["stats"]
    ax.text(0, -1.5, f"{mode.capitalize()}  |  seed {seed}", fontsize=12, weight="bold", color=COLORS["text"])
    caption = (f"{stats['selection_updates']}/130 selections  |  {stats.get('skipped_steps', 0)} skips  |  "
               f"{stats['steps_with_branch_errors']} steps with errors")
    ax.text(0, -1.0, caption, fontsize=7.3, color="#64748b")
    ax.text(0, 0, "Initial beam", fontsize=7.3, va="center", color="#64748b")
    for step, y in compressed["layout"].items():
        if not step:
            continue
        row = graph["rows"][step - 1]
        marker = "  SKIP" if row["skipped"] else "*" if row["has_errors"] else ""
        label = f"{step:03d}{marker}  {BENCHMARK_LABELS.get(row['benchmark'], row['benchmark'])}"
        ax.text(0, y, label, fontsize=6.9, va="center", color=COLORS["text"])
        if not row["skipped"]:
            for number in row["missing_candidates"]:
                ax.text(CANDIDATE_X[number], y, "missing", fontsize=5.8, ha="center", va="center",
                        color=COLORS["pruned"])
    for gap in compressed["omitted"]:
        ax.text(0.04, gap["y"], f"... {gap['start']}-{gap['end']} omitted", fontsize=6.4,
                va="center", color="#7a8798", style="italic")
    for edge in compressed["edges"]:
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        x1, y1 = source["plot_x"], source["plot_y"] + 0.19
        x2, y2 = target["plot_x"], target["plot_y"] - 0.19
        middle = (y1 + y2) / 2
        curve = MplPath([(x1, y1), (x1, middle), (x2, middle), (x2, y2)],
                        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        retained = bool(target.get("rank"))
        ax.add_patch(PathPatch(curve, facecolor="none", edgecolor=panel_node_color(target),
                               linewidth=1.0 if retained else 0.65, alpha=1 if retained else 0.65,
                               linestyle=(0, (3, 2)) if edge["collapsed"] or target["kind"] == "carry" else "-",
                               zorder=1))
    for node in nodes.values():
        x, y, color = node["plot_x"], node["plot_y"], panel_node_color(node)
        filled = bool(node.get("rank")) and node["kind"] != "carry"
        if node["kind"] == "root":
            label = f"Parent {node['rank']}"
        elif node["kind"] == "carry":
            label = f"Keep R{node['rank']}"
        else:
            label = f"C{node['number']}"
            if node["rank"]:
                label += f" / R{node['rank']}"
            elif not node["valid"]:
                label += " / ERR"
        box = FancyBboxPatch((x - 0.525, y - 0.19), 1.05, 0.38,
                             boxstyle="round,pad=0.015,rounding_size=0.07",
                             facecolor=color if filled else "white", edgecolor=color, linewidth=0.8, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=6.5, weight="medium", ha="center", va="center",
                color="white" if filled else color, zorder=4)
    ax.set_xlim(-0.12, 7.9)
    ax.set_ylim(max(node["plot_y"] for node in nodes.values()) + 0.6, -1.85)
    ax.axis("off")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--display-steps", default=",".join(map(str, DEFAULT_STEPS)))
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.display_steps.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    # A3 landscape gives six readable panels while keeping the PDF to one page.
    fig, axes = plt.subplots(2, 3, figsize=(16.54, 11.69))
    records = []
    for row, mode in enumerate(("sequential", "interleaved")):
        for column, seed in enumerate((42, 43, 44)):
            graph = build_graph(args.run_root / f"s{seed}_{mode}_t3", 130)
            if graph["config"]["seed"] != seed or graph["config"]["mode"] != mode:
                raise ValueError("Run configuration does not match the requested panel")
            compressed = contract_graph(graph, steps)
            draw_panel(axes[row, column], graph, compressed)
            records.append({
                "source_run": graph["source_run"], "config": graph["config"], "stats": graph["stats"],
                "skipped_steps": [r["iteration"] for r in graph["rows"] if r["skipped"]],
                "source_hashes": graph["source_hashes"], "displayed_steps": steps,
                "contracted_edges": compressed["edges"], "omitted_intervals": compressed["omitted"],
                "displayed_node_ids": [n["id"] for n in compressed["nodes"]],
            })
    fig.suptitle("PREMiSE search trees | six runs", fontsize=19, weight="bold", y=0.975, color=COLORS["text"])
    fig.text(0.5, 0.935, "Beam width 2: two parents expand to four candidates; two are retained.",
             ha="center", fontsize=10.5, color="#536378")
    handles = [
        Line2D([], [], color=COLORS["pruned"] if key in ("invalid", "carry") else COLORS[key],
               marker="s", linestyle="none", markersize=6,
               markerfacecolor="white" if key in ("invalid", "carry") else COLORS[key], label=label)
        for key, label in (
            ("rank1", "Next rank 1"), ("rank2", "Next rank 2"), ("pruned", "Pruned"),
            ("invalid", "Failed / missing"), ("carry", "Skipped: beam carried"),
        )
    ]
    handles.append(Line2D([], [], color="#64748b", linestyle="--", linewidth=1,
                          label="Actual ancestry through omitted steps"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.918), ncol=6,
               frameon=False, fontsize=8.3, handlelength=1.6, columnspacing=1.5)
    fig.subplots_adjust(top=0.875, bottom=0.10, left=0.035, right=0.985, wspace=0.12, hspace=0.19)
    fig.text(0.035, 0.067,
             "Dashed links follow the recorded ancestor chain; omitted steps are not drawn as direct mutations. "
             "* = recorded branch error at a displayed step.", fontsize=8, color="#536378")
    fig.text(0.035, 0.043,
             "All six runs contain 130 encounters, including skips. Counts refer to beam selection, not necessarily "
             "a text mutation. No run is completely error-free.", fontsize=8, color="#536378")
    fig.text(0.035, 0.019,
             "Qwen3.6-35B-A3B | past=3 | BFCL / AppWorld / Terminal-Bench 2.0 | "
             "Original scorefix six-run matrix (2026-09-28).", fontsize=8, color="#64748b")
    name = "PREMiSE_s42_s43_s44_seq_interleaved_one_page"
    pdf = args.output_dir / f"{name}.pdf"
    fig.savefig(pdf, facecolor="white", metadata={
        "Title": "PREMiSE six-run search trees with omitted intermediate steps",
        "Subject": "Exact recorded ancestry; sequential and interleaved seeds 42, 43, 44; one A3 landscape page",
    })
    fig.savefig(args.output_dir / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"pdf": str(pdf.resolve()), "displayed_steps": steps,
                      "runs": [{"seed": r["config"]["seed"], "mode": r["config"]["mode"],
                                "selections": r["stats"]["selection_updates"], "skips": r["skipped_steps"]}
                               for r in records]}, indent=2))


if __name__ == "__main__":
    main()
