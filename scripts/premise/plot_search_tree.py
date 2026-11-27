# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Export recorded PREMiSE beam genealogy, retaining skipped steps and failed branches."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

COLORS = {
    "rank1": "#087f74",
    "rank2": "#4169bc",
    "pruned": "#a4acb9",
    "invalid": "#d34848",
    "carry": "#b97812",
    "text": "#263449",
}
BENCHMARKS = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "Terminal-Bench", "swebench": "SWE-bench"}
SLOTS = {1: 420, 2: 630, 3: 840, 4: 1050}
ROOT_SLOTS = [525, 945]
ROW_HEIGHT = 128


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bullet_count(playbook: dict) -> int:
    if "entries" in playbook:
        return len(playbook["entries"])
    return sum(line.startswith("[") and " :: " in line for line in playbook.get("playbook", "").splitlines())


def build_graph(run: Path, steps: int) -> dict:
    """Resolve parent ranks through each recorded selection; skips carry the previous beam."""
    state = read_json(run / "state.json")
    config = state["config"]
    if steps > state["cursor"] or len(config["order"]) < steps:
        raise ValueError(f"Requested {steps} steps, but saved cursor is {state['cursor']}")
    nodes = [
        {"id": f"root{rank}", "kind": "root", "iteration": 0, "rank": rank, "x": x, "y": 0}
        for rank, x in enumerate(ROOT_SLOTS, 1)
    ]
    beam = [n["id"] for n in nodes]
    edges, rows, hashes = [], [], []
    stats = Counter()
    for iteration in range(1, steps + 1):
        source = run / "iterations" / f"iter_{iteration:04d}.json"
        raw = source.read_bytes()
        artifact = json.loads(raw)
        if artifact["iteration"] != iteration:
            raise ValueError(f"Iteration mismatch: {source}")
        if list(config["order"][iteration - 1]) != [artifact["benchmark"], artifact["task_id"]]:
            raise ValueError(f"Frozen task order mismatch: {source}")
        hashes.append({"path": str(source.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        selection = artifact.get("selection", {})
        selected = selection.get("selected", [])
        skipped = artifact.get("status") == "skipped" or artifact.get("train_update_applied") is False
        if not skipped and (len(selected) != 2 or len(set(selected)) != 2):
            raise ValueError(f"Expected two distinct selected candidates: {source}")
        if skipped and selected:
            raise ValueError(f"Skipped step unexpectedly contains a selection: {source}")
        candidates = artifact.get("candidates", [])
        row = {
            "iteration": iteration,
            "benchmark": artifact["benchmark"],
            "task_id": artifact["task_id"],
            "skipped": skipped,
            "reason": artifact.get("reason"),
            "errors": artifact.get("errors", []),
            "parents": list(beam),
            "parent_scores": [p.get("score") if p else None for p in artifact.get("parents", [])],
            "selection": selection,
            "past_sample": artifact.get("past_sample", []),
            "source": str(source.resolve()),
            "nodes": [],
        }
        by_candidate = {}
        for candidate in candidates:
            cid = candidate["id"]
            number = int(cid.rsplit("_", 1)[-1])
            parent_rank = candidate["parent_rank"]
            if number not in SLOTS or parent_rank not in (1, 2) or (number - 1) // 2 + 1 != parent_rank:
                raise ValueError(f"Unexpected candidate/parent pairing: {source}: {cid}")
            rank = selected.index(cid) + 1 if cid in selected else None
            if rank and not candidate.get("valid"):
                raise ValueError(f"Selected invalid candidate: {source}: {cid}")
            node = {
                "id": f"i{iteration:03d}-c{number}",
                "kind": "candidate",
                "iteration": iteration,
                "candidate_id": cid,
                "number": number,
                "parent_rank": parent_rank,
                "parent": beam[parent_rank - 1],
                "rank": rank,
                "valid": bool(candidate.get("valid")),
                "applied": candidate.get("applied", []),
                "errors": candidate.get("errors", []),
                "mutation_summary": candidate.get("mutation_summary", ""),
                "reflection": candidate.get("reflection", {}),
                "bullet_count": bullet_count(candidate["playbook"]),
                "metrics": {
                    key: candidate[key]
                    for key in (
                        "current_alignment", "past_alignment", "current_official_score", "past_official_score",
                        "current_pct", "past_pct",
                    )
                    if key in candidate
                },
                "replay_attempts": [
                    result.get("replay_attempts", 1)
                    for result in [candidate.get("current_result", {}), *candidate.get("past_results", [])]
                ],
                "x": SLOTS[number],
                "y": iteration * ROW_HEIGHT,
            }
            if cid in by_candidate:
                raise ValueError(f"Duplicate candidate: {source}: {cid}")
            by_candidate[cid] = node
            nodes.append(node)
            row["nodes"].append(node["id"])
            edges.append({"source": node["parent"], "target": node["id"], "carry": False})
            stats["candidates"] += 1
            stats["invalid_candidates"] += not node["valid"]
            stats["candidate_error_messages"] += len(node["errors"])
            stats["replay_jobs_with_retry"] += sum(a > 1 for a in node["replay_attempts"])
            if rank:
                stats["selected_candidates"] += 1
                stats["selected_candidates_without_operations"] += not bool(node["applied"])
                stats["primary_selections_with_operations"] += rank == 1 and bool(node["applied"])
        row["missing_candidates"] = [number for number in SLOTS if f"candidate_{number}" not in by_candidate]
        if skipped:
            next_beam = []
            for rank, (parent, x) in enumerate(zip(beam, ROOT_SLOTS, strict=True), 1):
                node = {
                    "id": f"i{iteration:03d}-carry{rank}", "kind": "carry", "iteration": iteration,
                    "rank": rank, "parent": parent, "x": x, "y": iteration * ROW_HEIGHT,
                }
                nodes.append(node)
                row["nodes"].append(node["id"])
                edges.append({"source": parent, "target": node["id"], "carry": True})
                next_beam.append(node["id"])
            stats["skipped_steps"] += 1
        else:
            beam_candidates = [by_candidate[cid] for cid in selected]
            next_beam = [node["id"] for node in beam_candidates]
            stats["selection_updates"] += 1
            stats["same_parent_selections"] += len({node["parent_rank"] for node in beam_candidates}) == 1
        row["next_beam"] = next_beam
        row["has_errors"] = bool(row["errors"]) or any(not c.get("valid") or c.get("errors") for c in candidates)
        stats["steps_with_branch_errors"] += row["has_errors"]
        stats["error_free_selection_updates"] += not skipped and not row["has_errors"]
        stats["top_level_error_messages"] += len(row["errors"])
        stats["missing_candidates"] += len(row["missing_candidates"])
        rows.append(row)
        beam = next_beam
    ids = {node["id"] for node in nodes}
    if len(ids) != len(nodes) or any(edge["source"] not in ids for edge in edges):
        raise ValueError("Graph has duplicate or unresolved nodes")
    if len(edges) != len(nodes) - 2:
        raise ValueError("Every non-root node must have exactly one parent")
    return {
        "source_run": str(run.resolve()),
        "config": {k: config.get(k) for k in ("seed", "mode", "model", "benchmarks", "past_count")},
        "source_cursor": state["cursor"],
        "source_order_length": len(config["order"]),
        "steps": steps,
        "stats": dict(stats),
        "benchmark_counts": dict(Counter(r["benchmark"] for r in rows)),
        "final_beam": beam,
        "nodes": nodes,
        "edges": edges,
        "rows": rows,
        "source_hashes": hashes,
        "metric_type": "official_score" if any("current_official_score" in n.get("metrics", {}) for n in nodes)
        else "legacy_pct_surrogate",
        "created_at": datetime.now(UTC).isoformat(),
    }


def node_color(node: dict) -> str:
    if node["kind"] == "carry":
        return COLORS["carry"]
    if node["kind"] == "root":
        return COLORS[f"rank{node['rank']}"]
    if not node["valid"]:
        return COLORS["invalid"]
    return COLORS[f"rank{node['rank']}"] if node["rank"] else COLORS["pruned"]


def node_label(node: dict) -> str:
    if node["kind"] == "root":
        return f"initial P{node['rank']}"
    if node["kind"] == "carry":
        return f"keep R{node['rank']}"
    suffix = f" R{node['rank']}" if node["rank"] else " error" if not node["valid"] else " pruned"
    return f"C{node['number']}{suffix}"


def metric_label(node: dict) -> str:
    if node["kind"] != "candidate" or not node["valid"]:
        return ""
    m = node["metrics"]
    performance = "official_score" if "current_official_score" in m else "pct"

    def pair(kind: str) -> str:
        values = [m.get(f"current_{kind}"), m.get(f"past_{kind}")]
        return "/".join("-" if value is None else f"{value:.2f}" for value in values)

    return f"A {pair('alignment')}   {'S' if performance == 'official_score' else 'PCT'} {pair(performance)}"


def make_svg(graph: dict) -> str:
    """Draw the full chronological forest with one row per encountered training task."""
    height = (graph["steps"] + 1) * ROW_HEIGHT + 110
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#263449}.node{cursor:pointer}.node:hover rect'
        '{stroke-width:3}.node.active rect{stroke:#ee9b24;stroke-width:4}.step{scroll-margin-top:20px}</style>',
        f'<rect width="1200" height="{height}" fill="#ffffff"/>',
        '<text x="30" y="34" font-size="21" font-weight="700">PREMiSE recorded search tree</text>',
        '<text x="30" y="61" font-size="14">Two parents expand to four candidates; selection retains two.</text>',
    ]
    offset = 108
    by_id = {node["id"]: node for node in graph["nodes"]}
    for row in graph["rows"]:
        y = row["iteration"] * ROW_HEIGHT + offset
        fill = "#fff8eb" if row["skipped"] else "#f8fafc" if row["iteration"] % 2 == 0 else "#ffffff"
        parts.append(f'<g class="step" id="step-{row["iteration"]}">')
        parts.append(f'<rect x="0" y="{y - 45}" width="1200" height="128" fill="{fill}"/>')
        parts.append(f'<text x="25" y="{y - 3}" font-size="18" font-weight="700">{row["iteration"]:03d}</text>')
        benchmark = BENCHMARKS.get(row["benchmark"], row["benchmark"])
        parts.append(f'<text x="82" y="{y - 3}" font-size="14">{html.escape(benchmark)}</text>')
        task = row["task_id"] if len(row["task_id"]) <= 41 else row["task_id"][:38] + "..."
        parts.append(f'<text x="25" y="{y + 23}" font-size="11">{html.escape(task)}</text>')
        status = "SKIP: beam unchanged" if row["skipped"] else "branch error" if row["has_errors"] else "updated"
        parts.append(f'<text x="25" y="{y + 47}" font-size="11">{status}</text></g>')
        if not row["skipped"]:
            for number in row["missing_candidates"]:
                parts.append(
                    f'<text x="{SLOTS[number]}" y="{y + 5}" text-anchor="middle" '
                    'font-size="13" style="fill:#d34848">not produced</text>'
                )
    for edge in graph["edges"]:
        start, end = by_id[edge["source"]], by_id[edge["target"]]
        x1, y1, x2, y2 = start["x"], start["y"] + offset + 22, end["x"], end["y"] + offset - 22
        color = node_color(end)
        dash = 'stroke-dasharray="5 4"' if edge["carry"] else ""
        opacity = "1" if end.get("rank") else "0.55"
        middle = (y1 + y2) / 2
        parts.append(
            f'<path d="M{x1},{y1} C{x1},{middle} {x2},{middle} {x2},{y2}" '
            f'fill="none" stroke="{color}" stroke-width="2" opacity="{opacity}" {dash}/>'
        )
    for node in graph["nodes"]:
        x, y, color = node["x"], node["y"] + offset, node_color(node)
        selected = bool(node.get("rank"))
        title = f"{node['id']} | {node_label(node)} | {metric_label(node)}"
        parts.append(f'<g class="node" id="{node["id"]}" data-node="{node["id"]}"><title>{html.escape(title)}</title>')
        parts.append(
            f'<rect x="{x - 85}" y="{y - 22}" width="170" height="44" rx="9" '
            f'fill="{color if selected else "#fff"}" stroke="{color}" stroke-width="1.5"/>'
        )
        text_color = "#fff" if selected else color
        parts.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="14" '
            f'style="fill:{text_color}" font-weight="600">{node_label(node)}</text>'
        )
        metrics = metric_label(node)
        if metrics:
            parts.append(f'<text x="{x}" y="{y + 42}" text-anchor="middle" font-size="10">{metrics}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def plot_panel(ax, graph: dict, start: int, end: int, *, detailed: bool = False) -> None:
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MplPath

    by_id = {node["id"]: node for node in graph["nodes"]}
    selected_rows = graph["rows"][start - 1 : end]
    parent_ids = selected_rows[0]["parents"]
    display = []
    for rank, pid in enumerate(parent_ids, 1):
        source = by_id[pid]
        display.append({**source, "x": ROOT_SLOTS[rank - 1], "plot_y": 0, "label": f"in R{rank}: {pid}"})
    for row in selected_rows:
        for nid in row["nodes"]:
            node = by_id[nid]
            display.append({**node, "plot_y": row["iteration"] - start + 1, "label": node_label(node)})
    locations = {n["id"]: (n["x"], n["plot_y"]) for n in display}
    text_size = 7 if detailed else 5.8
    for row in selected_rows:
        y = row["iteration"] - start + 1
        if row["skipped"] or row["has_errors"]:
            ax.axhspan(y - 0.36, y + 0.50, color="#fff3d9" if row["skipped"] else "#fff8f7", zorder=0)
        bm = BENCHMARKS.get(row["benchmark"], row["benchmark"])
        ax.text(15, y, f"{row['iteration']:03d} {bm}", va="center", fontsize=text_size, weight="medium")
        if detailed:
            task = row["task_id"]
            ax.text(15, y + 0.27, task[:45], va="center", fontsize=5.4, color="#69778b")
        if row["skipped"]:
            ax.text(15, y + 0.28, "SKIP: keep both parents", fontsize=5.5, color=COLORS["carry"])
        elif row["missing_candidates"]:
            for number in row["missing_candidates"]:
                ax.text(SLOTS[number], y, "not produced", ha="center", fontsize=5, color=COLORS["invalid"])
    for edge in graph["edges"]:
        if edge["target"] not in locations or edge["source"] not in locations:
            continue
        x1, y1 = locations[edge["source"]]
        x2, y2 = locations[edge["target"]]
        y1, y2 = y1 + 0.15, y2 - 0.15
        middle = (y1 + y2) / 2
        path = MplPath([(x1, y1), (x1, middle), (x2, middle), (x2, y2)],
                       [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        target = by_id[edge["target"]]
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=node_color(target),
                               linewidth=1.1 if target.get("rank") else 0.65,
                               alpha=1 if target.get("rank") else 0.5,
                               linestyle="--" if edge["carry"] else "-", zorder=1))
    for node in display:
        x, y, color = node["x"], node["plot_y"], node_color(node)
        selected = bool(node.get("rank"))
        box = FancyBboxPatch((x - 84, y - 0.145), 168, 0.29, boxstyle="round,pad=0.02,rounding_size=0.06",
                             facecolor=color if selected else "white", edgecolor=color, linewidth=0.85, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, node["label"], va="center", ha="center", fontsize=text_size - 0.4,
                color="white" if selected else color, zorder=4)
        if detailed and node["id"] not in parent_ids:
            ax.text(x, y + 0.31, metric_label(node), ha="center", fontsize=4.8, color="#66758a")
    ax.set_xlim(0, 1190)
    ax.set_ylim(end - start + 1.6, -0.6)
    ax.set_title(f"Steps {start:03d}-{end:03d}", fontsize=11, weight="bold", loc="left", pad=12)
    ax.axis("off")


def export_plots(graph: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    title = f"PREMiSE search tree | seed {graph['config']['seed']} | {graph['config']['mode']}"
    s = graph["stats"]
    summary = (f"{graph['steps']} encounters | {s['selection_updates']} successful selections | "
               f"{s.get('skipped_steps', 0)} skips | {s['invalid_candidates']} failed candidates")
    legend = "Green: next rank 1     Blue: next rank 2     Gray: pruned     Red: failed     Amber: skipped / carried"
    spans = [(start, min(start + 9, graph["steps"])) for start in range(1, graph["steps"] + 1, 10)]
    columns = min(4, len(spans))
    row_count = math.ceil(len(spans) / columns)
    fig, axes = plt.subplots(row_count, columns, figsize=(6.2 * columns, 7.8 * row_count), squeeze=False)
    for ax, span in zip(axes.flat, spans, strict=False):
        plot_panel(ax, graph, *span)
    for ax in list(axes.flat)[len(spans):]:
        ax.axis("off")
    fig.suptitle(f"{title}\n{summary}\n{legend}", fontsize=17, y=0.997, linespacing=1.65)
    fig.subplots_adjust(top=0.945, bottom=0.012, left=0.01, right=0.99, wspace=0.04, hspace=0.12)
    fig.savefig(output / "overview.png", dpi=140, facecolor="white")
    plt.close(fig)
    with PdfPages(output / "search_tree.pdf") as pdf:
        for start, end in spans:
            fig, ax = plt.subplots(figsize=(13, 11))
            plot_panel(ax, graph, start, end, detailed=True)
            fig.suptitle(f"{title}\n{summary}", fontsize=13, y=0.99)
            fig.text(0.05, 0.025, legend + "\nA: alignment current/past; S: official score; PCT: legacy surrogate.",
                     fontsize=8, color="#536378")
            fig.subplots_adjust(top=0.90, bottom=0.07, left=0.02, right=0.98)
            pdf.savefig(fig, facecolor="white")
            if start == 1:
                fig.savefig(output / "first_10_steps.png", dpi=150, facecolor="white")
            plt.close(fig)


def export_report(graph: dict, output: Path) -> None:
    s = graph["stats"]
    warning = (
        "This run does not satisfy 130 completely error-free updates. "
        f"There are {s.get('skipped_steps', 0)} skipped steps and "
        f"{s['steps_with_branch_errors']} steps with recorded branch errors."
    )
    if graph["source_order_length"] != graph["steps"]:
        warning += f" This is the first {graph['steps']} steps of a {graph['source_order_length']}-task plan."
    if graph["metric_type"] == "legacy_pct_surrogate":
        warning += " This older protocol includes SWE-bench and uses legacy PCT surrogate performance."
    lines = [
        "# Recorded PREMiSE search tree", "", warning, "",
        f"Source: `{graph['source_run']}`", "",
        f"Seed {graph['config']['seed']}, {graph['config']['mode']}, model `{graph['config']['model']}`, "
        f"sampled past count {graph['config']['past_count']}.", "",
        "| Measure | Count |", "| --- | ---: |",
        f"| Training encounters plotted | {graph['steps']} |",
        f"| Successful two-candidate beam selections | {s['selection_updates']} |",
        f"| Skipped updates | {s.get('skipped_steps', 0)} |",
        f"| Selections with no recorded branch errors | {s['error_free_selection_updates']} |",
        f"| Steps with branch errors | {s['steps_with_branch_errors']} |",
        f"| Generated candidate records | {s['candidates']} |",
        f"| Invalid candidate records | {s['invalid_candidates']} |",
        f"| Candidate slots not produced | {s['missing_candidates']} |",
        f"| Selected candidates with no applied operations | {s['selected_candidates_without_operations']} |",
        f"| Primary selections with applied operations | {s['primary_selections_with_operations']} |",
        f"| Selections taking both children from one parent | {s['same_parent_selections']} |", "",
        "A successful selection is a beam replacement, including a selected candidate with no mutation operations. "
        "It is not a claim that every candidate completed without error or that playbook text changed at every step.",
        "",
        "The forest starts with two initial parent slots. Each candidate is connected to its actual recorded "
        "parent rank, resolved through the preceding selection. Green and blue nodes are the next ranks 1 and 2; "
        "gray nodes terminate. Red nodes failed. Amber skip nodes carry both preceding playbooks unchanged. "
        "The two selected candidates can both descend from the same parent. Ten-step panels repeat their "
        "incoming beam nodes for readability; the continuous SVG contains each node once.", "",
        "Candidate details retain reflection spans, mutation operations, current/past objectives, selection "
        "fronts, and recorded errors. Official benchmark scores are not interchangeable with alignment scores "
        "or legacy PCT surrogate scores. The visualization reads existing artifacts and makes no model calls.", "",
        "| Skipped step | Benchmark | Task |", "| --- | --- | --- |",
    ]
    for row in graph["rows"]:
        if row["skipped"]:
            lines.append(f"| {row['iteration']} | {row['benchmark']} | {row['task_id']} |")
    lines.extend(["", "Source iteration SHA-256 hashes are in `graph.json`.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=130)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    graph = build_graph(args.run, args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = make_svg(graph)
    (args.output_dir / "search_tree.svg").write_text(svg, encoding="utf-8")
    (args.output_dir / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    template = Path(__file__).with_name("search_tree_template.html").read_text(encoding="utf-8")
    graph_json = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("__TREE_SVG__", svg).replace("__GRAPH_JSON__", graph_json)
    (args.output_dir / "search_tree.html").write_text(page, encoding="utf-8")
    export_report(graph, args.output_dir)
    export_plots(graph, args.output_dir)
    print(json.dumps({"output": str(args.output_dir.resolve()), "stats": graph["stats"]}, indent=2))


if __name__ == "__main__":
    main()
