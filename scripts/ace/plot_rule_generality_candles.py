# SPDX-License-Identifier: Apache-2.0
"""Plot context/rule candles and inspect Tukey outliers from saved fixed-reference scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

MODELS = ("qwen38", "qwen36", "qwen35", "gemma26")
MODES = ("thinking", "nonthinking")
STRATEGIES = ("best", "random", "worst")
LABELS = {"qwen38": "Qwen3.8\n27B", "qwen36": "Qwen3.6\n35B-A3B", "qwen35": "Qwen3.5\n9B",
          "gemma26": "Gemma4\n26B-A4B"}
COLORS = {"best": "#197d62", "random": "#8290a3", "worst": "#c25443"}


def tukey_stats(values):
    """Linear sample quartiles; strict exterior of Q1/Q3 +/-1.5 IQR, inclusive whisker endpoints."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("A nonempty finite score sample is required")
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    lower = values < low
    upper = values > high
    inside = values[~(lower | upper)]
    return {"n": len(values), "mean": float(values.mean()), "population_std": float(values.std()),
            "q1": float(q1), "median": float(median), "q3": float(q3),
            "lower_fence": float(low), "upper_fence": float(high),
            "lower_whisker": float(inside.min()), "upper_whisker": float(inside.max()),
            "low_count": int(lower.sum()), "high_count": int(upper.sum()),
            "outlier_count": int(lower.sum() + upper.sum()),
            "outlier_percent": float(100 * (lower.sum() + upper.sum()) / len(values))}


def build_groups(selections, contexts, rules):
    """A saved context and each of its rules count once within a strategy, even when selected by several seeds."""
    selected = defaultdict(dict)
    for row in selections:
        key = row["model"], row["mode"], row["strategy"]
        cid = row["context_id"]
        expected = contexts[cid]
        assert (expected["model"], expected["mode"], expected["task_id"]) == (
            row["model"], row["mode"], row["task_id"])
        assert expected["context_mean_cosine"] == row["context_mean_cosine"]
        selected[key].setdefault(cid, set()).add(row["seed"])
    by_context = defaultdict(list)
    for row in rules:
        by_context[row["context_id"]].append(row)
    groups = {}
    for key, chosen in selected.items():
        model, mode, strategy = key
        context_items, rule_items = [], []
        for cid, seeds in sorted(chosen.items()):
            context = contexts[cid]
            common = {"model": model, "mode": mode, "strategy": strategy, "context_id": cid,
                      "task_id": context["task_id"], "source_repeat": context["source_repeat"],
                      "selected_seeds": ";".join(sorted(seeds))}
            if context["context_mean_cosine"] != "":
                context_items.append({**common, "item_id": cid,
                                      "score": float(context["context_mean_cosine"])})
            assert len(by_context[cid]) == int(context["rules"])
            for rule in by_context[cid]:
                rule_items.append({**common, "item_id": rule["rule_id"],
                                   "score": float(rule["mean_context_max_cosine"]), "rule_text": rule["rule_text"]})
        assert len({r["item_id"] for r in rule_items}) == len(rule_items)
        groups[key] = {"context": context_items, "rule": rule_items, "selected_unique_contexts": len(chosen),
                       "empty_unique_contexts": len(chosen) - len(context_items)}
    return groups


def csv_file(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(groups):
    summaries, outliers = [], {"context": [], "rule": []}
    context_tails = {}
    for unit in ("context", "rule"):
        for key in sorted(groups, key=lambda k: (MODES.index(k[1]), MODELS.index(k[0]), STRATEGIES.index(k[2]))):
            model, mode, strategy = key
            items = groups[key][unit]
            stats = tukey_stats([r["score"] for r in items])
            summaries.append({"unit": unit, "model": model, "mode": mode, "strategy": strategy, **stats,
                              "selected_unique_contexts": groups[key]["selected_unique_contexts"],
                              "empty_unique_contexts": groups[key]["empty_unique_contexts"]})
            for row in items:
                tail = "low" if row["score"] < stats["lower_fence"] else (
                    "high" if row["score"] > stats["upper_fence"] else None)
                if tail is None:
                    continue
                detail = {"unit": unit, **row, "tail": tail, "lower_fence": stats["lower_fence"],
                          "upper_fence": stats["upper_fence"],
                          "distance_outside_fence": stats["lower_fence"] - row["score"] if tail == "low"
                          else row["score"] - stats["upper_fence"]}
                if unit == "context":
                    context_tails[key, row["context_id"]] = tail
                else:
                    detail["parent_context_outlier_tail"] = context_tails.get((key, row["context_id"]), "none")
                outliers[unit].append(detail)
    return summaries, outliers


def plot(groups, summaries, unit, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    lookup = {(s["model"], s["mode"], s["strategy"]): s for s in summaries if s["unit"] == unit}
    fig, axes = plt.subplots(1, 2, figsize=(24, 9), sharey=True)
    all_values = [r["score"] for group in groups.values() for r in group[unit]]
    ymin = max(-1, np.floor((min(all_values) - .035) * 20) / 20)
    ymax = min(1, np.ceil((max(all_values) + .035) * 20) / 20)
    centers = np.arange(4) * 1.5
    for ax, mode in zip(axes, MODES, strict=True):
        for index, model in enumerate(MODELS):
            for offset, strategy in zip((-.32, 0, .32), STRATEGIES, strict=True):
                key = model, mode, strategy
                s = lookup[key]
                x, width, color = centers[index] + offset, .22, COLORS[strategy]
                ax.plot([x, x], [s["lower_whisker"], s["upper_whisker"]], color=color, lw=1.3, zorder=2)
                for y in (s["lower_whisker"], s["upper_whisker"]):
                    ax.plot([x - width * .35, x + width * .35], [y, y], color=color, lw=1.3)
                ax.add_patch(Rectangle((x - width / 2, s["q1"]), width, max(s["q3"] - s["q1"], .00002),
                                       facecolor=color, edgecolor=color, alpha=.4, lw=1.5, zorder=3))
                ax.plot([x - width / 2, x + width / 2], [s["median"], s["median"]],
                        color="#243744", lw=1.8, zorder=4)
                ax.scatter([x], [s["mean"]], marker="D", s=15, color="#172b39", zorder=5)
                values = np.array([r["score"] for r in groups[key][unit]])
                exterior = values[(values < s["lower_fence"]) | (values > s["upper_fence"])]
                seed = int(hashlib.sha256(f"{unit}/{key}".encode()).hexdigest()[:8], 16)
                jitter = np.random.default_rng(seed).uniform(-.07, .07, size=len(exterior))
                ax.scatter(x + jitter, exterior, color=color, s=14 if unit == "rule" else 20,
                           alpha=.65, linewidths=.25, edgecolors="white", zorder=4)
                ax.text(x, -.145, f"n={s['n']}\n↓{s['low_count']}  ↑{s['high_count']}",
                        transform=ax.get_xaxis_transform(), ha="center", va="top", color=color, fontsize=9)
        ax.set_xticks(centers, [LABELS[m] for m in MODELS], fontsize=12)
        ax.set_xlim(-.8, centers[-1] + .8)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{'Thinking' if mode == 'thinking' else 'Non-thinking'} · w/o GT", fontsize=17, pad=14)
        ax.grid(axis="y", color="#dae1e5", alpha=.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=11)
    axes[0].set_ylabel("Context mean of rule commonality" if unit == "context" else
                       "Per-rule mean of per-other-context maximum cosine", fontsize=13)
    fig.suptitle(f"AppWorld · {'Context' if unit == 'context' else 'Rule'} commonality distributions",
                 fontsize=23, y=.98)
    fig.text(.5, .931, "Body: Q1-Q3 · Median line · Whiskers: last observations within 1.5*IQR · All outliers shown",
             ha="center", fontsize=13)
    handles = [Patch(facecolor=COLORS[s], edgecolor=COLORS[s], alpha=.4, label=s.title()) for s in STRATEGIES]
    handles += [Line2D([], [], marker="D", color="#172b39", ls="none", markersize=5, label="Mean")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .90), ncol=4, frameon=False, fontsize=12)
    fig.text(.5, .072, "Under each candle: n unique observations · ↓ lower outlier count · ↑ upper outlier count",
             ha="center", fontsize=12)
    fig.text(.5, .032, "Deduplicated across selection seeds within each strategy; strategies can share observations. "
             "Same frozen reference as previous analysis; empty contexts excluded.", ha="center", fontsize=10,
             color="#586875")
    fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.27, wspace=.085)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{unit}_commonality_candles_outlier_counts.{suffix}", dpi=180)
    plt.close(fig)


def render_rules(output, rows):
    """An offline, searchable full-text artifact; never changes the blinded live review UI."""
    unique = {}
    for row in rows:
        item = unique.setdefault(row["item_id"], {"row": row, "groups": [], "seeds": set()})
        assert item["row"]["score"] == row["score"]
        item["groups"].append(f"{row['strategy']}:{row['tail']}")
        item["seeds"].update(row["selected_seeds"].split(";"))
    cards = []
    for item in sorted(unique.values(), key=lambda x: x["row"]["score"]):
        r, groups = item["row"], item["groups"]
        tag = f"{r['model']} · {r['mode']} · {', '.join(groups)}"
        cards.append(f'<details class="card"><summary><b>{r["score"]:.4f}</b> '
                     f'{html.escape(tag)} · Task {html.escape(r["task_id"])} '
                     f'· r{html.escape(r["source_repeat"])}</summary>'
                     f'<pre>{html.escape(r["rule_text"])}</pre>'
                     f'<small>rule ID {html.escape(r["item_id"])} · '
                     f'selected seeds {html.escape(";".join(sorted(item["seeds"])))}</small></details>')
    report = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Rule outlier inspection</title>
<style>body{max-width:1250px;margin:35px auto;padding:0 20px;font:15px/1.7 system-ui;color:#263642;background:#f5f7f6}
input{width:100%;padding:12px;border:1px solid #bccbc2;border-radius:6px;font:inherit;box-sizing:border-box}
.card{background:white;border:1px solid #dce4df;border-radius:8px;padding:12px 17px;margin:12px 0}
summary{cursor:pointer}b{color:#176850;margin-right:10px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.8 monospace}
small{color:#7b8782;overflow-wrap:anywhere}[hidden]{display:none!important}</style>
<h1>Rule 유사도 아웃라이어 원문</h1>
<p>모델·모드·선정 전략별 rule 점수 분포에서 Q1-1.5*IQR 미만은 low, Q3+1.5*IQR 초과는 high.
동일 rule이 여러 전략에서 선정되면 각 그룹 판정을 함께 표시했습니다.
낮은/높은 의미 유사도이며 내용 오류 판정은 아닙니다.</p>
<input id="search" placeholder="모델, thinking / nonthinking, best / random / worst, low / high, Task, 원문 검색">
<p id="count"></p><div id="cards">''' + "".join(cards) + '''</div><script>
const cards=[...document.querySelectorAll('.card')];
function filter(){const q=document.getElementById('search').value.toLowerCase().trim().split(/\\s+/).filter(Boolean);
let count=0;for(const card of cards){card.hidden=!q.every(w=>card.textContent.toLowerCase().includes(w));
if(!card.hidden)count++;}
document.getElementById('count').textContent=count+' unique outlier rules / '+cards.length;}
document.getElementById('search').addEventListener('input',filter);filter();</script></html>'''
    (output / "outlier_rules.html").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read_csv(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return list(csv.DictReader(io.StringIO(raw.decode())))

    selections = read_csv(args.source / "selected_context_scores.csv")
    contexts = {r["context_id"]: r for r in read_csv(args.source / "all_context_scores.csv")}
    rules = []
    for mode in MODES:
        for model in MODELS:
            rules.extend(read_csv(args.source / mode / model / "rule_scores.csv"))
    assert len(selections) == 3600 and len(contexts) == 1610 and len(rules) == 7605
    groups = build_groups(selections, contexts, rules)
    assert len(groups) == 24
    summaries, outliers = analyze(groups)
    csv_file(args.output / "outlier_counts_and_thresholds.csv", summaries)
    for unit in ("context", "rule"):
        csv_file(args.output / f"{unit}_outliers.csv", outliers[unit])
        plot(groups, summaries, unit, args.output)
    render_rules(args.output, outliers["rule"])
    receipt = {"generated_at": datetime.now(UTC).isoformat(), "source": str(args.source.resolve()),
               "unit_definitions": {"context": "Equal mean of rule scores in one unique selected context",
                                    "rule": "One unique rule_id from one unique source context"},
               "outlier_definition": "Strict below Q1-1.5IQR or above Q3+1.5IQR; linear sample quartiles; "
                                     "computed separately within each model/mode/strategy/unit.",
               "deduplication": "Collapse repeated selection across seeds per strategy, using context_id/rule_id; "
                                "the same observation may be shared across strategies.",
               "outlier_counts": {unit: {"low_flags": sum(r["tail"] == "low" for r in outliers[unit]),
                                          "high_flags": sum(r["tail"] == "high" for r in outliers[unit]),
                                          "unique_observations": len({r["item_id"] for r in outliers[unit]})}
                                  for unit in ("context", "rule")},
               "rule_outlier_flags_in_nonoutlier_context": sum(r["parent_context_outlier_tail"] == "none"
                                                              for r in outliers["rule"]),
               "source_sha256": sources, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "new_embedding_calls": 0, "new_cosine_computations": 0}
    (args.output / "provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("outlier_counts", "rule_outlier_flags_in_nonoutlier_context")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
