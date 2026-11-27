# SPDX-License-Identifier: Apache-2.0
"""Add rule-gated completion counterfactual bars without changing recorded evaluations."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent
SPEC = importlib.util.spec_from_file_location("original_four_panel_plot", ORIGINAL / "plot_model_strategy_bars.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
SOURCES = {}

# Manual review of actual final Contexts; rule IDs are not inferred from failed answers.
# Task-specific advice is applied only within its stated subject, not to every task in the Context.
REVIEWS = {
    "qwen35/thinking/without_gt/best/42": (["misc-00056", "misc-00086"], "unrequested_payload"),
    "qwen35/thinking/without_gt/best/43": (["misc-00145"], "unrequested_payload"),
    "qwen35/thinking/with_gt/best/44": (["err-00022"], "shopping"),
    "qwen35/thinking/without_gt/best_advantage/44": (["str-00113"], "created_order_id"),
    "qwen35/nonthinking/without_gt/best/42": (["str-00056"], "unrequested_payload"),
    "qwen35/nonthinking/without_gt/best/44": (["err-00045"], "unrequested_payload"),
    "qwen35/nonthinking/without_gt/best_advantage/43": (["err-00083", "misc-00031"], "unrequested_payload"),
    "qwen35/nonthinking/without_gt/best_advantage/44": (["misc-00036", "misc-00054"], "unrequested_payload"),
    "oss20/high/with_gt/best/42": (["err-00142", "err-00199"], "password_or_sms_null_literal"),
    "oss120/low/without_gt/best/42": (["str-00136"], "expired_cards_summary"),
}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def csv_rows(path):
    SOURCES[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as stream:
        return list(csv.DictReader(stream))


def export_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def completion_state(path):
    """Read saved supervisor SQLite delta rows as data; never execute SQL or model code."""
    if not path.exists():
        return {"completed": False, "answer": None, "evidence": "missing_saved_supervisor_state"}
    payload = path.read_bytes()
    SOURCES[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
    state = {}
    for line in payload.decode().splitlines():
        sql, params, _ = json.loads(line)
        matched = re.match(r"UPDATE\s+tasks\s+SET\s+(.*?)\s+WHERE", sql, re.I)
        if not matched:
            continue
        columns = [part.split("=", 1)[0].strip().strip('"') for part in matched[1].split(",")]
        state.update(zip(columns, params, strict=False))
    answer = state.get("answer")
    if isinstance(answer, str):
        answer = json.loads(answer)
    return {"completed": state.get("status") == "success", "answer": answer,
            "evidence": "saved_task_status_and_answer", "state_path": str(path.resolve())}


def in_scope(scope, task, answer, tracker):
    shopping = bool(re.search(r"\bbuy\b|\bplace (?:a |an |the )?(?:new )?order\b", task, re.I))
    if scope == "unrequested_payload":
        return True
    if scope == "shopping":
        return shopping
    if scope == "created_order_id":
        order_added = any("amazon.Order" in r["requirement"] and "added" in r["requirement"]
                          for r in tracker["passes"])
        return shopping and str(answer).isdigit() and order_added
    if scope == "password_or_sms_null_literal":
        return answer == "null" and bool(re.search(r"password|text|sms", task, re.I))
    if scope == "expired_cards_summary":
        return bool(re.search(r"expired.*(?:card|payment)|(?:card|payment).*expired", task, re.I))
    raise ValueError(scope)


def calculate():
    original_rows = csv_rows(ORIGINAL / "metrics_by_seed.csv")
    lookup = {(r["model"], r["mode"], r["arm"], r["strategy"], int(r["seed"])): r for r in original_rows}
    contexts = read(OUT / "all_completion_rules.json")
    audit, reviewed, results = [], [], []
    for context in contexts:
        model, mode, arm, strategy, seed = (context[k] for k in ("model", "mode", "arm", "strategy", "seed"))
        key = "/".join([model, mode, arm, strategy, str(seed)])
        directory = Path(context["cohort"])
        path = (directory / f"seed{seed}/train_final_results.json" if strategy == "best" else
                directory.parents[1] / "evaluations" / mode / context["cid"] / "train_final_results.json")
        rows = [r for r in read(path) if r["condition_id"] == context["cid"]]
        assert len(rows) == 50
        BASE.DATA.validate(rows)
        checkpoint = Path(context["checkpoint"])
        raw_context = checkpoint.read_bytes()
        context_hash = hashlib.sha256(raw_context).hexdigest()
        SOURCES[str(checkpoint)] = context_hash
        assert {r["initial_context_sha256"] for r in rows} == {context_hash}
        ids, scope = REVIEWS.get(key, ([], "none"))
        evidence = [line for line in raw_context.decode().splitlines()
                    if any(line.startswith(f"[{rid}]") for rid in ids)]
        assert len(evidence) == len(ids), (key, ids)
        reviewed.append({"context": key, "checkpoint": str(checkpoint), "checkpoint_sha256": context_hash,
                         "eligible_rule_ids": ids, "scope": scope, "rule_quotes": evidence,
                         "review": "explicit_conflicting_payload_rule" if ids else
                                   "no_explicit_applicable_rule; unrelated bad workflows do not qualify"})
        original = BASE.DATA.metrics(rows)
        corrected_pct, corrected_tgc, changed = [], [], []
        for row in rows:
            old_pct = float(BASE.DATA.exact_reward(row))
            pct, tgc = old_pct, int(row["success"])
            score_path = Path(row["session_dir"]) / "benchmark/scores.json"
            if not row.get("budget_failure") and score_path.exists():
                tracker = read(score_path)["test_tracker"]
                assert len(tracker["passes"]) == row["passed_tests"]
                assert tracker["num_tests"] == row["total_tests"]
                failures = [r for r in tracker["failures"] if r["requirement"].strip() == "assert answers match."]
                if failures:
                    assert len(failures) == 1
                    null_target = bool(re.search(r'''==\s*(['"])null\1\s*$''', failures[0]["trace"]))
                    state = completion_state(Path(row["session_dir"]) / "benchmark/task_output/dbs/supervisor.jsonl")
                    reason = "wrong_answer_content_or_missing_required_value"
                    if null_target:
                        if not state["completed"]:
                            reason = "no_executed_completion; no synthetic_finish_added"
                        elif state["answer"] is None:
                            reason = "no_extra_answer_proven"
                        elif not ids:
                            reason = "no_explicit_conflicting_payload_rule"
                        else:
                            capture = read(Path(row["capture_path"]))
                            assert capture["playbook"] == raw_context.decode()
                            if not in_scope(scope, capture["task"], state["answer"], tracker):
                                reason = "wrong_rule_is_outside_this_task_or_payload_scope"
                            else:
                                reason = "omit_rule_induced_extra_answer_only"
                                pct = (row["passed_tests"] + 1) / row["total_tests"]
                                tgc = int(len(tracker["failures"]) == 1)
                                changed.append(row["task_id"])
                    audit.append({"model": model, "mode": mode, "arm": arm, "strategy": strategy, "seed": seed,
                                  "task_id": row["task_id"], "rollout_id": row["id"], "reason": reason,
                                  "null_target": null_target, "completion_executed": state["completed"],
                                  "submitted_answer": json.dumps(state["answer"], ensure_ascii=False),
                                  "rule_ids": ",".join(ids), "original_pct": old_pct, "counterfactual_pct": pct,
                                  "original_tgc": int(row["success"]), "counterfactual_tgc": tgc,
                                  "other_failed_tests": len(tracker["failures"]) - 1,
                                  "official_score_path": str(score_path)})
            assert old_pct <= pct <= 1
            corrected_pct.append(pct)
            corrected_tgc.append(tgc)
        original_csv = lookup[model, mode, arm, strategy, seed]
        assert abs(original["pct"] - float(original_csv["accumulated_pct"])) < 1e-12
        results.append({"model": model, "mode": mode, "arm": arm, "strategy": strategy, "seed": seed,
                        "original_pct": original["pct"], "counterfactual_pct": mean(corrected_pct),
                        "original_tgc": original["tgc"], "counterfactual_tgc": mean(corrected_tgc),
                        "corrected_cases": len(changed), "corrected_task_ids": ",".join(changed)})
    summary = {}
    for key in {(r["model"], r["mode"], r["arm"], r["strategy"]) for r in results}:
        rows = [r for r in results if (r["model"], r["mode"], r["arm"], r["strategy"]) == key]
        assert len(rows) == 3
        record = dict(zip(("model", "mode", "arm", "strategy"), key, strict=True))
        for field in ("original_pct", "counterfactual_pct", "original_tgc", "counterfactual_tgc"):
            record[field + "_mean"] = mean(r[field] for r in rows)
            record[field + "_std"] = stdev(r[field] for r in rows)
        record["corrected_cases"] = sum(r["corrected_cases"] for r in rows)
        summary[key] = record
    (OUT / "reviewed_completion_rules.json").write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n")
    export_csv("case_audit.csv", audit)
    export_csv("counterfactual_by_seed.csv", results)
    export_csv("counterfactual_summary.csv", sorted(summary.values(), key=lambda r: (
        r["model"], r["mode"], r["arm"], r["strategy"])))
    return summary, audit


def plot(summary):
    initial = {(r["model"], r["mode"]): {**r, "pct": float(r["pct"]), "pass_at_5": float(r["pass_at_5"])}
               for r in csv_rows(ORIGINAL / "initial_metrics.csv")}
    records = csv_rows(ORIGINAL / "metrics_mean_std.csv")
    original = {}
    for row in records:
        converted = {k: float(v) if k.endswith(("_mean", "_std")) else v for k, v in row.items()}
        original[row["model"], row["mode"], row["arm"], row["strategy"]] = converted
    BASE.OUT, BASE.WIDTH = OUT, 0.45
    close = BASE.plt.close
    BASE.plt.close = lambda *args, **kwargs: None
    try:
        BASE.draw(initial, original)
        fig = BASE.plt.gcf()
    finally:
        BASE.plt.close = close
    extra = []
    for row in range(2):
        for col, arm in enumerate(BASE.ARMS):
            ax = fig.axes[row * 2 + col]
            for index, model in enumerate(BASE.MODELS):
                if model not in ("qwen35", "oss120", "oss20"):
                    continue
                mode = tuple(BASE.DATA.MODEL_RUNS[model])[row]
                for strategy in ("best", "best_advantage"):
                    record = summary[model, mode, arm, strategy]
                    center = BASE.TICK_OFFSETS[BASE.STRATEGIES.index(strategy) + 2]
                    x = index * BASE.STRIDE + center + 1.5 * BASE.WIDTH
                    value, std = (record[f"counterfactual_pct_{v}"] * 100 for v in ("mean", "std"))
                    BASE.bar(ax, x, value, "#e5a635", std=std)
                    ax.texts[-1].set_text(f"{value:.1f}*")
                    extra.append({"row": row, "arm": arm, "model": model, "mode": mode, "category": strategy,
                                  "variant": "completion_counterfactual", "x": x, "mean_percent": value,
                                  "std_percent": std, "n": 3})
    fig._suptitle.set_text("AppWorld Train · Context-rule-gated completion counterfactual")
    for text in fig.texts:
        if text.get_text().startswith("50 matched training tasks"):
            text.set_text("Original bars retained · Gold bar added after Best / Best advantage "
                          "for Qwen3.5, GPT-OSS-120B and GPT-OSS-20B")
        if text.get_text().startswith("Initial and pass@5 are shared"):
            text.set_text("* Estimated accumulated PCT: omit an unnecessary answer only when the actual Context "
                          "has an explicit conflicting rule within the task's scope. Other failed tests remain failed.")
    fig.text(0.5, 0.013, "No new actor calls. No missing finish is inserted. "
             "Wrong factual answers, formatting errors and unrelated failures are unchanged. "
             "This is a saved-state counterfactual, not a rerun with a repaired Context.",
             ha="center", fontsize=11.5, color="#536176")
    for legend in list(fig.legends):
        legend.remove()
    handles = [Patch(facecolor=BASE.COLORS["initial"], label="Initial PCT"),
               Patch(facecolor=BASE.COLORS["pass_at_5"], label="Initial pass@5 (official TGC)"),
               Patch(facecolor=BASE.tinted("#657488"), edgecolor="#657488", hatch="///", label="Independent"),
               Patch(facecolor="#657488", label="Accumulated"),
               Patch(facecolor="#e5a635", label="Completion-rule counterfactual (estimated)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.933), ncol=5,
               frameon=False, fontsize=12, handlelength=2.4, columnspacing=2.0)
    name = "model_strategy_with_completion_counterfactual_pct_4panel"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=90, facecolor="white")
    BASE.plt.close(fig)
    original_name = "model_strategy_initial_pass5_independent_accumulated_pct_4panel"
    for extension in ("png", "pdf", "svg", "_preview.png"):
        file = OUT / (original_name + (extension if extension.startswith("_") else "." + extension))
        file.unlink()
    existing = csv_rows(OUT / "plotted_bars.csv")
    assert len(extra) == 24 and len(existing) == 240
    export_csv("plotted_bars.csv", existing + extra)


def main():
    preserved = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ORIGINAL.glob("*4panel.*")}
    summary, audit = calculate()
    plot(summary)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in preserved.items())
    provenance = {"generated_at": datetime.now(UTC).isoformat(), "estimated": True,
                  "new_actor_calls": 0, "target_seed_contexts": 72, "new_bars": 24, "total_bars": 264,
                  "corrected_cases": sum(r["reason"] == "omit_rule_induced_extra_answer_only" for r in audit),
                  "reason_counts": dict(Counter(r["reason"] for r in audit)),
                  "method": "Manual rule review + official null target + saved executed completion with non-null "
                            "answer + rule/task scope match. Convert only the standalone answers-match test to pass; "
                            "keep every other outcome, action, context and baseline unchanged.",
                  "limitation": "The rule's causal effect is not established. Gold bars are local termination "
                                "counterfactuals and do not predict a whole trajectory rerun with rewritten context.",
                  "manual_rules": REVIEWS, "source_sha256": SOURCES,
                  "original_figure_hashes_unchanged": preserved,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k not in (
        "source_sha256", "manual_rules", "original_figure_hashes_unchanged")}, ensure_ascii=False, indent=2))
    for row in sorted(summary.values(), key=lambda r: (r["model"], r["mode"], r["arm"], r["strategy"])):
        if row["corrected_cases"]:
            print(row["model"], row["mode"], row["arm"], row["strategy"], row["corrected_cases"],
                  round(row["original_pct_mean"] * 100, 2), "->", round(row["counterfactual_pct_mean"] * 100, 2))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
