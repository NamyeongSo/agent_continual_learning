# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Measure repeated-action F1 noise and learned-context effects on identical prefixes."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import os
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.agents.premise.llm import configure_llm_slots  # noqa: E402
from exgentic.agents.premise.pipeline import write_json  # noqa: E402
from exgentic.agents.premise.span_policy_shift import (  # noqa: E402
    SpanPolicyShiftEvaluator,
    canonical,
    normalize_action,
    strict_f1,
    terminal_parser,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_span(anchor, policy):
    request = {"messages": copy.deepcopy(anchor["messages"])}
    if anchor.get("tools"):
        request.update(tools=copy.deepcopy(anchor["tools"]), tool_choice="auto")
    span = anchor["policy_span"]
    message = request["messages"][span["message"]]
    content = message["content"] if span["part"] is None else message["content"][span["part"]]["text"]
    if hashlib.sha256(content[span["start"]:span["end"]].encode()).hexdigest() != span["sha256"]:
        raise ValueError("Frozen policy span changed")
    changed = content[:span["start"]] + policy + content[span["end"]:]
    if span["part"] is None:
        message["content"] = changed
    else:
        message["content"][span["part"]]["text"] = changed
    return request


def prepare(args):
    db = sqlite3.connect(f"file:{args.source / 'policy_shift.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rng, cases = random.Random(args.seed), []
    config = json.loads(args.model_call_config.read_text())
    settings = {k: copy.deepcopy(v) for k, v in config["model_settings"].items()
                if k not in ("timeout", "num_retries")}
    if args.model != config["model"] or args.temperature != settings["temperature"]:
        raise ValueError("Audit model and temperature must match the live model-call configuration")
    settings["model"] = args.model
    parser = args.source / "frozen/terminus_json_plain_parser.py"
    for benchmark in args.benchmarks.split(","):
        policies = db.execute(
            "SELECT p.point,t.content FROM policies p JOIN prompts t ON t.sha=p.prompt "
            "WHERE p.benchmark=? AND p.method=? AND p.seed=? ORDER BY p.point",
            (benchmark, args.method, args.seed),
        ).fetchall()
        if len(policies) < 2:
            raise ValueError(f"Missing distinct initial/learned policies for {benchmark}")
        contexts = {"initial": policies[0]["content"], "learned": policies[-1]["content"]}
        if contexts["initial"] == contexts["learned"]:
            raise ValueError("Initial and learned context bytes are identical")
        rows = db.execute(
            "SELECT a.* FROM anchors a JOIN anchor_owners o ON a.id=o.anchor "
            "WHERE a.benchmark=? AND a.seed=? AND o.method=? ORDER BY a.task,a.id",
            (benchmark, args.seed, args.method),
        ).fetchall()
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(dict(row))
        tasks = sorted(by_task)
        rng.shuffle(tasks)
        selected_tasks = 0
        for task in tasks:
            candidates = []
            for row in by_task[task]:
                path = Path(row["file"])
                if path.stat().st_size > args.max_history_bytes:
                    continue
                anchor = json.loads(path.read_text())
                step = sum(m["role"] == "assistant" for m in anchor["messages"])
                if step >= (30 if benchmark == "bfcl" else 40):
                    continue
                requests = {name: replace_span(anchor, content) for name, content in contexts.items()}
                # All non-context request fields must be byte-equivalent across arms.
                for request in requests.values():
                    request["generation_settings"] = copy.deepcopy(settings)
                baseline, changed = requests.values()
                message_index = anchor["policy_span"]["message"]
                for i, (a, b) in enumerate(zip(baseline["messages"], changed["messages"], strict=True)):
                    if i != message_index and a != b:
                        raise ValueError("A non-policy history message differs")
                if {k: v for k, v in baseline.items() if k != "messages"} != {
                    k: v for k, v in changed.items() if k != "messages"
                }:
                    raise ValueError("A non-context generation setting differs")
                candidates.append({
                    "id": row["id"], "benchmark": benchmark, "task": task, "action_index": step,
                    "source": str(path), "source_sha256": sha(path), "requests": requests,
                    "context_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in contexts.items()},
                    "context_points": [policies[0]["point"], policies[-1]["point"]],
                    **({"parser": {"path": str(parser), "sha256": sha(parser)}}
                       if benchmark == "terminalbench2" else {}),
                })
            if len(candidates) < args.prefixes_per_task:
                continue
            cases.extend(rng.sample(candidates, args.prefixes_per_task))
            selected_tasks += 1
            if selected_tasks == args.tasks:
                break
        if selected_tasks != args.tasks:
            raise ValueError(f"Not enough eligible tasks for {benchmark}")
    db.close()
    return {"protocol": "independent_matched_qwen38_live_reanchoring_v2", "model": args.model,
            "temperature": args.temperature, "repeats": args.repeats, "seed": args.seed,
            "source": str(args.source), "method": args.method, "cases": cases,
            "model_call_source": {"path": str(args.model_call_config), "sha256": sha(args.model_call_config)},
            "model_call_settings": config["model_settings"],
            "interpretation": "Conditional action agreement on historical prefixes; reference actions resampled "
                              "under the initial context using the exact same generation settings as probes. "
                              "No environment execution or new success measurement."}


def run_draw(evaluator, output, case, context, repeat):
    label = f"{case['benchmark']}_{case['id']}_{context}_{repeat:02d}"
    path = output / "draws" / f"{label}.json"
    if path.exists():
        return json.loads(path.read_text())
    row = {"case": case["id"], "benchmark": case["benchmark"], "task": case["task"],
           "context": context, "repeat": repeat}
    try:
        message, usage = evaluator._complete(
            case["requests"][context], benchmark=case["benchmark"], action_index=case["action_index"],
            log_path=output / "calls" / f"{label}.jsonl",
        )
        row.update(message=message, usage=usage)
        parser = terminal_parser(**case["parser"]) if "parser" in case else None
        actions = normalize_action(message, case["benchmark"], parser)
        row.update(status="completed", actions=actions)
    except Exception as exc:
        row.update(status="error", error_type=type(exc).__name__, error=str(exc),
                   usage=getattr(exc, "usage_records", row.get("usage", [])))
    write_json(path, row)
    return row


def equal_task_mean(rows, key):
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row[key])
    return fmean(fmean(values) for values in tasks.values())


def action_prf(predicted, reference, mode="strict"):
    """Score multiset overlap; loose compares names/types and ignores arguments."""
    if mode not in ("strict", "loose"):
        raise ValueError("Agreement mode must be strict or loose")
    pred = Counter(canonical(a) if mode == "strict" else a["name"] for a in predicted)
    ref = Counter(canonical(a) if mode == "strict" else a["name"] for a in reference)
    matched = sum((pred & ref).values())
    n_pred, n_ref = sum(pred.values()), sum(ref.values())
    return {
        "precision": matched / n_pred if n_pred else float(n_ref == 0),
        "recall": matched / n_ref if n_ref else float(n_pred == 0),
        "f1": 2 * matched / (n_pred + n_ref) if n_pred + n_ref else 1.0,
    }


def mean_prf(pairs, mode):
    scores = [action_prf(predicted, reference, mode) for predicted, reference in pairs]
    return {key: fmean(score[key] for score in scores) for key in ("precision", "recall", "f1")}


def case_agreement_metrics(base, learned, reference):
    result = {}
    for mode in ("strict", "loose"):
        fixed = {}
        for context, actions in (("initial", base), ("learned", learned)):
            scores = [action_prf(a, reference, mode) for a in actions]
            fixed[context] = {key: [s[key] for s in scores] for key in ("precision", "recall", "f1")}
        # Self comparisons have no privileged prediction/reference: average both directions.
        initial_pairs = [(a, b) for i, a in enumerate(base) for j, b in enumerate(base) if i != j]
        learned_pairs = [(a, b) for i, a in enumerate(learned) for j, b in enumerate(learned) if i != j]
        result[mode] = {
            "fixed_reference": fixed,
            "pairwise": {
                "initial_self": mean_prf(initial_pairs, mode),
                "learned_self": mean_prf(learned_pairs, mode),
                "cross_context": mean_prf(itertools.product(learned, base), mode),
            },
        }
    return result


def aggregate_agreement_metrics(rows, repeats):
    result = {}
    for mode in ("strict", "loose"):
        fixed, pairwise = {}, {}
        for context in ("initial", "learned"):
            fixed[context] = {}
            for metric in ("precision", "recall", "f1"):
                means = [equal_task_mean([
                    {"task": row["task"],
                     "score": row["agreement_metrics"][mode]["fixed_reference"][context][metric][k]}
                    for row in rows
                ], "score") for k in range(repeats)]
                fixed[context][metric] = {"mean": fmean(means), "sd": pstdev(means), "repeats": means,
                    "distribution": distribution([value for row in rows
                        for value in row["agreement_metrics"][mode]["fixed_reference"][context][metric]])}
        for comparison in ("initial_self", "learned_self", "cross_context"):
            pairwise[comparison] = {metric: equal_task_mean([
                {"task": row["task"], "score": row["agreement_metrics"][mode]["pairwise"][comparison][metric]}
                for row in rows
            ], "score") for metric in ("precision", "recall", "f1")}
        result[mode] = {"fixed_reference": fixed, "pairwise": pairwise}
    return result


def distribution(values):
    ordered = sorted(values)

    def quantile(q):
        index = (len(ordered) - 1) * q
        lower = int(index)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)

    return {"count": len(values), "mean": fmean(values), "sd": pstdev(values),
            "min": ordered[0], "p10": quantile(0.1), "median": quantile(0.5),
            "p90": quantile(0.9), "max": ordered[-1],
            "zero_fraction": fmean(v == 0 for v in values), "one_fraction": fmean(v == 1 for v in values)}


def content_signature(actions, benchmark):
    """A diagnostic for formatting/duration sensitivity, not semantic equivalence."""
    result = copy.deepcopy(actions)
    for action in result:
        arguments = action.get("arguments", {})
        if benchmark == "terminalbench2" and action["name"] == "terminal_command":
            arguments.pop("duration", None)
        if benchmark == "appworld":
            for key in ("code", "python_code"):
                if isinstance(arguments.get(key), str):
                    try:
                        arguments[key] = ast.dump(ast.parse(arguments[key]), include_attributes=False)
                    except SyntaxError:
                        pass
    return result


def analyze(prepared, draws):
    lookup = {(r["case"], r["context"], r["repeat"]): r for r in draws}
    cases, examples = [], []
    repeats = prepared["repeats"]
    for case in prepared["cases"]:
        reference = lookup[(case["id"], "initial", -1)]
        samples = {c: [lookup[(case["id"], c, k)] for k in range(repeats)] for c in ("initial", "learned")}
        if reference["status"] != "completed" or any(
            r["status"] != "completed" for group in samples.values() for r in group
        ):
            continue  # Common complete-case intersection, no zero filling.
        base, learned = ([r["actions"] for r in samples[c]] for c in ("initial", "learned"))
        within = list(itertools.combinations(base, 2))
        within_learned = list(itertools.combinations(learned, 2))
        between = list(itertools.product(base, learned))
        base_content = [content_signature(a, case["benchmark"]) for a in base]
        learned_content = [content_signature(a, case["benchmark"]) for a in learned]
        initial_f1 = [strict_f1(a, reference["actions"]) for a in base]
        learned_f1 = [strict_f1(a, reference["actions"]) for a in learned]
        row = {"id": case["id"], "benchmark": case["benchmark"], "task": case["task"],
               "initial_fixed_reference_f1": initial_f1, "learned_fixed_reference_f1": learned_f1,
               "same_context_f1": fmean(strict_f1(a, b) for a, b in within),
               "learned_same_context_f1": fmean(strict_f1(a, b) for a, b in within_learned),
               "changed_context_f1": fmean(strict_f1(a, b) for a, b in between),
               "same_context_exact_match": fmean(a == b for a, b in within),
               "learned_same_context_exact_match": fmean(a == b for a, b in within_learned),
               "changed_context_exact_match": fmean(a == b for a, b in between),
               "same_context_content_match": fmean(a == b for a, b in itertools.combinations(base_content, 2)),
               "changed_context_content_match": fmean(
                   a == b for a, b in itertools.product(base_content, learned_content)),
               "same_context_names_match": fmean([x["name"] for x in a] == [x["name"] for x in b]
                                                for a, b in within),
               "changed_context_names_match": fmean([x["name"] for x in a] == [x["name"] for x in b]
                                                   for a, b in between),
               "fixed_reference_repeat_sd": pstdev(initial_f1),
               "learned_fixed_reference_repeat_sd": pstdev(learned_f1),
               "initial_empty_action_fraction": fmean(not a for a in base),
               "learned_empty_action_fraction": fmean(not a for a in learned),
               "unique_initial_actions": len(set(map(canonical, base))),
               "unique_learned_actions": len(set(map(canonical, learned)))}
        row["agreement_metrics"] = case_agreement_metrics(base, learned, reference["actions"])
        row["context_gap"] = row["same_context_f1"] - row["changed_context_f1"]
        cases.append(row)
        for kind, pairs in (("same_context", within), ("changed_context", between)):
            pair = next(((a, b) for a, b in pairs if a != b), None)
            if pair:
                examples.append({"case": case["id"], "benchmark": case["benchmark"], "task": case["task"],
                                 "kind": kind, "action_a": pair[0], "action_b": pair[1],
                                 "strict_f1": strict_f1(*pair)})
    benchmarks = {}
    for benchmark in sorted({c["benchmark"] for c in prepared["cases"]}):
        rows = [r for r in cases if r["benchmark"] == benchmark]
        errors = Counter(r.get("error_type") for r in draws if r["benchmark"] == benchmark and r["status"] == "error")
        if not rows:
            benchmarks[benchmark] = {"complete_prefixes": 0, "errors": dict(errors),
                                     "expected_prefixes": sum(c["benchmark"] == benchmark for c in prepared["cases"])}
            continue
        metrics = {key: equal_task_mean(rows, key) for key in (
            "same_context_f1", "changed_context_f1", "context_gap", "same_context_exact_match",
            "learned_same_context_f1", "learned_same_context_exact_match", "learned_fixed_reference_repeat_sd",
            "changed_context_exact_match", "same_context_names_match", "changed_context_names_match",
            "same_context_content_match", "changed_context_content_match",
            "fixed_reference_repeat_sd", "unique_initial_actions", "unique_learned_actions",
            "initial_empty_action_fraction", "learned_empty_action_fraction",
        )}
        for context in ("initial", "learned"):
            replicate_means = [equal_task_mean([
                {"task": r["task"], "score": r[f"{context}_fixed_reference_f1"][k]} for r in rows
            ], "score") for k in range(repeats)]
            metrics[f"{context}_aggregate_f1_repeats"] = replicate_means
            metrics[f"{context}_aggregate_f1_mean"] = fmean(replicate_means)
            metrics[f"{context}_aggregate_f1_sd"] = pstdev(replicate_means)
            metrics[f"{context}_fixed_reference_f1_distribution"] = distribution(
                [value for row in rows for value in row[f"{context}_fixed_reference_f1"]]
            )
        metrics.update(complete_prefixes=len(rows), tasks=len({r["task"] for r in rows}), errors=dict(errors),
                       expected_prefixes=sum(c["benchmark"] == benchmark for c in prepared["cases"]),
                       zero_reference_f1_prefix_fraction=fmean(max(r["initial_fixed_reference_f1"]) == 0 for r in rows),
                       perfectly_repeated_prefix_fraction=fmean(r["unique_initial_actions"] == 1 for r in rows))
        metrics["agreement_metrics"] = aggregate_agreement_metrics(rows, repeats)
        benchmarks[benchmark] = metrics
    return {"model": prepared["model"], "temperature": prepared["temperature"], "repeats": repeats,
            "aggregation": "prefixes equally within task, tasks equally within benchmark; complete-case intersection",
            "agreement_definitions": {
                "strict": "Exact canonical function/action name and parsed arguments; multiset overlap with duplicates",
                "loose": "Function/action name only, arguments and contents ignored; multiset overlap with duplicates",
                "precision": "Matched atoms / predicted atoms", "recall": "Matched atoms / reference atoms",
                "fixed_reference_direction": (
                    "Prediction = initial/learned draw; reference = fixed initial-context draw"),
                "cross_context_direction": "Prediction = learned draw; reference = initial draw",
                "self_comparison_direction": "Both directions averaged over all distinct draw pairs",
                "macro_f1": "Average per-draw F1; not harmonic mean of aggregate precision and recall",
                "empty_actions": "Both empty: P=R=F1=1; exactly one empty: P=R=F1=0",
            },
            "benchmarks": benchmarks, "cases": cases, "action_examples": examples,
            "errors": sum(r["status"] == "error" for r in draws), "requests": len(draws)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--model-call-config", type=Path,
                        default=ROOT / "outputs/qwen38_terminalbench21_terminus2_timeout48_c10_20261113/config.json")
    parser.add_argument("--benchmarks", default="bfcl,appworld,terminalbench2")
    parser.add_argument("--method", default="peace_no_alignment_aud")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--prefixes-per-task", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-history-bytes", type=int, default=150000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Recompute summaries from saved draws without inference")
    args = parser.parse_args()
    if min(args.tasks, args.prefixes_per_task, args.workers) < 1 or args.repeats < 2:
        parser.error("Counts must be positive; repeat count must be at least two")
    args.source, args.output = args.source.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "prepared.json"
    prepared = json.loads(path.read_text()) if args.analyze_only else prepare(args)
    if path.exists() and json.loads(path.read_text()) != prepared:
        raise ValueError("Existing audit configuration differs")
    write_json(path, prepared)
    if args.prepare_only:
        print(f"Prepared {len(prepared['cases'])} fixed prefixes", flush=True)
        return
    if args.analyze_only:
        draws = [json.loads(p.read_text()) for p in (args.output / "draws").glob("*.json")]
        expected = len(prepared["cases"]) * (1 + 2 * args.repeats)
        if len(draws) != expected:
            raise ValueError(f"Audit is incomplete: {len(draws)}/{expected} draws")
        summary = analyze(prepared, draws)
        summary["analysis_source_sha256"] = sha(Path(__file__))
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary["benchmarks"], indent=2), flush=True)
        return
    configure_llm_slots(args.workers)
    settings = prepared["model_call_settings"]
    if settings.get("max_tokens") is None:
        os.environ["EXGENTIC_ACTOR_OUTPUT_UNLIMITED"] = "1"
    evaluator = SpanPolicyShiftEvaluator(model=args.model, base_url=args.base_url, output=args.output,
                                         temperature=args.temperature, match_source_settings=True,
                                         max_tokens=settings.get("max_tokens"),
                                         timeout=settings.get("timeout", 600),
                                         num_retries=settings.get("num_retries", 1))
    (args.output / "calls").mkdir(exist_ok=True)
    jobs = [(case, "initial", -1) for case in prepared["cases"]]
    jobs += [(case, context, repeat) for case in prepared["cases"]
             for context in ("initial", "learned") for repeat in range(args.repeats)]
    random.Random(args.seed).shuffle(jobs)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_draw, evaluator, args.output, *job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 20 == 0 or len(results) == len(jobs):
                write_json(args.output / "status.json", {"finished": len(results), "total": len(jobs),
                                                         "errors": sum(r["status"] == "error" for r in results)})
                print(f"draws {len(results)}/{len(jobs)}", flush=True)
    summary = analyze(prepared, results)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["benchmarks"], indent=2), flush=True)


if __name__ == "__main__":
    main()
