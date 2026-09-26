# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# --- ExGentic imports ---
from exgentic.interfaces.lib.api import evaluate
from exgentic.interfaces.registry import load_agent, load_benchmark
from exgentic.agents.ace.playbook_store import PlaybookStore
from exgentic.agents.ace.playbook_utils import get_playbook_stats
from exgentic.core.types import ModelSettings

from task_ordering import SEQUENTIAL_BENCHMARK_ORDER, get_unified_task_order, group_by_benchmark


BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "browsecompplus": {
        "bm_kwargs": {
            "searcher_type": "faiss",
            "include_get_document": True,
            "eval_model_id": "openai/gpt-5.4",
        },
        "agent_kwargs": {},
    },
    "swebench": {
        "bm_kwargs": {
            "subset": "princeton-nlp/SWE-bench_Verified",
        },
        "agent_kwargs": {},
    },
    "terminalbench2": {
        "bm_kwargs": {"subset": "2.0"},
        "agent_kwargs": {},
    },
    "appworld": {
        "bm_kwargs": {
            "subset": "test_challenge",
        },
        "agent_kwargs": {
            "enable_tool_shortlisting": True,
            "max_selected_tools": 30,
        },
    },
    "bfcl": {
        "bm_kwargs": {
            "subset": "multi_turn_base",
        },
        "agent_kwargs": {},
    },
    "tau2": {
        "bm_kwargs": {
            "subset": "telecom",
            "user_simulator_model": "openai/gpt-5.4",
        },
        "agent_kwargs": {},
    },
    "hle": {
        "bm_kwargs": {
            "judge_model": "openai/gpt-5.4",
            "runner": "direct",
        },
        "agent_kwargs": {},
    },
}

def extract_token_counts(cost_reports: dict) -> tuple[int, int]:
    """Extract total input/output tokens from cost_reports dict."""
    total_in, total_out = 0, 0
    for report in cost_reports.values():
        if isinstance(report, dict):
            total_in += report.get("input_tokens", 0)
            total_out += report.get("output_tokens", 0)
        elif hasattr(report, "input_tokens"):
            total_in += report.input_tokens
            total_out += report.output_tokens
    return total_in, total_out


def get_memory_tokens(mode: str, bm_slug: str) -> tuple[int, int]:
    stores = PlaybookStore.list_stores()
    if mode == "isolated":
        store = stores.get(f"ace_isolated_{bm_slug}")
    elif mode == "sequential":
        store = stores.get("ace_sequential_global")
    elif mode == "interleaved":
        store = stores.get("ace_interleaved_global")
    else:
        return 0, 0
    if store is None:
        return 0, 0
    stats = get_playbook_stats(store.playbook)
    mem_tokens = len(store.playbook) // 4
    return mem_tokens, stats["total_bullets"]



def record_online_metrics(
    metrics_path: Path,
    session_index: int,
    bm_slug: str,
    task_id: str,
    sr: Any,
    mode: str,
    seed: int,
    model: str,
    all_scores: list[float],
    bm_scores: dict[str, list[float]],
    training_time: bool = False,
    max_turns: int | None = None,
):
    score = sr.score if sr.score is not None else (1.0 if sr.success else 0.0)
    all_scores.append(score)
    bm_scores[bm_slug].append(score)

    input_tokens, output_tokens = extract_token_counts(sr.cost_reports)
    memory_tokens, playbook_bullets = get_memory_tokens(mode, bm_slug)

    record = {
        "session_index": session_index,
        "seed": seed,
        "mode": mode,
        "agent": "ace",
        "model": model,
        "benchmark_slug": bm_slug,
        "task_id": task_id,
        "score": score,
        "cumulative_avg_score": sum(all_scores) / len(all_scores),
        "benchmark_cumulative_avg_score": (
            sum(bm_scores[bm_slug]) / len(bm_scores[bm_slug])
        ),
        "steps": sr.steps,
        "action_count": sr.action_count,
        "agent_cost": sr.agent_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "memory_tokens": memory_tokens,
        "playbook_bullets": playbook_bullets,
        "execution_time": sr.execution_time,
        "status": sr.status.value if hasattr(sr.status, "value") else str(sr.status),
        "training_time": training_time,
        "max_turns": max_turns,
        "timestamp": datetime.now().isoformat(),
    }

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def resolve_max_turns(args: argparse.Namespace, benchmark_slug: str) -> int | None:
    """Use a benchmark-specific limit, falling back to the legacy global limit."""
    specific = getattr(args, f"{benchmark_slug}_max_turns", None)
    return specific if specific is not None else args.max_turns


def task_order_from_train_manifest(
    path: Path, benchmarks: list[str], seed: int, count: int, mode: str = "sequential"
) -> list[tuple[str, str]]:
    """Keep frozen training IDs fixed; vary order or interleaving by run seed."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest_count = manifest.get("train_count_per_benchmark")
    if not isinstance(manifest_count, int) or not 0 < count <= manifest_count:
        raise ValueError(f"Train manifest must have at least {count} tasks per benchmark")
    ordered_slugs = [slug for slug in SEQUENTIAL_BENCHMARK_ORDER if slug in benchmarks]
    ordered_slugs.extend(slug for slug in benchmarks if slug not in ordered_slugs)
    per_benchmark: dict[str, deque[str]] = {}
    for slug in ordered_slugs:
        split = manifest["benchmarks"][slug]
        train = list(split["train"])
        val = split["val"]
        if len(train) != manifest_count or len(set(train)) != manifest_count or set(train) & set(val):
            raise ValueError(f"Invalid train/val split for {slug}")
        order_seed = int(hashlib.md5(f"{seed}_{slug}".encode()).hexdigest(), 16) % (2**31)
        random.Random(order_seed).shuffle(train)
        per_benchmark[slug] = deque(train[:count])
    if mode == "sequential":
        return [(slug, task_id) for slug in ordered_slugs for task_id in per_benchmark[slug]]
    if mode == "interleaved":
        rng = random.Random(seed)
        task_order: list[tuple[str, str]] = []
        while any(per_benchmark.values()):
            slug = rng.choice(sorted(slug for slug, tasks in per_benchmark.items() if tasks))
            task_order.append((slug, per_benchmark[slug].popleft()))
        return task_order
    raise ValueError(f"Unsupported manifest training mode: {mode}")


def run_experiment(args):
    benchmarks_to_run = [s.strip() for s in args.benchmarks.split(",")]
    benchmark_max_turns = {
        slug: resolve_max_turns(args, slug)
        for slug in benchmarks_to_run
    }
    configs = {
        k: {**BENCHMARK_REGISTRY[k], "bm_kwargs": dict(BENCHMARK_REGISTRY[k]["bm_kwargs"])}
        for k in benchmarks_to_run
    }
    if args.appworld_max_interactions is not None and "appworld" in configs:
        configs["appworld"]["bm_kwargs"]["max_interactions"] = args.appworld_max_interactions
    if args.swebench_max_interactions is not None and "swebench" in configs:
        configs["swebench"]["bm_kwargs"]["max_interactions"] = args.swebench_max_interactions
    if "terminalbench2" in configs:
        configs["terminalbench2"]["bm_kwargs"]["actor"] = args.terminalbench2_actor
        if args.terminalbench2_max_turns is not None:
            configs["terminalbench2"]["bm_kwargs"]["max_interactions"] = args.terminalbench2_max_turns

    settings_kwargs = {}
    if args.max_tokens is not None:
        settings_kwargs["max_tokens"] = args.max_tokens
    if args.reasoning_effort is not None:
        settings_kwargs["reasoning_effort"] = args.reasoning_effort
    model_settings = ModelSettings(**settings_kwargs)

    print(f"\n{'=' * 70}")
    print(f"  ACE Experiment: mode={args.mode}  seed={args.seed}")
    print(f"  model={args.model}  num_tasks={args.num_tasks}")
    print(f"  model_settings={settings_kwargs or 'default'}")
    print(f"  thinking={'off' if args.disable_thinking else 'default'}")
    print(f"  training_time={args.training_time}")
    print(f"  max_turns={benchmark_max_turns}")
    print(f"  benchmarks={benchmarks_to_run}")
    print(f"  output_dir={args.output_dir}")
    print(f"{'=' * 70}\n")

    if args.task_split_manifest is not None:
        if args.mode not in ("sequential", "interleaved"):
            raise ValueError("--task-split-manifest requires sequential or interleaved mode")
        task_order = task_order_from_train_manifest(
            args.task_split_manifest, benchmarks_to_run, args.seed, args.num_tasks, args.mode
        )
    else:
        task_order = get_unified_task_order(configs, args.num_tasks, args.seed, args.mode)
    print(f"Total tasks: {len(task_order)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "online_metrics.jsonl"

    exp_config = {
        "mode": args.mode,
        "seed": args.seed,
        "agent": "ace",
        "model": args.model,
        "num_tasks": args.num_tasks,
        "benchmarks": benchmarks_to_run,
        "task_order": [(s, t) for s, t in task_order],
        "disable_thinking": args.disable_thinking,
        "training_time": args.training_time,
        "max_turns": args.max_turns,
        "benchmark_max_turns": benchmark_max_turns,
        "task_split_manifest": str(args.task_split_manifest) if args.task_split_manifest else None,
        "terminalbench2_actor": args.terminalbench2_actor,
    }
    if args.appworld_max_interactions is not None:
        exp_config["appworld_max_interactions"] = args.appworld_max_interactions
    if args.swebench_max_interactions is not None:
        exp_config["swebench_max_interactions"] = args.swebench_max_interactions
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    PlaybookStore.reset_all()
    if args.mode == "interleaved":
        _ckpt_ids = ["ace_interleaved_global"]
    elif args.mode == "sequential":
        _ckpt_ids = ["ace_sequential_global"]
    else:
        _ckpt_ids = [f"ace_isolated_{b}" for b in benchmarks_to_run]

    _restored = False
    _restored_session_count = 0
    for sid in _ckpt_ids:
        ckpt_path = output_dir / f"playbook_{sid}.json"
        if ckpt_path.exists():
            store = PlaybookStore.get_or_create(
                shuffle_mode=args.mode,
                benchmark_id=sid.replace("ace_isolated_", "") if args.mode == "isolated" else None,
            )
            store.load_checkpoint(str(ckpt_path))
            _restored_session_count = max(_restored_session_count, store.session_count)
            _restored = True
    if _restored:
        print(f"  ♻️  Restored playbook from checkpoint (session_count={_restored_session_count})")
    # A checkpoint contains playbook state, not the in-memory updates from a
    # partially completed benchmark. Give resumed sessions fresh run IDs so
    # the orchestrator cannot reuse their old results without replaying those
    # updates into the playbook.
    resume_run_suffix = (
        f"_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        if _restored else ""
    )

    all_scores: list[float] = []
    bm_scores: defaultdict[str, list[float]] = defaultdict(list)
    session_index = 0

    if metrics_path.exists():
        kept_lines: list[str] = []
        with open(metrics_path, "r") as f:
            for line in f:
                if _restored and session_index >= _restored_session_count:
                    break
                rec = json.loads(line)
                all_scores.append(rec["score"])
                bm_scores[rec["benchmark_slug"]].append(rec["score"])
                kept_lines.append(line)
                session_index += 1
        with open(metrics_path, "w") as f:
            f.writelines(kept_lines)
        if session_index > 0:
            print(f"  ♻️  Restored {session_index} metrics records (cum_avg={sum(all_scores)/len(all_scores):.3f})")

    if args.mode in ("isolated", "sequential"):
        _completed_benchmarks: set[str] = set()
        if _restored and session_index > 0:
            _bm_counts: dict[str, int] = defaultdict(int)
            with open(metrics_path, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    _bm_counts[rec["benchmark_slug"]] += 1
            for bm_slug, task_ids in group_by_benchmark(task_order):
                if _bm_counts.get(bm_slug, 0) >= len(task_ids):
                    _completed_benchmarks.add(bm_slug)
            if _completed_benchmarks:
                print(f"  ⏭️  Skipping completed benchmarks: {sorted(_completed_benchmarks)}")

        for bm_slug, task_ids in group_by_benchmark(task_order):
            if bm_slug in _completed_benchmarks:
                continue

            print(f"\n{'=' * 60}")
            print(f"  {args.mode.upper()} — {bm_slug} ({len(task_ids)} tasks)")
            print(f"{'=' * 60}\n")

            bm_kwargs = configs[bm_slug]["bm_kwargs"]
            agent_kwargs = configs[bm_slug].get("agent_kwargs", {})

            benchmark = load_benchmark(bm_slug)(**bm_kwargs)
            agent = load_agent("ace")(
                model=args.model,
                curator_model=args.model,
                shuffle_mode=args.mode,
                benchmark_id=bm_slug,
                runner="direct",
                model_settings=model_settings,
                enable_thinking=False if args.disable_thinking else None,
                training_time=args.training_time,
                **agent_kwargs,
            )

            results = evaluate(
                benchmark=benchmark,
                agent=agent,
                task_ids=task_ids,
                max_workers=1,
                output_dir=str(output_dir),
                run_id=f"ace_{args.mode}_s{args.seed}_{bm_slug}{resume_run_suffix}",
                **({"max_steps": benchmark_max_turns[bm_slug]} if benchmark_max_turns[bm_slug] is not None else {}),
            )

            print(f"  {bm_slug} score={results.benchmark_score}")

            stores = PlaybookStore.list_stores()
            store_key = (f"ace_isolated_{bm_slug}" if args.mode == "isolated"
                         else "ace_sequential_global")
            store = stores.get(store_key)
            if store:
                stats = get_playbook_stats(store.playbook)
                print(f"  playbook: sessions={store.session_count}, "
                      f"bullets={stats['total_bullets']}")
                store.save_checkpoint(str(output_dir / f"playbook_{store_key}.json"))

            for i, sr in enumerate(results.session_results):
                tid = task_ids[i] if i < len(task_ids) else sr.task_id or "?"
                rec = record_online_metrics(
                    metrics_path, session_index, bm_slug, tid, sr,
                    args.mode, args.seed, args.model,
                    all_scores, bm_scores,
                    training_time=args.training_time,
                    max_turns=benchmark_max_turns[bm_slug],
                )
                print(f"    [{session_index}] {bm_slug}::{tid}  "
                      f"score={rec['score']:.2f}  cum={rec['cumulative_avg_score']:.3f}  "
                      f"steps={rec['steps']}")
                session_index += 1

    elif args.mode == "interleaved":
        for i, (bm_slug, task_id) in enumerate(task_order):
            if _restored and i < _restored_session_count:
                print(f"  ⏭️  Skipping Interleaved [{i+1}/{len(task_order)}] {bm_slug}::{task_id} (cached)")
                continue

            print(f"\n--- Interleaved [{i+1}/{len(task_order)}] {bm_slug}::{task_id} ---")

            bm_kwargs = configs[bm_slug]["bm_kwargs"]
            agent_kwargs = configs[bm_slug].get("agent_kwargs", {})

            benchmark = load_benchmark(bm_slug)(**bm_kwargs)
            agent = load_agent("ace")(
                model=args.model,
                curator_model=args.model,
                shuffle_mode="interleaved",
                benchmark_id=bm_slug,
                runner="direct",
                model_settings=model_settings,
                enable_thinking=False if args.disable_thinking else None,
                training_time=args.training_time,
                **agent_kwargs,
            )

            results = evaluate(
                benchmark=benchmark,
                agent=agent,
                task_ids=[task_id],
                max_workers=1,
                output_dir=str(output_dir),
                run_id=f"ace_interleaved_s{args.seed}_{i:03d}_{bm_slug}{resume_run_suffix}",
                **({"max_steps": benchmark_max_turns[bm_slug]} if benchmark_max_turns[bm_slug] is not None else {}),
            )

            sr = results.session_results[0]

            rec = record_online_metrics(
                metrics_path, session_index, bm_slug, task_id, sr,
                "interleaved", args.seed, args.model,
                all_scores, bm_scores,
                training_time=args.training_time,
                max_turns=benchmark_max_turns[bm_slug],
            )
            print(f"  score={rec['score']:.2f}  cum={rec['cumulative_avg_score']:.3f}  "
                  f"steps={rec['steps']}")
            stores = PlaybookStore.list_stores()
            store = stores.get("ace_interleaved_global")
            if store:
                stats = get_playbook_stats(store.playbook)
                print(f"  playbook: sessions={store.session_count}, "
                      f"bullets={stats['total_bullets']}")
                store.save_checkpoint(str(output_dir / "playbook_ace_interleaved_global.json"))
            session_index += 1

    print(f"\n{'=' * 70}")
    print("  Final Summary")
    print(f"{'=' * 70}")
    print(f"  Overall avg score: {sum(all_scores)/len(all_scores):.3f}")
    for bm, scores in sorted(bm_scores.items()):
        print(f"  {bm:20s}: avg={sum(scores)/len(scores):.3f}  n={len(scores)}")

    stores = PlaybookStore.list_stores()
    for store_id, store in stores.items():
        stats = get_playbook_stats(store.playbook)
        print(f"  playbook[{store_id}]: sessions={store.session_count}, "
              f"bullets={stats['total_bullets']}")

    print(f"\n  Metrics: {metrics_path}")
    print(f"  Output:  {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run ACE with optional training-time grading feedback")
    parser.add_argument("--mode", required=True, choices=["isolated", "sequential", "interleaved"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, default=50, help="Tasks per benchmark")
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--benchmarks", default="browsecompplus,swebench,bfcl,tau2",
                        help="Comma-separated benchmark slugs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-split-manifest", type=Path, default=None,
                        help="Use only frozen train IDs from a train/val manifest")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--appworld-max-interactions", type=int, default=None,
                        help="Optional AppWorld interaction limit for smoke runs")
    parser.add_argument("--swebench-max-interactions", type=int, default=None,
                        help="Optional SWE-bench interaction limit for smoke runs")
    parser.add_argument("--reasoning-effort", default=None, help="Reasoning effort (low/medium/high)")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Pass enable_thinking=false to the vLLM chat template")
    parser.add_argument("--training-time", action="store_true",
                        help="Pass official post-session grading to ACE reflection")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Fallback maximum agent/environment steps per task")
    parser.add_argument("--appworld-max-turns", type=int, default=None,
                        help="AppWorld max steps (overrides --max-turns)")
    parser.add_argument("--bfcl-max-turns", type=int, default=None,
                        help="BFCL max steps (overrides --max-turns)")
    parser.add_argument("--swebench-max-turns", type=int, default=None,
                        help="SWE-bench max steps (overrides --max-turns)")
    parser.add_argument("--terminalbench2-max-turns", type=int, default=None,
                        help="Terminal-Bench 2.0 max steps (overrides --max-turns)")
    parser.add_argument("--terminalbench2-actor", choices=["terminus2", "interactive"], default="terminus2",
                        help="Terminal-Bench 2.0 ACE actor (default: Harbor Terminus-2)")
    args = parser.parse_args()
    if args.appworld_max_interactions is not None and args.appworld_max_interactions < 1:
        parser.error("--appworld-max-interactions must be positive")
    if args.swebench_max_interactions is not None and args.swebench_max_interactions < 1:
        parser.error("--swebench-max-interactions must be positive")
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns must be positive")
    for slug in ("appworld", "bfcl", "swebench", "terminalbench2"):
        value = getattr(args, f"{slug}_max_turns")
        if value is not None and value < 1:
            parser.error(f"--{slug}-max-turns must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
