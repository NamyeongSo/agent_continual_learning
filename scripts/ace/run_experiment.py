# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# --- ExGentic imports ---
from benchmark_metrics import APPWORLD_METRIC, reported_score
from checkpoint_recovery import recover_checkpoint
from task_ordering import SEQUENTIAL_BENCHMARK_ORDER, get_unified_task_order, group_by_benchmark

from exgentic.actor_context import (
    ACTOR_CONTEXT_MODES,
    DEFAULT_ACTOR_CONTEXT,
    actor_context_metadata,
    require_matching_actor_context,
)
from exgentic.agents.ace.playbook_store import PlaybookStore
from exgentic.agents.ace.playbook_utils import PLAYBOOK_EDIT_MODES, allowed_playbook_operations, get_playbook_stats
from exgentic.core.experiment_limits import BENCHMARK_MAX_TURNS, experiment_limits
from exgentic.core.types import ModelSettings, SessionResults
from exgentic.interfaces.lib.api import evaluate
from exgentic.interfaces.registry import load_agent, load_benchmark
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

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
            "interaction_mode": "code_generation",
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

def evaluate_training_tasks(task_ids, *, run_id, **kwargs):
    """Commit one ordered task at a time; infrastructure errors must stop learning."""
    for index, task_id in enumerate(task_ids):
        results = evaluate(task_ids=[task_id], run_id=f"{run_id}_t{index:03d}", **kwargs)
        if len(results.session_results) != 1:
            raise RuntimeError(f"Training task {task_id} did not produce exactly one result")
        result = results.session_results[0]
        require_valid_training_result(result, task_id)
        yield task_id, result


def require_valid_training_result(result, task_id):
    if result.status in {"error", "cancelled"} or result.score is None:
        raise RuntimeError(f"Training task {task_id} has no valid outcome: {result.status}: {result.details}")


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
    partial_score = sr.score if sr.score is not None else (1.0 if sr.success else 0.0)
    score = partial_score
    if bm_slug == "appworld":
        if not isinstance(sr.success, bool):
            raise ValueError("AppWorld TGC requires the official success flag")
        score = float(sr.success)
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
        **({"partial_score": partial_score, "success": sr.success, "tgc": score,
            "score_metric": APPWORLD_METRIC} if bm_slug == "appworld" else {}),
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
    requested = specific if specific is not None else args.max_turns
    cap = BENCHMARK_MAX_TURNS.get(benchmark_slug)
    return min(requested, cap) if requested is not None and cap is not None else requested or cap


def task_order_from_train_manifest(
    path: Path, benchmarks: list[str], seed: int, count: int | None, mode: str = "sequential"
) -> list[tuple[str, str]]:
    """Keep frozen training IDs fixed; vary order or interleaving by run seed."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sections = select_task_split_sections(manifest, benchmarks, seed)
    ordered_slugs = [slug for slug in SEQUENTIAL_BENCHMARK_ORDER if slug in benchmarks]
    ordered_slugs.extend(slug for slug in benchmarks if slug not in ordered_slugs)
    per_benchmark: dict[str, deque[str]] = {}
    for slug in ordered_slugs:
        split = sections[slug]
        train = validate_task_split(split, slug)["train"]
        manifest_count = manifest.get("train_count_per_benchmark", len(train))
        if len(train) != manifest_count:
            raise ValueError(f"Invalid training task count for {slug}")
        if count is not None and not 0 < count <= len(train):
            raise ValueError(f"Train manifest for {slug} must have at least {count} tasks")
        order_seed = int(hashlib.md5(f"{seed}_{slug}".encode()).hexdigest(), 16) % (2**31)
        if manifest.get("seed_scope") != "within_split_order_only":
            random.Random(order_seed).shuffle(train)
        per_benchmark[slug] = deque(train[:count])
    if mode in ("isolated", "sequential"):
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
    actor_context = getattr(args, "actor_context", DEFAULT_ACTOR_CONTEXT)
    enable_thinking = getattr(args, "enable_thinking", False)
    if enable_thinking and args.disable_thinking:
        raise ValueError("Thinking cannot be both enabled and disabled")
    disable_thinking = args.disable_thinking or (actor_context != "legacy" and not enable_thinking)
    edit_mode = getattr(args, "playbook_edit_mode", "add_only")
    allowed_playbook_operations(edit_mode)
    benchmarks_to_run = [s.strip() for s in args.benchmarks.split(",")]
    benchmark_max_turns = {
        slug: resolve_max_turns(args, slug)
        for slug in benchmarks_to_run
    }
    configs = {
        k: {**BENCHMARK_REGISTRY[k], "bm_kwargs": dict(BENCHMARK_REGISTRY[k]["bm_kwargs"])}
        for k in benchmarks_to_run
    }
    policy_path = getattr(args, "appworld_policy", None)
    if policy_path is not None and "appworld" in configs:
        configs["appworld"]["bm_kwargs"].update(policy_override=policy_path.read_text(), use_cache=False)
    if args.appworld_max_interactions is not None and "appworld" in configs:
        configs["appworld"]["bm_kwargs"]["max_interactions"] = args.appworld_max_interactions
    if args.swebench_max_interactions is not None and "swebench" in configs:
        configs["swebench"]["bm_kwargs"]["max_interactions"] = args.swebench_max_interactions
    if "terminalbench2" in configs:
        configs["terminalbench2"]["bm_kwargs"]["subset"] = getattr(args, "terminalbench2_version", "2.0")
        configs["terminalbench2"]["bm_kwargs"]["actor"] = args.terminalbench2_actor
        if args.terminalbench2_max_turns is not None:
            configs["terminalbench2"]["bm_kwargs"]["max_interactions"] = args.terminalbench2_max_turns

    settings_path = getattr(args, "model_settings_file", None)
    settings_kwargs = json.loads(settings_path.read_text()) if settings_path is not None else {}
    if args.max_tokens is not None:
        settings_kwargs["max_tokens"] = args.max_tokens
    if args.reasoning_effort is not None:
        settings_kwargs["reasoning_effort"] = args.reasoning_effort
    if getattr(args, "temperature", None) is not None:
        settings_kwargs["temperature"] = args.temperature
        for config in configs.values():
            config["agent_kwargs"] = {**config.get("agent_kwargs", {}), "temperature_override": args.temperature}
    actor_temperature = getattr(args, "actor_temperature", None)
    analysis_temperature = getattr(args, "analysis_temperature", None)
    if actor_temperature is not None:
        settings_kwargs["temperature"] = actor_temperature
    for config in configs.values():
        if actor_temperature is not None:
            config["agent_kwargs"] = {**config.get("agent_kwargs", {}), "temperature_override": actor_temperature}
        if analysis_temperature is not None:
            config["agent_kwargs"] = {**config.get("agent_kwargs", {}),
                                      "analysis_temperature_override": analysis_temperature}
    for config in configs.values():
        config["agent_kwargs"] = {
            **config.get("agent_kwargs", {}), "playbook_edit_mode": edit_mode, "actor_context": actor_context,
            "allow_shared_thinking": enable_thinking,
        }
        if actor_context != "legacy":
            config["agent_kwargs"]["enable_tool_shortlisting"] = False
    model_settings = ModelSettings(**settings_kwargs)

    print(f"\n{'=' * 70}")
    print(f"  ACE Experiment: mode={args.mode}  seed={args.seed}")
    task_count_label = "all manifest train IDs" if args.all_train_tasks else args.num_tasks
    print(f"  model={args.model}  num_tasks={task_count_label}")
    print(f"  model_settings={settings_kwargs or 'default'}")
    print(f"  thinking={'off' if disable_thinking else 'default'}")
    print(f"  training_time={args.training_time}")
    print(f"  playbook_edit_mode={edit_mode}")
    print(f"  max_turns={benchmark_max_turns}")
    print(f"  benchmarks={benchmarks_to_run}")
    print(f"  output_dir={args.output_dir}")
    print(f"{'=' * 70}\n")

    if args.task_split_manifest is not None:
        task_order = task_order_from_train_manifest(
            args.task_split_manifest, benchmarks_to_run, args.seed,
            None if args.all_train_tasks else args.num_tasks, args.mode
        )
    else:
        task_order = get_unified_task_order(configs, args.num_tasks, args.seed, args.mode)
    print(f"Total tasks: {len(task_order)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_config = output_dir / "experiment_config.json"
    if previous_config.exists():
        previous = json.loads(previous_config.read_text())
        require_matching_actor_context(previous, actor_context, benchmarks_to_run)
        previous_thinking = previous.get("effective_enable_thinking", not previous.get("disable_thinking", True))
        if previous_thinking != (not disable_thinking):
            raise ValueError("Existing thinking setting differs; use a separate output directory")
        if "model_settings" in previous and previous["model_settings"] != model_settings.model_dump(mode="json"):
            raise ValueError("Existing model settings differ; use a separate output directory")
        if "terminalbench2" in configs:
            previous_subset = json.loads(previous_config.read_text()).get("benchmark_configs", {}).get(
                "terminalbench2", {}).get("subset", "2.0")
            if previous_subset != configs["terminalbench2"]["bm_kwargs"]["subset"]:
                raise ValueError("Existing Terminal-Bench release differs; use a separate output directory")
        previous_mode = json.loads(previous_config.read_text()).get("playbook_edit_mode", "add_only")
        if previous_mode != edit_mode:
            raise ValueError("Existing experiment playbook edit mode differs; use a separate output directory")
    metrics_path = output_dir / "online_metrics.jsonl"
    recovered = recover_checkpoint(output_dir, task_order, args.mode) if args.resume_sessions else None
    if args.continue_from is not None:
        source = args.continue_from.resolve()
        previous = json.loads((source / "experiment_config.json").read_text())
        require_matching_actor_context(previous, actor_context, benchmarks_to_run)
        if "terminalbench2" in configs:
            previous_subset = previous.get("benchmark_configs", {}).get("terminalbench2", {}).get("subset", "2.0")
            if previous_subset != configs["terminalbench2"]["bm_kwargs"]["subset"]:
                raise ValueError("Continuation Terminal-Bench release differs; use a separate experiment")
        if previous.get("playbook_edit_mode", "add_only") != edit_mode:
            raise ValueError("Continuation configuration differs: playbook_edit_mode")
        for key, expected in {
            "mode": args.mode, "seed": args.seed, "model": args.model,
            "benchmarks": benchmarks_to_run, "training_time": args.training_time,
            "disable_thinking": disable_thinking,
            "benchmark_max_turns": benchmark_max_turns,
        }.items():
            if previous.get(key) != expected:
                raise ValueError(f"Continuation configuration differs: {key}")
        if args.mode != "sequential":
            raise ValueError("--continue-from currently supports sequential mode only")
        prior_rows = [json.loads(line) for line in (source / "online_metrics.jsonl").read_text().splitlines()]
        prefix = [(row["benchmark_slug"], row["task_id"]) for row in prior_rows]
        if len(set(prefix)) != len(prefix) or not set(prefix) <= set(task_order):
            raise ValueError("Continuation tasks must be unique members of the target training set")
        checkpoint = source / "playbook_ace_sequential_global.json"
        if json.loads(checkpoint.read_text())["session_count"] != len(prefix):
            raise ValueError("Continuation checkpoint and metrics counts differ")
        if metrics_path.exists() or (output_dir / checkpoint.name).exists():
            raise ValueError("Continuation destination must be fresh")
        task_order = prefix + [item for item in task_order if item not in set(prefix)]
        shutil.copy2(checkpoint, output_dir / checkpoint.name)
        shutil.copy2(source / "online_metrics.jsonl", metrics_path)

    stop_after = getattr(args, "stop_after", None)
    if stop_after is not None:
        if not 1 <= stop_after <= len(task_order):
            raise ValueError("Training stop must be within the full task order")
        if args.continue_from is not None and stop_after < len(prefix):
            raise ValueError("Training stop precedes the reused prefix")
        task_order = task_order[:stop_after]

    exp_config = {
        "mode": args.mode,
        "seed": args.seed,
        "agent": "ace",
        "model": args.model,
        "num_tasks": None if args.all_train_tasks else args.num_tasks,
        "benchmarks": benchmarks_to_run,
        "benchmark_configs": {slug: configs[slug]["bm_kwargs"] for slug in benchmarks_to_run},
        "task_order": [(s, t) for s, t in task_order],
        "disable_thinking": disable_thinking,
        "effective_enable_thinking": not disable_thinking,
        "model_settings": model_settings.model_dump(mode="json"),
        "model_settings_file_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest() if settings_path else None,
        "training_time": args.training_time,
        "max_turns": args.max_turns,
        "benchmark_max_turns": benchmark_max_turns,
        "experiment_limits": {slug: experiment_limits(slug) for slug in benchmarks_to_run},
        "task_split_manifest": str(args.task_split_manifest) if args.task_split_manifest else None,
        "task_split_manifest_sha256": (
            hashlib.sha256(args.task_split_manifest.read_bytes()).hexdigest()
            if args.task_split_manifest else None
        ),
        "all_train_tasks": args.all_train_tasks,
        "terminalbench2_actor": args.terminalbench2_actor,
        "continued_from": str(args.continue_from) if args.continue_from else None,
        "context_error_policy": "skip_current_task_no_learning",
        "terminalbench2_full_trajectory": True,
        "max_tokens": args.max_tokens,
        "temperature_all_roles": getattr(args, "temperature", None),
        "playbook_edit_mode": edit_mode,
        "actor_context": actor_context,
        "actor_context_metadata": actor_context_metadata(actor_context, benchmarks_to_run),
    }
    if args.reasoning_effort is not None:
        exp_config["reasoning_effort"] = args.reasoning_effort
    if actor_temperature is not None or analysis_temperature is not None:
        exp_config["role_temperatures"] = {"actor": actor_temperature, "analysis": analysis_temperature}
    if policy_path is not None:
        exp_config["appworld_policy_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if stop_after is not None:
        exp_config["stop_after"] = stop_after
    if args.appworld_max_interactions is not None:
        exp_config["appworld_max_interactions"] = args.appworld_max_interactions
    if args.swebench_max_interactions is not None:
        exp_config["swebench_max_interactions"] = args.swebench_max_interactions
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)
    if getattr(args, "dry_run", False):
        print(json.dumps({"config": str(output_dir / "experiment_config.json"),
                          "train_tasks": len(task_order), "actor_context": actor_context}))
        return

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
            payload = json.loads(ckpt_path.read_text())
            _restored_session_count = max(
                _restored_session_count, payload.get("completed_task_count", store.session_count)
            )
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

    if recovered is not None:
        metrics_path.write_text("")
        recovered_scores: list[float] = []
        recovered_bm_scores: defaultdict[str, list[float]] = defaultdict(list)
        for index, item in enumerate(recovered):
            if "metrics" in item:
                record = dict(item["metrics"])
                score = reported_score(item["benchmark"], record)
                if score is not None:
                    recovered_scores.append(score)
                    recovered_bm_scores[item["benchmark"]].append(score)
                record["cumulative_avg_score"] = (
                    sum(recovered_scores) / len(recovered_scores) if recovered_scores else None
                )
                benchmark_scores = recovered_bm_scores[item["benchmark"]]
                record["benchmark_cumulative_avg_score"] = (
                    sum(benchmark_scores) / len(benchmark_scores) if benchmark_scores else None
                )
                with metrics_path.open("a") as metrics_file:
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            record_online_metrics(
                metrics_path, index, item["benchmark"], item["task_id"],
                SessionResults.model_validate(item["result"]), args.mode, args.seed, args.model,
                recovered_scores, recovered_bm_scores, training_time=args.training_time,
                max_turns=benchmark_max_turns[item["benchmark"]],
            )
        print(f"  Recovered {len(recovered)} completed tasks from per-session checkpoints")

    if metrics_path.exists():
        kept_lines: list[str] = []
        with open(metrics_path) as f:
            for line in f:
                if _restored and session_index >= _restored_session_count:
                    break
                rec = json.loads(line)
                score = reported_score(rec["benchmark_slug"], rec)
                if score is not None:
                    all_scores.append(score)
                    bm_scores[rec["benchmark_slug"]].append(score)
                kept_lines.append(line)
                session_index += 1
        with open(metrics_path, "w") as f:
            f.writelines(kept_lines)
        if session_index > 0:
            print(f"  ♻️  Restored {session_index} metrics records (cum_avg={sum(all_scores)/len(all_scores):.3f})")

    if args.mode in ("isolated", "sequential"):
        _completed_benchmarks: set[str] = set()
        _bm_counts: dict[str, int] = defaultdict(int)
        if _restored and session_index > 0:
            _bm_counts: dict[str, int] = defaultdict(int)
            with open(metrics_path) as f:
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
            if _restored:
                task_ids = task_ids[_bm_counts.get(bm_slug, 0):]

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
                enable_thinking=False if disable_thinking else (True if enable_thinking else None),
                training_time=args.training_time,
                **agent_kwargs,
            )

            for tid, sr in evaluate_training_tasks(
                task_ids,
                benchmark=benchmark,
                agent=agent,
                max_workers=1,
                output_dir=str(output_dir),
                run_id=f"ace_{args.mode}_s{args.seed}_{bm_slug}{resume_run_suffix}",
                **({"max_steps": benchmark_max_turns[bm_slug]} if benchmark_max_turns[bm_slug] is not None else {}),
            ):
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
                store_key = (f"ace_isolated_{bm_slug}" if args.mode == "isolated"
                             else "ace_sequential_global")
                store = PlaybookStore.list_stores().get(store_key)
                if store:
                    path = output_dir / f"playbook_{store_key}.json"
                    store.save_checkpoint(str(path))
                    payload = json.loads(path.read_text())
                    payload["completed_task_count"] = session_index
                    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

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
                enable_thinking=False if disable_thinking else (True if enable_thinking else None),
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
            require_valid_training_result(sr, task_id)

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
    if session_index != len(task_order):
        raise RuntimeError(f"Incomplete training: {session_index}/{len(task_order)} tasks")
    (output_dir / "training_complete.json").write_text(json.dumps({
        "mode": args.mode, "seed": args.seed, "completed_task_count": session_index,
        "finished_at": datetime.now().isoformat(),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run ACE with optional training-time grading feedback")
    parser.add_argument("--mode", required=True, choices=["isolated", "sequential", "interleaved"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, default=50, help="Tasks per benchmark")
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--actor-context", choices=ACTOR_CONTEXT_MODES, default=DEFAULT_ACTOR_CONTEXT,
                        help="Common actor context with empty learned policy; use legacy for historical runs")
    parser.add_argument("--temperature", type=float, help="Override temperature for every actor and learning role")
    parser.add_argument("--actor-temperature", type=float)
    parser.add_argument("--analysis-temperature", type=float)
    parser.add_argument("--playbook-edit-mode", choices=PLAYBOOK_EDIT_MODES, default="add_only",
                        help="Allowed curator edits; use a separate output directory for each variant")
    parser.add_argument("--benchmarks", default="browsecompplus,swebench,bfcl,tau2",
                        help="Comma-separated benchmark slugs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Write and validate the plan without model/environment calls")
    parser.add_argument("--appworld-policy", type=Path, help="Experiment-specific AppWorld policy template")
    parser.add_argument("--stop-after", type=int, help="Stop after this many total training tasks")
    parser.add_argument("--continue-from", type=Path, default=None,
                        help="Continue a completed sequential pilot in a fresh output directory")
    parser.add_argument("--resume-sessions", action="store_true",
                        help="Recover the completed prefix from per-session checkpoints in the output directory")
    parser.add_argument("--task-split-manifest", type=Path, default=None,
                        help="Use only frozen train IDs from a train/val manifest")
    parser.add_argument("--all-train-tasks", action="store_true",
                        help="Use each benchmark's complete manifest train split instead of --num-tasks")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--model-settings-file", type=Path,
                        help="JSON ModelSettings baseline; explicit CLI overrides take precedence")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Explicitly enable thinking, including with shared actor context")
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
    parser.add_argument("--terminalbench2-version", choices=["2.0", "2.1"], default="2.0",
                        help="Pinned Terminal-Bench task release; use a separate output directory per release")
    parser.add_argument("--terminalbench2-actor", choices=["terminus2", "interactive"], default="terminus2",
                        help="Terminal-Bench 2.0 ACE actor (default: Harbor Terminus-2)")
    args = parser.parse_args()
    if args.all_train_tasks and args.task_split_manifest is None:
        parser.error("--all-train-tasks requires --task-split-manifest")
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
