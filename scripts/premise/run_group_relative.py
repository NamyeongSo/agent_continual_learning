# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Optimize PEACE prompts from fixed task groups without environment replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
LIVE_CONFIG = ROOT / "outputs/qwen38_terminalbench21_terminus2_timeout48_c10_20261113/config.json"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.actor_context import ACTOR_CONTEXT_MODES, DEFAULT_ACTOR_CONTEXT  # noqa: E402
from exgentic.agents.ace.playbook_utils import PLAYBOOK_EDIT_MODES  # noqa: E402
from exgentic.agents.premise.action_agreement import METRICS as AGREEMENT_METRICS  # noqa: E402
from exgentic.agents.premise.action_nli import DEFAULT_MODEL, DEFAULT_REVISION  # noqa: E402
from exgentic.agents.premise.group_relative import CANONICAL_SCORE_MODES, GroupRelativePipeline  # noqa: E402
from exgentic.core.experiment_limits import BENCHMARK_MAX_TURNS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", default="bfcl")
    parser.add_argument("--mode", choices=("isolated", "sequential", "interleaved"), default="isolated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tasks", type=int, help="Tasks per benchmark; default: every manifest train task")
    parser.add_argument("--rollouts", type=int, default=5, help="Initial-policy environment rollouts per task")
    parser.add_argument("--candidates", "-C", type=int, default=5, help="Proposed prompts, plus the incumbent")
    parser.add_argument("--epochs", type=int, default=1, help="Passes over the same frozen rollout bank")
    parser.add_argument("--past", type=int, default=0, help="Additional seen task groups per update")
    parser.add_argument("--metric", choices=AGREEMENT_METRICS, default="canonical_f1")
    parser.add_argument("--canonical-score-mode", choices=CANONICAL_SCORE_MODES, default="lower",
                        help="Use lower canonical agreement, or the previous conservative interval selection")
    parser.add_argument("--agreement-text-model", default=DEFAULT_MODEL)
    parser.add_argument("--agreement-text-revision", default=DEFAULT_REVISION)
    parser.add_argument("--agreement-text-threshold", type=float, default=.95)
    parser.add_argument("--agreement-text-max-tokens", type=int, default=512)
    parser.add_argument("--agreement-cpu-threads", type=int, default=4)
    parser.add_argument("--shift-penalty", type=float, default=0.0, help="Penalty for shift from the current prompt")
    parser.add_argument("--max-policy-shift", type=float, help="Optional hard bound on same-prefix policy shift")
    parser.add_argument("--advantage-epsilon", type=float, default=1e-8)
    parser.add_argument("--actor-context", choices=ACTOR_CONTEXT_MODES, default=DEFAULT_ACTOR_CONTEXT)
    parser.add_argument("--playbook-edit-mode", choices=PLAYBOOK_EDIT_MODES, default="add_only")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--model-call-config", type=Path, default=LIVE_CONFIG,
                        help="Copy model_settings from the live Qwen3.8 evaluation configuration")
    parser.add_argument("--actor-temperature", type=float, default=1.0)
    parser.add_argument("--reflection-temperature", type=float, default=1.0)
    parser.add_argument("--mutation-temperature", type=float, default=1.0)
    parser.add_argument("--policy-shift-temperature", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--rollout-workers", type=int, default=5)
    parser.add_argument("--proposal-workers", type=int, default=5)
    parser.add_argument("--action-workers", type=int, default=128,
                        help="Global prefix queue workers; default matches the current GPU1 serving capacity")
    parser.add_argument("--llm-calls", type=int, default=128,
                        help="Shared in-process model-call limit, including prefix probes")
    parser.add_argument("--llm-timeout", type=float, default=600)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--appworld-policy", type=Path)
    parser.add_argument("--terminalbench2-version", choices=("2.0", "2.1"))
    for benchmark, cap in BENCHMARK_MAX_TURNS.items():
        parser.add_argument(f"--{benchmark}-max-steps", type=int, default=cap)
    parser.add_argument("--stop-after", type=int, help="Total optimization steps, including previous steps")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without model/environment calls")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if not benchmarks or len(benchmarks) != len(set(benchmarks)):
        parser.error("--benchmarks must contain distinct benchmark names")
    if any(b not in BENCHMARK_MAX_TURNS for b in benchmarks):
        parser.error("Supported benchmarks: bfcl, appworld, terminalbench2")
    counts = (args.num_tasks if args.num_tasks is not None else 1, args.rollout_workers,
              args.proposal_workers, args.action_workers, args.llm_calls,
              *(getattr(args, f"{b}_max_steps") for b in BENCHMARK_MAX_TURNS))
    if any(n < 1 for n in counts) or args.llm_retries < 0 or args.llm_timeout <= 0:
        parser.error("Worker/step counts and timeout must be positive; retries must be nonnegative")
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    live = json.loads(args.model_call_config.read_text())
    settings = live["model_settings"]
    if args.model != live["model"] or args.actor_temperature != settings["temperature"]:
        parser.error("Model and rollout temperature must match --model-call-config")
    if args.policy_shift_temperature != settings["temperature"]:
        parser.error("Probe temperature must match --model-call-config")
    if args.reflection_temperature != settings["temperature"] or args.mutation_temperature != settings["temperature"]:
        parser.error("Reflection and mutation temperatures must match --model-call-config")
    if args.reasoning_effort is not None and args.reasoning_effort != settings.get("reasoning_effort"):
        parser.error("Reasoning effort must match --model-call-config")
    if settings.get("max_tokens") is None:
        os.environ["EXGENTIC_ACTOR_OUTPUT_UNLIMITED"] = "1"
    if not os.environ.get("DOCKER_HOST") and os.environ.get("ACE_DOCKER_HOST"):
        os.environ["DOCKER_HOST"] = os.environ["ACE_DOCKER_HOST"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_dir / "group_relative.lock"), timeout=0):
        try:
            pipeline = GroupRelativePipeline(
                output=args.output_dir, manifest=args.task_split_manifest, benchmarks=benchmarks,
                mode=args.mode, seed=args.seed, count=args.num_tasks,
                model=args.model, base_url=args.base_url,
                rollout_count=args.rollouts, candidate_count=args.candidates, epochs=args.epochs,
                past_count=args.past, metric=args.metric, shift_penalty=args.shift_penalty,
                canonical_score_mode=args.canonical_score_mode,
                agreement_text_model=args.agreement_text_model, agreement_text_revision=args.agreement_text_revision,
                agreement_text_threshold=args.agreement_text_threshold,
                agreement_text_max_tokens=args.agreement_text_max_tokens,
                agreement_cpu_threads=args.agreement_cpu_threads,
                max_policy_shift=args.max_policy_shift, advantage_epsilon=args.advantage_epsilon,
                actor_context=args.actor_context, playbook_edit_mode=args.playbook_edit_mode,
                actor_temperature=args.actor_temperature, reflection_temperature=args.reflection_temperature,
                mutation_temperature=args.mutation_temperature, policy_shift_temperature=args.policy_shift_temperature,
                reasoning_effort=args.reasoning_effort,
                model_call_settings=settings,
                model_call_source={"path": str(args.model_call_config.resolve()),
                                   "sha256": hashlib.sha256(args.model_call_config.read_bytes()).hexdigest()},
                parent_workers=args.rollout_workers, mutation_workers=args.proposal_workers,
                replay_workers=args.action_workers, llm_calls=args.llm_calls,
                llm_timeout=args.llm_timeout, llm_retries=args.llm_retries,
                appworld_policy_override=args.appworld_policy.read_text() if args.appworld_policy else None,
                terminalbench2_version=args.terminalbench2_version,
                max_steps={b: getattr(args, f"{b}_max_steps") for b in BENCHMARK_MAX_TURNS},
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"{len(pipeline.collection_order)} tasks x {args.rollouts} initial rollouts; "
              f"{args.candidates} proposals/update; {len(pipeline.order)} optimization steps; "
              f"resume={pipeline.cursor}; candidate environment replays=0", flush=True)
        if args.dry_run:
            return
        if args.collect_only:
            pipeline.collect_rollouts()
        else:
            pipeline.run(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
