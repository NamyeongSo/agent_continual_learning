# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

#!/usr/bin/env python3
"""Train a two-parent Pareto span-replay playbook on frozen ExGentic tasks."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.actor_context import ACTOR_CONTEXT_MODES, DEFAULT_ACTOR_CONTEXT  # noqa: E402
from exgentic.agents.ace.playbook_utils import PLAYBOOK_EDIT_MODES  # noqa: E402
from exgentic.agents.premise.pipeline import PremisePipeline  # noqa: E402
from exgentic.core.experiment_limits import ACTOR_OUTPUT_TOKENS, BENCHMARK_MAX_TURNS  # noqa: E402


def resolve_docker_ssh_port(value: str) -> str:
    """Pass SSH-config ports explicitly for SWE-bench's Python Docker SDK."""
    parsed = urlsplit(value)
    if parsed.scheme != "ssh" or parsed.port is not None or not parsed.hostname:
        return value
    target = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname
    config = subprocess.run(["ssh", "-G", target], check=True, capture_output=True,
                            text=True, timeout=10)
    port = next((line.split(None, 1)[1].strip() for line in config.stdout.splitlines()
                 if line.lower().startswith("port ")), "22")
    if not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError(f"Invalid SSH port for Docker host {parsed.hostname!r}: {port!r}")
    return urlunsplit(parsed._replace(netloc=f"{parsed.netloc}:{port}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-tag", default="",
                        help="Separate session/environment IDs when resuming with revised runtime behavior")
    parser.add_argument("--appworld-policy", type=Path, help="Experiment-specific AppWorld policy template")
    parser.add_argument("--actor-context", choices=ACTOR_CONTEXT_MODES, default=DEFAULT_ACTOR_CONTEXT,
                        help="Common actor context and empty learned policy; legacy preserves historical prompts")
    parser.add_argument("--stop-after", type=int, help="Stop after this many total training tasks")
    parser.add_argument("--benchmarks", default="bfcl,appworld,terminalbench2")
    parser.add_argument("--mode", choices=("isolated", "sequential", "interleaved"), default="sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tasks", type=int, default=None, help="Training tasks per benchmark; default: all manifest train IDs")
    parser.add_argument("--all-train-tasks", action="store_true",
                        help="Use every manifest train ID, including different train sizes per benchmark")
    parser.add_argument("--past", type=int, choices=(3, 6, 9), default=3)
    parser.add_argument("--method", choices=("premise", "ps_st_pareto"), default="premise")
    parser.add_argument("--disable-alignment", action="store_true",
                        help="Skip alignment judging and select by current/past official scores only")
    parser.add_argument("--candidate-selection",
                        choices=("pareto", "alignment_product", "pfs_first", "joint_pareto", "random"), default="pareto",
                        help="alignment_product: one parent, four children, keep the highest mean(A)*mean(score) child")
    parser.add_argument("--appworld-candidate-metric", choices=("tgc", "pct"), default="tgc",
                        help="AppWorld candidate reward; Online and final evaluation continue to report TGC")
    parser.add_argument("--benchmark-candidate-metric", choices=("official", "auxiliary"), default="official",
                        help="Auxiliary: BFCL scored-user-turn pass rate or Terminal-Bench collected-test pass rate")
    parser.add_argument("--candidate-evaluation", choices=("environment_replay", "span_policy_shift"),
                        default="environment_replay",
                        help="Score candidates by environment replay or inference-only, outcome-directed strict F1")
    parser.add_argument("--policy-shift-temperature", type=float, default=0.0,
                        help="Temperature for span action generation only; original training actions are the reference")
    parser.add_argument("--proposal-retrieval", choices=("none", "a", "b", "ab", "random"), default="none",
                        help="Two past proposal examples selected using action strict F1 and MiniLM reflection cosine")
    parser.add_argument("--proposal-embedding-url", default="http://127.0.0.1:18886")
    parser.add_argument("--grounded-proposals", action="store_true", help="Review evidence and compare grounded cases")
    parser.add_argument("--grounded-retrieval", choices=("none", "a", "b", "ab", "random"), default="ab")
    parser.add_argument("--candidate-validation", choices=("current_past", "random_val4"),
                        default="current_past", help="Candidate selection task source")
    parser.add_argument("--validation-scope", choices=("current", "seen"), default="current",
                        help="Benchmark pool for four random validation tasks")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--temperature", type=float, help="Override temperature for every actor and analysis role")
    parser.add_argument("--actor-temperature", type=float, help="Temperature for parent, replay and evaluation actors")
    parser.add_argument("--reflection-temperature", type=float, help="Independent reflection sampling temperature")
    parser.add_argument("--mutation-temperature", type=float, help="Independent proposal sampling temperature")
    parser.add_argument("--playbook-edit-mode", choices=PLAYBOOK_EDIT_MODES, default="add_only",
                        help="Allowed mutation edits; checkpoint policy must match on resume")
    parser.add_argument("--parent-workers", type=int, default=2)
    parser.add_argument("--reflection-workers", type=int, default=4)
    parser.add_argument("--mutation-workers", type=int, default=4)
    parser.add_argument("--replay-workers", type=int, default=28,
                        help="Concurrent replay sessions, or independent action workers for span_policy_shift")
    parser.add_argument("--replay-retries", type=int, default=2)
    parser.add_argument("--skip-task-errors", choices=("none", "nontransient", "all"), default="nontransient",
                        help="Record a failed task and continue without updating beam or memory")
    parser.add_argument("--llm-calls", type=int, default=32)
    parser.add_argument("--tool-calling-max-tokens", type=int, default=ACTOR_OUTPUT_TOKENS,
                        help="Actor output request cap; the shared 2,048-token turn budget also applies")
    parser.add_argument("--llm-timeout", type=float, default=600.0,
                        help="Request timeout for PREMiSE acting/analysis calls; native Terminus-2 is unchanged")
    parser.add_argument("--llm-retries", type=int, default=1,
                        help="Retries per PREMiSE LLM request, without restarting an exhausted replay")
    parser.add_argument("--appworld-sessions", type=int, default=8)
    parser.add_argument("--bfcl-sessions", type=int, default=16)
    parser.add_argument("--swebench-sessions", type=int, default=2)
    parser.add_argument("--terminalbench2-sessions", type=int, default=2)
    parser.add_argument("--terminalbench2-version", choices=("2.0", "2.1"), default=None,
                        help="Override the manifest's Terminal-Bench task release")
    parser.add_argument("--unlimited-benchmark-sessions", action="store_true", default=True,
                        help="Disable all per-benchmark session semaphores")
    parser.add_argument("--limit-benchmark-sessions", action="store_false", dest="unlimited_benchmark_sessions")
    parser.add_argument("--appworld-max-steps", type=int, default=BENCHMARK_MAX_TURNS["appworld"])
    parser.add_argument("--bfcl-max-steps", type=int, default=BENCHMARK_MAX_TURNS["bfcl"])
    parser.add_argument("--swebench-max-steps", type=int, default=100)
    parser.add_argument("--terminalbench2-max-steps", "--terminalbench2-max-turns", type=int,
                        default=BENCHMARK_MAX_TURNS["terminalbench2"])
    parser.add_argument("--evaluate-val", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate frozen checkpoints 50/100/130 on all test sets")
    parser.add_argument("--evaluate-test-checkpoint", type=int, choices=(50, 100, 130),
                        help="Evaluate one frozen checkpoint while training may still be running")
    parser.add_argument("--evaluate-test-benchmark", choices=("bfcl", "appworld", "terminalbench2"),
                        help="Test benchmark for --evaluate-test-checkpoint")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in (
        args.num_tasks or 1, args.parent_workers, args.reflection_workers,
        args.mutation_workers, args.replay_workers, args.llm_calls,
        args.appworld_sessions, args.bfcl_sessions, args.swebench_sessions,
        args.terminalbench2_sessions,
        args.tool_calling_max_tokens,
    )):
        parser.error("Counts and worker limits must be positive")
    if args.replay_retries < 0:
        parser.error("--replay-retries must be nonnegative")
    if args.candidate_evaluation == "span_policy_shift" and (
        args.method != "premise" or args.candidate_validation != "current_past"
    ):
        parser.error("span_policy_shift requires --method premise and --candidate-validation current_past")
    if not 0 <= args.policy_shift_temperature <= 2:
        parser.error("--policy-shift-temperature must be between 0 and 2")
    if bool(args.evaluate_test_checkpoint) != bool(args.evaluate_test_benchmark):
        parser.error("--evaluate-test-checkpoint and --evaluate-test-benchmark must be used together")
    if args.evaluate_test_checkpoint and (not args.evaluation_only or args.evaluate_test):
        parser.error("Partial checkpoint evaluation requires --evaluation-only without --evaluate-test")
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    if not os.environ.get("DOCKER_HOST") and os.environ.get("ACE_DOCKER_HOST"):
        os.environ["DOCKER_HOST"] = os.environ["ACE_DOCKER_HOST"]
    benchmarks = [slug.strip() for slug in args.benchmarks.split(",") if slug.strip()]
    if not benchmarks or len(set(benchmarks)) != len(benchmarks):
        parser.error("--benchmarks must contain distinct slugs")
    if any(slug in benchmarks for slug in ("swebench", "terminalbench2")) and not args.dry_run:
        if os.environ.get("DOCKER_HOST"):
            try:
                os.environ["DOCKER_HOST"] = resolve_docker_ssh_port(os.environ["DOCKER_HOST"])
            except (OSError, ValueError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as exc:
                parser.error(f"Could not resolve Docker SSH port: {exc}")
        docker = shutil.which("docker") or str(ROOT / ".venv" / "bin" / "docker")
        if not Path(docker).is_file():
            parser.error(f"Docker CLI is unavailable: {docker}")
        os.environ["PATH"] = str(Path(docker).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
        try:
            subprocess.run([docker, "info"], check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=10)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            parser.error(f"Docker benchmark requires a reachable Docker daemon: {exc}")
    pipeline = PremisePipeline(
        output=args.output_dir, manifest=args.task_split_manifest,
        actor_context=args.actor_context,
        benchmarks=benchmarks, mode=args.mode, seed=args.seed,
        count=None if args.all_train_tasks else args.num_tasks, past_count=args.past, model=args.model,
        base_url=args.base_url, parent_workers=args.parent_workers,
        reflection_workers=args.reflection_workers,
        mutation_workers=args.mutation_workers, replay_workers=args.replay_workers,
        replay_retries=args.replay_retries,
        skip_task_errors=args.skip_task_errors,
        llm_calls=args.llm_calls,
        llm_timeout=args.llm_timeout, llm_retries=args.llm_retries,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        actor_temperature=args.actor_temperature, reflection_temperature=args.reflection_temperature,
        mutation_temperature=args.mutation_temperature,
        playbook_edit_mode=args.playbook_edit_mode,
        tool_calling_max_tokens=args.tool_calling_max_tokens,
        execution_tag=args.execution_tag,
        terminalbench2_version=args.terminalbench2_version,
        appworld_policy_override=args.appworld_policy.read_text() if args.appworld_policy else None,
        benchmark_limits=(
            {slug: None for slug in benchmarks} if args.unlimited_benchmark_sessions else
            {"appworld": args.appworld_sessions, "bfcl": args.bfcl_sessions,
             "swebench": args.swebench_sessions, "terminalbench2": args.terminalbench2_sessions}
        ),
        max_steps={"appworld": args.appworld_max_steps, "bfcl": args.bfcl_max_steps,
                   "swebench": args.swebench_max_steps,
                   "terminalbench2": args.terminalbench2_max_steps},
        method=args.method,
        candidate_evaluation=args.candidate_evaluation,
        policy_shift_temperature=args.policy_shift_temperature,
        proposal_retrieval=args.proposal_retrieval,
        proposal_embedding_url=args.proposal_embedding_url,
        grounded_proposals=args.grounded_proposals, grounded_retrieval=args.grounded_retrieval,
        alignment_scoring=not args.disable_alignment,
        candidate_selection=args.candidate_selection,
        appworld_candidate_metric=args.appworld_candidate_metric,
        benchmark_candidate_metric=args.benchmark_candidate_metric,
        candidate_validation=args.candidate_validation,
        validation_scope=args.validation_scope,
    )
    print(f"{args.method}: {len(pipeline.order)} train tasks, past={args.past}, "
          f"candidate_evaluation={args.candidate_evaluation}, "
          f"candidate_validation={args.candidate_validation}, validation_scope={args.validation_scope}, "
          f"candidate span/task evaluations per iteration up to {4 * (1 + args.past)}, "
          f"replay workers={args.replay_workers}, "
          f"benchmark limits={'disabled' if args.unlimited_benchmark_sessions else 'enabled'}, "
          f"resume cursor={pipeline.cursor}")
    if args.dry_run:
        return
    if args.evaluation_only and not args.evaluate_test_checkpoint and pipeline.cursor != len(pipeline.order):
        parser.error("--evaluation-only requires completed training")
    if not args.evaluation_only:
        pipeline.run(stop_after=args.stop_after)
    if args.evaluate_val:
        pipeline.evaluate_validation()
    if args.evaluate_test:
        if len(benchmarks) == 1:
            pipeline.evaluate_test_checkpoints(checkpoints=(len(pipeline.order),), benchmarks=(benchmarks[0],))
        else:
            pipeline.evaluate_test_checkpoints()
    if args.evaluate_test_checkpoint:
        pipeline.evaluate_test_checkpoints(
            checkpoints=(args.evaluate_test_checkpoint,),
            benchmarks=(args.evaluate_test_benchmark,), allow_incomplete=True,
        )


if __name__ == "__main__":
    main()
