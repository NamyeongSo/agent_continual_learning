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

from exgentic.agents.premise.pipeline import PremisePipeline  # noqa: E402


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
    parser.add_argument("--benchmarks", default="bfcl,appworld,swebench")
    parser.add_argument("--mode", choices=("isolated", "sequential", "interleaved"), default="sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tasks", type=int, default=50, help="Training tasks per benchmark")
    parser.add_argument("--past", type=int, choices=(3, 6), default=3)
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--parent-workers", type=int, default=2)
    parser.add_argument("--reflection-workers", type=int, default=4)
    parser.add_argument("--mutation-workers", type=int, default=4)
    parser.add_argument("--replay-workers", type=int, default=28)
    parser.add_argument("--replay-retries", type=int, default=2)
    parser.add_argument("--skip-task-errors", choices=("none", "nontransient", "all"), default="none",
                        help="Record a failed task and continue without updating beam or memory")
    parser.add_argument("--llm-calls", type=int, default=32)
    parser.add_argument("--appworld-sessions", type=int, default=8)
    parser.add_argument("--bfcl-sessions", type=int, default=16)
    parser.add_argument("--swebench-sessions", type=int, default=2)
    parser.add_argument("--terminalbench2-sessions", type=int, default=2)
    parser.add_argument("--unlimited-benchmark-sessions", action="store_true",
                        help="Disable all per-benchmark session semaphores")
    parser.add_argument("--appworld-max-steps", type=int, default=50)
    parser.add_argument("--bfcl-max-steps", type=int, default=50)
    parser.add_argument("--swebench-max-steps", type=int, default=100)
    parser.add_argument("--terminalbench2-max-steps", type=int, default=100)
    parser.add_argument("--evaluate-val", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in (
        args.num_tasks, args.parent_workers, args.reflection_workers,
        args.mutation_workers, args.replay_workers, args.llm_calls,
        args.appworld_sessions, args.bfcl_sessions, args.swebench_sessions,
        args.terminalbench2_sessions,
    )):
        parser.error("Counts and worker limits must be positive")
    if args.replay_retries < 0:
        parser.error("--replay-retries must be nonnegative")
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
        benchmarks=benchmarks, mode=args.mode, seed=args.seed,
        count=args.num_tasks, past_count=args.past, model=args.model,
        base_url=args.base_url, parent_workers=args.parent_workers,
        reflection_workers=args.reflection_workers,
        mutation_workers=args.mutation_workers, replay_workers=args.replay_workers,
        replay_retries=args.replay_retries,
        skip_task_errors=args.skip_task_errors,
        llm_calls=args.llm_calls,
        benchmark_limits=(
            {slug: None for slug in benchmarks} if args.unlimited_benchmark_sessions else
            {"appworld": args.appworld_sessions, "bfcl": args.bfcl_sessions,
             "swebench": args.swebench_sessions, "terminalbench2": args.terminalbench2_sessions}
        ),
        max_steps={"appworld": args.appworld_max_steps, "bfcl": args.bfcl_max_steps,
                   "swebench": args.swebench_max_steps,
                   "terminalbench2": args.terminalbench2_max_steps},
    )
    print(f"Premise: {len(pipeline.order)} train tasks, past={args.past}, "
          f"replay jobs/iteration up to {4 * (1 + args.past)}, "
          f"replay workers={args.replay_workers}, "
          f"benchmark limits={'disabled' if args.unlimited_benchmark_sessions else 'enabled'}, "
          f"resume cursor={pipeline.cursor}")
    if args.dry_run:
        return
    pipeline.run()
    if args.evaluate_val:
        pipeline.evaluate_validation()


if __name__ == "__main__":
    main()
