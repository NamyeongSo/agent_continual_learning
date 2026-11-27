# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Continue one repository across milestone sessions, updating ACE after each one."""

from __future__ import annotations

import argparse

from exgentic import ACEAgent, SWEMilestoneBenchmark, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ripgrep")
    parser.add_argument("--model", required=True)
    parser.add_argument("--curator-model")
    parser.add_argument("--actor", choices=("terminus2", "interactive"), default="terminus2")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--agent-timeout", type=int, default=10800)
    parser.add_argument("--terminus-python", help="Override the isolated Terminus-2 Python executable")
    parser.add_argument("--num-tasks", type=int, help="Run an ordered prefix, including ungraded milestones")
    parser.add_argument("--run-id", default="swemilestone-ace")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument(
        "--training-feedback", action="store_true", help="Allow official score feedback during training"
    )
    args = parser.parse_args()
    benchmark = SWEMilestoneBenchmark(
        subset=args.repo,
        actor=args.actor,
        max_interactions=args.max_turns,
        agent_timeout_sec=args.agent_timeout,
        terminus_python=args.terminus_python,
    )
    agent = ACEAgent(
        model=args.model,
        curator_model=args.curator_model,
        benchmark_id="swemilestone",
        runner="direct",
        shuffle_mode="sequential",
        curator_frequency=1,
        training_time=args.training_feedback,
    )
    evaluate(
        benchmark=benchmark,
        agent=agent,
        max_workers=1,
        num_tasks=args.num_tasks,
        max_steps=200,
        max_actions=200,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
