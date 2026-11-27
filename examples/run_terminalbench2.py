# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Run one Terminal-Bench 2.0 task with an Exgentic tool-calling agent."""

from exgentic import LiteLLMToolCallingAgent, TerminalBench2Benchmark, evaluate


def main() -> None:
    benchmark = TerminalBench2Benchmark()
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, num_tasks=1, output_dir="./outputs")


if __name__ == "__main__":
    main()
