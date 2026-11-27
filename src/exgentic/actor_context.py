# SPDX-License-Identifier: Apache-2.0
"""Method-independent actor instructions; deliberately standard-library only."""

from __future__ import annotations

import hashlib
import os
import runpy
from pathlib import Path
from typing import Literal

ActorContextMode = Literal["legacy", "shared-empty-v1"]
ACTOR_CONTEXT_MODES = ("legacy", "shared-empty-v1")
DEFAULT_ACTOR_CONTEXT: ActorContextMode = "shared-empty-v1"

# ACE keeps its native empty section structure internally. It conveys no learned
# knowledge and renders identically to GEPA/SkillOpt's empty strings.
EMPTY_PLAYBOOK = "\n\n".join(
    f"## {title}"
    for title in (
        "STRATEGIES & INSIGHTS",
        "FORMULAS & CALCULATIONS",
        "CODE SNIPPETS & TEMPLATES",
        "COMMON MISTAKES TO AVOID",
        "PROBLEM-SOLVING HEURISTICS",
        "CONTEXT CLUES & INDICATORS",
        "OTHERS",
    )
)

COMMON_INSTRUCTIONS = """You are an expert agent completing tasks in the provided environment.
Read the task, environment policy, and available tool descriptions before acting.
Use only the supplied capabilities and schema-valid arguments. Ground actions in the task and observed results.
Inspect results, adapt to errors, preserve unrelated state, and verify the requested outcome.
Follow the environment's interaction and completion protocol.
Use learned guidance when relevant. If a strategy has an ID, cite it when it influences an action."""

BENCHMARK_INSTRUCTIONS = {
    "bfcl": (
        "Use the available function-calling tools with their exact names and argument schemas. "
        "Complete the current user turn before calling finish, then continue when the next turn arrives. "
        "Carry forward relevant state and apply later corrections."
    ),
    "appworld": (
        "Complete the task through the provided AppWorld applications. "
        "Follow the supplied environment policy for the execution interface and API documentation. "
        "Resolve identities, dates, credentials, and record IDs from the current task and application data. "
        "Inspect required pages of results, verify changes, and report completion using the provided finish action."
    ),
    "terminalbench2": (
        "Complete the task in its Linux workspace using the provided terminal interface. "
        "Inspect relevant files and tools, make focused changes, and run relevant verification before finishing. "
        "Follow the terminal agent's command and completion format."
    ),
}


def validate_actor_context(mode: str) -> None:
    if mode not in ACTOR_CONTEXT_MODES:
        raise ValueError(f"Unknown actor context: {mode}")


def learned_policy_text(learned: str) -> str:
    if not learned.strip() or learned.strip() == EMPTY_PLAYBOOK:
        return ""
    return learned


def shared_actor_prompt(benchmark: str, learned: str = "") -> str:
    parts = [COMMON_INSTRUCTIONS]
    if benchmark in BENCHMARK_INSTRUCTIONS:
        parts.extend(["## Environment guidance", BENCHMARK_INSTRUCTIONS[benchmark]])
    parts.extend(["## Learned policy", learned_policy_text(learned) or "(No learned guidance yet.)"])
    return "\n\n".join(parts)


def terminal_actor_instruction(task: str, learned: str, mode: str = "legacy") -> str:
    validate_actor_context(mode)
    if mode != "legacy":
        return task + "\n\n" + shared_actor_prompt("terminalbench2", learned)
    if learned.strip():
        return task + (
            "\n\nACE playbook from previous tasks (use relevant strategies; "
            "mention a bullet ID when it influences an action):\n" + learned
        )
    return task


def actor_context_metadata(mode: str, benchmarks: list[str]) -> dict:
    validate_actor_context(mode)
    if mode == "legacy":
        return {"version": "legacy"}
    metadata = {
        "version": mode,
        "initial_learned_policy": "empty",
        "initial_learned_policy_sha256": hashlib.sha256(b"").hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tool_selection": "all_public_tools",
        "enable_thinking": False,
        "initial_actor_prompt_sha256": {
            benchmark: hashlib.sha256(shared_actor_prompt(benchmark).encode()).hexdigest() for benchmark in benchmarks
        },
    }
    if os.environ.get("EXGENTIC_INFERENCE_PROFILE"):
        active = runpy.run_path(str(Path(__file__).with_name("inference_profile.py")))["profile"]()
        metadata.update(enable_thinking=True, inference_profile_sha256=active["sha256"],
                        reasoning_effort=active["reasoning_effort"])
    return metadata


def require_matching_actor_context(saved: dict, mode: str, benchmarks: list[str]) -> None:
    """Never resume learned state under a different context or changed renderer."""
    validate_actor_context(mode)
    if saved.get("actor_context", "legacy") != mode:
        raise ValueError("Existing actor context differs; use a fresh output directory for the new protocol")
    if mode != "legacy" and saved.get("actor_context_metadata") != actor_context_metadata(mode, benchmarks):
        raise ValueError("Existing actor context renderer or initial prompt hashes differ")
