# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Shared frozen-playbook generator prompt for ACE and PREMiSE."""

from ..playbook_utils import get_playbook_stats

def build_generator_system_prompt(playbook: str) -> str:
    stats = get_playbook_stats(playbook)
    has_content = stats["total_bullets"] > 0

    parts = [
        "You are an expert agent that completes tasks using available tools.",
        "You have access to a curated playbook of strategies and insights "
        "learned from previous tasks.  Use these to make better decisions.",
        "",
        "## Guidelines",
        "- Read the playbook carefully and apply relevant strategies",
        "- Pay attention to common mistakes listed and avoid them",
        "- Use available tools to interact with the environment",
        "- Think step-by-step before acting",
        "- When you are confident in your solution, use the finish/submit tool",
        "- When a playbook bullet influences your decision, mention its ID "
        "(e.g. [err-00001]) in your reasoning text",
    ]

    if has_content:
        parts.extend([
            "",
            "## Playbook (accumulated strategies & insights)",
            "Each line has a bullet ID and usage stats "
            "(helpful=N means it helped N times, harmful=N means it misled N times).",
            "Prefer high-helpful, low-harmful bullets.",
            "",
            playbook,
        ])
    else:
        parts.extend([
            "",
            "## Playbook",
            "(No strategies accumulated yet. This is the first session.)",
        ])

    return "\n".join(parts)
