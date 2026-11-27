# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Linux process identities independent of wall-clock corrections and PID reuse."""
from __future__ import annotations

from pathlib import Path


def process_start_ticks(pid: int) -> str | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()
        return fields[19] if fields[0] not in {"Z", "X"} else None
    except (OSError, IndexError):
        return None


def current_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def stable_identity(pid: int) -> dict:
    return {"start_ticks": process_start_ticks(pid), "boot_id": current_boot_id()}


def matches_identity(record: dict, created: float) -> bool:
    if record.get("start_ticks") is not None:
        return (process_start_ticks(record["pid"]) == record["start_ticks"]
                and (record.get("boot_id") is None or record["boot_id"] == current_boot_id()))
    return created == record.get("created")  # Legacy records require an explicit verified upgrade.
