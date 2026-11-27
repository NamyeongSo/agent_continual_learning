# SPDX-License-Identifier: Apache-2.0
"""Bound the native loop of an already running task at its tmux liveness check."""

from __future__ import annotations

import shlex


def tmux_turn_guard(real_binary: str, session_name: str, state_dir: str, limit: int) -> str:
    """Stop before the next episode while retaining the container for its verifier.

    Harbor calls ``has-session`` exactly once at the start of each agent episode.
    Other tmux operations pass through. Seed ``state_dir/count`` with the episode
    already in progress while the worker is paused inside its model call.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("Turn limit must be a positive integer")
    return f'''#!/bin/sh
real={shlex.quote(real_binary)}
session={shlex.quote(session_name)}
state={shlex.quote(state_dir)}
if [ "$#" -eq 3 ] && [ "$1" = has-session ] && [ "$2" = -t ] && [ "$3" = "$session" ]; then
    count=$(cat "$state/count") || exit 1
    case "$count" in ''|*[!0-9]*) exit 1 ;; esac
    if [ "$count" -ge {limit} ]; then
        printf '%s\\n' '{{"limit":{limit},"next_episode_blocked":true}}' > "$state/reached.json"
        exit 1
    fi
    next=$((count + 1))
    printf '%s\\n' "$next" > "$state/count.$$" && mv "$state/count.$$" "$state/count" || exit 1
fi
exec "$real" "$@"
'''
