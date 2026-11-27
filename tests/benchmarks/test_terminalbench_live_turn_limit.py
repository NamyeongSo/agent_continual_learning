# SPDX-License-Identifier: Apache-2.0
"""A live native episode guard leaves the environment available to the verifier."""

import json
import subprocess

import pytest

from exgentic.benchmarks.terminalbench2.native_turn_limit import tmux_turn_guard


def test_live_guard_allows_turn_100_blocks_101_and_passes_other_tmux_calls(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "count").write_text("98\n")
    binary = tmp_path / "real-tmux"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    binary.chmod(0o755)
    guard = tmp_path / "tmux"
    guard.write_text(tmux_turn_guard(str(binary), "terminus-2", str(state), 100))
    guard.chmod(0o755)

    def run(*args):
        return subprocess.run([str(guard), *args], capture_output=True, text=True)

    for turn in [99, 100]:
        assert run("has-session", "-t", "terminus-2").returncode == 0
        assert int((state / "count").read_text()) == turn
    blocked = run("has-session", "-t", "terminus-2")
    assert blocked.returncode == 1 and blocked.stdout == ""
    assert int((state / "count").read_text()) == 100
    assert json.loads((state / "reached.json").read_text())["next_episode_blocked"] is True
    assert run("send-keys", "-t", "terminus-2", "echo ok").returncode == 0
    assert run("has-session", "-t", "other-session").returncode == 0
    assert int((state / "count").read_text()) == 100


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_live_guard_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        tmux_turn_guard("tmux-real", "terminus-2", "/state", limit)
