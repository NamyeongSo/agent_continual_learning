# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Launch Harbor in the benchmark's isolated client environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ...environment.manager import EnvironmentManager


def run_terminus(request: dict, *, python: str | None = None) -> dict:
    executable = Path(python or EnvironmentManager().venv_python("benchmarks/swemilestone")).expanduser()
    if not executable.is_file():
        raise RuntimeError("Install Terminus-2 with: exgentic install --benchmark swemilestone --force")
    logs = Path(request["logs_dir"])
    logs.mkdir(parents=True, exist_ok=True)
    # Use the current bridge source, while Harbor remains in the isolated venv.
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    with (logs / "runtime.log").open("w") as log:
        result = subprocess.run(
            [str(executable), "-m", "exgentic.benchmarks.swemilestone.terminus_actor"],
            input=json.dumps(request),
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            timeout=request["agent_timeout_sec"] + 120,
            env=env,
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Terminus-2 failed; see {logs / 'runtime.log'}") from exc
    if result.returncode or response.get("error"):
        raise RuntimeError(f"Terminus-2 failed: {response.get('error', 'process exited')}; see {logs / 'runtime.log'}")
    return response["result"]
