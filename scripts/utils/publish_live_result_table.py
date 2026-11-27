# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Keep derived JSON and LaTeX results current without changing experiment artifacts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research_journal.result_tables import latex  # noqa: E402


def write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def publish(output, base_url):
    config = json.loads((output / "results_table_manifest.json").read_text())
    with urllib.request.urlopen(base_url + "/api/live-progress", timeout=90) as response:
        data = json.load(response)
    if data.get("initializing"):
        raise ValueError("Monitor is collecting its first snapshot; previous reports are retained")
    table = next(table for table in data["result_tables"] if table["id"] == config["id"])
    tex = (f"% Snapshot: {data['generated_at']}\n" + latex(table)).encode()
    write(output / "main_results.json", json.dumps({"generated_at": data["generated_at"], **table},
                                                   ensure_ascii=False, indent=2).encode() + b"\n")
    write(output / "main_results.tex", tex)
    write(output / "main_results_body.tex", (f"% Snapshot: {data['generated_at']}\n"
                                           + latex(table, standalone=False)).encode())
    return table["completed_cells"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8878")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least five seconds")
    with (args.output / "publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(args.output / "publisher_process.json", json.dumps({"pid": os.getpid(),
              "command": [os.sys.executable, *os.sys.argv], "started_at": datetime.now(UTC).isoformat()}).encode())
        while True:
            try:
                completed = publish(args.output, args.base_url.rstrip("/"))
                state = {"phase": "watching", "completed_cells": completed}
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                if args.once:
                    raise
                state = {"phase": "waiting_server", "error": f"{type(exc).__name__}: {exc}"}
            write(args.output / "publisher_status.json", json.dumps({**state,
                  "at": datetime.now(UTC).isoformat()}).encode())
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
