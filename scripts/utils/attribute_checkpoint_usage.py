# SPDX-License-Identifier: Apache-2.0
"""Attribute shared-bridge checkpoint costs using exact recorded response IDs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.utils.run_shared_reflection_matrix import read, write


class Attribution:
    def __init__(self):
        self.ids = set()
        self.shared = {}
        self.dedicated = {}

    def summary(self):
        shared = {key: value for key, value in self.shared.items() if key in self.ids}
        result = {}
        for port, records in (("8006", shared), ("8007", self.dedicated)):
            rows = list(records.values())
            result[port] = {
                "requests": len(rows),
                **{key: sum(row["usage"].get(key, 0) for row in rows)
                   for key in ("prompt_tokens", "completion_tokens")},
                **{key: sum(c.get(key, 0) for row in rows for c in row["choices"])
                   for key in ("reasoning_tokens", "visible_tokens")},
            }
        return result, sorted(shared)


def new_rows(path, positions):
    if not path.exists():
        return
    offset = positions.get(path, 0)
    if path.stat().st_size == offset:
        return
    with path.open("rb") as stream:
        if path.stat().st_size < offset:
            offset = 0
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            positions[path] = stream.tell()
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
            except (ValueError, UnicodeError):
                continue


def run(output, once=False):
    manifest = read(output / "checkpoint_queue_manifest.json")
    source = Path(manifest["source"])
    started = datetime.fromisoformat(manifest["evaluation_started_at"]).timestamp()
    positions, ledger = {}, Attribution()
    target = source / "monitor/checkpoint_usage_exclusions" / f"{output.name}.json"
    while True:
        for path in output.glob("sessions/**/litellm/*.jsonl"):
            for row in new_rows(path, positions):
                response = row.get("response")
                if isinstance(response, dict) and isinstance(response.get("id"), str):
                    ledger.ids.add(response["id"])
        for path, records in ((source / "serving/requests.jsonl", ledger.shared),
                              (output / "serving8007/requests.jsonl", ledger.dedicated)):
            for row in new_rows(path, positions):
                if row.get("role") == "actor" and row.get("at", 0) >= started:
                    records[row["id"]] = {key: row[key] for key in ("usage", "choices")}
        usage, excluded = ledger.summary()
        write(target, {"checkpoint_output": str(output), "updated_at": datetime.now(UTC).isoformat(),
                       "response_ids": excluded,
                       "proof": "Exact response.id in this checkpoint evaluator's SDK traces and original bridge log"})
        write(output / "usage_attribution.json", {"at": datetime.now(UTC).isoformat(), "by_port": usage,
              "sdk_response_ids": len(ledger.ids), "excluded_from_original_cohort": len(excluded),
              "exclusion_file": str(target),
              "scope": "Actor costs. Shared8006 is matched by exact SDK response ID;8007 is dedicated to this queue. "
                       "In-flight shared responses enter the ledger after their SDK trace is recorded. "
                       "Health probes remain separate from actor usage."})
        if once or read(output / "dashboard.json").get("phase") in {"completed", "error"}:
            return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.once)
