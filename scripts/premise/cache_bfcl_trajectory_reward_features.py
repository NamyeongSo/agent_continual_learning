# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Cache six observed BFCL trajectories for each frozen training-task query group."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.bfcl_trajectory_reward_features import FEATURE_VERSION, extract_features  # noqa: E402

CONTEXTS = ("parent", *(f"candidate_{number:02d}" for number in range(1, 6)))


def extract_group(group: dict, max_text_chars: int = 1200) -> dict:
    if group["model"] not in ("Qwen", "Gemma") or set(group["result_paths"]) != set(CONTEXTS):
        raise ValueError("Expected only Qwen/Gemma and parent plus exactly five candidate results")
    requests = group["public_task"]["user_turns"]
    first_request = requests[0]["text"] if requests else None
    traces = []
    for context in CONTEXTS:
        path = Path(group["result_paths"][context])
        values = extract_features(path, first_request=first_request, max_text_chars=max_text_chars)
        identity = values["identity"]
        if (identity["checkpoint_key"], identity["context"], identity["task_id"]) != (
                group["key"], context, group["task_id"]):
            raise ValueError(f"Manifest/result identity mismatch: {group['id']} / {context}")
        result = json.loads(path.read_text())
        if result["score"] not in (0, 1):
            raise ValueError("Official archived score must be binary and remains outside judge evidence")
        traces.append({**values, "context": context, "official_success": float(result["score"]),
                       "official_result_status": result["status"]})
    return {key: value for key, value in group.items() if key != "result_paths"} | {"traces": traces}


def build_cache(manifest_path: Path, output: Path, workers: int = 16, max_text_chars: int = 1200) -> dict:
    started = time.perf_counter()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    groups = manifest["groups"]
    if workers < 1 or len({group["id"] for group in groups}) != len(groups):
        raise ValueError("Positive extraction concurrency and unique frozen query group ids are required")
    if manifest.get("group_count", len(groups)) != len(groups) or manifest.get("trace_count", 6 * len(groups)) != (
            6 * len(groups)):
        raise ValueError("Frozen manifest coverage counts differ")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "group_features.jsonl.tmp"
    summary = {"created_at": datetime.now(UTC).isoformat(), "new_model_calls": 0,
               "manifest_path": str(manifest_path.resolve()),
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "feature_version": FEATURE_VERSION,
               "feature_source_sha256": hashlib.sha256(
                   (ROOT / "scripts/premise/bfcl_trajectory_reward_features.py").read_bytes()).hexdigest(),
               "cache_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "workers": workers, "compact_field_character_limit": max_text_chars,
               "group_count": 0, "trace_count": 0, "available_traces": 0, "missing_traces": 0,
               "truncated_traces": 0, "truncated_fields": 0, "by_model": {}, "missing": []}
    digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as executor, temporary.open("wb") as handle:
        values = executor.map(lambda group: extract_group(group, max_text_chars), groups)
        for group in values:
            summary["group_count"] += 1
            model = summary["by_model"].setdefault(group["model"], {
                "groups": 0, "traces": 0, "available": 0, "missing": 0, "truncated": 0,
            })
            model["groups"] += 1
            for trace in group["traces"]:
                summary["trace_count"] += 1
                model["traces"] += 1
                if trace["available"]:
                    summary["available_traces"] += 1
                    model["available"] += 1
                    evidence = trace["compact_trajectory"]
                    summary["truncated_fields"] += evidence["truncated_fields"]
                    summary["truncated_traces"] += evidence["lost_evidence"]
                    model["truncated"] += evidence["lost_evidence"]
                else:
                    summary["missing_traces"] += 1
                    model["missing"] += 1
                    summary["missing"].append({"group": group["id"], "model": group["model"],
                                               "context": trace["context"], "result_path": trace["result_path"]})
            data = (json.dumps(group, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
            handle.write(data)
            digest.update(data)
            if summary["group_count"] % 100 == 0:
                print(f"Cached {summary['group_count']}/{len(groups)} groups; "
                      f"{summary['available_traces']}/{summary['trace_count']} traces available", flush=True)
    final = output / "group_features.jsonl"
    temporary.replace(final)
    summary.update({"cache_path": str(final.resolve()), "cache_sha256": digest.hexdigest(),
                    "cache_bytes": final.stat().st_size, "wall_seconds": time.perf_counter() - started})
    (output / "feature_cache_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(build_cache(args.manifest, args.output, args.workers, args.max_text_chars), indent=2), flush=True)


if __name__ == "__main__":
    main()
