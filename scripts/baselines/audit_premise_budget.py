# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Count retained PREMiSE rollout attempts without running benchmarks or models.

Iteration artifacts omit some failed candidate results. Report a lower bound,
and count available session directories separately rather than declaring that
the theoretical 18 sessions/encounter were actually executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RUN_PATTERN = re.compile(
    r"^(?P<namespace>.+)_i(?P<iteration>\d+)_(?P<role>parent\d+|"
    r"candidate_\d+_(?:current|past\d+)_try\d+)_(?P<benchmark>bfcl|appworld|terminalbench2)$"
)


def relocated(path: Path, archive_root: Path | None) -> Path:
    """Resolve archived outputs using an explicit replacement root."""
    if path.exists() or archive_root is None:
        return path
    if "outputs" in path.parts:
        index = path.parts.index("outputs")
        return archive_root.joinpath(*path.parts[index + 1 :])
    return path


def trace_run(trace: dict, archive_root: Path | None) -> Path | None:
    raw = trace.get("session_dir")
    if not raw:
        return None
    session = relocated(Path(raw), archive_root)
    if session.parent.name != "sessions":
        raise ValueError(f"Unrecognized PREMiSE session path: {raw}")
    return session.parent.parent


def audit_run(run: Path, archive_root: Path | None = None) -> dict:
    run = run.resolve()
    state_bytes = (run / "state.json").read_bytes()
    state = json.loads(state_bytes)
    config = state["config"]
    if config.get("candidate_validation", "current_past") != "current_past":
        raise ValueError(f"{run}: expected the current/past PREMiSE reference protocol")
    if config.get("method", "premise") != "premise":
        raise ValueError(f"{run}: not the PREMiSE reference method")
    artifacts = sorted((run / "iterations").glob("iter_*.json"))
    if not artifacts:
        raise ValueError(f"No iteration artifacts: {run}")
    digest = hashlib.sha256()
    totals: Counter = Counter()
    by_encounter: dict[str, Counter] = defaultdict(Counter)
    refs: dict[str, Path] = {}
    roots = {run / "sessions"}
    accepted_namespaces: set[tuple[str, int]] = set()
    encounters: set[tuple[str, str]] = set()
    iteration_ids: set[int] = set()
    for artifact in artifacts:
        raw = artifact.read_bytes()
        digest.update(artifact.name.encode() + b"\0" + raw)
        item = json.loads(raw)
        iteration = int(item["iteration"])
        key = (item["benchmark"], item["task_id"])
        if key in encounters or iteration in iteration_ids:
            raise ValueError(f"Duplicate encounter or iteration: {artifact}")
        encounters.add(key)
        iteration_ids.add(iteration)
        counts = Counter(encounters=1, skipped_updates=int(item.get("status") == "skipped"))
        traces = []
        for parent in item.get("parents", []):
            if parent:
                counts["parent_results"] += 1
                traces.append(parent)
        for candidate in item.get("candidates", []):
            results = [("current_results", candidate.get("current_result"))]
            results.extend(("past_results", result) for result in candidate.get("past_results", []))
            for role, result in results:
                if result and result.get("replay"):
                    counts[role] += 1
                    counts["recorded_retry_extras"] += int(result.get("replay_attempts", 1)) - 1
                    counts["recorded_prefix_replays"] += int(result["replay"].get("prefix_length", 0) > 0)
                    traces.append(result["replay"])
        for trace in traces:
            source = trace_run(trace, archive_root)
            if source is None:
                continue
            refs[source.name] = source
            roots.add(source.parent)
            match = RUN_PATTERN.fullmatch(source.name)
            if match:
                accepted_namespaces.add((match["namespace"], iteration))
        counts["artifact_attempt_lower_bound"] = sum(
            counts[name] for name in ("parent_results", "current_results", "past_results", "recorded_retry_extras")
        )
        totals.update(counts)
        by_encounter[key[0]].update(counts)

    # Copied/resumed prefixes can refer to an older source directory. Deduplicate
    # by run ID across those roots; exclude test/checkpoint evaluation names.
    observed: dict[str, tuple[str, int, str]] = {}
    missing_roots = []
    for root in sorted(roots):
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        for path in root.iterdir():
            match = RUN_PATTERN.fullmatch(path.name)
            if match and path.is_dir():
                observed[path.name] = (match["namespace"], int(match["iteration"]), match["benchmark"])
    lineage = {name: meta for name, meta in observed.items() if meta[:2] in accepted_namespaces}
    missing_refs = sorted(set(refs) - observed.keys())
    lower_bound = max(totals["artifact_attempt_lower_bound"], len(set(lineage) | refs.keys()))
    return {
        "run": str(run),
        "mode": config["mode"],
        "seed": config["seed"],
        "model": config.get("model"),
        "past_count": config.get("past_count"),
        "cursor": state["cursor"],
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "iteration_artifacts_sha256": digest.hexdigest(),
        "counts": dict(totals),
        "by_encounter_benchmark": {key: dict(value) for key, value in by_encounter.items()},
        "lineage_observed_sessions": len(lineage),
        "lineage_observed_by_execution_benchmark": dict(Counter(meta[2] for meta in lineage.values())),
        "source_observed_sessions_including_abandoned_attempts": len(observed),
        "missing_final_session_dirs": len(missing_refs),
        "missing_source_roots": missing_roots,
        "retained_attempt_lower_bound": lower_bound,
        "attempts_per_unique_encounter_lower_bound": round(lower_bound / len(encounters), 4),
        "exact_total_known": False,
        "limitations": [
            "Failed candidate results and failed updates may omit session references.",
            "Lineage count includes available sessions with a namespace/iteration referenced by retained artifacts.",
            "Other source sessions may belong to abandoned/restarted attempts; they are reported separately.",
            "A prefix replay session and a fresh rollout need not use equal LLM tokens or wall time.",
            "Reflection/mutation/alignment LM calls and frozen checkpoint evaluations are excluded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--archive-root", type=Path, help="Replacement for historical outputs/ paths")
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are preserved")
    args = parser.parse_args()
    reports = [audit_run(run, args.archive_root.resolve() if args.archive_root else None) for run in args.run]
    result = {"schema_version": 1, "unit": "environment_session_attempt", "runs": reports}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    for report in reports:
        print(json.dumps({key: report[key] for key in (
            "run", "mode", "seed", "retained_attempt_lower_bound", "lineage_observed_sessions",
            "missing_final_session_dirs", "attempts_per_unique_encounter_lower_bound",
        )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
