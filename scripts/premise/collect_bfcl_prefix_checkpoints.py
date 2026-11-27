# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Recover BFCL-only PREMiSE checkpoints and verify the entire saved terminal state."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.premise.run_bfcl_alignment import SPLITS, save_immutable  # noqa: E402
from scripts.utils import evaluate_isolated_checkpoint_queue as shared  # noqa: E402


def recover(run: Path, output: Path, points: tuple[int, ...], seed: int, split_hash: str,
            *, minimum_point: int = 10, allow_terminal: bool = False) -> list[dict]:
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.playbook_utils import update_bullet_counts
    from exgentic.agents.premise.selection import sample_past

    state_path = run / "state.json"
    terminal_bytes = state_path.read_bytes()
    saved = json.loads(terminal_bytes)
    config, cursor = saved["config"], saved["cursor"]
    if config["mode"] not in ("sequential", "isolated") or (
        config["seed"], config["manifest_sha256"]
    ) != (seed, split_hash):
        raise ValueError("Expected the requested BFCL seed and frozen task split")
    state_key = "bfcl" if config["mode"] == "isolated" else "global"
    if set(saved["states"]) != {state_key}:
        raise ValueError("Recovery expects a single BFCL-only state")
    if (config.get("method", "premise") != "premise"
            or config.get("candidate_validation", "current_past") != "current_past"):
        raise ValueError("Expected PREMiSE with the original current/past selection")
    if not 0 < cursor <= len(config["order"]) or any(b != "bfcl" for b, _ in config["order"][:cursor]):
        raise ValueError("Saved training state must be a BFCL-only prefix")
    upper = cursor if allow_terminal else cursor - 1
    if minimum_point < 1 or not points or len(set(points)) != len(points) or any(
        not minimum_point <= p <= upper for p in points
    ):
        raise ValueError(f"Each checkpoint must be distinct and between {minimum_point} and {upper}")
    initial = {"playbook": DEFAULT_PLAYBOOK, "next_global_id": 1}
    state = {"beam": [copy.deepcopy(initial), copy.deepcopy(initial)], "memory": []}
    rng = random.Random(seed)
    snapshots, artifacts = {}, []
    parents_verified = 0
    for index, (benchmark, task) in enumerate(config["order"][:cursor], 1):
        path = run / "iterations" / f"iter_{index:04d}.json"
        raw = path.read_bytes()
        artifact = json.loads(raw)
        artifacts.append({"path": str(path), "sha256": shared.digest(raw)})
        if (artifact["iteration"], artifact["benchmark"], artifact["task_id"]) != (index, benchmark, task):
            raise ValueError("Iteration differs from the frozen training order")
        for rank, parent in enumerate(artifact.get("parents", [])):
            if parent:
                parent_config = shared.read_json(Path(parent["session_dir"]) / "config.json")
                if parent_config["agent_kwargs"]["playbook"] != state["beam"][rank]["playbook"]:
                    raise ValueError(f"Reconstructed parent differs at iteration {index}")
                parents_verified += 1
        eligible = [item for item in state["memory"] if (item["benchmark"], item["task_id"]) != (benchmark, task)]
        if "selection" in artifact or artifact.get("reason") == "insufficient_valid_candidates":
            past = sample_past(eligible, config["past_count"], rng)
        elif (artifact.get("status") == "skipped" and artifact.get("train_update_applied") is False
              and "rng_state_after" in artifact):
            rng.setstate(ast.literal_eval(artifact["rng_state_after"]))
        else:
            raise ValueError("Unsupported skipped iteration; RNG advancement cannot be verified")
        if "selection" in artifact:
            sample = [{"benchmark": item["benchmark"], "task_id": item["task_id"]} for item in past]
            if sample != artifact["past_sample"]:
                raise ValueError(f"Replay sample differs at iteration {index}")
            candidates = {item["id"]: item for item in artifact["candidates"]}
            selected = [candidates[name] for name in artifact["selection"]["selected"]]
            state["beam"] = [copy.deepcopy(item["playbook"]) for item in selected]
            for rank, chosen in enumerate(selected):
                tags = [{"id": item.get("entry_id", item.get("id")), "tag": item.get("tag", "neutral")}
                        for item in chosen["reflection"].get("entry_assessments", [])]
                state["beam"][rank]["playbook"] = update_bullet_counts(state["beam"][rank]["playbook"], tags)
            primary = selected[0]
            for sampled, result in zip(past, primary.get("past_results", []), strict=True):
                entry = next(item for item in state["memory"] if item["memory_id"] == sampled["memory_id"])
                entry.update(latest_selected_replay_official_score=result["official_score"],
                             latest_selected_replay_candidate_id=primary["id"], latest_selected_replay_iteration=index)
            if primary["applied"]:
                state["memory"].append({
                    "memory_id": f"{index}|{benchmark}|{task}", "benchmark": benchmark, "task_id": task,
                    "trace_artifact": f"iterations/iter_{index:04d}.json", "parent_index": primary["parent_rank"] - 1,
                    "reflection": primary["reflection"], "accepted_candidate_id": primary["id"],
                    "selected_official_score": primary["current_official_score"],
                    "latest_selected_replay_official_score": primary["current_official_score"],
                })
        elif artifact.get("status") != "skipped" or artifact.get("train_update_applied") is not False:
            raise ValueError("Skipped update lacks explicit unchanged-state evidence")
        snapshot = {"config": config, "cursor": index, "states": {state_key: copy.deepcopy(state)},
                    "rng_state": repr(rng.getstate())}
        if index in points:
            snapshots[index] = snapshot
    if snapshot != saved:
        raise ValueError("Reconstructed terminal beam, memory, or RNG differs from the entire saved state")
    if state_path.read_bytes() != terminal_bytes:
        raise ValueError("Source training state changed while recovering checkpoints")
    groups = []
    for point in sorted(points):
        path = output / f"checkpoint_{point:03d}.json"
        save_immutable(path, snapshots[point])
        groups.append({"benchmark": "bfcl", "method": "premise", "seed": seed, "checkpoint": point,
                       "checkpoint_path": str(path), "checkpoint_sha256": shared.digest(path.read_bytes()),
                       "model": config["model"], "source_run": str(run), "reconstructed": True,
                       "source_mode": config["mode"], "bfcl_only_prefix_verified": True,
                       "terminal_cursor": cursor, "entire_terminal_state_verified": True})
    save_immutable(output / "recovery_manifest.json", {
        "source_run": str(run), "source_state": str(state_path), "source_state_sha256": shared.digest(terminal_bytes),
        "terminal_cursor": cursor, "entire_terminal_state_verified": True,
        "parent_configs_verified": parents_verified, "iteration_artifacts": artifacts, "groups": groups,
    })
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output, source = args.output.resolve(), args.source_root.resolve()
    split_bytes = SPLITS.read_bytes()
    split_hash = shared.digest(split_bytes)
    groups = []
    for seed in (42, 43, 44):
        isolated = source / f"s{seed}_isolated_t3"
        run = isolated if isolated.is_dir() else source / f"s{seed}_sequential_t3"
        saved = shared.read_json(run / "state.json")
        points = tuple(range(10, saved["cursor"]))
        groups.extend(recover(run, output / "frozen/bfcl/premise" / f"s{seed}", points, seed, split_hash))
        print(f"RECOVERED seed={seed} points={len(points)} verified_cursor={saved['cursor']}", flush=True)
    save_immutable(output / "launch_manifest.json", {
        "manifest_sha256": split_hash, "source_root": str(source), "groups": groups,
        "source_context_scope": "BFCL-only PREMiSE training prefixes; source modes are recorded per group",
    })


if __name__ == "__main__":
    main()
