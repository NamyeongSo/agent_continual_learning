# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Recover frozen interleaved checkpoints from completed training artifacts."""

from __future__ import annotations

import copy
import random
from pathlib import Path

from scripts.utils import evaluate_isolated_checkpoint_queue as shared


def recover_premise(run: Path, output: Path, points: tuple[int, ...], seed: int = 44) -> dict[int, Path]:
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.playbook_utils import update_bullet_counts
    from exgentic.agents.premise.selection import sample_past

    final_path = run / "checkpoints/checkpoint_130.json"
    saved = shared.read_json(final_path)
    config = saved["config"]
    if (config["mode"], config["seed"], saved["cursor"], len(config["order"])) != (
        "interleaved", seed, 130, 130,
    ) or config.get("candidate_validation", "current_past") != "current_past":
        raise ValueError(f"Expected the original seed-{seed} interleaved PREMiSE training run")
    initial = {"playbook": DEFAULT_PLAYBOOK, "next_global_id": 1}
    state = {"beam": [copy.deepcopy(initial), copy.deepcopy(initial)], "memory": []}
    rng = random.Random(seed)
    snapshots, anchors, artifacts = {}, [], []
    parents_verified = 0
    for index, (benchmark, task) in enumerate(config["order"], 1):
        path = run / "iterations" / f"iter_{index:04d}.json"
        raw = path.read_bytes()
        artifact = shared.read_json(path)
        artifacts.append({"path": str(path), "sha256": shared.digest(raw)})
        if (artifact["iteration"], artifact["benchmark"], artifact["task_id"]) != (index, benchmark, task):
            raise ValueError("PREMiSE iteration differs from its recorded training order")
        for rank, parent in enumerate(artifact["parents"]):
            if parent:
                parent_config = shared.read_json(Path(parent["session_dir"]) / "config.json")
                if parent_config["agent_kwargs"]["playbook"] != state["beam"][rank]["playbook"]:
                    raise ValueError(f"PREMiSE reconstructed parent differs at iteration {index}")
                parents_verified += 1
        past = sample_past(state["memory"], config["past_count"], rng)
        if "selection" in artifact:
            expected_sample = [{"benchmark": r["benchmark"], "task_id": r["task_id"]} for r in past]
            if artifact["past_sample"] != expected_sample:
                raise ValueError(f"PREMiSE replay sample differs at iteration {index}")
            candidates = {r["id"]: r for r in artifact["candidates"]}
            selected = [candidates[name] for name in artifact["selection"]["selected"]]
            state["beam"] = [copy.deepcopy(r["playbook"]) for r in selected]
            for rank, chosen in enumerate(selected):
                tags = [{"id": r.get("entry_id", r.get("id")), "tag": r.get("tag", "neutral")}
                        for r in chosen["reflection"].get("entry_assessments", [])]
                state["beam"][rank]["playbook"] = update_bullet_counts(state["beam"][rank]["playbook"], tags)
            primary = selected[0]
            for sampled, result in zip(past, primary.get("past_results", []), strict=False):
                entry = next(r for r in state["memory"] if r["memory_id"] == sampled["memory_id"])
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
            raise ValueError("PREMiSE iteration has no recorded selection or explicit skipped update")
        if index in points or index in (50, 100, 130):
            snapshot = {"config": config, "cursor": index, "states": {"global": copy.deepcopy(state)},
                        "rng_state": repr(rng.getstate())}
            if index in (50, 100, 130):
                anchor = run / "checkpoints" / f"checkpoint_{index:03d}.json"
                if snapshot != shared.read_json(anchor):
                    raise ValueError(f"PREMiSE recovered checkpoint {index} differs from the entire original state")
                anchors.append({"checkpoint": index, "path": str(anchor),
                                "sha256": shared.digest(anchor.read_bytes()), "full_state_verified": True})
            if index in points:
                snapshots[index] = snapshot
    if set(snapshots) != set(points):
        raise ValueError("Requested PREMiSE checkpoints were not recovered")
    paths = {}
    for point, snapshot in snapshots.items():
        path = output / f"checkpoint_{point:03d}.json"
        shared.write_json(path, snapshot)
        paths[point] = path
    shared.write_json(output / "recovery_manifest.json", {
        "source_run": str(run), "source_final_sha256": shared.digest(final_path.read_bytes()),
        "parent_configs_verified": parents_verified, "anchors": anchors, "iteration_artifacts": artifacts,
        "checkpoints": [{"checkpoint": point, "path": str(path), "sha256": shared.digest(path.read_bytes())}
                        for point, path in paths.items()],
    })
    return paths


def collect(
    output: Path, ace_root: Path, premise_root: Path, points: tuple[int, ...], seed: int = 44,
) -> dict[tuple[str, int], dict]:
    sources = {}
    premise_run = premise_root / f"s{seed}_interleaved_t3"
    missing_premise = tuple(p for p in points if not (premise_run / f"checkpoints/checkpoint_{p:03d}.json").exists())
    recovered = (
        recover_premise(premise_run, output / "recovered_sources/premise", missing_premise, seed)
        if missing_premise else {}
    )
    for point in points:
        path = recovered.get(point, premise_run / f"checkpoints/checkpoint_{point:03d}.json")
        sources[("premise", point)] = {"path": str(path), "reconstructed": point in recovered}
        if point in recovered:
            manifest = output / "recovered_sources/premise/recovery_manifest.json"
            sources[("premise", point)].update(recovery_manifest=str(manifest),
                                               recovery_manifest_sha256=shared.digest(manifest.read_bytes()))
    snapshots = ace_root / f"test_evaluation/snapshots/interleaved_s{seed}"
    missing_ace = [p for p in points if not (snapshots / f"checkpoint_{p:03d}.json").exists()]
    prefix = []
    if missing_ace:
        from scripts.ace.checkpoint_recovery import completed_prefix

        run = ace_root / f"interleaved_s{seed}"
        config = shared.read_json(run / "experiment_config.json")
        if (config["mode"], config["seed"]) != ("interleaved", seed):
            raise ValueError(f"Expected the seed-{seed} interleaved ACE run")
        prefix = completed_prefix(run, config["task_order"])
        if len(prefix) != 130:
            raise ValueError("ACE training prefix is incomplete")
        for point in (50, 100, 130):
            old = shared.read_json(snapshots / f"checkpoint_{point:03d}.json")
            recorded = prefix[point - 1]["checkpoint"]
            if not recorded or any(old.get(k) != v for k, v in recorded.items()):
                raise ValueError(f"ACE training checkpoint {point} differs from its previous frozen export")
    for point in points:
        path = snapshots / f"checkpoint_{point:03d}.json"
        if point in missing_ace:
            entry = prefix[point - 1]
            if not entry["checkpoint"] or entry["checkpoint"]["session_count"] != point:
                raise ValueError(f"ACE has no exact training checkpoint {point}")
            path = Path(entry["checkpoint_path"])
        sources[("ace", point)] = {"path": str(path), "reconstructed": False,
                                   "training_session_checkpoint": point in missing_ace}
    return sources
