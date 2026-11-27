# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Recover actor policies at completed five-task training boundaries without training."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path

METHODS = {"ace", "premise", "gepa", "skillopt"}
BENCHMARKS = {"bfcl", "appworld", "terminalbench2"}


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


class Reader:
    def __init__(self):
        self.cache = {}

    def raw(self, path):
        path = Path(path)
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        previous = self.cache.get(path)
        if previous and previous[0] == stamp:
            return previous[1]
        data = path.read_bytes()
        self.cache[path] = (stamp, data)
        return data

    def json(self, path):
        return json.loads(self.raw(path))


class PlainState(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Optimizer checkpoint must contain plain data, not {module}.{name}")


def streams(roots, reader):
    result = []
    for root in map(Path, roots):
        if (root / "superseded_by.json").exists():
            continue
        plan = reader.json(root / "launch_manifest.json")
        for job in plan.get("jobs", []):
            if job.get("method") not in METHODS or job.get("benchmark") not in BENCHMARKS:
                continue
            output = Path(job["output"])
            spec = {"id": digest(str(output.resolve()))[:20], "root": str(root), "output": str(output),
                    "variant": root.name, "method": job["method"], "benchmark": job["benchmark"],
                    "seed": job["seed"], "snapshot": str(root / "code_snapshot"),
                    "model": plan.get("model", "openai/Qwen/Qwen3.6-35B-A3B"), "temperature": 0.7,
                    "playbook_edit_mode": job.get("playbook_edit_mode", "add_only")}
            if job["method"] in {"ace", "premise"}:
                folder = Path(job["train_dir"])
                command = job["command"]
                manifest = Path(command[command.index("--task-split-manifest") + 1])
                data = reader.json(manifest)
                split = data[f"{job['benchmark']}_{job['seed']}"]
                spec.update(train_dir=str(folder), splits={name: list(map(str, split[name]))
                                                         for name in ("train", "val", "test")},
                            split_sha256=digest(reader.raw(manifest)))
            else:
                config = output / "config.json"
                if not config.exists():
                    continue
                cfg = reader.json(config)
                spec.update(splits={name: [str(x["task_id"]) for x in cfg[key]] for name, key in
                                    (("train", "train"), ("val", "validation"), ("test", "test"))},
                            train_dir=str(output), split_sha256=cfg["split_manifest_sha256"],
                            model=cfg["model"], benchmark_kwargs=cfg["benchmark_kwargs"])
            if job["method"] in {"ace", "premise"}:
                context_path = Path(spec["train_dir"]) / (
                    "experiment_config.json" if job["method"] == "ace" else "state.json")
                context_config = reader.json(context_path) if context_path.exists() else {}
                if job["method"] == "premise":
                    context_config = context_config.get("config", {})
            else:
                context_config = cfg
            if context_config.get("actor_context", "legacy") != "legacy":
                spec["actor_context"] = context_config["actor_context"]
                spec["actor_context_metadata"] = context_config["actor_context_metadata"]
                if job["method"] in {"ace", "premise"}:
                    spec["benchmark_kwargs"] = context_config["benchmark_configs"][job["benchmark"]]
            split_sets = [set(spec["splits"][name]) for name in ("train", "val", "test")]
            if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i)):
                raise ValueError("Training, validation and test splits overlap")
            spec["train_total"] = len(spec["splits"]["train"])
            if job["benchmark"] == "appworld":
                policy = None
                if job["method"] in {"ace", "premise"} and "--appworld-policy" in job["command"]:
                    path = Path(job["command"][job["command"].index("--appworld-policy") + 1])
                    policy = reader.raw(path).decode("utf-8")
                elif job["method"] in {"gepa", "skillopt"}:
                    policy = spec["benchmark_kwargs"].get("policy_override")
                if policy is not None:
                    sha = digest(policy)
                    expected = job.get("appworld_policy_sha256", plan.get("appworld_policy_sha256"))
                    if expected is not None and sha != expected:
                        raise ValueError("AppWorld policy changed after the launch manifest was frozen")
                    spec.update(appworld_policy_override=policy, appworld_policy_sha256=sha)
            result.append(spec)
    return result


def ace(spec, reader):
    folder = Path(spec["train_dir"])
    cfg = folder / "experiment_config.json"
    if not cfg.exists():
        return []
    order = reader.json(cfg)["task_order"]
    metrics = {}
    metrics_path = folder / "online_metrics.jsonl"
    if metrics_path.exists():
        lines = reader.raw(metrics_path).decode().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError:
                if index == len(lines) - 1:
                    break
                raise
            if (index >= len(order) or row["session_index"] != index
                    or (row["benchmark_slug"], str(row["task_id"])) != tuple(order[index])):
                raise ValueError("ACE completed metrics differ from the frozen task order")
            if row.get("status") not in {"error", "cancelled"}:
                metrics[index + 1] = row
    records = {}
    for path in sorted(folder.glob("*/sessions/*/config.json"), key=lambda p: p.stat().st_mtime_ns):
        session = path.parent
        checkpoint = session / "agent/playbook_checkpoint.json"
        result = session / "results.json"
        if not result.exists():
            continue
        if reader.json(result).get("status") in {"error", "cancelled"}:
            continue
        config = reader.json(path)
        records[(config["benchmark"], str(config["task_id"]))] = (checkpoint, path)
    recovered = []
    for point, task in enumerate(order, 1):
        record = records.get((task[0], str(task[1])))
        if record is None:
            break
        path, _config = record
        if not path.exists():
            if point not in metrics:
                break  # Actor results can precede curator completion.
            if point % 5:
                continue
            following = records.get(tuple(order[point])) if point < len(order) else None
            if following:
                path = following[1]
                content = reader.json(path).get("agent_kwargs", {}).get("initial_playbook")
                if isinstance(content, str):
                    recovered.append({"point": point, "content": content,
                                      "selection": "ACE completed task; next actor's frozen input playbook",
                                      "source": str(path), "source_sha256": digest(reader.raw(path))})
                    continue
            break  # A boundary still requires evidence of its exact actor policy.
        if point % 5 == 0:
            recovered.append({"point": point, "content": reader.json(path)["playbook"],
                              "selection": "ACE post-curation playbook", "source": str(path),
                              "source_sha256": digest(reader.raw(path))})
    return recovered


def premise(spec, reader):
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.playbook_utils import update_bullet_counts

    folder = Path(spec["train_dir"])
    path = folder / "state.json"
    if not path.exists():
        return []
    raw = reader.raw(path)
    saved = json.loads(raw)
    if saved["config"]["mode"] != "isolated":
        raise ValueError("Checkpoint stream must have isolated training scope")
    beam = [DEFAULT_PLAYBOOK, DEFAULT_PLAYBOOK]
    recovered = []
    order = saved["config"]["order"]
    for index, (benchmark, task_id) in enumerate(order[:saved["cursor"]], 1):
        artifact_path = folder / "iterations" / f"iter_{index:04d}.json"
        artifact = reader.json(artifact_path)
        if (artifact["iteration"], artifact["benchmark"], artifact["task_id"]) != (index, benchmark, task_id):
            raise ValueError("PREMiSE iteration differs from its frozen training order")
        for rank, parent in enumerate(artifact.get("parents", [])):
            if parent:
                config = reader.json(Path(parent["session_dir"]) / "config.json")
                if config["agent_kwargs"]["playbook"] != beam[rank]:
                    raise ValueError(f"PREMiSE parent policy does not match recovered iteration {index}")
        if "selection" in artifact:
            candidates = {item["id"]: item for item in artifact["candidates"]}
            chosen = [candidates[name] for name in artifact["selection"]["selected"]]
            beam = []
            for candidate in chosen:
                tags = [{"id": x.get("entry_id", x.get("id")), "tag": x.get("tag", "neutral")}
                        for x in candidate["reflection"].get("entry_assessments", [])]
                beam.append(update_bullet_counts(candidate["playbook"]["playbook"], tags))
        elif artifact.get("status") != "skipped" or artifact.get("train_update_applied") is not False:
            raise ValueError("PREMiSE skipped update has no unchanged-policy evidence")
        if index % 5 == 0:
            recovered.append({"point": index, "content": beam[0], "selection": "PREMiSE selected beam rank 1",
                              "source": str(artifact_path), "source_sha256": digest(reader.raw(artifact_path))})
    expected = [item["playbook"] for item in saved["states"][spec["benchmark"]]["beam"]]
    if beam != expected:
        raise ValueError("Reconstructed PREMiSE terminal beam differs from the saved state")
    if path.read_bytes() != raw:
        raise BlockingIOError("PREMiSE committed a new iteration during checkpoint discovery")
    return recovered


def gepa(spec, reader):
    output = Path(spec["output"])
    cfg = reader.json(output / "config.json")
    recovered = {}
    for generation in sorted((output / "upstream").glob("replay_*"), reverse=True):
        path = generation / "gepa_state.bin"
        if not path.exists():
            continue
        state = PlainState(io.BytesIO(reader.raw(path))).load()
        trace = state["full_program_trace"]
        born = {entry["iteration_id"]: entry["i"] for entry in trace}
        for entry in trace:
            batch = entry["i"]
            point = min((batch + 1) * 5, len(cfg["train"]))
            if point % 5 or point in recovered or "subsample_ids" not in entry:
                continue
            expected = [x["task_id"] for x in cfg["train"][batch * 5:point]]
            if entry["subsample_ids"] != expected:
                raise ValueError("GEPA batch differs from the ordered train stream")
            eligible = []
            for index, candidate in enumerate(state["program_candidates"]):
                origin = state["iteration_ids_by_candidate_idx"][index]
                if index and born.get(origin, float("inf")) > batch:
                    continue
                scores = state["prog_candidate_val_subscores"][index]
                if len(scores) != len(cfg["validation"]):
                    continue
                score = sum(scores.values()) / len(scores)
                eligible.append((score, -index, candidate["shared"], index))
            if not eligible:
                continue
            score, _, content, candidate_index = max(eligible)
            recovered[point] = {"point": point, "content": content, "selection": "GEPA highest full-val mean as of CP",
                                "selected_val_score": score, "candidate_index": candidate_index,
                                "source": str(path), "source_sha256": digest(reader.raw(path))}
    return [recovered[point] for point in sorted(recovered)]


def skillopt(spec, reader):
    output = Path(spec["output"])
    recovered = {}
    for generation in sorted((output / "upstream").glob("replay_*"), reverse=True):
        path, state = generation / "history.json", generation / "runtime_state.json"
        if not path.exists() or not state.exists():
            continue
        history = reader.json(path)
        completed = reader.json(state)["last_completed_step"]
        best_step = 0
        for expected, row in enumerate(history, 1):
            if row["step"] != expected:
                raise ValueError("SkillOpt step history is not contiguous")
            if expected > completed:
                break
            best_step = row.get("best_step", best_step)
            point = min(expected * 5, spec["train_total"])
            if point % 5 or point in recovered:
                continue
            skill = generation / "skills" / f"skill_v{best_step:04d}.md"
            if not skill.exists():
                continue
            recovered[point] = {"point": point, "content": reader.raw(skill).decode(),
                                "selection": "SkillOpt official validation-gate best skill as of CP",
                                "selected_val_score": row.get("best_score"), "best_step": best_step,
                                "source": str(skill), "source_sha256": digest(reader.raw(skill))}
    return [recovered[point] for point in sorted(recovered)]


def checkpoints(spec, reader):
    return {"ace": ace, "premise": premise, "gepa": gepa, "skillopt": skillopt}[spec["method"]](spec, reader)
