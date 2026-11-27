# SPDX-License-Identifier: Apache-2.0
"""Freeze paired original/other-training-task evaluations of archived PEACE candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from exgentic.agents.premise.ablation_selection import select_ablation_beam
from exgentic.agents.premise.pipeline import write_json
from exgentic.agents.premise.selection import select_beam
from exgentic.core.experiment_limits import BENCHMARK_MAX_TURNS
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relocate_trace(trace):
    trace = copy.deepcopy(trace)
    for field in ["session_dir", "terminus_episode_path"]:
        if isinstance(trace.get(field), str):
            trace[field] = trace[field].replace("/outputs/", "/outputs_arr_oct/")
    return trace


def memory_experience(run, item):
    path = Path(run) / item["trace_artifact"]
    artifact = read(path)
    return {
        "benchmark": item["benchmark"], "task_id": item["task_id"],
        "trace": relocate_trace(artifact["parents"][item["parent_index"]]),
        "reflection": copy.deepcopy(item["reflection"]), "source": str(path),
        "source_sha256": digest(path.read_bytes()), "parent_index": item["parent_index"],
    }


def prefix_supported(experience):
    trace, reflection = experience["trace"], experience["reflection"]
    start = reflection["span"]["start"]
    if not 0 <= start < BENCHMARK_MAX_TURNS[trace["benchmark"]]:
        return False
    if trace["benchmark"] == "terminalbench2":
        return bool(trace.get("terminus_episode_path")) and Path(trace["terminus_episode_path"]).is_file()
    return start < len(trace["actions"])


def prepare_panels(source, manifest, output):
    """One terminal eligible update per benchmark, setting and seed; no outcome-based sampling."""
    source, output = Path(source), Path(output)
    split_document = read(manifest)
    panels, exclusions = [], []
    for run in sorted(source.glob("s*_t3")):
        state = read(run / "state.json")
        mode, seed = state["config"]["mode"], state["config"]["seed"]
        sections = select_task_split_sections(split_document, ["bfcl", "appworld", "terminalbench2"], seed)
        trains = {slug: set(validate_task_split(section, slug)["train"]) for slug, section in sections.items()}
        memories = [memory_experience(run, item) for substate in state["states"].values()
                    for item in substate["memory"]]
        lookup = {(item["benchmark"], item["task_id"]): item for item in memories}
        latest = {}
        for path in sorted((run / "iterations").glob("iter_*.json")):
            artifact = read(path)
            if not artifact.get("selection", {}).get("selected"):
                continue
            benchmark = artifact["benchmark"]
            if artifact["task_id"] not in trains[benchmark]:
                raise ValueError("Proposal source is outside its pinned training split")
            candidates = artifact["candidates"]
            if len(candidates) != 4 or not all(c.get("valid") for c in candidates):
                continue
            sampled = artifact.get("past_sample", [])
            if len(sampled) != 3:
                continue
            past = [lookup[item["benchmark"], item["task_id"]] for item in sampled]
            parents = [relocate_trace(trace) for trace in artifact["parents"] if isinstance(trace, dict)]
            if len(parents) != 2:
                continue
            current = [{"trace": parents[c["parent_rank"] - 1], "reflection": c["reflection"]}
                       for c in candidates]
            if not all(prefix_supported(item) for item in [*past, *current]):
                exclusions.append({"source": str(path), "reason": "prefix_exceeds_current_shared_budget_or_missing"})
                continue
            blocked = {(benchmark, artifact["task_id"]), *((p["benchmark"], p["task_id"]) for p in past)}
            alternatives = sorted([
                item for item in memories if item["benchmark"] == benchmark and item["task_id"] in trains[benchmark]
                and (item["benchmark"], item["task_id"]) not in blocked and prefix_supported(item)
            ], key=lambda item: item["task_id"])
            if not alternatives:
                continue
            key = f"s{seed}_{mode}_{benchmark}_i{artifact['iteration']:04d}"
            other = random.Random(f"peace-other-train-v1:{key}").choice(alternatives)
            original_contexts = []
            for parent in parents:
                config_path = Path(parent["session_dir"]) / "config.json"
                config = read(config_path)
                text = config["agent_kwargs"]["playbook"]
                original_contexts.append({"playbook": text, "sha256": digest(text.encode()),
                                          "config_source": str(config_path)})
            latest[benchmark] = {
                "id": key, "kind": "frozen_panel", "mode": mode, "seed": seed, "benchmark": benchmark,
                "iteration": artifact["iteration"], "task_id": artifact["task_id"],
                "source": str(path), "source_sha256": digest(path.read_bytes()),
                "original_parent_contexts": original_contexts, "parents": parents,
                "candidates": copy.deepcopy(candidates), "past": past, "other_training": other,
                "sampling": "Last fully eligible update per benchmark; other task uniformly sampled with a fixed seed",
                "candidate_count": 4, "historical_count": 3, "learning": False,
            }
        for benchmark in ["bfcl", "appworld", "terminalbench2"]:
            if benchmark not in latest:
                raise ValueError(f"No complete fixed panel for {run.name}/{benchmark}")
            panel = latest[benchmark]
            path = output / "panels" / f"{panel['id']}.json"
            write_json(path, panel)
            panels.append({k: panel[k] for k in ["id", "kind", "mode", "seed", "benchmark", "iteration"]} | {
                "panel": str(path.resolve()), "panel_sha256": digest(path.read_bytes()),
            })
    write_json(output / "panel_sampling.json", {"panels": panels, "excluded_rounds": exclusions,
                                              "no_model_calls": True, "validation_and_test_used": False})
    return panels


def evaluate_contexts(pipeline, policies, splits, output, workers=100):
    """Evaluate frozen rank-1 policies; infrastructure errors remain missing and are never selected on."""
    output = Path(output)
    def policy(arm, benchmark):
        value = policies[arm]
        return value["by_benchmark"][benchmark] if "by_benchmark" in value else value

    jobs = [(arm, benchmark, task_id) for arm in policies for benchmark, ids in splits.items() for task_id in ids]
    records = {}
    for arm, benchmark, task in jobs:
        key = digest(json.dumps([arm, benchmark, task]).encode())[:24]
        path = output / "records" / f"{key}.json"
        if path.exists():
            record = read(path)
            if (record["arm"], record["benchmark"], record["task_id"]) != (arm, benchmark, task):
                raise ValueError("Frozen evaluation identity changed")
            if record["policy_sha256"] != digest(policy(arm, benchmark)["playbook"].encode()):
                raise ValueError("Frozen policy changed")
            records[arm, benchmark, task] = record

    def one(arm, benchmark, task, attempt):
        try:
            trace = pipeline._run_agent(
                benchmark=benchmark, task_id=task, playbook=copy.deepcopy(policy(arm, benchmark)),
                label=f"test_{arm}_{benchmark}_{digest(task.encode())[:12]}_a{attempt}",
            )
            if benchmark == "appworld":
                success = trace.get("success")
                if not isinstance(success, bool):
                    success = trace.get("details", {}).get("success")
                if not isinstance(success, bool):
                    raise ValueError("Missing official AppWorld success; cannot infer TGC from PCT")
                score = int(success)
            else:
                score = trace["score"]
                if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Official evaluation score is not a finite fraction")
            return {"status": "completed", "score": score, "partial_score": trace.get("partial_score"),
                    "session_dir": trace["session_dir"], "attempt": attempt}
        except Exception as error:
            return {"status": "error", "score": None, "error": f"{type(error).__name__}: {error}",
                    "attempt": attempt}

    for attempt in range(1, 4):
        pending = [job for job in jobs if job not in records or
                   (records[job]["status"] != "completed" and records[job].get("attempt", 0) < attempt)]
        if not pending:
            continue
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {pool.submit(one, *job, attempt): job for job in pending}
            for future in as_completed(futures):
                arm, benchmark, task = futures[future]
                record = {"arm": arm, "benchmark": benchmark, "task_id": task,
                          "policy_sha256": digest(policy(arm, benchmark)["playbook"].encode()), **future.result()}
                records[arm, benchmark, task] = record
                key = digest(json.dumps([arm, benchmark, task]).encode())[:24]
                write_json(output / "records" / f"{key}.json", record)
                write_json(output / "progress.json", {
                    "total": len(jobs), "completed": sum(r["status"] == "completed" for r in records.values()),
                    "errors": sum(r["status"] == "error" for r in records.values()),
                })
    groups = []
    for arm in policies:
        for benchmark, ids in splits.items():
            measured = [records[arm, benchmark, task]["score"] for task in ids
                        if records[arm, benchmark, task]["status"] == "completed"]
            groups.append({"arm": arm, "benchmark": benchmark, "planned": len(ids), "measured": len(measured),
                           "score": sum(measured) / len(measured) if measured else None,
                           "infrastructure_errors": len(ids) - len(measured)})
    result = {"complete": all(r["status"] == "completed" for r in records.values()) and len(records) == len(jobs),
              "groups": groups, "learning": False}
    write_json(output / "summary.json", result)
    return result


def run_panel(pipeline, panel_path, output, manifest):
    panel, output = read(panel_path), Path(output)
    if digest(Path(panel["source"]).read_bytes()) != panel["source_sha256"]:
        raise ValueError("Archived candidate source changed")
    candidates = copy.deepcopy(panel["candidates"])
    by_id = {c["id"]: c for c in candidates}
    jobs = []
    for candidate in candidates:
        original = {"trace": panel["parents"][candidate["parent_rank"] - 1], "reflection": candidate["reflection"]}
        for role, experience in [("original_current", original), ("other_current", panel["other_training"]),
                                 *[(f"past{i}", value) for i, value in enumerate(panel["past"])]]:
            jobs.append((candidate["id"], role, experience))
    results = {}
    pending = []
    for identity, role, experience in jobs:
        path = output / "scoring" / f"{identity}_{role}.json"
        if path.exists():
            value = read(path)
            if value.get("status") == "completed":
                results[identity, role] = value["result"]
                continue
        pending.append((identity, role, experience))
    with ThreadPoolExecutor(max_workers=min(100, len(pending)) or 1) as pool:
        futures = {
            pool.submit(pipeline._score_job, by_id[cid], item["trace"], item["reflection"],
                        f"{cid}_{role}"): (cid, role)
            for cid, role, item in pending
        }
        for future in as_completed(futures):
            cid, role = futures[future]
            try:
                value = future.result()
                results[cid, role] = value
                record = {"status": "completed", "result": value}
            except Exception as error:
                record = {"status": "error", "score": None, "error": f"{type(error).__name__}: {error}"}
            write_json(output / "scoring" / f"{cid}_{role}.json", record)
    if len(results) != len(jobs):
        raise RuntimeError("Paired panel incomplete; do not select from unequal candidate pools")
    selections, policies = {}, {}
    for arm, role in [("original_task", "original_current"), ("other_training_task", "other_current")]:
        scored = []
        for candidate in candidates:
            current = results[candidate["id"], role]
            past = [results[candidate["id"], f"past{i}"] for i in range(3)]
            clone = {**candidate, "current_result": current, "past_results": past,
                     "current_alignment": current["alignment"]["score"],
                     "past_alignment": sum(r["alignment"]["score"] for r in past) / 3,
                     "current_official_score": current["official_score"],
                     "past_official_score": sum(r["official_score"] for r in past) / 3,
                     "current_candidate_score": current["candidate_score"],
                     "past_candidate_score": sum(r["candidate_score"] for r in past) / 3}
            scored.append(clone)
        selected, evidence = select_beam(scored, score_key="candidate_score")
        selections[arm] = evidence
        policies[arm] = selected[0]["playbook"]
        if arm == "original_task":
            for rule in ["pfs_first", "joint_pareto"]:
                chosen, detail = select_ablation_beam(scored, rule, score_key="candidate_score")
                selections[rule] = detail
                policies[rule] = chosen[0]["playbook"]
    write_json(output / "selections.json", {"panel": panel["id"], "selections": selections,
                                           "rank_one_test_policy_hashes": {
                                               key: digest(value["playbook"].encode())
                                               for key, value in policies.items()
                                           }})
    sections = select_task_split_sections(read(manifest), [panel["benchmark"]], panel["seed"])
    ids = validate_task_split(sections[panel["benchmark"]], panel["benchmark"])["test"]
    return evaluate_contexts(pipeline, policies, {panel["benchmark"]: ids}, output / "evaluation")
