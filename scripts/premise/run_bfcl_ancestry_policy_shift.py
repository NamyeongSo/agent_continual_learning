# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Probe the BFCL-stage final policy and all recorded ancestors for three sequential methods."""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import fcntl
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.premise.run_bfcl_policy_shift import (  # noqa: E402
    METRICS,
    call_keys,
    normalize_calls,
    response_message,
    save_immutable,
)
from scripts.utils.evaluate_isolated_checkpoint_queue import digest, read_json, timestamp, write_json  # noqa: E402

MODEL = "openai/Qwen/Qwen3.6-35B-A3B"
SEEDS = (42, 43, 44)
METHODS = ("gepa", "skillopt", "premise")
DISPLAY = {"gepa": "GEPA", "skillopt": "SkillOpt", "premise": "PREMiSE"}
GEPA_RUNS = {42: "17_gepa_sequential_seed42_legacy", 43: "22_gepa_sequential_seed43_legacy",
             44: "23_gepa_sequential_seed44_legacy"}
SKILL_RUNS = {42: "24_skillopt_sequential_seed42_legacy", 43: "28_skillopt_sequential_seed43_legacy",
              44: "31_skillopt_sequential_seed44_legacy"}


class Sources:
    def __init__(self):
        self.checks = {}

    def raw(self, path: Path) -> bytes:
        raw = path.read_bytes()
        self.checks[str(path.resolve())] = digest(raw)
        return raw

    def json(self, path: Path):
        return json.loads(self.raw(path))

    def text(self, path: Path) -> str:
        return self.raw(path).decode("utf-8")


def ancestor_paths(nodes: dict, final: str) -> list[list[str]]:
    """Retain every root-to-final path, including both parents of a merge."""
    def visit(key, active):
        if key in active:
            raise ValueError("Cycle in policy ancestry")
        if key not in nodes:
            raise ValueError(f"Missing policy ancestor: {key}")
        parents = nodes[key]["parents"]
        if len(parents) != len(set(parents)):
            raise ValueError("Duplicate policy parent")
        if not parents:
            return [[key]]
        return [[*path, key] for parent in parents for path in visit(parent, active | {key})]

    return visit(final, set())


def hop_distance(nodes: dict, child: str, ancestor: str) -> int | None:
    queue, seen = deque([(child, 0)]), set()
    while queue:
        node, distance = queue.popleft()
        if node == ancestor:
            return distance
        if node in seen:
            continue
        seen.add(node)
        queue.extend((parent, distance + 1) for parent in nodes[node]["parents"])
    return None


def selected_lineage(nodes: dict, final: str) -> dict:
    paths = ancestor_paths(nodes, final)
    retained = {key for path in paths for key in path}
    depths = {key: max(path.index(key) for path in paths if key in path) for key in retained}
    selected = [dict(nodes[key], depth=depths[key]) for key in sorted(retained, key=lambda k: (depths[k], k))]
    return {"final": final, "roots": sorted(key for key in retained if not nodes[key]["parents"]),
            "paths": paths, "nodes": selected, "excluded_nonancestors": len(nodes) - len(retained)}


def validate_source(config: dict, seed: int, tasks: list[dict], expected: set[str], model_field: str) -> None:
    if config[model_field] != MODEL or (config["mode"], config["seed"]) != ("sequential", seed):
        raise ValueError("Expected the requested model's sequential learning")
    ids = [row["task_id"] for row in tasks if row["benchmark"] == "bfcl"]
    if len(ids) != 50 or len(set(ids)) != 50 or set(ids) != expected:
        raise ValueError("Source learning does not use the fifty frozen BFCL training tasks")


def gepa_lineage(team: Path, seed: int, expected: set[str], sources: Sources) -> dict:
    raw = team / "training_history_20261009/runs" / GEPA_RUNS[seed] / "bfcl/raw"
    result, config = sources.json(raw / "gepa_result.json"), sources.json(raw / "experiment_config.json")
    validate_source(config, seed, config["train_task_ids"], expected, "task_model")
    if result["seed"] != seed or len(result["candidates"]) != len(result["parents"]):
        raise ValueError("Invalid GEPA candidate collection")
    best = result["best_idx"]
    if result["val_aggregate_scores"][best] != max(result["val_aggregate_scores"]):
        raise ValueError("GEPA final candidate is not the recorded validation winner")
    exported = team / f"gepa_sequential_seed{seed}/cp050_bfcl/candidate.json"
    if sources.json(exported)["best_candidate"] != result["candidates"][best]:
        raise ValueError("GEPA BFCL export differs from the selected final candidate")
    nodes = {}
    for index, candidate in enumerate(result["candidates"]):
        parents = result["parents"][index]
        if None in parents and parents != [None]:
            raise ValueError("Ambiguous GEPA root")
        nodes[f"c{index:04d}"] = {"id": f"c{index:04d}", "parents": [f"c{p:04d}" for p in parents if p is not None],
                                  "iteration": index, "prompt": candidate["system_prompt"],
                                  "candidate_index": index, "source": str(raw / "gepa_result.json")}
    return {**selected_lineage(nodes, f"c{best:04d}"), "selection": "BFCL-stage highest validation score",
            "source_run": str(raw), "selection_score": result["val_aggregate_scores"][best]}


def skillopt_lineage(team: Path, seed: int, expected: set[str], sources: Sources) -> dict:
    raw = team / "training_history_20261009/runs" / SKILL_RUNS[seed] / "bfcl/raw"
    task_ids, summary = sources.json(raw / "task_ids.json"), sources.json(raw / "summary.json")
    validate_source(task_ids, seed, task_ids["tasks"]["train"], expected, "model")
    history = sources.json(raw / "history.json")
    root_path = raw / "skills/skill_v0000.md"
    nodes = {"root": {"id": "root", "parents": [], "iteration": 0, "prompt": sources.text(root_path),
                      "source": str(root_path)}}
    current, best = "root", "root"
    for index, record in enumerate(history, 1):
        if record["step"] != index:
            raise ValueError("Missing SkillOpt step; cannot infer the actual parent")
        candidate_path = raw / "steps" / f"step_{index:04d}" / "candidate_skill.md"
        candidate = sources.text(candidate_path)
        key = f"step_{index:04d}"
        nodes[key] = {"id": key, "parents": [current], "iteration": index, "prompt": candidate,
                      "source": str(candidate_path), "action": record["action"]}
        if digest(candidate.encode())[:16] != record["candidate_hash"]:
            raise ValueError("SkillOpt candidate differs from its recorded hash")
        action = record["action"]
        if action in ("accept", "accept_new_best"):
            current = key
        elif action != "reject" and not action.startswith("skip"):
            raise ValueError(f"Unsupported SkillOpt update: {action}")
        if action == "accept_new_best":
            best = key
        origin = "initial_skill" if current == "root" else current
        if record["current_origin"] != origin:
            raise ValueError("SkillOpt recorded origin differs from reconstructed acceptance history")
        # An epoch-end placeholder can rewrite the last version after the final
        # candidate. Earlier saved versions must exactly match accepted/carry state.
        if index < len(history):
            version = sources.text(raw / "skills" / f"skill_v{index:04d}.md")
            if version != nodes[current]["prompt"]:
                raise ValueError("SkillOpt saved current state differs from the reconstructed parent")
    final = "root" if summary["best_step"] == 0 else f"step_{summary['best_step']:04d}"
    if final != best or summary["best_origin"] != ("initial_skill" if final == "root" else final):
        raise ValueError("SkillOpt final selection differs from the saved best origin")
    exported = team / f"skillopt_sequential_seed{seed}/cp050_bfcl/skill.md"
    if (sources.text(raw / "best_skill.md") != nodes[final]["prompt"]
            or sources.text(exported) != nodes[final]["prompt"]):
        raise ValueError("SkillOpt selected export contains an unrecorded rewrite")
    return {**selected_lineage(nodes, final), "selection": "BFCL-stage recorded best skill",
            "source_run": str(raw), "selection_score": summary["best_selection_hard"]}


def premise_lineage(run: Path, seed: int, expected: set[str], sources: Sources, finish: int = 50) -> dict:
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.playbook_utils import update_bullet_counts
    from exgentic.agents.ace.prompts.generator import build_generator_system_prompt

    state = sources.json(run / "state.json")
    config = state["config"]
    tasks = [{"benchmark": b, "task_id": t} for b, t in config["order"][:50]]
    validate_source(config, seed, tasks, expected, "model")
    if config.get("method", "premise") != "premise" or state["cursor"] < finish:
        raise ValueError("Expected completed PREMiSE BFCL learning")
    nodes = {"root": {"id": "root", "parents": [], "iteration": 0, "prompt":
                      build_generator_system_prompt(DEFAULT_PLAYBOOK), "playbook": DEFAULT_PLAYBOOK}}
    beam, skips, verifications = ["root", "root"], [], 0

    def verify_parents(artifact):
        nonlocal verifications
        for rank, parent in enumerate(artifact.get("parents", [])):
            if parent is None:
                continue
            path = Path(parent["session_dir"]) / "config.json"
            saved = sources.json(path)
            if saved["agent_kwargs"]["playbook"] != nodes[beam[rank]]["playbook"]:
                raise ValueError("PREMiSE actual parent prompt differs from the reconstructed selected beam")
            verifications += 1

    for index in range(1, finish + 1):
        path = run / "iterations" / f"iter_{index:04d}.json"
        artifact = sources.json(path)
        if (artifact["iteration"], artifact["benchmark"], artifact["task_id"]) != (
            index, *config["order"][index - 1]
        ):
            raise ValueError("PREMiSE iteration differs from its frozen BFCL order")
        verify_parents(artifact)
        selected = artifact.get("selection", {}).get("selected", [])
        if not selected:
            if artifact.get("status") != "skipped" or artifact.get("train_update_applied") is not False:
                raise ValueError("Unexplained PREMiSE missing selection")
            skips.append(index)
            continue
        if len(selected) != 2 or len(set(selected)) != 2:
            raise ValueError("Expected two distinct PREMiSE selected candidates")
        candidates = {c["id"]: c for c in artifact["candidates"]}
        next_beam = []
        for candidate_id in selected:
            candidate = candidates[candidate_id]
            rank = candidate["parent_rank"]
            if rank not in (1, 2) or not candidate["valid"]:
                raise ValueError("Invalid PREMiSE selected parent/candidate")
            tags = [{"id": a.get("entry_id", a.get("id")), "tag": a.get("tag", "neutral")}
                    for a in candidate["reflection"].get("entry_assessments", [])]
            playbook = update_bullet_counts(candidate["playbook"]["playbook"], tags)
            key = f"i{index:04d}_{candidate_id}"
            nodes[key] = {"id": key, "parents": [beam[rank - 1]], "iteration": index, "playbook": playbook,
                          "prompt": build_generator_system_prompt(playbook), "source": str(path),
                          "parent_rank": rank, "mutation_applied": bool(candidate.get("applied"))}
            next_beam.append(key)
        beam = next_beam
    if state["cursor"] > finish:
        verify_parents(sources.json(run / "iterations" / f"iter_{finish + 1:04d}.json"))
    elif [nodes[k]["playbook"] for k in beam] != [b["playbook"] for b in state["states"]["global"]["beam"]]:
        raise ValueError("PREMiSE final reconstructed beam differs from the saved checkpoint")
    return {**selected_lineage(nodes, beam[0]), "selection": f"BFCL CP{finish} post-selection rank-1 beam",
            "source_run": str(run), "skipped_iterations": skips, "parent_prompts_verified": verifications,
            "final_secondary_beam": beam[1]}


def prepare(output: Path, team: Path, premise: Path, bank: Path, base_url: str, workers: int) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    path = output / "prepared.json"
    if path.exists():
        prepared = read_json(path)
        if (prepared["base_url"], prepared["workers"], prepared["team"], prepared["premise_root"],
                prepared["bank_source"]) != (base_url, workers, str(team), str(premise), str(bank)):
            raise ValueError("Resume settings differ from the frozen ancestry study")
        for source, expected in prepared["source_checks"].items():
            if digest(Path(source).read_bytes()) != expected:
                raise ValueError(f"Original source changed: {source}")
        return prepared
    sources = Sources()
    histories = sources.json(bank / "history_bank.json")
    split = sources.json(ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json")
    anchors = histories["anchors"]
    for anchor in anchors:
        raw = sources.raw(Path(anchor["path"]))
        if digest(raw) != anchor["sha256"]:
            raise ValueError("Original Qwen history changed")
        history = json.loads(raw)
        if history["seed"] != anchor["seed"] or history["task_id"] != anchor["task_id"]:
            raise ValueError("Wrong recorded history identity")
    cohorts = []
    for method in METHODS:
        for seed in SEEDS:
            expected = set(validate_task_split(split[f"bfcl_{seed}"], "bfcl")["train"])
            if {a["task_id"] for a in anchors if a["seed"] == seed} != expected:
                raise ValueError("The existing history bank does not cover the frozen BFCL training set")
            if method == "gepa":
                lineage = gepa_lineage(team, seed, expected, sources)
            elif method == "skillopt":
                lineage = skillopt_lineage(team, seed, expected, sources)
            else:
                lineage = premise_lineage(premise / f"s{seed}_sequential_t3", seed, expected, sources)
            for node in lineage["nodes"]:
                if not isinstance(node["prompt"], str) or not node["prompt"].strip():
                    raise ValueError("Empty ancestor policy")
                node["prompt_sha256"] = digest(node["prompt"].encode())
                frozen = output / "policies" / method / f"s{seed}" / (node["id"] + ".json")
                save_immutable(frozen, node)
                node["frozen_path"], node["frozen_sha256"] = str(frozen), digest(frozen.read_bytes())
            cohort = {"method": method, "seed": seed, "key": f"{method}_s{seed}", **lineage}
            save_immutable(output / "lineages" / (cohort["key"] + ".json"), cohort)
            cohorts.append(cohort)
    prepared = {"model": MODEL, "base_url": base_url, "workers": workers, "benchmark": "BFCL",
                "mode": "sequential", "stage": "BFCL stage finish at CP50; subsequent training excluded",
                "team": str(team), "premise_root": str(premise), "bank_source": str(bank),
                "histories": anchors, "cohorts": cohorts, "source_checks": sources.checks,
                "thinking": False, "temperature": 0.0, "max_tokens": 8192, "environment_execution": False,
                "history_source": "Existing Qwen initial-context BFCL train50, at most twenty acting inputs per task",
                "prompt_application": "Exact full system prompt; every later message and tool remains frozen",
                "distance": "Shortest directed ancestor path, measured in recorded policy-generation edges",
                "created_at": timestamp()}
    save_immutable(path, prepared)
    return prepared


def request_body(history: dict, prompt: str) -> dict:
    messages = copy.deepcopy(history["messages"])
    if not messages or messages[0]["role"] != "system":
        raise ValueError("Missing first system prompt")
    messages[0]["content"] = prompt
    return {"model": MODEL.removeprefix("openai/"), "messages": messages, "tools": copy.deepcopy(history["tools"]),
            "tool_choice": "auto", "temperature": 0.0, "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False}}


def probe_path(output: Path, anchor: dict, prompt_sha: str) -> Path:
    key = f"s{anchor['seed']}_{digest(anchor['key'].encode())[:20]}_{prompt_sha[:20]}"
    return output / "probe_results" / (key + ".json")


async def probe_cohort(output: Path, prepared: dict, cohort: dict, finished: list[str]) -> None:
    import httpx

    workers = prepared["workers"]
    policies = {n["prompt_sha256"]: n["prompt"] for n in cohort["nodes"]}
    anchors = [a for a in prepared["histories"] if a["seed"] == cohort["seed"]]
    jobs = [(a, sha) for a in anchors for sha in policies]
    random.Random(cohort["key"]).shuffle(jobs)
    queue = asyncio.Queue()
    processed, errors, active = 0, 0, 0
    started = time.monotonic()
    for anchor, sha in jobs:
        path = probe_path(output, anchor, sha)
        if path.exists():
            old = read_json(path)
            if (old["anchor_sha256"], old["prompt_sha256"], old["model"]) != (anchor["sha256"], sha, MODEL):
                raise ValueError("Cached probe provenance changed")
            if old["status"] == "completed":
                processed += 1
                continue
        queue.put_nowait((anchor, sha))

    def status():
        write_json(output / "status.json", {"phase": "same_history_probes", "controller_pid": os.getpid(),
                   "current_cohort": cohort["key"], "completed_cohorts": finished,
                   "cohorts_total": len(prepared["cohorts"]), "total": len(jobs), "processed": processed,
                   "errors": errors, "active": active, "queued": queue.qsize(), "concurrency": workers,
                   "policy_nodes": len(cohort["nodes"]), "unique_prompts": len(policies),
                   "elapsed_seconds": round(time.monotonic() - started, 1), "environment_execution": False,
                   "updated_at": timestamp()})

    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=workers,
                                                               max_keepalive_connections=workers)) as client:
        async def worker():
            nonlocal processed, errors, active
            while not queue.empty():
                anchor, sha = queue.get_nowait()
                active += 1
                probe_started = time.monotonic()
                history_raw = Path(anchor["path"]).read_bytes()
                if digest(history_raw) != anchor["sha256"]:
                    raise ValueError("Frozen history changed before inference")
                body = request_body(json.loads(history_raw), policies[sha])
                row = {"anchor_key": anchor["key"], "anchor_sha256": anchor["sha256"], "prompt_sha256": sha,
                       "model": MODEL, "seed": anchor["seed"], "task_id": anchor["task_id"],
                       "request_sha256": digest(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()),
                       "started_at": timestamp()}
                attempts = []
                for attempt in range(6):
                    try:
                        response = await client.post(prepared["base_url"] + "/chat/completions", json=body,
                                                     headers={"Authorization": "Bearer EMPTY"})
                        response.raise_for_status()
                        payload = response.json()
                        calls = normalize_calls(response_message(payload))
                        row.update(status="completed", predicted_calls=calls, response=payload)
                        break
                    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                        detail = str(exc)
                        if isinstance(exc, httpx.HTTPStatusError):
                            detail += " " + exc.response.text[:2000]
                        attempts.append({"attempt": attempt + 1, "error": detail, "type": type(exc).__name__})
                        row.update(status="error", error=detail)
                        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404):
                            break
                        if attempt < 5:
                            await asyncio.sleep(min(2 ** attempt, 30))
                row.update(attempts=attempts, completed_at=timestamp(),
                           wall_seconds=time.monotonic() - probe_started)
                write_json(probe_path(output, anchor, sha), row)
                processed += 1
                errors += row["status"] != "completed"
                active -= 1
                queue.task_done()
                if processed % 50 == 0:
                    status()

        async def heartbeat():
            while True:
                status()
                await asyncio.sleep(30)

        ticker = asyncio.create_task(heartbeat())
        try:
            await asyncio.gather(*(worker() for _ in range(min(workers, queue.qsize()))))
        finally:
            ticker.cancel()
        status()


def compiled_calls(calls: list[dict]) -> dict:
    return {mode: (tuple(keys := call_keys(calls, mode == "strict")), Counter(keys)) for mode in ("loose", "strict")}


def compiled_score(predicted: dict, reference: dict) -> dict:
    result = {}
    for mode in ("loose", "strict"):
        pred, pred_counts = predicted[mode]
        ref, ref_counts = reference[mode]
        matched = sum(min(count, ref_counts.get(key, 0)) for key, count in pred_counts.items())
        precision = matched / len(pred) if pred else float(not ref)
        recall = matched / len(ref) if ref else float(not pred)
        result.update({f"{mode}_accuracy": float(pred == ref), f"{mode}_precision": precision,
                       f"{mode}_recall": recall,
                       f"{mode}_f1": 2 * matched / (len(pred) + len(ref)) if pred or ref else 1.0})
    return result


def balanced_pair(common: list[dict], cache: dict, left: str, right: str) -> dict:
    sums, counts = defaultdict(Counter), Counter()
    for anchor in common:
        task = anchor["task_id"]
        values = compiled_score(cache[(anchor["key"], left)], cache[(anchor["key"], right)])
        sums[task].update(values)
        counts[task] += 1
    return {metric: statistics.mean(sums[task][metric] / count for task, count in counts.items())
            if counts else None for metric in METRICS}


def csv_file(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_cohort(output: Path, prepared: dict, cohort: dict) -> dict:
    folder = output / "analysis" / cohort["method"] / f"s{cohort['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    anchors = [a for a in prepared["histories"] if a["seed"] == cohort["seed"]]
    nodes = {n["id"]: n for n in cohort["nodes"]}
    prompts = {n["prompt_sha256"] for n in nodes.values()}
    cache, failures = {}, []
    for anchor in anchors:
        for sha in prompts:
            path = probe_path(output, anchor, sha)
            row = read_json(path) if path.exists() else {"status": "missing"}
            if row["status"] == "completed":
                if (row["anchor_sha256"], row["prompt_sha256"], row["model"]) != (anchor["sha256"], sha, MODEL):
                    raise ValueError("Analysis probe provenance changed")
                cache[(anchor["key"], sha)] = compiled_calls(row["predicted_calls"])
            else:
                failures.append({"anchor_key": anchor["key"], "prompt_sha256": sha, "status": row["status"]})
    common = [a for a in anchors if all((a["key"], sha) in cache for sha in prompts)]
    pairs = []
    for i, left in enumerate(cohort["nodes"]):
        for right in cohort["nodes"][i + 1:]:
            distance = hop_distance(nodes, right["id"], left["id"])
            reverse = hop_distance(nodes, left["id"], right["id"])
            if distance is None and reverse is not None:
                left_node, right_node, distance = right, left, reverse
            else:
                left_node, right_node = left, right
            pairs.append({"method": cohort["method"], "seed": cohort["seed"], "ancestor": left_node["id"],
                          "descendant": right_node["id"], "distance": distance, "comparable": distance is not None,
                          **balanced_pair(common, cache, left_node["prompt_sha256"], right_node["prompt_sha256"])})
    final = nodes[cohort["final"]]
    final_rows = [{"node": n["id"], "iteration": n["iteration"],
                   "distance_to_final": hop_distance(nodes, final["id"], n["id"]),
                   **balanced_pair(common, cache, n["prompt_sha256"], final["prompt_sha256"])}
                  for n in cohort["nodes"]]
    roots = [{"root": root, "node": n["id"], "distance_from_root": distance,
              **balanced_pair(common, cache, n["prompt_sha256"], nodes[root]["prompt_sha256"])}
             for root in cohort["roots"] for n in cohort["nodes"]
             if (distance := hop_distance(nodes, n["id"], root)) is not None]
    distances = sorted({r["distance"] for r in pairs if r["distance"] is not None})
    curves = [{"method": cohort["method"], "seed": cohort["seed"], "distance": distance,
               "pairs": len(group := [r for r in pairs if r["distance"] == distance]),
               **{metric: statistics.mean(r[metric] for r in group) if common else None for metric in METRICS}}
              for distance in distances]
    csv_file(folder / "pairwise_metrics.csv", pairs)
    csv_file(folder / "distance_metrics.csv", curves)
    csv_file(folder / "final_agreement.csv", final_rows)
    csv_file(folder / "root_agreement.csv", roots)
    summary = {"method": cohort["method"], "seed": cohort["seed"], "final": cohort["final"],
               "roots": cohort["roots"], "paths": cohort["paths"], "policy_nodes": len(nodes),
               "unique_prompts": len(prompts), "total_histories": len(anchors), "common_histories": len(common),
               "common_tasks": len({a['task_id'] for a in common}), "failed_probes": len(failures),
               "failures": failures, "aggregation": "Mean acting inputs within task, then equal task weights",
               "errors": "Missing/protocol errors excluded by the all-ancestor successful-history intersection",
               "distance": prepared["distance"], "selection": cohort["selection"], "completed_at": timestamp()}
    write_json(folder / "summary.json", summary)
    return summary


def overview(output: Path, prepared: dict, finished: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "analysis"
    combined, summaries = [], []
    for cohort in prepared["cohorts"]:
        if cohort["key"] not in finished:
            continue
        p = folder / cohort["method"] / f"s{cohort['seed']}"
        summaries.append(read_json(p / "summary.json"))
        path = p / "distance_metrics.csv"
        if path.exists():
            with path.open() as f:
                combined.extend(csv.DictReader(f))
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.15), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.84, wspace=0.18)
    aggregated, correlations = [], []
    for axis, method in zip(axes, METHODS, strict=True):
        axis.set_title(DISPLAY[method], fontsize=10)
        rows = [r for r in combined if r["method"] == method]
        distances = sorted({int(r["distance"]) for r in rows})
        for metric, label, color, marker, style in (
            ("strict_accuracy", "ACC", "#2467A9", "o", "-"), ("strict_f1", "F1", "#D07825", "s", "--")
        ):
            x, y, sd = [], [], []
            for distance in distances:
                group = [r for r in rows if int(r["distance"]) == distance and r[metric]]
                if not group:
                    continue
                values = [1 - float(r[metric]) for r in group]
                mean = statistics.mean(values)
                spread = statistics.stdev(values) if len(values) > 1 else 0.0
                x.append(distance)
                y.append(mean)
                sd.append(spread)
                aggregated.append({"method": method, "distance": distance, "metric": metric,
                                   "mean_shift": mean, "seed_sample_sd": spread, "seeds": len(values)})
            if x:
                r = statistics.correlation(x, y) if len(x) > 1 and len(set(y)) > 1 else None
                correlations.append({"method": method, "metric": metric, "distances": len(x), "pearson_r": r})
                suffix = rf",\ r={r:.4f}" if r is not None else ""
                axis.plot(x, y, color=color, marker=marker, markersize=2.7, linestyle=style,
                          linewidth=1.1, label=rf"$1-\mathrm{{{label}}}{suffix}$")
                axis.fill_between(x, [max(0, v - s) for v, s in zip(y, sd, strict=True)],
                                  [min(1, v + s) for v, s in zip(y, sd, strict=True)], color=color, alpha=0.12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0, 1)
        if not rows:
            axis.text(0.5, 0.5, "Pending", transform=axis.transAxes, ha="center", va="center", color="gray")
        else:
            axis.legend(loc="lower right", fontsize=6.5, frameon=False)
            maximum = max(distances)
            axis.set_xlim(0.5, maximum + 0.5)
            if maximum <= 5:
                axis.set_xticks(distances)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Policy shift")
    fig.supxlabel("Ancestor distance (policy generations)", y=0.035, fontsize=9)
    completed = {m: sum(s["method"] == m for s in summaries) for m in METHODS}
    if len(finished) != len(prepared["cohorts"]):
        fig.text(0.5, 0.97, "Completed seeds: " + "; ".join(f"{DISPLAY[m]} {completed[m]}/3" for m in METHODS),
                 ha="center", fontsize=7, color="gray")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(folder / f"policy_shift_overview.{extension}", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    csv_file(folder / "distance_summary.csv", aggregated)
    csv_file(folder / "correlations.csv", correlations)
    write_json(folder / "summary.json", {"benchmark": "BFCL", "mode": "sequential", "model": MODEL,
               "completed_cohorts": finished, "cohorts": summaries, "correlations": correlations,
               "aggregation": "Equal tasks within seed, mean ancestor pairs at each distance, then equal seeds",
               "bands": "One sample seed SD; no iid significance claims",
               "distance": prepared["distance"],
               "histories": "All-ancestor successful intersection within each seed/method",
               "reference": "New root and ancestor predictions on the existing fixed Qwen history bank",
               "identical_prompt_cache": "Each history/prompt pair queried once; identical copies share predictions",
               "few_points": "GEPA/SkillOpt may have only one or two distances; correlations are descriptive",
               "environment_execution": False, "updated_at": timestamp()})
    caption = (
        "BFCL policy shift along the final policy's ancestry in sequential learning. "
        "All recorded ancestors of the BFCL-stage final GEPA, SkillOpt, and PREMiSE policies are queried "
        "on identical saved Qwen histories. Curves show strict $1-\\mathrm{ACC}$ and $1-\\mathrm{F1}$ "
        "versus directed ancestor distance in policy generations. Tasks and seeds receive equal weight; "
        "bands show one sample seed standard deviation. Each seed and method uses histories with valid "
        "predictions at all retained policies. Correlations are descriptive, particularly for short lineages."
    )
    (folder / "policy_shift_overview.tex").write_text(
        "\\begin{figure*}[t]\n  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{figures/bfcl_ancestry_policy_shift.pdf}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:bfcl-ancestry-policy-shift}}\n\\end{{figure*}}\n")


def run(output: Path, prepared: dict) -> None:
    import httpx

    response = httpx.get(prepared["base_url"] + "/models", timeout=15)
    response.raise_for_status()
    if MODEL.removeprefix("openai/") not in {r["id"] for r in response.json()["data"]}:
        raise ValueError("Endpoint is not serving Qwen")
    for cohort in prepared["cohorts"]:
        for node in cohort["nodes"]:
            if digest(Path(node["frozen_path"]).read_bytes()) != node["frozen_sha256"]:
                raise ValueError("Frozen ancestor policy changed")
    unique = {(a["key"], n["prompt_sha256"]) for c in prepared["cohorts"] for a in prepared["histories"]
              if a["seed"] == c["seed"] for n in c["nodes"]}
    write_json(output / "runtime_manifest.json", {"pid": os.getpid(), "started_at": timestamp(),
               "model": MODEL, "base_url": prepared["base_url"], "workers": prepared["workers"],
               "unique_probes": len(unique), "cohorts": [{"key": c["key"], "nodes": len(c["nodes"]),
                   "paths": len(c["paths"]), "final": c["final"]} for c in prepared["cohorts"]],
               "environment_execution": False})
    finished = []
    for cohort in prepared["cohorts"]:
        print(f"START {cohort['key']}: {len(cohort['nodes'])} ancestors including final/root", flush=True)
        asyncio.run(probe_cohort(output, prepared, cohort, finished))
        write_json(output / "status.json", {"phase": "analyzing", "current_cohort": cohort["key"],
                   "completed_cohorts": finished, "controller_pid": os.getpid(), "updated_at": timestamp()})
        summary = analyze_cohort(output, prepared, cohort)
        finished.append(cohort["key"])
        overview(output, prepared, finished)
        print(f"DONE {cohort['key']}: {summary['common_histories']} common histories; "
              f"{summary['failed_probes']} failed probes", flush=True)
    combined = read_json(output / "analysis/summary.json")
    errors = sum(c["failed_probes"] for c in combined["cohorts"])
    write_json(output / "status.json", {"phase": "complete_with_errors" if errors else "complete",
               "completed_cohorts": finished, "failed_probe_instances": errors, "controller_pid": os.getpid(),
               "updated_at": timestamp(), "environment_execution": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs"
                        / "bfcl_sequential_final_ancestry_policy_shift_qwen_queue150_20261013")
    parser.add_argument("--team", type=Path, default=ROOT / "team_prompt")
    parser.add_argument("--premise-root", type=Path, default=ROOT / "outputs"
                        / "premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928")
    parser.add_argument("--bank", type=Path, default=ROOT / "outputs"
                        / "bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/qwen")
    parser.add_argument("--base-url", default="http://127.0.0.1:18018/v1")
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepared = prepare(output, args.team.resolve(), args.premise_root.resolve(), args.bank.resolve(),
                               args.base_url.rstrip("/"), args.workers)
            if args.prepare_only:
                print(json.dumps({"cohorts": [{"key": c["key"], "nodes": len(c["nodes"]), "paths": c["paths"],
                                  "final": c["final"]} for c in prepared["cohorts"]],
                                  "histories": len(prepared["histories"])}, indent=2))
                return
            run(output, prepared)
        except Exception as exc:
            write_json(output / "status.json", {"phase": "error", "error": str(exc), "controller_pid": os.getpid(),
                                               "updated_at": timestamp()})
            raise


if __name__ == "__main__":
    main()
