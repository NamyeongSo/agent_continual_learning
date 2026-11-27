# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Audit frozen trajectory judgments under exactly reversed anonymous positions.

This independent audit never changes primary judgments or reads evaluation targets.
It selects two tasks from the first planned checkpoint in each model/seed stratum.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

try:
    from .bfcl_trajectory_reward_judge import RISK_METRICS, build_judge_request, judge_group
except ImportError:
    from bfcl_trajectory_reward_judge import RISK_METRICS, build_judge_request, judge_group


AUDIT_SEED = 20261021


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def select_audit_groups(groups: list[dict]) -> list[dict]:
    """Freeze two query positions in the first planned CP, without reading outcomes."""
    strata = defaultdict(list)
    for group in groups:
        if group["model"] in {"Qwen", "Gemma"}:
            strata[(group["model"], group["seed"])].append(group)
    selected = []
    for stratum in sorted(strata):
        first_cp = min(group["checkpoint"] for group in strata[stratum])
        first = sorted((group for group in strata[stratum] if group["checkpoint"] == first_cp),
                       key=lambda group: group["query_position"])
        if len(first) < 2 or len({group["query_position"] for group in first[:2]}) != 2:
            raise ValueError(f"Two unique planned queries are required in {stratum}/{first_cp}")
        selected.extend(first[:2])
    if len(strata) != 6 or len(selected) != 12:
        raise ValueError("The fixed audit requires Qwen/Gemma and three seeds each (12 task groups)")
    return selected


def compact_value(value, limit: int):
    """Reproduce existing recorded text compaction; never invent missing evidence."""
    if isinstance(value, str) and len(value) > limit:
        return {"text_prefix": value[:limit], "truncated": True, "original_chars": len(value)}, 1
    if isinstance(value, list):
        parts = [compact_value(item, limit) for item in value]
        return [part[0] for part in parts], sum(part[1] for part in parts)
    if isinstance(value, dict):
        if value.get("truncated") is True and "text_prefix" in value:
            prefix = value["text_prefix"]
            return ({**value, "text_prefix": prefix[:limit]}, 1) if len(prefix) > limit else (value, 0)
        parts = {key: compact_value(item, limit) for key, item in value.items()}
        return {key: part[0] for key, part in parts.items()}, sum(part[1] for part in parts.values())
    return value, 0


def observable_inputs(group: dict, text_limit: int | None) -> list[dict]:
    """Only observable trace data cross the private/public input boundary."""
    output = []
    for trace in group["traces"]:
        if not trace["available"]:
            raise ValueError("An audit group has a missing trajectory")
        compact = trace["compact_trajectory"]
        events, shortened = [], 0
        for event in compact["events"]:
            copied = dict(event)
            if text_limit is not None:
                for field in ("arguments", "result", "text", "content"):
                    if field in copied:
                        copied[field], count = compact_value(copied[field], text_limit)
                        shortened += count
            events.append(copied)
        output.append({"key": trace["context"], "events": events,
                       "lost_evidence": compact["lost_evidence"] or shortened > 0,
                       "truncated_fields": compact["truncated_fields"] + shortened})
    return output


def original_text_limit(original: dict) -> int | None:
    request_hash = original["result"]["request_sha256"]
    matching = [attempt for attempt in original.get("attempts", [])
                if attempt["result"].get("ok") and attempt["result"].get("request_sha256") == request_hash]
    if not matching:
        raise ValueError("Accepted original request is not identifiable in its measured attempts")
    return matching[-1].get("text_limit")


def reversed_inputs(trajectories: list[dict], original_map: dict, *, seed: int = AUDIT_SEED) -> list[dict]:
    """Invert the builder shuffle so A..F are exactly the original F..A."""
    by_key = {trajectory["key"]: trajectory for trajectory in trajectories}
    original_order = [original_map[identity]["key"] for identity in sorted(original_map)]
    if len(by_key) != len(trajectories) or set(by_key) != set(original_order):
        raise ValueError("Audit traces must exactly match the original anonymous group")
    desired = list(reversed(original_order))
    shuffle = list(range(len(desired)))
    random.Random(seed).shuffle(shuffle)
    arranged = [None] * len(desired)
    for position, input_index in enumerate(shuffle):
        arranged[input_index] = by_key[desired[position]]
    return arranged


def mean_ranks(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = np.empty(len(values), dtype=float)
    for value in np.unique(values):
        positions = np.flatnonzero(values == value)
        ranks[positions] = (np.count_nonzero(values < value) + 1 + np.count_nonzero(values <= value)) / 2
    return ranks


def compare_judgments(original: dict, repeated: dict) -> dict:
    """Compare observable scores and candidate rankings, without held-out targets."""
    if set(original) != set(repeated):
        raise ValueError("Reliability comparison requires the exact same private trace keys")
    contexts = sorted(original)
    rows = []
    for context in contexts:
        first, second = original[context], repeated[context]
        fm, sm = first["metrics"], second["metrics"]
        first_h, second_h = fm["material_violation"]["value"], sm["material_violation"]["value"]
        rows.append({
            "context": context,
            "fulfillment_original": fm["fulfillment"], "fulfillment_reversed": sm["fulfillment"],
            "fulfillment_lower_bound_difference": sm["fulfillment"]["lower_bound"]
            - fm["fulfillment"]["lower_bound"],
            "attainment_changed_turns": sum(a["value"] != b["value"]
                                            for a, b in zip(first["attainment"], second["attainment"], strict=True)),
            "material_original": first_h, "material_reversed": second_h,
            "material_known_flip": first_h is not None and second_h is not None and first_h != second_h,
            "material_unknown_transition": (first_h is None) != (second_h is None),
            "severities": {metric: {"original": fm[metric]["value"], "reversed": sm[metric]["value"]}
                           for metric in RISK_METRICS},
        })
    candidates = [context for context in contexts if context != "parent"]
    first_q = [original[context]["metrics"]["fulfillment"]["lower_bound"] for context in candidates]
    second_q = [repeated[context]["metrics"]["fulfillment"]["lower_bound"] for context in candidates]
    first_ranks, second_ranks = mean_ranks(first_q), mean_ranks(second_q)
    constant = bool(np.std(first_ranks) == 0 or np.std(second_ranks) == 0)
    top_first = {key for key, value in zip(candidates, first_q, strict=True) if value == max(first_q)}
    top_second = {key for key, value in zip(candidates, second_q, strict=True) if value == max(second_q)}
    return {
        "attempts": rows,
        "candidate_fulfillment_spearman": None if constant else float(np.corrcoef(first_ranks, second_ranks)[0, 1]),
        "candidate_spearman_undefined_due_to_ties": constant,
        "candidate_top_original": sorted(top_first), "candidate_top_reversed": sorted(top_second),
        "candidate_top_set_jaccard": len(top_first & top_second) / len(top_first | top_second),
    }


def summarize_comparisons(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row.get("ok")]
    attempts = [attempt for row in accepted for attempt in row["comparison"]["attempts"]]
    differences = [abs(attempt["fulfillment_lower_bound_difference"]) for attempt in attempts]
    correlations = [row["comparison"]["candidate_fulfillment_spearman"] for row in accepted
                    if row["comparison"]["candidate_fulfillment_spearman"] is not None]
    return {
        "task_groups": len(rows), "accepted_groups": len(accepted), "attempt_judgments": len(attempts),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "fulfillment_mean_absolute_difference": float(np.mean(differences)) if differences else None,
        "fulfillment_changed_attempts": sum(value > 0 for value in differences),
        "material_known_flips": sum(attempt["material_known_flip"] for attempt in attempts),
        "material_unknown_transitions": sum(attempt["material_unknown_transition"] for attempt in attempts),
        "candidate_spearman_defined_groups": len(correlations),
        "candidate_mean_spearman_when_defined": float(np.mean(correlations)) if correlations else None,
        "candidate_mean_top_set_jaccard": float(np.mean([
            row["comparison"]["candidate_top_set_jaccard"] for row in accepted])) if accepted else None,
        "scope": "Position sensitivity audit only; no primary scores or held-out targets are changed.",
    }


def model_ready(base_url: str, model: str, timeout: float = 10) -> bool:
    """Cheap health check; never submits a completion or restarts a server."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as response:
            data = json.load(response)
        return model in {entry.get("id") for entry in data.get("data", [])}
    except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError):
        return False


async def run(args) -> None:
    folder = args.folder.resolve()
    cache = folder / "group_features.jsonl"
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    groups = select_audit_groups([json.loads(line) for line in cache.read_text().splitlines()])
    output = folder / "reliability_audit"
    output.mkdir(exist_ok=True)
    result_folder = output / "results"
    result_folder.mkdir(exist_ok=True)
    manifest = {
        "version": "reversed_positions_v1", "source_cache_sha256": cache_hash,
        "selection": "First two query positions of first planned checkpoint, each Qwen/Gemma x seed stratum",
        "group_ids": [group["id"] for group in groups], "seed": AUDIT_SEED,
        "request": "Exact reverse of accepted original anonymous A..F order, same public evidence and rubric",
        "judge_model": args.model, "input_budget": 24000, "output_budget": 2048,
        "max_sent_chat_calls_per_group": args.max_sent_attempts, "max_unsent_tokenization_attempts_per_group": 3,
        "changes_primary_judgments": False, "uses_full_evaluation_targets": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing reliability manifest does not match this frozen audit")
    atomic_json(manifest_path, manifest)
    if args.prepare_only:
        atomic_json(output / "status.json", {"state": "prepared", "updated_at": utc_now(), **manifest})
        print(f"Prepared {len(groups)} groups without any model request: {output}", flush=True)
        return

    started = time.monotonic()
    health_deadline = started + args.health_wait_seconds
    stop_for_health = asyncio.Event()
    health_lock = asyncio.Lock()
    results = {}
    active = 0

    def write_status(state="running"):
        atomic_json(output / "status.json", {
            "state": state, "updated_at": utc_now(), "total": len(groups), "finished": len(results),
            "active": active, "elapsed_seconds": time.monotonic() - started,
            **summarize_comparisons(list(results.values())),
        })

    async def wait_ready() -> bool:
        if stop_for_health.is_set():
            return False
        async with health_lock:
            while not await asyncio.to_thread(model_ready, args.base_url, args.model):
                write_status("waiting_for_judge_server")
                remaining = health_deadline - time.monotonic()
                if remaining <= 0:
                    stop_for_health.set()
                    return False
                await asyncio.sleep(min(args.health_poll_seconds, remaining))
        return True

    queue = asyncio.Queue()
    for group in groups:
        path = result_folder / f"{group['id']}.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and (previous.get("source_cache_sha256") != cache_hash
                         or previous.get("judge_model") != args.model):
            raise ValueError(f"Incompatible cached reliability result: {path}")
        if previous and previous.get("status") not in {"waiting_for_judge_server", "original_judgment_unavailable"}:
            results[group["id"]] = previous
        else:
            queue.put_nowait((group, previous))

    async def worker():
        nonlocal active
        while not queue.empty() and not stop_for_health.is_set():
            group, previous = queue.get_nowait()
            active += 1
            path = result_folder / f"{group['id']}.json"
            row = {key: group[key] for key in ("id", "model", "seed", "checkpoint", "query_position", "task_id")}
            row.update(source_cache_sha256=cache_hash, judge_model=args.model,
                       attempts=list(previous.get("attempts", [])) if previous else [])
            try:
                original_path = folder / "judge_results" / f"{group['id']}.json"
                original = json.loads(original_path.read_text()) if original_path.exists() else None
                if not original or not original.get("ok"):
                    row.update(ok=False, status="original_judgment_unavailable")
                elif original.get("source_cache_sha256") != cache_hash or original.get("judge_model") != args.model:
                    raise ValueError("Original judgment has a different source cache or judge model")
                else:
                    public = observable_inputs(group, original_text_limit(original))
                    reversed_public = reversed_inputs(public, original["result"]["position_map"])
                    _, actual_map = build_judge_request(group["public_task"], reversed_public,
                                                        seed=AUDIT_SEED, model=args.model)
                    desired_keys = list(reversed([original["result"]["position_map"][key]["key"]
                                                  for key in sorted(original["result"]["position_map"])]))
                    if [actual_map[key]["key"] for key in sorted(actual_map)] != desired_keys:
                        raise ValueError("Constructed request did not reverse the original anonymous order")
                    row.update(original_request_sha256=original["result"]["request_sha256"],
                               original_result_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                               original_position_map=original["result"]["position_map"],
                               reversed_position_map=actual_map,
                               text_limit=original_text_limit(original))
                    sent = sum(bool(attempt["result"].get("chat_request_started")) for attempt in row["attempts"])
                    unsent = len(row["attempts"]) - sent
                    while sent < args.max_sent_attempts and unsent < 3:
                        if not await wait_ready():
                            row.update(ok=False, status="waiting_for_judge_server")
                            break
                        before = time.monotonic()
                        result = await judge_group(
                            group["public_task"], reversed_public, seed=AUDIT_SEED, base_url=args.base_url,
                            model=args.model, timeout=args.timeout, max_input_tokens=24000, max_output_tokens=2048,
                        )
                        row["attempts"].append({"attempt": len(row["attempts"]) + 1,
                                                "wall_seconds": time.monotonic() - before,
                                                "completed_at": utc_now(), "result": result})
                        sent += bool(result.get("chat_request_started"))
                        unsent += not bool(result.get("chat_request_started"))
                        row.update(ok=result["ok"], status=result["status"], result=result)
                        if result["ok"]:
                            row["comparison"] = compare_judgments(original["result"]["judgments"], result["judgments"])
                            break
                        if result["status"] in {"invalid_input", "input_budget_exceeded", "token_count_unavailable"}:
                            break
                    row.setdefault("ok", False)
                    row.setdefault("status", "finite_attempt_limit")
                    row.update(sent_chat_calls=sent, unsent_tokenization_attempts=unsent)
                row["completed_at"] = utc_now()
                atomic_json(path, row)
                results[group["id"]] = row
            finally:
                active -= 1
                queue.task_done()
                write_status()

    write_status()
    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    summary = summarize_comparisons(list(results.values()))
    summary.update(updated_at=utc_now(), planned_groups=len(groups),
                   by_model={model: summarize_comparisons([row for row in results.values() if row["model"] == model])
                             for model in ("Qwen", "Gemma")})
    atomic_json(output / "summary.json", summary)
    write_status("waiting_for_judge_server" if stop_for_health.is_set() else "completed")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--health-wait-seconds", type=float, default=1800)
    parser.add_argument("--health-poll-seconds", type=float, default=15)
    parser.add_argument("--max-sent-attempts", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.concurrency <= 16 or args.timeout <= 0 or args.health_wait_seconds < 0
            or not 0 < args.health_poll_seconds <= 60 or not 1 <= args.max_sent_attempts <= 3):
        parser.error("Require concurrency 1..16, timeout >0, health wait >=0, poll in (0,60], sent attempts 1..3")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
