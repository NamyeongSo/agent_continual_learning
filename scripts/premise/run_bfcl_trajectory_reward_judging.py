# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Resume bounded, anonymized trajectory judgments without executing BFCL actors."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from bfcl_trajectory_reward_judge import judge_group


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def compact_value(value, limit: int):
    """Preserve every event, marking any further shortening as missing evidence."""
    if isinstance(value, str) and len(value) > limit:
        return {"text_prefix": value[:limit], "truncated": True, "original_chars": len(value)}, 1
    if isinstance(value, list):
        results = [compact_value(item, limit) for item in value]
        return [item[0] for item in results], sum(item[1] for item in results)
    if isinstance(value, dict):
        # Existing truncation markers already represent unavailable evidence.
        if value.get("truncated") is True and "text_prefix" in value:
            prefix = value["text_prefix"]
            if len(prefix) > limit:
                return {**value, "text_prefix": prefix[:limit]}, 1
            return value, 0
        results = {key: compact_value(item, limit) for key, item in value.items()}
        return {key: item[0] for key, item in results.items()}, sum(item[1] for item in results.values())
    return value, 0


def judge_inputs(group: dict, text_limit: int | None = None) -> list[dict]:
    traces = []
    for trace in group["traces"]:
        if not trace["available"]:
            raise ValueError("Missing trajectories must not be passed to the judge")
        compact = trace["compact_trajectory"]
        events, shortened = [], 0
        for event in compact["events"]:
            copy = dict(event)
            if text_limit is not None:
                for field in ("arguments", "result", "text", "content"):
                    if field in copy:
                        copy[field], count = compact_value(copy[field], text_limit)
                        shortened += count
            events.append(copy)
        traces.append({
            "key": trace["context"], "events": events,
            "lost_evidence": compact["lost_evidence"] or shortened > 0,
            "truncated_fields": compact["truncated_fields"] + shortened,
        })
    return traces


def interleave_checkpoints(groups: list[dict]) -> list[dict]:
    """Finish prefixes across model/seed strata rather than all Qwen before Gemma."""
    strata = defaultdict(lambda: defaultdict(list))
    for group in groups:
        strata[(group["model"], group["seed"])][group["checkpoint"]].append(group)
    ordered, keys = [], sorted(strata)
    checkpoints = {key: sorted(strata[key]) for key in keys}
    for index in range(max(map(len, checkpoints.values()))):
        for key in keys:
            if index < len(checkpoints[key]):
                ordered.extend(sorted(strata[key][checkpoints[key][index]], key=lambda item: item["query_position"]))
    return ordered


def counted_chat_attempts(attempts: list[dict]) -> int:
    """Count sent completions, conservatively inferring old transport metadata."""
    count = 0
    for attempt in attempts:
        result = attempt["result"]
        sent = result.get("chat_request_started")
        if sent is None:
            sent = (result.get("prompt_tokens_counted", 24001) <= 24000
                    and result.get("status") != "input_budget_exceeded")
        count += bool(sent)
    return count


def probe_model(base_url: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
            data = json.load(response)
        return any(entry.get("id") == model for entry in data.get("data", []))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def retryable_infrastructure(row: dict) -> bool:
    return (not row["ok"] and row["status"] == "judge_transport_error"
            and counted_chat_attempts(row.get("attempts", [])) < 3)


async def run(args) -> None:
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=args.concurrency))
    folder = args.folder.resolve()
    lock = (folder / "judge_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cache = folder / "group_features.jsonl"
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    groups = interleave_checkpoints([json.loads(line) for line in cache.read_text().splitlines()])
    output = folder / "judge_results"
    output.mkdir(exist_ok=True)
    status_path = folder / "judge_status.json"
    statuses, successes, attempts = {}, {}, []
    started = time.monotonic()
    active = 0
    waiting_for_server = False
    health_lock = asyncio.Lock()
    health_checked = 0.0

    def ingest(group_id, row):
        statuses[group_id] = row["status"]
        successes[group_id] = bool(row["ok"])
        attempts.extend(row.get("attempts", []))

    def save_status(state=None):
        usage = Counter()
        for attempt in attempts:
            result = attempt["result"]
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[field] += (result.get("usage") or {}).get(field, 0) or 0
        atomic_json(status_path, {
            "state": state or ("waiting_server" if waiting_for_server else "running"),
            "updated_at": utc_now(), "source_cache_sha256": cache_hash,
            "total": len(groups), "finished": len(statuses), "successful": sum(successes.values()),
            "status_counts": dict(Counter(statuses.values())), "active": active,
            "pending": len(groups) - len(statuses) - active,
            "attempts": len(attempts), "usage": dict(usage), "elapsed_seconds_this_run": time.monotonic() - started,
            "concurrency": args.concurrency, "base_url": args.base_url, "judge_model": args.model,
            "input_budget": 24000, "output_budget": 2048, "thinking": False, "temperature": 0,
            "per_attempt_timeout": args.timeout, "finite_attempt_limit": 3,
            "chat_requests_started": counted_chat_attempts(attempts),
            "retry_infrastructure": args.retry_infrastructure,
        })

    async def ensure_server():
        nonlocal waiting_for_server, health_checked
        if not args.wait_server:
            return
        async with health_lock:
            if time.monotonic() - health_checked < 10:
                return
            while not await asyncio.to_thread(probe_model, args.base_url, args.model):
                waiting_for_server = True
                save_status()
                await asyncio.sleep(20)
            health_checked = time.monotonic()
            waiting_for_server = False

    async def invalidate_health():
        nonlocal health_checked
        health_checked = 0.0

    queue = asyncio.Queue()
    for group in groups:
        path = output / f"{group['id']}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row.get("source_cache_sha256") != cache_hash or row.get("judge_model") != args.model:
                raise ValueError(f"Incompatible cached judgment: {path}")
            if args.retry_infrastructure and retryable_infrastructure(row):
                history = folder / "judge_history"
                history.mkdir(exist_ok=True)
                archived = history / path.name
                if not archived.exists():
                    atomic_json(archived, row)
                attempts.extend(row.get("attempts", []))
                group["previous_attempts"] = row.get("attempts", [])
                queue.put_nowait(group)
            else:
                ingest(group["id"], row)
        else:
            queue.put_nowait(group)

    preflight_path = folder / "judge_preflight.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}

    async def worker():
        nonlocal active
        while not queue.empty():
            group = queue.get_nowait()
            active += 1
            row = {key: group[key] for key in (
                "id", "model", "key", "seed", "checkpoint", "task_id", "query_position", "train_order_index"
            )}
            previous = group.get("previous_attempts", [])
            row.update(source_cache_sha256=cache_hash, judge_model=args.model, attempts=list(previous))
            try:
                if not all(trace["available"] for trace in group["traces"]):
                    row.update(ok=False, status="missing_trajectory", result=None)
                elif preflight.get("ok") and preflight.get("source_group_id") == group["id"]:
                    row["attempts"] = [{"attempt": 0, "reused_preflight": True,
                                        "wall_seconds": preflight["wall_seconds"], "result": preflight}]
                    row.update(ok=True, status="ok", result=preflight)
                else:
                    limit = None
                    format_failures, transport_failures, compactions = 0, 0, 0
                    while counted_chat_attempts(row["attempts"]) < 3 and format_failures < 3:
                        await ensure_server()
                        attempt = len(row["attempts"])
                        seed = int.from_bytes(hashlib.sha256(f"{group['id']}/{attempt}".encode()).digest()[:4], "big")
                        before = time.monotonic()
                        result = await judge_group(
                            group["public_task"], judge_inputs(group, limit), seed=seed,
                            base_url=args.base_url, model=args.model, timeout=args.timeout,
                            max_input_tokens=24000, max_output_tokens=2048,
                        )
                        row["attempts"].append({"attempt": attempt + 1, "text_limit": limit,
                                                "wall_seconds": time.monotonic() - before, "result": result})
                        row.update(ok=result["ok"], status=result["status"], result=result)
                        if result["ok"]:
                            break
                        if result["status"] == "input_budget_exceeded":
                            compactions += 1
                            if compactions >= 3:
                                break
                            limit = 600 if limit is None else 300
                        elif result["status"] == "judge_transport_error":
                            transport_failures += 1
                            await invalidate_health()
                            if not args.wait_server or transport_failures >= 3:
                                break
                        else:
                            format_failures += 1
                row["completed_at"] = utc_now()
                atomic_json(output / f"{group['id']}.json", row)
                # Previous attempts already contributed to cumulative costs at queue construction.
                statuses[group["id"]] = row["status"]
                successes[group["id"]] = bool(row["ok"])
                attempts.extend(row["attempts"][len(previous):])
            finally:
                active -= 1
                queue.task_done()
                save_status()

    async def heartbeat():
        while True:
            save_status()
            await asyncio.sleep(10)

    save_status()
    pulse = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    finally:
        pulse.cancel()
    save_status("completed" if all(successes.values()) else "completed_with_missing_judgments")
    print(json.dumps(json.loads(status_path.read_text()), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retry-infrastructure", action="store_true")
    parser.add_argument("--wait-server", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 128 or args.timeout <= 0:
        parser.error("Require concurrency 1..128 and a positive timeout")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
