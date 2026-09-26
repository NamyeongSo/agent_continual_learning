# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Two-parent span-replay Pareto optimization across ExGentic benchmarks."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ...core.types import RunConfig
from ...interfaces.lib.api import execute
from .llm import PremiseLLM
from .selection import sample_past, select_beam


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def is_nontransient_task_error(exc: Exception) -> bool:
    """Only skip failures attributable to this task, not service outages."""
    message = str(exc).lower()
    context_markers = (
        "maximum context length", "context length exceeded", "context_length_exceeded",
        "context window exceeded", "prompt is too long", "input tokens exceeds",
    )
    return any(marker in message for marker in context_markers) or (
        isinstance(exc, RuntimeError) and message.startswith("reflection_") and " failed:" in message
    )


def playbook_text(playbook: dict[str, Any], benchmark: str) -> str:
    entries = [
        item for item in playbook.get("entries", [])
        if item.get("benchmark") in (benchmark, "global")
    ]
    return "\n".join(
        f"[{item['id']}] ({item['benchmark']}) helpful={item.get('helpful', 0)} "
        f"harmful={item.get('harmful', 0)} :: {item['content']}"
        for item in entries
    )


def apply_operations(parent: dict[str, Any], operations: list[dict[str, Any]], benchmark: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    child = copy.deepcopy(parent)
    applied = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        op = str(operation.get("op", "")).upper()
        entry_id = str(operation.get("entry_id", ""))
        entries = child["entries"]
        target = next((item for item in entries if item["id"] == entry_id), None)
        if op == "ADD":
            content = str(operation.get("content", "")).strip()
            scope = str(operation.get("benchmark") or benchmark)
            if not content or scope not in (benchmark, "global"):
                continue
            new_id = f"p{child['next_id']:06d}"
            child["next_id"] += 1
            entries.append({"id": new_id, "benchmark": scope, "content": content, "helpful": 0, "harmful": 0})
            applied.append({"op": op, "entry_id": new_id, "content": content, "benchmark": scope})
        elif op == "UPDATE" and target is not None and target["benchmark"] in (benchmark, "global"):
            content = str(operation.get("content", "")).strip()
            if content:
                target["content"] = content
                applied.append({"op": op, "entry_id": entry_id, "content": content})
        elif op == "DELETE" and target is not None and target["benchmark"] in (benchmark, "global"):
            entries.remove(target)
            applied.append({"op": op, "entry_id": entry_id})
    return child, applied


def manifest_order(path: Path, benchmarks: list[str], seed: int, count: int, mode: str) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pools: dict[str, deque[str]] = {}
    configs: dict[str, Any] = {}
    for slug in benchmarks:
        section = manifest["benchmarks"][slug]
        train = [str(item) for item in section["train"]]
        val = [str(item) for item in section.get("val", [])]
        if len(train) < count or len(set(train)) != len(train) or set(train) & set(val):
            raise ValueError(f"Invalid train/val manifest for {slug}")
        order_seed = int(hashlib.md5(f"{seed}_{slug}".encode()).hexdigest(), 16) % (2**31)
        random.Random(order_seed).shuffle(train)
        pools[slug] = deque(train[:count])
        configs[slug] = dict(section.get("benchmark_kwargs") or {})
    if mode in ("isolated", "sequential"):
        slugs = sorted(benchmarks) if mode == "isolated" else [
            s for s in ("bfcl", "appworld", "swebench", "terminalbench2") if s in benchmarks
        ]
        slugs += [s for s in benchmarks if s not in slugs]
        return [(slug, task) for slug in slugs for task in pools[slug]], configs
    if mode != "interleaved":
        raise ValueError(f"Unknown mode {mode}")
    rng = random.Random(seed)
    order = []
    while any(pools.values()):
        slug = rng.choice(sorted(s for s, tasks in pools.items() if tasks))
        order.append((slug, pools[slug].popleft()))
    return order, configs


def trajectory_from_session(root: Path, run_id: str, session_result: Any, benchmark: str) -> dict[str, Any]:
    session_dir = root / run_id / "sessions" / session_result.session_id
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (session_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    observations = {event["step"]: event.get("observation") for event in events if event.get("event") == "observation" and not event.get("initial")}
    actions = [{
        "action": event["action"],
        "action_class": event.get("action_class", "SingleAction"),
        "observation": observations.get(event["step"]),
    } for event in events if event.get("event") == "action"]
    score = session_result.score if session_result.score is not None else (1.0 if session_result.success else 0.0)
    if not 0 <= float(score) <= 1:
        raise ValueError(f"{benchmark} score {score} is outside [0,1]; add an explicit normalization adapter")
    return {
        "benchmark": benchmark, "task_id": str(manifest["task_id"]),
        "task": manifest["task"], "context": manifest.get("context", {}),
        "actions": actions, "score": float(score), "details": session_result.details,
        "session_dir": str(session_dir),
    }


class PremisePipeline:
    def __init__(self, *, output: Path, manifest: Path, benchmarks: list[str], mode: str,
                 seed: int, count: int, past_count: int, model: str, base_url: str,
                 parent_workers: int = 2, reflection_workers: int = 4,
                 mutation_workers: int = 4, replay_workers: int = 28,
                 replay_retries: int = 2,
                 llm_calls: int = 32, benchmark_limits: dict[str, int | None] | None = None,
                 skip_task_errors: str = "none",
                 max_steps: dict[str, int] | None = None):
        if past_count not in (3, 6):
            raise ValueError("past_count must be 3 or 6")
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        # AppWorld writes outside this output tree, so concurrent experiments
        # must not use the same experiment_name for a matching replay label.
        self.run_namespace = hashlib.sha256(str(self.output).encode()).hexdigest()[:10]
        self.manifest = manifest.resolve()
        self.benchmarks = benchmarks
        self.mode, self.seed, self.count = mode, seed, count
        self.past_count = past_count
        self.model, self.base_url = model, base_url
        self.parent_workers = parent_workers
        self.reflection_workers = reflection_workers
        self.mutation_workers = mutation_workers
        self.replay_workers = replay_workers
        if replay_retries < 0:
            raise ValueError("replay_retries must be nonnegative")
        if skip_task_errors not in ("none", "nontransient", "all"):
            raise ValueError("skip_task_errors must be none, nontransient, or all")
        self.replay_retries = replay_retries
        self.skip_task_errors = skip_task_errors
        self.llm = PremiseLLM(model, base_url, max_calls=llm_calls)
        self.rng = random.Random(seed)
        self.order, self.benchmark_configs = manifest_order(self.manifest, benchmarks, seed, count, mode)
        limits = {"appworld": 8, "bfcl": 16, "swebench": 2, "terminalbench2": 2} if benchmark_limits is None else benchmark_limits
        self.benchmark_slots = {
            slug: nullcontext() if limits.get(slug) is None else threading.BoundedSemaphore(limits.get(slug, 4))
            for slug in benchmarks
        }
        self._replay_lock = threading.Lock()
        self._active_replays = 0
        self._peak_replays = 0
        self.max_steps = max_steps or {"appworld": 50, "bfcl": 50, "swebench": 100, "terminalbench2": 100}
        self.states = {slug: self._empty_state() for slug in (benchmarks if mode == "isolated" else ["global"])}
        self.cursor = 0
        self._restore_or_initialize()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        initial = {"entries": [], "next_id": 1}
        return {"beam": [copy.deepcopy(initial), copy.deepcopy(initial)], "memory": []}

    def _restore_or_initialize(self):
        state_path = self.output / "state.json"
        config = {
            "schema_version": 1, "manifest": str(self.manifest),
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "benchmarks": self.benchmarks, "mode": self.mode, "seed": self.seed,
            "count": self.count, "past_count": self.past_count, "model": self.model,
            "order": self.order, "benchmark_configs": self.benchmark_configs,
            "max_steps": self.max_steps, "replay_retries": self.replay_retries,
        }
        if state_path.exists():
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if saved["config"] != json.loads(json.dumps(config)):
                raise ValueError("Existing checkpoint configuration differs from this run")
            self.cursor = int(saved["cursor"])
            self.states = saved["states"]
            self.rng.setstate(ast.literal_eval(saved["rng_state"]))
        else:
            write_json(state_path, {"config": config, "cursor": 0, "states": self.states, "rng_state": repr(self.rng.getstate())})

    def _save(self):
        saved = json.loads((self.output / "state.json").read_text(encoding="utf-8"))
        saved.update({"cursor": self.cursor, "states": self.states, "rng_state": repr(self.rng.getstate())})
        write_json(self.output / "state.json", saved)

    def _run_agent(self, *, benchmark: str, task_id: str, playbook: dict[str, Any],
                   label: str, replay_source: dict[str, Any] | None = None,
                   span_start: int | None = None) -> dict[str, Any]:
        run_id = f"premise_{self.run_namespace}_{label}_{benchmark}".replace("/", "_")
        kwargs = dict(self.benchmark_configs[benchmark])
        kwargs["use_cache"] = False
        # AppWorld's experiment directory is per job, including concurrent
        # executions of the same task under different candidates.
        if benchmark == "appworld":
            env = dict(kwargs.get("env_kwargs") or {})
            env["experiment_name"] = run_id
            kwargs["env_kwargs"] = env
        replay_spec = None
        prefix_action_count = 0
        if replay_source is not None:
            assert span_start is not None
            prefix = replay_source["actions"][:span_start]
            replay_spec = {
                "task_id": str(task_id),
                "actions": [
                    {"action": item["action"], "action_class": item["action_class"]}
                    for item in prefix
                ],
            }
            prefix_action_count = sum(
                len(item["action"].get("actions", [])) if "actions" in item["action"] else 1
                for item in prefix
            )
        step_limit = self.max_steps.get(benchmark, 100)
        continuation_limit = step_limit - (span_start or 0)
        continuation_actions = max(step_limit, 100) - prefix_action_count
        if continuation_limit < 1 or continuation_actions < 1:
            raise ValueError("Replay prefix consumes the entire action or step budget")
        config = RunConfig(
            benchmark=benchmark, agent="premise", task_ids=[str(task_id)],
            output_dir=str(self.output / "sessions"), run_id=run_id,
            model=self.model, benchmark_kwargs=kwargs,
            agent_kwargs={
                "model": self.model,
                "playbook": playbook_text(playbook, benchmark),
                "runner": "direct", "enable_thinking": False,
                "model_settings": {"temperature": 0.0, "max_tokens": 8192},
                "enable_tool_shortlisting": benchmark == "appworld",
                "max_selected_tools": 30,
            },
            replay_prefix=replay_spec, max_steps=continuation_limit,
            max_actions=continuation_actions,
            overwrite_sessions=True,
        )
        result = execute(config=config)
        if len(result.session_results) != 1:
            raise RuntimeError(f"{label}: expected one completed session, got {len(result.session_results)}")
        session_result = result.session_results[0]
        if session_result.status.value in ("error", "cancelled"):
            raise RuntimeError(f"{label}: {session_result.status}: {session_result.details}")
        return trajectory_from_session(self.output / "sessions", run_id, session_result, benchmark)

    def _one_reflection(self, rank: int, index: int, trace: dict[str, Any], playbook: dict[str, Any]):
        last = None
        for _ in range(3):
            try:
                value = self.llm.reflect(trace, playbook, correction=str(last) if last else None)
                value.update({"parent_rank": rank, "id": f"reflection_{index}"})
                return value
            except (ValueError, KeyError) as exc:
                last = exc
        raise RuntimeError(f"reflection_{index} failed: {last}")

    def _one_candidate(self, reflection: dict[str, Any], trace: dict[str, Any], parent: dict[str, Any]):
        try:
            response = self.llm.mutate(trace, parent, reflection)
        except Exception as exc:
            response = {"operations": [], "mutation_summary": f"No-op after mutator error: {type(exc).__name__}: {exc}"}
        operations = response.get("operations")
        child, applied = apply_operations(parent, operations if isinstance(operations, list) else [], trace["benchmark"])
        return {"id": f"candidate_{reflection['id'].rsplit('_', 1)[-1]}",
                "parent_rank": reflection["parent_rank"], "reflection": reflection,
                "playbook": child, "applied": applied,
                "mutation_summary": response.get("mutation_summary", ""), "valid": True}

    def _score_job(self, candidate: dict[str, Any], source: dict[str, Any], reflection: dict[str, Any], label: str):
        benchmark = source["benchmark"]
        for attempt in range(self.replay_retries + 1):
            try:
                with self.benchmark_slots[benchmark]:
                    with self._replay_lock:
                        self._active_replays += 1
                        self._peak_replays = max(self._peak_replays, self._active_replays)
                    try:
                        replay = self._run_agent(
                            benchmark=benchmark, task_id=source["task_id"],
                            playbook=candidate["playbook"], label=f"{label}_try{attempt + 1}",
                            replay_source=source, span_start=reflection["span"]["start"],
                        )
                    finally:
                        with self._replay_lock:
                            self._active_replays -= 1
                break
            except (KeyError, ValueError):
                raise
            except Exception:
                if attempt >= self.replay_retries:
                    raise
        judge = self.llm.judge(source, reflection, replay)
        # The official replay score is retained, while detailed grader output
        # remains available in the session files instead of every iteration.
        replay.pop("details", None)
        return {"replay": replay, "alignment": judge, "pct": replay["score"],
                "replay_attempts": attempt + 1}

    def _memory_trace(self, item: dict[str, Any]) -> dict[str, Any]:
        if "trace" in item:  # Checkpoints created before trace references.
            return item["trace"]
        artifact = json.loads((self.output / item["trace_artifact"]).read_text(encoding="utf-8"))
        return artifact["parents"][item["parent_index"]]

    def _score_candidates(self, iteration: int, candidates: list[dict[str, Any]],
                          parent_traces: list[dict[str, Any]], past: list[dict[str, Any]]):
        jobs = []
        for candidate in candidates:
            rank = candidate["parent_rank"]
            jobs.append((candidate["id"], "current", None, parent_traces[rank - 1], candidate["reflection"]))
            for index, item in enumerate(past):
                jobs.append((candidate["id"], "past", index, item["trace"], item["reflection"]))
        by_id = {candidate["id"]: candidate for candidate in candidates}
        self._peak_replays = 0
        outcomes: dict[tuple[str, str, int | None], dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(self.replay_workers, len(jobs))) as pool:
            futures = {}
            for candidate_id, kind, past_index, source, reflection in jobs:
                label = f"i{iteration:04d}_{candidate_id}_{kind}{past_index if past_index is not None else ''}"
                future = pool.submit(self._score_job, by_id[candidate_id], source, reflection, label)
                futures[future] = (candidate_id, kind, past_index)
            for future in as_completed(futures):
                key = futures[future]
                try:
                    outcomes[key] = future.result()
                except Exception as exc:
                    outcomes[key] = {"error": f"{type(exc).__name__}: {exc}"}
        for candidate in candidates:
            cid = candidate["id"]
            current = outcomes[(cid, "current", None)]
            past_results = [outcomes[(cid, "past", i)] for i in range(len(past))]
            if "error" in current or any("error" in item for item in past_results):
                candidate["valid"] = False
                candidate["errors"] = [item["error"] for item in [current, *past_results] if "error" in item]
                continue
            candidate.update({
                "current_alignment": current["alignment"]["score"],
                "current_pct": current["pct"],
                "past_alignment": sum(item["alignment"]["score"] for item in past_results) / len(past_results) if past_results else None,
                "past_pct": sum(item["pct"] for item in past_results) / len(past_results) if past_results else None,
                "current_result": current,
                "past_results": past_results,
            })

    def step(self) -> dict[str, Any]:
        if self.cursor >= len(self.order):
            raise StopIteration
        iteration = self.cursor + 1
        iteration_started = time.monotonic()
        benchmark, task_id = self.order[self.cursor]
        state_key = benchmark if self.mode == "isolated" else "global"
        state = self.states[state_key]
        parents = [copy.deepcopy(item) for item in state["beam"]]

        def run_parent(rank: int):
            with self.benchmark_slots[benchmark]:
                return self._run_agent(
                    benchmark=benchmark, task_id=task_id,
                    playbook=parents[rank], label=f"i{iteration:04d}_parent{rank + 1}",
                )

        with ThreadPoolExecutor(max_workers=min(self.parent_workers, 2)) as pool:
            futures = [pool.submit(run_parent, rank) for rank in range(2)]
            traces = [future.result() for future in futures]
        parent_seconds = time.monotonic() - iteration_started
        specs = [(rank, index, traces[rank - 1], parents[rank - 1])
                 for rank in (1, 2) for index in (2 * rank - 1, 2 * rank)]
        with ThreadPoolExecutor(max_workers=min(self.reflection_workers, 4)) as pool:
            futures = [pool.submit(self._one_reflection, *spec) for spec in specs]
            reflections = [future.result() for future in futures]
        reflection_seconds = time.monotonic() - iteration_started - parent_seconds
        with ThreadPoolExecutor(max_workers=min(self.mutation_workers, 4)) as pool:
            futures = [pool.submit(self._one_candidate, reflection, traces[reflection["parent_rank"] - 1],
                                   parents[reflection["parent_rank"] - 1]) for reflection in reflections]
            candidates = [future.result() for future in futures]
        mutation_seconds = time.monotonic() - iteration_started - parent_seconds - reflection_seconds
        eligible_memory = [item for item in state["memory"] if not (item["benchmark"] == benchmark and item["task_id"] == task_id)]
        past = [dict(item, trace=self._memory_trace(item))
                for item in sample_past(eligible_memory, self.past_count, self.rng)]
        self._score_candidates(iteration, candidates, traces, past)
        replay_seconds = time.monotonic() - iteration_started - parent_seconds - reflection_seconds - mutation_seconds
        selected, selection = select_beam(candidates)
        new_state = copy.deepcopy(state)
        new_state["beam"] = [copy.deepcopy(item["playbook"]) for item in selected]
        for rank, chosen in enumerate(selected):
            entries = {item["id"]: item for item in new_state["beam"][rank]["entries"]}
            for assessment in chosen["reflection"].get("entry_assessments", []):
                entry = entries.get(str(assessment.get("entry_id", "")))
                tag = assessment.get("tag")
                if entry is not None and tag in ("helpful", "harmful"):
                    entry[tag] = int(entry.get(tag, 0)) + 1
        primary = selected[0]
        for sampled, result in zip(past, primary.get("past_results", [])):
            memory_id = sampled.get("memory_id")
            for entry in new_state["memory"]:
                if entry.get("memory_id") == memory_id:
                    entry["latest_selected_replay_pct"] = result["pct"]
                    entry["latest_selected_replay_candidate_id"] = primary["id"]
                    entry["latest_selected_replay_iteration"] = iteration
                    break
        if primary["applied"]:
            new_state["memory"].append({
                "memory_id": f"{iteration}|{benchmark}|{task_id}",
                "benchmark": benchmark, "task_id": task_id,
                "trace_artifact": f"iterations/iter_{iteration:04d}.json",
                "parent_index": primary["parent_rank"] - 1,
                "reflection": primary["reflection"],
                "accepted_candidate_id": primary["id"],
                "selected_pass_pct": primary["current_pct"],
                "latest_selected_replay_pct": primary["current_pct"],
            })
        artifact = {
            "iteration": iteration, "benchmark": benchmark, "task_id": task_id,
            "parents": traces, "reflections": reflections, "candidates": candidates,
            "past_sample": [{"benchmark": item["benchmark"], "task_id": item["task_id"]} for item in past],
            "selection": selection,
            "parallelism": {"replay_jobs": 4 * (1 + len(past)), "peak_replay_sessions": self._peak_replays,
                            "replay_workers": self.replay_workers},
            "timings_seconds": {"parents": round(parent_seconds, 2), "reflections": round(reflection_seconds, 2),
                                "mutations": round(mutation_seconds, 2), "replays_and_judges": round(replay_seconds, 2)},
        }
        write_json(self.output / "iterations" / f"iter_{iteration:04d}.json", artifact)
        self.states[state_key] = new_state
        self.cursor = iteration
        self._save()
        return artifact

    def run(self):
        while self.cursor < len(self.order):
            try:
                artifact = self.step()
            except Exception as exc:
                if self.skip_task_errors == "none" or (
                    self.skip_task_errors == "nontransient" and not is_nontransient_task_error(exc)
                ):
                    raise
                iteration = self.cursor + 1
                benchmark, task_id = self.order[self.cursor]
                artifact = {
                    "iteration": iteration, "benchmark": benchmark, "task_id": task_id,
                    "status": "skipped", "train_update_applied": False,
                    "error_type": type(exc).__name__, "error_message": str(exc)[:4000],
                }
                write_json(self.output / "iterations" / f"iter_{iteration:04d}.json", artifact)
                # step() commits beam and memory only after successful selection.
                # A skipped task advances the frozen order without training on it.
                self.cursor = iteration
                self._save()
                print(f"[{iteration}/{len(self.order)}] SKIPPED {benchmark}::{task_id} "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"[{artifact['iteration']}/{len(self.order)}] {artifact['benchmark']}::{artifact['task_id']} "
                  f"selected={artifact['selection']['selected']} past={len(artifact['past_sample'])}", flush=True)

    def evaluate_validation(self):
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        jobs = []
        for slug in self.benchmarks:
            state = self.states[slug if self.mode == "isolated" else "global"]
            for index, task_id in enumerate(manifest["benchmarks"][slug].get("val", [])):
                jobs.append((slug, str(task_id), index, copy.deepcopy(state["beam"][0])))

        def one(job):
            slug, task_id, index, playbook = job
            with self.benchmark_slots[slug]:
                result = self._run_agent(benchmark=slug, task_id=task_id,
                                         playbook=playbook, label=f"validation_{slug}_{index:04d}")
            return {"benchmark": slug, "task_id": task_id, "score": result["score"]}

        result_path = self.output / "validation.json"
        results = [None] * len(jobs)
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(previous, list) and len(previous) == len(jobs):
                for index, (record, job) in enumerate(zip(previous, jobs)):
                    if isinstance(record, dict) and (record.get("benchmark"), record.get("task_id")) == job[:2]:
                        results[index] = record
        pending = [(index, job) for index, job in enumerate(jobs) if results[index] is None]
        with ThreadPoolExecutor(max_workers=min(self.replay_workers, len(pending)) or 1) as pool:
            futures = {pool.submit(one, job): index for index, job in pending}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                write_json(result_path, results)
        return results
