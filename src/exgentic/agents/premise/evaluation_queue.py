# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""One work-conserving evaluation queue shared by seeds, checkpoints and benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

from ...utils.context_errors import is_context_length_error
from ...utils.task_splits import select_task_split_sections, validate_task_split
from .pipeline import PremisePipeline, write_json


class CheckpointEvaluationQueue:
    def __init__(self, workers: int = 50):
        if workers < 1:
            raise ValueError("Evaluation worker count must be positive")
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="checkpoint-eval")
        self.groups = {}
        self.futures = {}
        self.submitted = 0
        self.finished = 0

    def enqueue(self, pipeline: PremisePipeline, checkpoint: int,
                benchmarks=("appworld", "terminalbench2", "bfcl")) -> list[tuple]:
        snapshot_bytes = (pipeline.output / "checkpoints" / f"checkpoint_{checkpoint:03d}.json").read_bytes()
        snapshot = json.loads(snapshot_bytes)
        state = json.loads((pipeline.output / "state.json").read_bytes())
        manifest_bytes = pipeline.manifest.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if snapshot["cursor"] != checkpoint or snapshot["config"] != state["config"]:
            raise ValueError("Checkpoint does not match the training configuration")
        if snapshot["config"]["manifest_sha256"] != manifest_hash:
            raise ValueError("Task manifest changed after training")
        sections = select_task_split_sections(json.loads(manifest_bytes), pipeline.benchmarks, pipeline.seed)
        keys, batches = [], []
        for slug in benchmarks:
            ids = validate_task_split(sections[slug], slug)["test"]
            if not ids:
                raise ValueError(f"No test tasks for {slug}")
            key = (str(pipeline.output), checkpoint, slug)
            keys.append(key)
            if key in self.groups:
                continue
            identity = {
                "checkpoint": checkpoint, "benchmark": slug, "task_ids": ids,
                "checkpoint_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "manifest_sha256": manifest_hash, "beam_rank": 1, "learning": False,
            }
            path = pipeline.output / "test_evaluation" / f"cp{checkpoint:03d}_{slug}.json"
            records = [None] * len(ids)
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                records = previous["results"]
                if previous.get("identity") != identity or len(records) != len(ids):
                    raise ValueError(f"Existing evaluation differs: {path}")
                for record, task_id in zip(records, ids, strict=True):
                    if record is not None and record.get("task_id") != task_id:
                        raise ValueError(f"Existing evaluation task differs: {path}")
            state_key = slug if pipeline.mode == "isolated" else "global"
            frozen = copy.deepcopy(snapshot["states"][state_key]["beam"][0])
            self.groups[key] = {"identity": identity, "results": records, "path": path}
            self._save_group(key)
            batches.append([
                (key, index, pipeline, task_id, frozen)
                for index, task_id in enumerate(ids) if records[index] is None
            ])
        # Mix benchmarks immediately. Every free worker takes the next tuple;
        # no benchmark, checkpoint or seed has its own worker quota or barrier.
        for row in zip_longest(*batches):
            for job in row:
                if job is not None:
                    future = self.pool.submit(self._execute, job)
                    self.futures[future] = job[:2]
                    self.submitted += 1
        return keys

    @staticmethod
    def _execute(job):
        key, index, pipeline, task_id, frozen = job
        _, checkpoint, slug = key
        sampling = {}
        try:
            temperature = getattr(pipeline, "evaluation_temperature_overrides", {}).get((checkpoint, slug, task_id))
            sampling = {"temperature": temperature} if temperature is not None else {}
            result = pipeline._run_agent(
                benchmark=slug, task_id=task_id, playbook=copy.deepcopy(frozen),
                label=f"test_cp{checkpoint:03d}_{slug}_{index:04d}",
                **sampling,
            )
            return {"task_id": task_id, "score": result["score"], "status": "completed", **sampling}
        except Exception as exc:
            # Preserve the existing evaluation convention, with execution errors
            # visible separately from tasks scored by the official verifier.
            return {"task_id": task_id, "score": 0.0, "status": "error",
                    "context_error": is_context_length_error(exc), "error": f"{type(exc).__name__}: {exc}", **sampling}

    def _save_group(self, key):
        group = self.groups[key]
        records = group["results"]
        complete = all(record is not None for record in records)
        payload = {"identity": group["identity"], "results": records, "complete": complete}
        if complete:
            payload.update(
                mean_official_score=sum(record["score"] for record in records) / len(records),
                execution_errors=sum(record["status"] == "error" for record in records),
            )
        write_json(group["path"], payload)

    def poll(self):
        for future, (key, index) in list(self.futures.items()):
            if future.done():
                self.groups[key]["results"][index] = future.result()
                del self.futures[future]
                self.finished += 1
                self._save_group(key)

    def complete(self, keys):
        return bool(keys) and all(all(record is not None for record in self.groups[key]["results"]) for key in keys)

    def status(self):
        return {
            "workers": self.workers, "submitted": self.submitted, "finished": self.finished,
            "running": sum(future.running() for future in self.futures),
            "queued": sum(not future.running() and not future.done() for future in self.futures),
        }

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
