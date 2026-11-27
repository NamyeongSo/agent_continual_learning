# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Preference-grounded rubric discovery and selection from PReMISE (2605.30803).

This is an independent paper adaptation, not the existing beam/playbook trainer.
All supervised operations use train data; validation is evaluation-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist

from . import rubric_prompts as prompts
from .reward_data import BENCHMARKS, digest, write_json


class RubricStageError(RuntimeError):
    """Incomplete required stages cannot silently yield a trained rubric."""


@dataclass(frozen=True)
class RubricConfig:
    target_size: int = 20
    cluster_size: int = 25
    max_levels: int = 10
    pool_size: int = 100
    min_support: int = 10
    mmr_weight: float = 0.7
    preference_threshold: float = 0.6
    min_applicable: int = 10
    criteria_min: int = 3
    criteria_max: int = 5
    score_batch_size: int = 5
    workers: int = 50
    seed: int = 42
    train_limit_per_benchmark: int = 0
    val_limit_per_benchmark: int = 0
    audit_tasks_per_benchmark: int = 0

    def __post_init__(self):
        positive = (
            self.target_size,
            self.cluster_size,
            self.max_levels,
            self.pool_size,
            self.min_support,
            self.min_applicable,
            self.criteria_min,
            self.criteria_max,
            self.score_batch_size,
            self.workers,
        )
        if min(positive) < 1 or self.criteria_max < self.criteria_min or self.criteria_max > 15:
            raise ValueError("Invalid positive counts or extraction range")
        if not 0 <= self.mmr_weight <= 1 or not 0 <= self.preference_threshold <= 1:
            raise ValueError("MMR weight and preference threshold must be probabilities")
        if min(self.train_limit_per_benchmark, self.val_limit_per_benchmark, self.audit_tasks_per_benchmark) < 0:
            raise ValueError("Limits must be nonnegative")


def normalized(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite matrix of embeddings")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero embedding cannot define cosine similarity")
    return array / norms


def cosine_clusters(vectors, target_size: int) -> list[list[int]]:
    """Average-link cosine clustering, recursively split clusters larger than 2T."""
    vectors = normalized(vectors)

    def split(indices: list[int], count: int) -> list[list[int]]:
        if len(indices) == 1:
            return [indices]
        distances = pdist(vectors[indices], metric="cosine").clip(0, 2)
        labels = cut_tree(linkage(distances, method="average"), n_clusters=count).ravel()
        return [
            [idx for idx, label in zip(indices, labels, strict=True) if label == group] for group in sorted(set(labels))
        ]

    groups = split(list(range(len(vectors))), max(1, len(vectors) // target_size))
    result = []
    while groups:
        group = groups.pop(0)
        if len(group) > 2 * target_size:
            groups[:0] = split(group, math.ceil(len(group) / target_size))
        else:
            result.append(group)
    return sorted(result, key=lambda group: group[0])


def validate_partition(value: dict, ids: list[str]) -> dict:
    groups, singletons = value.get("subgroups"), value.get("unmerged")
    if not isinstance(groups, list) or not isinstance(singletons, list):
        raise ValueError("Expected subgroups and unmerged lists")
    if any(not isinstance(group, list) or not 2 <= len(group) <= 5 for group in groups):
        raise ValueError("Each subgroup must contain 2-5 IDs")
    flattened = [item for group in groups for item in group] + singletons
    if any(not isinstance(item, str) for item in flattened) or Counter(flattened) != Counter(ids):
        raise ValueError("Partition must cover every input ID exactly once")
    return value


def normalize_partition(value: dict, ids: list[str]) -> dict:
    """Carry omitted/singleton criteria unchanged; reject invented or duplicated IDs."""
    groups, singletons = value.get("subgroups"), value.get("unmerged")
    if not isinstance(groups, list) or not isinstance(singletons, list):
        raise ValueError("Expected partition lists")
    bounded, carried = [], list(singletons)
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("Invalid subgroup")
        for start in range(0, len(group), 5):
            chunk = group[start : start + 5]
            if len(chunk) == 1:
                carried.extend(chunk)
            else:
                bounded.append(chunk)
    flattened = [item for group in bounded for item in group] + carried
    if any(not isinstance(item, str) for item in flattened):
        raise ValueError("Partition IDs must be strings")
    counts = Counter(flattened)
    unknown = sorted(set(counts) - set(ids))
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    if unknown or duplicated:
        raise ValueError(
            f"Partition must cover every input ID exactly once; unknown={unknown}, duplicated={duplicated}"
        )
    missing = [key for key in ids if key not in counts]
    result = validate_partition({"subgroups": bounded, "unmerged": carried + missing}, ids)
    if missing:
        result["carried_missing"] = missing
    return result


def criterion_text(value: dict, *, minimum_words: int = 20) -> dict:
    name, description = value.get("name"), value.get("description")
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise ValueError("Invalid criterion name")
    if not isinstance(description, str) or not minimum_words <= len(description.split()) <= 150:
        raise ValueError("Invalid criterion description length")
    return {"name": name.strip(), "description": description.strip()}


def parse_scores(value: dict, ids: list[str]) -> list[dict]:
    rows = value.get("scores")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected criterion scores")
    if Counter(row.get("id") for row in rows) != Counter(ids):
        raise ValueError("Score IDs must exactly match the requested criteria")
    for row in rows:
        if type(row.get("applicable")) is not bool:
            raise ValueError("Applicability must be a boolean")
        if any(type(row.get(key)) is not int or not 0 <= row[key] <= 10 for key in ("a", "b")):
            raise ValueError("Scores must be integers in [0,10]")
        if not row["applicable"] and (row["a"] or row["b"]):
            raise ValueError("Inapplicable criteria must use neutral zero scores")
    return rows


def mmr_select(criteria: list[dict], vectors, size: int, weight: float) -> list[dict]:
    """Log-support relevance with a maximum-cosine redundancy penalty."""
    if not criteria:
        return []
    vectors = normalized(vectors)
    if len(criteria) != len(vectors):
        raise ValueError("Criterion/embedding mismatch")
    relevance = np.log1p([row["support"] for row in criteria])
    relevance = relevance / relevance.max()
    selected = []
    while len(selected) < min(size, len(criteria)):
        candidates = []
        for i, row in enumerate(criteria):
            if i in selected:
                continue
            redundancy = float(max(vectors[i] @ vectors[j] for j in selected)) if selected else 0.0
            score = weight * relevance[i] - (1 - weight) * redundancy
            candidates.append((-score, row["id"], i))
        selected.append(min(candidates)[2])
    return [criteria[i] for i in selected]


def preference_statistics(rows: list[dict], labels: dict[str, str]) -> dict:
    applicable = [row for row in rows if row["applicable"]]
    wins = sum((row["a"] > row["b"]) if labels[row["pair_id"]] == "A" else (row["b"] > row["a"]) for row in applicable)
    ties = sum(row["a"] == row["b"] for row in applicable)
    return {
        "applicable": len(applicable),
        "wins": wins,
        "ties": ties,
        "preference_fit": wins / len(applicable) if applicable else None,
    }


class LocalEmbeddings:
    """Run a cached CPU embedding worker using a separate Python environment."""

    def __init__(self, python: str, model: str, worker: Path, cache: Path):
        if not Path(model).is_dir():
            hub = Path(
                os.environ.get(
                    "HF_HUB_CACHE",
                    os.environ.get(
                        "HUGGINGFACE_HUB_CACHE",
                        str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"),
                    ),
                )
            )
            repository = hub / f"models--{model.replace('/', '--')}"
            model = str(repository)
        repository = Path(model)
        if (repository / "snapshots").is_dir():
            reference = repository / "refs/main"
            if reference.exists():
                model = str(repository / "snapshots" / reference.read_text().strip())
            else:
                snapshots = [path for path in (repository / "snapshots").iterdir() if (path / "config.json").exists()]
                if len(snapshots) != 1:
                    raise ValueError("Embedding cache has no unique snapshot; supply an explicit snapshot directory")
                model = str(snapshots[0])
        model = str(Path(model).resolve())
        pooling = Path(model) / "1_Pooling/config.json"
        pooling_config = json.loads(pooling.read_text())
        self.python, self.model, self.worker, self.cache = python, model, worker, cache
        self.identity = {
            "python": python,
            "model": model,
            "pooling": {"config": pooling_config, "normalization": "L2", "device": "CPU", "max_length": 256},
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "model_files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(model).iterdir()
                if path.suffix in (".json", ".safetensors", ".bin", ".txt")
            },
        }

    def __call__(self, texts: list[str]) -> np.ndarray:
        key = digest({"identity": self.identity, "texts": texts})
        output = self.cache / f"{key}.npy"
        if not output.exists():
            source = self.cache / f"{key}.json"
            write_json(source, texts)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false"}
            subprocess.run(
                [self.python, str(self.worker), "--input", str(source), "--output", str(output), "--model", self.model],
                env=env,
                check=True,
                timeout=600,
            )
        vectors = normalized(np.load(output, allow_pickle=False))
        if len(vectors) != len(texts):
            raise ValueError("Embedding cache row mismatch")
        return vectors


class RubricPipeline:
    def __init__(
        self, dataset: Path, output: Path, llm, embeddings, config: RubricConfig, *, retry_errors: bool = False
    ):
        self.dataset, self.output, self.llm, self.embeddings, self.config = dataset, output, llm, embeddings, config
        self.retry_errors = retry_errors
        manifest_bytes = (dataset / "manifest.json").read_bytes()
        self.manifest = json.loads(manifest_bytes)
        self.records = self.manifest["records"]
        if len({r["id"] for r in self.records}) != len(self.records):
            raise ValueError("Duplicate pair IDs")
        if any(r["label"] not in ("A", "B") or r["split"] not in ("train", "val") for r in self.records):
            raise ValueError("Invalid label/split")
        self.by_id = {r["id"]: r for r in self.records}
        groups = {
            split: {(r["benchmark"], r["task_id"]) for r in self.records if r["split"] == split}
            for split in ("train", "val")
        }
        if groups["train"] & groups["val"]:
            raise ValueError("Train/validation task overlap")
        self.train, self.val, self.audit = [], [], []
        for benchmark in BENCHMARKS:
            train = sorted(
                (r for r in self.records if r["split"] == "train" and r["benchmark"] == benchmark),
                key=lambda r: digest([config.seed, r["id"]]),
            )
            val = sorted(
                (r for r in self.records if r["split"] == "val" and r["benchmark"] == benchmark), key=lambda r: r["id"]
            )
            tasks = sorted({r["task_id"] for r in train}, key=lambda task: digest([config.seed, benchmark, task]))
            reserved = set(tasks[: config.audit_tasks_per_benchmark])
            self.audit.extend(r for r in train if r["task_id"] in reserved)
            train = [r for r in train if r["task_id"] not in reserved]
            if not train or not val:
                raise ValueError(f"No discovery train/validation pairs for {benchmark}")
            self.train.extend(train[: config.train_limit_per_benchmark] if config.train_limit_per_benchmark else train)
            self.val.extend(val[: config.val_limit_per_benchmark] if config.val_limit_per_benchmark else val)
        self.visible = {}
        for record in [*self.train, *self.val, *self.audit]:
            value = json.loads((dataset / "pairs" / f"{record['id']}.json").read_text())
            expected = {"task", "context", "public_tools", "common_prefix", "trajectory_a", "trajectory_b"}
            if set(value) != expected or digest(value) != record["input_sha256"]:
                raise ValueError(f"Input contract/hash mismatch for {record['id']}")
            self.visible[record["id"]] = value
        self.identity = {
            "paper": "https://arxiv.org/html/2605.30803v1",
            "implementation": "independent trajectory adaptation",
            "dataset_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "settings": asdict(config),
            "model": llm.model,
            "base_url": llm.base_url,
            "request": {
                "max_tokens": 2048,
                "timeout": llm.timeout,
                "request_retries": llm.request_retries,
                "output_retries": llm.output_retries,
            },
            "embeddings": embeddings.identity,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "prompts_sha256": hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest(),
            "train_ids": [r["id"] for r in self.train],
            "val_ids": [r["id"] for r in self.val],
            "audit_ids": [r["id"] for r in self.audit],
            "labels": "Higher stored official environment SessionScore; ties excluded; AppWorld uses partial score",
        }
        identity_path = output / "config.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != self.identity:
            raise ValueError("Existing rubric run differs in code, prompts, model, inputs or configuration")
        write_json(identity_path, self.identity)
        self.lock = threading.Lock()
        self.request_locks = {}
        self.progress = {"status": "initialized", "pid": os.getpid()}

    def status(self, **values):
        with self.lock:
            self.progress.update(values, updated_at=datetime.now(UTC).isoformat())
            write_json(self.output / "status.json", self.progress)

    def call(
        self,
        stage: str,
        key,
        system: str,
        payload: dict,
        parser=lambda value: value,
        *,
        temperature: float = 0.0,
        llm=None,
    ):
        llm = llm or self.llm
        request = prompts.messages(system, payload)
        identity = {
            "stage": stage,
            "key": key,
            "messages": request,
            "temperature": temperature,
            "model": llm.model,
            "base_url": llm.base_url,
            "run": digest(self.identity),
        }
        path = self.output / "jobs" / stage / f"{digest(identity)}.json"
        with self.lock:
            request_lock = self.request_locks.setdefault(str(path), threading.Lock())
        with request_lock:
            return self._cached_call(path, identity, request, stage, parser, temperature, llm)

    def _cached_call(self, path, identity, request, stage, parser, temperature, llm):
        cached = None
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["identity"] != identity:
                raise ValueError("Cached request mismatch")
            if cached["status"] == "completed":
                return parser(cached["response"])
            if not self.retry_errors:
                raise RubricStageError(f"Cached failed job {stage}/{digest(identity)}; use --retry-errors")
        recovery = []
        try:
            previous_recovery = cached.get("validation_recovery", []) if cached else []
            response = (
                previous_recovery[-1]["response"]
                if previous_recovery
                else llm.ask(request, f"rubric_{stage}", temperature=temperature)
            )
            for attempt in range(3):
                try:
                    parsed = parser(response)
                    break
                except ValueError as exc:
                    recovery.append({"error": str(exc), "response": response})
                    if attempt == 2:
                        raise
                    correction = [
                        *request,
                        {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
                        {
                            "role": "user",
                            "content": (
                                f"Your JSON failed validation: {exc}. Return a corrected JSON object satisfying "
                                "every constraint in the original instructions. Keep criterion descriptions "
                                "concise and general (20-60 words); omit case-specific reasoning, examples, "
                                "and speculation about hidden evaluation. Use only supplied IDs and evidence. "
                                "Do not change the requested criterion count or scoring scale."
                            ),
                        },
                    ]
                    response = llm.ask(correction, f"rubric_{stage}_format_repair", temperature=temperature)
            write_json(
                path,
                {
                    "identity": identity,
                    "status": "completed",
                    "response": response,
                    "validation_recovery": recovery,
                },
            )
            return parsed
        except Exception as exc:
            previous = json.loads(path.read_text()) if path.exists() else None
            write_json(
                path,
                {
                    "identity": identity,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "validation_recovery": recovery,
                    "previous_failure": previous,
                },
            )
            raise

    def jobs(self, stage: str, items: list, function) -> list:
        self.status(status="running", stage=stage, processed=0, total=len(items), errors=0)
        results, errors = [None] * len(items), []
        with ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix=f"rubric-{stage}") as pool:
            futures = {pool.submit(function, item): i for i, item in enumerate(items)}
            for processed, future in enumerate(as_completed(futures), 1):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    errors.append({"index": i, "error_type": type(exc).__name__, "message": str(exc)})
                self.status(processed=processed, errors=len(errors))
        write_json(self.output / "stages" / f"{stage}.json", {"total": len(items), "errors": errors})
        if errors:
            raise RubricStageError(f"{stage}: {len(errors)}/{len(items)} jobs failed; completed jobs are resumable")
        return results

    @staticmethod
    def public_criteria(criteria: list[dict]) -> list[dict]:
        return [{key: row[key] for key in ("id", "name", "description")} for row in criteria]

    @classmethod
    def aliased_criteria(cls, criteria: list[dict]) -> tuple[list[dict], dict[str, str]]:
        aliases = {f"r{i + 1}": row["id"] for i, row in enumerate(criteria)}
        public = [{**row, "id": alias} for alias, row in zip(aliases, cls.public_criteria(criteria), strict=True)]
        return public, aliases

    def extract(self) -> list[dict]:
        def extract_pair(record):
            def parse(value):
                criteria = value.get("criteria")
                if (
                    not isinstance(criteria, list)
                    or not self.config.criteria_min <= len(criteria) <= self.config.criteria_max
                ):
                    raise ValueError("Extraction count outside configured range")
                result = []
                for index, row in enumerate(criteria):
                    text = criterion_text(row)
                    for score in ("model_a_score", "model_b_score"):
                        if type(row.get(score)) is not int or not 0 <= row[score] <= 10:
                            raise ValueError("Invalid extraction score")
                    result.append(
                        {
                            "id": f"c{digest([record['id'], index])[:12]}",
                            **text,
                            "support": 1,
                            "sources": [record["id"]],
                            "extraction_scores": [row["model_a_score"], row["model_b_score"]],
                        }
                    )
                return result

            return self.call(
                "extract",
                record["id"],
                prompts.EXTRACT,
                {
                    "comparison": self.visible[record["id"]],
                    "winner": record["label"],
                    "criterion_count": [self.config.criteria_min, self.config.criteria_max],
                },
                parse,
            )

        criteria = [row for group in self.jobs("extract", self.train, extract_pair) for row in group]
        write_json(self.output / "discovery" / "level_00.json", criteria)
        return criteria

    def consolidate(self, criteria: list[dict]) -> list[dict]:
        for level in range(self.config.max_levels):
            if level >= 2 and len(criteria) <= self.config.pool_size:
                break
            vectors = self.embeddings([row["description"] for row in criteria])
            groups = [[criteria[i] for i in group] for group in cosine_clusters(vectors, self.config.cluster_size)]

            def merge(group, level=level):
                if len(group) == 1:
                    return group
                ids = [row["id"] for row in group]
                public, aliases = self.aliased_criteria(group)

                def parse_partition(value):
                    value = normalize_partition(value, list(aliases))
                    if value.get("carried_missing"):
                        write_json(
                            self.output / "partition_recovery" / f"{level:02d}_{digest(ids)}.json",
                            {
                                "level": level,
                                "input_ids": ids,
                                "omitted_aliases": value["carried_missing"],
                                "carried_unchanged_ids": [aliases[key] for key in value["carried_missing"]],
                                "action": "Preserve omitted criteria unchanged as unmerged; no inferred assignments",
                            },
                        )
                    return {
                        "subgroups": [[aliases[key] for key in subset] for subset in value["subgroups"]],
                        "unmerged": [aliases[key] for key in value["unmerged"]],
                    }

                partition = self.call(
                    "subgroup",
                    [level, ids],
                    prompts.SUBGROUP,
                    {"criteria": public},
                    parse_partition,
                )
                by_id = {row["id"]: row for row in group}
                result = [by_id[key] for key in partition["unmerged"]]
                for subgroup in partition["subgroups"]:
                    members = [by_id[key] for key in subgroup]
                    text = self.call(
                        "describe",
                        [level, sorted(subgroup)],
                        prompts.DESCRIBE,
                        {"criteria": self.public_criteria(members)},
                        criterion_text,
                    )
                    result.append(
                        {
                            "id": f"c{digest([level, sorted(subgroup)])[:12]}",
                            **text,
                            "support": sum(row["support"] for row in members),
                            "sources": sorted({key for row in members for key in row["sources"]}),
                        }
                    )
                return sorted(result, key=lambda row: row["id"])

            updated = [row for group in self.jobs(f"consolidation_{level:02d}", groups, merge) for row in group]
            if sum(row["support"] for row in updated) != sum(row["support"] for row in criteria):
                raise RuntimeError("Consolidation lost source support")
            write_json(self.output / "discovery" / f"level_{level + 1:02d}.json", updated)
            no_reduction = len(updated) >= 0.95 * len(criteria)
            criteria = updated
            if no_reduction:
                break
        return criteria

    def select(self, criteria: list[dict]) -> tuple[list[dict], list[dict]]:
        pool = [row for row in criteria if row["support"] >= self.config.min_support]
        if not pool:
            raise RubricStageError("No criteria meet min-support; inspect discovery or explicitly adjust the setting")
        batches = [
            pool[i : i + self.config.score_batch_size] for i in range(0, len(pool), self.config.score_batch_size)
        ]

        def reusable(group):
            public, aliases = self.aliased_criteria(group)

            def parse(value):
                rows = value.get("criteria", [])
                if Counter(row.get("id") for row in rows) != Counter(list(aliases)):
                    raise ValueError("Genericity IDs do not match")
                if any(type(row.get("reusable")) is not bool for row in rows):
                    raise ValueError("Genericity must be boolean")
                return {aliases[row["id"]]: row["reusable"] for row in rows}

            return self.call(
                "genericity",
                [row["id"] for row in group],
                prompts.GENERICITY,
                {"criteria": public},
                parse,
            )

        checks = {key: value for batch in self.jobs("genericity", batches, reusable) for key, value in batch.items()}
        pool = [row for row in pool if checks[row["id"]]]
        if not pool:
            raise RubricStageError("Genericity filter rejected every supported criterion")
        write_json(self.output / "discovery" / "supported_pool.json", pool)
        rubric = mmr_select(
            pool, self.embeddings([row["description"] for row in pool]), self.config.target_size, self.config.mmr_weight
        )
        write_json(self.output / "rubrics" / "discovered.json", rubric)
        return rubric, pool

    def score_pair(self, record: dict, criteria: list[dict], *, stage: str = "fit") -> list[dict]:
        ids = [row["id"] for row in criteria]
        public, aliases = self.aliased_criteria(criteria)
        rows = self.call(
            stage,
            [record["id"], ids],
            prompts.SCORE,
            {"comparison": self.visible[record["id"]], "criteria": public},
            lambda value: parse_scores(value, list(aliases)),
        )
        return [{**row, "id": aliases[row["id"]], "pair_id": record["id"]} for row in rows]

    def fit(self, pool: list[dict]) -> list[dict]:
        batches = [
            pool[i : i + self.config.score_batch_size] for i in range(0, len(pool), self.config.score_batch_size)
        ]
        tasks = [(record, batch) for record in self.train for batch in batches]
        scores = [
            row for batch in self.jobs("preference_fit", tasks, lambda task: self.score_pair(*task)) for row in batch
        ]
        labels = {record["id"]: record["label"] for record in self.train}
        statistics = []
        for criterion in pool:
            rows = [row for row in scores if row["id"] == criterion["id"]]
            if len(rows) != len(self.train):
                raise RuntimeError("Incomplete preference fit score matrix")
            stats = preference_statistics(rows, labels)
            keep = (
                stats["applicable"] >= self.config.min_applicable
                and stats["preference_fit"] >= self.config.preference_threshold
            )
            statistics.append({"id": criterion["id"], **stats, "keep": keep})
        by_id = {row["id"]: row for row in pool}
        kept = sorted(
            (row for row in statistics if row["keep"]),
            key=lambda row: (-row["preference_fit"], -by_id[row["id"]]["support"], row["id"]),
        )
        # Strict subset and reordering of the supported pool: no textual rewrite or validation feedback.
        repaired = [by_id[row["id"]] for row in kept[: self.config.target_size]]
        write_json(
            self.output / "preference_fit.json",
            {
                "split": "train",
                "statistics": statistics,
                "scores": scores,
                "selection_status": "selected" if repaired else "no_qualifying_criteria",
                "threshold": self.config.preference_threshold,
                "selected_count": len(repaired),
                "max_preference_fit": max(
                    (row["preference_fit"] for row in statistics if row["preference_fit"] is not None), default=None
                ),
            },
        )
        write_json(self.output / "rubrics" / "preference_repaired.json", repaired)
        return repaired

    def judge_pair(
        self,
        record: dict,
        rubric: list[dict],
        swapped: bool = False,
        *,
        temperature: float = 0.0,
        key_suffix=None,
        llm=None,
    ) -> dict:
        visible = dict(self.visible[record["id"]])
        if swapped:
            visible["trajectory_a"], visible["trajectory_b"] = visible["trajectory_b"], visible["trajectory_a"]

        def parse(value):
            if value.get("choice") not in ("A", "B"):
                raise ValueError("Expected A/B verdict")
            return value["choice"]

        choice = self.call(
            "judge",
            [record["id"], swapped, key_suffix],
            prompts.JUDGE,
            {"comparison": visible, "criteria": self.public_criteria(rubric)},
            parse,
            temperature=temperature,
            llm=llm,
        )
        canonical = ("B" if choice == "A" else "A") if swapped else choice
        return {
            "pair_id": record["id"],
            "benchmark": record["benchmark"],
            "swapped": swapped,
            "choice": canonical,
            "reward": int(canonical == record["label"]),
        }

    def evaluate(self, rubrics: dict[str, list[dict]]) -> dict:
        tasks = [
            (name, rubric, record, swapped)
            for name, rubric in rubrics.items()
            for record in self.val
            for swapped in (False, True)
        ]

        def evaluate_one(task):
            name, rubric, record, swapped = task
            try:
                return {"rubric": name, "status": "completed", **self.judge_pair(record, rubric, swapped)}
            except Exception as exc:
                return {
                    "rubric": name,
                    "pair_id": record["id"],
                    "benchmark": record["benchmark"],
                    "swapped": swapped,
                    "status": "error",
                    "error_type": type(exc).__name__,
                }

        rows = self.jobs("validation", tasks, evaluate_one)
        metrics = {}
        for name in rubrics:
            metrics[name] = {}
            for benchmark in BENCHMARKS:
                subset = [row for row in rows if row["rubric"] == name and row["benchmark"] == benchmark]
                errors = sum(row["status"] != "completed" for row in subset)
                normal = [row for row in subset if not row["swapped"] and row["status"] == "completed"]
                reversed_rows = [row for row in subset if row["swapped"] and row["status"] == "completed"]
                lookup = {row["pair_id"]: row["choice"] for row in reversed_rows}
                metrics[name][benchmark] = {
                    "pairs": len(subset) // 2,
                    "errors": errors,
                    "accuracy": sum(row["reward"] for row in normal) / len(normal) if normal and not errors else None,
                    "swapped_accuracy": sum(row["reward"] for row in reversed_rows) / len(reversed_rows)
                    if reversed_rows and not errors
                    else None,
                    "order_flip_rate": sum(row["choice"] != lookup[row["pair_id"]] for row in normal) / len(normal)
                    if normal and not errors
                    else None,
                }
        report = {"split": "val", "learning": False, "by_rubric": metrics, "results": rows}
        write_json(self.output / "validation.json", report)
        return report

    def run(self) -> dict[str, list[dict]]:
        criteria = self.consolidate(self.extract())
        discovered, pool = self.select(criteria)
        repaired = self.fit(pool)
        rubrics = {"no_rubric": [], "discovered": discovered, "preference_repaired": repaired}
        evaluated = (
            rubrics if repaired else {key: value for key, value in rubrics.items() if key != "preference_repaired"}
        )
        report = self.evaluate(evaluated)
        report["preference_selection"] = {
            "status": "selected" if repaired else "no_qualifying_criteria",
            "threshold": self.config.preference_threshold,
            "selected_count": len(repaired),
            "skipped_variants": [] if repaired else ["preference_repaired"],
        }
        write_json(self.output / "validation.json", report)
        errors = sum(row["status"] != "completed" for row in report["results"])
        self.status(
            status="completed_with_errors" if errors else "completed",
            stage="finished",
            errors=errors,
            rubric_sizes={key: len(value) for key, value in rubrics.items()},
            preference_selection=report["preference_selection"],
        )
        return rubrics
