# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Controlled diagnostic of specificity restrictions on frozen seed-42 judge data."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise import reward_pipeline, reward_prompts, rubric_pipeline, rubric_prompts
from exgentic.agents.premise.reward_data import BENCHMARKS, digest, write_json
from exgentic.agents.premise.reward_pipeline import RewardLLM, mutate_rules
from exgentic.agents.premise.rubric_pipeline import LocalEmbeddings, mmr_select

SCOPED = """
Interpret every criterion using the actual task and supplied public API/command contract.
Technical terms, API/field names, pagination, authentication, arithmetic and shell examples can be useful.
Apply their stated mechanism only when relevant; an example does not define an undocumented capability.
A return value, nonempty answer, explicit verification or particular procedure is required only when
the user request or public contract requires it. Optional empty completion arguments can be valid.
Assess achieved outcomes and requested constraints. A recovered intermediate error does not by itself
make an attempt worse. Process efficiency is not proof of outcome correctness. Preserve uncertainty
where omitted observations remove decisive evidence. Do not use benchmark identity or A/B position as a verdict.
Keep the original output schema.
"""

RELAXED_REFLECTOR = (
    reward_prompts.REFLECTOR_SYSTEM.replace(
        "Reject shortcuts based on A/B position, original/replay status, benchmark "
        "identity, tool name, task identifiers,",
        "Reject shortcuts based on A/B position, original/replay status, benchmark identity alone, task identifiers,",
    )
    .replace(
        "Concrete steps may be cited as evidence here, but the proposed reusable "
        "criterion must not memorize their literals.\n"
        "In criterion and limits, abstract concrete cases into requested goals, "
        "target entities, inputs, outputs, constraints,\n"
        "side effects, evidence, error causes and recovery. Omit specific "
        "artifacts, applications, API names and examples.",
        "Concrete technical terms, API/field names, domain mechanisms and short examples may be retained.\n"
        "State their applicability using the current public task/contract, and their exceptions. Do not memorize\n"
        "a task's answer, secret, identifier or winner. A criterion need not apply in every environment.",
    )
    .replace(
        "Never recommend an execution plan, algorithm, application operation or "
        "domain-specific problem-solving technique.",
        "Describe how concrete execution evidence distinguishes the attempts; do "
        "not prescribe an actor execution plan.",
    )
)

RELAXED_MUTATOR = (
    reward_prompts.MUTATOR_SYSTEM.replace(
        "Do not store benchmark names, tool/API names, task IDs, concrete paths, "
        "numeric answers, credentials, winner letters,\n"
        "score thresholds, particular examples, or presumed hidden checks.",
        "Do not store benchmark-based verdict shortcuts, task IDs, concrete paths, "
        "answer keys, credentials, winner letters,\n"
        "score thresholds or presumed hidden checks. Technical terms, tool/API names and short examples are allowed.",
    )
    .replace(
        "side effects, evidence, errors and recovery. Do not include illustrative "
        "examples, quotations from the case, code,\n"
        "function identifiers (including unqualified aliases), or category-specific "
        "actions such as mail or document procedures.",
        "side effects, evidence, errors and recovery. Preserve concrete mechanisms, field/function names and short\n"
        "examples when grounded in the public contract; specify when they apply and when they do not.",
    )
    .replace(
        "Abstract all domain nouns into goals, entities, outputs, constraints or "
        "effects. The same rule must apply outside the\n"
        "task's domain; naming a technical procedure or a particular artifact does "
        "not become general by adding a condition.",
        "A conditional rule may apply only to tasks involving its mechanism. It belongs in the shared playbook.\n"
        "Do not treat an example API as available where the current public contract does not expose it.",
    )
    .replace(
        'Before returning, remove ALL case illustrations, including "e.g.", "for example", and "such as" clauses.',
        "Preserve useful short illustrations while avoiding exact task answers and A/B winner shortcuts.",
    )
)

RESTRICTION_ERRORS = {
    "Rule contains a concrete example or function identifier",
    "Rule describes a particular task domain or artifact",
    "Rule memorizes a concrete task/tool identifier",
}


def relaxed_mutation(parent, response, task_id):
    """Change only specificity bans; preserve structure and metadata-shortcut guards."""
    rules = copy.deepcopy(parent)
    operations = response.get("operations")
    if not isinstance(operations, list) or len(operations) > 3:
        raise ValueError("Expected at most three mutation operations")
    next_id = max((int(row["id"].rsplit("-", 1)[-1]) for row in rules), default=0) + 1
    for operation in operations:
        kind, content = operation.get("type"), operation.get("content", "")
        if kind not in ("ADD", "UPDATE"):
            raise ValueError("Only ADD and UPDATE are supported")
        if not isinstance(content, str) or not content.strip() or len(content) > 1000:
            raise ValueError("Invalid rule content")
        if not content.startswith("Prefer "):
            raise ValueError("Rule must start with Prefer")
        if (task_id and task_id.casefold() in content.casefold()) or re.search(
            r"\bbfcl\b|appworld|terminal[ _-]?bench|multi_turn|/home/|/workspace/|/tmp/", content, re.I
        ):
            raise ValueError("Task/benchmark/path shortcut")
        if re.search(r"\b(?i:(?:always\s+)?(?:choose|select|prefer)\s+(?:trajectory\s+)?)[AB]\b", content):
            raise ValueError("Winner-position shortcut")
        if kind == "ADD":
            if not any(row["content"].casefold() == content.casefold() for row in rules):
                rules.append({"id": f"rule-{next_id:05d}", "content": content.strip()})
                next_id += 1
        else:
            matches = [i for i, row in enumerate(rules) if row["id"] == operation.get("id")]
            if not matches:
                raise ValueError("Unknown UPDATE rule ID")
            rules[matches[0]]["content"] = content.strip()
    return rules


def forbidden(record, visible):
    return [
        record["task_id"],
        *[
            name
            for tool in visible["public_tools"]
            if "_" in tool["name"]
            for name in (tool["name"], tool["name"].rsplit("__", 1)[-1])
        ],
    ]


def strict_mutation(parent, response, record, visible):
    next_id = max((int(row["id"].rsplit("-", 1)[-1]) for row in parent), default=0) + 1
    return mutate_rules(parent, next_id, response, forbidden(record, visible))[0]


def replay_records(current, records, count=10):
    """Hold out the source task from transfer replay; sample unique training task groups."""
    groups = defaultdict(list)
    for row in records.values():
        if row["split"] == "train" and (row["benchmark"], row["task_id"]) != (current["benchmark"], current["task_id"]):
            groups[row["benchmark"], row["task_id"]].append(row)
    queues = {}
    for bench in BENCHMARKS:
        keys = [key for key in groups if key[0] == bench]
        queues[bench] = sorted(keys, key=lambda key: digest([42, current["id"], key]))
    chosen = []
    while len(chosen) < count and any(queues.values()):
        for bench in BENCHMARKS:
            if queues[bench] and len(chosen) < count:
                group = groups[queues[bench].pop(0)]
                chosen.append(min(group, key=lambda row: digest([42, current["id"], row["id"]])))
    if len(chosen) != count:
        raise ValueError("Insufficient independent train task groups")
    return [current, *chosen]


class Analysis:
    def __init__(self, args):
        self.args, self.root = args, args.output
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock, self.request_locks = threading.Lock(), {}
        self.progress = {"pid": os.getpid(), "status": "running"}
        manifest_bytes = (args.rubric_source / "dataset/manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.records = {row["id"]: row for row in manifest["records"]}
        if len(self.records) != len(manifest["records"]):
            raise ValueError("Duplicate frozen pair IDs")
        groups = {
            split: {(r["benchmark"], r["task_id"]) for r in self.records.values() if r["split"] == split}
            for split in ("train", "val")
        }
        if groups["train"] & groups["val"]:
            raise ValueError("Train/validation task overlap")
        self.visible = {}
        for row in self.records.values():
            value = json.loads((args.rubric_source / "dataset/pairs" / f"{row['id']}.json").read_text())
            if digest(value) != row["input_sha256"] or set(value) != {
                "task",
                "context",
                "public_tools",
                "common_prefix",
                "trajectory_a",
                "trajectory_b",
            }:
                raise ValueError("Frozen public input mismatch")
            self.visible[row["id"]] = value
        self.llm = RewardLLM(
            args.base_url, args.model, self.root / "llm_calls", threading.BoundedSemaphore(args.workers)
        )
        self.embeddings = LocalEmbeddings(
            args.embedding_python,
            args.embedding_model,
            ROOT / "scripts/premise/embed_rubric_criteria.py",
            self.root / "embeddings",
        )
        identity = {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "base_url": args.base_url,
            "model": args.model,
            "workers": args.workers,
            "embeddings": self.embeddings.identity,
            "feedback_tokens": 500,
            "output_tokens": 2048,
            "seed": 42,
            "cases_per_benchmark": args.cases_per_benchmark,
            "generation_cases_per_benchmark": args.generation_cases_per_benchmark,
            "protocol": "Development analysis; generation uses train labels only; frozen source results preserved",
            "prompt_sources": {
                "reward": hashlib.sha256(Path(reward_prompts.__file__).read_bytes()).hexdigest(),
                "rubric": hashlib.sha256(Path(rubric_prompts.__file__).read_bytes()).hexdigest(),
            },
            "implementation_sources": {
                "reward": hashlib.sha256(Path(reward_pipeline.__file__).read_bytes()).hexdigest(),
                "rubric": hashlib.sha256(Path(rubric_pipeline.__file__).read_bytes()).hexdigest(),
            },
            "source_artifacts": {
                str(path.relative_to(args.rubric_source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    *(args.rubric_source / "discovery").glob("level_*.json"),
                    args.rubric_source / "discovery/supported_pool.json",
                    args.rubric_source / "rubrics/discovered.json",
                ]
            },
            "source_roots": {"rubric": str(args.rubric_source), "reward": str(args.reward_source)},
        }
        config = self.root / "config.json"
        if config.exists() and json.loads(config.read_text()) != identity:
            raise ValueError("Existing analysis differs")
        self.identity = identity
        write_json(config, identity)
        shutil.copy2(Path(__file__), self.root / "source_snapshot.py")
        write_json(
            self.root / "prompt_variants.json",
            {
                "strict_reflector": reward_prompts.REFLECTOR_SYSTEM,
                "relaxed_reflector": RELAXED_REFLECTOR,
                "strict_mutator": reward_prompts.MUTATOR_SYSTEM,
                "relaxed_mutator": RELAXED_MUTATOR,
                "rubric_plain": rubric_prompts.JUDGE,
                "rubric_scoped": rubric_prompts.JUDGE + SCOPED,
                "reward_judge_unchanged": reward_prompts.JUDGE_SYSTEM,
            },
        )

    def status(self, **kwargs):
        with self.lock:
            self.progress.update(kwargs, updated_at=datetime.now(UTC).isoformat())
            write_json(self.root / "status.json", self.progress)

    def call(self, messages, stage, parser, temperature=0):
        identity = {"messages": messages, "temperature": temperature, "run": digest(self.identity)}
        path = self.root / "jobs" / f"{digest(identity)}.json"
        with self.lock:
            lock = self.request_locks.setdefault(str(path), threading.Lock())
        with lock:
            if path.exists():
                row = json.loads(path.read_text())
                if row["status"] == "completed":
                    return parser(row["response"])
                if not self.args.retry_errors:
                    raise ValueError(f"Cached failed analysis request: {path.name}")
                write_json(self.root / "previous_errors" / path.name, row)
            recoveries = []
            try:
                response = self.llm.ask(messages, stage, temperature=temperature)
                for attempt in range(3):
                    try:
                        value = parser(response)
                        break
                    except ValueError as exc:
                        recoveries.append({"error": str(exc), "response": response})
                        if attempt == 2:
                            raise
                        corrective = [
                            *messages,
                            {"role": "assistant", "content": json.dumps(response)},
                            {
                                "role": "user",
                                "content": f"Validation failed: {exc}. Correct only the output to satisfy "
                                "the original schema and constraints. Keep explanations concise; use only "
                                "supplied evidence.",
                            },
                        ]
                        response = self.llm.ask(corrective, stage + "_format_repair", temperature=temperature)
                write_json(
                    path,
                    {
                        "identity": identity,
                        "status": "completed",
                        "response": response,
                        "validation_recovery": recoveries,
                    },
                )
                return value
            except Exception as exc:
                write_json(
                    path,
                    {
                        "identity": identity,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "validation_recovery": recoveries,
                    },
                )
                raise

    def parallel(self, stage, items, function):
        self.status(stage=stage, processed=0, total=len(items), errors=0)
        results, failures = [None] * len(items), []
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = {executor.submit(function, item): i for i, item in enumerate(items)}
            for processed, future in enumerate(as_completed(futures), 1):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    failures.append({"index": i, "error_type": type(exc).__name__, "error_message": str(exc)})
                self.status(processed=processed, errors=len(failures))
        write_json(self.root / "stages" / f"{stage}.json", {"total": len(items), "errors": failures})
        if failures:
            raise ValueError(f"{stage}: {len(failures)} failed jobs; completed requests are cached")
        return results

    def recover_proposals(self):
        path = self.root / "rejected_proposals.json"
        if path.exists():
            return json.loads(path.read_text())
        by_hash = defaultdict(list)
        for row in self.records.values():
            if row["split"] == "train":
                by_hash[row["input_sha256"]].append(row)
        proposals, counts, seen = [], Counter(), set()
        for variant in ("p3_c3", "p2_c2"):
            for mode in ("sequential", "interleaved"):
                folder = self.args.reward_source / variant / f"s42_{mode}" / "llm_calls"
                self.status(stage="recover_rejected_proposals", source=f"{variant}/{mode}")
                for trace in sorted(folder.glob("*.json")):
                    with trace.open() as handle:
                        header = handle.read(160)
                    if not re.search(r'"stage"\s*:\s*"mutation"', header):
                        continue
                    row = json.loads(trace.read_text())
                    if row["status"] != "completed":
                        continue
                    payload = json.loads(row["request"]["messages"][1]["content"])
                    matching = by_hash.get(digest(payload["comparison"]))
                    if not matching:
                        counts["no_frozen_train_match"] += 1
                        continue
                    record = min(matching, key=lambda item: item["id"])
                    parent, response = payload["criteria"], row["response"]
                    try:
                        strict_mutation(parent, response, record, payload["comparison"])
                        continue
                    except ValueError as exc:
                        reason = str(exc)
                    counts[reason] += 1
                    if reason not in RESTRICTION_ERRORS:
                        continue
                    try:
                        rules = relaxed_mutation(parent, response, record["task_id"])
                    except ValueError as exc:
                        counts[f"still_invalid:{exc}"] += 1
                        continue
                    key = digest([record["id"], parent, response])
                    if key in seen or rules == parent:
                        continue
                    seen.add(key)
                    proposals.append(
                        {
                            "id": key[:16],
                            "record_id": record["id"],
                            "benchmark": record["benchmark"],
                            "task_id": record["task_id"],
                            "rejection": reason,
                            "trace": str(trace),
                            "parent": parent,
                            "rules": rules,
                            "mutation": response,
                            "payload": payload,
                        }
                    )
        write_json(path, proposals)
        write_json(self.root / "restriction_audit.json", {"counts": dict(counts), "recoverable_unique": len(proposals)})
        return proposals

    def select_cases(self, proposals):
        cases, used = [], set()
        for bench in BENCHMARKS:
            candidates = sorted((p for p in proposals if p["benchmark"] == bench), key=lambda p: digest([42, p["id"]]))
            selected = []
            for proposal in candidates:
                group = bench, proposal["task_id"]
                if group not in used:
                    selected.append(proposal)
                    used.add(group)
                    if len(selected) == self.args.cases_per_benchmark:
                        break
            if len(selected) != self.args.cases_per_benchmark:
                raise ValueError(f"Only {len(selected)} distinct rejected-proposal tasks for {bench}")
            cases.extend(selected)
        write_json(self.root / "selected_cases.json", cases)
        return cases

    def rubric_arms(self):
        source = self.args.rubric_source
        levels = sorted((source / "discovery").glob("level_*.json"))
        final = json.loads(levels[-1].read_text())
        supported = [row for row in final if row["support"] >= 10]
        strict = json.loads((source / "rubrics/discovered.json").read_text())
        self.status(stage="embed_supported_pool", processed=0, total=len(supported), errors=0)
        expanded = mmr_select(supported, self.embeddings([r["description"] for r in supported]), 20, 0.7)
        keep = {row["id"] for row in json.loads((source / "discovery/supported_pool.json").read_text())}
        if len(strict) != 20 or len(expanded) != 20:
            raise ValueError("Controlled rubric comparison requires exactly 20 criteria per arm")
        write_json(
            self.root / "rubric_selection.json",
            {
                "strict": strict,
                "expanded": expanded,
                "supported_pool": supported,
                "previously_filtered_in_expanded": [r["id"] for r in expanded if r["id"] not in keep],
                "same_count": len(strict) == len(expanded) == 20,
                "new_discovery": False,
                "validation_used_for_selection": False,
            },
        )
        return {
            "no_rubric": ([], rubric_prompts.JUDGE),
            "strict_plain": (strict, rubric_prompts.JUDGE),
            "expanded_plain": (expanded, rubric_prompts.JUDGE),
            "strict_scoped": (strict, rubric_prompts.JUDGE + SCOPED),
            "expanded_scoped": (expanded, rubric_prompts.JUDGE + SCOPED),
        }

    @staticmethod
    def parse_choice(value):
        if value.get("choice") not in ("A", "B"):
            raise ValueError("Expected choice A or B")
        return value["choice"]

    def judge(self, task):
        visible = copy.deepcopy(self.visible[task["record_id"]])
        if task["swapped"]:
            visible["trajectory_a"], visible["trajectory_b"] = visible["trajectory_b"], visible["trajectory_a"]
        if task["style"] == "rubric":
            criteria = [{key: row[key] for key in ("id", "name", "description")} for row in task["rules"]]
            messages = rubric_prompts.messages(task["system"], {"comparison": visible, "criteria": criteria})
        else:
            messages = reward_prompts.judge_messages(visible, task["rules"])
        result = {key: value for key, value in task.items() if key not in ("rules", "system", "style")}
        try:
            choice = self.call(messages, "specificity_judge", self.parse_choice)
            canonical = ("B" if choice == "A" else "A") if task["swapped"] else choice
            result.update(
                status="completed",
                choice=canonical,
                raw_choice=choice,
                reward=int(canonical == self.records[task["record_id"]]["label"]),
            )
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__, error_message=str(exc))
        return result

    def rubric_tasks(self, arms):
        tasks = []
        for name, (rules, system) in arms.items():
            for record in self.records.values():
                if record["split"] == "val":
                    for swapped in (False, True):
                        tasks.append(
                            {
                                "experiment": "rubric",
                                "arm": name,
                                "record_id": record["id"],
                                "benchmark": record["benchmark"],
                                "task_id": record["task_id"],
                                "swapped": swapped,
                                "style": "rubric",
                                "rules": rules,
                                "system": system,
                            }
                        )
        return tasks

    def case_tasks(self, case, arms, experiment):
        current = self.records[case["record_id"]]
        tasks = []
        for name, rules in arms.items():
            for record in replay_records(current, self.records):
                subset = (
                    "current"
                    if record["id"] == current["id"]
                    else (
                        "same_benchmark_other_task"
                        if record["benchmark"] == current["benchmark"]
                        else "other_benchmark"
                    )
                )
                for swapped in (False, True):
                    tasks.append(
                        {
                            "experiment": experiment,
                            "case_id": case["id"],
                            "source_benchmark": current["benchmark"],
                            "arm": name,
                            "record_id": record["id"],
                            "benchmark": record["benchmark"],
                            "task_id": record["task_id"],
                            "subset": subset,
                            "swapped": swapped,
                            "style": "reward",
                            "rules": rules,
                        }
                    )
        return tasks

    def generate_one(self, item):
        case, condition, replicate = item
        payload = copy.deepcopy(case["payload"])
        payload.pop("reflection", None)
        record = self.records[case["record_id"]]
        assert record["split"] == "train"
        assert payload["training_feedback"]["preferred"] == record["label"]
        # Distinct cache identities make each stochastic replicate an actual pair of requests.
        payload["proposal_replicate"] = replicate
        system = reward_prompts.REFLECTOR_SYSTEM if condition == "strict" else RELAXED_REFLECTOR

        def parse_reflection(value):
            if not all(
                isinstance(value.get(key), str)
                for key in ("diagnosis", "evidence", "criterion", "limits", "uncertainty")
            ):
                raise ValueError("Invalid reflection fields")
            return value

        reflection = self.call(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}],
            "specificity_reflection",
            parse_reflection,
            temperature=0.7,
        )
        payload["reflection"] = reflection
        system = reward_prompts.MUTATOR_SYSTEM if condition == "strict" else RELAXED_MUTATOR
        response = self.call(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}],
            "specificity_mutation",
            lambda value: value,
            temperature=0.7,
        )
        result = {
            "case_id": case["id"],
            "condition": condition,
            "replicate": replicate,
            "reflection": reflection,
            "mutation": response,
        }
        try:
            rules = (
                strict_mutation(case["parent"], response, record, self.visible[record["id"]])
                if condition == "strict"
                else (relaxed_mutation(case["parent"], response, record["task_id"]))
            )
            result.update(status="valid", rules=rules, changed=rules != case["parent"])
        except ValueError as exc:
            result.update(status="rejected", error_message=str(exc))
        return result

    def run(self):
        proposals = self.recover_proposals()
        cases = self.select_cases(proposals)
        arms = self.rubric_arms()
        if self.args.prepare_only:
            self.status(status="prepared", stage="prepared", selected_cases=len(cases))
            return
        generation_cases = []
        for bench in BENCHMARKS:
            generation_cases.extend(
                [c for c in cases if c["benchmark"] == bench][: self.args.generation_cases_per_benchmark]
            )
        generation_items = [
            (case, condition, replicate)
            for case in generation_cases
            for condition in ("strict", "relaxed")
            for replicate in (0, 1)
        ]
        generated = self.parallel("generation", generation_items, self.generate_one)
        write_json(self.root / "generated_proposals.json", generated)
        tasks = self.rubric_tasks(arms)
        for case in cases:
            tasks.extend(self.case_tasks(case, {"parent": case["parent"], "recovered": case["rules"]}, "recovery"))
        for case in generation_cases:
            contexts = {"parent": case["parent"]}
            for proposal in generated:
                if proposal["case_id"] == case["id"] and proposal["status"] == "valid":
                    contexts[f"{proposal['condition']}_{proposal['replicate']}"] = proposal["rules"]
            tasks.extend(self.case_tasks(case, contexts, "generation"))
        order = sorted(
            tasks, key=lambda task: digest([42, {k: v for k, v in task.items() if k not in ("system", "rules")}])
        )
        write_json(
            self.root / "queue_manifest.json",
            {
                "total": len(order),
                "groups": dict(Counter(t["experiment"] for t in order)),
                "workers": self.args.workers,
                "unit": "context, datapoint, orientation",
                "all_experiments_share_queue": True,
            },
        )
        rows = self.parallel("judge_replays", order, self.judge)
        write_json(self.root / "results.json", rows)
        summarize(self.root, rows, generated)
        errors = sum(row["status"] != "completed" for row in rows)
        self.status(status="completed_with_errors" if errors else "completed", stage="finished", errors=errors)


def decision_metrics(rows):
    """Average both orientations without treating them as independent pairs."""
    successful = [row for row in rows if row["status"] == "completed"]
    errors = len(rows) - len(successful)
    pairs, task_scores = defaultdict(list), defaultdict(list)
    for row in successful:
        pairs[row["record_id"]].append(row)
        task_scores[row["benchmark"], row["task_id"]].append(row["reward"])
    complete_pairs = [pair for pair in pairs.values() if {r["swapped"] for r in pair} == {False, True}]

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    return {
        "decisions": len(rows),
        "successful_decisions": len(successful),
        "errors": errors,
        "pairs": len(pairs),
        "complete_pairs": len(complete_pairs),
        "task_groups": len(task_scores),
        "accuracy": mean([r["reward"] for r in successful]) if not errors else None,
        "successful_only_accuracy": mean([r["reward"] for r in successful]),
        "task_macro_accuracy": mean([mean(values) for values in task_scores.values()]) if not errors else None,
        "flip_rate": mean([len({r["choice"] for r in pair}) > 1 for pair in complete_pairs]),
        "raw_B_fraction": mean([r["raw_choice"] == "B" for r in successful]),
        "both_orders_correct": sum(all(r["reward"] for r in pair) for pair in complete_pairs),
        "both_orders_wrong": sum(not any(r["reward"] for r in pair) for pair in complete_pairs),
    }


def paired_difference(left, right):
    """Task-cluster bootstrap of paired correctness differences (right minus left)."""

    def key(row):
        return row["record_id"], row["swapped"]

    lmap, rmap = {key(r): r for r in left}, {key(r): r for r in right}
    if lmap.keys() != rmap.keys() or any(r["status"] != "completed" for r in [*left, *right]):
        return {"status": "incomplete", "accuracy_delta": None, "task_macro_delta": None}
    groups = defaultdict(list)
    for identity, row in lmap.items():
        groups[row["benchmark"], row["task_id"]].append(rmap[identity]["reward"] - row["reward"])
    if not groups:
        return {"status": "empty", "accuracy_delta": None, "task_macro_delta": None}
    cluster_values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(42)
    samples = cluster_values[rng.integers(0, len(cluster_values), size=(4000, len(cluster_values)))].mean(axis=1)
    return {
        "status": "completed",
        "accuracy_delta": float(np.mean([v for values in groups.values() for v in values])),
        "task_macro_delta": float(cluster_values.mean()),
        "task_groups": len(groups),
        "task_macro_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "bootstrap_unit": "benchmark/task group; both orientations stay together",
        "bootstrap_samples": 4000,
    }


def generation_scores(rows, proposals):
    """Include rejected proposals as parent retention, and label valid-only results separately."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"], row["arm"], row["subset"]].append(row)
    details = []
    for proposal in proposals:
        for subset in ("current", "same_benchmark_other_task", "other_benchmark"):
            parent = groups[proposal["case_id"], "parent", subset]
            arm = f"{proposal['condition']}_{proposal['replicate']}"
            child = groups[proposal["case_id"], arm, subset] if proposal["status"] == "valid" else parent
            diff = paired_difference(parent, child)
            details.append(
                {
                    "case_id": proposal["case_id"],
                    "condition": proposal["condition"],
                    "replicate": proposal["replicate"],
                    "proposal_status": proposal["status"],
                    "subset": subset,
                    "parent_retained": proposal["status"] != "valid",
                    **diff,
                }
            )
    summary = {}
    for condition in ("strict", "relaxed"):
        own = [p for p in proposals if p["condition"] == condition]
        summary[condition] = {
            "proposals": len(own),
            "valid": sum(p["status"] == "valid" for p in own),
            "changed": sum(p.get("changed", False) for p in own),
            "rejection_reasons": dict(Counter(p["error_message"] for p in own if p["status"] != "valid")),
            "by_subset": {},
        }
        for subset in ("current", "same_benchmark_other_task", "other_benchmark"):
            own_details = [r for r in details if r["condition"] == condition and r["subset"] == subset]
            valid_only = [r for r in own_details if r["proposal_status"] == "valid"]
            complete = all(r["status"] == "completed" for r in own_details)
            summary[condition]["by_subset"][subset] = {
                "intention_to_treat_delta": float(np.mean([r["accuracy_delta"] for r in own_details]))
                if complete and own_details
                else None,
                "valid_only_delta": float(np.mean([r["accuracy_delta"] for r in valid_only]))
                if valid_only and all(r["status"] == "completed" for r in valid_only)
                else None,
                "proposal_count": len(own_details),
                "valid_only_count": len(valid_only),
                "independent_source_tasks": len({r["case_id"] for r in own_details}),
            }
    return {"summary": summary, "details": details, "rejected_proposal_policy": "retain parent for this diagnostic"}


def summarize(root, rows, generated):
    rubric = [row for row in rows if row["experiment"] == "rubric"]
    arms = sorted({row["arm"] for row in rubric})
    metrics = {
        arm: {
            bench: decision_metrics([r for r in rubric if r["arm"] == arm and r["benchmark"] == bench])
            for bench in BENCHMARKS
        }
        for arm in arms
    }
    contrasts = {}
    for left, right in (
        ("strict_plain", "expanded_plain"),
        ("strict_scoped", "expanded_scoped"),
        ("strict_plain", "strict_scoped"),
        ("expanded_plain", "expanded_scoped"),
        ("no_rubric", "expanded_scoped"),
    ):
        contrasts[f"{right}_minus_{left}"] = {
            bench: paired_difference(
                [r for r in rubric if r["arm"] == left and r["benchmark"] == bench],
                [r for r in rubric if r["arm"] == right and r["benchmark"] == bench],
            )
            for bench in BENCHMARKS
        }
    recovery = [row for row in rows if row["experiment"] == "recovery"]
    recovery_details = []
    for case_id in sorted({r["case_id"] for r in recovery}):
        source_bench = next(r["source_benchmark"] for r in recovery if r["case_id"] == case_id)
        for subset in ("current", "same_benchmark_other_task", "other_benchmark"):
            own = [r for r in recovery if r["case_id"] == case_id and r["subset"] == subset]
            recovery_details.append(
                {
                    "case_id": case_id,
                    "source_benchmark": source_bench,
                    "subset": subset,
                    **paired_difference(
                        [r for r in own if r["arm"] == "parent"], [r for r in own if r["arm"] == "recovered"]
                    ),
                }
            )
    recovery_summary = {}
    for subset in ("current", "same_benchmark_other_task", "other_benchmark"):
        own = [r for r in recovery_details if r["subset"] == subset]
        complete = all(r["status"] == "completed" for r in own)
        values = [r["accuracy_delta"] for r in own] if complete else []
        recovery_summary[subset] = {
            "source_cases": len(own),
            "mean_case_delta": float(np.mean(values)) if values else None,
            "improved": sum(v > 0 for v in values),
            "worsened": sum(v < 0 for v in values),
            "unchanged": sum(v == 0 for v in values),
            "all_complete": complete,
        }
    generation = generation_scores([r for r in rows if r["experiment"] == "generation"], generated)
    summary = {
        "rubric_metrics": metrics,
        "rubric_paired_contrasts": contrasts,
        "recovered_rules": {"summary": recovery_summary, "details": recovery_details},
        "generation": generation,
        "errors": sum(r["status"] != "completed" for r in rows),
        "decisions": len(rows),
        "limitations": [
            "Development validation reused: these are diagnostic estimates, not a new independent final test.",
            "Only 5/5/4 validation task groups; confidence intervals cluster tasks, not orientations.",
            "Rejected-proposal pilot deliberately samples affected tasks; it does not estimate all-update effects.",
            "Transfer cases are other training tasks, which old playbooks may already have seen.",
            "Both rules and prompts remain affected by earlier abstract discovery; "
            "restoring a pool cannot recover ungenerated knowledge.",
            "Scope prompt changes contract interpretation, efficiency priority and uncertainty together.",
            "Case-specific transfer sets can overlap across source cases; pilot case deltas are descriptive.",
            "No model weights or full learning trajectories are retrained; all comparisons use frozen source data.",
        ],
    }
    write_json(root / "summary.json", summary)

    def percent(value):
        return "NA" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# Judge specificity diagnostic",
        "",
        "## Rubric comparison",
        "",
        "Both orientations averaged. Flip means canonical preference changes when A/B is swapped.",
        "",
        "| Arm | BFCL accuracy / flip | AppWorld accuracy / flip | Terminal accuracy / flip |",
        "|---|---:|---:|---:|",
    ]
    for arm in arms:
        cells = [
            f"{percent(metrics[arm][b]['accuracy'])} / {percent(metrics[arm][b]['flip_rate'])}" for b in BENCHMARKS
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Paired rubric contrasts",
            "",
            "Delta in percentage points. CI covers task-macro delta, not pair-weighted delta.",
            "",
            "| Contrast | Benchmark | Pair-weighted delta | Task-macro delta [95% CI] |",
            "|---|---|---:|---:|",
        ]
    )
    for contrast, results in contrasts.items():
        for bench, result in results.items():
            ci = result.get("task_macro_ci95")
            interval = f"[{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]" if ci else "NA"
            lines.append(
                f"| {contrast} | {bench} | {percent(result['accuracy_delta'])} | "
                f"{percent(result['task_macro_delta'])} {interval} |"
            )
    lines.extend(
        [
            "",
            "## Previously rejected mutations",
            "",
            "```json",
            json.dumps(recovery_summary, indent=2),
            "```",
            "",
            "## Strict / relaxed generation",
            "",
            "Invalid proposals retain the parent in intention-to-treat comparisons.",
            "",
            "```json",
            json.dumps(generation["summary"], indent=2),
            "```",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in summary["limitations"])
    lines.extend(["", f"Requested decisions: {len(rows)}; errors: {summary['errors']}.", ""])
    (root / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric-source", type=Path, required=True)
    parser.add_argument("--reward-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--embedding-python", default="/home/thskadud/.vllm/bin/python")
    parser.add_argument("--embedding-model", default="/home/thskadud/.cache/huggingface/hub/models--BAAI--bge-m3")
    parser.add_argument("--cases-per-benchmark", type=int, default=4)
    parser.add_argument("--generation-cases-per-benchmark", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.cases_per_benchmark, args.generation_cases_per_benchmark) < 1:
        parser.error("Counts must be positive")
    if args.generation_cases_per_benchmark > args.cases_per_benchmark:
        parser.error("Generation count exceeds selected case count")
    for name in ("rubric_source", "reward_source", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        analysis = Analysis(args)
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(20):
                analysis.status()

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            analysis.run()
        except BaseException as exc:
            analysis.status(status="error", error_type=type(exc).__name__, error_message=str(exc))
            traceback.print_exc()
            raise
        finally:
            stop.set()
            thread.join(timeout=1)


if __name__ == "__main__":
    main()
