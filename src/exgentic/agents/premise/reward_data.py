# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Frozen pairwise reward data with reconstructed continuations and task holdouts."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

BENCHMARKS = ("bfcl", "appworld", "terminalbench2")


class FeedbackTruncator:
    def __init__(self, tokenizer, max_tokens: int = 500):
        self.tokenizer, self.max_tokens = tokenizer, max_tokens
        self.marker = f"\n[환경 feedback 생략됨: 턴당 {max_tokens}토큰 제한]"
        self.turns = self.truncated_turns = 0
        if max_tokens <= len(tokenizer.encode(self.marker, add_special_tokens=False).ids):
            raise ValueError("Feedback token limit must leave room for the omission marker")

    def observation(self, value) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
        self.turns += 1
        if len(tokens) <= self.max_tokens:
            return text
        self.truncated_turns += 1
        budget = self.max_tokens - len(self.tokenizer.encode(self.marker, add_special_tokens=False).ids)
        while budget >= 0:
            result = self.tokenizer.decode(tokens[:budget], skip_special_tokens=False) + self.marker
            if len(self.tokenizer.encode(result, add_special_tokens=False).ids) <= self.max_tokens:
                return result
            budget -= 1
        raise ValueError("Could not fit feedback omission marker")

    def steps(self, steps: list[dict]) -> list[dict]:
        return [
            {
                **step,
                **(
                    {"observation": self.observation(step["observation"])}
                    if step.get("observation") is not None
                    else {}
                ),
            }
            for step in steps
        ]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _trace_choice_message(choice) -> dict:
    if isinstance(choice, str):
        # Older LiteLLM logs store SDK objects as repr strings. Convert only
        # known data constructors into literal dictionaries; never eval them.
        class Literals(ast.NodeTransformer):
            def visit_Call(self, node):
                constructors = {"Choices", "Message", "ChatCompletionMessageToolCall", "Function", "FunctionCall"}
                if (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in constructors
                    or node.args
                    or any(keyword.arg is None for keyword in node.keywords)
                ):
                    raise ValueError("Unsupported legacy trace constructor")
                return ast.Dict(
                    keys=[ast.Constant(keyword.arg) for keyword in node.keywords],
                    values=[self.visit(keyword.value) for keyword in node.keywords],
                )

        try:
            choice = ast.literal_eval(Literals().visit(ast.parse(choice, mode="eval")))
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return {}
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return {}
    return choice["message"]


def _public_call_arguments(root: Path, benchmark: str) -> dict[tuple[str, str], dict]:
    """Recover historical call parameters, without reading scores or verifier output."""
    from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows

    calls = {}

    def remember(call_id, name, arguments):
        if not call_id or not name or not isinstance(arguments, dict):
            return
        key = call_id, name
        if key in calls and calls[key] != arguments:
            raise ValueError(f"Conflicting public parameters for call {call_id}")
        calls[key] = arguments

    trace_path = root / "benchmark/bfcl_trace.json"
    if benchmark == "bfcl" and trace_path.exists():
        trace = json.loads(trace_path.read_text())
        for turn in trace.get("turn_step_actions", []):
            for step in turn:
                for action in step:
                    remember(action.get("id"), action.get("name"), action.get("arguments"))
        # The executed arguments include validated/defaulted parameters and
        # take precedence over the model's raw JSON for BFCL calls.
        return calls

    log_path = root / "agent/litellm/trace.jsonl"
    if log_path.exists():
        for row in iter_expanded_trace_rows(log_path):
            messages = list(row.get("request", {}).get("messages", []) or [])
            response = row.get("response", {})
            if isinstance(response, dict):
                for choice in response.get("choices", []) or []:
                    messages.append(_trace_choice_message(choice))
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                for call in message.get("tool_calls", []) or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function", {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            continue
                    remember(call.get("id"), function.get("name"), arguments)
    return calls


def _restore_composite_arguments(action: dict, calls: dict[tuple[str, str], dict]) -> dict:
    if not action.get("actions"):
        return action
    restored = []
    for child in action["actions"]:
        arguments = calls.get((child.get("id"), child.get("name")))
        if child.get("arguments") == {} and arguments:
            child = {**child, "arguments": arguments}
        restored.append(child)
    return {**action, "actions": restored}


def _recorded_replay_prefix(root: Path) -> list[dict] | None:
    """Read the actual prefix outputs shown to the continuation agent."""
    from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows

    path = root / "agent/litellm/trace.jsonl"
    if not path.exists():
        return None
    header = "The environment has been restored by executing the following recorded actions"
    for row in iter_expanded_trace_rows(path):
        for message in row.get("request", {}).get("messages", []) or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.startswith(header):
                prefix = json.loads(content.partition("\n")[2])
                if not isinstance(prefix, list):
                    raise ValueError("Recorded continuation prefix is not a list")
                return prefix
    return None


def _prefix_action(action: dict) -> dict:
    # PrefixReplaySession rebuilds validation; the saved validation report
    # belongs to the parent and can incorrectly claim that a lost call is valid.
    result = {key: value for key, value in action.items() if key != "validation"}
    if result.get("actions"):
        result["actions"] = [_prefix_action(child) for child in result["actions"]]
    return result


@lru_cache(maxsize=256)
def public_trace(session: str, benchmark: str) -> dict:
    """Read public observations only; omit grader feedback, costs and model metadata."""
    root = Path(session)
    manifest = json.loads((root / "session.json").read_text())
    context = {k: v for k, v in manifest.get("context", {}).items() if k not in ("benchmark", "task_id")}
    if benchmark == "terminalbench2":
        # The adapter observation contains verifier rewards. Use native agent
        # steps, which already include the restored prefix, instead.
        native = json.loads((root / "benchmark/harbor/agent/trajectory.json").read_text())
        steps = [
            {k: step[k] for k in ("message", "tool_calls", "observation") if k in step}
            for step in native["steps"]
            if step["source"] == "agent"
        ]
        tools = []
    else:
        events = [json.loads(line) for line in (root / "trajectory.jsonl").read_text().splitlines() if line]
        observations = {
            event["step"]: event.get("observation")
            for event in events
            if event.get("event") == "observation" and not event.get("initial")
        }
        steps = [
            {"action": event["action"], "observation": observations.get(event["step"])}
            for event in events
            if event.get("event") == "action"
        ]
        tools = manifest.get("actions", [])
        if any(step["action"].get("actions") for step in steps):
            calls = _public_call_arguments(root, benchmark)
            required = {tool["name"]: tool.get("arguments_schema", {}).get("required", []) for tool in tools}
            for step in steps:
                step["action"] = _restore_composite_arguments(step["action"], calls)
                for child in step["action"].get("actions", []):
                    if (
                        child.get("arguments") == {}
                        and required.get(child.get("name"))
                        and child.get("validation", {}).get("valid") is True
                    ):
                        raise ValueError(f"Cannot recover required public parameters for call {child.get('id')}")
    return {"task": manifest["task"], "context": context, "steps": steps, "tools": tools}


def reconstruct_pair(row: dict) -> dict:
    source = public_trace(row["original_session_dir"], row["benchmark"])
    replay = public_trace(row["replay_session_dir"], row["benchmark"])
    if source["task"] != replay["task"]:
        raise ValueError("Source and replay refer to different tasks")
    replay_steps = replay["steps"]
    if row["benchmark"] != "terminalbench2":
        config_path = Path(row["replay_session_dir"]).parent.parent / "run/config.json"
        config = json.loads(config_path.read_text())
        prefix = (config.get("replay_prefix") or {}).get("actions", [])
        source_calls = {
            (action.get("id"), action.get("name")): action.get("arguments", {})
            for step in source["steps"]
            for action in step["action"].get("actions", [])
            if action.get("id")
        }
        if len(prefix) > len(source["steps"]) or any(
            _restore_composite_arguments(saved["action"], source_calls) != step["action"]
            for saved, step in zip(prefix, source["steps"], strict=False)
        ):
            raise ValueError("Recorded replay prefix differs from the original trajectory")
        if prefix:
            actual_prefix = _recorded_replay_prefix(Path(row["replay_session_dir"]))
            if actual_prefix is not None:
                if len(actual_prefix) != len(prefix) or any(
                    entry.get("index") != index or entry.get("action") != prefix[index]["action"]
                    for index, entry in enumerate(actual_prefix)
                ):
                    raise ValueError("Actual replay prefix differs from its configured actions")
                restored_prefix = [
                    {"action": _prefix_action(entry["action"]), "observation": entry["observation"]}
                    for entry in actual_prefix
                ]
            else:
                # Reuse parent feedback only when execution parameters agree.
                # Lost parameters can produce different replay outputs.
                if any(
                    entry["action"] != step["action"]
                    for entry, step in zip(prefix, source["steps"], strict=False)
                ):
                    raise ValueError("Actual replay feedback unavailable for a prefix with lost parameters")
                restored_prefix = source["steps"][: len(prefix)]
            replay_steps = restored_prefix + replay_steps
    observed = set()
    for step in source["steps"] + replay_steps:
        action = step.get("action") or {}
        observed.add(action.get("name"))
        observed.update(a.get("name") for a in action.get("actions", []))
    tools = [
        {k: tool[k] for k in ("name", "description", "arguments_schema", "arguments", "is_finish") if k in tool}
        for tool in source["tools"]
        if not tool.get("is_hidden") and (tool.get("name") in observed or tool.get("is_finish"))
    ]
    return {
        "task": source["task"],
        "context": source["context"],
        "public_tools": tools,
        "original": source["steps"],
        "replay": replay_steps,
    }


def holdout_tasks(rows: list[dict], fraction: float, seed: int) -> tuple[set[str], dict]:
    """Find the closest pair-count holdout while keeping whole task groups together."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    if len(groups) < 2:
        raise ValueError("At least two task groups per benchmark are needed for train/val")
    tasks = sorted(groups)
    random.Random(seed).shuffle(tasks)
    target = max(1, min(len(rows) - 1, round(len(rows) * fraction)))
    reachable = {0: ()}
    for task in tasks:
        size = len(groups[task])
        for total, subset in list(reachable.items()):
            if total + size not in reachable:
                reachable[total + size] = (*subset, task)
    best = min((count for count in reachable if 0 < count < len(rows)), key=lambda n: (abs(n - target), n))
    chosen = set(reachable[best])
    return chosen, {
        "target_val_pairs": target,
        "val_pairs": best,
        "train_pairs": len(rows) - best,
        "val_tasks": len(chosen),
        "train_tasks": len(groups) - len(chosen),
    }


def prepare_dataset(
    metadata: Path,
    output: Path,
    *,
    source_seed: int = 42,
    split_seed: int = 42,
    val_fraction: float = 0.2,
    tokenizer_path: Path,
) -> dict:
    from tokenizers import Tokenizer

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if (output / "manifest.json").exists():
        raise FileExistsError("Dataset is already frozen")
    raw_bytes = metadata.read_bytes()
    tokenizer_bytes = tokenizer_path.read_bytes()
    truncator = FeedbackTruncator(Tokenizer.from_file(str(tokenizer_path)), 500)
    rows = [json.loads(line) for line in raw_bytes.decode().splitlines() if line]
    rows = [row for row in rows if row["seed"] == source_seed]
    seen = set()
    selected = []
    for row in rows:
        if math.isclose(row["original_score"], row["replay_score"], abs_tol=1e-9, rel_tol=0):
            continue
        key = row["original_session_dir"], row["replay_session_dir"]
        if key not in seen:
            selected.append(row)
            seen.add(key)
    rng = random.Random(split_seed)
    records = []
    split_counts = {}
    for benchmark in BENCHMARKS:
        subset = [row for row in selected if row["benchmark"] == benchmark]
        val_tasks, counts = holdout_tasks(subset, val_fraction, split_seed)
        split_counts[benchmark] = counts
        for split in ("train", "val"):
            bucket = [row for row in subset if (row["task_id"] in val_tasks) == (split == "val")]
            rng.shuffle(bucket)
            for index, row in enumerate(bucket):
                pair = reconstruct_pair(row)
                pair["original"] = truncator.steps(pair["original"])
                pair["replay"] = truncator.steps(pair["replay"])
                pair_id = digest([row["original_session_dir"], row["replay_session_dir"]])[:24]
                winner = "A" if index % 2 == 0 else "B"
                original_wins = row["original_score"] > row["replay_score"]
                a_is_original = original_wins == (winner == "A")
                visible = {k: pair[k] for k in ("task", "context", "public_tools")}
                visible.update(
                    trajectory_a=pair["original"] if a_is_original else pair["replay"],
                    trajectory_b=pair["replay"] if a_is_original else pair["original"],
                )
                common = 0
                for a, b in zip(visible["trajectory_a"], visible["trajectory_b"], strict=False):
                    if a != b:
                        break
                    common += 1
                visible["common_prefix"] = visible["trajectory_a"][:common]
                visible["trajectory_a"] = visible["trajectory_a"][common:]
                visible["trajectory_b"] = visible["trajectory_b"][common:]
                write_json(output / "pairs" / f"{pair_id}.json", visible)
                records.append(
                    {
                        "id": pair_id,
                        "benchmark": benchmark,
                        "task_id": row["task_id"],
                        "split": split,
                        "label": winner,
                        "input_sha256": digest(visible),
                        "provenance": row,
                        "a_is_original": a_is_original,
                    }
                )
    train_groups = {(r["benchmark"], r["task_id"]) for r in records if r["split"] == "train"}
    val_groups = {(r["benchmark"], r["task_id"]) for r in records if r["split"] == "val"}
    if train_groups & val_groups:
        raise RuntimeError("Train/val task overlap")
    manifest = {
        "source_metadata": str(metadata.resolve()),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_seed": source_seed,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "split_unit": "benchmark/task_id",
        "counts": split_counts,
        "records": records,
        "label_balance": "A/B balanced separately within each benchmark and split",
        "environment_feedback": {
            "max_tokens_per_turn": 500,
            "marker_included_in_limit": True,
            "tokenizer_path": str(tokenizer_path.resolve()),
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "turns_processed": truncator.turns,
            "truncated_turns": truncator.truncated_turns,
        },
        "input_contract": "Public task/context/tools and full A/B trajectories; no scores or provenance",
        "public_tool_contract": {
            "schema_field": "arguments_schema",
            "selection": "Observed actions and completion action; hidden actions excluded",
        },
        "action_arguments": {
            "legacy_recovery": "Composite call arguments restored by exact call ID and tool name",
            "sources": ["Public assistant tool-call history", "BFCL executed turn_step_actions"],
            "missing_required_parameters": "Reject unrecoverable valid composite calls",
            "replay_prefix": "Use actual recorded prefix actions and feedback shown to the continuation agent",
        },
    }
    write_json(output / "manifest.json", manifest)
    public_trace.cache_clear()
    return manifest


def training_order(records: list[dict], mode: str, seed: int) -> list[str]:
    pools = {
        slug: sorted(r["id"] for r in records if r["split"] == "train" and r["benchmark"] == slug)
        for slug in BENCHMARKS
    }
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    if mode == "sequential":
        return [pair_id for slug in BENCHMARKS for pair_id in pools[slug]]
    if mode != "interleaved":
        raise ValueError("Unknown training setting")
    order = []
    while any(pools.values()):
        slug = rng.choice([slug for slug in BENCHMARKS if pools[slug]])
        order.append(pools[slug].pop())
    return order
