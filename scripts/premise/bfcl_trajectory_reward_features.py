# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Extract observable BFCL trajectory features without answers, private state, or model calls."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

FEATURE_VERSION = "bfcl-visible-trajectory-v1"


def _moved(path: Path) -> Path:
    parts = list(path.parts)
    if "outputs" in parts:
        parts[parts.index("outputs")] = "outputs_arr_oct"
        relocated = Path(*parts)
        if relocated.exists():
            return relocated
    return path


@lru_cache(maxsize=32)
def _contributors(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in (root / "contributors").glob("*") if path.is_dir()))


def resolve_trajectory(result_path: Path, result: dict | None = None) -> Path | None:
    """Resolve direct and contributed sessions using the exact recorded session identity."""
    result_path = Path(result_path)
    result = result if result is not None else json.loads(result_path.read_text())
    if not result.get("session_id"):
        return None
    root = next((parent for parent in result_path.parents if parent.name == "results"), None)
    if root is None:
        raise ValueError("Expected an archived results/checkpoint/context/partition result path")
    root = root.parent
    run = result.get("run_id") or (
        f"alignment_{result['checkpoint_key']}_{result['context']}_{result['partition']}_{result['index']:03d}"
    )
    roots = []
    if result.get("contributor_output"):
        roots.append(_moved(Path(result["contributor_output"])))
    roots.extend((root, *_contributors(root)))
    suffix = (Path("sessions") / result["checkpoint_key"] / result["context"] / result["partition"]
              / run / "sessions" / result["session_id"] / "trajectory.jsonl")
    for folder in dict.fromkeys(roots):
        candidate = folder / suffix
        if candidate.is_file():
            return candidate
    return None


def _actions(action: dict) -> list[dict]:
    if "actions" in action:
        return [leaf for child in action["actions"] for leaf in _actions(child)]
    return [action] if "name" in action else []


def _observations(observation: dict) -> list[dict]:
    if "observations" in observation:
        return [leaf for child in observation["observations"] for leaf in _observations(child)]
    return [observation]


def _canonical(value) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_error(value) -> bool:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            return _tool_error(decoded)
        return bool(re.match(r"^(?:Error during execution:|Error:|Traceback \(most recent call last\):|"
                             r"(?:[A-Za-z_][A-Za-z0-9_]*Error|Exception):)", value.strip()))
    if isinstance(value, dict):
        # Domain outcomes such as success:false, found:false, empty lists, and negative returns are not errors.
        return any(value.get(key) not in (None, False, "", [], {}) for key in ("error", "exception"))
    if isinstance(value, list):
        return any(_tool_error(item) for item in value)
    return False


def features_from_events(
    events: list[dict], first_request: str | None, result: dict | None = None, max_text_chars: int = 1200
) -> dict:
    """Summarize only logged actions, observations, and requests actually shown to the actor."""
    result = result or {}
    if max_text_chars < 1:
        raise ValueError("A positive compact evidence character limit is required")
    features = dict.fromkeys((
        "environment_steps", "tool_invocations", "observed_tool_calls", "invalid_actions", "invalid_names",
        "invalid_arguments", "tool_error_observations", "tool_error_calls", "finish_actions",
        "acknowledged_finishes", "visible_turn_advances", "repeated_action_observation_within_turn",
        "immediate_exact_repeat_calls", "avoidable_error_repeat_calls", "unmatched_observed_calls",
        "action_events_without_observation",
    ), 0)
    compact, requests, pending, completed = [], [], {}, set()
    seen, last_signature = set(), None
    turn, call_number, truncated_fields = 0, 0, 0

    def clipped(value):
        nonlocal truncated_fields
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        if len(text) <= max_text_chars:
            return value
        truncated_fields += 1
        return {"text_prefix": text[:max_text_chars], "truncated": True, "original_chars": len(text)}

    if first_request:
        requests.append(first_request)
        compact.append({"type": "request", "step": 0, "turn": turn, "text": clipped(first_request)})
    finish_step = None
    action_steps, observed_steps = set(), set()
    for event in events:
        step = event.get("step", 0)
        if event.get("event") == "action":
            features["environment_steps"] += 1
            action_steps.add(step)
            leaves = _actions(event.get("action", {}))
            if any(action["name"] == "finish" for action in leaves):
                finish_step = step
            for action in leaves:
                call_number += 1
                validation = action.get("validation") or {}
                name = action["name"]
                features["finish_actions"] += name == "finish"
                features["tool_invocations"] += name != "finish"
                features["invalid_names"] += validation.get("name_valid") is False
                features["invalid_arguments"] += validation.get("args_valid") is False
                features["invalid_actions"] += any(validation.get(key) is False
                                                    for key in ("valid", "name_valid", "args_valid"))
                identifier = action.get("id") or f"synthetic:{step}:{call_number}"
                pending[identifier] = (action, call_number)
                compact.append({"type": "action", "step": step, "turn": turn, "name": name,
                                "arguments": clipped(action.get("arguments", {})), "call_number": call_number,
                                "validation": {key: validation[key] for key in
                                               ("valid", "name_valid", "args_valid", "error") if key in validation}})
        elif event.get("event") == "observation":
            if not event.get("initial"):
                observed_steps.add(step)
            for observation in _observations(event.get("observation", {})):
                value = observation.get("result")
                invoking = [leaf for action in observation.get("invoking_actions", []) for leaf in _actions(action)]
                if not invoking:
                    if value is not None and str(value).strip() and step == finish_step:
                        turn += 1
                        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        requests.append(text)
                        features["visible_turn_advances"] += 1
                        compact.append({"type": "request", "step": step, "turn": turn, "text": clipped(text)})
                        seen.clear()
                        last_signature = None
                        finish_step = None
                    elif value is not None:
                        compact.append({"type": "observation", "step": step, "turn": turn,
                                        "names": [], "calls": [], "result": clipped(value)})
                    continue
                calls = []
                tool_actions = []
                for action in invoking:
                    identifier = action.get("id")
                    match = pending.get(identifier)
                    if match is None and identifier is None:
                        possible = [item for item in pending.values() if item[0]["name"] == action["name"]]
                        match = possible[0] if len(possible) == 1 else None
                    if match:
                        original, number = match
                        calls.append(number)
                    else:
                        original, number = action, None
                    if action["name"] == "finish":
                        features["acknowledged_finishes"] += str(value).strip() == "Turn finished."
                        continue
                    if number is not None and number in completed:
                        continue
                    if number is not None:
                        completed.add(number)
                    features["observed_tool_calls"] += 1
                    features["unmatched_observed_calls"] += match is None
                    tool_actions.append(original)
                    # Require original arguments; observation copies can intentionally omit them.
                    if match:
                        signature = _canonical([original["name"], original.get("arguments", {}), value])
                        features["repeated_action_observation_within_turn"] += signature in seen
                        features["immediate_exact_repeat_calls"] += signature == last_signature
                        features["avoidable_error_repeat_calls"] += signature == last_signature and _tool_error(value)
                        seen.add(signature)
                        last_signature = signature
                if tool_actions and _tool_error(value):
                    features["tool_error_observations"] += 1
                    features["tool_error_calls"] += len(tool_actions)
                compact.append({"type": "observation", "step": step, "turn": turn,
                                "names": [action["name"] for action in invoking], "calls": calls,
                                "result": clipped(value)})
    features["action_events_without_observation"] = len(action_steps - observed_steps)
    features.update({"reported_steps": result.get("steps"), "max_steps": result.get("max_steps"),
                     "visible_request_count": len(requests), "result_status": result.get("status"),
                     "session_status": result.get("session_status"),
                     "at_step_limit": bool(result.get("max_steps") and result.get("steps", 0) >= result["max_steps"]),
                     "infrastructure_error": bool(result.get("status") == "error" and re.search(
                         r"BadGateway|Connection refused|ConnectionError|ServiceUnavailable|Timeout|429|502|503",
                         str(result.get("error", "")), re.IGNORECASE)),
                     "limit_termination": bool(re.search(
                         r"max[_ ]steps|step limit|turn limit|token limit|output[_ ]length|truncated actor",
                         str(result.get("error", "")), re.IGNORECASE))})
    evidence = {"events": compact, "visible_requests": [clipped(request) for request in requests],
                "lost_evidence": truncated_fields > 0, "truncated_fields": truncated_fields}
    return {"features": features, "compact_trajectory": evidence, "visible_requests": requests}


def extract_features(result_path: Path, *, first_request: str | None = None, max_text_chars: int = 1200) -> dict:
    """Extract one training-task result, preserving provenance and explicit missing-trace coverage."""
    result_path = Path(result_path)
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    if result.get("partition") != "train":
        raise ValueError("Selection features may only read sampled training-task trajectories")
    trajectory = resolve_trajectory(result_path, result)
    output = {"version": FEATURE_VERSION, "available": trajectory is not None,
              "identity": {key: result.get(key) for key in ("checkpoint_key", "context", "task_id", "session_id")},
              "result_path": str(result_path.resolve()), "result_sha256": hashlib.sha256(result_bytes).hexdigest()}
    if trajectory is None:
        return {**output, "missing_reason": "trajectory_not_found", "features": None, "compact_trajectory": None,
                "visible_requests": []}
    trajectory_bytes = trajectory.read_bytes()
    events = [json.loads(line) for line in trajectory_bytes.decode().splitlines() if line.strip()]
    run = result.get("run_id") or (
        f"alignment_{result['checkpoint_key']}_{result['context']}_{result['partition']}_{result['index']:03d}"
    )
    if any(event.get("session_id") not in (None, result["session_id"])
           or event.get("task_id") not in (None, result["task_id"])
           or event.get("run_id") not in (None, run) for event in events):
        raise ValueError("Trajectory task/session identity differs from the selected result")
    session = trajectory.with_name("session.json")
    session_bytes = session.read_bytes() if session.is_file() else None
    if session_bytes is not None:
        row = json.loads(session_bytes)
        if row.get("task_id") not in (None, result["task_id"]) or row.get("run_id") not in (None, run):
            raise ValueError("Session task identity differs from the selected result")
        first_request = row.get("task") or first_request
    values = features_from_events(events, first_request, result, max_text_chars)
    return {**output, **values, "trajectory_path": str(trajectory.resolve()),
            "trajectory_sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
            "session_sha256": hashlib.sha256(session_bytes).hexdigest() if session_bytes is not None else None,
            "definitions": "Tool invocations are flattened attempted non-finish actions; observed_tool_calls counts "
            "matched or unmatched logged tool responses. Exact repeats are proxies within a visible user request, "
            "not proven unnecessary actions. Finish acknowledgement is not certified subgoal success. No wall time "
            "or official success score enters features/compact evidence."}
