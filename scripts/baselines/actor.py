# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Execute one frozen actor in the core environment, without optimizer dependencies."""

from __future__ import annotations

import argparse
import copy
import math
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.baselines.runtime import digest, read, write  # noqa: E402


def official_protocol_failure(session, session_dir, benchmark):
    """Recover a measured outcome for a model response with no visible action."""
    metadata = (session.details or {}).get("session_metadata") or {}
    if (benchmark not in {"bfcl", "appworld"} or session.status.value != "error" or session.score != 0.0
            or session.success is not False or metadata.get("error_source") != "agent"
            or metadata.get("error") != "LiteLLM completion missing assistant content."):
        return None
    path = Path(session_dir) / ("benchmark/bfcl_score.json" if benchmark == "bfcl" else "benchmark/results.json")
    try:
        outcome = read(path)
    except (OSError, ValueError):
        return None
    if (not isinstance(outcome, dict) or type(outcome.get("score")) not in (int, float)
            or not math.isfinite(outcome["score"]) or not 0 <= outcome["score"] <= 1
            or type(outcome.get("success")) is not bool):
        return None
    if benchmark == "bfcl" and (outcome["score"] != 0.0 or outcome["success"] is not False):
        return None
    if benchmark == "appworld" and (
        outcome.get("session_id") != getattr(session, "session_id", None)
        or outcome.get("task_id") != getattr(session, "task_id", None)
        or not outcome.get("session_id") or not outcome.get("task_id")
    ):
        return None
    return {"error": metadata["error"], "official_outcome_source": str(path),
            "score": float(outcome["score"]), "success": outcome["success"]}


def restore_protocol_outcome(session, failure):
    """Keep the raw error artifacts and use the separately recorded official score."""
    result = copy.copy(session)
    result.score = failure["score"]
    result.success = failure["success"]
    result.details = {**session.details, "score": result.score, "success": result.success,
                      "session_metadata": {**session.details.get("session_metadata", {}),
                                           "agent_protocol_error": failure["error"],
                                           "official_outcome_source": failure["official_outcome_source"]}}
    return result


def run(request):
    from exgentic.actor_context import shared_actor_prompt, validate_actor_context
    from exgentic.agents.premise.pipeline import trajectory_from_session
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    benchmark = request["benchmark"]
    actor_context = request.get("actor_context", "legacy")
    validate_actor_context(actor_context)
    policy_kwargs = {"playbook": request["system_prompt"], "system_prompt_override": request["system_prompt"]}
    if actor_context != "legacy":
        learned = request["learned_policy"]
        if request["system_prompt"] != shared_actor_prompt(benchmark, learned):
            raise ValueError("Stored actor prompt differs from the shared context renderer")
        policy_kwargs = {"playbook": learned, "actor_context": actor_context, "actor_benchmark": benchmark}
    steps = 30 if benchmark == "bfcl" else 40
    actor_temperature = request.get("actor_temperature")
    if actor_temperature is None:
        actor_temperature = request.get("temperature", 0.7)
    kwargs = dict(request["benchmark_kwargs"])
    kwargs["use_cache"] = False
    run_id = f"{request['method']}_{benchmark}_{uuid.uuid4().hex}"
    if benchmark == "appworld":
        kwargs["env_kwargs"] = {**kwargs.get("env_kwargs", {}), "experiment_name": run_id}
    if benchmark == "terminalbench2":
        kwargs.update(actor="terminus2", max_interactions=40)
    config = RunConfig(
        benchmark=benchmark, agent="premise", task_ids=[request["item"]["task_id"]],
        output_dir=request["session_output"], run_id=run_id, model=request["model"],
        benchmark_kwargs=kwargs, max_steps=steps, max_actions=100, overwrite_sessions=True,
        agent_kwargs={
            "model": request["model"], "temperature_override": actor_temperature, "enable_thinking": False,
            **policy_kwargs,
            "runner": "direct", "enable_tool_shortlisting": benchmark == "appworld" and actor_context == "legacy",
            "max_selected_tools": 30,
            "model_settings": {"temperature": actor_temperature, "max_tokens": 2048, "timeout": 1800,
                               "num_retries": 1, **({"reasoning_effort": request["reasoning_effort"]}
                                                    if request.get("reasoning_effort") else {})},
        },
    )
    result = execute(config=config)
    if len(result.session_results) != 1:
        raise RuntimeError("Actor did not return exactly one official outcome")
    session = result.session_results[0]
    protocol_failure = None
    if session.status.value == "error" and benchmark in {"bfcl", "appworld"}:
        session_dir = Path(request["session_output"]) / run_id / "sessions" / session.session_id
        protocol_failure = official_protocol_failure(session, session_dir, benchmark)
    if (session.status.value in ("error", "cancelled") and not protocol_failure) or session.score is None:
        raise RuntimeError(f"Actor infrastructure error: {session.status}: {session.details}")
    if protocol_failure:
        session = restore_protocol_outcome(session, protocol_failure)
    metadata = (session.details or {}).get("session_metadata") or {}
    try:
        trace = trajectory_from_session(Path(request["session_output"]), run_id, session, benchmark)
    except (ValueError, RuntimeError, OSError):
        if not metadata.get("experiment_budget_exceeded"):
            raise
        # A zero-turn native output failure may have no completed replay episode.
        # It is still an official score-zero outcome, never an infrastructure retry.
        session_dir = Path(request["session_output"]) / run_id / "sessions" / session.session_id
        manifest = read(session_dir / "session.json")
        trace = {"task": manifest["task"], "actions": [], "details": session.details,
                 "public_tool_contract": {"definitions": manifest.get("actions", [])},
                 "session_dir": str(session_dir), "budget_failure": metadata["experiment_budget_failure"]}
    if session.success is None:
        raise RuntimeError("Missing official binary success flag")
    if protocol_failure:
        trace["agent_protocol_error"] = protocol_failure["error"]
        trace["official_outcome_source"] = protocol_failure["official_outcome_source"]
    hard = int(session.success)
    if benchmark == "terminalbench2":
        # The adapter's success flag records successful verifier execution.
        # Terminal-Bench's task outcome is the official Harbor reward itself.
        hard = int(float(session.score) == 1.0)
    return {"id": request["item"]["task_id"], "hard": hard, "soft": float(session.score),
            "trace": trace, "prompt_sha256": digest(request["system_prompt"]), "run_id": run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = read(args.request)
        marker = Path(request.get("output", ".")) / "checkpoint_queue.json"
        if request.get("label") == "final_test" and marker.exists():
            from scripts.utils.checkpoint_eval_proxy import baseline_final

            result = baseline_final(request)
        else:
            result = run(request)
        write(args.response, result)
    except Exception as exc:
        from exgentic.utils.context_errors import is_context_length_error

        write(args.response, {"error": f"{type(exc).__name__}: {exc}",
                              "retryable": not is_context_length_error(exc) and not isinstance(exc, ValueError)})
        raise


if __name__ == "__main__":
    main()
