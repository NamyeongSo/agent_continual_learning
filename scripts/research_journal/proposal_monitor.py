# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Public, read-only projections of recorded proposal retrieval evidence."""

from __future__ import annotations


def retrieval_summary(artifact):
    candidates = []
    for candidate in artifact.get("candidates", []):
        audit = candidate.get("proposal_retrieval")
        if not isinstance(audit, dict):
            continue
        reflection = candidate.get("reflection") or {}
        candidates.append({
            "id": candidate.get("id"), "parent_rank": candidate.get("parent_rank"),
            "reflection_id": reflection.get("id"), "valid": candidate.get("valid"),
            **{key: audit.get(key) for key in (
                "mode", "requested", "selected", "eligible_tasks", "query_embedding_truncated")},
            "examples": [{key: example.get(key) for key in (
                "id", "iteration", "task_id", "benchmark", "retrieval_role", "action_f1",
                "reflection_cosine", "embedding_truncated")} for example in audit.get("examples", [])],
            **({"grounded_proposal": candidate["grounded_proposal"]} if candidate.get("grounded_proposal") else {}),
        })
    return {"iteration": artifact.get("iteration"), "task_id": artifact.get("task_id"),
            "status": artifact.get("status"), "reason": artifact.get("reason"),
            "error_type": artifact.get("error_type"),
            "error_message": str(artifact.get("error_message") or "")[:1500], "candidates": candidates}


def evidence(artifact, reflection_id):
    reflection = next((row for row in artifact.get("reflections", []) if row.get("id") == reflection_id), None)
    if reflection is None:
        return {"available": False}
    rank = reflection.get("parent_rank")
    parents = artifact.get("parents") or []
    if not isinstance(rank, int) or not 1 <= rank <= len(parents) or not parents[rank - 1]:
        return {"available": False}
    trace = parents[rank - 1]
    return {"available": True, "iteration": artifact.get("iteration"),
            **{key: trace.get(key) for key in ("task_id", "benchmark", "task", "success", "actions",
                                              "public_tool_contract")},
            "reflection": {key: reflection.get(key) for key in (
                "id", "parent_rank", "span", "analysis", "expected_behavior", "entry_assessments", "evidence_card")},
            **({"grounding": {key: reflection["grounding"].get(key) for key in ("version", "verified", "reviews")}}
               if reflection.get("grounding") else {})}
