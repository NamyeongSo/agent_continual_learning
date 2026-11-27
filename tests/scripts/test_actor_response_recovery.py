# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from scripts.baselines.runtime import digest, recover_actor_response, write


def test_interrupted_official_budget_zero_is_recovered_but_wrong_prompt_and_partial_responses_are_rejected(tmp_path):
    request = {"model": "same-model", "item": {"task_id": "task"}, "system_prompt": "same-policy"}
    write(tmp_path / "request.json", request)
    write(tmp_path / "attempt_0.json", {"error": "Transport error", "retryable": True})
    result = {"id": "task", "run_id": "official-run", "prompt_sha256": digest("same-policy"),
              "hard": 0, "soft": 0.0, "trace": {"budget_failure": {"reason": "output_truncated"}}}
    write(tmp_path / "attempt_1.json", result)
    assert recover_actor_response(tmp_path, request) == result
    assert recover_actor_response(tmp_path, {**request, "model": "other-model"}) is None
    result["prompt_sha256"] = digest("different-policy")
    write(tmp_path / "attempt_1.json", result)
    assert recover_actor_response(tmp_path, request) is None
