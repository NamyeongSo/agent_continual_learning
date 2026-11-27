# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json

import pytest

from scripts.premise.bfcl_trajectory_reward_features import extract_features, features_from_events, resolve_trajectory


def _action(step, name="read", arguments=None, **validation):
    return {"event": "action", "step": step,
            "action": {"name": name, "arguments": arguments or {}, "id": f"call{step}",
                       "validation": {"valid": True, "name_valid": True, "args_valid": True, **validation}}}


def _observation(step, result, name="read"):
    return {"event": "observation", "step": step,
            "observation": {"result": result, "invoking_actions": [{"name": name, "id": f"call{step}"}]}}


def test_parallel_children_count_as_distinct_tool_invocations_in_one_environment_step():
    first, second = _action(1, "read", {"file": "a"}), _action(2, "write", {"file": "b"})
    events = [{"event": "action", "step": 1, "action": {"actions": [first["action"], second["action"]]}},
              {"event": "observation", "step": 1, "observation": {"observations": [
                  _observation(1, '{"contents": "hello"}')["observation"],
                  _observation(2, '{"written": true}', "write")["observation"],
              ]}}]
    output = features_from_events(events, "Read a and write b")
    assert output["features"]["environment_steps"] == 1
    assert output["features"]["tool_invocations"] == output["features"]["observed_tool_calls"] == 2
    assert [event["name"] for event in output["compact_trajectory"]["events"] if event["type"] == "action"] == [
        "read", "write",
    ]


@pytest.mark.parametrize("result", [
    '{"success": false}', '{"found": false, "items": []}', '{"error": null}', '{"error": 0}',
    '{"error_rate": 0.3}', "No matching record.", "File was not found in the listing.",
])
def test_normal_negative_outcomes_are_not_broadly_counted_as_tool_errors(result):
    output = features_from_events([_action(1), _observation(1, result)], "Read a")
    assert output["features"]["tool_error_observations"] == output["features"]["tool_error_calls"] == 0


@pytest.mark.parametrize("result", [
    '{"error": "No such file"}', {"exception": "Invalid path"}, "Error during execution: Invalid route.",
    "Error: Invalid route.", "ValueError: Invalid route.", "Traceback (most recent call last):\nFailure",
])
def test_explicit_structured_and_execution_errors_are_counted_from_observed_tool_outputs(result):
    output = features_from_events([_action(1), _observation(1, result)], "Read a")
    assert output["features"]["tool_error_observations"] == output["features"]["tool_error_calls"] == 1


def test_invalid_name_and_schema_are_read_from_action_validation_without_double_counting_observation_copies():
    events = [_action(1, "invented", valid=False, name_valid=False),
              _observation(1, '{"error": "Unknown action"}', "invented"),
              _action(2, args_valid=False, valid=False), _observation(2, '{"error": "Invalid arguments"}')]
    features = features_from_events(events, "Read a")["features"]
    assert features["invalid_actions"] == 2
    assert features["invalid_names"] == features["invalid_arguments"] == 1
    assert features["tool_invocations"] == 2


def test_repeated_reads_reset_after_a_new_visible_request_and_finish_is_not_success_certification():
    finish = _action(3, "finish", {"content": "Done"})
    boundary = {"event": "observation", "step": 3, "observation": {"observations": [
        _observation(3, "Turn finished.", "finish")["observation"],
        {"result": "Read the same file once more.", "invoking_actions": []},
    ]}}
    events = [_action(1), _observation(1, '{"value": 1}'), _action(2), _observation(2, '{"value": 1}'),
              finish, boundary, _action(4), _observation(4, '{"value": 1}')]
    output = features_from_events(events, "Read the file.")
    features = output["features"]
    assert features["repeated_action_observation_within_turn"] == features["immediate_exact_repeat_calls"] == 1
    assert features["finish_actions"] == features["acknowledged_finishes"] == 1
    assert features["visible_turn_advances"] == 1 and features["visible_request_count"] == 2
    assert output["visible_requests"] == ["Read the file.", "Read the same file once more."]
    assert [event["turn"] for event in output["compact_trajectory"]["events"] if event["type"] == "request"] == [0, 1]


def test_avoidable_error_repeats_exclude_the_initial_error_changed_arguments_and_normal_verification():
    error = '{"error": "No such file"}'
    events = [_action(1, arguments={"file": "a"}), _observation(1, error),
              _action(2, arguments={"file": "a"}), _observation(2, error),
              _action(3, arguments={"file": "b"}), _observation(3, error),
              _action(4, arguments={"file": "b"}), _observation(4, '{"contents": "ok"}'),
              _action(5, arguments={"file": "b"}), _observation(5, '{"contents": "ok"}')]
    features = features_from_events(events, "Read a")["features"]
    assert features["tool_error_observations"] == 3
    assert features["immediate_exact_repeat_calls"] == 2
    assert features["avoidable_error_repeat_calls"] == 1


def test_error_repeat_after_a_new_user_request_is_not_counted_as_avoidable():
    error = '{"error": "No such file"}'
    events = [_action(1), _observation(1, error), _action(2, "finish"),
              {"event": "observation", "step": 2, "observation": {"observations": [
                  _observation(2, "Turn finished.", "finish")["observation"],
                  {"result": "Try that same file again.", "invoking_actions": []}]}},
              _action(3), _observation(3, error)]
    assert features_from_events(events, "Read a")["features"]["avoidable_error_repeat_calls"] == 0


def test_truncation_is_explicit_and_does_not_change_full_trace_feature_counts():
    events = [_action(1, arguments={"text": "x" * 100}), _observation(1, {"output": "y" * 200})]
    full = features_from_events(events, "Read " + "z" * 100, max_text_chars=1000)
    clipped = features_from_events(events, "Read " + "z" * 100, max_text_chars=20)
    assert clipped["features"] == full["features"]
    assert clipped["compact_trajectory"]["lost_evidence"] is True
    assert clipped["compact_trajectory"]["truncated_fields"] >= 3
    for event in clipped["compact_trajectory"]["events"]:
        value = event.get("text", event.get("arguments", event.get("result")))
        assert value["truncated"] is True and value["original_chars"] > 20


def test_infrastructure_errors_and_step_limits_are_separate_from_observed_policy_errors_and_wall_time():
    result = {"status": "error", "session_status": "error", "steps": 20, "max_steps": 20,
              "error": "BadGateway: Connection refused", "wall_seconds": 100000, "score": 0}
    output = features_from_events([], "Read a", result)
    assert output["features"]["infrastructure_error"] is True
    assert output["features"]["at_step_limit"] is True
    assert output["features"]["tool_error_observations"] == 0
    assert "wall_seconds" not in json.dumps(output) and '"score"' not in json.dumps(output)
    assert "session_status" not in json.dumps(output["compact_trajectory"])


def _archive(tmp_path, contributor=False):
    root = tmp_path / "outputs_arr_oct" / "experiment"
    result_path = root / "results" / "cp10" / "candidate_01" / "train" / "000.json"
    result_path.parent.mkdir(parents=True)
    result = {"checkpoint_key": "cp10", "context": "candidate_01", "partition": "train", "index": 0,
              "task_id": "task1", "session_id": "session1", "steps": 1, "max_steps": 20,
              "status": "completed", "session_status": "success", "score": 1}
    session_root = root
    if contributor:
        session_root = root / "contributors" / "front_8006"
        result["contributor_output"] = str(tmp_path / "outputs" / "experiment" / "contributors" / "front_8006")
    result_path.write_text(json.dumps(result))
    run = "alignment_cp10_candidate_01_train_000"
    session = session_root / "sessions" / "cp10" / "candidate_01" / "train" / run / "sessions" / "session1"
    session.mkdir(parents=True)
    events = [_action(1, arguments={"file": "/visible/task/path"}), _observation(1, '{"value": 42}')]
    trajectory = session / "trajectory.jsonl"
    trajectory.write_text("".join(json.dumps({**event, "task_id": "task1", "session_id": "session1",
                                            "run_id": run}) + "\n" for event in events))
    (session / "session.json").write_text(json.dumps({"task": "Read the visible file", "task_id": "task1",
                                                     "run_id": run, "context": {"hidden": "NEVER EXPOSE"},
                                                     "initial_config": {"answer": "PRIVATE ANSWER"}}))
    return result_path, trajectory


@pytest.mark.parametrize("contributor", [False, True])
def test_resolver_handles_direct_and_moved_contributor_archives_and_keeps_only_visible_evidence(tmp_path, contributor):
    result, trajectory = _archive(tmp_path, contributor)
    assert resolve_trajectory(result) == trajectory
    output = extract_features(result)
    assert output["available"] is True and output["features"]["observed_tool_calls"] == 1
    assert output["visible_requests"] == ["Read the visible file"]
    compact = json.dumps(output["compact_trajectory"])
    assert "/visible/task/path" in compact
    assert "NEVER EXPOSE" not in compact and "PRIVATE ANSWER" not in compact
    assert '"score"' not in compact and "session_status" not in compact


def test_resolver_can_find_a_contributor_when_the_recorded_worker_root_is_missing(tmp_path):
    result_path, trajectory = _archive(tmp_path, True)
    row = json.loads(result_path.read_text())
    row["contributor_output"] = "/missing/obsolete/worker"
    result_path.write_text(json.dumps(row))
    assert resolve_trajectory(result_path) == trajectory


def test_missing_trace_is_explicit_missing_coverage_not_a_zero_cost_policy(tmp_path):
    result, trajectory = _archive(tmp_path)
    trajectory.unlink()
    output = extract_features(result)
    assert output["available"] is False
    assert output["features"] is output["compact_trajectory"] is None
    assert output["missing_reason"] == "trajectory_not_found"


def test_feature_extraction_rejects_validation_results_and_mismatched_session_identity(tmp_path):
    result_path, trajectory = _archive(tmp_path)
    row = json.loads(result_path.read_text())
    row["partition"] = "val"
    result_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="sampled training"):
        extract_features(result_path)
    row["partition"] = "train"
    result_path.write_text(json.dumps(row))
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    records[0]["task_id"] = "different-task"
    trajectory.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="identity differs"):
        extract_features(result_path)
