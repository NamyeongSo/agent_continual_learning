# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from scripts.litellm.evaluate_terminalbench21_retry_queue import predecessor_finished, validate_retry
from scripts.litellm.evaluate_terminus2_after_queue import validate_tasks


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_dependency_waits_for_every_native_task_and_final_marker(tmp_path):
    tasks = [{"task_id": str(i)} for i in range(89)]
    write(tmp_path / "config.json", {"tasks": tasks})
    write(tmp_path / "status.json", {"phase": "evaluating"})
    for task in tasks:
        write(tmp_path / "records" / f"{task['task_id']}.json", {"status": "completed"})
    assert not predecessor_finished(tmp_path)
    write(tmp_path / "status.json", {"phase": "infrastructure_errors"})
    write(tmp_path / "records/88.json", {"status": "infrastructure_error"})
    assert predecessor_finished(tmp_path)
    (tmp_path / "records/88.json").unlink()
    assert not predecessor_finished(tmp_path)


def test_retry_subset_excludes_measured_zeros(tmp_path):
    tasks = [{"task_id": str(i)} for i in range(6)]
    write(tmp_path / "config.json", {"tasks": tasks})
    for i in range(6):
        write(tmp_path / "records" / f"{i}.json",
              {"status": "infrastructure_error" if i < 5 else "completed", "score": 0})
    config = {"retry_source": str(tmp_path), "tasks": tasks[:5], "workers": 10,
              "model_settings": {"disable_request_timeout": True}}
    validate_retry(config)
    config["tasks"] = tasks[1:]
    with pytest.raises(ValueError, match="five original unmeasured"):
        validate_retry(config)


def test_native_timeout_retry_excludes_successful_timeouts_and_other_failures(tmp_path):
    tasks = [{"task_id": str(i)} for i in range(4)]
    write(tmp_path / "config.json", {"tasks": tasks})
    for i, (success, timed_out) in enumerate([(False, True), (True, True), (False, False), (True, False)]):
        write(tmp_path / "records" / f"{i}.json", {
            "status": "completed", "success": success,
            "session": {"session_metadata": {"agent_timed_out": timed_out}},
        })
    config = {"tasks": tasks[:1], "retry_timeout_source": str(tmp_path)}
    validate_tasks(config)
    for selection in (tasks[:2], tasks[1:2], tasks[:1] * 2, []):
        with pytest.raises(ValueError):
            validate_tasks({**config, "tasks": selection})
