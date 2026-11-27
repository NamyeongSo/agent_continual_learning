# SPDX-License-Identifier: Apache-2.0

import json

from scripts.experiment_atlas.checkpoint_progress import snapshot


def test_current_gpu_and_all_split_progress_are_read_without_exposing_private_data(tmp_path):
    (tmp_path / "checkpoint_queue_manifest.json").write_text(json.dumps({
        "journal_id": "record", "serving_gpu_indices": [1], "query_limits": {"8006": 80},
        "expected_checkpoints": 156, "initial_reused_outcomes": 3454, "environment": "PRIVATE"}))
    row = {"method": "premise", "benchmark": "appworld", "seed": 42, "point": 10, "alias": None,
           "state": "running", "policy": "PRIVATE", "splits": {"train": {
               "done": 1, "total": 50, "running": 1, "errors": 0, "score": 0, "trace": "PRIVATE"}}}
    path = tmp_path / "dashboard.json"
    path.write_text(json.dumps({"phase": "running", "checkpoints": 153, "rows": [row, {**row, "alias": "cp10"}],
                               "task_counts": {"completed": 1, "running": 1, "error": 0},
                               "gpu_queries": {"active": {"8006": 80}, "endpoints": "PRIVATE"}}))
    before = path.read_bytes()
    data = snapshot(tmp_path)
    assert data["gpu_indices"] == [1] and data["query_limits"] == {"8006": 80}
    assert data["splits"]["train"] == {"done": 1, "total": 50, "running": 1, "errors": 0}
    assert data["rows"][0]["splits"]["train"]["score"] == 0
    assert "PRIVATE" not in json.dumps(data) and path.read_bytes() == before
    assert data["expected_checkpoints"] == 156 and data["reused_outcomes"] == 3454


def test_missing_or_partial_checkpoint_files_are_unavailable(tmp_path):
    assert snapshot(tmp_path) == {"available": False}
    (tmp_path / "checkpoint_queue_manifest.json").write_text('{"journal_id":')
    assert snapshot(tmp_path) == {"available": False}
