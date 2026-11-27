# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.utils import schedule_peace_auxiliary as scheduler


def test_dependency_waits_for_all_six_valid_official_test_markers(tmp_path, monkeypatch):
    plan = {"version": "peace-appworld-pct-comparison-v1", "inference_profile_sha256": "profile",
            "manifest_sha256": "split", "jobs": [{"id": str(i), "benchmark": "appworld"} for i in range(6)]}
    monkeypatch.setattr(scheduler.shared, "read", lambda _: plan)
    completed = {str(i) for i in range(5)}
    monkeypatch.setattr(scheduler.shared, "complete", lambda job, _: job["id"] in completed)
    expected = {"job_ids": [str(i) for i in range(6)], "inference_profile_sha256": "profile",
                "manifest_sha256": "split"}
    assert scheduler.ready(tmp_path, expected) == (False, 5)
    completed.add("5")
    assert scheduler.ready(tmp_path, expected) == (True, 6)
    with pytest.raises(ValueError, match="identities"):
        scheduler.ready(tmp_path, {**expected, "inference_profile_sha256": "changed"})
