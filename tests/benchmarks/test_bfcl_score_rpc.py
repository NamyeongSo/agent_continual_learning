"""BFCL scores must cross the venv RPC boundary without Gorilla installed."""

import json

import cloudpickle

from exgentic.benchmarks.bfcl.bfcl_eval import BFCLSession


class _GorillaLikeObject:
    def __str__(self) -> str:
        return "<Gorilla object>"


def test_score_sanitizes_nested_checker_objects_for_rpc():
    session = object.__new__(BFCLSession)
    session._result_payload = {
        "score": 0.0,
        "success": False,
        "is_finished": True,
        "session_metrics": {"accuracy": 0.0},
        "session_metadata": {
            "checker_result": {
                "valid": False,
                "details": {"model_instance_state": {"root": _GorillaLikeObject()}},
            }
        },
    }

    score = session.score()
    restored = cloudpickle.loads(cloudpickle.dumps(score))

    assert restored.session_metadata["checker_result"]["details"]["model_instance_state"]["root"] == "<Gorilla object>"
    json.dumps(restored.session_metadata)
