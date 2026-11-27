# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import pytest

from scripts.utils.benchmark_qwen_serving import request_body
from scripts.utils.run_qwen_serve_variant import command_with_batch
from scripts.utils.tune_qwen_h200_moe import candidates, pick_config, require_idle_gpu


def test_kernel_selection_requires_correctness_and_avoids_noise():
    baseline = {"BLOCK_SIZE_M": 64}
    faster = {"BLOCK_SIZE_M": 16}
    invalid = {"BLOCK_SIZE_M": 128}
    rows = [{"config": baseline, "correct": True, "microseconds": 100},
            {"config": faster, "correct": True, "microseconds": 98},
            {"config": invalid, "correct": False, "microseconds": 10}]
    assert pick_config(rows, baseline)["config"] == baseline
    rows[1]["microseconds"] = 70
    assert pick_config(rows, baseline)["config"] == faster
    with pytest.raises(RuntimeError, match="validated"):
        pick_config([rows[2]], baseline)


def test_kernel_candidates_retain_the_measured_baseline_and_valid_dimensions():
    baseline = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3}
    reference = {**baseline, "BLOCK_SIZE_M": 16}
    configs = candidates(baseline, reference)
    assert baseline in configs and reference in configs
    assert len(configs) == len({tuple(sorted(c.items())) for c in configs})
    assert all(512 % c["BLOCK_SIZE_N"] == 0 and 512 % c["BLOCK_SIZE_K"] == 0 for c in configs)


def test_serving_variant_changes_only_the_prefill_batch_budget():
    original = ["vllm", "serve", "Qwen/Qwen3.6-35B-A3B", "--max-num-batched-tokens", "65536",
                "--max-model-len", "262144", "--max-num-seqs", "512", "--kv-cache-dtype", "fp8"]
    changed = command_with_batch(original, 8192)
    assert changed[4] == "8192"
    assert changed[:4] == original[:4] and changed[5:] == original[5:]
    assert original[4] == "65536"


def test_performance_replay_does_not_mutate_the_actual_actor_request():
    source = {"kind": "action", "body": {"messages": [{"role": "user", "content": "actual task"}],
                                         "tools": [{"type": "function"}]}}
    body = request_body(source, 12, 256, fixed_output=True)
    assert source["body"]["messages"][0]["content"] == "actual task"
    assert body["tool_choice"] == "none" and body["temperature"] == 0.7
    assert body["max_tokens"] == body["min_tokens"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    natural = request_body(source, 12, 2048)
    assert natural["max_tokens"] == 2048
    assert "min_tokens" not in natural and "ignore_eos" not in natural


def test_tuner_refuses_a_gpu_used_by_another_compute_process(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "12345\n")
    with pytest.raises(RuntimeError, match="another compute process"):
        require_idle_gpu()
