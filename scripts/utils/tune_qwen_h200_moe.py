# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Measure BF16 Triton MoE tiles for Qwen3.6-35B-A3B on an otherwise idle H200."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def candidates(default, reference):
    configs = [default, reference]
    for block_m, block_n, block_k, warps, stages in itertools.product(
        (16, 32, 64, 128), (64, 128), (64, 128), (4, 8), (2, 3)
    ):
        if block_m == 16 and warps == 8:
            continue
        configs.append({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                        "GROUP_SIZE_M": 1, "num_warps": warps, "num_stages": stages})
    return list({json.dumps(c, sort_keys=True): c for c in configs}.values())


def pick_config(measurements, default):
    valid = [row for row in measurements if row.get("correct") and row.get("microseconds", 0) > 0]
    if not valid:
        raise RuntimeError("No numerically validated MoE configuration remains")
    best = min(valid, key=lambda row: row["microseconds"])
    baseline = next((row for row in valid if row["config"] == default), None)
    # Avoid installing a winner inside measurement noise.
    return baseline if baseline and best["microseconds"] >= baseline["microseconds"] * 0.97 else best


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def require_idle_gpu():
    output = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    pids = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    if any(pid != os.getpid() for pid in pids):
        raise RuntimeError("GPU0 has another compute process; stop owned serving before kernel tuning")


def tune(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0 explicitly; this tuner must not touch another GPU")
    require_idle_gpu()
    import torch
    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, get_default_config

    torch.manual_seed(42)
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_name(0)
    if "H200" not in device or torch.cuda.device_count() != 1:
        raise ValueError("Expected exactly one visible H200")
    w1 = torch.randn((256, 1024, 2048), dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn((256, 2048, 512), dtype=torch.bfloat16, device="cuda") * 0.02
    import vllm

    reference_path = Path(vllm.__file__).parent / (
        "model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_H100_80GB_HBM3.json"
    )
    references = {int(k): v for k, v in json.loads(reference_path.read_text()).items() if k != "triton_version"}
    winners, reports, best_pools = {}, {}, []
    sizes = [128, 256, 2048, 8192, 1, 8, 16, 32, 64, 96, 192, 384, 512, 1024,
             4096, 16384, 32768, 65536]
    for size in sizes:
        x = torch.randn((size, 2048), dtype=torch.bfloat16, device="cuda") * 0.1
        logits = torch.randn((size, 256), dtype=torch.float32, device="cuda")
        weights, ids = torch.topk(torch.softmax(logits, dim=-1), 8, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()
        ids = ids.to(torch.int32).contiguous()
        default = get_default_config(size, 256, 512, 2048, 8, None)
        reference = references[min(references, key=lambda k: abs(k - size))]
        pool = candidates(default, reference) if size in (128, 256, 2048, 8192) else (
            list({json.dumps(c, sort_keys=True): c for c in [default, reference, *best_pools]}.values())
        )
        # Large prefills need a small set of proven candidates, rather than a full search.
        if size >= 16384:
            pool = [default, reference, winners["8192"]]
        with override_config(default):
            expected = fused_experts(x, w1, w2, weights, ids).clone()
        torch.cuda.synchronize()
        rows = []
        for index, config in enumerate(pool):
            graph = None
            try:
                with override_config(config):
                    actual = fused_experts(x, w1, w2, weights, ids)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
                    graph = torch.cuda.CUDAGraph()
                    invocations = 3 if size <= 8192 else 1
                    with torch.cuda.graph(graph):
                        for _ in range(invocations):
                            fused_experts(x, w1, w2, weights, ids)
                graph.replay()
                torch.cuda.synchronize()
                measurements = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    measurements.append(start.elapsed_time(end) * 1000 / invocations)
                rows.append({"config": config, "correct": True,
                             "microseconds": statistics.median(measurements)})
            except Exception as exc:
                rows.append({"config": config, "correct": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
                # override_config in this installed vLLM lacks a finally block.
                import vllm.model_executor.layers.fused_moe as package

                package._config = None
            finally:
                if graph is not None:
                    graph.reset()
            if index % 24 == 0:
                print(f"M={size} candidate {index + 1}/{len(pool)}", flush=True)
        selected = pick_config(rows, default)
        winner = selected["config"]
        # A different routing distribution must also retain the same BF16 operation.
        biased = logits.clone()
        biased[:, :32] += 3
        test_weights, test_ids = torch.topk(torch.softmax(biased, dim=-1), 8, dim=-1)
        test_weights = (test_weights / test_weights.sum(dim=-1, keepdim=True)).contiguous()
        test_ids = test_ids.to(torch.int32).contiguous()
        with override_config(default):
            reference_output = fused_experts(x, w1, w2, test_weights, test_ids).clone()
        with override_config(winner):
            test_output = fused_experts(x, w1, w2, test_weights, test_ids)
        torch.testing.assert_close(test_output, reference_output, atol=0.002, rtol=0.02)
        winners[str(size)] = winner
        baseline = next(row for row in rows if row["config"] == default)
        reports[str(size)] = {"selected": selected, "baseline": baseline, "candidates": rows,
                              "uniform_and_biased_routing_correct": True}
        if size in (128, 256, 2048, 8192):
            best_pools.extend(row["config"] for row in sorted(
                (row for row in rows if row["correct"]), key=lambda row: row["microseconds"]
            )[:3])
        write(output / "kernel_results.json", {"device": device, "vllm": vllm.__version__, "sizes": reports})
        write(output / "configs/E=256,N=512,device_name=NVIDIA_H200.json", dict(
            sorted(winners.items(), key=lambda item: int(item[0]))
        ))
        print(f"DONE M={size}: {baseline['microseconds'] / selected['microseconds']:.2f}x", flush=True)
        del x, logits, biased, weights, ids, expected, actual, reference_output, test_output
        torch.cuda.empty_cache()
    write(output / "kernel_tuning_complete.json", {"sizes": sizes, "at": time.time(), "device": device})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    tune(parser.parse_args().output.resolve())
