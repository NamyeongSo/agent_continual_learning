# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Replay private actor prompts against controlled Qwen serving configurations."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def request_body(template, index, output_tokens, fixed_output=False):
    body = copy.deepcopy(template["body"])
    # Diversify the first task/history message consistently across configurations.
    # Shared system/tool prefixes remain eligible for the same prefix cache policy.
    for message in body["messages"]:
        if message["role"] == "user" and isinstance(message.get("content"), str):
            message["content"] = f"[Performance replay {index:05d}]\n" + message["content"]
            break
    body.update(model="Qwen/Qwen3.6-35B-A3B", temperature=0.7, seed=42 + index,
                max_tokens=output_tokens,
                stream=True, stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": False})
    if fixed_output:
        body.update(min_tokens=output_tokens, ignore_eos=True)
    if body.get("tools"):
        body["tool_choice"] = "none"  # Same prompt rendering used by the production parser bridge.
    return body


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def one(template, index, output_tokens, endpoint, fixed_output=False):
    payload = json.dumps(request_body(template, index, output_tokens, fixed_output)).encode()
    start, first, usage = time.perf_counter(), None, None
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.perf_counter()
    end = time.perf_counter()
    if usage is None or not 0 < usage.get("completion_tokens", 0) <= output_tokens:
        raise ValueError(f"Performance request did not return valid token usage: {usage}")
    if fixed_output and usage["completion_tokens"] != output_tokens:
        raise ValueError(f"Fixed-output request did not produce exactly {output_tokens} tokens: {usage}")
    return {"index": index, "kind": template["kind"], "elapsed_seconds": end - start,
            "ttft_seconds": first - start if first else end - start,
            "decode_seconds": end - first if first else 0,
            "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]}


def run(output, name, profiles, output_tokens=2048, fixed_output=False):
    templates = json.loads((output / "private_benchmark_requests.json").read_text())
    result_path = output / f"serving_benchmark_{name}.json"
    if result_path.exists():
        raise ValueError("Use a unique benchmark name; never replace a completed comparison")
    endpoint = "http://127.0.0.1:8006/v1/chat/completions"
    report = {"name": name, "at": time.time(), "output_tokens": output_tokens,
              "fixed_output": fixed_output, "profiles": []}
    # Warm up the same representative prefixes and kernels for every server.
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda i: one(templates[i % len(templates)], 10000 + i, 16, endpoint, True), range(32)))
    for concurrency in profiles:
        count, start, rows = concurrency * 2, time.perf_counter(), []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = [pool.submit(one, templates[i % len(templates)], concurrency * 1000 + i,
                                   output_tokens, endpoint, fixed_output) for i in range(count)]
            for future in as_completed(pending):
                rows.append(future.result())
        elapsed = time.perf_counter() - start
        latencies, ttft = [r["elapsed_seconds"] for r in rows], [r["ttft_seconds"] for r in rows]
        profile = {"concurrency": concurrency, "requests": count, "elapsed_seconds": elapsed,
                   "output_tokens_per_second": sum(r["output_tokens"] for r in rows) / elapsed,
                   "p50_seconds": statistics.median(latencies), "p95_seconds": percentile(latencies, 0.95),
                   "ttft_p50_seconds": statistics.median(ttft), "rows": rows}
        report["profiles"].append(profile)
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in profile.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--output-tokens", type=int, default=2048)
    p.add_argument("--fixed-output", action="store_true")
    args = p.parse_args()
    run(args.output.resolve(), args.name, args.concurrency, args.output_tokens, args.fixed_output)
