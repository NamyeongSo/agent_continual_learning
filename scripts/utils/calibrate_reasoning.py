# SPDX-License-Identifier: Apache-2.0
"""Run small real training/inference diagnostics using canonical training tasks only."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def environment(output):
    original = read(ROOT / "outputs/qwen_epoch2_continuation_four_baselines_20261109/launch_manifest.json")
    profile = output / "pilot_profile.json"
    env = {
        **os.environ,
        **original.get("environment", {}),
        "EXGENTIC_INFERENCE_PROFILE": str(profile.resolve()),
        "EXGENTIC_INFERENCE_PROFILE_SHA256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "OPENAI_API_BASE": "http://127.0.0.1:18007/v1",
        "OPENAI_BASE_URL": "http://127.0.0.1:18007/v1",
        "OPENAI_API_KEY": "EMPTY",
        "PYTHONUNBUFFERED": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
    return env


def prepare(output):
    manifest_path = (
        ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json"
    )
    manifest = read(manifest_path)
    original = read(ROOT / "outputs/qwen_epoch2_continuation_four_baselines_20261109/launch_manifest.json")
    ids = manifest["bfcl_42"]["train"]
    split = {
        "seed_scope": "within_split_order_only",
        "bfcl_42": {
            "train": ids[:5],
            "val": ids[5:7],
            "test": ids[7:9],
            "benchmark_kwargs": {"subset": "multi_turn_base"},
        },
    }
    write(output / "pilot_split.json", split)
    write(
        output / "data_provenance.json",
        {
            "source": str(manifest_path),
            "all_ids_are_canonical_train": True,
            "canonical_test_accessed": False,
            "diagnostic_labels_only": split,
        },
    )
    jobs = []
    for index, method in enumerate(("ace", "premise", "gepa", "skillopt")):
        directory = output / "training" / method
        common = [
            "--model",
            "openai/Qwen/Qwen3.6-35B-A3B",
            "--reasoning-effort",
            "medium",
            "--actor-context",
            "shared-empty-v1",
            "--seed",
            "42",
        ]
        if method in {"ace", "premise"}:
            command = [
                sys.executable,
                "-u",
                f"scripts/{method}/run_experiment.py",
                "--mode",
                "isolated",
                "--benchmarks",
                "bfcl",
                "--task-split-manifest",
                str(output / "pilot_split.json"),
                "--output-dir",
                str(directory),
                "--num-tasks",
                "2",
                "--playbook-edit-mode",
                "add_update_delete",
            ]
            if method == "ace":
                command += [
                    "--training-time",
                    "--actor-temperature",
                    "0.0",
                    "--analysis-temperature",
                    "0.7",
                    "--max-tokens",
                    "8192",
                ]
            else:
                command += [
                    "--base-url",
                    "http://127.0.0.1:18007/v1",
                    "--disable-alignment",
                    "--past",
                    "3",
                    "--actor-temperature",
                    "0.0",
                    "--reflection-temperature",
                    "0.7",
                    "--mutation-temperature",
                    "0.7",
                    "--llm-timeout",
                    "1800",
                ]
        else:
            command = [
                sys.executable,
                "-u",
                "scripts/baselines/run_experiment.py",
                "--method",
                method,
                "--benchmark",
                "bfcl",
                "--manifest",
                str(output / "pilot_split.json"),
                "--output",
                str(directory),
                "--workers",
                "4",
                "--actor-temperature",
                "0.0",
                "--analysis-temperature",
                "0.7",
                "--base-url",
                "http://127.0.0.1:18007/v1",
            ]
        jobs.append({"id": method, "command": command + common, "port_block_start": 50000 + 500 * index})
    for benchmark in ("appworld", "terminalbench2"):
        source = next(j for j in original["jobs"] if j["method"] == "premise" and j["benchmark"] == benchmark)
        request = {
            "method": "calibration",
            "benchmark": benchmark,
            "benchmark_kwargs": source["source_spec"]["benchmark_kwargs"],
            "item": {"task_id": manifest[f"{benchmark}_42"]["train"][0], "split": "train"},
            "model": "openai/Qwen/Qwen3.6-35B-A3B",
            "actor_context": "shared-empty-v1",
            "learned_policy": "",
            "actor_temperature": 0.0,
            "reasoning_effort": "medium",
            "session_output": str(output / "inference" / benchmark),
        }
        from exgentic.actor_context import shared_actor_prompt

        request["system_prompt"] = shared_actor_prompt(benchmark)
        if benchmark == "appworld":
            request["benchmark_kwargs"]["policy_override"] = Path(source["appworld_policy"]).read_text()
        path = output / "inference" / f"{benchmark}_request.json"
        write(path, request)
        jobs.append(
            {
                "id": benchmark,
                "port_block_start": 53000 if benchmark == "appworld" else 54000,
                "command": [
                    sys.executable,
                    "-u",
                    "scripts/baselines/actor.py",
                    "--request",
                    str(path),
                    "--response",
                    str(path.with_name(f"{benchmark}_response.json")),
                ],
            }
        )
    write(output / "pilot_jobs.json", jobs)
    return jobs


def run(output, workers):
    jobs = read(output / "pilot_jobs.json") if (output / "pilot_jobs.json").exists() else prepare(output)
    base = environment(output)

    def execute(job):
        marker = output / "results" / f"{job['id']}.json"
        if marker.exists() and read(marker)["returncode"] == 0:
            return read(marker)
        env = {**base, "EXGENTIC_PORT_BLOCK_START": str(job["port_block_start"])}
        with (output / f"pilot_{job['id']}.log").open("a") as log:
            process = subprocess.run(job["command"], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = {"id": job["id"], "returncode": process.returncode, "command": job["command"]}
        write(marker, result)
        print(job["id"], "finished", process.returncode, flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(execute, jobs))
    write(output / "pilot_summary.json", {"jobs": results, "all_completed": all(r["returncode"] == 0 for r in results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output.resolve(), args.workers)


if __name__ == "__main__":
    main()
