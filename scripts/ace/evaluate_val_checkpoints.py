# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

#!/usr/bin/env python3
"""Evaluate frozen ACE playbooks at three training checkpoints on all val sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ace"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.agents.ace.playbook_store import PlaybookStore  # noqa: E402
from exgentic.core.types import ModelSettings  # noqa: E402
from exgentic.interfaces.lib.api import evaluate  # noqa: E402
from exgentic.interfaces.registry import load_agent, load_benchmark  # noqa: E402
from run_experiment import BENCHMARK_REGISTRY  # noqa: E402

BENCHMARKS = ("bfcl", "appworld", "swebench")
STAGES = (50, 100, 150)
TRAIN_BASES = {
    "sequential": ROOT / "scripts/ace/outputs/ace_parallel_train50_20260924_retry_bfcl_rpc",
    "interleaved": ROOT / "scripts/ace/outputs/ace_parallel_interleaved_train50_20260924",
}
DEFAULT_MANIFEST = ROOT / "scripts/utils/task_splits/seed42_train50_val50.json"
DEFAULT_OUTPUT = ROOT / "scripts/ace/outputs/ace_val_train50_20260926"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_dir(mode: str, seed: int) -> Path:
    tag = (f"ace_qwen_training_time_nothink_{mode}_s{seed}_n50_4096_"
           "aw50_bf50_sw100_bfcl_appworld_swebench_all")
    path = TRAIN_BASES[mode] / tag
    if not path.is_dir():
        raise FileNotFoundError(f"Missing ACE training output: {path}")
    return path


def snapshots(path: Path, mode: str) -> dict[int, tuple[Path, dict]]:
    candidates = []
    for checkpoint in path.glob("*/sessions/*/agent/playbook_checkpoint.json"):
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        candidates.append((checkpoint, payload))
    if mode == "sequential":
        expected = {
            50: {"bfcl": 50},
            100: {"bfcl": 50, "appworld": 50},
            150: {"bfcl": 50, "appworld": 50, "swebench": 50},
        }
        matches = {stage: [(p, x) for p, x in candidates
                           if x.get("benchmark_counts") == counts]
                   for stage, counts in expected.items()}
    else:
        matches = {stage: [(p, x) for p, x in candidates
                           if x.get("session_count") == stage]
                   for stage in STAGES}
    for stage, found in matches.items():
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {mode} checkpoint at {stage} in {path}; found {len(found)}")
        expected_store = f"ace_{mode}_global"
        if found[0][1].get("store_id") != expected_store:
            raise ValueError(f"Wrong store ID in {found[0][0]}")
    return {stage: found[0] for stage, found in matches.items()}


def build_jobs(manifest: dict, modes: list[str], seeds: list[int]) -> list[dict]:
    jobs = []
    for mode in modes:
        for seed in seeds:
            source = train_dir(mode, seed)
            config = json.loads((source / "experiment_config.json").read_text(encoding="utf-8"))
            if config["mode"] != mode or config["seed"] != seed or not config["disable_thinking"]:
                raise ValueError(f"Training configuration mismatch: {source}")
            if not config["training_time"] or config["benchmark_max_turns"] != {
                "bfcl": 50, "appworld": 50, "swebench": 100,
            }:
                raise ValueError(f"Unexpected training protocol: {source}")
            if len(config["task_order"]) != 150 or sum(1 for _ in (source / "online_metrics.jsonl").open()) != 150:
                raise ValueError(f"Training run is incomplete: {source}")
            for benchmark in BENCHMARKS:
                train_ids = [str(task) for slug, task in config["task_order"] if slug == benchmark]
                split = manifest["benchmarks"][benchmark]
                val_ids = list(map(str, split["val"]))
                if len(train_ids) != 50 or set(train_ids) != set(split["train"]):
                    raise ValueError(f"ACE training IDs differ from frozen train split: {source}, {benchmark}")
                if len(val_ids) != 50 or len(set(val_ids)) != 50 or set(val_ids) & set(train_ids):
                    raise ValueError(f"Invalid frozen val split: {benchmark}")
            checkpoints = snapshots(source, mode)
            for stage in STAGES:
                checkpoint, payload = checkpoints[stage]
                for benchmark in BENCHMARKS:
                    jobs.append({
                        "mode": mode, "seed": seed, "stage": stage,
                        "benchmark": benchmark, "train_dir": source,
                        "checkpoint": checkpoint, "snapshot": payload,
                        "model": config["model"],
                        "max_steps": config["benchmark_max_turns"][benchmark],
                        "task_ids": list(map(str, manifest["benchmarks"][benchmark]["val"])),
                        "benchmark_kwargs": dict(manifest["benchmarks"][benchmark]["benchmark_kwargs"]),
                    })
    return jobs


def resolve_docker_host() -> None:
    value = os.environ.get("DOCKER_HOST") or os.environ.get("ACE_DOCKER_HOST")
    if not value and not Path("/var/run/docker.sock").is_socket():
        raise RuntimeError("Set DOCKER_HOST or ACE_DOCKER_HOST, or mount a Docker socket")
    parsed = urlsplit(value or "")
    if parsed.scheme == "ssh" and parsed.port is None and parsed.hostname:
        target = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname
        config = subprocess.run(["ssh", "-G", target], capture_output=True, text=True,
                                check=True, timeout=10)
        port = next((line.split(None, 1)[1].strip() for line in config.stdout.splitlines()
                     if line.lower().startswith("port ")), "22")
        value = urlunsplit(parsed._replace(netloc=f"{parsed.netloc}:{port}"))
    if value:
        os.environ["DOCKER_HOST"] = value
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   check=True, timeout=10)


def run_job(job: dict, output_root: Path, manifest_hash: str, max_workers: int) -> dict:
    mode, seed, stage, slug = (job[key] for key in ("mode", "seed", "stage", "benchmark"))
    output = output_root / f"{mode}_s{seed}" / f"checkpoint_{stage:03d}" / slug
    summary_path = output / "summary.json"
    checkpoint_hash = sha256(job["checkpoint"])
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (existing.get("checkpoint_sha256") != checkpoint_hash
                or existing.get("manifest_sha256") != manifest_hash
                or set(existing.get("task_ids", [])) != set(job["task_ids"])
                or len(existing.get("sessions", [])) != 50):
            raise ValueError(f"Existing validation summary does not match this job: {summary_path}")
        print(f"SKIP completed {mode} s{seed} checkpoint={stage} {slug}", flush=True)
        return existing

    print(f"START {mode} s{seed} checkpoint={stage} {slug} 50 val tasks workers={max_workers}", flush=True)
    PlaybookStore.reset_all()
    agent = load_agent("ace")(
        model=job["model"], curator_model=job["model"], shuffle_mode=mode,
        benchmark_id=slug, runner="direct", model_settings=ModelSettings(max_tokens=4096),
        enable_thinking=False, training_time=False, evaluation_mode=True,
        initial_playbook=job["snapshot"]["playbook"],
        **BENCHMARK_REGISTRY[slug]["agent_kwargs"],
    )
    benchmark = load_benchmark(slug)(**job["benchmark_kwargs"])
    results = evaluate(
        benchmark=benchmark, agent=agent, task_ids=job["task_ids"],
        max_workers=max_workers, max_steps=job["max_steps"],
        output_dir=str(output), run_id=f"ace_val_{mode}_s{seed}_c{stage}_{slug}",
    )
    sessions = [{
        "task_id": str(sr.task_id),
        "score": sr.score if sr.score is not None else (1.0 if sr.success else 0.0),
        "success": sr.success,
        "status": sr.status.value if hasattr(sr.status, "value") else str(sr.status),
        "steps": sr.steps,
        "session_id": sr.session_id,
        **({"error": (sr.details.get("session_metadata") or {}).get("error")}
           if str(sr.status) == "error" else {}),
    } for sr in results.session_results]
    if len(sessions) != 50 or {item["task_id"] for item in sessions} != set(job["task_ids"]):
        raise RuntimeError(f"Validation returned incomplete or mismatched sessions for {mode} s{seed} {stage} {slug}")
    summary = {
        "mode": mode, "seed": seed, "checkpoint_stage": stage, "benchmark": slug,
        "checkpoint_path": str(job["checkpoint"]), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_session_count": job["snapshot"]["session_count"],
        "checkpoint_benchmark_counts": job["snapshot"].get("benchmark_counts", {}),
        "manifest_sha256": manifest_hash, "task_ids": job["task_ids"],
        "model": job["model"], "evaluation_mode": True,
        "max_steps": job["max_steps"], "max_workers": max_workers,
        "benchmark_score": results.benchmark_score,
        "mean_task_score": sum(float(item["score"]) for item in sessions) / 50,
        "error_count": sum(item["status"] == "error" for item in sessions),
        "errored_task_ids": [item["task_id"] for item in sessions if item["status"] == "error"],
        "sessions": sessions, "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(summary_path, summary)
    print(f"DONE {mode} s{seed} checkpoint={stage} {slug} score={summary['mean_task_score']:.3f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--port-block-start", type=int, default=42000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 12:
        parser.error("--max-workers must be 1..12")
    if not 32000 <= args.port_block_start <= 64000:
        parser.error("--port-block-start must be 32000..64000")
    os.environ["EXGENTIC_PORT_BLOCK_START"] = str(args.port_block_start)
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8006/v1")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = build_jobs(manifest, ["sequential", "interleaved"], [42, 43, 44])
    print(f"ACE validation: {len(jobs)} evaluations, 50 tasks each, max_workers={args.max_workers}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job['mode']} s{job['seed']} checkpoint={job['stage']} val={job['benchmark']} "
                  f"snapshot_session_count={job['snapshot']['session_count']}")
        return
    resolve_docker_host()
    manifest_hash = sha256(args.manifest)
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}]", flush=True)
        run_job(job, args.output_root, manifest_hash, args.max_workers)
    print("All 54 ACE validation evaluations complete", flush=True)


if __name__ == "__main__":
    main()
