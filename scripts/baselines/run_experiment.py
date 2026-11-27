# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Launch one isolated official baseline in its pinned optimizer environment."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.baselines.prepare import LOCK, check_sources, verify_prompts  # noqa: E402
from scripts.baselines.runtime import (  # noqa: E402
    ACTOR_CONTEXT_MODES,
    DEFAULT_ACTOR_CONTEXT,
    Runtime,
    actor_context_metadata,
    digest,
    read,
    write,
)


def make_config(args):
    from exgentic.agents.premise.pipeline import manifest_order
    from exgentic.core.experiment_limits import experiment_limits
    from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

    order, configs = manifest_order(args.manifest, [args.benchmark], args.seed, None, "isolated")
    terminalbench2_version = getattr(args, "terminalbench2_version", None)
    if terminalbench2_version is not None:
        if args.benchmark != "terminalbench2":
            raise ValueError("--terminalbench2-version requires the Terminal-Bench benchmark")
        if terminalbench2_version not in ("2.0", "2.1"):
            raise ValueError(f"Unsupported Terminal-Bench release: {terminalbench2_version}")
        configs["terminalbench2"]["subset"] = terminalbench2_version
    policy = getattr(args, "appworld_policy", None)
    policy_config = {}
    if policy is not None:
        if args.benchmark != "appworld":
            raise ValueError("--appworld-policy requires the AppWorld benchmark")
        text = policy.read_text(encoding="utf-8")
        configs["appworld"]["policy_override"] = text
        policy_config = {"appworld_policy_sha256": digest(text), "appworld_policy_source": str(policy.resolve())}
    section = select_task_split_sections(read(args.manifest), [args.benchmark], args.seed)[args.benchmark]
    splits = validate_task_split(section, args.benchmark)
    train = [{"task_id": task_id, "split": "train"} for _, task_id in order]
    validation = [{"task_id": task_id, "split": "val"} for task_id in splits["val"]]
    test = [{"task_id": task_id, "split": "test"} for task_id in splits["test"]]
    if args.pilot:
        train, validation, test = train[:5], validation[:2], test[:2]
    actor_context = getattr(args, "actor_context", DEFAULT_ACTOR_CONTEXT)
    return {
        "schema_version": 1, "method": args.method, "benchmark": args.benchmark, "seed": args.seed,
        "mode": "isolated", "model": args.model, "base_url": args.base_url, "workers": args.workers,
        "temperature": 0.7, "thinking": False, "actor_input_tokens": 100000, "actor_output_tokens": 2048,
        "actor_max_turns": 30 if args.benchmark == "bfcl" else 40, "analysis_output_tokens": 8192,
        "experiment_limits": experiment_limits(args.benchmark),
        "analysis_input_limit": "model_context_only", "total_training_limit": None,
        "termination": "single_pass_stream_exhausted", "pilot": args.pilot,
        "output": str(args.output.resolve()), "core_python": str(ROOT / ".venv/bin/python"),
        "train": train, "validation": validation, "test": test, "benchmark_kwargs": configs[args.benchmark],
        "split_manifest_sha256": digest(args.manifest.read_text(encoding="utf-8")),
        "protocol_sha256": digest((Path(__file__).parent / "protocol.toml").read_text(encoding="utf-8")),
        "actor_context": actor_context,
        "actor_context_metadata": actor_context_metadata(actor_context, [args.benchmark]),
        "actor_temperature": getattr(args, "actor_temperature", None),
        "analysis_temperature": getattr(args, "analysis_temperature", 0.7),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "optimizer_seed": (getattr(args, "optimizer_seed", None)
                           if getattr(args, "optimizer_seed", None) is not None else args.seed),
        **policy_config,
    }


def worker(config_path, scripted):
    config = read(config_path)
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    for name in lock["environments"][config["method"]]["sources"]:
        if importlib.metadata.version(name) != lock["sources"][name]["version"]:
            raise RuntimeError(f"Unexpected installed {name} version")
    runtime = Runtime(config, scripted=scripted)
    with (runtime.output / "optimizer.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (runtime.output / "summary.json").exists():
            print(json.dumps(read(runtime.output / "summary.json")), flush=True)
            return
        # Reconstruct optimizer state from its initial state and immutable cached
        # actor/teacher decisions. This also restores SkillOpt's rejected-edit
        # buffer, which its upstream checkpoint loader does not restore.
        generations = list((runtime.output / "upstream").glob("replay_*"))
        upstream_dir = runtime.output / "upstream" / f"replay_{len(generations):04d}"
        upstream_dir.mkdir(parents=True, exist_ok=False)
        write(runtime.output / "status.json", {"phase": "running", "pid": os.getpid(),
                                               "upstream_dir": str(upstream_dir), "scripted": scripted})
        try:
            if config["method"] == "gepa":
                from scripts.baselines.gepa_runtime import train
            else:
                from scripts.baselines.skillopt_runtime import train
            result = train(runtime, upstream_dir)
            if len(list((runtime.output / "online").glob("*.json"))) != len(config["train"]):
                raise RuntimeError("Train stream not completely exposed exactly once")
            write(runtime.output / "status.json", {"phase": "completed", "result": result, "scripted": scripted})
        except Exception as exc:
            write(runtime.output / "status.json", {"phase": "error", "error": f"{type(exc).__name__}: {exc}"})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("gepa", "skillopt"))
    parser.add_argument("--benchmark", choices=("bfcl", "appworld", "terminalbench2"))
    parser.add_argument("--terminalbench2-version", choices=("2.0", "2.1"), default=None,
                        help="Override the manifest's Terminal-Bench release for train, validation and test")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--actor-temperature", type=float)
    parser.add_argument("--analysis-temperature", type=float, default=0.7)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--optimizer-seed", type=int)
    parser.add_argument("--actor-context", choices=ACTOR_CONTEXT_MODES, default=DEFAULT_ACTOR_CONTEXT,
                        help="Shared empty initial actor policy; legacy reproduces method-specific prompts")
    parser.add_argument("--appworld-policy", type=Path, help="Frozen AppWorld environment policy template")
    parser.add_argument("--pilot", action="store_true", help="Separate 5 train / 2 val / 2 test diagnostic run")
    parser.add_argument("--scripted", action="store_true", help="Exercise official loops without any LLM/environment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config, args.scripted)
        return
    if not args.method or not args.benchmark or not args.output or not 1 <= args.workers <= 50:
        parser.error("--method, --benchmark, --output and workers in [1, 50] are required")
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    check_sources(lock, lock["environments"][args.method]["sources"])
    if not verify_prompts(lock)["ready"]:
        raise RuntimeError("Team prompt bytes differ from their provenance hashes")
    config = make_config(args)
    path = args.output.resolve() / "config.json"
    if path.exists() and read(path) != config:
        raise ValueError("Output already contains a different immutable experiment configuration")
    write(path, config)
    if args.dry_run:
        print(json.dumps({"config": str(path), "train": len(config["train"]),
                          "val": len(config["validation"]), "test": len(config["test"])}))
        return
    python = ROOT / lock["environments"][args.method]["path"] / "bin/python"
    argv = [str(python), str(Path(__file__).resolve()), "--worker-config", str(path)]
    if args.scripted:
        argv.append("--scripted")
    environment = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True", PYTHONUNBUFFERED="1")
    environment["OPENAI_BASE_URL"] = args.base_url
    environment["OPENAI_API_BASE"] = args.base_url
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    environment.setdefault("OPENAI_API_KEY", "EMPTY")
    raise SystemExit(subprocess.call(argv, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
