# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Learn trajectory-judge rubrics using PReMISE (arXiv:2605.30803v1)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import threading
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise import rubric_audit, rubric_pipeline, rubric_prompts
from exgentic.agents.premise.reward_data import write_json
from exgentic.agents.premise.reward_pipeline import RewardLLM
from exgentic.agents.premise.rubric_audit import AuditConfig, RubricAuditor
from exgentic.agents.premise.rubric_pipeline import LocalEmbeddings, RubricConfig, RubricPipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True, help="Existing frozen reward dataset directory")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    result.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    result.add_argument(
        "--embedding-python",
        default=sys.executable,
        help="Interpreter with torch/transformers; embeddings execute on CPU",
    )
    result.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="Cached HuggingFace model name or local snapshot directory",
    )
    for flag, default in (
        ("target-size", 20),
        ("cluster-size", 25),
        ("max-levels", 10),
        ("pool-size", 100),
        ("min-support", 10),
        ("min-applicable", 10),
        ("criteria-min", 3),
        ("criteria-max", 5),
        ("score-batch-size", 5),
        ("workers", 50),
        ("seed", 42),
        ("train-limit-per-benchmark", 0),
        ("val-limit-per-benchmark", 0),
        ("audit-tasks-per-benchmark", 2),
    ):
        result.add_argument(f"--{flag}", type=int, default=default)
    result.add_argument("--mmr-weight", type=float, default=0.7)
    result.add_argument("--preference-threshold", type=float, default=0.6)
    result.add_argument("--audit", action="store_true")
    result.add_argument("--repair", action="store_true", help="Requires --audit and three distinct audit judges")
    result.add_argument(
        "--audit-judges", type=Path, help='JSON list of {"base_url": ".../v1", "model": "..."}; contains no API keys'
    )
    result.add_argument("--audit-pairs-per-benchmark", type=int, default=6)
    result.add_argument("--repair-beam", type=int, default=2)
    result.add_argument("--repair-depth", type=int, default=2)
    result.add_argument("--reliability-slack", type=float, default=0)
    result.add_argument("--preference-slack", type=float, default=0)
    result.add_argument("--repair-packet-token-limit", type=int, default=16384)
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--retry-errors", action="store_true")
    return result


def execute(args):
    if args.repair and not args.audit:
        raise ValueError("--repair requires --audit")
    if (args.output / "config.json").exists() and not args.resume:
        raise ValueError("Existing run: use --resume, or choose a separate output directory")
    config = RubricConfig(
        **{key: getattr(args, key) for key in asdict(RubricConfig()) if key != "audit_tasks_per_benchmark"},
        audit_tasks_per_benchmark=args.audit_tasks_per_benchmark if args.audit else 0,
    )
    audit_config = AuditConfig(
        pairs_per_benchmark=args.audit_pairs_per_benchmark,
        repair_beam=args.repair_beam,
        repair_depth=args.repair_depth,
        reliability_slack=args.reliability_slack,
        preference_slack=args.preference_slack,
        packet_max_tokens=args.repair_packet_token_limit,
    )
    specs = (
        json.loads(args.audit_judges.read_text())
        if args.audit_judges
        else [{"base_url": args.base_url, "model": args.model}]
    )
    if not isinstance(specs, list) or not specs or any(set(spec) != {"base_url", "model"} for spec in specs):
        raise ValueError("Invalid audit judge configuration")
    if args.repair and len({(spec["base_url"], spec["model"]) for spec in specs}) < 3:
        raise ValueError(
            "Repair requires three distinct judge models/endpoints; repeated calls are not independent judges"
        )
    dataset = args.output / "dataset"
    if not dataset.exists():
        shutil.copytree(args.dataset, dataset)
    if (dataset / "manifest.json").read_bytes() != (args.dataset / "manifest.json").read_bytes():
        raise ValueError("Dataset source differs from the frozen run snapshot")
    slots = threading.BoundedSemaphore(args.workers)
    llm = RewardLLM(args.base_url, args.model, args.output / "llm_calls", slots)
    worker = ROOT / "scripts/premise/embed_rubric_criteria.py"
    embeddings = LocalEmbeddings(args.embedding_python, args.embedding_model, worker, args.output / "embeddings")
    pipeline = RubricPipeline(dataset, args.output, llm, embeddings, config, retry_errors=args.retry_errors)
    launch = {
        "run": pipeline.identity,
        "audit": args.audit,
        "repair": args.repair,
        "audit_settings": asdict(audit_config),
        "audit_judges": specs,
        "source_files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(rubric_pipeline.__file__),
                Path(rubric_prompts.__file__),
                Path(rubric_audit.__file__),
                worker,
            )
        },
    }
    launch_path = args.output / "launch_manifest.json"
    if launch_path.exists() and json.loads(launch_path.read_text()) != launch:
        raise ValueError("Existing launch/audit configuration differs")
    write_json(launch_path, launch)
    for relative in launch["source_files"]:
        destination = args.output / "source_snapshot" / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train": len(pipeline.train),
                "val": len(pipeline.val),
                "reserved_audit": len(pipeline.audit),
                "workers": config.workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        pipeline.status(status="prepared", stage="prepared")
        return
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(20):
            pipeline.status()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        rubrics = pipeline.run()
        if args.audit and not rubrics["preference_repaired"]:
            write_json(
                args.output / "audit" / "skipped.json",
                {"reason": "No qualifying preference-repaired criteria", "audit": args.audit, "repair": args.repair},
            )
            return
        if args.audit:
            judges = [RewardLLM(spec["base_url"], spec["model"], args.output / "llm_calls", slots) for spec in specs]
            auditor = RubricAuditor(pipeline, judges, audit_config)
            audit = auditor.run(rubrics["preference_repaired"])
            if args.repair:
                rubrics["robustness_repaired"] = auditor.repair(rubrics["preference_repaired"], audit)
                shutil.copy2(args.output / "validation.json", args.output / "validation_before_robustness_repair.json")
                pipeline.evaluate(rubrics)
            validation = json.loads((args.output / "validation.json").read_text())
            errors = sum(row["status"] != "completed" for row in validation["results"])
            pipeline.status(status="completed_with_errors" if errors else "completed", stage="finished", errors=errors)
    except Exception as exc:
        pipeline.status(status="error", error_type=type(exc).__name__, error_message=str(exc))
        raise
    finally:
        stopped.set()
        thread.join(timeout=2)


def main():
    args = parser().parse_args()
    args.output = args.output.absolute()
    args.dataset = args.dataset.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == "__main__":
    main()
