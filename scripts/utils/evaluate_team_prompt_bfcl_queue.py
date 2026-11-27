# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate frozen final GEPA/SkillOpt prompts on BFCL through one common task queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import evaluate_team_prompt_queue as team  # noqa: E402
from scripts.utils import evaluate_team_prompt_validation_queue as controller  # noqa: E402

shared = team.shared


def prepare(
    source: Path, splits_path: Path, output: Path, workers: int, base_url: str,
    modes: tuple[str, ...] = ("sequential", "interleaved"), seeds: tuple[int, ...] = (42, 43, 44),
    partitions: tuple[str, ...] = ("val",), max_tokens: int = 8192, num_retries: int = 5,
) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    if workers < 1 or max_tokens < 1 or num_retries < 0:
        raise ValueError("Workers/output limit must be positive; retries must be nonnegative")
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset({42, 43, 44}):
        raise ValueError("Seeds must be unique values from 42, 43, 44")
    if not modes or len(set(modes)) != len(modes) or not set(modes).issubset({"sequential", "interleaved"}):
        raise ValueError("Modes must be unique sequential/interleaved settings")
    if not partitions or len(set(partitions)) != len(partitions) or not set(partitions).issubset({"val", "test"}):
        raise ValueError("Splits must be unique val/test values")
    split_raw = splits_path.read_bytes()
    sections = json.loads(split_raw)
    groups = []
    for method in ("gepa", "skillopt"):
        for mode in modes:
            folder = "cp130_terminalbench2" if mode == "sequential" else (
                "cp130_shared" if method == "gepa" else "cp130_shared_best"
            )
            filename = "candidate.json" if method == "gepa" else "skill.md"
            for seed in sorted(seeds):
                original = source / f"{method}_{mode}_seed{seed}" / folder / filename
                raw = original.read_bytes()
                provenance = shared.read_json(original.with_name("provenance.json"))
                identity = tuple(provenance[key] for key in ("method", "mode", "seed", "checkpoint"))
                expected_basis = "stage_best" if mode == "sequential" else (
                    "final_best" if method == "gepa" else "final_best_selected_skill"
                )
                if identity != (method, mode, seed, 130) or provenance["basis"] != expected_basis:
                    raise ValueError(f"Expected final CP130 prompt provenance: {original}")
                if provenance["training_policy"] != "legacy" or provenance["sha256"] != shared.digest(raw):
                    raise ValueError(f"Checkpoint differs from original final prompt: {original}")
                prompt = team.exported_prompt(method, raw)
                if method == "gepa" and original.with_name("prompt.md").read_text() != prompt:
                    raise ValueError(f"GEPA prompt differs from its selected candidate: {original}")
                frozen = output / "checkpoints" / method / mode / f"s{seed}" / filename
                if frozen.exists() and frozen.read_bytes() != raw:
                    raise ValueError("Frozen final checkpoint differs")
                frozen.parent.mkdir(parents=True, exist_ok=True)
                frozen.write_bytes(raw)
                shared.write_json(frozen.with_name("provenance.json"), provenance)
                selection = {"basis": provenance["basis"], "stage": provenance.get("stage", "shared")}
                if method == "gepa":
                    selection["candidate_index"] = json.loads(raw)["best_candidate_index"]
                section = validate_task_split(sections[f"bfcl_{seed}"], "bfcl")
                for partition in partitions:
                    tasks = section[partition]
                    if len(tasks) != 50:
                        raise ValueError(f"Expected 50 frozen BFCL {partition} tasks for seed {seed}")
                    groups.append({
                        "group": f"bfcl/{method}/{mode}/s{seed}/cp130/{partition}", "benchmark": "bfcl",
                        "method": method, "mode": mode, "seed": seed, "checkpoint": 130, "split": partition,
                        "task_ids": tasks, "subset": "multi_turn_base", "selection": selection,
                        "checkpoint_path": str(frozen), "checkpoint_sha256": shared.digest(raw),
                        "prompt_sha256": shared.digest(prompt.encode()), "source": str(original),
                        "provenance": provenance, "model": "openai/Qwen/Qwen3.6-35B-A3B", "temperature": 0.0,
                        "max_tokens": max_tokens, "model_num_retries": num_retries,
                        "transport_additional_retries": num_retries, "transport_retry_after": 0.5,
                        "evaluation_label": output.name,
                    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "task_split_manifest.json").write_bytes(split_raw)
    return {
        "created_at": shared.timestamp(), "source": str(source), "manifest_source": str(splits_path),
        "manifest_sha256": shared.digest(split_raw), "benchmark": "bfcl", "subset": "multi_turn_base",
        "methods": ["gepa", "skillopt"], "modes": list(modes), "seeds": sorted(seeds),
        "splits": list(partitions), "split": partitions[0] if len(partitions) == 1 else "val_test",
        "workers": workers, "base_url": base_url, "checkpoint_count": 2 * len(modes) * len(seeds),
        "evaluation_group_count": len(groups), "tasks_per_group": 50, "total_tasks": 50 * len(groups),
        "max_tokens": max_tokens, "temperature": 0.0, "model_num_retries": num_retries,
        "transport_additional_retries": num_retries, "output_length_additional_retries": shared.LENGTH_RETRIES,
        "output_length_max_attempts": shared.LENGTH_RETRIES + 1, "learning": False,
        "max_steps": 50, "max_actions": 100, "enable_thinking": False, "enable_tool_shortlisting": False,
        "prompt_application": "Exact frozen exported full system prompt through PREMiSE tool-calling runtime",
        "runner_sha256": shared.digest(Path(__file__).read_bytes()),
        "worker_runner_sha256": shared.digest(Path(team.__file__).read_bytes()),
        "controller_runner_sha256": shared.digest(Path(controller.__file__).read_bytes()),
        "groups": groups,
    }


def aggregate(output: Path, launch: dict) -> None:
    scores = shared.read_json(output / "scores.json")
    aggregates = []
    for method in launch["methods"]:
        for mode in launch["modes"]:
            for partition in launch["splits"]:
                rows = [row for row in scores["scores"] if (
                    row["method"], row["mode"], row["split"]
                ) == (method, mode, partition)]
                if len(rows) != len(launch["seeds"]) or not all(row["complete"] for row in rows):
                    raise ValueError("Final scores do not cover every requested seed")
                values = [row["hard"] for row in rows]
                aggregates.append({
                    "method": method, "mode": mode, "split": partition, "benchmark": "bfcl",
                    "checkpoint": 130, "by_seed": {str(row["seed"]): row["hard"] for row in rows},
                    "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None,
                    "std_ddof": 1, "error_scoring": "zero",
                    "execution_errors": sum(row["execution_errors"] for row in rows),
                })
    scores["aggregates"] = aggregates
    shared.write_json(output / "scores.json", scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "team_prompt")
    parser.add_argument("--splits-manifest", type=Path, default=shared.SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("sequential", "interleaved"),
                        default=["sequential", "interleaved"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"])
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-retries", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "automatic_resume_disabled.json").exists() and not args.dry_run:
            print("Evaluation remains paused; no workers started.", flush=True)
            return 0
        manifest = output / "launch_manifest.json"
        if manifest.exists():
            launch = shared.read_json(manifest)
            expected = {
                "source": str(args.source.resolve()), "manifest_source": str(args.splits_manifest.resolve()),
                "workers": args.workers, "base_url": args.base_url, "seeds": sorted(args.seeds),
                "modes": args.modes, "splits": args.splits, "max_tokens": args.max_tokens,
                "model_num_retries": args.num_retries,
            }
            if any(launch[key] != value for key, value in expected.items()):
                raise ValueError("Resume settings differ from the frozen BFCL evaluation")
        else:
            launch = prepare(
                args.source.resolve(), args.splits_manifest.resolve(), output, args.workers, args.base_url,
                tuple(args.modes), tuple(args.seeds), tuple(args.splits), args.max_tokens, args.num_retries,
            )
            shared.write_json(manifest, launch)
        if args.dry_run:
            print(json.dumps({key: value for key, value in launch.items() if key != "groups"}, indent=2))
            return 0
        result = controller.run(output, launch)
        aggregate(output, launch)
        return result


if __name__ == "__main__":
    raise SystemExit(main())
