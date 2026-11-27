# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Wait for AppWorld reevaluation, validate remaining checkpoints, then test each validation winner."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import evaluate_interleaved_validation_queue as runner  # noqa: E402

shared = runner.shared
POINTS = tuple(point for point in range(10, 131, 10) if point not in (50, 100, 130))
PREVIOUS = ROOT / "outputs/premise_ace_interleaved_s44_val_queue150_20261009"


def predecessor_ready(dependency: dict) -> bool:
    output = Path(dependency["output"])
    if shared.digest((output / "launch_manifest.json").read_bytes()) != dependency["launch_sha256"]:
        raise ValueError("Predecessor evaluation inputs changed")
    marker = output / "validation_complete.json"
    if not marker.exists():
        return False
    completion = shared.read_json(marker)
    observed = (
        completion.get("all_tasks_terminal"), completion.get("processed_tasks"), completion.get("checkpoint_count"),
    )
    if observed != (
        True, dependency["total_tasks"], dependency["checkpoint_count"],
    ):
        raise ValueError("Invalid predecessor completion marker")
    status = shared.read_json(output / "status.json")
    if status.get("phase") != "complete" or status.get("processed_tasks") != dependency["total_tasks"]:
        return False
    with (output / "controller.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def historical_validation(previous: Path, output: Path, launch: dict) -> tuple[list[dict], list[dict]]:
    original = shared.read_json(previous / "launch_manifest.json")
    settings = ("manifest_sha256", "max_tokens", "output_length_additional_retries", "learning", "premise_beam_rank",
                "terminalbench2_actor", "terminalbench2_enforce_session_timeouts")
    if any(original[k] != launch[k] for k in settings):
        raise ValueError("Existing validation settings differ from the new validation protocol")
    if shared.read_json(previous / "status.json")["phase"] != "complete":
        raise ValueError("The previous checkpoint 50/100/130 validation is incomplete")
    rows = shared.read_json(previous / "scores.json")["scores"]
    if {(r["method"], r["checkpoint"]) for r in rows} != {
        (method, point) for method in runner.METHODS for point in (50, 100, 130)
    } or any(not r["complete"] or r["total"] != 130 for r in rows):
        raise ValueError("Expected six complete historical validation checkpoints")
    splits = shared.read_json(output / "validation/task_split_manifest.json")
    groups = []
    for group in original["groups"]:
        if (group["seed"], group["mode"], group["split"], group["temperature"]) != (44, "interleaved", "val", 0.0):
            raise ValueError("Historical validation identity differs")
        expected = splits[f"{group['benchmark']}_44"]["val"]
        if group["task_ids"] != expected:
            raise ValueError("Historical validation task list differs")
        raw = Path(group["checkpoint_path"]).read_bytes()
        playbook = runner.checkpoint_playbook(group["method"], json.loads(raw), group["checkpoint"])
        hashes = (shared.digest(raw), shared.digest(playbook.encode()))
        if hashes != (group["checkpoint_sha256"], group["playbook_sha256"]):
            raise ValueError("Historical checkpoint changed")
        if group["model"] != launch["groups"][0]["model"]:
            raise ValueError("Historical validation model differs")
        frozen = output / "historical_checkpoints" / group["method"] / f"cp{group['checkpoint']:03d}.json"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(raw)
        groups.append({**group, "checkpoint_path": str(frozen)})
    scores = [compact_score(row, str(previous / "scores.json")) for row in rows]
    shared.write_json(output / "historical_validation_scores.json", {"scores": scores, "groups": groups,
                      "source_scores_sha256": shared.digest((previous / "scores.json").read_bytes())})
    return scores, groups


def compact_score(row: dict, source: str) -> dict:
    keys = ("method", "seed", "mode", "checkpoint", "split", "processed", "total", "complete",
            "execution_errors", "hard", "soft")
    score = {k: row[k] for k in keys}
    score["source"] = source
    score["by_benchmark"] = {
        benchmark: {"hard": summary["score_with_errors_zero"], "soft": summary["partial_score_with_errors_zero"],
                    "total": summary["total"], "execution_errors": summary["execution_errors"]}
        for benchmark, summary in row["by_benchmark"].items()
    }
    return score


def select_winners(scores: list[dict]) -> list[dict]:
    if any(not r["complete"] or r["total"] != 130 or r["hard"] is None or r["soft"] is None for r in scores):
        raise ValueError("Every candidate needs a complete 130-task validation score before test selection")
    identities = [(r["method"], r["checkpoint"]) for r in scores]
    expected = {(method, point) for method in runner.METHODS for point in range(10, 131, 10)}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Validation selection must cover all 26 unique checkpoints")
    return [max((r for r in scores if r["method"] == method),
                key=lambda r: (r["hard"], r["soft"], -r["checkpoint"])) for method in runner.METHODS]


def prepare_test(output: Path, validation: dict, groups: list[dict], winners: list[dict]) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    splits = shared.read_json(output / "validation/task_split_manifest.json")
    selected = {(r["method"], r["checkpoint"]) for r in winners}
    test_groups = []
    expected_counts = {"bfcl": 50, "appworld": 50, "terminalbench2": 29}
    for group in groups:
        if (group["method"], group["checkpoint"]) not in selected:
            continue
        benchmark = group["benchmark"]
        section = validate_task_split(splits[f"{benchmark}_44"], benchmark)
        if len(section["test"]) != expected_counts[benchmark]:
            raise ValueError("Unexpected test split size")
        identity = f"{group['method']}/interleaved/s44/cp{group['checkpoint']:03d}/test/{benchmark}"
        test_groups.append({**group, "group": identity, "split": "test", "task_ids": section["test"]})
    if len(test_groups) != 6 or sum(len(g["task_ids"]) for g in test_groups) != 258:
        raise ValueError("Expected two winners and 258 test tasks across three benchmarks")
    return {**validation, "created_at": shared.timestamp(), "split": "test", "checkpoint_count": 2,
            "total_tasks": 258, "test_by_benchmark": expected_counts, "groups": test_groups,
            "selected_checkpoints": {r["method"]: r["checkpoint"] for r in winners},
            "selection_metric": "overall validation hard; ties: soft, then earlier checkpoint"}


def write_report(output: Path, scores: list[dict], filename: str) -> None:
    columns = ["method", "checkpoint", "seed", "split", "total", "hard_pct", "soft_pct", "errors"]
    columns += [f"{benchmark}_{metric}" for benchmark in runner.BENCHMARKS for metric in ("hard_pct", "soft_pct")]
    with (output / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for score in sorted(scores, key=lambda r: (r["method"], r["checkpoint"])):
            row = {k: score[k] for k in ("method", "checkpoint", "seed", "split", "total")}
            row.update(hard_pct=100 * score["hard"], soft_pct=100 * score["soft"], errors=score["execution_errors"])
            for benchmark, values in score["by_benchmark"].items():
                row[f"{benchmark}_hard_pct"] = 100 * values["hard"]
                row[f"{benchmark}_soft_pct"] = 100 * values["soft"]
            writer.writerow(row)


def run(output: Path, plan: dict) -> int:
    while not predecessor_ready(plan["dependency"]):
        if (output / "automatic_resume_disabled.json").exists():
            shared.write_json(output / "status.json", {"phase": "paused", "active_tasks": 0})
            return 0
        shared.write_json(output / "status.json", {
            "phase": "waiting", "controller_pid": os.getpid(), "active_tasks": 0,
            "reason": "Waiting for GEPA/SkillOpt AppWorld validation and predecessor controller shutdown",
            "queued_validation_tasks": 2600, "planned_test_tasks": 258, "dependency": plan["dependency"]["output"],
            "updated_at": shared.timestamp(),
        })
        time.sleep(10)
    for stage in ("validation", "test"):
        if (output / "automatic_resume_disabled.json").exists():
            shared.write_json(output / "status.json", {"phase": "paused", "active_tasks": 0})
            return 0
        directory = output / stage
        if stage == "test":
            new = shared.read_json(output / "validation/scores.json")["scores"]
            historical = shared.read_json(output / "historical_validation_scores.json")
            scores = historical["scores"] + [compact_score(r, str(output / "validation/scores.json")) for r in new]
            winners = select_winners(scores)
            shared.write_json(output / "all_validation_scores.json", {"scores": scores})
            shared.write_json(output / "selected_checkpoints.json", {
                "criterion": "overall validation hard; ties: soft, then earlier checkpoint", "winners": winners,
            })
            write_report(output, scores, "all_validation_scores.csv")
            groups = plan["validation"]["groups"] + historical["groups"]
            launch = prepare_test(output, plan["validation"], groups, winners)
            path = directory / "launch_manifest.json"
            directory.mkdir(parents=True, exist_ok=True)
            if path.exists() and shared.read_json(path) != launch:
                prior = shared.read_json(path)
                if (prior["selected_checkpoints"], prior["groups"]) != (
                    launch["selected_checkpoints"], launch["groups"],
                ):
                    raise ValueError("Test winner or frozen inputs differ on resume")
                launch = prior
            shared.write_json(path, launch)
            records = {g["group"]: [None] * len(g["task_ids"]) for g in launch["groups"]}
            jobs = runner.queue_jobs(launch["groups"], records)
            (directory / "queue.jsonl").write_text("".join(json.dumps(job) + "\n" for job in jobs))
        else:
            launch = plan["validation"]
        shared.write_json(output / "status.json", {"phase": stage, "controller_pid": os.getpid(),
                          "stage_status": str(directory / "status.json"), "updated_at": shared.timestamp()})
        with (directory / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner.run(directory, launch)
    test_scores = [compact_score(r, str(output / "test/scores.json"))
                   for r in shared.read_json(output / "test/scores.json")["scores"]]
    write_report(output, test_scores, "selected_checkpoint_test_scores.csv")
    shared.write_json(output / "status.json", {"phase": "complete", "controller_pid": os.getpid(),
                      "validation_tasks": 2600, "test_tasks": 258, "active_tasks": 0,
                      "updated_at": shared.timestamp()})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--previous-validation", type=Path, default=PREVIOUS)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output, upstream, previous = args.output.resolve(), args.after.resolve(), args.previous_validation.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "automatic_resume_disabled.json").exists() and not args.dry_run:
            print("Scheduled evaluation remains paused; no workers started.", flush=True)
            return 0
        manifest = output / "launch_manifest.json"
        if manifest.exists():
            plan = shared.read_json(manifest)
            if (plan["workers"], plan["base_url"], plan["dependency"]["output"], plan["previous_validation"]) != (
                args.workers, args.base_url, str(upstream), str(previous),
            ):
                raise ValueError("Schedule resume settings differ")
        else:
            upstream_raw = (upstream / "launch_manifest.json").read_bytes()
            predecessor = json.loads(upstream_raw)
            validation_output = output / "validation"
            validation_output.mkdir(exist_ok=True)
            validation = runner.prepare(validation_output, args.workers, args.base_url, points=POINTS)
            shared.write_json(validation_output / "launch_manifest.json", validation)
            records = {g["group"]: [None] * len(g["task_ids"]) for g in validation["groups"]}
            jobs = runner.queue_jobs(validation["groups"], records)
            (validation_output / "queue.jsonl").write_text("".join(json.dumps(job) + "\n" for job in jobs))
            historical_validation(previous, output, validation)
            plan = {"created_at": shared.timestamp(), "workers": args.workers, "base_url": args.base_url,
                    "checkpoints": list(POINTS), "excluded_validation_checkpoints": [50, 100, 130],
                    "new_validation_tasks": 2600, "test_tasks": 258, "previous_validation": str(previous),
                    "dependency": {"output": str(upstream), "launch_sha256": shared.digest(upstream_raw),
                                   "total_tasks": predecessor["total_tasks"],
                                   "checkpoint_count": predecessor["checkpoint_count"]},
                    "validation": validation, "runner_sha256": shared.digest(Path(__file__).read_bytes())}
            shared.write_json(manifest, plan)
        if args.dry_run:
            print(json.dumps({k: v for k, v in plan.items() if k != "validation"}, indent=2))
            return 0
        return run(output, plan)


if __name__ == "__main__":
    raise SystemExit(main())
