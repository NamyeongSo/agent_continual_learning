# SPDX-License-Identifier: Apache-2.0
"""Run parallel ADD-only training on sign-filtered, saved Reflection groups."""

import argparse
import hashlib
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, wait
from fractions import Fraction
from math import sqrt
from pathlib import Path

from filelock import FileLock

from scripts.ace.plot_worst_reflection_repeats import RUNS, SEEDS
from scripts.ace.run_reflection_accumulation import base, protocol_arms, train_condition


def score_groups(bank):
    groups = {}
    for arm in protocol_arms({"arms": list(bank)}):
        groups[arm] = {}
        for task, candidates in bank[arm].items():
            assert len(candidates) == 5
            rewards = [Fraction(c["score_numerator"], c["score_denominator"]) for c in candidates]
            mean = sum(rewards) / 5
            residuals = [r - mean for r in rewards]
            std = sqrt(float(sum(r * r for r in residuals) / 5))
            groups[arm][task] = {
                "zero_signal_group": std == 0,
                "rewards_pct": list(map(float, rewards)),
                "advantages": [float(r) / (std + 1e-8) for r in residuals],
            }
    return groups


def prepare(output, runs=RUNS):
    jobs = []
    expected_jobs = 0
    for mode, source in runs.items():
        protocol = base.read(source / "protocol.json")
        bank = base.read(source / "reflection_bank.json")
        groups = score_groups(bank)
        analysis_path = output / f"{mode}_advantage_groups.json"
        base.write(analysis_path, groups)
        source_hash = hashlib.sha256((source / "reflection_bank.json").read_bytes()).hexdigest()
        target = output / mode
        target.mkdir(parents=True, exist_ok=True)
        arms = protocol_arms(protocol)
        if set(bank) != set(arms):
            raise ValueError("Reflection bank GT arms differ from the learning protocol")
        expected_jobs += len(arms) * len(SEEDS) * 2
        for arm in arms:
            by_task = groups[arm]
            cohort = target / arm
            orders = {}
            for seed in SEEDS:
                split = protocol["task_orders"][str(seed)]
                assert set(split["train"]) == set(by_task)
                train = [task for task in split["train"] if not by_task[task]["zero_signal_group"]]
                assert set(train).isdisjoint(set(split["val"]) | set(split["test"]))
                orders[str(seed)] = {**split, "train": train}
                for strategy, sign in (("best_advantage", 1), ("worst_advantage", -1)):
                    condition = {"seed": seed, "arm": arm, "mode": strategy, "id": f"s{seed}_{arm}_{strategy}"}
                    selections = []
                    for task in train:
                        group = by_task[task]
                        candidates = bank[arm][task]
                        assert [c["mean_pct"] for c in candidates] == group["rewards_pct"]
                        eligible = [i for i, advantage in enumerate(group["advantages"]) if sign * advantage >= 0]
                        rng_seed = int(base.digest(f"advantage-v1|{seed}|{task}|{arm}|{strategy}"), 16)
                        index = random.Random(rng_seed).choice(eligible)
                        selections.append(
                            {
                                "candidate": candidates[index],
                                "advantage": group["advantages"][index],
                                "eligible_indices": eligible,
                                "tie_indices": eligible,  # Sampling pool used by the checkpoint writer.
                                "selection_rng_seed": str(rng_seed),
                            }
                        )
                    base.write(cohort / "conditions" / condition["id"] / "selections.json", selections)
                    jobs.append({"path": str(cohort), "condition": condition, "steps": len(train)})
            base.write(
                cohort / "protocol.json",
                {
                    **protocol,
                    "task_orders": orders,
                    "playbook_edit_mode": "add_only",
                    "experiment": "ace_advantage_sign_accumulation",
                    "source_accumulation": str(source),
                    "source_reflection_bank_sha256": source_hash,
                    "advantage_analysis": str(analysis_path),
                    "advantage_analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                    "selection_rule": "uniform eligible candidate per task, including zero. Skip all-zero groups.",
                },
            )
    assert len(jobs) == expected_jobs
    base.write(output / "plan.json", jobs)
    return jobs


def run(output, jobs):
    for directory in sorted({job["path"] for job in jobs}):
        protocol = base.read(Path(directory) / "protocol.json")
        base.write(
            output / f"endpoint_{Path(directory).parent.name}_{Path(directory).name}.json",
            base.check_endpoint(protocol),
        )

    def status(state, active=0, errors=()):
        counts = {}
        for job in jobs:
            directory = Path(job["path"]) / "conditions" / job["condition"]["id"]
            progress = directory / "training_status.json"
            counts[f"{Path(job['path']).parent.name}/{job['condition']['id']}"] = {
                "completed": base.read(progress)["completed"] if progress.exists() else 0,
                "total": job["steps"],
                "done": (directory / "training_completed.json").exists(),
            }
        base.write(
            output / "status.json",
            {
                "state": state,
                "pid": os.getpid(),
                "active_contexts": active,
                "errors": list(errors),
                "completed_contexts": sum(row["done"] for row in counts.values()),
                "total_contexts": len(jobs),
                "completed_steps": sum(row["completed"] for row in counts.values()),
                "total_steps": sum(row["total"] for row in counts.values()),
                "contexts": counts,
                "updated_at": base.now(),
            },
        )

    errors = []
    status("starting")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        active = {pool.submit(train_condition, job["path"], job["condition"]): job for job in jobs}
        while active:
            status("running", len(active), errors)
            finished, _ = wait(active, timeout=5, return_when="FIRST_COMPLETED")
            for future in finished:
                job = active.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    errors.append({"job": job, "error": repr(exc)})
    status("error" if errors else "completed", errors=errors)
    if errors:
        raise RuntimeError(f"{len(errors)} context trainings failed; see status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        plan = output / "plan.json"
        jobs = base.read(plan) if plan.exists() else prepare(output)
        if not args.prepare_only:
            run(output, jobs)
