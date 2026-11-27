# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Freeze and join 18 AppWorld teammate-policy runs to the existing Qwen services."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.baselines.run_experiment import make_config  # noqa: E402
from scripts.baselines.run_queue import source_files  # noqa: E402
from scripts.utils import run_qwen_isolated_when_free as runner  # noqa: E402


def api(path, payload=None, method=None):
    url = "http://127.0.0.1:8878"
    if payload is None:
        return json.load(urllib.request.urlopen(url + path, timeout=30))
    token = api("/api/info")["csrf_token"]
    request = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        method=method or "POST",
        headers={"Content-Type": "application/json", "X-Journal-Token": token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def snapshot_source(output):
    snapshot = output / "code_snapshot"
    snapshot.mkdir(parents=True)
    for relative in source_files(ROOT):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, snapshot / name)
    (snapshot / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    baseline = snapshot / ".exgentic/baselines"
    baseline.mkdir(parents=True)
    for name in ("venvs", "upstream"):
        (baseline / name).symlink_to(ROOT / ".exgentic/baselines" / name, target_is_directory=True)
    hashes = {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("src", "scripts")
        for path in (snapshot / name).rglob("*")
        if path.is_file()
    }
    runner.write(output / "source_hashes.json", hashes)
    return snapshot


def create_journal(root, title, method, policy_sha, command):
    existing = [d for d in api("/api/experiments")["experiments"] if d.get("run_path") == str(root)]
    if existing:
        raise ValueError("This experiment already has a journal record")
    return api(
        "/api/experiments",
        {
            "title": title,
            "purpose": "AppWorld 환경 폴리시만 팀원 버전으로 바꾼 학습을 원래 폴리시와 비교",
            "hypothesis": "팀원 폴리시의 API·완료 안내가 같은 조건의 학습과 TGC에 미치는 영향 비교",
            "status": "planned",
            "method": method,
            "benchmark": "AppWorld · isolated",
            "model": runner.MODEL,
            "seeds": "42, 43, 44",
            "run_path": str(root),
            "split": "기존 frozen AppWorld train/val/test 각50개, 같은 seed별 학습 순서",
            "budget": "actor-turn-v1: input100000/output2048/40turns; all-role temperature0.7; "
            "누적 학습 예산 무제한; 기존 내부 병렬성 유지; 5-task CP 공용큐100 workers",
            "command": command,
            "tags": ["appworld", "teammate-policy", "isolated", "qwen36"],
            "summary": "새로운 초기 학습. AppWorld Online/Offline/Offline-val은 TGC. policy SHA256=" + policy_sha,
        },
    )["id"]


def prepare(output, service_owner, queue_output, policy):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_source(output)
    frozen_policy = output / "appworld_teammate_policy.txt"
    shutil.copy2(policy, frozen_policy)
    policy_sha = runner.digest(frozen_policy)
    tag = output.name.removeprefix("qwen_appworld_teammate_")
    manifest = snapshot / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    children = []
    source_roots = {
        "add_only": ROOT / "outputs/qwen36_isolated_ace_premise_temp07_20261021_025529",
        "add_update_delete": ROOT
        / ("outputs/qwen36_isolated_ace_premise_curator_add_update_delete_temp07_20261021_045725"),
    }
    for index, (mode, original) in enumerate(source_roots.items()):
        root = output / f"qwen_appworld_team_{mode}_{tag}"
        root.mkdir()
        (root / "code_snapshot").symlink_to(snapshot, target_is_directory=True)
        plan = copy.deepcopy(runner.read(original / "launch_manifest.json"))
        jobs = []
        for old in plan["jobs"]:
            if old["benchmark"] != "appworld":
                continue
            folder = root / "appworld" / old["method"] / f"s{old['seed']}"
            command = [
                str(snapshot / token.split("/code_snapshot/", 1)[1]) if "/code_snapshot/" in token else token
                for token in old["command"]
            ]
            command[command.index("--task-split-manifest") + 1] = str(manifest)
            command[command.index("--output-dir") + 1] = str(folder / "train")
            command += ["--appworld-policy", str(frozen_policy)]
            jobs.append(
                {
                    **old,
                    "output": str(folder),
                    "train_dir": str(folder / "train"),
                    "command": command,
                    "status": "pending",
                    "attempts": 0,
                    "appworld_policy": str(frozen_policy),
                    "appworld_policy_sha256": policy_sha,
                }
            )
        assert len(jobs) == 6
        plan.update(
            created_at=runner.now(),
            jobs=jobs,
            train_workers=6,
            train_workers_per_method={"ace": 3, "premise": 3},
            training_iterations=300,
            test_tasks=300,
            gpu=0,
            mode="isolated",
            manifest=str(manifest),
            manifest_sha256=runner.digest(manifest),
            port_block_base=28000 + index * 7000,
            source_snapshot=str(snapshot),
            playbook_edit_mode=mode,
            services_from=str(service_owner),
            appworld_policy=str(frozen_policy),
            appworld_policy_sha256=policy_sha,
            policy_name="teammate",
            joined_services=True,
        )
        plan.pop("after_output", None)
        command = (
            f"{snapshot}/.venv/bin/python {snapshot}/scripts/utils/run_appworld_team_policy.py --legacy-root {root}"
        )
        plan["journal_id"] = create_journal(
            root,
            f"AppWorld 팀원 폴리시 · ACE/PREMiSE · {mode} · seeds42-44",
            "ACE/PREMiSE " + mode,
            policy_sha,
            command,
        )
        runner.write(root / "journal_record.json", {"id": plan["journal_id"]})
        runner.write(root / "launch_manifest.json", plan)
        runner.write(root / "progress.json", {"at": runner.now(), "jobs": jobs})
        runner.write(root / "status.json", {"phase": "planned", "at": runner.now()})
        children.append({"root": str(root), "kind": "legacy", "journal_id": plan["journal_id"]})
    root = output / f"qwen_appworld_team_optimizers_{tag}"
    root.mkdir()
    (root / "code_snapshot").symlink_to(snapshot, target_is_directory=True)
    jobs = []
    for method in ("gepa", "skillopt"):
        for seed in (42, 43, 44):
            folder = root / "runs" / f"{method}_appworld_{seed}"
            args = SimpleNamespace(
                manifest=manifest,
                benchmark="appworld",
                seed=seed,
                pilot=False,
                method=method,
                model=runner.MODEL,
                base_url=runner.BASE_URL,
                workers=50,
                output=folder,
                appworld_policy=frozen_policy,
            )
            config = make_config(args)
            config["core_python"] = str(snapshot / ".venv/bin/python")
            runner.write(folder / "config.json", config)
            command = [
                str(snapshot / ".venv/bin/python"),
                str(snapshot / "scripts/baselines/run_experiment.py"),
                "--method",
                method,
                "--benchmark",
                "appworld",
                "--seed",
                str(seed),
                "--workers",
                "50",
                "--output",
                str(folder),
                "--appworld-policy",
                str(frozen_policy),
            ]
            jobs.append(
                {
                    "id": f"{method}_appworld_{seed}",
                    "method": method,
                    "benchmark": "appworld",
                    "seed": seed,
                    "output": str(folder),
                    "command": command,
                    "status": "pending",
                    "appworld_policy_sha256": policy_sha,
                }
            )
    plan = {
        "created_at": runner.now(),
        "after_output": str(service_owner),
        "join_output": str(service_owner),
        "snapshot": str(snapshot),
        "jobs": jobs,
        "max_running": 6,
        "max_running_per_method": 3,
        "internal_workers": 50,
        "model": runner.MODEL,
        "temperature": 0.7,
        "mode": "isolated",
        "playbook_edit_mode": "add_only",
        "total_training_limit": None,
        "termination": "single_pass_stream_exhausted",
        "manifest_sha256": runner.digest(manifest),
        "appworld_policy": str(frozen_policy),
        "appworld_policy_sha256": policy_sha,
        "policy_name": "teammate",
    }
    plan["journal_id"] = create_journal(
        root,
        "AppWorld 팀원 폴리시 · GEPA/SkillOpt · seeds42-44",
        "GEPA/SkillOpt",
        policy_sha,
        f"{snapshot}/scripts/baselines/run_queue.py --run --output {root}",
    )
    runner.write(root / "launch_manifest.json", plan)
    runner.write(root / "progress.json", {"at": runner.now(), "jobs": jobs})
    runner.write(root / "status.json", {"phase": "planned", "at": runner.now()})
    children.append({"root": str(root), "kind": "native", "journal_id": plan["journal_id"]})
    cohort = {
        "created_at": runner.now(),
        "service_owner": str(service_owner),
        "queue_output": str(queue_output),
        "snapshot": str(snapshot),
        "children": children,
        "experiment_count": 18,
        "appworld_policy": str(frozen_policy),
        "appworld_policy_sha256": policy_sha,
        "manifest_sha256": runner.digest(manifest),
        "actor_policy": "actor-turn-v1",
        "appworld_metric": "appworld_tgc_v1",
        "shared_queue_concurrency": 100,
    }
    runner.write(output / "cohort.json", cohort)
    return cohort


def legacy(root):
    from filelock import FileLock

    from scripts.utils.run_qwen_followup import reuse_services

    plan = runner.read(root / "launch_manifest.json")
    with FileLock(str(root / "controller.lock"), timeout=0):
        runner.write(root / "controller_process.json", {"pid": os.getpid(), "started_at": runner.now()})
        server, bridge = reuse_services(Path(plan["services_from"]), root)
        runner.journal(plan, root, "running", "팀원 폴리시 AppWorld 학습 시작. 각 방법3 seeds 동시; 내부 병렬성 유지.")
        runner.write(root / "status.json", {"phase": "running", "at": runner.now()})
        try:
            jobs = runner.run_training_and_evaluation(plan, root, server, bridge, resume=True)
            errors = [j["id"] for j in jobs if j["status"] != "completed"]
            phase = "blocked" if errors else "completed"
            runner.write(root / "status.json", {"phase": phase, "at": runner.now(), "failed_jobs": errors})
            runner.journal(
                plan, root, phase, f"팀원 폴리시 {len(jobs) - len(errors)}/{len(jobs)} 완료, 미완료 {errors}"
            )
        except Exception as exc:
            runner.write(root / "status.json", {"phase": "blocked", "error": f"{type(exc).__name__}: {exc}"})
            raise


def start(output):
    import psutil

    from scripts.utils.checkpoint_sources import digest
    from scripts.utils.process_identity import stable_identity

    cohort = runner.read(output / "cohort.json")
    if any((Path(c["root"]) / "launched_controller.json").exists() for c in cohort["children"]):
        raise ValueError("This cohort has already been started; inspect its existing controllers")
    snapshot, queue = Path(cohort["snapshot"]), Path(cohort["queue_output"])
    owner = runner.read(queue / "controller_process.json")
    process = psutil.Process(owner["pid"])
    environment = runner.experiment_environment(process.environ())
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        environment[key] = "1"
    private = output / "private_environment.json"
    descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(environment, stream)
    for child in cohort["children"]:
        root = Path(child["root"])
        plan = runner.read(root / "launch_manifest.json")
        for job in plan["jobs"]:
            runner.write(
                Path(job["output"]) / "checkpoint_queue.json",
                {
                    "output": str(queue),
                    "stream": digest(str(Path(job["output"]).resolve()))[:20],
                    "point": 50,
                    "proxy_module": str(snapshot / "scripts/utils/checkpoint_eval_proxy.py"),
                },
            )
        if child["kind"] == "legacy":
            command = [
                str(snapshot / ".venv/bin/python"),
                str(snapshot / "scripts/utils/run_appworld_team_policy.py"),
                "--legacy-root",
                str(root),
            ]
        else:
            command = [
                str(snapshot / ".venv/bin/python"),
                str(snapshot / "scripts/baselines/run_queue.py"),
                "--run",
                "--output",
                str(root),
                "--poll-seconds",
                "10",
            ]
        with (root / "controller.log").open("ab") as log:
            p = subprocess.Popen(
                command,
                cwd=snapshot,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        runner.write(
            root / "launched_controller.json",
            {"pid": p.pid, "command": command, **stable_identity(p.pid), "at": runner.now()},
        )
    old = {"pid": process.pid, "command": process.cmdline(), **stable_identity(process.pid)}
    runner.write(output / "queue_handoff.json", {"old_controller": old, "phase": "draining", "at": runner.now()})
    # Finish currently running evaluations before replacing their loaded code.
    runner.write(
        queue / "pause_dispatch.json", {"reason": "AppWorld policy-aware worker upgrade", "owner": str(output)}
    )
    command = [
        str(snapshot / ".venv/bin/python"),
        str(snapshot / "scripts/utils/run_appworld_team_policy.py"),
        "--handoff",
        str(output),
    ]
    with (output / "queue_handoff.log").open("ab") as log:
        p = subprocess.Popen(
            command,
            cwd=snapshot,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    runner.write(output / "handoff_process.json", {"pid": p.pid, "command": command, **stable_identity(p.pid)})


def handoff(output):
    import psutil

    from scripts.utils.process_identity import matches_identity, stable_identity

    cohort = runner.read(output / "cohort.json")
    queue = Path(cohort["queue_output"])
    old = runner.read(output / "queue_handoff.json")["old_controller"]
    process = psutil.Process(old["pid"])
    assert matches_identity(old, process.create_time()) and process.cmdline() == old["command"]
    while True:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("The original queue controller exited while draining")
        heartbeat = runner.read(queue / "heartbeat.json")
        marker = runner.read(queue / "pause_dispatch.json")
        if marker.get("owner") != str(output):
            raise RuntimeError("Another operation replaced the queue pause marker")
        runner.write(
            output / "queue_handoff_status.json",
            {"phase": "draining", "active": heartbeat["active"], "at": runner.now()},
        )
        if heartbeat["active"] == 0:
            break
        time.sleep(10)
    children = [{"pid": child.pid, **stable_identity(child.pid)} for child in process.children(recursive=True)]
    process.terminate()
    try:
        process.wait(timeout=5)
    except psutil.TimeoutExpired:
        process.kill()
    for identity in children:
        try:
            child = psutil.Process(identity["pid"])
            if matches_identity(identity, child.create_time()):
                child.terminate()
        except psutil.NoSuchProcess:
            pass
    manifest = runner.read(queue / "checkpoint_queue_manifest.json")
    runner.write(output / "queue_manifest_before.json", manifest)
    manifest["roots"] = list(dict.fromkeys([*manifest["roots"], *[c["root"] for c in cohort["children"]]]))
    manifest["source_journal_ids"] = list(
        dict.fromkeys([*manifest["source_journal_ids"], *[c["journal_id"] for c in cohort["children"]]])
    )
    runner.write(queue / "checkpoint_queue_manifest.json", manifest)
    (queue / "pause_dispatch.json").unlink()
    snapshot = Path(cohort["snapshot"])
    command = [
        str(snapshot / ".venv/bin/python"),
        str(snapshot / "scripts/utils/checkpoint_queue.py"),
        "--output",
        str(queue),
    ]
    with (output / "shared_queue_controller.log").open("ab") as log:
        p = subprocess.Popen(
            command,
            cwd=snapshot,
            env=runner.read(output / "private_environment.json"),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    runner.write(
        output / "queue_handoff_status.json",
        {"phase": "resumed", "at": runner.now(), "pid": p.pid, "command": command, **stable_identity(p.pid)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--service-owner", type=Path)
    parser.add_argument("--queue-output", type=Path)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "scripts/premise/policies/appworld_teammate_completion.txt"
    )
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if args.legacy_root:
        legacy(args.legacy_root.resolve())
    elif args.handoff:
        handoff(args.handoff.resolve())
    elif args.output and args.start:
        start(args.output.resolve())
    elif args.output and args.service_owner and args.queue_output:
        prepare(args.output.resolve(), args.service_owner.resolve(), args.queue_output.resolve(), args.policy.resolve())
    else:
        parser.error("Provide preparation inputs, --output --start, --legacy-root, or --handoff")


if __name__ == "__main__":
    main()
