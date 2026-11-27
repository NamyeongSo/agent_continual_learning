# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Resume a frozen ACE/PREMiSE teammate batch while deferring checkpoint/test inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils import run_qwen_isolated_when_free as runner  # noqa: E402
from scripts.utils.schedule_qwen_pause import identify  # noqa: E402


def validate(plan):
    if (len(plan["jobs"]) != 6 or {(j["method"], j["seed"]) for j in plan["jobs"]}
            != {(m, s) for m in ("ace", "premise") for s in (42, 43, 44)}):
        raise ValueError("Expected ACE/PREMiSE seeds42-44 only")
    if not plan.get("evaluation_deferred") or plan.get("policy_name") != "teammate":
        raise ValueError("This controller requires teammate policy and deferred inference")
    if runner.digest(Path(plan["appworld_policy"])) != plan["appworld_policy_sha256"]:
        raise ValueError("Frozen teammate policy bytes changed")
    if runner.digest(Path(plan["manifest"])) != plan["manifest_sha256"]:
        raise ValueError("Frozen task split changed")
    for job in plan["jobs"]:
        if job["benchmark"] != "appworld" or job["appworld_policy_sha256"] != plan["appworld_policy_sha256"]:
            raise ValueError("A job has a different benchmark or environment policy")
        command = job["command"]
        if command[command.index("--temperature") + 1] != "0.7":
            raise ValueError("All roles must retain temperature0.7")
        if job["method"] == "premise" and command[command.index("--base-url") + 1] != plan["base_url"]:
            raise ValueError("Actor and analysis endpoints differ")
        if any(flag in command for flag in ("--evaluate-test", "--evaluate-val", "--evaluation-only")):
            raise ValueError("Training commands must not launch extra inference")


def run(root, resume_output):
    plan = runner.read(root / "launch_manifest.json")
    validate(plan)
    service = runner.read(resume_output / "services.json")
    processes = []
    for key in ("upstream_process", "bridge_process"):
        meta = service[key]
        current = identify(meta["pid"])
        if current is None or current["command"] != meta["command"]:
            raise ValueError("Validated endpoint process changed identity")
        processes.append(runner.ExistingProcess(meta["pid"], command=meta["command"]))
    models = runner.request(service["upstream_url"] + "/v1/models")["data"]
    if runner.RAW_MODEL not in [model["id"] for model in models]:
        raise ValueError("The requested Qwen model is unavailable")
    health = runner.request(plan["base_url"].removesuffix("/v1") + "/healthz")
    if health.get("parser") != "qwen3_coder" or health.get("upstream") != service["upstream_url"]:
        raise ValueError("The tool parser is not connected to the requested endpoint")
    with FileLock(str(root / "controller.lock"), timeout=0):
        runner.write(root / "controller_process.json", {**identify(os.getpid()), "started_at": runner.now()})
        runner.write(root / "status.json", {"phase": "running", "at": runner.now(),
                                            "evaluation_deferred": True, "resume_output": str(resume_output)})
        runner.journal(plan, root, "running", "팀원 AppWorld ACE/PREMiSE seeds42·43·44 학습 재개. "
                       "19006 upstream 사용; CP 및 최종 test 추론 보류, 기존 체크포인트·내부 병렬성 유지.")
        try:
            jobs = runner.run_training_and_evaluation(plan, root, *processes, resume=True, defer_evaluation=True)
            failed = [j["id"] for j in jobs if j["status"] not in {"completed", "test_pending"}]
            phase = "blocked" if failed else "training_complete_evaluation_deferred"
            runner.write(root / "status.json", {"phase": phase, "at": runner.now(), "failed_jobs": failed,
                                                "evaluation_deferred": True})
            runner.journal(plan, root, "blocked",
                           f"학습 단계 종료. 미해결 학습 {failed}; CP·최종 test는 사용자 요청으로 보류.")
        except Exception as exc:
            runner.write(root / "status.json", {"phase": "blocked", "at": runner.now(),
                                                "error": f"{type(exc).__name__}: {exc}"})
            runner.journal(plan, root, "blocked", f"학습 실행기 중단: {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume-output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.resume_output.resolve())
