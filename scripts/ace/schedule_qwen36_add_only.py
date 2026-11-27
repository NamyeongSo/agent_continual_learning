# SPDX-License-Identifier: Apache-2.0
"""Schedule the matched Qwen3.6 FP8 ACE studies behind the server handoff."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from scripts.ace import evaluate_advantage_accumulation as evaluator
from scripts.ace import run_advantage_accumulation as advantage
from scripts.ace import run_reflection_accumulation as accumulation
from scripts.ace.analyze_context_proposal_bottleneck import analyze as analyze_reflections
from scripts.ace.analyze_reflection_accumulation import analyze as analyze_accumulation
from scripts.ace.compare_reflection_accumulation import compare
from scripts.ace.plot_reflection_group_advantages import analyze as analyze_group_advantages
from scripts.ace.plot_worst_reflection_repeats import ROOT, RUNS, SEEDS
from scripts.ace.plot_worst_reflection_repeats import analyze as analyze_worst
from scripts.ace.schedule_nonthinking_replication import freeze_reflection, verify_sources

base = accumulation.base
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
HANDOFF = ROOT / "outputs/ace_appworld_advantage_add_20261118/qwen36_handoff/status.json"


def prepare(output, *, model=MODEL, handoff=HANDOFF):
    for mode, prior in RUNS.items():
        source = Path(base.read(prior / "protocol.json")["source_reflection_run"])
        reflection = output / mode / "reflection"
        reflection.parent.mkdir(exist_ok=True)
        freeze_reflection(source, reflection)
        protocol = base.read(reflection / "protocol.json")
        settings = protocol["model_settings"]
        thinking = mode == "thinking"
        settings.update(temperature=1.0 if thinking else 0.7, top_p=0.95 if thinking else 0.80, presence_penalty=1.5)
        settings["extra_body"]["chat_template_kwargs"] = {
            "enable_thinking": thinking,
            "preserve_thinking": thinking,
        }
        settings_path = reflection / "model_settings.json"
        base.write(settings_path, settings)
        protocol.update(
            model=f"openai/{model}",
            generation_mode="thinking" if thinking else "non_thinking",
            generation_settings_authority=f"User-specified {model} samplers",
            model_settings_source_sha256=base.digest(settings_path.read_bytes()),
        )
        base.write(reflection / "protocol.json", protocol)
        accumulation.prepare(output / mode / "accumulation", reflection)
    base.write(
        output / "plan.json",
        {
            "model": model,
            "dependency": str(handoff),
            "modes": list(RUNS),
            "steps": [
                "reflection",
                "full_train_accumulation",
                "advantage_sign_accumulation",
                "advantage_sign_evaluation",
            ],
            "final_contexts": 60,
        },
    )


def run(output, *, model=MODEL, handoff_path=HANDOFF, modes=tuple(RUNS)):
    def emit(state, phase, **extra):
        base.write(
            output / "status.json",
            {"state": state, "phase": phase, "updated_at": base.now(), "pid": os.getpid(), **extra},
        )
        print(f"{base.now()} {state} {phase} {extra}", flush=True)

    def execute(mode, phase, filename):
        target = output / mode / phase
        verify_sources(target)
        protocol = base.read(target / "protocol.json")
        if protocol["model"] != f"openai/{model}":
            raise ValueError("Wrong model in frozen protocol")
        endpoint_name = "endpoint_qwen36.json" if model == MODEL else "endpoint_verification_at_start.json"
        base.write(target / endpoint_name, base.check_endpoint(protocol))
        if base.read(target / "status.json")["state"] == "completed":
            return
        command = [
            sys.executable,
            str(target / "source_snapshot/scripts/ace" / filename),
            "--run",
            "--output",
            str(target),
        ]
        with (target / "scheduler.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            emit("running", f"{mode}/{phase}", child_pid=child.pid)
            result = child.wait()
        if result or base.read(target / "status.json")["state"] != "completed":
            raise RuntimeError(f"Incomplete {mode}/{phase} (exit {result})")

    while True:
        handoff = base.read(handoff_path)
        if handoff["state"] == "ready" and handoff["model"] == model:
            break
        if handoff["state"] == "error":
            raise RuntimeError(f"{model} handoff failed: {handoff}")
        emit("scheduled", "waiting_for_qwen36", handoff_state=handoff["state"])
        time.sleep(30)
    accumulation.journal(output, "running", f"{model} 서버 준비 완료. ACE ADD-only 실험을 시작합니다.")
    for mode in modes:
        execute(mode, "reflection", "run_context_proposal_bottleneck.py")
        reflection = output / mode / "reflection"
        analyze_reflections(reflection, reflection / "analysis" / "reflection_final")
    runs = {mode: output / mode / "accumulation" for mode in modes}
    for mode, target in runs.items():
        execute(mode, "accumulation", "run_reflection_accumulation.py")
        for seed in SEEDS:
            analyze_accumulation(target, seed)
            compare(target, seed)
    analyze_worst(output / "analysis" / "worst_reflection_repeats", runs)
    analyze_group_advantages(output / "analysis" / "reflection_group_advantages", runs)
    target = output / "advantage"
    plan_path = target / "plan.json"
    jobs = base.read(plan_path) if plan_path.exists() else advantage.prepare(target, runs)
    training_status = target / "status.json"
    if not training_status.exists() or base.read(training_status)["state"] != "completed":
        emit("running", "advantage_training", contexts=len(jobs), steps=sum(j["steps"] for j in jobs))
        advantage.run(target, jobs)
    emit("running", "advantage_evaluation")
    evaluator.run(target, runs)
    emit("completed", "all_complete")
    accumulation.journal(output, "completed", "두 모드의 Reflection 평가와 ADD-only 누적 Context 60개 평가 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        if not (output / "plan.json").exists():
            prepare(output)
        if not args.prepare_only:
            try:
                run(output)
            except Exception as exc:
                status_path = output / "status.json"
                previous = base.read(status_path) if status_path.exists() else {}
                base.write(status_path, {**previous, "state": "error", "error": str(exc), "updated_at": base.now()})
                raise
