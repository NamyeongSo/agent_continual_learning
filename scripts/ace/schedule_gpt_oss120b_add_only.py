# SPDX-License-Identifier: Apache-2.0
"""Replicate the GPT-OSS-20B ACE study after its complete GPU handoff."""

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

from filelock import FileLock

from scripts.ace import gpt_oss_recovery
from scripts.ace import schedule_gpt_oss20b_add_only as oss20

base, qwen, study = oss20.base, oss20.qwen, oss20.study
MODEL = "openai/gpt-oss-120b"
PREDECESSOR = study.ROOT / "outputs/ace_appworld_gpt_oss20b_add_only_20261120"
HF_CACHE = Path.home() / ".cache/huggingface/hub"
HF = Path.home() / ".local/bin/hf"
WORKERS = 100
REFLECTION_INPUTS = (
    "appworld_policy.txt",
    "initial_playbook.txt",
    "model_settings.json",
    "references.json",
    "task_splits.json",
    "benchmark_dependency_hashes.json",
    "stage1_queue.json",
    "stage2_queue.json",
    "stage3_queue.json",
)


def prepare(output, revision):
    if not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise ValueError("Prepare with the pinned Hugging Face revision")
    prior_plan = base.read(PREDECESSOR / "plan.json")
    if prior_plan["model"] != oss20.MODEL or tuple(prior_plan["modes"]) != oss20.MODES:
        raise ValueError("Unexpected GPT-OSS-20B predecessor")
    serving = output / "serving"
    serving.mkdir(exist_ok=True)
    prior_script = PREDECESSOR / "serving/serve.sh"
    script = prior_script.read_bytes()
    if base.digest(script) != prior_plan["serving_sha256"]["serve.sh"] or script.count(oss20.MODEL.encode()) != 2:
        raise ValueError("Unexpected predecessor serving script")
    script = script.replace(oss20.MODEL.encode(), MODEL.encode())
    script, replacements = re.subn(rb"(?m)(^  --max-num-seqs )\d+",
                                   lambda match: match[1] + str(WORKERS).encode(), script)
    if replacements != 1:
        raise ValueError("Expected one vLLM sequence concurrency setting")
    (serving / "serve.sh").write_bytes(script)
    subprocess.run(["bash", "-n", str(serving / "serve.sh")], check=True)
    (serving / "model_revision.txt").write_text(revision + "\n")

    protocol_hashes = {}
    for mode in oss20.MODES:
        reflection = output / mode / "reflection"
        for phase in ("reflection", "accumulation"):
            relative = Path(mode) / phase
            source, target = PREDECESSOR / relative, output / relative
            study.verify_sources(source)
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            for name in ("source_snapshot", "benchmark_dependencies"):
                shutil.copytree(source / name, target / name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copy2(source / "source_hashes.json", target / "source_hashes.json")
            if phase == "reflection":
                for name in REFLECTION_INPUTS:
                    shutil.copy2(source / name, target / name)
            protocol = base.read(source / "protocol.json")
            if protocol["model"] != f"openai/{oss20.MODEL}" or protocol["model_settings"]["reasoning_effort"] != mode:
                raise ValueError(f"Unexpected predecessor protocol for {relative}")
            protocol.update(
                model=f"openai/{MODEL}", model_revision=revision, prepared_at=base.now(),
                benchmark_dependency_overlay=str(target / "benchmark_dependencies"),
                model_settings_source=str(reflection / "model_settings.json"),
                matched_20b_protocol_sha256=base.digest((source / "protocol.json").read_bytes()),
            )
            status = {"state": "scheduled", "updated_at": base.now()}
            if phase == "accumulation":
                protocol.update(source_reflection_run=str(reflection),
                                source_protocol_sha256=base.digest((reflection / "protocol.json").read_bytes()),
                                global_concurrency=WORKERS, evaluation_workers=WORKERS)
                status["phase"] = "waiting_for_reflection_experiment"
            else:
                for stage in protocol["stages"]:
                    stage["max_workers"] = WORKERS
                status.update(phase="waiting_for_gpt_oss120b_pipeline", stage=0)
            base.write(target / "status.json", status)
            base.write(target / "protocol.json", protocol)
            study.verify_sources(target)
            protocol_hashes[str(relative / "protocol.json")] = base.digest((target / "protocol.json").read_bytes())

    base.write(output / "plan.json", {
        **prior_plan, "model": MODEL, "model_revision": revision, "predecessor": str(PREDECESSOR),
        "reflection_concurrency": WORKERS, "evaluation_concurrency": WORKERS,
        "protocol_sha256": protocol_hashes,
        "serving_sha256": {name: base.digest((serving / name).read_bytes())
                           for name in ("serve.sh", "model_revision.txt")},
    })


def checkpoint_ready(revision):
    checkpoint = HF_CACHE / "models--openai--gpt-oss-120b" / "snapshots" / revision
    index = checkpoint / "model.safetensors.index.json"
    if not index.is_file():
        return False
    shards = set(base.read(index)["weight_map"].values())
    required = shards | {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    return bool(shards) and all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
                                for name in required)


def ensure_checkpoint(output, revision):
    serving = output / "serving"
    download = serving / "download_process.json"
    while not checkpoint_ready(revision) and download.exists() and qwen.pid_alive(base.read(download)["pid"]):
        qwen.publish(output / "status.json", "scheduled", phase="waiting_for_checkpoint_download")
        time.sleep(30)
    if checkpoint_ready(revision):
        return
    command = [str(HF), "download", MODEL, "--cache-dir", str(HF_CACHE),
               "--revision", revision, "--exclude", "original/*", "--exclude", "metal/*"]
    qwen.publish(output / "status.json", "scheduled", phase="retrying_checkpoint_download")
    with (serving / "download.log").open("a") as log:
        for attempt in range(3):
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0 and checkpoint_ready(revision):
                return
            if attempt < 2:
                time.sleep(15)
    raise RuntimeError("GPT-OSS-120B checkpoint download failed. See serving/download.log")


def run(output):
    status = output / "status.json"
    previous = base.read(status) if status.exists() else {}
    plan = base.read(output / "plan.json")
    predecessor = Path(plan["predecessor"])
    serving = output / "serving"
    for directory, hashes in ((output, plan["protocol_sha256"]), (serving, plan["serving_sha256"])):
        for name, expected in hashes.items():
            if base.digest((directory / name).read_bytes()) != expected:
                raise ValueError(f"Prepared file changed: {directory / name}")
    while not oss20.predecessor_complete(predecessor, oss20.MODES):
        prior = base.read(predecessor / "status.json")
        qwen.publish(status, "scheduled", phase="waiting_for_gpt_oss20b_completion",
                     predecessor_state=prior["state"], predecessor_phase=prior["phase"],
                     checkpoint_ready=checkpoint_ready(plan["model_revision"]))
        time.sleep(30)
    ensure_checkpoint(output, plan["model_revision"])
    qwen.publish(status, "running", phase="starting_gpt_oss120b")
    endpoint = oss20.start_server(output, model=MODEL, predecessor=predecessor,
                                  predecessor_model=oss20.MODEL, predecessor_mode="high")
    base.write(serving / "endpoint_verification.json", endpoint)
    oss20.check_generation(output)
    handoff = serving / "status.json"
    qwen.publish(handoff, "ready", model=MODEL, server_pid=endpoint["server"]["pid"], port=8006, physical_gpu=1)
    gpt_oss_recovery.run_study(output, model=MODEL, handoff_path=handoff, modes=oss20.MODES,
                              previous_failure=previous if previous.get("state") == "error" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        status = output / "status.json"
        try:
            if not (output / "plan.json").exists():
                prepare(output, args.revision)
            if not args.prepare_only and (not status.exists() or base.read(status)["state"] != "completed"):
                run(output)
        except Exception as exc:
            phase = base.read(status).get("phase", "prepare") if status.exists() else "prepare"
            qwen.publish(status, "error", phase=phase, error=str(exc))
            study.accumulation.journal(output, "blocked", f"GPT-OSS-120B 실험 중단 ({phase}) — {exc}")
            raise


if __name__ == "__main__":
    main()
