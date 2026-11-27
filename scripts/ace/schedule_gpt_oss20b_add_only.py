# SPDX-License-Identifier: Apache-2.0
"""Queue the ACE ADD-only study for GPT-OSS high and low after Qwen3.5."""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

from filelock import FileLock

from scripts.ace import gpt_oss_recovery
from scripts.ace import schedule_qwen35_9b_add_only as qwen
from scripts.ace import schedule_qwen36_add_only as study

base = study.base
MODEL = "openai/gpt-oss-20b"
MODES = ("high", "low")
PREDECESSOR = study.ROOT / "outputs/ace_appworld_qwen35_9b_add_only_20261119"
CACHE = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b"
WORKERS = 200


def prepare(output):
    serving = output / "serving"
    serving.mkdir(exist_ok=True)
    revision = (CACHE / "refs/main").read_text().strip()
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Invalid cached GPT-OSS revision")
    checkpoint = CACHE / "snapshots" / revision
    shards = set(base.read(checkpoint / "model.safetensors.index.json")["weight_map"].values())
    if not shards or not all((checkpoint / filename).is_file() for filename in shards | {"tokenizer.json"}):
        raise ValueError("Pinned GPT-OSS checkpoint is incomplete")
    shutil.copy2(Path(__file__).with_name("serve_gpt_oss20b.sh"), serving / "serve.sh")
    (serving / "model_revision.txt").write_text(revision + "\n")

    source = PREDECESSOR / "thinking/reflection"
    for effort in MODES:
        (output / effort).mkdir(exist_ok=True)
        reflection = output / effort / "reflection"
        study.freeze_reflection(source, reflection)
        protocol = base.read(reflection / "protocol.json")
        settings = {
            "temperature": 1.0,
            "top_p": 1.0,
            "reasoning_effort": effort,
            **{k: protocol["model_settings"][k] for k in ("max_tokens", "timeout", "num_retries")},
        }
        settings_path = reflection / "model_settings.json"
        base.write(settings_path, settings)
        protocol.update(
            model=f"openai/{MODEL}", model_revision=revision, generation_mode="reasoning_effort",
            generation_settings_authority="User-specified GPT-OSS high/low, temperature=1.0, top_p=1.0",
            model_settings=settings, model_settings_source_sha256=base.digest(settings_path.read_bytes()),
            playbook_edit_mode="add_only",
        )
        for stage in protocol["stages"]:
            stage["max_workers"] = WORKERS
        base.write(reflection / "protocol.json", protocol)

        accumulation = output / effort / "accumulation"
        study.accumulation.prepare(accumulation, reflection, global_concurrency=WORKERS)
        protocol_path = accumulation / "protocol.json"
        base.write(protocol_path, {**base.read(protocol_path), "evaluation_workers": WORKERS})
    base.write(output / "plan.json", {
        "model": MODEL, "model_revision": revision, "predecessor": str(PREDECESSOR),
        "modes": MODES, "temperature": 1.0, "top_p": 1.0, "gpu_physical_id": 1, "port": 8006,
        "reflection_concurrency": WORKERS, "evaluation_concurrency": WORKERS,
        "final_contexts": 60, "full_train_contexts": 36, "advantage_contexts": 24,
        "steps": ["reflection", "full_train_accumulation", "advantage_sign_accumulation", "advantage_sign_evaluation"],
        "serving_sha256": {name: base.digest((serving / name).read_bytes())
                           for name in ("serve.sh", "model_revision.txt")},
    })


def predecessor_complete(predecessor=PREDECESSOR, modes=("thinking", "nonthinking")):
    status = base.read(predecessor / "status.json")
    if (status["state"], status["phase"]) != ("completed", "all_complete"):
        return False
    for mode in modes:
        if base.read(predecessor / mode / "reflection/status.json")["state"] != "completed":
            raise ValueError(f"Incomplete {predecessor.name}/{mode} reflection results")
        for seed in study.SEEDS:
            target = predecessor / mode / "accumulation" / f"seed{seed}"
            if base.read(target / "completed.json")["evaluations"] != 900:
                raise ValueError(f"Incomplete {predecessor.name}/{mode}/seed{seed} accumulation evaluations")
    advantage = predecessor / "advantage"
    learners = base.read(advantage / "plan.json")
    totals = study.evaluator.phase_totals(learners)
    evaluations = advantage / "evaluations"
    if len(learners) != 24 or len(base.read(evaluations / "summary.json")) != 24:
        raise ValueError(f"Incomplete {predecessor.name} Advantage summary")
    for phase, count in totals.items():
        if len(base.read(evaluations / f"{phase}_results.json")) != count:
            raise ValueError(f"Incomplete {predecessor.name} Advantage {phase} results")
    return True


def start_server(output, *, model=MODEL, predecessor=PREDECESSOR,
                 predecessor_model=qwen.MODEL, predecessor_mode="thinking"):
    serving = output / "serving"
    handoff = serving / "status.json"
    protocol = base.read(output / "high/reflection/protocol.json")
    if handoff.exists():
        saved = base.read(handoff)
        if saved["model"] == model and qwen.pid_alive(saved["server_pid"]):
            return qwen.ready(protocol, saved["server_pid"])

    old = base.read(predecessor / "serving/status.json")
    pid = old["server_pid"]
    if qwen.pid_alive(pid):
        endpoint = base.check_endpoint(base.read(predecessor / predecessor_mode / "reflection/protocol.json"))
        if old["model"] != predecessor_model or endpoint["server"]["pid"] != pid:
            raise ValueError(f"The running server is not the verified {predecessor_model} predecessor")
        os.kill(pid, signal.SIGTERM)
        for _ in range(90):
            if not qwen.pid_alive(pid):
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"{predecessor_model} server did not stop after handoff")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8006)) == 0:
            raise RuntimeError("Port 8006 is occupied by a different server")
    with (serving / "vllm.log").open("a") as log:
        child = subprocess.Popen(["bash", str(serving / "serve.sh")], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    qwen.publish(handoff, "starting", model=model, server_pid=child.pid, port=8006, physical_gpu=1)
    return qwen.ready(protocol, child.pid)


def check_generation(output):
    import litellm

    checks = {}
    for effort in MODES:
        protocol = base.read(output / effort / "reflection/protocol.json")
        sampler = {k: protocol["model_settings"][k] for k in ("temperature", "top_p", "reasoning_effort")}
        kwargs = {
            "model": protocol["model"], "api_base": protocol["base_url"], "api_key": "EMPTY",
            "allowed_openai_params": ["reasoning_effort"], "timeout": 120, "max_tokens": 4096, **sampler,
        }
        json_request = {
            "messages": [{"role": "user", "content": 'Return the JSON object {"ok": true}.'}],
            "response_format": {"type": "json_object"},
        }
        tool_request = {
            "messages": [{"role": "user", "content": "Call the report function with the value hello."}],
            "tool_choice": "auto",
            "tools": [{"type": "function", "function": {
                "name": "report", "description": "Report the requested value.",
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                               "required": ["value"], "additionalProperties": False},
            }}],
        }
        for _ in range(3):
            content = litellm.completion(**kwargs, **json_request).choices[0].message.content
            tools = litellm.completion(**kwargs, **tool_request).choices[0].message.tool_calls
            try:
                if json.loads(content) == {"ok": True} and tools and tools[0].function.name == "report":
                    if json.loads(tools[0].function.arguments) == {"value": "hello"}:
                        checks[effort] = {**sampler, "structured_json": True, "auto_tool_call": True}
                        break
            except (TypeError, ValueError):
                continue
        else:
            raise RuntimeError(f"GPT-OSS {effort} generation check failed")
    base.write(output / "serving/generation_canaries.json", checks)


def run(output):
    status = output / "status.json"
    previous = base.read(status) if status.exists() else {}
    plan = base.read(output / "plan.json")
    serving = output / "serving"
    for name, sha256 in plan["serving_sha256"].items():
        if base.digest((serving / name).read_bytes()) != sha256:
            raise ValueError(f"Frozen serving file changed: {name}")
    while not predecessor_complete():
        prior = base.read(PREDECESSOR / "status.json")
        qwen.publish(status, "scheduled", phase="waiting_for_qwen35_completion",
                     predecessor_state=prior["state"], predecessor_phase=prior["phase"])
        time.sleep(30)
    qwen.publish(status, "running", phase="starting_gpt_oss")
    endpoint = start_server(output)
    base.write(serving / "endpoint_verification.json", endpoint)
    check_generation(output)
    handoff = serving / "status.json"
    qwen.publish(handoff, "ready", model=MODEL, server_pid=endpoint["server"]["pid"], port=8006, physical_gpu=1)
    gpt_oss_recovery.run_study(output, model=MODEL, handoff_path=handoff, modes=MODES,
                              previous_failure=previous if previous.get("state") == "error" else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-active", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output / "scheduler.lock"))
    with lock.acquire(timeout=-1 if args.wait_for_active else 0, poll_interval=1):
        status = output / "status.json"
        try:
            if not (output / "plan.json").exists():
                prepare(output)
            if not args.prepare_only and (not status.exists() or base.read(status)["state"] != "completed"):
                run(output)
        except Exception as exc:
            phase = base.read(status).get("phase", "prepare") if status.exists() else "prepare"
            qwen.publish(status, "error", phase=phase, error=str(exc))
            study.accumulation.journal(output, "blocked", f"GPT-OSS 실험 중단 ({phase}) — {exc}")
            raise
