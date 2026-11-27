# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Wait until the requested time, check Qwen generation, and resume cached CP130 probes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace.run_appworld_own_model_policy_shift import identity  # noqa: E402
from scripts.utils.evaluate_isolated_checkpoint_queue import read_json, timestamp, write_json  # noqa: E402

MODEL = "Qwen/Qwen3.6-35B-A3B"


def ready_route(output: Path, plan: dict) -> None:
    upstream, base_url = plan["upstream"], plan["base_url"]
    response = httpx.get(upstream + "/models", timeout=10)
    response.raise_for_status()
    if MODEL not in {row["id"] for row in response.json()["data"]}:
        raise RuntimeError("The requested Qwen model is not loaded yet")
    messages = [{"role": "system", "content": "Use tools to fulfill the user request."},
                {"role": "user", "content": "Call the ping tool with value 1."}]
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {
        "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}]
    tokens = []
    for choice in ("auto", "none"):
        response = httpx.post(upstream.removesuffix("/v1") + "/tokenize", json={
            "model": MODEL, "messages": messages, "tools": tools, "tool_choice": choice,
            "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}, timeout=15)
        response.raise_for_status()
        tokens.append(response.json()["tokens"])
    if tokens[0] != tokens[1]:
        raise RuntimeError("Restarted chat template differs for auto and none; parsing route needs review")
    try:
        response = httpx.get(base_url.removesuffix("/v1") + "/healthz", timeout=5)
    except httpx.ConnectError:
        command = ["/home/thskadud/.vllm/bin/python", "-u",
                   str(ROOT / "scripts/utils/serve_vllm_tool_parser_bridge.py"), "--port", "18018",
                   "--upstream", upstream.removesuffix("/v1"), "--tool-call-parser", "qwen3_coder",
                   "--read-timeout", "610"]
        with (output / "qwen_parser_bridge_after_reboot.log").open("ab", buffering=0) as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / "post_reboot_router.json", {"pid": process.pid, "command": command,
                                                      "started_at": timestamp()})
        raise RuntimeError("Started the client parsing bridge; waiting for its health check") from None
    response.raise_for_status()
    health = response.json()
    if (health.get("parser"), health.get("upstream")) != ("qwen3_coder", upstream.removesuffix("/v1")):
        raise RuntimeError("Port 18018 is not the expected Qwen parsing route")
    request = {"model": MODEL, "messages": messages, "tools": tools, "tool_choice": "auto",
               "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
    response = httpx.post(base_url + "/chat/completions", json=request, timeout=25)
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    if (payload.get("model") != MODEL or choice["finish_reason"] != "tool_calls"
            or not calls or calls[0]["function"]["name"] != "ping"
            or json.loads(calls[0]["function"]["arguments"]) != {"value": 1}
            or message.get("reasoning_content")):
        raise RuntimeError("Qwen generation/tool parsing/thinking check did not pass")
    write_json(output / "post_reboot_preflight.json", {"at": timestamp(), "request": request,
               "response": payload, "auto_none_prompt_tokens_identical": True, "token_count": len(tokens[0])})


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    plan = read_json(output / "delayed_resume_plan.json")
    deadline = datetime.fromisoformat(plan["not_before"]).timestamp()
    with (output / "delayed_resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(phase: str, **extra):
            data = {"phase": phase, "scheduler_pid": os.getpid(), "not_before": plan["not_before"],
                    "checkpoint": 130, "workers": plan["workers"], "updated_at": timestamp(), **extra}
            write_json(output / "delayed_resume_status.json", data)
            if phase in ("scheduled_resume", "waiting_for_qwen"):
                write_json(output / "status.json", data)

        while time.time() < deadline:
            status("scheduled_resume", remaining_seconds=round(deadline - time.time(), 1))
            time.sleep(min(30, max(0, deadline - time.time())))
        while True:
            status("waiting_for_qwen")
            try:
                ready_route(output, plan)
                break
            except (httpx.HTTPError, RuntimeError, ValueError, KeyError, IndexError) as exc:
                status("waiting_for_qwen", last_error=f"{type(exc).__name__}: {exc}")
                print(f"Waiting for Qwen: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(15)
        current_launch = read_json(output / "controller_launch.json")
        previous = identity(current_launch["pid"])
        if previous is not None:
            if previous["argv"] != current_launch["command"]:
                raise RuntimeError("Saved controller PID belongs to a different process")
            status("existing_controller", controller_pid=previous["pid"])
            return
        command = [sys.executable, "-u", str(ROOT / "scripts/premise/run_bfcl_root_final_policy_shift.py"),
                   "--checkpoint", "130", "--output", str(output), "--base-url", plan["base_url"],
                   "--workers", str(plan["workers"]), "--reuse-from", plan["cache_source"]]
        with (output / "controller_after_reboot.log").open("ab", buffering=0) as stream:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        write_json(output / "controller_launch.json", {"pid": process.pid, "command": command,
                   "started_at": timestamp(), "checkpoint": 130, "resumed_after_reboot": True})
        status("controller_running", controller_pid=process.pid)
        print(f"Resumed CP130 controller {process.pid} after generation preflight", flush=True)
        code = process.wait()
        status("controller_finished" if code == 0 else "controller_error",
               controller_pid=process.pid, return_code=code)


if __name__ == "__main__":
    main()
