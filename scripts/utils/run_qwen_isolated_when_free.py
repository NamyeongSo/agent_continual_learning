# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Wait for a free GPU, start the supplied Qwen server, and run frozen isolated jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.utils.process_identity import current_boot_id, process_start_ticks  # noqa: E402

MODEL = "openai/Qwen/Qwen3.6-35B-A3B"
RAW_MODEL = MODEL.removeprefix("openai/")
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
SUBSETS = {"bfcl": "multi_turn_base", "appworld": "test_challenge", "terminalbench2": "2.0"}
TEMPERATURE = 0.7
TRAIN_WORKERS = 6
EVAL_WORKERS = 100
BASE_URL = "http://127.0.0.1:18006/v1"


def experiment_environment(environ=None, *, base_url=BASE_URL):
    """Give resumed actors the local endpoint and Docker CLI without relying on the invoking shell."""
    env = dict(os.environ if environ is None else environ)
    env["OPENAI_API_BASE"] = base_url
    env["OPENAI_BASE_URL"] = base_url
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    docker_bin = str(ROOT / ".venv/bin")
    previous_path = env.get("PATH", os.defpath).split(os.pathsep)
    env["PATH"] = os.pathsep.join([docker_bin, *[p for p in previous_path if p != docker_bin]])
    return env


def configure_experiment_environment():
    os.environ.update(experiment_environment())
    if not shutil.which("docker"):
        raise RuntimeError("Qwen experiments require the project Docker CLI on PATH before training starts")


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class ExistingProcess:
    """Monitor a recorded process without restarting it or signalling its children."""

    def __init__(self, pid, *, command=None, completion=None):
        import psutil

        self.pid = pid
        self.returncode = None
        self.completion = completion
        self.identity = None
        self.start_ticks = None
        self.boot_id = None
        try:
            process = psutil.Process(pid)
            if process.status() != psutil.STATUS_ZOMBIE:
                if command is not None and process.cmdline() != command:
                    raise ValueError(f"Recorded PID {pid} belongs to a different command")
                self.identity = process.create_time()
                self.start_ticks = process_start_ticks(pid)
                self.boot_id = current_boot_id()
        except psutil.NoSuchProcess:
            pass

    def poll(self):
        import psutil

        if self.returncode is not None:
            return self.returncode
        try:
            process = psutil.Process(self.pid)
            same = (
                process_start_ticks(self.pid) == self.start_ticks and current_boot_id() == self.boot_id
                if self.start_ticks is not None else process.create_time() == self.identity
            )
            if same and process.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            pass
        # A non-child process has no recoverable exit status. Require its explicit
        # completion marker; frozen_evaluation_jobs validates the count/checkpoint.
        self.returncode = 0 if self.completion is not None and self.completion.exists() else 1
        return self.returncode


def resume_jobs(plan, output):
    progress = read(output / "progress.json")
    recorded = {job["id"]: job for job in progress["jobs"]}
    if len(recorded) != len(plan["jobs"]) or set(recorded) != {job["id"] for job in plan["jobs"]}:
        raise ValueError("Resume job IDs differ from launch manifest")
    for job in plan["jobs"]:
        previous = recorded[job["id"]]
        for field in ("method", "benchmark", "seed", "command", "train_dir", "test_ids", "train_count", "output"):
            if previous[field] != job[field]:
                raise ValueError(f"Resume {job['id']} {field} differs from launch manifest")
        if previous.get("playbook_edit_mode", "add_only") != job.get("playbook_edit_mode", "add_only"):
            raise ValueError("Resume playbook edit mode differs from launch manifest")
    return progress["jobs"]


def adopt_training(job):
    import psutil

    process = ExistingProcess(job["pid"], command=job["command"],
                              completion=Path(job["train_dir"]) / "training_complete.json")
    slot = job.get("slot", 0)
    if process.poll() is None:
        port = int(psutil.Process(process.pid).environ()["EXGENTIC_PORT_BLOCK_START"])
        slot, remainder = divmod(port - job.get("port_block_base", 27000), 1000)
        if slot < 0 or remainder:
            raise ValueError("Existing training worker has an invalid port block")
    log = (Path(job["output"]) / "training.log").open("ab")
    job["slot"] = slot
    return process, job, slot, log


def existing_services(output):
    server_record = read(output / "server_process.json")
    bridge_record = read(output / "bridge_process.json")
    server = ExistingProcess(server_record["pid"], command=server_record.get("command"))
    bridge = ExistingProcess(bridge_record["pid"], command=bridge_record.get("command"))
    if server.poll() is not None or bridge.poll() is not None:
        raise RuntimeError("Recorded vLLM server or parser bridge is no longer running")
    if not owns_port(server, 8006) or not owns_port(bridge, 18006):
        raise RuntimeError("Recorded processes do not own the Qwen server/bridge ports")
    if RAW_MODEL not in [row["id"] for row in request("http://127.0.0.1:8006/v1/models")["data"]]:
        raise RuntimeError("Recorded Qwen model is not served")
    if request("http://127.0.0.1:18006/healthz")["parser"] != "qwen3_coder":
        raise RuntimeError("Recorded Qwen parser bridge is not healthy")
    return server, bridge


def training_workers(plan, output):
    """Read scheduling capacity each cycle so later changes need no restart."""
    settings = output / "runtime_schedule.json"
    workers = read(settings).get("train_workers") if settings.exists() else plan.get("train_workers", TRAIN_WORKERS)
    if type(workers) is not int or not 1 <= workers <= 18:
        raise ValueError("train_workers must be an integer between 1 and 18")
    return workers


def next_training_job(plan, jobs, running):
    limits = plan.get("train_workers_per_method", {})
    if any(method not in {"ace", "premise"} or type(cap) is not int or cap < 1 for method, cap in limits.items()):
        raise ValueError("Method training limits must be positive integers for ACE/PREMiSE")
    counts = {method: sum(item[1]["method"] == method for item in running.values()) for method in limits}
    return next((job for job in jobs if job["status"] == "pending"
                 and (job["method"] not in limits or counts[job["method"]] < limits[job["method"]])), None)


def training_port_base(plan, workers):
    base = plan.get("port_block_base", 27000)
    if type(base) is not int or base < 1024 or base + workers * 1000 > 65536:
        raise ValueError("Training port blocks must fit below port 65536")
    return base


def request(url, payload=None, timeout=20):
    query = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(query, timeout=timeout) as response:
        return json.load(response)


def gpu_state(index):
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                       text=True, timeout=15)

    values = query([f"--id={index}", "--query-gpu=uuid,memory.used,utilization.gpu"]).strip().split(",")
    uuid, memory, utilization = (value.strip() for value in values)
    processes = []
    for line in query(["--query-compute-apps=gpu_uuid,pid,used_memory"]).splitlines():
        fields = [item.strip() for item in line.split(",")]
        if fields[0] == uuid:
            processes.append({"pid": int(fields[1]), "memory_mib": fields[2]})
    return {"index": index, "uuid": uuid, "memory_mib": int(memory),
            "utilization": int(utilization), "processes": processes}


def gpu_is_free(state):
    return not state["processes"] and state["memory_mib"] <= 512 and state["utilization"] == 0


def port_free(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def owns_port(process, port):
    import psutil

    for connection in psutil.net_connections(kind="tcp"):
        if connection.status != "LISTEN" or not connection.laddr or connection.laddr.port != port:
            continue
        if connection.pid is not None:
            # Adopted services can record the listening process itself, while
            # services launched by this runner record the session leader. Accept
            # either exact ownership form; comparing only the PGID rejects an
            # adopted vLLM child even when its PID is the socket owner.
            if connection.pid == process.pid:
                return True
            try:
                if os.getpgid(connection.pid) == process.pid:
                    return True
            except ProcessLookupError:
                pass
    return False


def train_command(job, code, manifest):
    common = ["--mode", "isolated", "--seed", str(job["seed"]), "--benchmarks", job["benchmark"],
              "--task-split-manifest", str(manifest), "--all-train-tasks", "--model", MODEL,
              "--temperature", str(TEMPERATURE), "--output-dir", job["train_dir"]]
    if job.get("playbook_edit_mode", "add_only") != "add_only":
        common.extend(["--playbook-edit-mode", job["playbook_edit_mode"]])
    python = str(code / ".venv/bin/python")
    if job["method"] == "ace":
        return [python, "-u", str(code / "scripts/ace/run_experiment.py"), *common,
                "--resume-sessions", "--max-tokens", "8192", "--disable-thinking", "--training-time",
                "--bfcl-max-turns", "30", "--appworld-max-turns", "40",
                "--terminalbench2-max-turns", "40", "--terminalbench2-actor", "terminus2"]
    return [python, "-u", str(code / "scripts/premise/run_experiment.py"), *common,
            "--base-url", BASE_URL, "--past", "3", "--parent-workers", "2", "--reflection-workers", "4",
            "--mutation-workers", "4", "--replay-workers", "28", "--llm-calls", "32",
            "--skip-task-errors", "nontransient", "--unlimited-benchmark-sessions"]


def build_plan(output, code, manifest, *, playbook_edit_mode="add_only"):
    from exgentic.agents.ace.playbook_utils import allowed_playbook_operations
    from exgentic.core.experiment_limits import experiment_limits
    from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

    allowed_playbook_operations(playbook_edit_mode)
    data = read(manifest)
    jobs = []
    for method in ("ace", "premise"):
        for benchmark in BENCHMARKS:
            for seed in (42, 43, 44):
                section = select_task_split_sections(data, [benchmark], seed)[benchmark]
                split = validate_task_split(section, benchmark)
                expected = (30, 29) if benchmark == "terminalbench2" else (50, 50)
                if (len(split["train"]), len(split["test"])) != expected:
                    raise ValueError("Frozen train/test counts differ")
                folder = output / benchmark / method / f"s{seed}"
                job = {"id": f"{method}_{benchmark}_s{seed}", "method": method, "benchmark": benchmark,
                       "seed": seed, "train_count": expected[0], "test_ids": split["test"],
                       "train_dir": str(folder / "train"), "output": str(folder),
                       "status": "pending", "attempts": 0}
                if playbook_edit_mode != "add_only":
                    job["playbook_edit_mode"] = playbook_edit_mode
                job["command"] = train_command(job, code, manifest)
                jobs.append(job)
    return {"created_at": now(), "model": MODEL, "temperature_all_roles": TEMPERATURE,
            "mode": "isolated", "manifest": str(manifest), "manifest_sha256": digest(manifest),
            "playbook_edit_mode": playbook_edit_mode,
            "gpu": 1, "serve_script": "/home/thskadud/vllm_serve_qwen.sh", "base_url": BASE_URL,
            "train_workers": TRAIN_WORKERS, "evaluation_workers": EVAL_WORKERS,
            "training_schedule": "first pending stream in manifest order; fill all available training slots",
            "premise_workers": {"parent": 2, "reflection": 4, "mutation": 4, "replay": 28, "llm_slots": 32},
            "past_count": 3, "evaluation": "final rank-1 playbook, frozen official test split, no learning",
            "experiment_limits": {slug: experiment_limits(slug) for slug in BENCHMARKS},
            "training_iterations": sum(job["train_count"] for job in jobs),
            "test_tasks": sum(len(job["test_ids"]) for job in jobs), "jobs": jobs}


def frozen_evaluation_jobs(job):
    train = Path(job["train_dir"])
    if read(train / "training_complete.json")["completed_task_count"] != job["train_count"]:
        raise ValueError("Training completion count differs")
    if job["method"] == "ace":
        config = read(train / "experiment_config.json")
        path = train / f"playbook_ace_isolated_{job['benchmark']}.json"
        checkpoint = read(path)
        playbook = checkpoint["playbook"]
        if checkpoint["session_count"] != job["train_count"]:
            from scripts.ace.checkpoint_recovery import completed_prefix

            order = config.get("task_order", [])
            prefix = completed_prefix(train, order)
            if (len(order) != job["train_count"] or len(prefix) != len(order)
                    or len({(slug, str(task)) for slug, task in order}) != len(order)
                    or any(slug != job["benchmark"] for slug, _task in order)
                    or any(item["result"].get("status") in {"error", "cancelled"} for item in prefix)
                    or not prefix[-1]["checkpoint"]
                    or prefix[-1]["checkpoint"]["playbook"] != playbook):
                raise ValueError("ACE checkpoint session count differs without complete task/policy evidence")
    else:
        path = train / "state.json"
        checkpoint = read(path)
        config = checkpoint["config"]
        if checkpoint["cursor"] != job["train_count"]:
            raise ValueError("PREMiSE cursor differs")
        playbook = checkpoint["states"][job["benchmark"]]["beam"][0]["playbook"]
    actor_temperature = config.get("role_temperatures", {}).get("actor", config.get("temperature_all_roles"))
    if actor_temperature != TEMPERATURE or config["seed"] != job["seed"]:
        raise ValueError("Training temperature/seed differs")
    if config["mode"] != "isolated" or config["model"] != MODEL:
        raise ValueError("Training method/model differs")
    edit_mode = job.get("playbook_edit_mode", "add_only")
    if config.get("playbook_edit_mode", "add_only") != edit_mode:
        raise ValueError("Training playbook edit mode differs")
    target = Path(job["output"]) / "evaluation/frozen_checkpoint.json"
    frozen = {"playbook": playbook, "source": str(path), "source_sha256": digest(path),
              "model": MODEL, "temperature_all_roles": TEMPERATURE}
    if config.get("role_temperatures"):
        frozen.pop("temperature_all_roles")
        frozen.update(actor_temperature=actor_temperature, training_role_temperatures=config["role_temperatures"])
    appworld_policy = {}
    if job.get("appworld_policy") is not None:
        policy_text = Path(job["appworld_policy"]).read_text(encoding="utf-8")
        policy_sha = hashlib.sha256(policy_text.encode()).hexdigest()
        if policy_sha != job["appworld_policy_sha256"] or config.get("appworld_policy_sha256") != policy_sha:
            raise ValueError("Training and final evaluation AppWorld policies differ")
        appworld_policy = {"appworld_policy_override": policy_text, "appworld_policy_sha256": policy_sha}
        frozen.update(appworld_policy)
    if edit_mode != "add_only":
        frozen["playbook_edit_mode"] = edit_mode
    context_profile = {}
    if config.get("actor_context", "legacy") != "legacy":
        from exgentic.actor_context import require_matching_actor_context

        mode = config["actor_context"]
        benchmarks = list(config["actor_context_metadata"]["initial_actor_prompt_sha256"])
        require_matching_actor_context(config, mode, benchmarks)
        context_profile = {"actor_context": mode, "actor_context_metadata": config["actor_context_metadata"]}
        frozen.update(context_profile)
    if target.exists() and read(target) != frozen:
        raise ValueError("Frozen evaluation checkpoint changed")
    write(target, frozen)
    policy = {"playbook_edit_mode": edit_mode} if edit_mode != "add_only" else {}
    policy.update(context_profile)
    if context_profile:
        policy["benchmark_kwargs"] = config["benchmark_configs"][job["benchmark"]]
    return [{**policy, **appworld_policy, "base_url": job.get("base_url", BASE_URL),
             "method": job["method"], "benchmark": job["benchmark"], "seed": job["seed"],
             "index": index, "task_id": task, "output": job["output"],
             "frozen": str(target), "frozen_sha256": digest(target)}
            for index, task in enumerate(job["test_ids"])]


def evaluate_one(job):
    evaluation_worker_init(job.get("base_url", BASE_URL))
    if (Path(job["output"]) / "checkpoint_queue.json").exists():
        from scripts.utils.checkpoint_eval_proxy import evaluate_one as shared_checkpoint_evaluation

        return shared_checkpoint_evaluation(job)
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    started = now()
    try:
        if digest(job["frozen"]) != job["frozen_sha256"]:
            raise ValueError("Frozen playbook hash differs")
        frozen_policy = read(job["frozen"])
        playbook = frozen_policy["playbook"]
        if job.get("actor_context", "legacy") != "legacy":
            from exgentic.actor_context import require_matching_actor_context

            mode = job["actor_context"]
            benchmarks = list(job["actor_context_metadata"]["initial_actor_prompt_sha256"])
            require_matching_actor_context(job, mode, benchmarks)
            pinned = frozen_policy.get("stream", frozen_policy)
            if (pinned.get("actor_context") != mode or
                    pinned.get("actor_context_metadata") != job["actor_context_metadata"]):
                raise ValueError("Frozen evaluation actor context differs from the training checkpoint")
        PlaybookStore.reset_all()
        kwargs = {
            "runner": "direct", "enable_thinking": False, "temperature_override": TEMPERATURE,
            "model_settings": {"temperature": TEMPERATURE, "max_tokens": 2048, "timeout": 600, "num_retries": 1},
            "enable_tool_shortlisting": job["benchmark"] == "appworld", "max_selected_tools": 30,
        }
        if job.get("actor_context", "legacy") != "legacy":
            kwargs["actor_context"] = job["actor_context"]
            kwargs["enable_tool_shortlisting"] = False
        if job["method"] == "ace":
            kwargs.update(initial_playbook=playbook, evaluation_mode=True, training_time=False,
                          shuffle_mode="isolated", benchmark_id=job["benchmark"], curator_model=MODEL)
            if job.get("playbook_edit_mode", "add_only") != "add_only":
                kwargs["playbook_edit_mode"] = job["playbook_edit_mode"]
        else:
            kwargs["playbook"] = playbook
            if job.get("actor_context", "legacy") != "legacy":
                kwargs["actor_benchmark"] = job["benchmark"]
        run_id = f"test_{job['method']}_{job['benchmark']}_s{job['seed']}_{job['index']:03d}"
        bm = {"subset": SUBSETS[job["benchmark"]], **job.get("benchmark_kwargs", {}), "use_cache": False}
        if job["benchmark"] == "appworld":
            if job.get("appworld_policy_override") is not None:
                text = job["appworld_policy_override"]
                if hashlib.sha256(text.encode()).hexdigest() != job["appworld_policy_sha256"]:
                    raise ValueError("Frozen AppWorld evaluation policy changed")
                bm["policy_override"] = text
            namespace = hashlib.sha256(job["output"].encode()).hexdigest()[:8]
            bm["env_kwargs"] = {"experiment_name": run_id + "_" + namespace}
        if job["benchmark"] == "terminalbench2":
            bm.update(actor="terminus2", max_interactions=40)
        result = execute(config=RunConfig(
            benchmark=job["benchmark"], agent=job["method"], model=MODEL, task_ids=[job["task_id"]],
            run_id=run_id, output_dir=str(Path(job["output"]) / "evaluation/sessions"), max_workers=1,
            benchmark_kwargs=bm, agent_kwargs=kwargs,
        ))
        if len(result.session_results) != 1:
            raise RuntimeError("Expected exactly one session result")
        session = result.session_results[0]
        if session.status.value in ("error", "cancelled") or session.score is None:
            raise RuntimeError(str(session.details))
        metadata = session.details.get("session_metadata") or {}
        score = session.score
        if job["benchmark"] == "appworld":
            score = 0.0 if metadata.get("experiment_budget_exceeded") else float(metadata["test_tracker"]["success"])
        return {**job, "status": "completed", "score": score, "partial_score": session.score,
                **({"score_metric": "appworld_tgc_v1", "tgc": score} if job["benchmark"] == "appworld" else {}),
                "session_id": session.session_id, "session_status": session.status.value,
                "budget_failure": metadata.get("experiment_budget_failure"), "started_at": started, "ended_at": now()}
    except Exception as exc:
        return {**job, "status": "error", "score": None, "error": f"{type(exc).__name__}: {exc}",
                "started_at": started, "ended_at": now()}


def evaluation_worker_init(base_url=None):
    os.environ.pop("EXGENTIC_PORT_BLOCK_START", None)
    os.environ.update(experiment_environment(base_url=base_url or BASE_URL))


def journal(plan, output, phase, text):
    record = read(output / "journal_record.json") if (output / "journal_record.json").exists() else None
    if record is None:
        return
    try:
        info = request("http://127.0.0.1:8878/api/info")
        url = f"http://127.0.0.1:8878/api/experiments/{record['id']}"
        current = request(url)
        current = current.get("experiment", current)
        payload = {"revision": current["revision"], "status": phase}
        query = urllib.request.Request(url, data=json.dumps(payload).encode(), method="PUT", headers={
            "Content-Type": "application/json", "X-Journal-Token": info["csrf_token"],
        })
        with urllib.request.urlopen(query, timeout=15) as response:
            response.read()
        query = urllib.request.Request(url + "/notes", data=json.dumps({"text": text}).encode(), headers={
            "Content-Type": "application/json", "X-Journal-Token": info["csrf_token"],
        })
        with urllib.request.urlopen(query, timeout=15) as response:
            response.read()
    except Exception as exc:
        write(output / "journal_pending.json", {"phase": phase, "text": text, "error": str(exc), "at": now()})


def launch_server(plan, output):
    script = Path(plan["serve_script"])
    if digest(script) != plan["serve_script_sha256"]:
        raise ValueError("Supplied vLLM script changed since scheduling")
    log = (output / "server_launcher.log").open("ab")
    process = subprocess.Popen(["bash", "-c", 'source "$1"; wait', "qwen-server", str(script)],
                               cwd=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(output / "server_process.json", {"pid": process.pid, "started_at": now(), "script": str(script)})
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM launcher exited; inspect output_nohup_vllm_server_qwen_gpu.log")
        try:
            models = request("http://127.0.0.1:8006/v1/models")["data"]
            if RAW_MODEL in [model["id"] for model in models] and owns_port(process, 8006):
                return process
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(10)
    raise TimeoutError("vLLM did not become ready within 30 minutes")


def launch_bridge(output):
    script = ROOT / "scripts/utils/serve_vllm_tool_parser_bridge.py"
    log = (output / "parser_bridge.log").open("ab")
    process = subprocess.Popen(["/home/thskadud/.vllm/bin/python", "-u", str(script), "--port", "18006"],
                               cwd=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if process.poll() is not None:
            raise RuntimeError("Qwen parser bridge exited")
        try:
            if request("http://127.0.0.1:18006/healthz")["parser"] == "qwen3_coder":
                write(output / "bridge_process.json", {"pid": process.pid, "started_at": now()})
                return process
        except (OSError, KeyError):
            pass
        time.sleep(1)
    raise TimeoutError("Parser bridge did not become ready")


def preflight_model(output):
    configure_experiment_environment()
    tool = {"type": "function", "function": {"name": "runtime_probe", "description": "Runtime smoke check",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}}
    body = {"model": RAW_MODEL, "messages": [{"role": "user", "content": "Call runtime_probe with value ready."}],
            "tools": [tool], "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    tokens = [request("http://127.0.0.1:8006/tokenize", {**body, "tool_choice": choice}, 60)
              for choice in ("auto", "none")]
    if not tokens[0].get("tokens") or tokens[0]["tokens"] != tokens[1].get("tokens"):
        raise ValueError("auto/none tool-choice prompts differ; cannot use the parser bridge")
    body.pop("add_generation_prompt")
    body.pop("chat_template_kwargs")
    response = request(BASE_URL + "/chat/completions", {
        **body, "tool_choice": "auto", "temperature": TEMPERATURE, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }, 180)
    calls = response["choices"][0]["message"].get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != "runtime_probe":
        raise ValueError("Qwen did not produce a parsed tool call in the smoke check")
    write(output / "model_preflight.json", {"at": now(), "identical_tool_choice_tokens": True,
                                           "usage": response.get("usage"), "tool_call": calls[0]})


def run_training_and_evaluation(plan, output, server, bridge, *, resume=False, defer_evaluation=False):
    jobs = resume_jobs(plan, output) if resume else plan["jobs"]
    running = {job["id"]: adopt_training(job) for job in jobs if job["status"] == "training"}
    evaluations = {}
    successes = failures = 0
    evaluation_pool = nullcontext(None) if defer_evaluation else ProcessPoolExecutor(
        max_workers=EVAL_WORKERS, initializer=evaluation_worker_init)
    with evaluation_pool as pool:
        for job in jobs:
            if job["status"] == "test_pending" and not defer_evaluation:
                job["status"] = "evaluating"
                job["evaluation_deferred"] = False
            if defer_evaluation and job["status"] == "evaluating":
                job["status"] = "test_pending"
            if defer_evaluation:
                continue
            if job["status"] in {"evaluating", "completed", "evaluation_error"}:
                for task in frozen_evaluation_jobs(job):
                    folder = Path(task["output"]) / "evaluation"
                    record = folder / "records" / f"{task['index']:03d}.json"
                    if record.exists():
                        result = read(record)
                        if result.get("frozen_sha256") != task["frozen_sha256"]:
                            raise ValueError("Existing evaluation record has a different frozen checkpoint")
                        successes += int(result["status"] == "completed")
                        failures += int(result["status"] == "error")
                    elif job["status"] == "evaluating":
                        attempts = sorted((folder / "attempts").glob(f"{task['index']:03d}_*.json"))
                        if attempts:
                            latest = read(attempts[-1])
                            if latest["status"] != "error" or latest["attempt"] >= 3:
                                write(record, latest)
                                successes += int(latest["status"] == "completed")
                                failures += int(latest["status"] == "error")
                                continue
                        attempt = read(attempts[-1])["attempt"] + 1 if attempts else 1
                        evaluations[pool.submit(evaluate_one, task)] = (task, attempt)
                    else:
                        raise ValueError("Terminal evaluation job is missing a task record")
        while True:
            if server.poll() is not None or bridge.poll() is not None:
                raise RuntimeError("Owned vLLM server or parser bridge exited during experiments")
            for name, (process, job, _slot, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                del running[name]
                if process.returncode:
                    job.update(status="pending" if job["attempts"] < 3 else "training_error",
                               last_returncode=process.returncode)
                    continue
                try:
                    tasks = frozen_evaluation_jobs(job)
                    for task in tasks if not defer_evaluation else []:
                        path = Path(task["output"]) / "evaluation/records" / f"{task['index']:03d}.json"
                        if path.exists():
                            raise ValueError("Unexpected pre-existing evaluation record in fresh run")
                        evaluations[pool.submit(evaluate_one, task)] = (task, 1)
                    job.update(status="test_pending" if defer_evaluation else "evaluating", training_finished_at=now(),
                               evaluation_deferred=defer_evaluation)
                except Exception as exc:
                    job.update(status="training_error", error=f"{type(exc).__name__}: {exc}")
            for future in list(evaluations):
                if not future.done():
                    continue
                task, attempt = evaluations.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**task, "status": "error", "score": None, "error": str(exc)}
                result["attempt"] = attempt
                folder = Path(task["output"]) / "evaluation"
                write(folder / "attempts" / f"{task['index']:03d}_{attempt}.json", result)
                if result["status"] == "error" and attempt < 3:
                    evaluations[pool.submit(evaluate_one, task)] = (task, attempt + 1)
                    continue
                write(folder / "records" / f"{task['index']:03d}.json", result)
                successes += int(result["status"] == "completed")
                failures += int(result["status"] == "error")
            for job in jobs:
                if job["status"] == "evaluating":
                    records = sorted((Path(job["output"]) / "evaluation/records").glob("*.json"))
                    if len(records) == len(job["test_ids"]):
                        rows = [read(path) for path in records]
                        errors = sum(row["status"] == "error" for row in rows)
                        score = sum(row["score"] or 0 for row in rows) / len(rows)
                        write(Path(job["output"]) / "evaluation/summary.json", {
                            "complete": True, "tasks": len(rows), "execution_errors": errors,
                            "score": score if not errors else None, "score_with_errors_zero": score,
                            "temperature_all_roles": TEMPERATURE,
                            "playbook_edit_mode": job.get("playbook_edit_mode", "add_only"),
                        })
                        job["status"] = "completed" if not errors else "evaluation_error"
            capacity = training_workers(plan, output)
            port_base = training_port_base(plan, capacity)
            while len(running) < capacity:
                job = next_training_job(plan, jobs, running)
                if job is None:
                    break
                slot = next(index for index in range(max(capacity, len(jobs)))
                            if index not in [item[2] for item in running.values()])
                env = experiment_environment(base_url=plan.get("base_url", BASE_URL))
                env["EXGENTIC_PORT_BLOCK_START"] = str(port_base + slot * 1000)
                folder = Path(job["output"])
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / "training.log").open("ab")
                process = subprocess.Popen(job["command"], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job.update(status="training", pid=process.pid, slot=slot, port_block_base=port_base,
                           attempts=job["attempts"] + 1, started_at=now())
                running[job["id"]] = (process, job, slot, log)
            write(output / "progress.json", {"at": now(), "jobs": jobs, "running_training": len(running),
                                           "train_workers": capacity,
                                           "pending_evaluations": len(evaluations),
                                           "evaluated": successes, "evaluation_errors": failures})
            if not running and not evaluations and not any(job["status"] == "pending" for job in jobs):
                return jobs
            if evaluations:
                wait(evaluations, timeout=10, return_when=FIRST_COMPLETED)
            else:
                time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--resume-existing", action="store_true",
                        help="Adopt recorded live services/workers and preserve completed task records")
    args = parser.parse_args()
    configure_experiment_environment()
    output = args.output_root.resolve()
    plan = read(output / "launch_manifest.json")
    lock = FileLock(str(output / "controller.lock"), timeout=0)
    lease_path = Path.home() / ".cache/exgentic/qwen_gpu1_experiment.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lock, FileLock(str(lease_path), timeout=0):
        write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now()})
        try:
            if args.resume_existing:
                server, bridge = existing_services(output)
                resume_jobs(plan, output)
                journal(plan, output, "running",
                        f"기존 Qwen 서버·학습 프로세스·평가 결과를 유지하며 스케줄러 재개. "
                        f"동시 학습 실험 {training_workers(plan, output)}개, 최종 평가 {EVAL_WORKERS} workers.")
            else:
                stable = 0
                while stable < 3:
                    try:
                        gpu = gpu_state(plan["gpu"])
                        free = gpu_is_free(gpu) and port_free(8006) and port_free(18006)
                        stable = stable + 1 if free else 0
                        write(output / "status.json", {"phase": "waiting_gpu", "at": now(), "gpu": gpu,
                                                       "stable_free_samples": stable, "required_samples": 3})
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        stable = 0
                        write(output / "status.json", {"phase": "waiting_gpu_probe", "at": now(), "error": str(exc)})
                    if stable < 3:
                        time.sleep(args.poll_seconds)
                write(output / "status.json", {"phase": "starting_vllm", "at": now(), "gpu": gpu})
                server = launch_server(plan, output)
                bridge = launch_bridge(output)
                preflight_model(output)
                journal(plan, output, "running",
                        "GPU 1 반환 확인 후 지정 vLLM 서버와 Qwen parser bridge 시작. "
                        "18개 isolated 학습 및 최종 평가 실행.")
            write(output / "status.json", {"phase": "running", "at": now(), "server_pid": server.pid,
                                           "bridge_pid": bridge.pid})
            jobs = run_training_and_evaluation(plan, output, server, bridge, resume=args.resume_existing)
            errors = [job["id"] for job in jobs if job["status"] != "completed"]
            phase = "completed" if not errors else "blocked"
            write(output / "status.json", {"phase": phase, "at": now(), "failed_jobs": errors})
            journal(plan, output, phase, f"Isolated 실험 종료: {len(jobs) - len(errors)}/18 완료. 실패 작업: {errors}")
        except Exception as exc:
            write(output / "status.json", {"phase": "blocked", "at": now(), "error": f"{type(exc).__name__}: {exc}"})
            journal(plan, output, "blocked", f"실험 실행기 중단: {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
