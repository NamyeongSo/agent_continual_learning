# SPDX-License-Identifier: Apache-2.0
"""Run fixed-panel and training PEACE ablations without rerunning basic PEACE."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import psutil
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from exgentic.agents.premise.ablation_selection import ABLATION_SELECTION_RULES  # noqa: E402
from exgentic.agents.premise.pipeline import PremisePipeline, manifest_order, write_json  # noqa: E402
from exgentic.core.experiment_limits import BENCHMARK_MAX_TURNS, experiment_limits  # noqa: E402
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402
from scripts.premise.submission_panels import (  # noqa: E402
    evaluate_contexts,
    prepare_panels,
    run_panel,
)
from scripts.utils.run_qwen_isolated_when_free import experiment_environment  # noqa: E402

SOURCE = ROOT / "outputs_arr_oct/premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928"
LEGACY = ROOT / ("outputs_arr_oct/premise_terminalbench2_reflection_rerun_20261008/source_snapshot/"
                 "src/exgentic/agents/premise/llm.py")
PRIVATE = ROOT / "outputs/qwen_shared_reflection_medium_fresh1_four_baselines_20261111/private_environment.json"
BENCHMARKS = ["bfcl", "appworld", "terminalbench2"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def models(url):
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=5) as response:
        return json.load(response)["data"]


def prepare(output, *, upstream="http://127.0.0.1:8006", source=SOURCE, proxy_port=18118):
    output = Path(output).resolve()
    if (output / "launch_manifest.json").exists():
        raise ValueError("Study already prepared; use run to resume its queue")
    output.mkdir(parents=True, exist_ok=True)
    available = models(upstream)
    if len(available) != 1 or "Qwen3.6-35B-A3B" not in available[0]["id"]:
        raise ValueError("Expected the user-provided Qwen3.6 endpoint")
    source_config = read(source / "s42_sequential_t3/state.json")["config"]
    manifest = read(source_config["manifest"])
    if sha(source_config["manifest"]) != source_config["manifest_sha256"]:
        raise ValueError("Historical split source changed")
    policy = (ROOT / "outputs/qwen_shared_reflection_medium_fresh1_four_baselines_20261111/"
              "appworld_original_policy.txt")
    shutil.copy2(policy, output / "appworld_original_policy.txt")
    for seed in [42, 43, 44]:
        manifest[f"appworld_{seed}"].setdefault("benchmark_kwargs", {}).update(
            interaction_mode="function_calling",
        )
        manifest[f"terminalbench2_{seed}"].setdefault("benchmark_kwargs", {}).update(
            subset="2.0", actor="terminus2", max_interactions=40,
        )
    write_json(output / "task_splits.json", manifest)
    # Prove membership and order against the historical streams after adding runtime kwargs.
    for run in source.glob("s*_t3"):
        state = read(run / "state.json")
        order, _ = manifest_order(output / "task_splits.json", BENCHMARKS, state["config"]["seed"],
                                  None, state["config"]["mode"])
        if [list(x) for x in order] != state["config"]["order"]:
            raise ValueError("Historical task order was not preserved")
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for folder in ["src", "scripts"]:
        shutil.copytree(ROOT / folder, snapshot / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "installation", "outputs"))
    for name in ["pyproject.toml", "uv.lock"]:
        shutil.copy2(ROOT / name, snapshot / name)
    (snapshot / ".venv").symlink_to((ROOT / ".venv").resolve(), target_is_directory=True)
    (snapshot / ".exgentic").symlink_to((ROOT / ".exgentic").resolve(), target_is_directory=True)
    legacy = output / "frozen_historical_llm.py"
    shutil.copy2(LEGACY, legacy)
    # Docker details remain in an ignored, owner-readable file.
    shutil.copy2(PRIVATE, output / "private_environment.json")
    os.chmod(output / "private_environment.json", 0o600)
    jobs = []
    for seed in [42, 43, 44]:
        for mode in ["sequential", "interleaved"]:
            for rule in ABLATION_SELECTION_RULES:
                identity = f"{rule}_{mode}_s{seed}"
                jobs.append({"id": identity, "kind": "training", "rule": rule, "mode": mode, "seed": seed,
                             "output": str(output / "runs" / identity), "benchmark": "mixed"})
    panels = prepare_panels(source, output / "task_splits.json", output)
    for panel in panels:
        panel.update(output=str(output / "runs" / panel["id"]), rule="pareto")
    files = {str(p.relative_to(snapshot)): sha(p) for folder in ["src", "scripts"]
             for p in (snapshot / folder).rglob("*.py")}
    write_json(output / "source_sha256.json", files)
    plan = {
        "version": "peace-submission-ablation-v1", "created_at": now(), "output": str(output),
        "source": str(source), "snapshot": str(snapshot), "manifest": str(output / "task_splits.json"),
        "manifest_sha256": sha(output / "task_splits.json"),
        "historical_manifest_sha256": source_config["manifest_sha256"],
        "model": "openai/" + available[0]["id"], "upstream": upstream, "proxy_port": proxy_port,
        "proxy_base": f"http://127.0.0.1:{proxy_port}", "global_llm_concurrency": 100,
        "temperature_all_roles": 0.0, "past_count": 3, "parents": 2, "children": 4, "survivors": 2,
        "basic_peace_training_jobs": 0, "epochs": 1, "train_slots": 3, "test_slots": 1, "panel_slots": 1,
        "actor_context": "legacy", "appworld_policy": str(output / "appworld_original_policy.txt"),
        "appworld_policy_sha256": sha(output / "appworld_original_policy.txt"),
        "historical_llm": str(legacy), "historical_llm_sha256": sha(legacy),
        "experiment_limits": {b: experiment_limits(b) for b in BENCHMARKS},
        "training_jobs": jobs, "panel_jobs": panels,
        "test_selection": "Rank 1 only; Sequential at BFCL cp50/AppWorld cp100/TB cp130; Interleaved cp130",
        "panel_selection": "18 final eligible benchmark updates; identical 4 children and 3 histories across arms",
        "cost_rates_usd_per_million": {"input": 0.05, "output": 0.70},
        "limitations": [
            "User-provided FP8 weights differ from the historical unquantized model.",
            "New actors use the repository's shared 100k/2048 and 30/40-turn budgets, unlike older runs.",
            "The historical basic PEACE scores are references, not a freshly matched control.",
            "Native actor temperature is explicitly zero in this study; historical native actors omitted it.",
        ],
    }
    write_json(output / "launch_manifest.json", plan)
    write_json(output / "progress.json", {"at": now(), "jobs": {j["id"]: {"status": "pending"}
                                                              for j in [*jobs, *panels]}})
    return plan


def environment(plan, job=None, port=None):
    group = job.get("request_group", job["id"]) if job else "health"
    base = plan["proxy_base"] + "/jobs/" + group + "/v1"
    env = experiment_environment(base_url=base)
    env.update(read(Path(plan["output"]) / "private_environment.json"))
    for key in ["EXGENTIC_INFERENCE_PROFILE", "EXGENTIC_INFERENCE_PROFILE_SHA256",
                "EXGENTIC_ACTOR_OUTPUT_UNLIMITED", "EXGENTIC_APPWORLD_TURN_CAP_PCT"]:
        env.pop(key, None)
    env.update(PYTHONPATH=plan["snapshot"] + "/src:" + plan["snapshot"],
               PYTHONUNBUFFERED="1", LITELLM_LOCAL_MODEL_COST_MAP="True",
               EXGENTIC_LLM_TRACE_FORMAT="dedup_v1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if port is not None:
        env["EXGENTIC_PORT_BLOCK_START"] = str(port)
    return env


def pipeline_for(plan, job, stage=None):
    stage = stage or ("train" if job["kind"] == "training" else "panel")
    group = job["id"] + "__" + stage
    root = Path(job["output"]) / (
        "test_runner" if stage == "test" else "train" if job["kind"] == "training" else "runner"
    )
    pipeline = PremisePipeline(
        output=root, manifest=Path(plan["manifest"]),
        benchmarks=job.get("source_benchmarks", plan.get("benchmarks", BENCHMARKS)), mode=job["mode"],
        seed=job["seed"], count=None, past_count=3, model=plan["model"],
        base_url=plan["proxy_base"] + "/jobs/" + group + "/v1",
        temperature=0.0, actor_context="legacy", parent_workers=2, reflection_workers=4, mutation_workers=4,
        replay_workers=100, llm_calls=100, tool_calling_max_tokens=2048, replay_retries=2,
        max_steps=dict(BENCHMARK_MAX_TURNS), appworld_policy_override=Path(plan["appworld_policy"]).read_text(),
        appworld_candidate_metric="pct", candidate_selection=job["rule"],
    )
    if sha(plan["historical_llm"]) != plan["historical_llm_sha256"]:
        raise ValueError("Pinned historical reflection/mutation/judge implementation changed")
    name = "exgentic.agents.premise._submission_historical_llm"
    spec = importlib.util.spec_from_file_location(name, plan["historical_llm"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    pipeline.llm = module.PremiseLLM(
        plan["model"], pipeline.base_url, max_calls=100, timeout=600.0, num_retries=1,
        trace_dir=root / "llm_calls",
    )
    return pipeline


def work(plan, identity, stage):
    job = next(j for j in [*plan["training_jobs"], *plan["panel_jobs"]] if j["id"] == identity)
    root = Path(job["output"])
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "worker.lock"), timeout=0):
        pipeline = pipeline_for(plan, job, stage)
        write_json(root / "job_status.json", {"phase": stage, "at": now(), "temperature": 0.0})
        if stage == "train":
            target = plan.get("train_stop_after", len(pipeline.order))
            if plan.get("benchmarks") == ["bfcl"]:
                if target != 50 or any(b != "bfcl" for b, _ in pipeline.order[:target]):
                    raise ValueError("BFCL-only training must stop after its original 50 training tasks")
            pipeline.run(stop_after=target)
            if pipeline.cursor != target:
                raise RuntimeError("Training did not exhaust its one-pass stream")
            write_json(root / "training_complete.json", {
                "complete": True, "cursor": pipeline.cursor, "at": now(),
                "state_sha256": sha(root / "train/state.json"),
            })
            return
        if stage == "panel":
            if sha(job["panel"]) != job["panel_sha256"]:
                raise ValueError("Frozen panel changed")
            result = run_panel(pipeline, job["panel"], root, plan["manifest"])
        else:
            state = read(root / "train/state.json")
            policies = {}
            benchmarks = plan.get("benchmarks", BENCHMARKS)
            points = {"bfcl": 50, "appworld": 100, "terminalbench2": 130}
            for benchmark in benchmarks:
                point = points[benchmark]
                checkpoint = point if job["mode"] == "sequential" else 130
                snapshot = read(root / f"train/checkpoints/checkpoint_{checkpoint:03d}.json")
                if snapshot["config"] != state["config"]:
                    raise ValueError("Frozen checkpoint configuration differs from training")
                policies[benchmark] = copy.deepcopy(snapshot["states"]["global"]["beam"][0])
            sections = select_task_split_sections(read(plan["manifest"]), benchmarks, job["seed"])
            splits = {b: validate_task_split(s, b)["test"] for b, s in sections.items()}
            result = evaluate_contexts(pipeline, {"learned": {"by_benchmark": policies}}, splits, root / "evaluation")
        if not result["complete"]:
            raise RuntimeError("Unmeasured infrastructure failures remain in frozen evaluation")
        write_json(root / "job_complete.json", {
            "complete": True, "at": now(), "id": job["id"], "temperature": 0.0, "model": plan["model"], **result,
        })
        write_json(root / "job_status.json", {"phase": "completed", "at": now()})


def alive(state):
    try:
        process = psutil.Process(state["pid"])
        return abs(process.create_time() - state["created_at"]) < 0.1 and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError):
        return False


def cost_summary(plan):
    output = Path(plan["output"])
    stages = {}
    for job in [*plan["training_jobs"], *plan["panel_jobs"]]:
        folder = Path(job["output"]) / ("train" if job["kind"] == "training" else "runner") / "llm_calls"
        for file in folder.glob("*.json"):
            value = read(file)
            response = value.get("response") or {}
            if response.get("id"):
                stages[job["id"], response["id"]] = value.get("stage", "analysis")
    totals = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0,
                                 "unknown_usage_calls": 0, "request_seconds": 0.0})
    ledger = output / "serving/usage.jsonl"
    if ledger.exists():
        with ledger.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("path") not in ["/v1/chat/completions", "/v1/completions"]:
                    continue
                identity, _, phase = row["job"].rpartition("__")
                identity = identity or row["job"]
                category = "health" if row["health_probe"] else stages.get(
                    (identity, row.get("response_id")), "actor")
                result = totals[identity, phase or "unknown", category]
                result["calls"] += 1
                result["request_seconds"] += row["request_seconds"]
                usage = row.get("usage") or {}
                if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                    result["unknown_usage_calls"] += 1
                else:
                    result["input_tokens"] += usage["prompt_tokens"]
                    result["output_tokens"] += usage["completion_tokens"]
    rows = []
    for (job, phase, category), value in totals.items():
        rows.append({"job": job, "phase": phase, "category": category, **value,
                     "usd_equivalent": (value["input_tokens"] * 0.05 + value["output_tokens"] * 0.70) / 1e6})
    write_json(output / "cost_summary.json", {"rows": rows, "rates": plan["cost_rates_usd_per_million"],
                                            "unknown_usage_is_not_assumed_zero": True, "at": now()})


def run(plan):
    output = Path(plan["output"])
    with FileLock(str(output / "controller.lock"), timeout=0):
        for key in ["manifest", "appworld_policy", "historical_llm"]:
            if sha(plan[key]) != plan[key + "_sha256"]:
                raise ValueError(f"Pinned {key} changed")
        for file, expected in read(output / "source_sha256.json").items():
            if sha(Path(plan["snapshot"]) / file) != expected:
                raise ValueError(f"Frozen source changed: {file}")
        available = models(plan["upstream"])
        if len(available) != 1 or "openai/" + available[0]["id"] != plan["model"]:
            raise ValueError("The serving model differs from the prepared study")
        progress = read(output / "progress.json")
        jobs = [*plan["training_jobs"], *plan["panel_jobs"]]
        # Separate blocks from existing experiments; never take their ports or stop them.
        occupied = {c.laddr.port // 1000 * 1000 for c in psutil.net_connections("tcp") if c.status == "LISTEN"}
        owned = {state["port_block_start"] for state in progress["jobs"].values()
                 if state.get("status") in ["training", "evaluating", "panel"] and alive(state)}
        candidates = plan.get("environment_port_blocks", list(range(51000, 64000, 1000)))
        if len(candidates) != len(set(candidates)) or any(
            type(p) is not int or p % 1000 or not 2000 <= p <= 64000 for p in candidates
        ):
            raise ValueError("Environment blocks must be unique, valid 1000-port blocks")
        ports = [p for p in candidates if p not in occupied or p in owned]
        if len(ports) < plan["train_slots"] + plan["panel_slots"] + plan["test_slots"]:
            raise RuntimeError("Insufficient unused environment port blocks")
        write_json(output / "controller_process.json", {
            "pid": os.getpid(), "created_at": psutil.Process().create_time(),
        })
        last_cost = 0.0
        while True:
            active = defaultdict(list)
            for job in jobs:
                state = progress["jobs"][job["id"]]
                root = Path(job["output"])
                if (root / "job_complete.json").exists():
                    marker = read(root / "job_complete.json")
                    if not (marker["complete"] and marker["id"] == job["id"]
                            and marker["model"] == plan["model"] and marker["temperature"] == 0.0):
                        raise ValueError("Completion marker differs from the study")
                    state["status"] = "completed"
                elif state["status"] in ["training", "evaluating", "panel"]:
                    if alive(state):
                        active[state["stage"]].append(state)
                    elif state["stage"] == "train" and (root / "training_complete.json").exists():
                        state.update(status="evaluation_pending", training_ended_at=now())
                    else:
                        state.update(status="error", ended_at=now())
            used = {state["port_block_start"] for values in active.values() for state in values}
            for stage, limit in [("train", plan["train_slots"]), ("panel", plan["panel_slots"]),
                                 ("test", plan["test_slots"])]:
                for job in jobs:
                    if len(active[stage]) >= limit:
                        break
                    state = progress["jobs"][job["id"]]
                    eligible = ((stage == "train" and job["kind"] == "training" and state["status"] == "pending")
                                or (stage == "panel" and job["kind"] == "frozen_panel" and state["status"] == "pending")
                                or (stage == "test" and state["status"] == "evaluation_pending"))
                    if not eligible:
                        continue
                    port = next(p for p in ports if p not in used)
                    folder = Path(job["output"])
                    folder.mkdir(parents=True, exist_ok=True)
                    command = [str(Path(plan["snapshot"]) / ".venv/bin/python"), "-u", "-m",
                               "scripts.premise.run_submission_ablations", "work", "--output", plan["output"],
                               "--job", job["id"], "--stage", stage]
                    with (folder / f"{stage}.log").open("a") as log:
                        process = subprocess.Popen(
                            command, cwd=plan["snapshot"],
                            env=environment(plan, {**job, "request_group": job["id"] + "__" + stage}, port),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        )
                    state.update(status={"train": "training", "test": "evaluating", "panel": "panel"}[stage],
                                 stage=stage, pid=process.pid, created_at=psutil.Process(process.pid).create_time(),
                                 started_at=now(), command=command, port_block_start=port)
                    active[stage].append(state)
                    used.add(port)
            statuses = [state["status"] for state in progress["jobs"].values()]
            terminal = not any(active.values()) and all(s in ["completed", "error"] for s in statuses)
            progress["at"] = now()
            write_json(output / "progress.json", progress)
            write_json(output / "status.json", {
                "phase": ("completed" if all(s == "completed" for s in statuses)
                          else "completed_with_errors" if terminal else "running"),
                "at": now(), "global_llm_concurrency": 100, "temperature": 0.0,
                "completed": statuses.count("completed"), "total": len(jobs),
                "active": {key: len(value) for key, value in active.items()},
                "errors": [key for key, state in progress["jobs"].items() if state["status"] == "error"],
            })
            if time.monotonic() - last_cost > 300:
                cost_summary(plan)
                last_cost = time.monotonic()
            if terminal:
                cost_summary(plan)
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "work"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--stage", choices=["train", "test", "panel"])
    args = parser.parse_args()
    if args.action == "prepare":
        plan = prepare(args.output)
        print(json.dumps({"training_jobs": len(plan["training_jobs"]), "panels": len(plan["panel_jobs"]),
                          "basic_peace_training_jobs": 0, "temperature": 0.0, "concurrency": 100}))
        return
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    plan = read(args.output / "launch_manifest.json")
    if args.action == "work":
        work(plan, args.job, args.stage)
    else:
        run(plan)


if __name__ == "__main__":
    main()
