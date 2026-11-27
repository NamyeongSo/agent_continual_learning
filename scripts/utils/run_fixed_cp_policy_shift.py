# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Continue a drained CP10 queue with baseline-specific CP0 histories and ΔCP probes."""

from __future__ import annotations

import argparse
import ast
import asyncio
import fcntl
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.utils import run_checkpoint_policy_shift as common  # noqa: E402
from scripts.utils import run_pairwise_policy_shift as pairs  # noqa: E402
from scripts.utils.policy_shift_metrics import (  # noqa: E402
    canonical,
    digest,
    load_json,
    normalize_calls,
    normalize_terminal,
    response_message,
    terminal_parser,
    write_json,
)

LABELS = {"ace_aud": "ACE AUD", "gepa_val": "GEPA Val", "gepa_train_as_val": "GEPA Train-as-val",
          "skillopt_val": "SkillOpt Val", "skillopt_train_as_val": "SkillOpt Train-as-val",
          "peace_no_alignment_aud": "PEACE AUD w/o alignment"}


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def api(path, body=None, method=None):
    base = "http://127.0.0.1:8878"
    headers = {"Content-Type": "application/json"}
    if body is not None:
        headers["X-Journal-Token"] = api("/api/info")["csrf_token"]
    request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def journal(plan, status, summary):
    if not plan.get("journal_id"):
        return
    path = "/api/experiments/" + plan["journal_id"]
    record = api(path)
    api(path, {"revision": record["revision"], "status": status, "summary": summary}, "PUT")


def gate(plan):
    with common.readonly_queue(plan["queue_output"]) as db:
        counts = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
        errors = {str(row["id"]): digest(row["result"]) for row in db.execute(
            "SELECT id,result FROM tasks WHERE status='error'")}
        streams = {row[0] for row in db.execute("SELECT id FROM streams")}
        invalid = db.execute("SELECT count(*) FROM stream_invalidations").fetchone()[0]
    expected_errors = plan.get("terminal_timeout_exclusions", {})
    ready = (not counts.get("pending") and not counts.get("running") and counts.get("completed", 0) > 0
             and set(counts) <= {"completed", "error"}
             and sum(counts.values()) == plan.get("expected_evaluation_tasks", sum(counts.values()))
             and errors == expected_errors and streams == set(plan["selected_stream_ids"]) and not invalid)
    return {"ready": ready, "counts": counts, "preserved_timeout_errors": len(errors),
            "unreviewed_errors": sorted(set(errors) - set(expected_errors))}


def frozen_renderer(snapshot, checks):
    """Execute the trusted frozen renderer, including its original stats helper."""
    utils = snapshot / "src/exgentic/agents/ace/playbook_utils.py"
    generator = snapshot / "src/exgentic/agents/ace/prompts/generator.py"
    for path in (utils, generator):
        checks[str(path)] = digest(path.read_bytes())
    name = "_shift_playbook_utils_" + checks[str(utils)][:12]
    spec = importlib.util.spec_from_file_location(name, utils)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    tree = ast.parse(generator.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or not node.level]
    scope = {"get_playbook_stats": module.get_playbook_stats}
    exec(compile(tree, str(generator), "exec"), scope)
    return scope["build_generator_system_prompt"]


def actor_prompt(spec, content, checks, renderers):
    snapshot = Path(spec["snapshot"])
    if spec["method"] in {"gepa", "skillopt"}:
        suffix = "txt" if spec["method"] == "gepa" else "md"
        path = snapshot / f"scripts/baselines/prompts/{spec['method']}/base/{spec['benchmark']}.{suffix}"
        checks[str(path)] = digest(path.read_bytes())
        return path.read_text() + "\n\n" + content
    if spec["benchmark"] == "terminalbench2":
        return content
    if snapshot not in renderers:
        renderers[snapshot] = frozen_renderer(snapshot, checks)
    return renderers[snapshot](content)


def session_tools(metadata):
    """Recover the unchanged full tool contract when the callback omitted kwargs."""
    def clamp(value):
        if isinstance(value, dict):
            return {k: clamp(v) for k, v in value.items()}
        if isinstance(value, list):
            return list(map(clamp, value))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 2147483647:
            return type(value)(2147483647)
        return value

    tools = [{"type": "function", "function": {"name": a["name"], "description": a["description"],
                                                "parameters": clamp(a["arguments_schema"])}}
             for a in metadata["actions"] if not a.get("is_message")]
    return sorted(tools, key=lambda tool: (tool["type"], tool["function"]["name"]))


def build_jobs(db, plan):
    settings = canonical({k: plan[k] for k in ("model", "temperature", "input_tokens", "output_tokens")})

    def request(anchor, prompt, sample):
        identifier = digest(canonical([anchor, prompt, sample, settings]))
        db.execute("INSERT OR IGNORE INTO jobs(id,anchor,prompt,sample) VALUES(?,?,?,?)",
                   (identifier, anchor, prompt, sample))
        return identifier

    for policy in db.execute("SELECT * FROM policies ORDER BY stream,point").fetchall():
        root = db.execute("SELECT prompt FROM policies WHERE stream=? AND point=0", (policy["stream"],)).fetchone()[0]
        anchors = db.execute("""SELECT a.id FROM anchors a JOIN anchor_owners o ON o.anchor=a.id
            WHERE a.benchmark=? AND a.seed=? AND o.method=?""",
            (policy["benchmark"], policy["seed"], policy["method"])).fetchall()
        if not anchors:
            raise ValueError("No own-baseline histories for " + policy["stream"])
        for anchor in anchors:
            reference = request(anchor[0], root, "root:" + policy["stream"])
            prediction = (request(anchor[0], root, "repeat:" + policy["stream"]) if policy["point"] == 0 else
                          reference if policy["prompt"] == root else
                          request(anchor[0], policy["prompt"], "policy:" + policy["stream"]))
            db.execute("INSERT OR REPLACE INTO bindings VALUES(?,?,?,?,?)",
                       (policy["stream"], policy["point"], anchor[0], prediction, reference))
        db.commit()


def prepare_data(output, plan):
    from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows

    if (output / "prepared.json").exists():
        return load_json(output / "prepared.json")
    db = common.own_db(output)
    db.execute("CREATE TABLE IF NOT EXISTS anchor_owners(anchor TEXT,method TEXT,PRIMARY KEY(anchor,method))")
    checks, renderers, audit, banks = {}, {}, [], {}
    specs = load_json(output / "source_scope.json")
    parser = terminal_parser(plan["terminal_parser_path"])
    with common.readonly_queue(plan["queue_output"]) as source:
        for spec in specs:
            key = (spec["method"], spec["benchmark"])
            for cp in source.execute("SELECT * FROM checkpoints WHERE stream=? ORDER BY point", (spec["id"],)):
                path = Path(cp["file"])
                checks[str(path)] = digest(path.read_bytes())
                payload = load_json(path)
                if digest(payload["content"]) != cp["sha"]:
                    raise ValueError("A frozen checkpoint changed")
                prompt = actor_prompt(spec, payload["content"], checks, renderers)
                sha = digest(prompt)
                db.execute("INSERT OR IGNORE INTO prompts VALUES(?,?)", (sha, prompt))
                db.execute("INSERT OR REPLACE INTO policies VALUES(?,?,?,?,?,?)",
                           (spec["id"], cp["point"], spec["benchmark"], spec["run_variant"], spec["seed"], sha))
                if cp["point"] == 0:
                    if key in banks and banks[key]["prompt"] != prompt:
                        raise ValueError("Initial context differs across seeds/variations")
                    banks.setdefault(key, {"cp": cp["canonical"], "prompt": prompt, "spec": spec})
            db.commit()
        for (method, benchmark), bank in sorted(banks.items()):
            variants = sorted({s["run_variant"] for s in specs if s["method"] == method})
            tasks = source.execute("SELECT * FROM tasks WHERE cp=? AND split='train' ORDER BY ordinal,id",
                                   (bank["cp"],)).fetchall()
            usable = 0
            for task in tasks:
                if task["status"] != "completed":
                    audit.append({"method": method, "benchmark": benchmark, "task": task["task_id"],
                                  "reason": "source_evaluation_missing", "status": task["status"]})
                    continue
                folder = Path(json.loads(task["result"])["session_output"])
                metadata_paths = list(folder.rglob("session.json"))
                if len(metadata_paths) != 1:
                    raise ValueError("Ambiguous CP0 history session")
                metadata_path = metadata_paths[0]
                metadata = load_json(metadata_path)
                checks[str(metadata_path)] = digest(metadata_path.read_bytes())
                traces = (list(folder.glob("**/run/litellm/trace.jsonl")) if benchmark == "terminalbench2"
                          else [metadata_path.parent / "agent/litellm/trace.jsonl"])
                if len(traces) != 1 or not traces[0].exists():
                    audit.append({"method": method, "benchmark": benchmark, "task": task["task_id"],
                                  "reason": "missing_trace"})
                    continue
                trace = traces[0]
                checks[str(trace)] = digest(trace.read_bytes())
                tools = session_tools(metadata) if benchmark != "terminalbench2" else []
                config = load_json(metadata_path.parent / "config.json")
                kwargs = config.get("agent_kwargs", {})
                if kwargs.get("enable_tool_shortlisting") and len(tools) > kwargs.get("max_selected_tools", 30):
                    raise ValueError("Cannot reconstruct a missing shortlisted tool contract")
                seen, count, recovered = set(), 0, 0
                for row in iter_expanded_trace_rows(trace):
                    try:
                        if row.get("status") != "success":
                            continue
                        response = row["response"]
                        identity = response.get("id") or digest(canonical(response))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        message = response_message(response)
                        if benchmark == "terminalbench2":
                            normalize_terminal(message, parser)
                        elif not normalize_calls(message):
                            continue
                        request = row["request"]
                        acting_tools = request.get("tools", tools)
                        if benchmark != "terminalbench2" and request.get("tools") is not None:
                            if canonical(acting_tools) != canonical(tools):
                                raise ValueError("Recorded and reconstructed full tool contracts differ")
                        elif benchmark != "terminalbench2":
                            recovered += 1
                        span = common.find_span(request["messages"], bank["prompt"])
                        for seed in (42, 43, 44):
                            anchor = {"benchmark": benchmark, "seed": seed, "history_baseline": method,
                                      "task_id": task["task_id"], "messages": request["messages"],
                                      "tools": acting_tools, "policy_span": span}
                            identifier = digest(canonical(anchor))
                            path = output / "histories" / f"{identifier}.json"
                            common.freeze(path, anchor)
                            db.execute("INSERT OR IGNORE INTO anchors VALUES(?,?,?,?,?)",
                                       (identifier, benchmark, seed, task["task_id"], str(path)))
                            db.executemany("INSERT OR IGNORE INTO anchor_owners VALUES(?,?)",
                                           [(identifier, variant) for variant in variants])
                        count += 1
                    except (ValueError, KeyError, TypeError, IndexError, SyntaxError) as error:
                        audit.append({"method": method, "benchmark": benchmark, "task": task["task_id"],
                                      "reason": "invalid_reference", "error": str(error)[:300]})
                usable += count
                audit.append({"method": method, "benchmark": benchmark, "task": task["task_id"],
                              "reason": "coverage", "histories": count, "tools_recovered": recovered})
                db.commit()
                write_json(output / "status.json", {"phase": "preparing_histories", "pid": os.getpid(),
                                                     "bank": [method, benchmark], "histories": usable})
            if not usable:
                raise ValueError(f"No valid CP0 acting prefixes for {method}/{benchmark}")
    build_jobs(db, plan)
    prepared = {"source_hashes": checks, "histories": db.execute("SELECT count(*) FROM anchors").fetchone()[0],
                "physical_history_banks": len(banks), "policy_instances": db.execute(
                    "SELECT count(*) FROM policies").fetchone()[0], "effective_requests": db.execute(
                    "SELECT count(*) FROM jobs").fetchone()[0],
                "environment_execution": False, "learning": False,
                "request_sampling": "One draw per distinct stream/prompt/history; independent CP0 repeat control"}
    write_json(output / "history_audit.json", audit)
    write_json(output / "prepared.json", prepared)
    db.close()
    return prepared


def plot(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#7753A6", "#D58A21", "#D58A21", "#2B9472", "#2B9472", "#2877B6"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, benchmark in zip(axes, ("bfcl", "appworld", "terminalbench2"), strict=True):
        metric = "accuracy" if benchmark == "terminalbench2" else "strict_accuracy"
        for (method, label), color in zip(LABELS.items(), colors, strict=True):
            rows = sorted((r for r in summary["scores"] if (r["benchmark"], r["method"], r["metric"]) ==
                           (benchmark, method, metric)), key=lambda r: r["checkpoint_distance"])
            if not rows:
                continue
            x, y, sd = ([r[k] for r in rows] for k in ("checkpoint_distance", "mean_shift", "pair_sample_sd"))
            correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(y) > 0 else None
            caption = label + (f" (r={correlation:.2f})" if correlation is not None else " (r=undefined)")
            ax.plot(x, y, marker="o", color=color, linestyle="--" if "train_as_val" in method else "-",
                    label=caption, markersize=4)
            ax.fill_between(x, [max(0, a-b) for a, b in zip(y, sd, strict=True)],
                            [min(1, a+b) for a, b in zip(y, sd, strict=True)], color=color, alpha=.08)
        title = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "Terminal-Bench"}[benchmark]
        ax.set(title=title, xlabel="Checkpoint distance (ΔCP)", ylabel="Policy shift (1 - accuracy)",
               ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Baseline-specific CP0 histories · temperature 0 · mean ± pair SD")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"analysis/policy_shift_delta_cp.{suffix}", dpi=180)
    plt.close(fig)


def freeze_plan(output, queue_output):
    output.mkdir(parents=True, exist_ok=False)
    with common.readonly_queue(queue_output) as db:
        specs = [json.loads(r[0]) for r in db.execute("SELECT spec FROM streams ORDER BY id")]
        expected_tasks = db.execute("SELECT count(*) FROM tasks").fetchone()[0]
        exclusions = {}
        for row in db.execute("SELECT * FROM tasks WHERE status='error'"):
            result = json.loads(row["result"])
            if row["attempts"] != 3 or "TimeoutError" not in result.get("error", ""):
                raise ValueError("Unexpected CP error requires review before policy shift")
            exclusions[str(row["id"])] = digest(row["result"])
    if len(specs) != 54 or {s["run_variant"] for s in specs} != set(LABELS):
        raise ValueError("Expected the six-setting CP10 cohort")
    parser_paths = list((Path.home() / ".exgentic/benchmarks/terminalbench2/venv/lib").glob(
        "python*/site-packages/harbor/agents/terminus_2/terminus_json_plain_parser.py"))
    if len(parser_paths) != 1:
        raise ValueError("Official native action parser not found")
    parser = output / "frozen/terminus_json_plain_parser.py"
    parser.parent.mkdir()
    shutil.copy2(parser_paths[0], parser)
    snapshot = output / "code_snapshot"
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("policy_shift_metrics.py", "run_checkpoint_policy_shift.py", "run_pairwise_policy_shift.py",
                 "run_fixed_cp_policy_shift.py"):
        target = snapshot / "scripts/utils" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts/utils" / name, target)
    command = [sys.executable, "-u", str(snapshot / "scripts/utils/run_fixed_cp_policy_shift.py"),
               "--output", str(output)]
    plan = {"created_at": now(), "queue_output": str(queue_output), "selected_stream_ids": [s["id"] for s in specs],
            "expected_evaluation_tasks": expected_tasks,
            "terminal_timeout_exclusions": exclusions, "workers": 256, "per_gpu_query_limit": 128,
            "query_router_url": "http://127.0.0.1:18008/health", "model": "openai/Qwen/Qwen3.6-35B-A3B",
            "base_url": "http://127.0.0.1:18008/v1", "tokenize_url": "http://127.0.0.1:8007/tokenize",
            "temperature": 0.0, "input_tokens": 100000, "output_tokens": 2048,
            "checkpoint_mode": "all", "comparison": "all_within_seed_checkpoint_pairs_by_distance",
            "terminal_parser_path": str(parser), "terminal_parser_sha256": digest(parser.read_bytes()),
            "history_bank": "Each baseline's own CP0 train-evaluation acting prefixes; shared within its variations",
            "aggregation": "Task-equal within CP pair; pair mean/sample SD at each ΔCP across three seeds",
            "history_sampling_temperature": 0.7, "environment_execution": False, "learning": False,
            "method_labels": LABELS, "command": command}
    record = api("/api/experiments", {"title": "CP10 → ΔCP policy shift · Original AppWorld · PEACE w/o alignment",
                 "purpose": "CP 평가 큐 종료 후 각 baseline 자체 CP0 history에서 모든 CP쌍의 행동 drift 측정",
                 "status": "planned", "run_path": str(output), "model": plan["model"],
                 "method": "; ".join(LABELS.values()), "benchmark": "BFCL, AppWorld, Terminal-Bench",
                 "seeds": "42,43,44", "split": plan["history_bank"],
                 "budget": "temperature0; global256; dual-GPU router cap128/GPU; input100k/output2048",
                 "command": " ".join(command), "limitations": "CP terminal timeouts remain missing; "
                 "history banks differ "
                 "by baseline as requested. AppWorld loose tool-name metrics may be uninformative with execute_python.",
                 "tags": ["policy-shift", "cp10", "baseline-own-history", "temperature0", "queue256"]})
    plan["journal_id"] = record["id"]
    write_json(output / "plan.json", plan)
    write_json(output / "source_scope.json", specs)
    write_json(output / "code_hashes.json", {str(p.relative_to(snapshot)): digest(p.read_bytes())
                                             for p in snapshot.rglob("*") if p.is_file()})
    write_json(queue_output / "policy_shift_followup.json", {"output": str(output), "journal_id": record["id"]})
    return plan


def run(output):
    plan = load_json(output / "plan.json")
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ticks = (Path("/proc") / str(os.getpid()) / "stat").read_text().rpartition(")")[2].split()[19]
        write_json(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(),
                   "start_ticks": ticks,
                   "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
        while True:
            state = gate(plan)
            write_json(output / "status.json", {"phase": "preparing" if state["ready"] else "waiting_checkpoint_queue",
                                                 "pid": os.getpid(), "at": now(), "gate": state})
            if state["ready"]:
                break
            time.sleep(30)
        journal(plan, "running", "CP 큐의 모든 시도 종료 확인. 시간초과 누락을 보존하고 자체 CP0 history 준비 시작.")
        snapshot = output / plan.get("runtime_snapshot", "code_snapshot")
        for relative, sha in load_json(output / plan.get("runtime_code_hashes", "code_hashes.json")).items():
            if digest((snapshot / relative).read_bytes()) != sha:
                raise ValueError("Frozen policy-shift runtime changed")
        if digest(Path(plan["terminal_parser_path"]).read_bytes()) != plan["terminal_parser_sha256"]:
            raise ValueError("Frozen native action parser changed")
        prepared = prepare_data(output, plan)
        for path, sha in prepared["source_hashes"].items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError("A frozen policy/history source changed")
        journal(plan, "running", f"Policy shift 측정 시작: histories {prepared['histories']}, "
                f"requests {prepared['effective_requests']}; temperature0/global{plan['workers']}, "
                "baseline별 history 분리.")
        asyncio.run(common.probe(output, plan))
        summary = pairs.analyze(output, plan)
        plot(output, summary)
        write_json(output / "status.json", {"phase": summary["status"], "pid": os.getpid(), "at": now(),
                                             "counts": summary["counts"], "workers": plan["workers"]})
        journal(plan, "completed", f"Policy shift 완료: {summary['counts']}, common histories "
                f"{summary['common_histories']}. ΔCP별 accuracy/F1 및 pair SD 저장.")
        for path in [output / "prepared.json", output / "history_audit.json", output / "analysis/summary.json",
                     *sorted((output / "analysis").glob("*.csv")), *sorted((output / "analysis").glob("*.png"))]:
            api("/api/experiments/" + plan["journal_id"] + "/import", {"path": str(path), "caption": path.name})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        freeze_plan(args.output.resolve(), args.queue_output.resolve())
    if not args.freeze_only:
        try:
            run(args.output.resolve())
        except Exception as error:
            write_json(args.output / "status.json", {"phase": "failed", "pid": os.getpid(), "at": now(),
                                                     "error": f"{type(error).__name__}: {error}"})
            raise
