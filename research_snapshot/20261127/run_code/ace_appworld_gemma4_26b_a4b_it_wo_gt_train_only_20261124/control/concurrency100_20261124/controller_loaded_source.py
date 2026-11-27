# SPDX-License-Identifier: Apache-2.0
"""Run fresh Gemma ACE w/o-GT reflection banks, 30 ADD-only learners, and Train-only evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from statistics import mean, stdev

from filelock import FileLock
from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace import evaluate_advantage_accumulation as evaluator  # noqa: E402
from scripts.ace import run_advantage_accumulation as advantage  # noqa: E402
from scripts.ace import run_reflection_accumulation as accumulation  # noqa: E402
from scripts.ace import run_training_first as training  # noqa: E402
from scripts.ace.schedule_nonthinking_replication import freeze_reflection, verify_sources  # noqa: E402

base = accumulation.base
MODES = ("thinking", "nonthinking")
ARMS = ("without_gt",)
SEEDS = (42, 43, 44)
WORKERS = 50
training.MODES = MODES
training.PHASES = ("train_final",)
SOURCE = ROOT / "outputs/ace_appworld_gpt_oss120b_add_only_20261121/high/reflection"


def worker_limit(output):
    workers = base.read(Path(output) / "plan.json").get("workers", WORKERS)
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError("Invalid shared queue worker limit")
    return workers


def settings_for(mode):
    if mode not in MODES:
        raise ValueError(mode)
    thinking = mode == "thinking"
    return {"temperature": 1.0, "top_p": 0.95, "max_tokens": None, "timeout": 600, "num_retries": 1,
            "extra_body": {"top_k": 64, "chat_template_kwargs": {
                "enable_thinking": thinking, "preserve_thinking": thinking,
            }}}


def stage_jobs(tasks, stage):
    return [j for j in base.jobs_for_stage(tasks, stage, 3) if stage == 1 or j["arm"] in ARMS]


def history_canary(template):
    """Verify the actual frozen template, including the tool-call exception across a new user turn."""
    messages = [
        {"role": "system", "content": "History formatting test."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "VISIBLE_FINAL_ANSWER", "reasoning": "DROP_ORDINARY_THOUGHT"},
        {"role": "user", "content": "Use the report tool."},
        {"role": "assistant", "content": None, "reasoning": "KEEP_TOOL_CALL_THOUGHT",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {
             "name": "report", "arguments": {"value": "hello"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "report", "content": "ok"},
        {"role": "assistant", "content": "VISIBLE_SECOND_FINAL", "reasoning": "DROP_FINISHED_THOUGHT"},
        {"role": "user", "content": "Next user turn."},
    ]
    rendered = Environment().from_string(template).render(
        messages=messages, tools=[], bos_token="<bos>", add_generation_prompt=True,
        enable_thinking=True, preserve_thinking=True,
    )
    assert "DROP_ORDINARY_THOUGHT" not in rendered and "DROP_FINISHED_THOUGHT" not in rendered
    assert "KEEP_TOOL_CALL_THOUGHT" in rendered
    assert "VISIBLE_FINAL_ANSWER" in rendered and "VISIBLE_SECOND_FINAL" in rendered
    return rendered


def install_tool_reasoning_history(reflection):
    """Derive a Gemma-only frozen actor; keep the other model cohorts' actor bytes untouched."""
    source = reflection / "source_snapshot/src/exgentic/agents/ace/ace_instance.py"
    original = source.read_text()
    anchor = """            tool_calls = self._extract_tool_calls(message)
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    tool_calls=["""
    replacement = """            tool_calls = self._extract_tool_calls(message)
            # Gemma retains thoughts on tool-call turns, while ordinary replies
            # keep final content only. The frozen template also enforces this.
            reasoning = {
                key: value for key in ("reasoning_content", "reasoning")
                if (value := message.get(key)) is not None
            }
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    **({"content": message.content} if message.content is not None else {}),
                    **reasoning,
                    tool_calls=["""
    if original.count(anchor) != 1:
        raise ValueError("Frozen ACE tool-history patch anchor differs")
    source.write_text(original.replace(anchor, replacement))
    manifest = base.read(reflection / "source_hashes.json")
    relative = str(source.relative_to(reflection / "source_snapshot"))
    original_sha = manifest[relative]
    manifest[relative] = training.digest(source)
    base.write(reflection / "source_hashes.json", manifest)
    base.write(reflection / "gemma_history_patch.json", {
        "file": relative, "before_sha256": original_sha, "after_sha256": manifest[relative],
        "scope": "Gemma frozen actor only; tool-call reasoning retention, ordinary replies unchanged",
    })


def prepare(output, model, revision):
    if (output / "plan.json").exists():
        plan = base.read(output / "plan.json")
        if plan["model"] != model or plan["model_revision"] != revision:
            raise ValueError("Prepared model/revision differs")
        return plan
    rendering = history_canary((output / "serving/chat_template.jinja").read_text())
    (output / "serving/history_template_canary.txt").write_text(rendering)
    for mode in MODES:
        reflection = output / mode / "reflection"
        reflection.parent.mkdir(exist_ok=True)
        freeze_reflection(SOURCE, reflection)
        install_tool_reasoning_history(reflection)
        protocol = base.read(reflection / "protocol.json")
        settings = settings_for(mode)
        base.write(reflection / "model_settings.json", settings)
        protocol.update(
            model=f"openai/{model}", model_revision=revision, arms=list(ARMS),
            generation_mode="thinking" if mode == "thinking" else "non_thinking",
            generation_settings_authority="User Gemma4 T=1, top_p=.95, top_k=64; w/o GT; Train only",
            model_settings=settings,
            model_settings_source_sha256=base.digest((reflection / "model_settings.json").read_bytes()),
            actor_trajectories_planned=1000, proposal_chains_planned=250,
            apply_policy_to=["shared_initial_rollouts", "without_gt_updated_context_rollouts"],
            history_policy="Strip ordinary past reasoning; retain tool-call reasoning; official pinned Gemma template",
            history_policy_version="gemma-tool-reasoning-history-v1",
        )
        for stage, count in ((1, 250), (2, 250), (3, 750)):
            jobs = stage_jobs(protocol["task_ids"], stage)
            assert len(jobs) == count
            base.write(reflection / f"stage{stage}_queue.json", jobs)
            spec = protocol["stages"][stage - 1]
            spec.update(total_jobs=count, max_workers=WORKERS)
            if stage > 1:
                spec["jobs_per_arm"] = {"without_gt": count}
        for key in ("gt_reference_source", "gt_reference_sha256"):
            protocol.pop(key, None)
        (reflection / "references.json").unlink()
        base.write(reflection / "protocol.json", protocol)
        accumulation.prepare(output / mode / "accumulation", reflection, global_concurrency=WORKERS,
                             execution_order="phase_major")
        target = output / mode / "accumulation"
        learned = base.read(target / "protocol.json")
        learned.update(training_contexts=9, planned_curator_steps=450, planned_evaluations=450,
                       evaluations_per_seed=150, actor_trajectories_planned=450,
                       apply_policy_to=["train_final"], evaluation_phases=["train_final"],
                       seed_phase_order=["curator_training", "train_final"], evaluation_workers=WORKERS,
                       phase_barrier_scope="both_modes_all_contexts")
        base.write(target / "protocol.json", learned)
    plan = {
        "model": model, "model_revision": revision, "modes": list(MODES), "arms": list(ARMS),
        "seeds": list(SEEDS), "playbook_edit_mode": "add_only", "physical_gpu": 1, "port": 8006,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "workers": WORKERS, "stage_barriers": "Both modes complete each stage before the next",
        "phase_order": ["initial_rollouts", "reflection_update", "updated_context_rollouts",
                        "all_context_training", "train_final"],
        "stage_queue_counts": [500, 500, 1500], "training_contexts": 30,
        "normal_contexts": 18, "advantage_contexts": 12, "train_evaluations": 1500,
        "actor_trajectories_planned": 3500, "evaluation_phases": ["train_final"],
        "queue_order": "Train: seed, mode, context, original task order",
        "history_policy": "No ordinary reasoning in past history; tool-call reasoning retained",
        "prepared_at": base.now(),
    }
    base.write(output / "plan.json", plan)
    pinned = {}
    for mode in MODES:
        for phase in ("reflection", "accumulation"):
            path = output / mode / phase
            verify_sources(path)
            for name in ("protocol.json", "source_hashes.json"):
                file = path / name
                pinned[str(file)] = training.digest(file)
    for name in ("run_gemma4_train_only.py", "run_training_first.py", "run_advantage_accumulation.py",
                 "evaluate_advantage_accumulation.py"):
        file = ROOT / "scripts/ace" / name
        pinned[str(file)] = training.digest(file)
    for name in ("serve.sh", "chat_template.jinja", "model_revision.txt"):
        file = output / "serving" / name
        pinned[str(file)] = training.digest(file)
    base.write(output / "input_hashes.json", pinned)
    return plan


def request(protocol, payload):
    settings = protocol["model_settings"]
    body = {"model": protocol["model"].removeprefix("openai/"), "temperature": 1.0, "top_p": 0.95,
            **settings["extra_body"], "max_tokens": 4096, **payload}
    req = urllib.request.Request(protocol["base_url"] + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return {"request": body, "response": json.load(response)}


def canaries(output):
    saved = output / "serving/generation_canaries.json"
    if saved.exists():
        return
    checks = {}
    for mode in MODES:
        protocol = base.read(output / mode / "reflection/protocol.json")
        base.configure_environment(protocol)
        endpoint = base.check_endpoint(protocol)
        base.write(output / "serving/endpoint_verification.json", endpoint)
        math = request(protocol, {"messages": [
            {"role": "user", "content": "Multiply 47 by 83. Give the final number."}]})
        message = math["response"]["choices"][0]["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if bool(reasoning) != (mode == "thinking") or not re.search(r"\b3,?901\b", message.get("content") or ""):
            raise ValueError(f"{mode}: thinking/answer canary failed: {message}")
        structured = request(protocol, {"messages": [{"role": "user", "content": 'Return only {"ok":true}.'}],
                                        "response_format": {"type": "json_object"}})
        assert json.loads(structured["response"]["choices"][0]["message"]["content"]) == {"ok": True}
        tool = request(protocol, {"messages": [{"role": "user", "content": "Call report with value hello."}],
                                 "tool_choice": "auto", "tools": [{"type": "function", "function": {
                                     "name": "report", "description": "Report a value.", "parameters": {
                                         "type": "object", "properties": {"value": {"type": "string"}},
                                         "required": ["value"]}}}]})
        calls = tool["response"]["choices"][0]["message"].get("tool_calls")
        assert calls and calls[0]["function"]["name"] == "report"
        assert json.loads(calls[0]["function"]["arguments"]) == {"value": "hello"}
        checks[mode] = {"math": math, "structured_json": structured, "native_tool_call": tool}
        base.write(output / f"serving/{mode}_canaries.json", checks[mode])
    base.write(saved, checks)


def emit(output, phase, **fields):
    training.emit(output, output / "control", phase, **fields)


def validate_reflection(item):
    directory, job = Path(item["root"]), item["job"]
    protocol = base.read(directory / "protocol.json")
    row = base.read(base.job_dir(directory, job) / "result.json")
    assert all(row.get(k) == v for k, v in job.items())
    assert row["appworld_policy_sha256"] == protocol["appworld_policy_sha256"]
    if job["stage"] == 2:
        assert job["arm"] == "without_gt"
        assert training.digest(Path(row["updated_context_path"])) == row["updated_context_sha256"]
        reflection = base.read(row["reflection_path"])
        assert reflection["gt_trajectory_sha256"] is None
    else:
        assert type(row["success"]) is bool
        accumulation.exact_reward(row)
        context = protocol["initial_playbook"]
        if job["stage"] == 3:
            parent = base.read(directory / "stage2" / job["context_id"] / "result.json")
            context = Path(parent["updated_context_path"]).read_text()
        assert row["initial_context_sha256"] == base.digest(context)
    return row


def reflection_one(item):
    path = base.job_dir(item["root"], item["job"])
    if (path / "result.json").exists():
        return validate_reflection(item)
    module = training.frozen(item["source"])
    limit = base.read(Path(item["root"]) / "protocol.json")["infrastructure_attempts"]
    attempts = len(list(path.glob("attempt_*")))
    response = {"error": "Saved infrastructure attempt limit reached"}
    while attempts < limit:
        attempts += 1
        response = module["worker"](item["root"], item["job"], attempts)
        if response["ok"]:
            return validate_reflection(item)
        if not training.retryable(response.get("error", "")):
            break
        time.sleep(3)
    raise RuntimeError(f"Reflection stage job failed: {item['job']['id']}: {response}")


def run_queue(output, phase, items, function, validator):
    workers = worker_limit(output)
    pending, done, active, errors = deque(), {}, {}, []
    for item in items:
        key = item["root"], item["job"]["id"]
        if training.result_path(item).exists():
            done[key] = validator(item)
        else:
            pending.append(item)
    reused = len(done)

    def report():
        emit(output, phase, completed=len(done), total=len(items), running=len(active), pending=len(pending),
             worker_limit=workers, reused=reused, errors=errors,
             by_mode={mode: {"completed": sum(item["sampler"] == mode and
                                              (item["root"], item["job"]["id"]) in done for item in items),
                             "total": sum(item["sampler"] == mode for item in items)} for mode in MODES})
        if phase == "train_final":
            grouped = defaultdict(list)
            for item in items:
                key = item["root"], item["job"]["id"]
                if key in done:
                    grouped[item["sampler"], item["job"]["condition_id"]].append(done[key])
            base.write(output / "control/partial_summary.json", {
                "/".join(key): {"tasks": len(rows), "pct": mean(r["pct"] for r in rows),
                                "tgc": mean(r["success"] for r in rows)} for key, rows in grouped.items()
            })

    report()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        last = time.monotonic()
        while pending or active:
            if (output / "STOP").exists():
                raise InterruptedError("STOP requested")
            while pending and len(active) < workers and not errors:
                item = pending.popleft()
                active[pool.submit(function, item)] = item
            finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                item = active.pop(future)
                try:
                    done[item["root"], item["job"]["id"]] = future.result()
                except Exception as exc:
                    errors.append({"job": item["job"]["id"], "root": item["root"], "error": repr(exc)})
            if time.monotonic() - last >= 30 or not active:
                report()
                last = time.monotonic()
            if errors and not active:
                break
    if errors or len(done) != len(items):
        raise RuntimeError(f"Incomplete {phase}: {errors[:3]}")
    return done


def reflection_stage(output, stage):
    queues = {}
    for mode in MODES:
        directory = output / mode / "reflection"
        queues[mode] = base.read(directory / f"stage{stage}_queue.json")
    items = [{"root": str(output / mode / "reflection"), "job": queues[mode][i], "sampler": mode,
              "source": str(output / mode / "reflection/source_snapshot/scripts/ace" /
                            "run_context_proposal_bottleneck.py")}
             for i in range(len(queues[MODES[0]])) for mode in MODES]
    base.write(output / "control" / f"stage{stage}_shared_queue.json", items)
    run_queue(output, ("initial_rollouts", "reflection_update", "updated_context_rollouts")[stage - 1],
              items, reflection_one, validate_reflection)
    for mode in MODES:
        directory = output / mode / "reflection"
        training.frozen(items[0]["source"])["summarize"](directory, stage)
        base.write(directory / "status.json", {"state": "completed" if stage == 3 else "running",
                                               "stage": stage, "updated_at": base.now()})


def learners(output):
    runs = {mode: output / mode / "accumulation" for mode in MODES}
    jobs = []
    for mode, directory in runs.items():
        source = str(directory / "source_snapshot/scripts/ace/run_reflection_accumulation.py")
        module = training.frozen(source)
        if not (directory / "bank_ready.json").exists():
            bank = module["freeze_bank"](directory)
            module["plan_selections"](directory, bank)
            base.write(directory / "bank_ready.json", {"contexts": 250, "evaluations": 750, "at": base.now()})
        for seed in SEEDS:
            for condition in accumulation.conditions(seed, ARMS):
                jobs.append({"path": str(directory), "condition": condition, "steps": 50,
                             "source": source, "sampler": mode, "kind": "full"})
    target = output / "advantage"
    target.mkdir(exist_ok=True)
    extras = base.read(target / "plan.json") if (target / "plan.json").exists() else advantage.prepare(target, runs)
    for job in extras:
        mode = Path(job["path"]).parent.name
        jobs.append({**job, "source": str(runs[mode] / "source_snapshot/scripts/ace/run_reflection_accumulation.py"),
                     "sampler": mode, "kind": "advantage"})
    assert len(jobs) == 30 and all(j["condition"]["arm"] == "without_gt" for j in jobs)
    base.write(output / "control/training_plan.json", jobs)
    return jobs


def evaluation_queue(output, jobs):
    training.require_all_trained(jobs)
    queue = []
    runs = {mode: output / mode / "accumulation" for mode in MODES}
    for mode, directory in runs.items():
        source = str(directory / "source_snapshot/scripts/ace/run_reflection_accumulation.py")
        module = training.frozen(source)
        for seed in SEEDS:
            batch = module["eval_jobs"](directory, seed, "train_final")
            base.write(directory / f"seed{seed}/train_final_queue.json", batch)
            queue.extend({"root": str(directory / f"seed{seed}"), "job": job, "source": source,
                          "sampler": mode, "kind": "full", "ordinal": i} for i, job in enumerate(batch))
    extras = base.read(output / "advantage/plan.json")
    for i, (root, job) in enumerate(evaluator.build_queue(output / "advantage", extras, "train_final", runs)):
        mode = job["sampler"]
        queue.append({"root": root, "job": job, "sampler": mode, "kind": "advantage", "ordinal": i,
                      "source": str(runs[mode] / "source_snapshot/scripts/ace/run_reflection_accumulation.py")})
    queue.sort(key=lambda r: (r["job"]["seed"], MODES.index(r["sampler"]),
                              r["job"]["condition_id"], r["ordinal"]))
    assert len(queue) == 1500 and all(r["job"]["phase"] == "train_final" for r in queue)
    assert all(r["job"]["arm"] == "without_gt" for r in queue)
    base.write(output / "control/evaluation_queue.json", queue)
    return queue


def summarize_train(output, queue, results):
    batches, contexts = defaultdict(list), defaultdict(list)
    for item in queue:
        row = results[item["root"], item["job"]["id"]]
        batches[item["root"]].append(row)
        contexts[item["sampler"], row["seed"], row["condition_id"]].append(row)
    for directory, rows in batches.items():
        base.write(Path(directory) / "train_final_results.json", rows)
    records = []
    for (mode, seed, cid), rows in contexts.items():
        assert len(rows) == len({r["task_id"] for r in rows}) == 50
        records.append({"model": base.read(output / "plan.json")["model"], "mode": mode, "seed": seed,
                        "condition_id": cid, "strategy": cid.split("without_gt_", 1)[1], "tasks": 50,
                        "pct": mean(r["pct"] for r in rows), "tgc": mean(r["success"] for r in rows)})
    assert len(records) == 30
    base.write(output / "train_final_by_seed.json", records)
    with (output / "train_final_by_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for mode in MODES:
        for strategy in ("best", "random", "worst", "best_advantage", "worst_advantage"):
            group = [r for r in records if r["mode"] == mode and r["strategy"] == strategy]
            assert len(group) == 3
            summary.append({"mode": mode, "strategy": strategy, "seeds": list(SEEDS),
                            **{f"{metric}_{stat}": function(r[metric] for r in group)
                               for metric in ("pct", "tgc") for stat, function in (("mean", mean), ("std", stdev))}})
    base.write(output / "train_final_mean_std.json", summary)


def run(output):
    for path, sha in base.read(output / "input_hashes.json").items():
        if training.digest(Path(path)) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    canaries(output)
    for stage in (1, 2, 3):
        reflection_stage(output, stage)
    jobs = learners(output)
    training.train_all(output, output / "control", jobs)
    queue = evaluation_queue(output, jobs)
    results = run_queue(output, "train_final", queue, training.evaluate_one, training.validate_result)
    summarize_train(output, queue, results)
    base.write(output / "completed.json", {"contexts": 30, "train_evaluations": 1500, "at": base.now()})
    emit(output, "train_only_complete", state="completed", completed=1500, total=1500, running=0, errors=[])
    accumulation.journal(output, "completed", "Gemma w/o GT의 Reflection 분석·30개 ADD Context·Train 평가 완료.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "control").mkdir(exist_ok=True)
    with FileLock(str(output / "controller.lock"), timeout=0):
        prepare(output, args.model, args.revision)
        if args.prepare_only:
            return
        base.write(output / "control/process.json", {"pid": os.getpid(), "started_at": base.now(), "command": sys.argv})
        accumulation.journal(output, "running",
                             f"Gemma w/o GT·Train-only 시작: 500/500/1500 공유 queue, 동시성{worker_limit(output)}.")
        try:
            run(output)
        except BaseException as exc:
            emit(output, "stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "error",
                 state="stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "error",
                 error=repr(exc))
            accumulation.journal(output, "blocked", f"Gemma 실험 중단: {exc}")
            raise


if __name__ == "__main__":
    main()
