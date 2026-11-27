# SPDX-License-Identifier: Apache-2.0
"""Frozen, sequential Gemma31 -> Gemma26 format-repair diagnostics on GPU 0.

Only saved format-error rollouts are selected, including previously successful
ones. Original results, contexts, requests and prompts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from statistics import mean

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file() and (parent / "src/exgentic").is_dir())
DEFAULT_OUTPUT = ROOT / "outputs/ace_appworld_gemma_format_repair_diagnostic_20261127"
ORIGINAL_31 = ROOT / "outputs/ace_appworld_gemma4_31b_it_wo_gt_train_only_20261124"
ORIGINAL_26 = ROOT / "outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"
AUDIT = ORIGINAL_31 / "analysis/accumulation_pattern_diagnosis_20261127"
REJECTION = "Code-generation mode accepts exactly one execute_python code block per agent turn."
MODELS = (("31b", ORIGINAL_31, AUDIT / "execution_format_cases.json", 52),
          ("26b", ORIGINAL_26, AUDIT / "gemma26_execution_format_cases.json", 115))
WORKERS = 30


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def emit(output, **fields):
    status = {"updated_at": now(), "pid": os.getpid(), **fields}
    write(Path(output) / "status.json", status)
    line = f"[{status['updated_at']}] " + " ".join(f"{k}={v}" for k, v in fields.items() if k != "by_condition")
    print(line, flush=True)
    with (Path(output) / "progress.log").open("a") as stream:
        stream.write(line + "\n")


def journal(output, name, status, message, metrics=None):
    """Best effort journal writes; save failed payloads without losing run results."""
    directory = output / name
    record_path = directory / "journal_record.json"
    if not record_path.exists():
        return
    helper = Path.home() / ".codex/skills/experiment-journal/scripts/journal.py"
    record_id = read(record_path)["id"]
    event = directory / "journal_events" / str(time.time_ns())
    event.mkdir(parents=True)
    try:
        record = json.loads(subprocess.check_output([sys.executable, str(helper), "show", record_id], text=True))
        patch = {"status": status, "summary": message}
        if metrics is not None:
            patch["metrics"] = metrics
        write(event / "patch.json", patch)
        subprocess.run([sys.executable, str(helper), "update", record_id, "--revision", str(record["revision"]),
                        "--file", str(event / "patch.json")], check=True, capture_output=True, text=True)
        (event / "progress.txt").write_text(message + "\n")
        subprocess.run([sys.executable, str(helper), "log", record_id, "--file", str(event / "progress.txt")],
                       check=True, capture_output=True, text=True)
    except Exception as exc:
        write(event / "recording_error.json", {"error": repr(exc), "message": message})


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        verify(output)
        return
    own = output / "diagnostic_source"
    own.mkdir(exist_ok=True)
    for filename in ("rerun_gemma_format_errors.py", "gemma_python_format_repair.py"):
        shutil.copy2(ROOT / "scripts/ace" / filename, own / filename)
    manifest = {"created_at": now(), "variant": "gemma-single-python-fence-v1", "models": [],
                "workers": WORKERS, "gpu_physical_id": 0, "input_hashes": {},
                "selection": "Only saved final-Train format-error rollouts; retain successes too.",
                "limitations": "Selected-case diagnostic; not an unbiased new full-Train estimate."}
    pins = manifest["input_hashes"]
    for path in own.glob("*.py"):
        pins[str(path)] = digest(path)
    for name, original, cases_path, expected in MODELS:
        cases = read(cases_path)
        targets = {(r["mode"], r["condition"], r["task_id"]): r for r in cases}
        if len(targets) != expected:
            raise ValueError(f"Unexpected selected-case count: {name}")
        pins[str(cases_path)] = digest(cases_path)
        original_queue_path = original / "control/evaluation_queue.json"
        pins[str(original_queue_path)] = digest(original_queue_path)
        queue = []
        sources = {}
        model_root = output / name
        for old in read(original_queue_path):
            job, mode = old["job"], old["sampler"]
            key = mode, job["condition_id"], job["task_id"]
            if key not in targets:
                continue
            original_root = Path(old["root"])
            old_result_path = original_root / "stage3" / job["id"] / "result.json"
            baseline = read(old_result_path)
            capture_path = Path(baseline["capture_path"])
            captured = read(capture_path)
            invalid_turns = sum(action["action"] != "execute_python" for action in captured["actions"])
            if not invalid_turns or invalid_turns != targets[key]["format_error_turns"]:
                raise ValueError(f"Saved invalid message-action evidence differs: {key}")
            if baseline["pct"] != targets[key]["pct"] or baseline["success"] != targets[key]["success"]:
                raise ValueError(f"Audit and final result differ: {key}")
            if mode not in sources:
                original_snapshot = Path(old["source"]).parents[2]
                source_hashes = original_snapshot.parent / "source_hashes.json"
                pins[str(source_hashes)] = digest(source_hashes)
                snapshot = model_root / "snapshots" / mode / "source_snapshot"
                for relative, expected_hash in read(source_hashes).items():
                    if digest(original_snapshot / relative) != expected_hash:
                        raise ValueError(f"Frozen original changed: {original_snapshot / relative}")
                shutil.copytree(original_snapshot, snapshot, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                for relative, expected_hash in read(source_hashes).items():
                    target = snapshot / relative
                    if digest(target) != expected_hash:
                        raise ValueError(f"Copied snapshot differs: {target}")
                    pins[str(target)] = expected_hash
                sources[mode] = str(snapshot / "scripts/ace/run_reflection_accumulation.py")
            evaluation = model_root / "evaluations" / mode / job["condition_id"]
            protocol_path = original_root / "protocol.json"
            protocol = read(protocol_path)
            pins[str(protocol_path)] = digest(protocol_path)
            policy_sha = hashlib.sha256(protocol["benchmark_kwargs"]["policy_override"].encode()).hexdigest()
            if policy_sha != protocol["appworld_policy_sha256"] or job["arm"] != "without_gt":
                raise ValueError("Policy or GT scope mismatch")
            if job["task_id"] not in protocol["task_ids"] or job["phase"] != "train_final":
                raise ValueError("Only original Train final-Context cases are allowed")
            protocol.update(gpu_physical_id=0, evaluation_workers=WORKERS, global_concurrency=WORKERS,
                            diagnostic_variant=manifest["variant"], diagnostic_parent=str(original_root),
                            diagnostic_sampling_and_prompt="unchanged", diagnostic_selected_cases=expected)
            write(evaluation / "protocol.json", protocol)
            pins[str(evaluation / "protocol.json")] = digest(evaluation / "protocol.json")
            parent_path = original_root / "stage2" / job["context_id"] / "result.json"
            parent = read(parent_path)
            context = Path(parent["updated_context_path"])
            if (digest(context) != parent["updated_context_sha256"]
                    or digest(context) != baseline["initial_context_sha256"]):
                raise ValueError("Frozen Context hash mismatch")
            write(evaluation / "stage2" / job["context_id"] / "result.json", parent)
            write(evaluation / "baseline" / f"{job['id']}.json", baseline)
            for path in (old_result_path, capture_path, parent_path, context,
                         evaluation / "stage2" / job["context_id"] / "result.json",
                         evaluation / "baseline" / f"{job['id']}.json"):
                pins[str(path)] = digest(path)
            queue.append({**old, "root": str(evaluation), "source": sources[mode],
                          "baseline_result_path": str(old_result_path), "original_case": targets[key],
                          "repair_source": str(own / "gemma_python_format_repair.py")})
        if len(queue) != expected or len({(r["sampler"], r["job"]["id"]) for r in queue}) != expected:
            raise ValueError("Selection did not match the exact saved case set")
        serving = model_root / "serving"
        serving.mkdir(parents=True, exist_ok=True)
        for filename in ("model_revision.txt", "chat_template.jinja"):
            shutil.copy2(original / "serving" / filename, serving / filename)
            pins[str(serving / filename)] = digest(serving / filename)
        script = (original / "serving/serve.sh").read_text()
        script = script.replace("export CUDA_VISIBLE_DEVICES=1", "export CUDA_VISIBLE_DEVICES=0")
        script = script.replace("--max-num-seqs 100", "--max-num-seqs 30")
        if "export CUDA_VISIBLE_DEVICES=0" not in script or "--max-num-seqs 30" not in script:
            raise ValueError("Serving script is not pinned to GPU 0 / concurrency 30")
        (serving / "serve.sh").write_text(script)
        pins[str(serving / "serve.sh")] = digest(serving / "serve.sh")
        write(model_root / "queue.json", queue)
        pins[str(model_root / "queue.json")] = digest(model_root / "queue.json")
        if len(sources) > 1:
            hashes = [read(original / mode / "accumulation/source_hashes.json") for mode in sources]
            if any(h != hashes[0] for h in hashes):
                raise ValueError("Cross-mode worker reuse requires identical frozen framework sources")
        manifest["models"].append({"name": name, "original": str(original), "count": len(queue),
                                   "by_mode": dict(Counter(r["sampler"] for r in queue))})
    write(output / "manifest.json", manifest)
    emit(output, state="planned", phase="prepared", total=167, completed=0, workers=WORKERS,
         sequence="31b:52 -> 26b:115", physical_gpu=0)


def verify(output):
    manifest = read(Path(output) / "manifest.json")
    for path, expected in manifest["input_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Pinned diagnostic input changed: {path}")
    return manifest


@cache
def frozen(source):
    module = runpy.run_path(source)
    import exgentic

    snapshot = Path(source).parents[2]
    if not Path(exgentic.__file__).resolve().is_relative_to(snapshot.parents[1]):
        raise ValueError(f"Actor was imported outside the diagnostic snapshot: {exgentic.__file__}")
    return module


def result_path(item):
    return Path(item["root"]) / "stage3" / item["job"]["id"] / "result.json"


def evaluate_one(item):
    module = frozen(item["source"])
    root, job = Path(item["root"]), item["job"]
    protocol = read(root / "protocol.json")
    if result_path(item).exists():
        return module["validate_evaluation"](root, job, protocol)
    from exgentic.agents.ace.ace_instance import ACEAgentInstance

    helper = runpy.run_path(item["repair_source"])
    original = helper["install"](ACEAgentInstance)
    try:
        attempt = max((int(p.name.removeprefix("attempt_")) for p in result_path(item).parent.glob("attempt_*")),
                      default=0)
        retryable = ("timeout", "connection", "serviceunavailable", "temporarily unavailable", "remoteprotocolerror")
        for _ in range(3):
            attempt += 1
            outcome = module["base"].worker(root, job, attempt)
            if outcome["ok"]:
                return module["validate_evaluation"](root, job, protocol)
            if not any(token in outcome.get("error", "").lower() for token in retryable):
                break
        raise RuntimeError(f"Unresolved infrastructure failure: {outcome}")
    finally:
        ACEAgentInstance._completion = original


def endpoint(protocol):
    with urllib.request.urlopen(protocol["base_url"] + "/models", timeout=5) as response:
        return json.load(response)


def serving_process(port="8006"):
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if b"serve" not in args or b"--port" not in args:
                continue
            if args[args.index(b"--port") + 1] != port.encode() or not any(b"vllm" in x for x in args):
                continue
            env = dict(x.split(b"=", 1) for x in (proc / "environ").read_bytes().split(b"\0") if b"=" in x)
            if env.get(b"CUDA_VISIBLE_DEVICES") != b"0":
                raise ValueError("Refuse to manage a server outside physical GPU 0")
            matches.append({"pid": int(proc.name), "args": [x.decode() for x in args if x],
                            "start_ticks": (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(matches) > 1:
        raise ValueError("More than one server owns the diagnostic port")
    return matches[0] if matches else None


def ensure_server(output, name, protocol):
    expected = protocol["model"].removeprefix("openai/")
    try:
        model_ids = {r["id"] for r in endpoint(protocol)["data"]}
    except (OSError, ValueError):
        model_ids = set()
    process = serving_process()
    if expected in model_ids:
        if process is None or expected not in process["args"] or protocol["model_revision"] not in process["args"]:
            raise ValueError("Endpoint process/revision is not the pinned model")
        args = process["args"]
        template = Path(args[args.index("--chat-template") + 1])
        if digest(template) != digest(output / name / "serving/chat_template.jinja"):
            raise ValueError("Live server template differs from the pinned source model template")
        if args[args.index("--max-num-seqs") + 1] != str(WORKERS):
            raise ValueError("Live endpoint is not configured for concurrency 30")
        write(output / name / "serving/verified_endpoint.json", {"models": model_ids and sorted(model_ids),
                                                                 "process": process, "checked_at": now()})
        return None
    if process is not None:
        if name != "26b" or "google/gemma-4-31B-it" not in process["args"]:
            raise ValueError("Refuse to replace an unrelated serving process")
        # A completed first cohort is required before touching the 31B server.
        if read(output / "31b/status.json")["state"] != "completed":
            raise ValueError("31B completion barrier remains closed")
        with urllib.request.urlopen("http://127.0.0.1:8006/metrics", timeout=10) as response:
            metrics = response.read().decode()
        for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            values = [float(line.split()[-1]) for line in metrics.splitlines()
                      if line.startswith(metric + "{") or line.startswith(metric + " ")]
            if not values or any(values):
                raise RuntimeError(f"Refuse model switch while endpoint has requests: {metric}={values}")
        write(output / "26b/serving/pre_switch_process.json", process)
        os.kill(process["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 180
        while serving_process() is not None:
            if time.monotonic() >= deadline:
                raise RuntimeError("31B server did not stop after SIGTERM")
            time.sleep(2)
        time.sleep(5)
    serving = output / name / "serving"
    with (serving / "vllm.log").open("a") as stream:
        process = subprocess.Popen(["bash", str(serving / "serve.sh")], cwd=ROOT,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(serving / "process.json", {"pid": process.pid, "started_at": now(), "model": expected,
                                      "gpu_physical_id": 0, "script_sha256": digest(serving / "serve.sh")})
    deadline, last = time.monotonic() + 1200, 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} vLLM exited with {process.returncode}; see {serving / 'vllm.log'}")
        try:
            if expected in {r["id"] for r in endpoint(protocol)["data"]}:
                return ensure_server(output, name, protocol)
        except (OSError, ValueError):
            pass
        if time.monotonic() - last >= 30:
            emit(output, state="running", phase=f"{name}_server_startup", server_pid=process.pid,
                 completed=52 if name == "26b" else 0, total=167)
            last = time.monotonic()
        time.sleep(5)
    raise TimeoutError("vLLM startup exceeded 20 minutes")


def summarize(model_root, queue, rows):
    comparisons = []
    for item, row in zip(queue, rows, strict=True):
        old = read(item["baseline_result_path"])
        capture = read(row["capture_path"])
        events_path = Path(row["session_dir"]) / "agent/format_repair.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        comparisons.append({"mode": item["sampler"], "condition_id": row["condition_id"], "seed": row["seed"],
                            "task_id": row["task_id"], "original_pct": old["pct"], "rerun_pct": row["pct"],
                            "original_tgc": old["success"], "rerun_tgc": row["success"],
                            "original_format_error_turns": item["original_case"]["format_error_turns"],
                            "rerun_format_error_turns": sum(a["action"] != "execute_python"
                                                            for a in capture["actions"]),
                            "rerun_observed_rejections": json.dumps(capture["observations"]).count(REJECTION),
                            "repaired_turns": sum(e["decision"] == "repaired" for e in events),
                            "unrepaired_decisions": dict(Counter(e["decision"] for e in events
                                                                if e["decision"] != "repaired")),
                            "turn_limit_reached": row.get("turn_limit_reached", False),
                            "initial_context_sha256": row["initial_context_sha256"],
                            "result_path": str(result_path(item))})
    def aggregate(group):
        return {"count": len(group), "original_pct": mean(r["original_pct"] for r in group),
                "rerun_pct": mean(r["rerun_pct"] for r in group),
                "original_tgc": mean(r["original_tgc"] for r in group),
                "rerun_tgc": mean(r["rerun_tgc"] for r in group),
                "repaired_turns": sum(r["repaired_turns"] for r in group),
                "remaining_format_error_rollouts": sum(r["rerun_format_error_turns"] > 0 for r in group),
                "success_gains": sum(not r["original_tgc"] and r["rerun_tgc"] for r in group),
                "success_losses": sum(r["original_tgc"] and not r["rerun_tgc"] for r in group)}
    summary = {"selected_cases": aggregate(comparisons), "by_condition": {
        "/".join(key): aggregate([r for r in comparisons if (r["mode"], r["condition_id"]) == key])
        for key in sorted({(r["mode"], r["condition_id"]) for r in comparisons})},
        "limitations": "Selected-case diagnostic, stochastic reruns, unchanged prompt and Context. "
        "Not a full benchmark rescore or a pure causal estimate of format handling."}
    write(model_root / "comparison_rows.json", comparisons)
    write(model_root / "summary.json", summary)
    return summary


def run_cohort(output, name):
    model_root = output / name
    queue = read(model_root / "queue.json")
    protocol = read(Path(queue[0]["root"]) / "protocol.json")
    if (model_root / "status.json").exists() and read(model_root / "status.json")["state"] == "completed":
        if all(result_path(item).exists() for item in queue) and (model_root / "summary.json").exists():
            return
        raise ValueError("Completed marker without all selected results")
    emit(output, state="running", phase=f"{name}_endpoint_check", completed=52 if name == "26b" else 0, total=167)
    ensure_server(output, name, protocol)
    context = multiprocessing.get_context("spawn")
    active, rows, errors = {}, {}, []
    pending = list(queue)
    last = 0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        journal(output, name, "running", f"GPU 0에서 {name} 형식 오류 {len(queue)}건 진단 재평가 시작. "
                "동시성 30. 기존 Context·프롬프트·sampling 유지, Python fence 정규표현식 보정만 적용. "
                f"PID={os.getpid()}")
        while pending or active:
            if (output / "STOP").exists():
                raise InterruptedError("Diagnostic STOP requested")
            while pending and len(active) < WORKERS:
                item = pending.pop(0)
                active[pool.submit(evaluate_one, item)] = item
            finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                item = active.pop(future)
                key = item["sampler"], item["job"]["id"]
                try:
                    row = future.result()
                    if row["initial_context_sha256"] != read(item["baseline_result_path"])["initial_context_sha256"]:
                        raise ValueError("Diagnostic did not use the exact original Context")
                    rows[key] = row
                except Exception as exc:
                    errors.append({"item": item, "error": repr(exc)})
            if time.monotonic() - last >= 30 or not (pending or active):
                state = "running" if pending or active else "blocked" if errors else "validating"
                by_condition = dict(Counter(f"{mode}/{r['condition_id']}" for (mode, _), r in rows.items()))
                emit(model_root, state=state, phase="regex_repair_train_diagnostic", completed=len(rows),
                     total=len(queue), active=len(active), pending=len(pending), errors=len(errors),
                     workers=WORKERS, by_condition=by_condition)
                emit(output, state=state, phase=f"{name}_evaluation",
                     completed=len(rows) + (52 if name == "26b" else 0),
                     total=167, model_completed=len(rows), model_total=len(queue),
                     active=len(active), pending=len(pending), errors=len(errors), workers=WORKERS)
                last = time.monotonic()
    write(model_root / "errors.json", errors)
    if errors or len(rows) != len(queue):
        raise RuntimeError(f"{name}: {len(errors)} unresolved cases; next model remains blocked")
    ordered = [rows[item["sampler"], item["job"]["id"]] for item in queue]
    write(model_root / "results.json", ordered)
    summary = summarize(model_root, queue, ordered)
    emit(model_root, state="completed", phase="regex_repair_train_diagnostic", completed=len(queue), total=len(queue),
         summary=summary["selected_cases"])
    scores = summary["selected_cases"]
    metrics = [{"name": key, "value": scores[key] * 100, "unit": "%", "split": "selected_format_error_train_cases",
                "source": str(model_root / "summary.json")} for key in
               ("original_pct", "rerun_pct", "original_tgc", "rerun_tgc")]
    journal(output, name, "completed", f"선정된 {len(queue)}건 재평가 완료. "
            f"PCT {scores['original_pct'] * 100:.2f} → {scores['rerun_pct'] * 100:.2f}, "
            f"TGC {scores['original_tgc'] * 100:.2f} → {scores['rerun_tgc'] * 100:.2f}. "
            f"코드 변환 {scores['repaired_turns']}턴, 남은 형식 오류 {scores['remaining_format_error_rollouts']}건. "
            "오류가 있었던 사례만 재평가한 진단이며 전체 Train 성능 추정치는 아님.", metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output)
    if args.run:
        from filelock import FileLock

        with FileLock(str(output / "controller.lock"), timeout=0):
            verify(output)
            try:
                for name, *_ in MODELS:
                    run_cohort(output, name)
                    verify(output)
                emit(output, state="completed", phase="both_model_diagnostics_complete", completed=167, total=167)
            except BaseException as exc:
                emit(output, state="blocked", phase="diagnostic_interrupted", error=repr(exc))
                journal(output, name, "blocked", f"형식 보정 진단 중단: {exc}")
                raise


if __name__ == "__main__":
    main()
