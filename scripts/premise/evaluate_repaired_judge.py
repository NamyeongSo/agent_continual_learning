# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Paired validation of frozen judge inputs before and after trajectory repairs."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise import rubric_prompts
from exgentic.agents.premise.reward_data import BENCHMARKS, digest, write_json
from scripts.premise.analyze_judge_specificity import decision_metrics, paired_difference

PUBLIC_KEYS = {"task", "context", "public_tools", "common_prefix", "trajectory_a", "trajectory_b"}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "trajectory_choice",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    },
}


def validation_inputs(original: Path, repaired: Path):
    manifests = {name: json.loads((path / "manifest.json").read_text()) for name, path in (
        ("original", original), ("repaired", repaired)
    )}
    maps = {name: {r["id"]: r for r in m["records"]} for name, m in manifests.items()}
    if set(maps["original"]) != set(maps["repaired"]):
        raise ValueError("Repair changed the dataset pair IDs")
    for pair_id, record in maps["original"].items():
        if any(record[k] != maps["repaired"][pair_id][k] for k in (
            "benchmark", "task_id", "split", "label", "a_is_original", "provenance"
        )):
            raise ValueError("Repair changed labels, splits or source score provenance")
    records = sorted((r for r in maps["original"].values() if r["split"] == "val"), key=lambda r: r["id"])
    visible = {}
    for name, path in (("original", original), ("repaired", repaired)):
        visible[name] = {}
        for record in records:
            value = json.loads((path / "pairs" / f"{record['id']}.json").read_text())
            if set(value) != PUBLIC_KEYS or digest(value) != maps[name][record["id"]]["input_sha256"]:
                raise ValueError("Frozen input contract or hash mismatch")
            visible[name][record["id"]] = value
    return records, visible, manifests


def build_queue(records, visible, rubrics):
    """Deduplicate identical requests while preserving every evaluation condition."""
    jobs = {}
    for record in records:
        for variant in ("original", "repaired"):
            for rubric, rules in rubrics.items():
                for swapped in (False, True):
                    value = copy.deepcopy(visible[variant][record["id"]])
                    if swapped:
                        value["trajectory_a"], value["trajectory_b"] = value["trajectory_b"], value["trajectory_a"]
                    messages = rubric_prompts.messages(rubric_prompts.JUDGE, {"comparison": value, "criteria": rules})
                    key = digest(messages)
                    job = jobs.setdefault(key, {"request_id": key, "messages": messages, "cases": []})
                    job["cases"].append({
                        "variant": variant, "rubric": rubric, "record_id": record["id"],
                        "benchmark": record["benchmark"], "task_id": record["task_id"], "swapped": swapped,
                    })
    return sorted(jobs.values(), key=lambda job: digest([42, job["request_id"]]))


def score_cases(job, response, labels):
    rows = []
    for case in job["cases"]:
        row = {**case, "request_id": job["request_id"], "status": response["status"]}
        if response["status"] == "completed":
            raw = response["choice"]
            canonical = ("B" if raw == "A" else "A") if case["swapped"] else raw
            row.update(choice=canonical, raw_choice=raw, reward=int(canonical == labels[case["record_id"]]))
        else:
            row.update(error_type=response["error_type"], error_message=response["error_message"])
        rows.append(row)
    return rows


def report(root, rows, config):
    metrics, contrasts = {}, {}
    lines = ["# 수정 전·후 judge validation", "",
             f"같은 {config['validation_pairs']}쌍, A/B 양쪽 순서, 같은 rubric과 JSON choice decoder로 비교했다.",
             "재학습 없이 공개 입력만 교체했다. 전체 입력 수정에는 API 인자 스키마, "
             "복합 호출 인자 및 실제 replay prefix feedback 복원이 포함된다.",
             "", "| Rubric | Benchmark | 수정 전 | 수정 후 | 차이(pp) | 순서 변경 시 선택 뒤집힘: 전 → 후 |",
             "|---|---|---:|---:|---:|---:|"]
    for rubric in ("no_rubric", "discovered"):
        metrics[rubric], contrasts[rubric] = {}, {}
        for bench in (*BENCHMARKS, "ALL"):
            subsets = {variant: [r for r in rows if r["rubric"] == rubric and r["variant"] == variant
                                  and (bench == "ALL" or r["benchmark"] == bench)]
                       for variant in ("original", "repaired")}
            metrics[rubric][bench] = {v: decision_metrics(s) for v, s in subsets.items()}
            contrasts[rubric][bench] = paired_difference(subsets["original"], subsets["repaired"])
            a, b = (metrics[rubric][bench][v] for v in ("original", "repaired"))
            def pct(value):
                return "오류로 미집계" if value is None else f"{value * 100:.2f}%"
            delta = None if a["accuracy"] is None or b["accuracy"] is None else (b["accuracy"] - a["accuracy"]) * 100
            delta_text = "미집계" if delta is None else f"{delta:+.2f}"
            lines.append(f"| {rubric} | {bench} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {delta_text} | "
                         f"{pct(a['flip_rate'])} → {pct(b['flip_rate'])} |")
    errors = sum(r["status"] != "completed" for r in rows)
    summary = {"metrics": metrics, "paired_contrasts": contrasts, "errors": errors,
               "evaluation_rows": len(rows), "unique_requests": config["unique_requests"],
               "limitations": "Reused development validation, 5/5/4 task groups; no retraining. "
               "Both conditions use the same strict JSON decoder, unlike historical json_object runs. "
               "Input repairs are bundled; their individual effects are not isolated. "
               "Identical Terminal inputs reuse the same request across original/repaired."}
    write_json(root / "summary.json", summary)
    lines += ["", "정확도는 각 쌍의 두 순서 판정 평균이다. 독립 과제 수는 summary.json에 있다.",
              "전체 평균은 validation 쌍 비중으로 가중했다. "
              "task 평균 및 task-group bootstrap 구간은 summary.json에 있다.",
              "동일한 Terminal 입력은 수정 전·후에 같은 실제 요청을 재사용했다. "
              "오류가 있는 조건의 전체 정확도는 미집계하고 오류를 별도 보고한다.",
              "이번 비교는 양쪽 모두 동일한 json_schema를 사용했다. "
              "과거 json_object 형식의 결과와 직접 비교하지 않는다.",
              f"전체 판정 {len(rows)}개, 실제 고유 요청 {config['unique_requests']}개, 오류 판정 {errors}개."]
    (root / "report_ko.md").write_text("\n".join(lines) + "\n")
    return summary


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records, visible, manifests = validation_inputs(args.original_dataset, args.repaired_dataset)
    rules_path = args.rubric_source / "rubrics/discovered.json"
    rules = [{k: r[k] for k in ("id", "name", "description")} for r in json.loads(rules_path.read_text())]
    jobs = build_queue(records, visible, {"no_rubric": [], "discovered": rules})
    source_config = json.loads((args.rubric_source / "config.json").read_text())
    config = {
        "base_url": source_config["base_url"], "model": source_config["model"], "temperature": 0,
        "max_tokens": 2048, "response_format": RESPONSE_FORMAT, "enable_thinking": False,
        "workers": args.workers, "timeout_seconds": args.timeout, "request_retries": 1, "output_retries": 5,
        "datasets": {
            "original": str(args.original_dataset.resolve()), "repaired": str(args.repaired_dataset.resolve())
        },
        "dataset_sha256": {v: digest(m) for v, m in manifests.items()}, "rules_sha256": digest(rules),
        "rule_count": len(rules), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(Path(rubric_prompts.__file__).read_bytes()).hexdigest(),
        "unique_requests": len(jobs), "evaluation_rows": sum(len(j["cases"]) for j in jobs),
        "validation_pairs": len(records), "split_counts": dict(Counter(r["benchmark"] for r in records)),
        "protocol": "Frozen validation only. No training/selection; no labels or scores in judge messages.",
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing evaluation configuration differs")
    write_json(config_path, config)
    shutil.copy2(Path(__file__), root / "source_snapshot.py")
    shutil.copy2(Path(rubric_prompts.__file__), root / "prompt_snapshot.py")
    write_json(root / "rubric.json", rules)
    write_json(root / "queue_manifest.json", [
        {"request_id": j["request_id"], "cases": j["cases"]} for j in jobs
    ])
    labels = {r["id"]: r["label"] for r in records}
    local, clients, client_lock = threading.local(), [], threading.Lock()
    progress = {"pid": os.getpid(), "status": "running", "processed_requests": 0,
                "total_requests": len(jobs), "total_decisions": config["evaluation_rows"], "errors": 0}

    def status():
        write_json(root / "status.json", {**progress, "updated_at": datetime.now(UTC).isoformat()})

    def request(job):
        path = root / "calls" / f"{job['request_id']}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["status"] == "completed" or (cached["status"] == "error" and not args.retry_errors):
                return cached
            write_json(root / "previous_errors" / path.name, cached)
        if not hasattr(local, "client"):
            local.client = OpenAI(base_url=config["base_url"], api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                                  timeout=args.timeout, max_retries=0)
            with client_lock:
                clients.append(local.client)
        trace = {"request_id": job["request_id"], "status": "requesting", "started_at": datetime.now(UTC).isoformat(),
                 "request": {"messages": job["messages"], "model": config["model"], "temperature": 0,
                             "max_tokens": 2048, "response_format": RESPONSE_FORMAT,
                             "chat_template_kwargs": {"enable_thinking": False}}, "attempts": []}
        write_json(path, trace)
        started, failures, truncations = time.monotonic(), 0, 0
        while True:
            call_started = time.monotonic()
            try:
                result = local.client.chat.completions.create(
                    model=config["model"], messages=job["messages"], temperature=0, max_tokens=2048,
                    response_format=RESPONSE_FORMAT, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = result.choices[0]
                trace["attempts"].append({"finish_reason": choice.finish_reason, "content": choice.message.content,
                                          "usage": result.usage.model_dump() if result.usage else None,
                                          "seconds": time.monotonic() - call_started})
                if choice.finish_reason == "length" and truncations < 5:
                    truncations += 1
                    write_json(path, trace)
                    continue
                value = json.loads(choice.message.content or "")
                if choice.finish_reason != "stop" or set(value) != {"choice"} or value["choice"] not in ("A", "B"):
                    raise ValueError("Invalid choice response or exhausted output limit")
                trace.update(status="completed", choice=value["choice"])
                break
            except Exception as exc:
                trace["attempts"].append({"error_type": type(exc).__name__, "error_message": str(exc),
                                          "seconds": time.monotonic() - call_started})
                retryable = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
                    isinstance(exc, APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
                )
                if retryable and failures < 1:
                    failures += 1
                    write_json(path, trace)
                    time.sleep(1)
                    continue
                trace.update(status="error", error_type=type(exc).__name__, error_message=str(exc))
                break
        trace.update(seconds=time.monotonic() - started, request_retries=failures, output_retries=truncations)
        write_json(path, trace)
        return trace

    status()
    started, rows = time.monotonic(), []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(request, job): job for job in jobs}
            for future in as_completed(futures):
                response = future.result()
                rows.extend(score_cases(futures[future], response, labels))
                progress["processed_requests"] += 1
                progress["errors"] += int(response["status"] != "completed")
                progress["processed_decisions"] = len(rows)
                progress["elapsed_seconds"] = round(time.monotonic() - started, 2)
                status()
                if progress["processed_requests"] % 50 == 0:
                    print(json.dumps(progress), flush=True)
    finally:
        for client in clients:
            client.close()
    write_json(root / "results.json", rows)
    summary = report(root, rows, config)
    progress.update(status="completed_with_errors" if summary["errors"] else "completed", stage="finished")
    status()
    print(json.dumps({"status": progress["status"], "errors": summary["errors"], "report": str(root / "report_ko.md")}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--repaired-dataset", type=Path, required=True)
    parser.add_argument("--rubric-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("Workers and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except BaseException as exc:
            write_json(args.output / "status.json", {"pid": os.getpid(), "status": "error",
                       "error_type": type(exc).__name__, "error_message": str(exc)})
            raise


if __name__ == "__main__":
    main()
