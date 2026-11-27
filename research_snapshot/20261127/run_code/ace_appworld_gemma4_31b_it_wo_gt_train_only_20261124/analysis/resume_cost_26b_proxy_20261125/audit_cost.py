# SPDX-License-Identifier: Apache-2.0
"""Audit saved Gemma26 API usage as a task-matched proxy for remaining Gemma31 work."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

OUT = Path(__file__).resolve().parent
RUN31 = OUT.parents[1]
RUN26 = RUN31.parent / "ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
TOKENIZER_PATH = (Path.home() / ".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots"
                  / REVISION / "tokenizer.json")
TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
SOURCE_HASHES = {}
PROMPT_COUNTS = {}
UNKNOWN_CALLS = []
SEEN_RESPONSES = set()
DUPLICATE_IDS = []
CACHE_EVIDENCE = json.loads((OUT / "curator_local_cache_evidence.json").read_bytes())
CACHED_PATHS = {str(Path(row["path"]).resolve()) for row in CACHE_EVIDENCE
                if row["verified_local_cache_replay"]}
CALL_FIELDS = ["phase", "mode", "job", "path", "sha256", "has_usage", "input_tokens", "output_tokens",
               "reasoning_text_tokens_approx", "replayed_tool_reasoning_tokens_approx", "error_type", "response_id",
               "verified_local_response_cache"]


def read(path):
    raw = path.read_bytes()
    SOURCE_HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


@lru_cache(maxsize=65536)
def text_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)


def reasoning(message):
    provider = message.get("provider_specific_fields") or {}
    for field in (message.get("reasoning_content"), message.get("reasoning"),
                  provider.get("reasoning_content"), provider.get("reasoning")):
        if isinstance(field, str) and field:
            return field
    return ""


def report_tokens(row):
    value = row["cost_reports"]["agent"]
    return {name: int(re.search(r"\b" + name + r"=(\d+)\b", value)[1])
            for name in ("input_tokens", "output_tokens")}


def dollars(inputs, outputs):
    return float((Decimal(inputs) * Decimal("0.10") + Decimal(outputs) * Decimal("0.34")) / Decimal(1000000))


def fingerprint(request):
    fields = {k: request.get(k) for k in ("model", "messages", "tools", "extra_body", "chat_template_kwargs")}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def audit_call(path, phase, mode, job, writer):
    raw = path.read_bytes()
    entry = json.loads(raw)
    response, request = entry.get("response") or {}, entry.get("request") or {}
    usage = response.get("usage")
    counts = Counter(calls=1, input_tokens=0, output_tokens=0)
    replay = 0
    for message in request.get("messages", []):
        if not isinstance(message, dict):
            continue
        text = reasoning(message)
        if text:
            if message.get("tool_calls"):
                replay += text_tokens(text)
                counts["history_tool_reasoning_messages"] += 1
            else:
                counts["history_non_tool_reasoning_messages"] += 1
    counts["replayed_tool_reasoning_tokens_approx"] += replay
    kwargs = (request.get("extra_body") or {}).get("chat_template_kwargs") or request.get("chat_template_kwargs") or {}
    if kwargs.get("enable_thinking") is not (mode == "thinking"):
        counts["generation_mode_mismatch_or_missing"] += 1
    fp = fingerprint(request)
    reason_tokens, error_type = 0, ""
    if usage:
        inputs, outputs = usage["prompt_tokens"], usage["completion_tokens"]
        assert isinstance(inputs, int) and isinstance(outputs, int) and min(inputs, outputs) >= 0
        assert usage["total_tokens"] == inputs + outputs
        counts.update(input_tokens=inputs, output_tokens=outputs, calls_with_usage=1)
        if str(path.resolve()) in CACHED_PATHS:
            counts.update(local_cached_calls=1, local_cached_input_tokens=inputs,
                          local_cached_output_tokens=outputs)
        details = usage.get("completion_tokens_details") or {}
        if details.get("reasoning_tokens") is not None:
            counts["reasoning_tokens_explicit"] += details["reasoning_tokens"]
            counts["calls_with_explicit_reasoning_count"] += 1
        else:
            counts["calls_without_reasoning_breakdown"] += 1
        prompt_details = usage.get("prompt_tokens_details") or {}
        counts["cached_input_tokens_explicit"] += prompt_details.get("cached_tokens") or 0
        counts["calls_with_prompt_token_details"] += bool(prompt_details)
        texts = [reasoning(choice.get("message") or {}) for choice in response.get("choices", [])]
        reason_tokens = sum(text_tokens(text) for text in texts if text)
        counts["reasoning_text_tokens_approx"] += reason_tokens
        counts["calls_with_reasoning_text"] += any(texts)
        if fp in PROMPT_COUNTS:
            assert PROMPT_COUNTS[fp] == inputs, "Identical prompt fingerprint has different API token count"
        PROMPT_COUNTS[fp] = inputs
        response_id = response.get("id")
        if response_id in SEEN_RESPONSES:
            DUPLICATE_IDS.append({"path": str(path), "id": response_id})
        if response_id:
            SEEN_RESPONSES.add(response_id)
    else:
        counts["calls_without_usage"] += 1
        error = (entry.get("error") or "missing_usage").lower()
        error_type = "timeout" if "timeout" in error or "timed out" in error else (
            "context_length" if "context length" in error else "other")
        counts["unmeasured_" + error_type] += 1
        UNKNOWN_CALLS.append({"phase": phase, "mode": mode, "job": job, "path": str(path),
                              "prompt_fingerprint": fp, "error_type": error_type,
                              "num_retries": request.get("num_retries"), "seconds": entry.get("seconds")})
    writer.writerow({"phase": phase, "mode": mode, "job": job, "path": str(path),
                     "sha256": hashlib.sha256(raw).hexdigest(), "has_usage": bool(usage),
                     "input_tokens": counts["input_tokens"], "output_tokens": counts["output_tokens"],
                     "reasoning_text_tokens_approx": reason_tokens,
                     "replayed_tool_reasoning_tokens_approx": replay, "error_type": error_type,
                     "response_id": response.get("id", ""),
                     "verified_local_response_cache": str(path.resolve()) in CACHED_PATHS})
    return counts


def main():
    paused = read(RUN31 / "status.json")
    assert paused["state"] == "paused" and (RUN31 / "STOP").exists()
    jobs = []
    for mode in ("thinking", "nonthinking"):
        for job in read(RUN31 / mode / "reflection/stage3_queue.json"):
            if not (RUN31 / mode / "reflection/stage3" / job["id"] / "result.json").exists():
                folder = RUN26 / mode / "reflection/stage3" / job["id"]
                jobs.append(("candidate_remaining", mode, job["id"], folder,
                             report_tokens(read(folder / "result.json"))))
    assert len(jobs) == 722
    for item in read(RUN26 / "control/evaluation_queue.json"):
        folder = Path(item["root"]) / "stage3" / item["job"]["id"]
        jobs.append(("train_final", item["sampler"], item["job"]["id"], folder,
                     report_tokens(read(folder / "result.json"))))
    assert len(jobs) == 2222
    phases, job_rows, mismatches = defaultdict(Counter), [], []
    with (OUT / "api_call_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CALL_FIELDS)
        writer.writeheader()
        for ordinal, (phase, mode, name, folder, reported) in enumerate(jobs, 1):
            usage, chosen_usage = Counter(), Counter()
            result = read(folder / "result.json")
            chosen = f"attempt_{result['attempt']}"
            for path in sorted(folder.glob("attempt_*/llm_calls/*.json")):
                count = audit_call(path, phase, mode, name, writer)
                usage.update(count)
                if path.parents[1].name == chosen:
                    chosen_usage.update(count)
            row = {"phase": phase, "mode": mode, "job": name, "proxy_source": str(folder),
                   "attempt_directories": len(list(folder.glob("attempt_*"))),
                   "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                   "reported_input_tokens": reported["input_tokens"],
                   "reported_output_tokens": reported["output_tokens"],
                   "extra_attempt_input_tokens": usage["input_tokens"] - chosen_usage["input_tokens"],
                   "extra_attempt_output_tokens": usage["output_tokens"] - chosen_usage["output_tokens"],
                   "calls": usage["calls"], "calls_without_usage": usage["calls_without_usage"]}
            if any(chosen_usage[k] != reported[k] for k in ("input_tokens", "output_tokens")):
                mismatches.append({**row, "chosen_attempt_input_tokens": chosen_usage["input_tokens"],
                                   "chosen_attempt_output_tokens": chosen_usage["output_tokens"]})
            usage["units"] = 1
            phases[phase, mode].update(usage)
            job_rows.append(row)
            if ordinal % 250 == 0:
                print(json.dumps({"audited_rollout_jobs": ordinal, "total": len(jobs)}), flush=True)
        for job in read(RUN26 / "control/training_plan.json"):
            phase, mode = "context_training", job["sampler"]
            folder = Path(job["path"]) / "conditions" / job["condition"]["id"]
            usage = Counter()
            for path in sorted((folder / "calls").glob("*.json")):
                usage.update(audit_call(path, phase, mode, job["condition"]["id"], writer))
            checkpoint = read(folder / "checkpoints" / f"{job['steps']:03d}.json")
            assert all(usage[k] == checkpoint["cost_after"][k] for k in ("input_tokens", "output_tokens"))
            assert usage["calls_with_usage"] == job["steps"]
            usage["units"] = job["steps"]
            phases[phase, mode].update(usage)
    unknown = defaultdict(Counter)
    for row in UNKNOWN_CALLS:
        matched = PROMPT_COUNTS.get(row["prompt_fingerprint"])
        row["matched_successful_prompt_tokens"] = matched
        if matched is not None:
            unknown[row["phase"], row["mode"]].update(matched_error_calls=1, matched_error_prompt_tokens=matched)
        else:
            unknown[row["phase"], row["mode"]]["unmatched_error_calls"] += 1
    rows = []
    for (phase, mode), counts in phases.items():
        inputs, outputs = counts["input_tokens"], counts["output_tokens"]
        billable_inputs = inputs - counts["local_cached_input_tokens"]
        billable_outputs = outputs - counts["local_cached_output_tokens"]
        row = {"phase": phase, "mode": mode, **dict(counts), **dict(unknown[phase, mode]),
               "input_usd": dollars(inputs, 0), "output_usd": dollars(0, outputs),
               "known_usage_usd": dollars(inputs, outputs),
               "cache_adjusted_input_tokens": billable_inputs,
               "cache_adjusted_output_tokens": billable_outputs,
               "cache_adjusted_known_usage_usd": dollars(billable_inputs, billable_outputs)}
        rows.append(row)
    totals = sum(phases.values(), Counter())
    assert totals["local_cached_calls"] == 30
    cache_adjusted_inputs = totals["input_tokens"] - totals["local_cached_input_tokens"]
    cache_adjusted_outputs = totals["output_tokens"] - totals["local_cached_output_tokens"]
    SOURCE_HASHES[str(OUT / "curator_local_cache_evidence.json")] = hashlib.sha256(
        (OUT / "curator_local_cache_evidence.json").read_bytes()).hexdigest()
    report = {"generated_at": datetime.now(UTC).isoformat(), "remaining_candidate_rollouts": 722,
              "remaining_train_rollouts": 1500, "remaining_contexts": 30, "proxy_curator_calls": 1410,
              "prices_usd_per_million": {"input": 0.10, "output_including_reasoning": 0.34},
              "token_proxy": "Gemma26 only; task/context-index-matched 722 unfinished candidate jobs, "
                             "all1500 final Train jobs and all30 trained Contexts",
              "phases": rows, "totals": dict(totals),
              "known_usage_usd": dollars(totals["input_tokens"], totals["output_tokens"]),
              "input_usd": dollars(totals["input_tokens"], 0),
              "output_usd": dollars(0, totals["output_tokens"]),
              "local_response_cache_adjustment": {
                  "calls": totals["local_cached_calls"], "input_tokens": totals["local_cached_input_tokens"],
                  "output_tokens": totals["local_cached_output_tokens"],
                  "excluded_usd": dollars(totals["local_cached_input_tokens"], totals["local_cached_output_tokens"]),
                  "basis": "30 first-curator calls reuse saved stage2 response IDs, creation times, messages "
                           "and usage; normalized requests match; runtimes are under1 second. Independent actor calls "
                           "are never deduplicated."},
              "cache_adjusted_input_tokens": cache_adjusted_inputs,
              "cache_adjusted_output_tokens": cache_adjusted_outputs,
              "cache_adjusted_known_usage_usd": dollars(cache_adjusted_inputs, cache_adjusted_outputs),
              "error_input_imputation": {"matched_calls": sum(r.get("matched_error_calls", 0) for r in rows),
                  "input_tokens": sum(r.get("matched_error_prompt_tokens", 0) for r in rows),
                  "usd_if_each_error_call_bills_one_prompt": dollars(
                      sum(r.get("matched_error_prompt_tokens", 0) for r in rows), 0),
                  "error_completion_tokens": "Unknown; no API usage was returned; not silently set to zero billing"},
              "report_vs_chosen_attempt_mismatch_jobs": mismatches,
              "duplicate_response_ids": DUPLICATE_IDS,
              "reasoning_split": "No separate reasoning usage field when null. Reasoning text and replay counts "
                                 "are approximate re-tokenization of parsed strings, never added to API totals.",
              "input_accounting": "API prompt_tokens includes full rendered history, tool schema and current "
                                   "playbook, including preserved tool-call reasoning; no manual replay surcharge.",
              "cache_pricing": "No discounted cached-input rate supplied; full input count charged at0.10/M",
              "limitations": ["Gemma31 may generate different trajectories, reasoning length and playbook length",
                              "Gemma31 signed-advantage task counts unknown until complete G5 bank; "
                              "proxy uses Gemma26 N39/46 and1410 curator calls",
                              "Missing-usage timeouts and hidden provider retries cannot be fully priced "
                              "from these logs",
                              "Completed778 candidate jobs, initial500 and reflection/update500 excluded "
                              "from future cost",
                              "50 interrupted candidate jobs are already included in the722 restarted jobs",
                              "No BWT or Held-out evaluation in this protocol"],
              "new_model_calls": 0, "source_sha256": SOURCE_HASHES,
              "tokenizer_sha256": hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
              "tokenizer_revision": REVISION, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fields = sorted({k for r in rows for k in r})
    with (OUT / "phase_token_costs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "rollout_cost_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(job_rows[0]))
        writer.writeheader()
        writer.writerows(job_rows)
    (OUT / "unmeasured_calls.json").write_text(json.dumps(UNKNOWN_CALLS, indent=2) + "\n")
    (OUT / "cost_estimate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert read(RUN31 / "status.json")["state"] == "paused" and (RUN31 / "STOP").exists()
    print(json.dumps({k: report[k] for k in ("phases", "totals", "known_usage_usd", "input_usd", "output_usd",
                                             "error_input_imputation", "report_vs_chosen_attempt_mismatch_jobs",
                                             "local_response_cache_adjustment", "cache_adjusted_known_usage_usd")}))


if __name__ == "__main__":
    main()
