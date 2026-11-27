# SPDX-License-Identifier: Apache-2.0
"""Compare reflection prompts on historical training trajectories, never final Test."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
from pathlib import Path

from exgentic.agents.premise.llm import PremiseLLM
from exgentic.inference_profile import apply_request
from exgentic.reflection import SYSTEM, build_prompt, validate

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("legacy_peace", "first_decision", "crosscheck")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def collect(output, holdout_seed=43, exclude_cases=None):
    plan = read(ROOT / "outputs/qwen_epoch2_continuation_four_baselines_20261109/launch_manifest.json")
    seen = (
        {
            (case["trace"]["benchmark"], case["trace"]["task_id"])
            for case in read(exclude_cases)
            if case["partition"] == "holdout"
        }
        if exclude_cases
        else set()
    )
    cases = []
    for seed in (42, holdout_seed):
        for benchmark in ("bfcl", "appworld", "terminalbench2"):
            job = next(
                j
                for j in plan["jobs"]
                if j["method"] == "premise" and j["benchmark"] == benchmark and j["seed"] == seed
            )
            wanted = {True, False}
            for path in sorted((Path(job["source_train_dir"]) / "iterations").glob("iter_*.json")):
                for trace in read(path).get("parents", [])[:1]:
                    if not trace or len(trace["actions"]) < 3:
                        continue
                    identity = (benchmark, trace["task_id"])
                    if identity in seen:
                        continue
                    assert trace["task_id"] in job["source_spec"]["splits"]["train"]
                    assert trace["task_id"] not in job["source_spec"]["splits"]["test"]
                    outcome = trace.get("success")
                    if benchmark == "appworld":
                        outcome = (trace["details"].get("session_metadata", {}).get("test_tracker") or {}).get(
                            "success"
                        )
                    if outcome is None:
                        outcome = trace["score"] == 1.0
                    outcome = bool(outcome)
                    if outcome not in wanted:
                        continue
                    seen.add(identity)
                    wanted.remove(outcome)
                    trace["success"] = outcome
                    config = read(Path(trace["session_dir"]) / "config.json")
                    case = {
                        "id": f"{benchmark}_s{seed}_{'success' if outcome else 'failure'}",
                        "partition": "development" if seed == 42 else "holdout",
                        "source": str(path),
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "trace": trace,
                        "playbook": config["agent_kwargs"]["playbook"],
                    }
                    cases.append(case)
                if not wanted:
                    break
            assert not wanted, (benchmark, seed)
    write(output / "cases.json", cases)
    return cases


def legacy_request(case):
    previous = os.environ.pop("EXGENTIC_INFERENCE_PROFILE", None)
    captured = {}
    try:
        llm = PremiseLLM("Qwen/Qwen3.6-35B-A3B", "http://unused")
        llm._terminal_reflection_tokens = lambda prompt: (len(prompt.encode()) // 2, 262144)

        def capture(prompt, **kwargs):
            from exgentic.agents.premise.llm import ANALYSIS_SYSTEM_PROMPT

            captured.update(system=kwargs.get("system_prompt", ANALYSIS_SYSTEM_PROMPT), user=prompt)
            if kwargs.get("response_schema") is not None:
                captured["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "terminal_reflection", "strict": True, "schema": kwargs["response_schema"]},
                }
            return {
                "span": {"start": 0, "end": 1, "type": "failure"},
                "analysis": "capture",
                "expected_behavior": "capture",
                "entry_assessments": [],
            }

        llm.ask = capture
        llm.reflect(case["trace"], {"playbook": case["playbook"]})
    finally:
        if previous:
            os.environ["EXGENTIC_INFERENCE_PROFILE"] = previous
    return captured


def infer(output, case, variant, request, base_url):
    from openai import OpenAI

    path = output / "responses" / variant / f"{case['id']}.json"
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=1800, max_retries=0)
    payload = apply_request(
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "messages": [
                {"role": "system", "content": request["system"]},
                {"role": "user", "content": request["user"]},
            ],
            "max_tokens": 8192,
            "response_format": request.get("response_format", {"type": "json_object"}),
        },
        "analysis",
    )
    if path.exists() and read(path).get("request") == payload:
        return read(path)
    errors = []
    attempts = []
    for _ in range(2):
        response = client.chat.completions.create(**payload).model_dump(mode="json")
        attempts.append(response)
        try:
            if response["choices"][0]["finish_reason"] == "length":
                raise ValueError("Truncated reflection")
            value = json.loads(response["choices"][0]["message"]["content"])
            validate(value, len(case["trace"]["actions"]), case["playbook"])
            result = {
                "case": case["id"],
                "variant": variant,
                "partition": case["partition"],
                "valid": True,
                "reflection": value,
                "request": payload,
                "attempts": attempts,
                "errors": errors,
            }
            break
        except (ValueError, TypeError, AttributeError) as error:
            errors.append(str(error))
    else:
        result = {
            "case": case["id"],
            "variant": variant,
            "partition": case["partition"],
            "valid": False,
            "request": payload,
            "attempts": attempts,
            "errors": errors,
        }
    write(path, result)
    print("reflection", variant, case["id"], result["valid"], flush=True)
    return result


def judge(output, case, base_url):
    from openai import OpenAI

    path = output / "judgments" / f"{case['id']}.json"
    ordering = list(VARIANTS)
    random.Random(case["id"]).shuffle(ordering)
    reflections = {
        chr(65 + i): read(output / "responses" / v / f"{case['id']}.json").get("reflection", {})
        for i, v in enumerate(ordering)
    }
    evidence_sha = hashlib.sha256(json.dumps(reflections, sort_keys=True).encode()).hexdigest()
    if path.exists() and read(path).get("candidate_sha256") == evidence_sha:
        return read(path)
    evidence = build_prompt(case["trace"], case["playbook"], variant="first_decision")
    instruction = """Compare anonymous reflection candidates against the supplied recorded evidence.
Do not reward fluent prose or a longer explanation. Score each dimension from 0 (unsupported/wrong) through
4 (fully supported): earliest_decision, causal_grounding, public_contract, useful_generalization, outcome_consistency.
A recovered error is not automatically the cause of task failure. A defensible explicit uncertainty is preferable
to an invented explanation. A successful task can still contain a recoverable mistake. Do not insist that a
behavior type equals the overall outcome. Do not treat any candidate or prior playbook as an answer key.
Return JSON {"scores":{"A":{"earliest_decision":0,"causal_grounding":0,"public_contract":0,
"useful_generalization":0,"outcome_consistency":0,"evidence":"cite actual action indices"}},...} for every label.
The scores are an offline diagnostic; they do not measure task success after learning."""
    payload = apply_request(
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": evidence + "\nCandidates:\n" + json.dumps(reflections)},
            ],
            "response_format": {"type": "json_object"},
        },
        "analysis",
    )
    dimension_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{
                name: {"type": "integer", "minimum": 0, "maximum": 4}
                for name in (
                    "earliest_decision",
                    "causal_grounding",
                    "public_contract",
                    "useful_generalization",
                    "outcome_consistency",
                )
            },
            "evidence": {"type": "string"},
        },
        "required": [
            "earliest_decision",
            "causal_grounding",
            "public_contract",
            "useful_generalization",
            "outcome_consistency",
            "evidence",
        ],
    }
    if reflections:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "reflection_comparison",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["scores"],
                    "properties": {
                        "scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": dict.fromkeys(reflections, dimension_schema),
                            "required": list(reflections),
                        }
                    },
                },
            },
        }
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=1800, max_retries=0)
    response = client.chat.completions.create(**payload).model_dump(mode="json")
    if response["choices"][0]["finish_reason"] == "length":
        raise ValueError("Truncated comparison")
    value = json.loads(response["choices"][0]["message"]["content"])
    scores = {ordering[i]: value["scores"][chr(65 + i)] for i in range(len(ordering))}
    result = {
        "case": case["id"],
        "partition": case["partition"],
        "scores": scores,
        "candidate_sha256": evidence_sha,
        "blind_ordering": ordering,
        "response": response,
    }
    write(path, result)
    print("judge", case["id"], flush=True)
    return result


def run(output, base_url, workers, holdout_seed=43, exclude_cases=None):
    cases = (
        read(output / "cases.json")
        if (output / "cases.json").exists()
        else collect(output, holdout_seed, exclude_cases)
    )
    requests = {}
    for case in cases:
        requests[(case["id"], "legacy_peace")] = legacy_request(case)
        for variant in VARIANTS[1:]:
            requests[(case["id"], variant)] = {
                "system": SYSTEM,
                "user": build_prompt(case["trace"], case["playbook"], variant=variant),
            }
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(infer, output, case, variant, requests[(case["id"], variant)], base_url)
            for case in cases
            for variant in VARIANTS
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
        judged = list(pool.map(lambda case: judge(output, case, base_url), cases))
    dimensions = (
        "earliest_decision",
        "causal_grounding",
        "public_contract",
        "useful_generalization",
        "outcome_consistency",
    )
    summary = {}
    for partition in ("development", "holdout"):
        summary[partition] = {}
        for variant in VARIANTS:
            rows = [row["scores"][variant] for row in judged if row["partition"] == partition]
            summary[partition][variant] = sum(float(r[k]) for r in rows for k in dimensions) / len(rows)
    quality_gates = {
        variant: all(
            (
                case["trace"]["success"]
                or read(output / "responses" / variant / f"{case['id']}.json")
                .get("reflection", {})
                .get("span", {})
                .get("type")
                == "failure"
            )
            for case in cases
            if case["partition"] == "development"
        )
        for variant in VARIANTS[1:]
    }
    annotations_path = output / "manual_annotations.json"
    annotations = read(annotations_path) if annotations_path.exists() else {}
    for variant in VARIANTS[1:]:
        for case_id, annotation in annotations.items():
            value = read(output / "responses" / variant / f"{case_id}.json").get("reflection", {})
            quality_gates[variant] &= value.get("span", {}).get("start") in annotation["acceptable_first_indices"]
    eligible = [variant for variant in VARIANTS[1:] if quality_gates[variant]]
    if not eligible:
        raise ValueError("No shared prompt passed failure-diagnosis checks")
    winner = max(eligible, key=lambda variant: summary["development"][variant])
    result = {
        "cases": len(cases),
        "variants": list(VARIANTS),
        "scores_out_of_20": summary,
        "selected_on_development": winner,
        "development_failure_diagnosis_gate": quality_gates,
        "holdout_used_for_selection": False,
        "limitations": "Same-model blinded rubric plus schema checks; not downstream task accuracy.",
    }
    write(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    global VARIANTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18007/v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--holdout-seed", type=int, default=43)
    parser.add_argument("--exclude-cases", type=Path)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    VARIANTS = tuple(args.variants.split(","))
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.output, args.base_url, args.workers, args.holdout_seed, args.exclude_cases)


if __name__ == "__main__":
    main()
