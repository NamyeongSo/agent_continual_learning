# SPDX-License-Identifier: Apache-2.0
"""Teacher-force a single recorded AppWorld action per initial Qwen3.5 trajectory."""

import csv
import json
import math
import random
import runpy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from statistics import mean, pstdev
from urllib.request import Request, urlopen

import numpy as np
from filelock import FileLock
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
BANK = RUN / "thinking" / "reflection"
ACCUM = RUN / "thinking" / "accumulation"
MODEL = "Qwen/Qwen3.5-9B"
HOST = "http://127.0.0.1:8006"
ARMS = ("without_gt", "with_gt")
METHODS = ("best", "worst")
TASK_COUNT = 6
CALL_INDEX = 8
SEED = 42
POSITIONS_PER_ACTION = 8


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def post(path, value):
    request = Request(HOST + path, data=json.dumps(value).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        return json.load(response)


def reward(row):
    if row.get("budget_failure"):
        assert row["pct"] == 0
        return Fraction(0)
    result = Fraction(row["passed_tests"], row["total_tests"])
    assert abs(float(result) - row["pct"]) < 1e-12
    return result


def normalize(messages):
    messages = deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    return messages


def render(tokenizer, template, request, *, system=None, assistant=None):
    messages = normalize(request["messages"])
    if system is not None:
        assert messages[0]["role"] == "system"
        messages[0]["content"] = system
    if assistant is not None:
        messages.extend(normalize([assistant]))
    return tokenizer.apply_chat_template(
        messages, tools=request["tools"], tokenize=False, chat_template=template,
        add_generation_prompt=assistant is None, **request["extra_body"]["chat_template_kwargs"],
    )


def get_prefix(tokenizer, template, result, empty):
    folder = BANK / "stage1" / result["id"] / f"attempt_{result['attempt']}" / "llm_calls"
    path = folder / f"{CALL_INDEX:04d}.json"
    if not path.exists():
        return None
    entry = read(path)
    if "response" not in entry:
        return None
    request = entry["request"]
    response = entry["response"]
    choice = response["choices"][0]
    assistant = choice["message"]
    tools = assistant.get("tool_calls") or []
    if (len(tools) != 1 or tools[0]["function"]["name"] != "execute_python"
            or not assistant.get("reasoning_content") or choice["finish_reason"] != "tool_calls"):
        return None
    assert request["messages"][0]["content"] == empty and request["model"] == "openai/" + MODEL
    prefix = render(tokenizer, template, request)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(prefix_ids) != response["usage"]["prompt_tokens"] or len(prefix_ids) > 9000:
        return None
    completed = render(tokenizer, template, request, assistant=assistant)
    if not completed.startswith(prefix) or not completed.endswith("</tool_call><|im_end|>\n"):
        return None
    assistant_text = completed[len(prefix):].removesuffix("<|im_end|>\n")
    action_start = assistant_text.rfind("<tool_call>\n<function=execute_python>\n")
    if action_start < 0:
        return None
    before_action, action = assistant_text[:action_start], assistant_text[action_start:]
    before_ids = tokenizer.encode(prefix + before_action, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + assistant_text, add_special_tokens=False)
    if full_ids[:len(before_ids)] != before_ids or before_ids[:len(prefix_ids)] != prefix_ids:
        return None
    marker = "<parameter=code>\n"
    code_start = action.find(marker)
    code_end = action.rfind("\n</parameter>")
    if code_start < 0 or code_end <= code_start:
        return None
    encoded_action = tokenizer(action, add_special_tokens=False, return_offsets_mapping=True)
    if encoded_action["input_ids"] != full_ids[len(before_ids):]:
        return None
    code_positions = [
        i for i, (a, b) in enumerate(encoded_action["offset_mapping"])
        if code_start + len(marker) <= a < b <= code_end
        and encoded_action["input_ids"][i] not in tokenizer.all_special_ids
    ]
    if len(code_positions) < 4:
        return None
    return {
        "path": path, "request": request, "before_action": before_action, "action": action,
        "prefix_ids": prefix_ids, "action_ids": full_ids[len(before_ids):],
        "reasoning_ids": before_ids[len(prefix_ids):],
        "code_positions": code_positions,
    }


def prepare():
    protocol = read(BANK / "protocol.json")
    policy = protocol["appworld_policy_sha256"]
    checkpoint = (RUN / "serving" / "model_revision.txt").read_text().strip()
    model_cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots" / checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), local_files_only=True)
    template = (RUN / "serving" / "chat_template.jinja").read_text()
    shared = runpy.run_path(str(BANK / "source_snapshot/src/exgentic/actor_context.py"))["shared_actor_prompt"]
    empty = shared("appworld", protocol["initial_playbook"])
    selected = {}
    for arm in ARMS:
        for method in METHODS:
            path = ACCUM / "conditions" / f"s{SEED}_{arm}_{method}" / "selections.json"
            selected[arm, method] = {c["candidate"]["task_id"]: c["candidate"] for c in read(path)}
    eligible = []
    order = sorted(protocol["task_ids"])
    random.Random(SEED).shuffle(order)
    inputs = []
    for task in order:
        group = [read(BANK / "stage1" / f"{task}_r{i}" / "result.json") for i in range(5)]
        assert all(r["appworld_policy_sha256"] == policy for r in group)
        rewards = [reward(r) for r in group]
        if len(set(rewards)) == 1:
            continue
        prefixes = [get_prefix(tokenizer, template, row, empty) for row in group]
        if any(prefix is None for prefix in prefixes):
            continue
        contexts = {}
        for (arm, method), choices in selected.items():
            pick = choices[task]
            path = BANK / "stage2" / f"{arm}_{pick['source_id']}" / "result.json"
            proposal = read(path)
            assert proposal["appworld_policy_sha256"] == policy
            edited = Path(proposal["updated_context_path"]).read_text()
            assert sha256(edited.encode()).hexdigest() == proposal["updated_context_sha256"]
            assert proposal["source_repeat"] == pick["source_repeat"] and proposal["task_id"] == task
            sample = next((BANK / "stage3" / f"{arm}_{pick['source_id']}_e0").glob("attempt_*/llm_calls/0001.json"))
            system = shared("appworld", edited)
            assert read(sample)["request"]["messages"][0]["content"] == system
            contexts[f"{arm}_{method}"] = (system, pick["source_repeat"], proposal["updated_context_sha256"])
        eligible.append(task)
        for index, prefix in enumerate(prefixes):
            request = prefix["request"]
            if len(inputs) == 0:
                reference = post("/tokenize", {
                    "model": MODEL, "messages": request["messages"], "tools": request["tools"],
                    "chat_template_kwargs": request["extra_body"]["chat_template_kwargs"],
                })
                assert prefix["prefix_ids"] == reference["tokens"]
            for condition, (system, source_repeat, ctx_hash) in {
                "initial": (empty, None, sha256(protocol["initial_playbook"].encode()).hexdigest()), **contexts,
            }.items():
                before = render(tokenizer, template, request, system=system)
                before_ids = tokenizer.encode(before + prefix["before_action"], add_special_tokens=False)
                assert before_ids[-len(prefix["reasoning_ids"]):] == prefix["reasoning_ids"]
                delta = rewards[index] - mean(rewards)
                advantage = float(delta) / (pstdev(map(float, rewards)) + 1e-8)
                inputs.append({
                    "task": task, "trajectory": index, "condition": condition, "source_repeat": source_repeat,
                    "pct": float(rewards[index]), "advantage": advantage,
                    "advantage_sign": (delta > 0) - (delta < 0), "context_sha256": ctx_hash,
                    "recorded_call": str(prefix["path"]), "tokens": before_ids + prefix["action_ids"],
                    "action_start": len(before_ids), "reasoning_start": len(before_ids) - len(prefix["reasoning_ids"]),
                    "code_positions": prefix["code_positions"],
                })
        if len(eligible) == TASK_COUNT:
            break
    assert len(eligible) == TASK_COUNT
    write(HERE / "plan.json", {
        "model": MODEL, "revision": checkpoint, "mode": "thinking", "tasks": eligible, "registered_call": CALL_INDEX,
        "requests": len(inputs), "selection_seed": SEED, "policy_sha256": policy,
        "action": "Python code inside the recorded execute_python tool call, conditioned on recorded reasoning",
        "score": "vLLM raw token logprobs before sampling transformations, single-position forced decode",
        "sampled_action_positions": POSITIONS_PER_ACTION,
        "matched_probe": "Initial request history and tool schemas held fixed; only learned system policy changes.",
    })
    return inputs


def one_position(prompts, token):
    output = post("/v1/completions", {
        "model": MODEL, "prompt": prompts, "max_tokens": 1, "temperature": 1.0,
        "logprobs": 1, "allowed_token_ids": [token], "ignore_eos": True, "return_token_ids": True,
    })
    choices = sorted(output["choices"], key=lambda choice: choice["index"])
    assert len(choices) == len(prompts)
    if any(choice["token_ids"] != [token] or not choice.get("logprobs") for choice in choices):
        return None
    values = [choice["logprobs"]["token_logprobs"][0] for choice in choices]
    assert all(math.isfinite(x) and x <= 0 for x in values)
    return values


def score(batch):
    reference = batch[0]
    action = reference["tokens"][reference["action_start"]:]
    assert all(row["tokens"][row["action_start"]:] == action for row in batch)
    locations = reference["code_positions"]
    sample_ids = np.linspace(0, len(locations) - 1, min(len(locations), POSITIONS_PER_ACTION), dtype=int)
    scored = []
    for offset in sample_ids:
        idx = locations[offset]
        logprobs = one_position([row["tokens"][:row["action_start"] + idx] for row in batch], action[idx])
        if logprobs is not None:
            scored.append((idx, logprobs))
    if len(scored) < 4:
        raise ValueError(f"Insufficient matched positions for {reference['task']}/{reference['trajectory']}")
    positions, values = zip(*scored, strict=True)
    result = []
    for logprobs, row in zip(zip(*values, strict=True), batch, strict=True):
        result.append({
            **{k: v for k, v in row.items() if k not in {"tokens", "code_positions"}},
            "action_tokens": len(action), "sampled_tokens": len(logprobs),
            "sampled_positions": json.dumps(positions), "action_logp_per_token": mean(logprobs),
        })
    return result


def report(rows):
    baseline = {(r["task"], r["trajectory"]): r for r in rows if r["condition"] == "initial"}
    contrasts = defaultdict(lambda: defaultdict(list))
    for row in rows:
        old = baseline[row["task"], row["trajectory"]]
        assert row["action_tokens"] == old["action_tokens"]
        row["raw_log_ratio_per_action_token"] = row["action_logp_per_token"] - old["action_logp_per_token"]
        if row["condition"] != "initial":
            for cohort in ("all", "exclude_reflection_source"):
                if cohort == "exclude_reflection_source" and row["trajectory"] == row["source_repeat"]:
                    continue
                contrasts[cohort, row["condition"]][row["task"], row["advantage_sign"]].append(row)
    result = {}
    for (cohort, condition), records in contrasts.items():
        paired = sorted({t for t, a in records if (t, 1) in records and (t, -1) in records})
        r = {"paired_tasks": len(paired)}
        for key in ("raw_log_ratio_per_action_token",):
            plus = [mean(row[key] for row in records[t, 1]) for t in paired]
            minus = [mean(row[key] for row in records[t, -1]) for t in paired]
            diff = np.array(plus) - np.array(minus)
            bootstrap = diff[np.random.default_rng(SEED).integers(len(diff), size=(10000, len(diff)))].mean(axis=1)
            r[key] = {"positive": mean(plus), "negative": mean(minus), "positive_minus_negative": float(diff.mean()),
                      "positive_contrast_tasks": int((diff > 0).sum()),
                      "task_bootstrap_ci95": list(map(float, np.quantile(bootstrap, [0.025, 0.975])))}
        result.setdefault(cohort, {})[condition] = r
    with (HERE / "action_logprobs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(HERE / "summary.json", result)
    return result


def main():
    with FileLock(str(HERE / "controller.lock"), timeout=0):
        # Confirm the backend returns raw probabilities before the forced single-token constraint.
        sample = one_position([[785, 3476]], 13)[0]
        assert abs(sample - (-4.735692024230957)) < 0.15, sample
        inputs = prepare()
        completed = []
        size = 1 + len(ARMS) * len(METHODS)
        batches = [inputs[i:i + size] for i in range(0, len(inputs), size)]
        print(f"Prepared {len(inputs)} action scores over {TASK_COUNT} tasks.", flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(score, batch) for batch in batches]
            for future in as_completed(futures):
                completed.extend(future.result())
                if len(completed) % 15 == 0:
                    print(f"Scored {len(completed)}/{len(inputs)}", flush=True)
        assert len(completed) == len(inputs)
        complete = sorted(completed, key=lambda x: (x["task"], x["trajectory"], x["condition"]))
        print(json.dumps(report(complete), indent=2), flush=True)


if __name__ == "__main__":
    main()
