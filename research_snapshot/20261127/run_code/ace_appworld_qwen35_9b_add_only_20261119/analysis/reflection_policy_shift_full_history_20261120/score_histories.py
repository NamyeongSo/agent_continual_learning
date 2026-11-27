# SPDX-License-Identifier: Apache-2.0
"""Score all logged action turns in a stratified Qwen3.5 reflection sample."""

import csv
import random
import runpy
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from filelock import FileLock
from jinja2 import TemplateError
from transformers import AutoTokenizer

OUTPUT = Path(__file__).resolve().parent
PILOT = runpy.run_path(str(OUTPUT.parent / "reflection_policy_shift_pilot_20261120/measure_policy_shift.py"))
BANK, ACCUM, RUN = (PILOT[k] for k in ("BANK", "ACCUM", "RUN"))
ARMS, METHODS, SEED = (PILOT[k] for k in ("ARMS", "METHODS", "SEED"))
CONDITIONS = ("initial", *(f"{a}_{s}" for a in ARMS for s in METHODS))
read, write, reward = (PILOT[k] for k in ("read", "write", "reward"))
TASKS_PER_BAND = 3
SAMPLES_PER_ACTION = 4
MAX_BATCH = 10
WORKERS = 2


def build_plan():
    protocol = read(BANK / "protocol.json")
    tasks = []
    for task in protocol["task_ids"]:
        rows = [read(BANK / "stage1" / f"{task}_r{i}" / "result.json") for i in range(5)]
        assert all(r["task_id"] == task and r["appworld_policy_sha256"] == protocol["appworld_policy_sha256"]
                   for r in rows)
        values = [reward(r) for r in rows]
        if len(set(values)) > 1:
            tasks.append((mean(values), task, rows))
    tasks.sort()
    rng = random.Random(SEED)
    selected = []
    for band, indices in enumerate(np.array_split(np.arange(len(tasks)), 4), 1):
        sample = rng.sample([int(i) for i in indices], TASKS_PER_BAND)
        selected.extend({"band": band, "task": tasks[i][1], "trajectories": tasks[i][2]} for i in sample)
    plan = {"tasks": selected, "eligible_tasks": len(tasks), "conditions": CONDITIONS,
            "samples_per_action": SAMPLES_PER_ACTION, "workers": WORKERS, "batch": MAX_BATCH,
            "metric": "log probability per sampled action token, pre-sampler; recorded reasoning held fixed",
            "raw_sample": "uniformly spaced tokens over canonical code or other nonempty recorded action",
            "normalization": "mean turns per trajectory, mean trajectories per sign, mean task contrasts",
            "policy_sha256": protocol["appworld_policy_sha256"],
            "model_revision": (RUN / "serving/model_revision.txt").read_text().strip()}
    write(OUTPUT / "plan.json", plan)
    return plan


def load_contexts(protocol):
    shared = runpy.run_path(str(BANK / "source_snapshot/src/exgentic/actor_context.py"))["shared_actor_prompt"]
    contexts = {}
    for arm in ARMS:
        for method in METHODS:
            picks = read(ACCUM / "conditions" / f"s{SEED}_{arm}_{method}" / "selections.json")
            for item in picks:
                pick = item["candidate"]
                proposal = read(BANK / "stage2" / f"{arm}_{pick['source_id']}" / "result.json")
                text = Path(proposal["updated_context_path"]).read_text()
                assert sha256(text.encode()).hexdigest() == proposal["updated_context_sha256"]
                assert proposal["task_id"] == pick["task_id"] and proposal["source_repeat"] == pick["source_repeat"]
                assert proposal["appworld_policy_sha256"] == protocol["appworld_policy_sha256"]
                contexts[pick["task_id"], f"{arm}_{method}"] = shared("appworld", text), pick["source_repeat"]
    return shared("appworld", protocol["initial_playbook"]), contexts


def encode_action(tokenizer, template, entry, empty):
    if not entry.get("response"):
        return "no_model_output"
    request = entry["request"]
    response = entry["response"]
    emitted = response["choices"][0]["message"]
    messages = request["messages"]
    assert messages[0]["role"] == "system" and messages[0]["content"] == empty
    assert request["model"] == "openai/" + PILOT["MODEL"]
    try:
        prompt = PILOT["render"](tokenizer, template, request)
        completed = PILOT["render"](tokenizer, template, request, assistant=emitted)
    except (ValueError, TypeError, KeyError, TemplateError):
        return "unrenderable_history_or_action"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) != response["usage"]["prompt_tokens"]:
        return "history_token_count_mismatch"
    if not completed.startswith(prompt) or not completed.endswith("<|im_end|>\n"):
        return "no_canonical_continuation"
    output = completed[len(prompt):].removesuffix("<|im_end|>\n")
    end_thought = output.find("</think>")
    if end_thought < 0:
        return "missing_reasoning_boundary"
    action_from = end_thought + len("</think>")
    calls = emitted.get("tool_calls") or []
    if calls:
        action_from = output.find("<tool_call>", action_from)
        if action_from < 0:
            return "missing_tool_boundary"
    region_from, region_to = action_from, len(output)
    if len(calls) == 1 and calls[0]["function"]["name"] == "execute_python":
        marker = "<parameter=code>\n"
        code = output.find(marker, action_from)
        code_end = output.rfind("\n</parameter>")
        if code >= 0 and code_end > code:
            region_from, region_to = code + len(marker), code_end
    if not output[region_from:region_to].strip():
        return "empty_action"
    # Model generation starts after the separately encoded prompt, preventing BPE merges across this boundary.
    completion = tokenizer(output, add_special_tokens=False, return_offsets_mapping=True)
    completion_ids = completion["input_ids"]
    tokens = prompt_ids + completion_ids
    positions = [
        len(prompt_ids) + i for i, (a, b) in enumerate(completion["offset_mapping"])
        if region_from <= a < b <= region_to and completion_ids[i] not in tokenizer.all_special_ids
    ]
    if not positions:
        return "no_action_tokens"
    indices = np.linspace(0, len(positions) - 1, min(len(positions), SAMPLES_PER_ACTION), dtype=int)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    system_end = prompt_ids.index(im_end) + 1
    system_str = prompt.split("<|im_end|>", 1)[0] + "<|im_end|>"
    assert empty in system_str
    return system_str, tokens, system_end, [positions[i] for i in indices], len(positions)


def prepare_task(tokenizer, template, group, empty, contexts):
    count = Counter()
    records = {}
    batches = defaultdict(list)
    system_cache = {}
    scores = [reward(r) for r in group["trajectories"]]
    task = group["task"]
    for repeat, row in enumerate(group["trajectories"]):
        assert row["source_repeat"] == repeat
        path = BANK / "stage1" / row["id"] / f"attempt_{row['attempt']}" / "llm_calls"
        logs = sorted(path.glob("*.json"))
        for log in logs:
            count["logged_calls"] += 1
            encoded = encode_action(tokenizer, template, read(log), empty)
            if isinstance(encoded, str):
                count[encoded] += 1
                continue
            system, tokens, end, positions, length = encoded
            count["scorable_action_turns"] += 1
            turn = int(log.stem)
            a = scores[repeat] - mean(scores)
            record = {"task": task, "band": group["band"], "trajectory": repeat, "call": turn,
                      "advantage": float(a) / (pstdev(map(float, scores)) + 1e-8),
                      "advantage_sign": (a > 0) - (a < 0), "pct": float(scores[repeat]),
                      "recorded_action_tokens": length}
            records[repeat, turn] = record
            variants = {}
            for condition in CONDITIONS:
                if condition == "initial":
                    head, source = tokens[:end], None
                else:
                    new_system, source = contexts[task, condition]
                    key = (system, condition)
                    if key not in system_cache:
                        assert system.count(empty) == 1
                        text = system.replace(empty, new_system)
                        system_cache[key] = tokenizer.encode(text, add_special_tokens=False)
                    head = system_cache[key]
                variants[condition] = head, source
            for i, position in enumerate(positions):
                key = (repeat, turn, i)
                for condition, (head, source) in variants.items():
                    prompt = head + tokens[end:position]
                    batches[tokens[position]].append((key, condition, source, prompt))
    # All five policies for one token stay in the same request; reuse repeated next-token IDs within this Task.
    requests = [(t, members[i:i + MAX_BATCH]) for t, members in batches.items()
                for i in range(0, len(members), MAX_BATCH)]
    count["requested_token_scores"] = sum(len(members) for _, members in requests)
    return requests, records, count


def score_request(token, batch):
    prompts = [x[3] for x in batch]
    for attempt in range(3):
        try:
            values = PILOT["one_position"](prompts, token)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    return [(key, cond, source, value) for (key, cond, source, _), value in
            zip(batch, values if values is not None else [None] * len(batch), strict=True)]


def score_task(pool, tokenizer, template, empty, contexts, group):
    requests, records, coverage = prepare_task(tokenizer, template, group, empty, contexts)
    pending = [pool.submit(score_request, token, group) for token, group in requests]
    token_scores = defaultdict(dict)
    source_ids = {}
    for completed, future in enumerate(as_completed(pending), 1):
        for (trajectory, turn, position), condition, source, score in future.result():
            token_scores[trajectory, turn][position, condition] = score
            source_ids[condition] = source
        if completed % 100 == 0:
            print(f"Task {group['task']}: {completed}/{len(pending)} scoring batches", flush=True)
    rows = []
    for key, record in records.items():
        positions = {i for i, _ in token_scores[key]
                     if all(token_scores[key].get((i, c)) is not None for c in CONDITIONS)}
        if not positions:
            coverage["unscored_turns"] += 1
            continue
        coverage["scored_turns"] += 1
        coverage["matched_token_positions"] += len(positions)
        for condition in CONDITIONS:
            deltas = [token_scores[key][i, condition] - token_scores[key][i, "initial"] for i in sorted(positions)]
            rows.append({**record, "condition": condition, "source_repeat": source_ids[condition],
                         "matched_tokens": len(positions), "mean_log_ratio": mean(deltas)})
    write(OUTPUT / f"{group['task']}.json", {"coverage": dict(coverage), "rows": rows})
    return coverage


def summarize(plan, *, state):
    files = [OUTPUT / f"{x['task']}.json" for x in plan["tasks"] if (OUTPUT / f"{x['task']}.json").is_file()]
    if not files:
        return
    results = [read(f) for f in files]
    rows = [r for outcome in results for r in outcome["rows"]]
    coverage = Counter()
    for x in results:
        coverage.update(x["coverage"])
    by_trajectory = defaultdict(list)
    for row in rows:
        key = (row["task"], row["condition"], row["advantage_sign"], row["trajectory"], row["source_repeat"])
        by_trajectory[key].append(row["mean_log_ratio"])
    groups = defaultdict(lambda: defaultdict(list))
    for (task, condition, sign, trajectory, source), values in by_trajectory.items():
        if condition == "initial" or sign == 0:
            continue
        groups["all", condition][task, sign].append(mean(values))
        if trajectory != source:
            groups["without_source", condition][task, sign].append(mean(values))
    summary = {}
    for (variant, condition), groups_by_sign in groups.items():
        eligible = sorted({task for task, _ in groups_by_sign
                           if (task, 1) in groups_by_sign and (task, -1) in groups_by_sign})
        if not eligible:
            continue
        positive = [mean(groups_by_sign[task, 1]) for task in eligible]
        negative = [mean(groups_by_sign[task, -1]) for task in eligible]
        contrasts = np.asarray(positive) - np.asarray(negative)
        rng = np.random.default_rng(SEED)
        estimates = contrasts[rng.integers(len(contrasts), size=(10000, len(contrasts)))].mean(axis=1)
        summary.setdefault(variant, {})[condition] = {
            "paired_tasks": len(eligible), "positive": mean(positive), "negative": mean(negative),
            "positive_minus_negative": float(contrasts.mean()),
            "task_bootstrap_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        }
    if rows:
        with (OUTPUT / "action_logprobs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(OUTPUT / "summary.json", {"state": state, "completed_tasks": len(files), "coverage": coverage,
                                    "contrasts": summary, "updated_at": datetime.now(UTC).isoformat()})


def main():
    with FileLock(str(OUTPUT / "controller.lock"), timeout=0):
        plan = read(OUTPUT / "plan.json") if (OUTPUT / "plan.json").is_file() else build_plan()
        expected = (RUN / "serving/model_revision.txt").read_text().strip()
        assert expected == plan["model_revision"]
        cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots" / expected
        tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True)
        template = (RUN / "serving/chat_template.jinja").read_text()
        empty, contexts = load_contexts(read(BANK / "protocol.json"))
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for group in plan["tasks"]:
                if not (OUTPUT / f"{group['task']}.json").is_file():
                    print(f"Task {group['task']} (Q{group['band']}) started", flush=True)
                    c = score_task(pool, tokenizer, template, empty, contexts, group)
                    print(f"Task {group['task']} finished: {dict(c)}", flush=True)
                summarize(plan, state="running")
        summarize(plan, state="completed")


if __name__ == "__main__":
    main()
