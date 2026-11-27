# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Freeze completed checkpoint actors and probe their next actions on fixed histories."""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import fcntl
import io
import json
import os
import pickle
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.utils.policy_shift_metrics import (  # noqa: E402
    canonical,
    digest,
    load_json,
    normalize_calls,
    normalize_terminal,
    response_message,
    score_actions,
    splice_request,
    terminal_parser,
    write_json,
)

METHODS = ("ace", "gepa", "skillopt", "premise")
DISPLAY = {"ace": "ACE", "gepa": "GEPA", "skillopt": "SkillOpt", "premise": "PEACE"}


def readonly_queue(path):
    db = sqlite3.connect((Path(path).resolve() / "queue.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def checkpoint_gate(plan):
    """An empty pending queue alone does not establish successful completion."""
    root = Path(plan["queue_output"])
    current = load_json(root / "checkpoint_queue_manifest.json")
    expected = set(plan["selected_stream_ids"])
    if set(current.get("selected_stream_ids", [])) != expected:
        return {"ready": False, "reason": "Checkpoint queue scope changed"}
    with readonly_queue(root) as db:
        placeholders = ",".join("?" for _ in expected)
        invalid = [
            r[0]
            for r in db.execute(
                f"SELECT stream FROM stream_invalidations WHERE stream IN ({placeholders})", tuple(expected)
            )
        ]
        counts = dict(
            db.execute(
                f"""SELECT t.status,count(*) FROM tasks t JOIN checkpoints c ON c.id=t.cp
            WHERE c.stream IN ({placeholders}) GROUP BY t.status""",
                tuple(expected),
            ).fetchall()
        )
        points = {
            sid: [r[0] for r in db.execute("SELECT point FROM checkpoints WHERE stream=? ORDER BY point", (sid,))]
            for sid in sorted(expected)
        }
    missing = {
        sid: sorted(set(required) - set(points[sid]))
        for sid, required in plan["minimum_checkpoint_points"].items()
        if not set(required) <= set(points[sid])
    }
    dashboard = load_json(root / "dashboard.json")
    heartbeat = load_json(root / "heartbeat.json")
    ready = (
        not invalid
        and not missing
        and counts.get("completed", 0) > 0
        and not any(counts.get(k, 0) for k in ("pending", "running", "error"))
        and dashboard.get("phase") == "completed"
        and not dashboard.get("issues")
        and heartbeat.get("active") == 0
        and set(heartbeat.get("selected_stream_ids", [])) == expected
    )
    return {
        "ready": ready,
        "task_counts": counts,
        "invalid_streams": invalid,
        "missing_checkpoint_points": missing,
        "checkpoint_points": points,
        "controller_phase": dashboard.get("phase"),
        "active": heartbeat.get("active"),
    }


def own_db(output):
    db = sqlite3.connect(Path(output) / "policy_shift.sqlite3", timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS anchors(id TEXT PRIMARY KEY,benchmark TEXT,seed INTEGER,task TEXT,file TEXT);
        CREATE TABLE IF NOT EXISTS prompts(sha TEXT PRIMARY KEY,content TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,anchor TEXT,prompt TEXT,sample TEXT,
            status TEXT DEFAULT 'pending',attempts INTEGER DEFAULT 0,action TEXT,error TEXT,usage TEXT);
        CREATE TABLE IF NOT EXISTS policies(stream TEXT,point INTEGER,benchmark TEXT,method TEXT,seed INTEGER,
            prompt TEXT,PRIMARY KEY(stream,point));
        CREATE TABLE IF NOT EXISTS bindings(stream TEXT,point INTEGER,anchor TEXT,job TEXT,reference TEXT,
            PRIMARY KEY(stream,point,anchor));
        CREATE INDEX IF NOT EXISTS dispatch_policy_shift ON jobs(status);
    """)
    return db


class PlainState(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Optimizer state contains a non-data object: {module}.{name}")


def frozen_source(path, checks):
    path = Path(path)
    data = path.read_bytes()
    checks[str(path)] = digest(data)
    return data


def initial_content(spec, checks):
    snapshot = Path(spec["snapshot"])
    method = spec["method"]
    if method in {"gepa", "skillopt"}:
        name = "shared.txt" if method == "gepa" else "seed_skill.md"
        seed = frozen_source(snapshot / f"scripts/baselines/prompts/{method}/base/{name}", checks).decode()
        if method == "gepa":
            states = sorted(Path(spec["output"]).glob("upstream/replay_*/gepa_state.bin"))
            state = PlainState(io.BytesIO(frozen_source(states[-1], checks))).load()
            if state["program_candidates"][0]["shared"] != seed:
                raise ValueError("GEPA initial candidate differs from its frozen initial instruction")
        else:
            roots = sorted(Path(spec["output"]).glob("upstream/replay_*/skills/skill_v0000.md"))
            if not roots or any(frozen_source(p, checks).decode() != seed for p in roots):
                raise ValueError("SkillOpt initial skill differs from its frozen initial instruction")
        return seed
    source = frozen_source(snapshot / "src/exgentic/agents/ace/playbook_store.py", checks).decode()
    assignment = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEFAULT_PLAYBOOK" for t in n.targets)
    )
    root = ast.literal_eval(assignment.value)
    if method == "premise":
        iteration = json.loads(frozen_source(Path(spec["train_dir"]) / "iterations/iter_0001.json", checks))
        config = json.loads(frozen_source(Path(iteration["parents"][0]["session_dir"]) / "config.json", checks))
        if config["agent_kwargs"]["playbook"] != root:
            raise ValueError("PEACE initial parent differs from the recorded root")
    return root


def actor_prompt(spec, content, checks):
    if spec["method"] in {"gepa", "skillopt"}:
        suffix = "txt" if spec["method"] == "gepa" else "md"
        base = Path(spec["snapshot"]) / f"scripts/baselines/prompts/{spec['method']}/base"
        text = frozen_source(base / f"{spec['benchmark']}.{suffix}", checks).decode() + "\n\n" + content
    elif spec["benchmark"] == "terminalbench2":
        text = content
    else:
        from exgentic.agents.ace.prompts.generator import build_generator_system_prompt

        for relative in ("src/exgentic/agents/ace/prompts/generator.py", "src/exgentic/agents/ace/playbook_utils.py"):
            if frozen_source(Path(spec["snapshot"]) / relative, checks) != (ROOT / relative).read_bytes():
                raise ValueError("Actor prompt renderer differs from the source experiment")
        text = build_generator_system_prompt(content)
    return text


def find_span(messages, needle):
    found = []
    # Only the original instruction may change; quoted historical instructions cannot.
    content = messages[0]["content"]
    parts = (
        [(None, content)]
        if isinstance(content, str)
        else [(i, part["text"]) for i, part in enumerate(content) if part.get("type") == "text"]
    )
    for part, text in parts:
        if text.count(needle) == 1:
            start = text.index(needle)
            found.append(
                {"message": 0, "part": part, "start": start, "end": start + len(needle), "sha256": digest(needle)}
            )
    if len(found) != 1:
        raise ValueError("The source actor policy is not uniquely located in the initial instruction")
    return found[0]


def freeze(path, value):
    if Path(path).exists() and canonical(load_json(path)) != canonical(value):
        raise ValueError(f"Immutable policy-shift artifact changed: {path}")
    write_json(path, value)


def add_comparison_jobs(db, anchor, prompt, root, settings):
    """Reference and evaluation draws stay independent for an unchanged root."""

    def job(text, sample):
        key = digest(canonical([anchor, text, sample, settings]))
        db.execute("INSERT OR IGNORE INTO jobs(id,anchor,prompt,sample) VALUES(?,?,?,?)", (key, anchor, text, sample))
        return key

    reference = job(root, "primary")
    prediction = job(prompt, "repeat" if prompt == root else "primary")
    return reference, prediction


def repair_root_alias_bindings(db):
    """Repair only comparison references; preserve every inference job and response."""
    affected = db.execute("""SELECT p.benchmark,p.method,count(*) AS bindings,
        count(DISTINCT p.stream||':'||p.point) AS policies
        FROM bindings b JOIN policies p ON p.stream=b.stream AND p.point=b.point
        WHERE b.point>0 AND b.job=b.reference GROUP BY p.benchmark,p.method""").fetchall()
    missing = db.execute("""SELECT count(*) FROM bindings b
        LEFT JOIN bindings r ON r.stream=b.stream AND r.point=0 AND r.anchor=b.anchor
        LEFT JOIN jobs j ON j.id=r.job
        WHERE b.point>0 AND b.job=b.reference AND
        (r.job IS NULL OR r.job=b.reference OR j.sample!='repeat')""").fetchone()[0]
    if missing:
        raise ValueError("An independent root repeat is missing; cannot repair comparisons")
    with db:
        changed = db.execute("""UPDATE bindings AS b SET job=(SELECT r.job FROM bindings r
            WHERE r.stream=b.stream AND r.point=0 AND r.anchor=b.anchor)
            WHERE b.point>0 AND b.job=b.reference""").rowcount
    if db.execute("SELECT count(*) FROM bindings WHERE point>0 AND job=reference").fetchone()[0]:
        raise ValueError("Self-comparisons remain after repair")
    return {"changed_bindings": changed, "affected": [dict(row) for row in affected]}


def prepare(output, plan):
    from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows

    protocol = digest(
        canonical(
            {
                k: plan[k]
                for k in (
                    "model",
                    "temperature",
                    "input_tokens",
                    "output_tokens",
                    "checkpoint_mode",
                    "selected_stream_ids",
                    "terminal_parser_sha256",
                )
            }
        )
    )
    if digest(Path(plan["terminal_parser_path"]).read_bytes()) != plan["terminal_parser_sha256"]:
        raise ValueError("Frozen official Terminal-Bench parser changed")
    saved = output / "prepared.json"
    if saved.exists():
        prepared = load_json(saved)
        if prepared.get("protocol_sha256") != protocol:
            raise ValueError("Frozen policy-shift protocol changed")
        for path, sha in prepared["source_hashes"].items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError(f"A frozen source changed: {path}")
        return prepared
    db, checks, audit = own_db(output), {}, []
    parser = terminal_parser(plan["terminal_parser_path"])
    with readonly_queue(plan["queue_output"]) as source:
        specs = [
            json.loads(source.execute("SELECT spec FROM streams WHERE id=?", (sid,)).fetchone()[0])
            for sid in plan["selected_stream_ids"]
        ]
        grouped = defaultdict(list)
        source_cps = {}
        for spec in specs:
            grouped[(spec["benchmark"], spec["seed"])].append(spec)
            if spec.get("appworld_policy_sha256"):
                raise ValueError("Expected the selected original-policy cohort")
            cps = source.execute("SELECT * FROM checkpoints WHERE stream=? ORDER BY point", (spec["id"],)).fetchall()
            source_cps[spec["id"]] = cps
            if plan.get("checkpoint_mode") == "root_final":
                cps = [cps[-1]]
            content = initial_content(spec, checks)
            policies = [(0, content, "Initial policy")]
            for cp in cps:
                payload = json.loads(frozen_source(cp["file"], checks))
                if digest(payload["content"]) != cp["sha"]:
                    raise ValueError("Checkpoint content differs from the evaluation queue")
                policies.append((cp["point"], payload["content"], payload["selection"]))
            for point, content, selection in policies:
                prompt = actor_prompt(spec, content, checks)
                freeze(
                    output / "policies" / spec["id"] / f"cp{point:03d}.json",
                    {"point": point, "content": content, "prompt": prompt, "selection": selection},
                )
                sha = digest(prompt)
                db.execute("INSERT OR IGNORE INTO prompts VALUES(?,?)", (sha, prompt))
                db.execute(
                    "INSERT OR REPLACE INTO policies VALUES(?,?,?,?,?,?)",
                    (spec["id"], point, spec["benchmark"], spec["method"], spec["seed"], sha),
                )
            db.commit()
        for (benchmark, seed), group in sorted(grouped.items()):
            if {s["method"] for s in group} != set(METHODS):
                raise ValueError("Each benchmark/seed must include all four methods")
            spec = next(s for s in group if s["method"] == "ace")
            expected = set(spec["splits"]["train"])
            if any(set(s["splits"]["train"]) != expected for s in group):
                raise ValueError("The methods' frozen training task sets differ")
            cp = source_cps[spec["id"]][0]
            payload = load_json(cp["file"])
            needle = actor_prompt(spec, payload["content"], checks)
            tasks = source.execute(
                "SELECT * FROM tasks WHERE cp=? AND split='train' ORDER BY ordinal,id", (cp["canonical"],)
            ).fetchall()
            if {r["task_id"] for r in tasks} != expected:
                raise ValueError("History-bank task coverage differs from the frozen training set")
            for task in tasks:
                result = json.loads(task["result"])
                folder = Path(result["session_output"])
                metadata = list(folder.rglob("session.json"))
                if len(metadata) != 1:
                    raise ValueError(f"Ambiguous history source for queue task {task['id']}")
                session = metadata[0].parent
                traces = (
                    list(folder.glob("**/run/litellm/trace.jsonl"))
                    if benchmark == "terminalbench2"
                    else [session / "agent/litellm/trace.jsonl"]
                )
                try:
                    if len(traces) != 1:
                        raise ValueError("Ambiguous actor trace")
                    frozen_source(traces[0], checks)
                    rows = list(iter_expanded_trace_rows(traces[0]))
                except (OSError, KeyError, ValueError) as exc:
                    audit.append(
                        {
                            "benchmark": benchmark,
                            "seed": seed,
                            "task": task["task_id"],
                            "reason": "missing_or_undecodable_trace",
                            "error": str(exc),
                        }
                    )
                    continue
                seen = set()
                used = 0
                for index, row in enumerate(rows):
                    request = row.get("request") or {}
                    if benchmark != "terminalbench2" and not request.get("tools"):
                        continue  # Keep the acting tool set fixed; exclude selection requests.
                    try:
                        if row.get("status") != "success":
                            raise ValueError("Recorded request failed")
                        response = row["response"]
                        identity = response.get("id") or digest(canonical(response))
                        if identity in seen:
                            continue  # Harbor logs the same native completion twice.
                        seen.add(identity)
                        message = response_message(response)
                        if benchmark == "terminalbench2":
                            normalize_terminal(message, parser)
                        elif not normalize_calls(message):
                            raise ValueError("No recorded reference calls")
                        messages = request["messages"]
                        span = find_span(messages, needle)
                        anchor = {
                            "benchmark": benchmark,
                            "seed": seed,
                            "task_id": task["task_id"],
                            "messages": messages,
                            "tools": request.get("tools", []),
                            "policy_span": span,
                        }
                        key = digest(canonical(anchor))
                        path = output / "histories" / f"{key}.json"
                        freeze(path, anchor)
                        db.execute(
                            "INSERT OR IGNORE INTO anchors VALUES(?,?,?,?,?)",
                            (key, benchmark, seed, task["task_id"], str(path)),
                        )
                        used += 1
                    except (ValueError, KeyError, TypeError, IndexError, SyntaxError) as exc:
                        audit.append(
                            {
                                "benchmark": benchmark,
                                "seed": seed,
                                "task": task["task_id"],
                                "row": index,
                                "reason": "invalid_reference",
                                "error": str(exc),
                            }
                        )
                audit.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "task": task["task_id"],
                        "reason": "coverage",
                        "histories": used,
                        "source_checkpoint": cp["point"],
                    }
                )
                db.commit()
        settings = canonical({k: plan[k] for k in ("model", "temperature", "input_tokens", "output_tokens")})

        for policy in db.execute("SELECT * FROM policies").fetchall():
            root = db.execute("SELECT prompt FROM policies WHERE stream=? AND point=0", (policy["stream"],)).fetchone()[
                0
            ]
            anchors = db.execute(
                "SELECT id FROM anchors WHERE benchmark=? AND seed=?", (policy["benchmark"], policy["seed"])
            ).fetchall()
            if not anchors:
                raise ValueError("A benchmark/seed has no usable fixed histories")
            for anchor in anchors:
                reference, prediction = add_comparison_jobs(db, anchor[0], policy["prompt"], root, settings)
                db.execute(
                    "INSERT OR REPLACE INTO bindings VALUES(?,?,?,?,?)",
                    (policy["stream"], policy["point"], anchor[0], prediction, reference),
                )
            db.commit()
    prepared = {
        "protocol_sha256": protocol,
        "streams": specs,
        "source_hashes": checks,
        "history_bank": "ACE earliest CP train evaluation inputs",
        "histories": db.execute("SELECT count(*) FROM anchors").fetchone()[0],
        "distinct_requests": db.execute("SELECT count(*) FROM jobs").fetchone()[0],
        "policy_instances": db.execute("SELECT count(*) FROM policies").fetchone()[0],
        "environment_execution": False,
        "learning": False,
    }
    write_json(output / "history_audit.json", audit)
    write_json(saved, prepared)
    db.close()
    return prepared


async def probe(output, plan):
    import httpx

    db = own_db(output)
    with db:
        db.execute("UPDATE jobs SET status='pending' WHERE status='running'")
    rows = db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id").fetchall()
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(dict(row))
    parser = terminal_parser(plan["terminal_parser_path"])
    prompts = dict(db.execute("SELECT sha,content FROM prompts").fetchall())
    if any(digest(content) != sha for sha, content in prompts.items()):
        raise ValueError("A frozen actor prompt changed")
    files = {r["id"]: r["file"] for r in db.execute("SELECT id,file FROM anchors")}
    last_status = 0.0

    def status():
        nonlocal last_status
        if time.monotonic() - last_status < 2:
            return
        last_status = time.monotonic()
        counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
        write_json(
            output / "status.json",
            {
                "phase": "probing",
                "pid": os.getpid(),
                "workers": plan["workers"],
                "counts": counts,
                "queued": queue.qsize(),
            },
        )

    async with httpx.AsyncClient(
        timeout=600, limits=httpx.Limits(max_connections=plan["workers"], max_keepalive_connections=plan["workers"])
    ) as client:
        models = await client.get(plan["base_url"].rstrip("/") + "/models")
        models.raise_for_status()
        if plan["model"].removeprefix("openai/") not in {r["id"] for r in models.json()["data"]}:
            raise ValueError("Policy-shift endpoint serves another model")
        # Check protocol support before dispatching a large immutable queue.
        token_url = plan.get("tokenize_url") or plan["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize"
        check = await client.post(
            token_url,
            json={
                "model": plan["model"].removeprefix("openai/"),
                "messages": [{"role": "user", "content": "protocol check"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        check.raise_for_status()
        if not isinstance(check.json().get("count"), int):
            raise ValueError("Endpoint does not expose a usable input-token count")
        health_lock = asyncio.Lock()
        last_health = time.monotonic()

        async def wait_for_model(force=False):
            nonlocal last_health
            if not force and time.monotonic() - last_health < 15:
                return
            async with health_lock:
                while True:
                    try:
                        response = await client.get(plan["base_url"].rstrip("/") + "/models", timeout=10)
                        response.raise_for_status()
                        if plan["model"].removeprefix("openai/") not in {r["id"] for r in response.json()["data"]}:
                            raise ValueError("Policy-shift endpoint serves another model")
                        last_health = time.monotonic()
                        return
                    except (httpx.TransportError, httpx.HTTPStatusError):
                        write_json(
                            output / "status.json",
                            {
                                "phase": "waiting_model",
                                "pid": os.getpid(),
                                "workers": plan["workers"],
                                "queued": queue.qsize(),
                            },
                        )
                        await asyncio.sleep(15)

        async def worker():
            while not queue.empty():
                await wait_for_model()
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                anchor = load_json(files[row["anchor"]])
                if digest(canonical(anchor)) != row["anchor"]:
                    raise ValueError("A frozen history or tool set changed")
                body = splice_request(
                    anchor, prompts[row["prompt"]], plan["model"], plan["temperature"], plan["output_tokens"]
                )
                with db:
                    db.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
                error, action, usage, response = None, None, None, None
                for attempt in range(3):
                    try:
                        tokenize = {
                            k: v
                            for k, v in body.items()
                            if k in {"model", "messages", "tools", "tool_choice", "chat_template_kwargs"}
                        }
                        token_response = await client.post(token_url, json=tokenize)
                        token_response.raise_for_status()
                        count = token_response.json()["count"]
                        if count > plan["input_tokens"]:
                            raise ValueError(f"Actor input budget exceeded: {count}>{plan['input_tokens']}")
                        result = await client.post(plan["base_url"].rstrip("/") + "/chat/completions", json=body)
                        result.raise_for_status()
                        response = result.json()
                        usage = response.get("usage")
                        if not usage or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
                            raise ValueError("Provider token usage missing")
                        if (
                            usage["prompt_tokens"] > plan["input_tokens"]
                            or usage["completion_tokens"] > plan["output_tokens"]
                        ):
                            raise ValueError("Actor token budget exceeded")
                        message = response_message(response)
                        action = (
                            normalize_terminal(message, parser)
                            if anchor["benchmark"] == "terminalbench2"
                            else normalize_calls(message)
                        )
                        error = None
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                            break
                        if attempt < 2:
                            await asyncio.sleep(2**attempt)
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        break  # No resampling after measured output or token-budget failures.
                if error and response is None and attempt == 2:
                    try:
                        health = await client.get(plan["base_url"].rstrip("/") + "/models", timeout=10)
                        health.raise_for_status()
                    except (httpx.TransportError, httpx.HTTPStatusError):
                        # An unavailable model service must not consume the rest
                        # of the study as thousands of artificial probe errors.
                        with db:
                            db.execute("UPDATE jobs SET status='pending',attempts=attempts+3 WHERE id=?", (row["id"],))
                        queue.put_nowait(row)
                        queue.task_done()
                        await wait_for_model(force=True)
                        continue
                with db:
                    db.execute(
                        "UPDATE jobs SET status=?,action=?,error=?,usage=?,attempts=attempts+? WHERE id=?",
                        (
                            "error" if error else "completed",
                            canonical(action) if action is not None else None,
                            error,
                            canonical(usage),
                            attempt + 1,
                            row["id"],
                        ),
                    )
                write_json(
                    output / "responses" / f"{row['id']}.json",
                    {"response": response, "error": error, "usage": usage, "attempts": attempt + 1},
                )
                queue.task_done()
                status()

        await asyncio.gather(*(worker() for _ in range(min(plan["workers"], len(rows)))))
    db.close()


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output, plan):
    db = own_db(output)
    invalid = {
        r[0]
        for r in db.execute("""SELECT b.anchor FROM bindings b
        JOIN jobs j ON j.id=b.job JOIN jobs r ON r.id=b.reference
        GROUP BY b.anchor HAVING sum(j.status!='completed' OR r.status!='completed')>0""")
    }
    task_values = defaultdict(lambda: defaultdict(list))
    coverage = defaultdict(lambda: {"anchors": 0, "common_anchors": 0, "tasks": set()})
    for anchor in db.execute("SELECT * FROM anchors"):
        key = (anchor["benchmark"], anchor["seed"])
        coverage[key]["anchors"] += 1
        if anchor["id"] not in invalid:
            coverage[key]["common_anchors"] += 1
            coverage[key]["tasks"].add(anchor["task"])
    query = """SELECT b.anchor,b.point,p.method,p.seed,p.benchmark,a.task,j.action,r.action AS reference
        FROM bindings b JOIN policies p ON p.stream=b.stream AND p.point=b.point
        JOIN anchors a ON a.id=b.anchor JOIN jobs j ON j.id=b.job JOIN jobs r ON r.id=b.reference
        WHERE j.status='completed' AND r.status='completed'"""
    for row in db.execute(query):
        if row["anchor"] in invalid:
            continue
        scores = score_actions(json.loads(row["action"]), json.loads(row["reference"]), row["benchmark"])
        key = (row["benchmark"], row["method"], row["seed"], row["point"], row["task"])
        for metric, value in scores.items():
            task_values[key][metric].append(value)
    task_rows, seed_groups = [], defaultdict(lambda: defaultdict(list))
    for key, metrics in sorted(task_values.items()):
        benchmark, method, seed, point, task = key
        for metric, values in metrics.items():
            value = statistics.mean(values)
            task_rows.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "seed": seed,
                    "checkpoint": point,
                    "task_id": task,
                    "metric": metric,
                    "value": value,
                    "histories": len(values),
                }
            )
            seed_groups[(benchmark, method, seed, point, metric)][metric].append(value)
    seeds, groups = [], defaultdict(list)
    for (benchmark, method, seed, point, metric), values in sorted(seed_groups.items()):
        numbers = values[metric]
        value = statistics.mean(numbers)
        seeds.append(
            {
                "benchmark": benchmark,
                "method": method,
                "seed": seed,
                "checkpoint": point,
                "metric": metric,
                "agreement": value,
                "policy_shift": 1 - value,
                "tasks": len(numbers),
            }
        )
        groups[(benchmark, method, point, metric)].append(value)
    means = [
        {
            "benchmark": b,
            "method": m,
            "checkpoint": cp,
            "metric": metric,
            "agreement": statistics.mean(values),
            "policy_shift": 1 - statistics.mean(values),
            "seed_sd": statistics.stdev(values) if len(values) > 1 else 0,
            "seeds": len(values),
        }
        for (b, m, cp, metric), values in sorted(groups.items())
    ]
    write_csv(output / "analysis/task_metrics.csv", task_rows)
    write_csv(output / "analysis/seed_metrics.csv", seeds)
    write_csv(output / "analysis/checkpoint_metrics.csv", means)
    counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
    write_json(
        output / "analysis/coverage.json",
        [dict(benchmark=b, seed=s, **{**v, "tasks": sorted(v["tasks"])}) for (b, s), v in sorted(coverage.items())],
    )
    summary = {
        "status": "completed_with_errors" if counts.get("error") else "completed",
        "counts": counts,
        "workers": plan["workers"],
        "common_histories": sum(v["common_anchors"] for v in coverage.values()),
        "reference": "CP0 predictions; CP0 curve point is an independent repeatability control",
        "scores": means,
    }
    write_json(output / "analysis/summary.json", summary)
    if not means:
        raise RuntimeError("No common successful histories remain; cannot report policy shift")
    plot(output, means)
    db.close()
    return summary


def plot(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for benchmark in ("bfcl", "appworld", "terminalbench2"):
        metrics = (
            ["accuracy"]
            if benchmark == "terminalbench2"
            else ["loose_accuracy", "strict_accuracy", "loose_f1", "strict_f1"]
        )
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
        for ax, metric in zip(axes[0], metrics, strict=True):
            for method in METHODS:
                selected = sorted(
                    [r for r in rows if (r["benchmark"], r["method"], r["metric"]) == (benchmark, method, metric)],
                    key=lambda r: r["checkpoint"],
                )
                if not selected:
                    continue
                x = [r["checkpoint"] for r in selected]
                y = [r["agreement"] for r in selected]
                sd = [r["seed_sd"] for r in selected]
                line = ax.plot(x, y, marker="o", markersize=3, label=DISPLAY[method])[0]
                ax.fill_between(
                    x,
                    [max(0, a - b) for a, b in zip(y, sd, strict=True)],
                    [min(1, a + b) for a, b in zip(y, sd, strict=True)],
                    color=line.get_color(),
                    alpha=0.12,
                )
            ax.set(
                xlabel="Checkpoint (training task IDs)",
                ylabel="Conditional action agreement",
                ylim=(-0.02, 1.02),
                title=metric.replace("_", " "),
            )
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle(benchmark + " · fixed histories · CP0 = independent repeat")
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"analysis/{benchmark}_policy_shift.{suffix}", dpi=180)
        plt.close(fig)


def run(output, prepare_only=False):
    plan = load_json(output / "plan.json")
    if not checkpoint_gate(plan)["ready"]:
        raise RuntimeError("Checkpoint evaluations have not completed successfully")
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = prepare(output, plan)
        write_json(
            output / "status.json",
            {
                "phase": "prepared",
                "pid": os.getpid(),
                **{k: prepared[k] for k in ("histories", "distinct_requests", "policy_instances")},
            },
        )
        if prepare_only:
            return
        asyncio.run(probe(output, plan))
        summary = analyze(output, plan)
        write_json(
            output / "status.json",
            {"phase": summary["status"], "pid": os.getpid(), "counts": summary["counts"], "workers": plan["workers"]},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.prepare_only)


if __name__ == "__main__":
    main()
