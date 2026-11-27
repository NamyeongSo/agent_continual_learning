# SPDX-License-Identifier: Apache-2.0
"""Build a read-only research settings snapshot from frozen experiment artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import runpy
import shlex
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = REPO / "outputs/qwen_epoch2_continuation_four_baselines_20261109"
DEFAULT_OUTPUT = REPO / "outputs/experiment_settings_atlas_20261111"
SPLIT_PATH = REPO / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_seed42_v1.json"
METHOD_NAMES = {"ace": "ACE AUD", "premise": "PEACE AUD w/o alignment", "skillopt": "SkillOpt", "gepa": "GEPA"}
BENCH_NAMES = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "Terminal-Bench 2.0"}
SECRET_KEYS = re.compile(r"(?:api_key|password|credential|access_token|private_key|^secret$)", re.I)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {
            key: ("[비공개]" if SECRET_KEYS.search(key) and item else clean(item))
            for key, item in value.items()
            if key not in {"environment", "supervisor", "canary_string"}
        }
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str) and value.startswith("ssh://"):
        return "SSH Docker context · 비공개 연결 주소"
    return value


def literals(path):
    results = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    results[target.id] = value
    return results


class Sources:
    def __init__(self):
        self.entries = {}

    def add(self, path, title=None, *, text=False):
        path = Path(path).resolve()
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        if identity not in self.entries:
            self.entries[identity] = {
                "id": identity,
                "title": title or path.name,
                "path": str(path),
                "sha256": sha(path),
                "size": path.stat().st_size,
                "text": path.read_text(encoding="utf-8") if text else None,
            }
        elif text:
            self.entries[identity]["text"] = path.read_text(encoding="utf-8")
        return identity


def config_for(job, epoch):
    root = Path(job["source_train_dir"] if epoch == 1 else job["train_dir"])
    filename = {"ace": "experiment_config.json", "premise": "state.json"}.get(job["method"], "config.json")
    path = root / filename
    if not path.exists():
        return path, None
    value = read(path)
    return path, value["config"] if job["method"] == "premise" else value


def primitive_defaults(path, class_name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    values = {}
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
            try:
                values[item.target.id] = ast.literal_eval(item.value)
            except (ValueError, TypeError):
                pass
    return values


def make_tasks(manifest, plan):
    metadata_path = SPLIT_PATH.with_suffix(".metadata.csv")
    with metadata_path.open() as stream:
        metadata = {(r["benchmark"], r["task_id"]): r for r in csv.DictReader(stream)}
    records = []
    task_roots = manifest["sources"]
    bfcl_file = Path(task_roots["bfcl"]["file"])
    data = bfcl_file.read_text()
    bfcl = json.loads(data) if data.lstrip().startswith("[") else [json.loads(line) for line in data.splitlines()]
    backend = bfcl_file.parent.parent / "constants/executable_backend_config.py"
    mappings = next(
        v for v in literals(backend).values() if isinstance(v, dict) and v.get("TwitterAPI") == "posting_api.json"
    )
    aw = Path(task_roots["appworld"]["dataset_file"]).parent.parent
    tb = Path(task_roots["terminalbench2"]["task_root"])
    for bench, section in manifest["benchmarks"].items():
        membership = {tid: split for split in ("train", "val", "test") for tid in section[split]}
        if bench == "bfcl":
            iterable = [(r["id"], r) for r in bfcl]
        else:
            iterable = [(tid, None) for b, tid in metadata if b == bench]
        for task_id, raw in iterable:
            meta = {
                k: v for k, v in metadata[(bench, task_id)].items() if v and k not in {"task_id", "benchmark", "split"}
            }
            for key in ("involved_classes", "tags"):
                if key in meta:
                    meta[key] = json.loads(meta[key])
            for key in ("variant", "user_turns", "num_apps", "num_apis", "num_solution_code_lines"):
                if key in meta:
                    meta[key] = int(meta[key])
            item = {"benchmark": bench, "id": task_id, "split": membership.get(task_id, "unused"), "meta": meta}
            if bench == "bfcl":
                turns = []
                for index, messages in enumerate(raw["question"]):
                    turns.append(
                        {
                            "turn": index + 1,
                            "messages": messages,
                            "text": "\n\n".join(str(x.get("content", "")) for x in messages),
                        }
                    )
                tools = []
                for name in raw["involved_classes"]:
                    path = bfcl_file.parent / "multi_turn_func_doc" / mappings[name]
                    functions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                    tools.extend(f for f in functions if f.get("name") not in raw.get("excluded_function", []))
                item.update(
                    instruction=turns[0]["text"],
                    turns=turns,
                    tools=tools,
                    context={
                        "initial_config": raw.get("initial_config"),
                        "path": raw.get("path"),
                        "excluded_function": raw.get("excluded_function", []),
                    },
                    source={"path": str(bfcl_file), "sha256": sha(bfcl_file)},
                )
            elif bench == "appworld":
                path = aw / "tasks" / task_id / "specs.json"
                specs = read(path)
                item.update(
                    instruction=specs["instruction"],
                    turns=[],
                    tools=[],
                    context={
                        "datetime": specs.get("datetime"),
                        "db_version": specs.get("db_version"),
                        "mode": "code_generation",
                        "completion": "finish(answer?, status?)",
                    },
                    source={"path": str(path), "sha256": sha(path)},
                )
            else:
                path = tb / task_id / "instruction.md"
                config = tomllib.loads((path.parent / "task.toml").read_text())
                item.update(
                    instruction=path.read_text(),
                    turns=[],
                    tools=[],
                    context={k: config.get(k) for k in ("task", "metadata", "agent", "verifier", "environment")},
                    source={"path": str(path), "sha256": sha(path)},
                )
            item["orders"] = {}
            for method in METHOD_NAMES:
                per_seed = {}
                for seed in (42, 43, 44):
                    job = next(
                        j
                        for j in plan["jobs"]
                        if j["method"] == method and j["benchmark"] == bench and j["seed"] == seed
                    )
                    per_seed[str(seed)] = {
                        name: (ids.index(task_id) + 1 if task_id in ids else None)
                        for name, ids in (
                            ("epoch1", job["epoch1_order"]),
                            ("epoch2", job["epoch2_order"]),
                            ("val", job["val_ids"]),
                            ("test", job["test_ids"]),
                        )
                    }
                item["orders"][method] = per_seed
            records.append(item)
    if len(records) != 706 or sum(r["split"] != "unused" for r in records) != 389:
        raise ValueError("Unexpected task catalog size")
    if any(not task["instruction"].strip() for task in records):
        raise ValueError("A benchmark task has no instruction")
    return records


def prompt_library(plan, sources):
    prompts = []
    original = next(j for j in plan["jobs"] if j["method"] == "ace")
    base = Path(original["source_snapshot"])
    for relative, names, methods in (
        (
            "src/exgentic/agents/ace/prompts/reflector.py",
            ["REFLECTOR_PROMPT_NO_GT", "REFLECTOR_PROMPT_TRAINING_TIME"],
            ["ace"],
        ),
        (
            "src/exgentic/agents/ace/prompts/curator.py",
            ["CURATOR_PROMPT_NO_GT", "CURATOR_PROMPT_TRAINING_TIME"],
            ["ace", "premise"],
        ),
        ("src/exgentic/agents/ace/playbook_store.py", ["DEFAULT_PLAYBOOK"], ["ace", "premise"]),
    ):
        path = base / relative
        for key, content in literals(path).items():
            if key in names and isinstance(content, str):
                prompts.append(
                    {
                        "name": key,
                        "methods": methods,
                        "text": content,
                        "source": sources.add(path, text=True),
                        "kind": "template",
                    }
                )
    generator = base / "src/exgentic/agents/ace/prompts/generator.py"
    prompts.append(
        {
            "name": "ACE / PEACE actor system prompt 구성",
            "methods": ["ace", "premise"],
            "text": generator.read_text(),
            "source": sources.add(generator, text=True),
            "kind": "implementation",
        }
    )
    j = next(j for j in plan["jobs"] if j["method"] == "premise")
    path = Path(j["source_snapshot"]) / "src/exgentic/agents/premise/llm.py"
    for key, content in literals(path).items():
        if isinstance(content, str) and ("PROMPT" in key or key == "EVIDENCE_RULES"):
            prompts.append(
                {
                    "name": key,
                    "methods": ["premise"],
                    "text": content,
                    "source": sources.add(path, text=True),
                    "kind": "template",
                }
            )
    for path in sorted((base / "scripts/baselines/prompts").rglob("*")):
        if path.suffix in {".md", ".txt"}:
            relative = path.relative_to(base / "scripts/baselines/prompts")
            prompts.append(
                {
                    "name": str(relative),
                    "methods": [relative.parts[0]],
                    "text": path.read_text(),
                    "source": sources.add(path),
                    "kind": "template",
                }
            )
    policy = next(j["appworld_policy"] for j in plan["jobs"] if j.get("appworld_policy"))
    prompts.append(
        {
            "name": "AppWorld Original environment policy",
            "methods": list(METHOD_NAMES),
            "text": Path(policy).read_text(),
            "source": sources.add(policy),
            "kind": "environment",
        }
    )
    harbor = Path("/home/thskadud/.exgentic/benchmarks/terminalbench2/venv/lib")
    for filename in ("terminus-json-plain.txt", "timeout.txt"):
        paths = sorted(harbor.glob(f"python*/site-packages/harbor/agents/terminus_2/templates/{filename}"))
        if paths:
            path = paths[0]
            prompts.append(
                {
                    "name": f"Terminal-Bench native Terminus 2 / {filename}",
                    "methods": list(METHOD_NAMES),
                    "text": path.read_text(),
                    "source": sources.add(path, text=True),
                    "kind": "template",
                }
            )
    return prompts


def sample_request_settings(job):
    base = Path(job["source_train_dir"])
    if job["method"] == "ace":
        files = base.glob("*/run/config.json")
    elif job["method"] == "premise":
        files = (base / "sessions").glob("*/run/config.json")
    else:
        files = (base / "rollouts").glob("*/sessions/*/run/config.json")
    path = next(files, None)
    if path is None:
        return None
    d = read(path)
    return {
        "source": str(path),
        "model": d.get("model"),
        "max_steps": d.get("max_steps"),
        "agent_kwargs": clean(d.get("agent_kwargs", {})),
        "experiment_limits": d.get("experiment_limits"),
        "benchmark_kwargs": clean(d.get("benchmark_kwargs", {})),
    }


def build(cohort=DEFAULT_COHORT, output=DEFAULT_OUTPUT):
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    sources = Sources()
    plan = read(cohort / "launch_manifest.json")
    manifest = read(SPLIT_PATH)
    ordered = Path(read(Path(plan["jobs"][0]["source_plan"]))["manifest"])
    if sha(ordered) != plan["manifest_sha256"]:
        raise ValueError("Ordered manifest differs from the run")
    for path, title in [
        (cohort / "launch_manifest.json", "Epoch 2 launch manifest"),
        (SPLIT_PATH, "Canonical balanced split"),
        (ordered, "Seed-specific split orders"),
    ]:
        sources.add(path, title)
    validation = read(SPLIT_PATH.with_suffix(".validation.json"))
    tasks = make_tasks(manifest, plan)
    prompts = prompt_library(plan, sources)
    original_plans = {}
    runs = []
    examples = []
    native_configs = {}
    for j in plan["jobs"]:
        epochs = {}
        if j["source_plan"] not in original_plans:
            original_plans[j["source_plan"]] = read(Path(j["source_plan"]))
        original = next(x for x in original_plans[j["source_plan"]]["jobs"] if x["id"] == j["id"])
        for epoch in (1, 2):
            path, config = config_for(j, epoch)
            epochs[str(epoch)] = {
                "config": clean(config),
                "source": sources.add(path) if config else None,
                "availability": "recorded" if config else "not_created",
                "command": clean(original["train_command"] if epoch == 1 else j["train_command"]),
                "command_shell": shlex.join(clean(original["train_command"] if epoch == 1 else j["train_command"])),
                "train_dir": str(path.parent),
                "count": j["epoch1_count"] if epoch == 1 else j["train_count"],
            }
        assert set(j["epoch1_order"]) == set(manifest["benchmarks"][j["benchmark"]]["train"])
        order = j["epoch1_order"].copy()
        random.Random(j["seed"]).shuffle(order)
        if order != j["epoch2_order"]:
            raise ValueError("Epoch 2 order differs from the recorded shuffle")
        run = {
            k: j.get(k)
            for k in (
                "id",
                "method",
                "benchmark",
                "seed",
                "optimizer_seed",
                "data_order_seed",
                "variant",
                "epoch1_count",
                "train_count",
                "source_snapshot",
                "epoch1_order",
                "epoch2_order",
                "test_workers",
                "test_ids",
                "val_ids",
                "output",
                "playbook_edit_mode",
            )
        }
        run["epochs"] = epochs
        run["benchmark_kwargs"] = clean(j["source_spec"]["benchmark_kwargs"])
        run["actor_limits"] = j["source_spec"]["actor_limits"]
        run["source_spec"] = clean(j["source_spec"])
        run["source_plan"] = j["source_plan"]
        run["policy_sha256"] = j.get("appworld_policy_sha256")
        runs.append(run)
        if j["seed"] == 42:
            example = sample_request_settings(j)
            if example:
                examples.append({"method": j["method"], "benchmark": j["benchmark"], **example})
            if j["method"] == "skillopt":
                native = Path(j["source_upstream"]) / "config.json"
                continuation = Path(j["train_dir"]) / "upstream/continued/config.json"
                native_configs[j["benchmark"]] = {
                    "epoch1": clean(read(native)),
                    "epoch2": clean(read(continuation)) if continuation.exists() else None,
                    "source": sources.add(native),
                }
    snapshots = {j["method"]: Path(j["source_snapshot"]) for j in plan["jobs"]}
    source_files = {
        "ACE actor": snapshots["ace"] / "src/exgentic/agents/ace/ace_instance.py",
        "ACE defaults": snapshots["ace"] / "src/exgentic/agents/ace/ace_agent.py",
        "PEACE pipeline": snapshots["premise"] / "src/exgentic/agents/premise/pipeline.py",
        "PEACE selection": snapshots["premise"] / "src/exgentic/agents/premise/selection.py",
        "GEPA runtime": snapshots["gepa"] / "scripts/baselines/gepa_runtime.py",
        "SkillOpt runtime": snapshots["skillopt"] / "scripts/baselines/skillopt_runtime.py",
        "Baseline LLM runtime": snapshots["gepa"] / "scripts/baselines/runtime.py",
        "Baseline actor": snapshots["gepa"] / "scripts/baselines/actor.py",
        "Epoch 2 continuation": Path(plan["continuation_script"]),
        "Actor budgets": snapshots["ace"] / "src/exgentic/core/experiment_limits.py",
        "Split membership generator": REPO / "scripts/utils/create_balanced_task_splits.py",
        "Split order generator": REPO / "scripts/utils/create_fixed_split_orders.py",
        "Parser bridge": REPO / "scripts/utils/serve_vllm_tool_parser_bridge.py",
        "Model launcher": Path("/home/thskadud/vllm_serve_qwen.sh"),
        "Chat template": Path("/home/thskadud/chat_templates/qwen3_6-27B.jinja"),
        "Upstream versions": snapshots["gepa"] / "scripts/baselines/upstreams.lock.toml",
        "Experiment protocol": snapshots["gepa"] / "scripts/baselines/protocol.toml",
        "BFCL environment": snapshots["ace"] / "src/exgentic/benchmarks/bfcl/bfcl_benchmark.py",
        "AppWorld environment": snapshots["ace"] / "src/exgentic/benchmarks/appworld/appworld_benchmark.py",
        "Terminal-Bench environment": (
            snapshots["ace"] / "src/exgentic/benchmarks/terminalbench2/terminalbench2_benchmark.py"
        ),
        "Terminal-Bench budgets": snapshots["ace"] / "src/exgentic/benchmarks/terminalbench2/terminus_limits.py",
    }
    source_refs = {title: sources.add(path, title, text=True) for title, path in source_files.items()}
    runner_versions = [
        {
            **version,
            "source": sources.add(version["path"], f"Continuation runner revision {version['revision']}", text=True),
            "current_sha256": sha(version["path"]),
            "matches_recorded_hash": sha(version["path"]) == version["sha256"],
        }
        for version in plan.get("runner_revisions", [])
    ]
    epoch1_protocols = [
        {
            "path": path,
            "source": sources.add(path, "Epoch 1 source launch plan"),
            "run_ids": [job["id"] for job in plan["jobs"] if job["source_plan"] == path],
            "config": {
                key: original.get(key)
                for key in (
                    "created_at",
                    "max_parallel",
                    "schedule",
                    "base_url",
                    "model",
                    "temperature_all_roles",
                    "optimizer_seed",
                    "actor_input_tokens",
                    "actor_output_tokens",
                    "turns",
                    "server_command",
                    "server_max_num_seqs",
                    "gpu",
                    "appworld_policy_sha256",
                    "manifest_sha256",
                    "snapshot",
                    "total_training_limit",
                )
            },
        }
        for path, original in original_plans.items()
    ]
    model_cache = Path("/home/thskadud/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots")
    generation_configs = [
        {
            "revision": path.parent.name,
            "config": read(path),
            "source": sources.add(path, text=True),
            "architecture": read(path.with_name("config.json")),
            "architecture_source": sources.add(path.with_name("config.json"), text=True),
        }
        for path in sorted(model_cache.glob("*/generation_config.json"))
    ]
    serving_evidence = {"generation_configs": generation_configs, "model_revision_pinned_in_launcher": False}
    for path in Path("/home/thskadud/.vllm/lib").glob("python*/site-packages/vllm-*.dist-info/METADATA"):
        serving_evidence["installed_vllm_version"] = next(
            (line.removeprefix("Version: ") for line in path.read_text().splitlines() if line.startswith("Version: ")),
            None,
        )
    log_path = Path("/home/thskadud/output_nohup_vllm_server_qwen_gpu.log")
    if log_path.exists():
        with log_path.open(errors="replace") as stream:
            for line in stream:
                marker = "Default vLLM sampling parameters have been overridden"
                if marker in line:
                    serving_evidence["last_server_log_excerpt"] = line[line.index(marker) :].strip()
                    serving_evidence["last_server_log_path"] = str(log_path)
                    break
    ace_defaults = primitive_defaults(source_files["ACE defaults"], "ACEAgent")
    upstreams = tomllib.loads(source_files["Upstream versions"].read_text())
    audit_dir = cohort / "token_audit_20261111"
    token_tables = {}
    for path in [
        audit_dir / "01-method-token-summary.csv",
        audit_dir / "epoch1/e1-01-train-vs-test.csv",
        audit_dir / "epoch1/e1-03-stages.csv",
        audit_dir / "epoch1/e1-04-frozen-checkpoint-evaluation.csv",
    ]:
        if path.exists():
            with path.open(encoding="utf-8-sig") as stream:
                token_tables[path.name] = list(csv.reader(stream))
    benchmark_stats = {}
    for bench in manifest["benchmarks"]:
        rows = [t for t in tasks if t["benchmark"] == bench]
        distributions = {}
        for field in ("primary_domain", "difficulty", "category", "variant"):
            if any(field in t["meta"] for t in rows):
                distributions[field] = {
                    s: dict(Counter(str(t["meta"].get(field, "unknown")) for t in rows if t["split"] == s))
                    for s in ("train", "val", "test", "unused")
                }
        benchmark_stats[bench] = {
            "name": BENCH_NAMES[bench],
            "pool": len(rows),
            "counts": dict(Counter(t["split"] for t in rows)),
            "distributions": distributions,
            "source": manifest["sources"][bench],
            "validation": validation["benchmarks"][bench],
        }
    settings = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "title": "Qwen3.6 실험 설정",
        "cohort": str(cohort),
        "journal_id": plan["journal_id"],
        "methods": METHOD_NAMES,
        "protocol": {k: v for k, v in plan.items() if k not in {"environment", "jobs"}},
        "manifest": manifest,
        "ordered_manifest": read(ordered),
        "manifest_sha256": sha(ordered),
        "benchmark_stats": benchmark_stats,
        "runs": runs,
        "prompts": prompts,
        "examples": examples,
        "ace_defaults": ace_defaults,
        "skillopt_native": native_configs,
        "upstreams": clean(upstreams),
        "source_refs": source_refs,
        "sources": list(sources.entries.values()),
        "token_tables": token_tables,
        "policy": next(p for p in prompts if p["kind"] == "environment"),
        "historical_serving_preflight": read(cohort / "serving_preflight.json"),
        "serving_evidence": serving_evidence,
        "epoch1_protocols": clean(epoch1_protocols),
        "runner_versions": runner_versions,
    }
    common_path = REPO / "src/exgentic/actor_context.py"
    if common_path.exists():
        common = runpy.run_path(str(common_path))
        mode = common["DEFAULT_ACTOR_CONTEXT"]
        preflight_path = REPO / "outputs/shared_actor_context_preflight_20261111/verification.json"
        settings["next_actor_context"] = {
            "version": mode,
            "source": sources.add(common_path, "Common actor context · new experiments", text=True),
            "metadata": common["actor_context_metadata"](mode, list(BENCH_NAMES)),
            "initial_prompts": {bench: common["shared_actor_prompt"](bench) for bench in BENCH_NAMES},
            "preflight": (
                {k: v for k, v in read(preflight_path).items() if k != "rows"} if preflight_path.exists() else None
            ),
        }
        settings["sources"] = list(sources.entries.values())
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("settings.json", clean(settings)), ("tasks.json", tasks)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    verification = {
        "task_count": len(tasks),
        "selected_tasks": sum(t["split"] != "unused" for t in tasks),
        "run_count": len(runs),
        "prompt_count": len(prompts),
        "source_count": len(sources.entries),
        "all_instructions_present": True,
        "epoch2_orders_verified": True,
        "manifest_sha256": sha(ordered),
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.cohort, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
