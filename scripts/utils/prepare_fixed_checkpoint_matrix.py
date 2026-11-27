# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Freeze CP0/10-task policies from completed matrix jobs without changing training artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import checkpoint_sources as sources  # noqa: E402
from scripts.utils.checkpoint_queue import Queue, now  # noqa: E402
from scripts.utils.fixed_matrix_worker import validate_final  # noqa: E402
from scripts.utils.run_dual_gpu_experiment_pool import write  # noqa: E402

VARIANTS = {"ace_aud", "gepa_val", "gepa_train_as_val", "skillopt_val", "skillopt_train_as_val",
            "peace_no_alignment_aud"}
ORIGINAL_POLICY = "18c6ea8f3ab5d02a2495657eb77386af53430cc71555379e11ba2535a0856f04"


def initial_fingerprint(spec, content):
    fields = {key: spec.get(key) for key in (
        "method", "benchmark", "model", "temperature", "benchmark_kwargs", "actor_limits",
        "appworld_policy_sha256", "actor_wrapper_sha256",
    )}
    fields.update(content_sha256=sources.digest(content),
                  splits={name: sorted(ids) for name, ids in spec["splits"].items()})
    if spec.get("actor_context", "legacy") != "legacy":
        fields.update(actor_context=spec["actor_context"], actor_context_metadata=spec["actor_context_metadata"])
    return sources.digest(json.dumps(fields, sort_keys=True))


def selected_jobs(dispatch):
    for root in dispatch["sources"]:
        plan = json.loads((Path(root) / "launch_manifest.json").read_text())
        for job in plan["jobs"]:
            if job["variant"] not in VARIANTS:
                continue
            original = plan.get("appworld_policy_sha256") == ORIGINAL_POLICY
            if (job["benchmark"] == "appworld") != original:
                continue
            if not validate_final(job):
                raise ValueError("Training/final-test evidence is incomplete: " + job["id"])
            yield Path(root), plan, job


def first_actor(job, reader):
    folder = Path(job["train_dir"])
    if job["method"] == "premise":
        iteration = reader.json(folder / "iterations/iter_0001.json")
        parent = next(row for row in iteration["parents"] if row)
        path = Path(parent["session_dir"]) / "config.json"
        return path, reader.json(path)
    first = job["train_ids"][0]
    for path in sorted(folder.glob("*/sessions/*/config.json")):
        config = reader.json(path)
        if str(config["task_id"]) == first:
            return path, config
    raise ValueError("Missing first actor configuration: " + job["id"])


def make_stream(root, plan, job, reader):
    snapshot = Path(plan["snapshot"])
    method, benchmark = job["method"], job["benchmark"]
    spec = {"id": sources.digest(str(Path(job["output"]).resolve()))[:20], "root": str(root),
            "output": job["train_dir"] if method in {"gepa", "skillopt"} else job["output"],
            "source_output": job["output"], "variant": root.name, "run_variant": job["variant"],
            "train_dir": job["train_dir"], "snapshot": str(snapshot), "method": method,
            "benchmark": benchmark, "seed": job["seed"], "model": plan["model"], "temperature": 0.7,
            "playbook_edit_mode": job["playbook_edit_mode"], "train_total": job["train_count"],
            "split_sha256": plan["manifest_sha256"],
            "splits": {name: list(map(str, job[f"{name}_ids"])) for name in ("train", "val", "test")},
            "actor_limits": {"input_tokens": 100000, "output_tokens": 2048,
                             "max_turns": 30 if benchmark == "bfcl" else 40}}
    # The evaluation val split is the official held-out split even when training selected on train.
    splits = list(map(set, spec["splits"].values()))
    if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
        raise ValueError("Official evaluation split memberships overlap")
    if method in {"ace", "premise"}:
        path, actor = first_actor(job, reader)
        actor_context = actor["agent_kwargs"].get("actor_context", "legacy")
        source_config = reader.json(Path(job["train_dir"]) / (
            "experiment_config.json" if method == "ace" else "state.json")) if actor_context != "legacy" else {}
        if method == "premise":
            source_config = source_config.get("config", {})
        kwargs = dict(actor["benchmark_kwargs"])
        if method == "ace":
            content = actor["agent_kwargs"].get("initial_playbook")
            if not content:
                path = snapshot / "src/exgentic/agents/ace/playbook_store.py"
                tree = ast.parse(reader.raw(path).decode())
                content = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "DEFAULT_PLAYBOOK" for t in node.targets))
            wrapper = snapshot / "src/exgentic/agents/ace/prompts/generator.py"
        else:
            content = actor["agent_kwargs"]["playbook"]
            wrapper = snapshot / "src/exgentic/agents/premise/instance.py"
    else:
        cfg = reader.json(Path(job["train_dir"]) / "config.json")
        actor_context = cfg.get("actor_context", "legacy")
        source_config = cfg
        kwargs = dict(cfg["benchmark_kwargs"])
        base = snapshot / "scripts/baselines/prompts" / method / "base"
        path = base / ("shared.txt" if method == "gepa" else "seed_skill.md")
        content = reader.raw(path).decode()
        wrapper = base / (benchmark + (".txt" if method == "gepa" else ".md"))
    if actor_context != "legacy":
        spec.update(actor_context=actor_context, actor_context_metadata=source_config["actor_context_metadata"])
        content = ""
        path = Path(job["train_dir"]) / ("state.json" if method == "premise" else
                                        "experiment_config.json" if method == "ace" else "config.json")
        wrapper = snapshot / "src/exgentic/actor_context.py"
    if "env_kwargs" in kwargs:
        kwargs["env_kwargs"] = {k: v for k, v in kwargs["env_kwargs"].items() if k != "experiment_name"}
        if not kwargs["env_kwargs"]:
            kwargs.pop("env_kwargs")
    kwargs["use_cache"] = False
    if benchmark == "appworld":
        policy = reader.raw(Path(job["appworld_policy"])).decode()
        if sources.digest(policy) != ORIGINAL_POLICY:
            raise ValueError("Original AppWorld policy bytes changed")
        kwargs.update(policy_override=policy, interaction_mode="code_generation")
        spec.update(appworld_policy_override=policy, appworld_policy_sha256=ORIGINAL_POLICY)
    spec.update(benchmark_kwargs=kwargs, actor_wrapper_sha256=sources.digest(reader.raw(wrapper)))
    initial = {"point": 0, "content": content, "selection": "Initial baseline context; shared across seeds/variations",
               "source": str(path), "source_sha256": sources.digest(reader.raw(path)),
               "initial_group": initial_fingerprint(spec, content)}
    return spec, initial


def prepare(output, dispatch_path, base_url):
    output.mkdir(parents=True, exist_ok=True)
    reader = sources.Reader()
    prepared = []
    for root, plan, job in selected_jobs(reader.json(dispatch_path)):
        spec, initial = make_stream(root, plan, job, reader)
        snapshots = [initial] + [row for row in sources.checkpoints(spec, reader) if row["point"] % 10 == 0]
        if [row["point"] for row in snapshots] != list(range(0, spec["train_total"] + 1, 10)):
            raise ValueError("Missing requested checkpoint: " + job["id"])
        final = reader.json(Path(job["output"]) / "job_complete.json")
        if job["method"] in {"gepa", "skillopt"}:
            if sources.digest(snapshots[-1]["content"]) != final["prompt_sha256"]:
                raise ValueError("Recovered final policy differs from official final-test policy")
        prepared.append((spec, snapshots))
    if len(prepared) != 54:
        raise ValueError(f"Expected 6 settings x 3 benchmarks x 3 seeds, got {len(prepared)}")
    queue = Queue(output, interval=10, share_initial=True)
    prepared.sort(key=lambda item: (item[0]["method"] not in {"gepa", "skillopt"},
                                    item[0]["run_variant"], item[0]["benchmark"], item[0]["seed"]))
    for spec, snapshots in prepared:
        for snapshot in snapshots:
            queue.register(spec, snapshot)
    with queue.db:
        queue.db.execute("""UPDATE tasks SET priority=(SELECT
            CASE WHEN json_extract(payload,'$.stream.method') IN ('gepa','skillopt') THEN 0 ELSE 1000 END + point
            FROM checkpoints WHERE id=tasks.cp)""")
    initial_count = queue.db.execute("SELECT count(*) FROM checkpoints WHERE point=0 AND canonical=id").fetchone()[0]
    if initial_count != 12:
        raise ValueError(f"Expected one shared initial context per baseline/benchmark (12), got {initial_count}")
    manifest = {"prepared": True, "created_at": now(), "workers": 200, "per_gpu_query_limit": 100,
                "checkpoint_interval": 10, "share_initial": True, "base_url": base_url,
                "stop_when_drained": True, "auto_discover": False,
                "scope_label": ("GEPA·SkillOpt 우선 · CP0 공유 · Original AppWorld · "
                                "PEACE w/o alignment · GPU별100 query"),
                "source_journal_ids": [], "roots": [],
                "sources_manifest": str(dispatch_path),
                "sources_manifest_sha256": sources.digest(reader.raw(dispatch_path)),
                "reuse_existing_final_tests": False}
    old = output / "checkpoint_queue_manifest.json"
    if old.exists():
        previous = reader.json(old)
        for key in ("workers", "base_url", "checkpoint_interval", "share_initial"):
            if previous[key] != manifest[key]:
                raise ValueError("Existing checkpoint evaluation configuration differs")
        manifest = previous
    write(old, manifest)
    write(output / "prepared_streams.json", [{"spec": spec, "checkpoints": snapshots} for spec, snapshots in prepared])
    dashboard = queue.dashboard(workers=200)
    write(output / "dashboard.json", dashboard)
    summary = {"streams": len(prepared), "logical_checkpoints": dashboard["checkpoints"],
               "physical_contexts": dashboard["checkpoints"] - dashboard["skipped"],
               "shared_initial_contexts": initial_count, "task_counts": dashboard["task_counts"],
               "priority": "All GEPA/SkillOpt queries before ACE/PEACE, then increasing CP point"}
    write(output / "preparation_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18008/v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve(), args.sources.resolve(), args.base_url)))
