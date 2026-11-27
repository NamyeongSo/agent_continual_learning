# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Compare original roots with sequential CP50/130 policies on fixed BFCL histories."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.premise import run_bfcl_ancestry_policy_shift as common  # noqa: E402

METHODS = ("ace", "gepa", "skillopt", "premise")
DISPLAY = {"ace": "ACE", "gepa": "GEPA", "skillopt": "SkillOpt", "premise": "PREMiSE"}


def selected_prompts(method: str, seed: int, expected: set[str], sources: common.Sources,
                     checkpoint: int = 50) -> tuple[str, str, dict]:
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.prompts.generator import build_generator_system_prompt

    team = ROOT / "team_prompt"
    if checkpoint not in (50, 130):
        raise ValueError("Only CP50 and CP130 are supported")
    stage, export = ("bfcl", "cp050_bfcl") if checkpoint == 50 else ("terminalbench2", "cp130_terminalbench2")

    def validate_final_stage(config, tasks, field):
        if (config[field], config["mode"], config["seed"]) != (common.MODEL, "sequential", seed):
            raise ValueError("Final-stage model, mode, or seed differs")
        if len(tasks) != 30 or any(r["benchmark"] != "terminalbench2" for r in tasks):
            raise ValueError("CP130 must finish the thirty-task Terminal-Bench stage")

    if method == "gepa":
        run = team / "training_history_20261009/runs" / common.GEPA_RUNS[seed]
        raw = run / "bfcl/raw"
        config, initial = sources.json(raw / "experiment_config.json"), sources.json(raw / "gepa_result.json")
        common.validate_source(config, seed, config["train_task_ids"], expected, "task_model")
        if initial["parents"][0] not in ([], [None]):
            raise ValueError("GEPA initial candidate is not a root")
        result = initial
        if checkpoint == 130:
            raw = run / f"{stage}/raw"
            config, result = sources.json(raw / "experiment_config.json"), sources.json(raw / "gepa_result.json")
            validate_final_stage(config, config["train_task_ids"], "task_model")
        final = result["best_idx"]
        exported = sources.json(team / f"gepa_sequential_seed{seed}" / export / "candidate.json")
        if (result["val_aggregate_scores"][final] != max(result["val_aggregate_scores"])
                or exported["best_candidate"] != result["candidates"][final]):
            raise ValueError("GEPA selected final policy differs from its stage export")
        return (initial["candidates"][0]["system_prompt"], result["candidates"][final]["system_prompt"],
                {"selection": f"{stage} stage best validation candidate at CP{checkpoint}", "candidate_index": final})
    if method == "skillopt":
        run = team / "training_history_20261009/runs" / common.SKILL_RUNS[seed]
        raw = run / "bfcl/raw"
        tasks, summary = sources.json(raw / "task_ids.json"), sources.json(raw / "summary.json")
        common.validate_source(tasks, seed, tasks["tasks"]["train"], expected, "model")
        root = sources.text(raw / "skills/skill_v0000.md")
        if checkpoint == 130:
            raw = run / f"{stage}/raw"
            tasks, summary = sources.json(raw / "task_ids.json"), sources.json(raw / "summary.json")
            validate_final_stage(tasks, tasks["tasks"]["train"], "model")
        final = sources.text(raw / "best_skill.md")
        exported = sources.text(team / f"skillopt_sequential_seed{seed}" / export / "skill.md")
        selected = (sources.text(raw / "steps" / f"step_{summary['best_step']:04d}" / "candidate_skill.md")
                    if summary["best_step"] else sources.text(raw / "skills/skill_v0000.md"))
        if final != exported or final != selected:
            raise ValueError("SkillOpt selected skill differs from the recorded stage best step")
        return root, final, {"selection": f"{stage} stage best skill at CP{checkpoint}",
                             "best_step": summary["best_step"]}
    if method == "premise":
        run = ROOT / "outputs/premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928"
        run = run / f"s{seed}_sequential_t3"
        snapshot = sources.json(run / "checkpoints" / f"checkpoint_{checkpoint:03d}.json")
        config = snapshot["config"]
        tasks = [{"benchmark": b, "task_id": t} for b, t in config["order"][:50]]
        common.validate_source(config, seed, tasks, expected, "model")
        if snapshot["cursor"] != checkpoint or config.get("method", "premise") != "premise":
            raise ValueError("Expected the requested PREMiSE checkpoint")
        playbook = snapshot["states"]["global"]["beam"][0]["playbook"]
        if checkpoint == 50:
            next_step = sources.json(run / "iterations/iter_0051.json")
            parent = sources.json(Path(next_step["parents"][0]["session_dir"]) / "config.json")
            if parent["agent_kwargs"]["playbook"] != playbook:
                raise ValueError("PREMiSE CP50 rank-1 prompt differs from the next recorded parent")
        elif snapshot["states"] != sources.json(run / "state.json")["states"]:
            raise ValueError("PREMiSE CP130 differs from the completed training state")
        return (build_generator_system_prompt(DEFAULT_PLAYBOOK), build_generator_system_prompt(playbook),
                {"selection": f"Post-selection CP{checkpoint} rank-1 beam"})
    if method == "ace":
        root = ROOT / "scripts/ace/outputs/ace_appworld_scorefix_resume_bfcl50_20260928"
        config = sources.json(root / f"sequential_s{seed}/experiment_config.json")
        tasks = [{"benchmark": b, "task_id": t} for b, t in config["task_order"][:50]]
        common.validate_source(config, seed, tasks, expected, "model")
        snapshot = sources.json(root / "test_evaluation/snapshots" / f"sequential_s{seed}"
                                / f"checkpoint_{checkpoint:03d}.json")
        original = sources.raw(Path(snapshot["source_checkpoint"]))
        if (common.digest(original) != snapshot["source_checkpoint_sha256"]
                or snapshot["training_task_ordinal"] != checkpoint or snapshot["store_id"] != "ace_sequential_global"
                or json.loads(original)["playbook"] != snapshot["playbook"]):
            raise ValueError("ACE snapshot differs from its original checkpoint")
        return (build_generator_system_prompt(DEFAULT_PLAYBOOK), build_generator_system_prompt(snapshot["playbook"]),
                {"selection": f"ACE learned context after {checkpoint} sequential training tasks"})
    raise ValueError(f"Unsupported method: {method}")


def prepare(output: Path, base_url: str, workers: int, checkpoint: int = 50, reuse_from: Path | None = None) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    path = output / "prepared.json"
    if path.exists():
        saved = common.read_json(path)
        if (saved["base_url"], saved["workers"], saved["checkpoint"], saved.get("reuse_from")) != (
            base_url, workers, checkpoint, str(reuse_from) if reuse_from else None
        ):
            raise ValueError("Frozen root/final evaluation settings changed")
        for source, expected in saved["source_checks"].items():
            if common.digest(Path(source).read_bytes()) != expected:
                raise ValueError("Frozen root/final source changed")
        return saved
    sources = common.Sources()
    bank = ROOT / "outputs/bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/qwen/history_bank.json"
    anchors = sources.json(bank)["anchors"]
    splits = sources.json(ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json")
    for anchor in anchors:
        if common.digest(sources.raw(Path(anchor["path"]))) != anchor["sha256"]:
            raise ValueError("Existing Qwen BFCL history changed")
    cohorts = []
    for method in METHODS:
        for seed in common.SEEDS:
            tasks = set(validate_task_split(splits[f"bfcl_{seed}"], "bfcl")["train"])
            if {a["task_id"] for a in anchors if a["seed"] == seed} != tasks:
                raise ValueError("History bank differs from the frozen training task set")
            root, final, selection = selected_prompts(method, seed, tasks, sources, checkpoint)
            nodes = []
            for kind, prompt in (("root", root), ("final", final)):
                node = {"id": kind, "prompt": prompt, "prompt_sha256": common.digest(prompt.encode())}
                frozen = output / "policies" / method / f"s{seed}" / (kind + ".json")
                common.save_immutable(frozen, node)
                node.update(frozen_path=str(frozen), frozen_sha256=common.digest(frozen.read_bytes()))
                nodes.append(node)
            cohorts.append({"method": method, "seed": seed, "key": f"{method}_s{seed}", "nodes": nodes,
                            "selection": selection})
    unique = {(a["key"], n["prompt_sha256"]) for c in cohorts for a in anchors if a["seed"] == c["seed"]
              for n in c["nodes"]}
    prepared = {"model": common.MODEL, "base_url": base_url, "workers": workers,
                "benchmark": "BFCL", "mode": "sequential", "checkpoint": checkpoint,
                "histories": anchors, "cohorts": cohorts, "source_checks": sources.checks,
                "unique_probes": len(unique), "policy_instances": 2 * len(cohorts), "environment_execution": False,
                "scope": f"Original root versus sequential CP{checkpoint} on fixed BFCL histories",
                "reuse_from": str(reuse_from) if reuse_from else None,
                "created_at": common.timestamp()}
    common.save_immutable(path, prepared)
    return prepared


def reuse_cached_probes(output: Path, prepared: dict) -> dict:
    manifest = output / "cache_import.json"
    if manifest.exists():
        return common.read_json(manifest)
    source = prepared.get("reuse_from")
    if source is None:
        return {"imported": 0}
    source = Path(source)
    previous = common.read_json(source / "prepared.json")
    if (previous["model"] != common.MODEL
            or common.read_json(source / "status.json")["phase"] not in ("complete", "complete_with_errors")):
        raise ValueError("Reuse requires a completed study of the same Qwen model")
    eligible = {(a["key"], n["prompt_sha256"]): (a, n)
                for c in prepared["cohorts"] for a in prepared["histories"] if a["seed"] == c["seed"]
                for n in c["nodes"]}
    imported = []
    for anchor, node in eligible.values():
        original = common.probe_path(source, anchor, node["prompt_sha256"])
        target = common.probe_path(output, anchor, node["prompt_sha256"])
        if not original.exists() or target.exists():
            continue
        raw = original.read_bytes()
        row = json.loads(raw)
        if row["status"] != "completed":
            continue
        body = common.request_body(common.read_json(Path(anchor["path"])), node["prompt"])
        expected_request = common.digest(json.dumps(body, ensure_ascii=False, sort_keys=True).encode())
        if (row["anchor_sha256"], row["prompt_sha256"], row["model"], row["request_sha256"]) != (
            anchor["sha256"], node["prompt_sha256"], common.MODEL, expected_request
        ):
            raise ValueError("Cached input/policy/request does not match the new study")
        row.update(reused_from=str(original), reused_from_sha256=common.digest(raw))
        common.write_json(target, row)
        imported.append({"source": str(original), "source_sha256": common.digest(raw), "target": str(target)})
    result = {"source": str(source), "imported": len(imported), "records": imported,
              "new_inference_requests": prepared["unique_probes"] - len(imported),
              "validation": "Exact history, prompt, requested model, and full request hashes match"}
    common.save_immutable(manifest, result)
    return result


def analyze(output: Path, prepared: dict, finished: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "analysis"
    folder.mkdir(exist_ok=True)
    checkpoint = prepared.get("checkpoint", 50)
    completed = [c for c in prepared["cohorts"] if c["key"] in finished]
    cache = {}
    for c in completed:
        for a in prepared["histories"]:
            if a["seed"] != c["seed"]:
                continue
            for n in c["nodes"]:
                key = (a["key"], n["prompt_sha256"])
                if key in cache:
                    continue
                path = common.probe_path(output, a, n["prompt_sha256"])
                row = common.read_json(path) if path.exists() else {"status": "missing"}
                if row["status"] == "completed":
                    if (row["anchor_sha256"], row["prompt_sha256"], row["model"]) != (
                        a["sha256"], n["prompt_sha256"], common.MODEL
                    ):
                        raise ValueError("Root/final analysis provenance changed")
                    cache[key] = common.compiled_calls(row["predicted_calls"])
    # Use the same valid input intersection for every completed method in a seed.
    common_anchors = {}
    for seed in common.SEEDS:
        required = {n["prompt_sha256"] for c in completed if c["seed"] == seed for n in c["nodes"]}
        common_anchors[seed] = [a for a in prepared["histories"] if a["seed"] == seed
                               and required and all((a["key"], sha) in cache for sha in required)]
    rows = []
    for c in completed:
        root, final = c["nodes"]
        anchors = common_anchors[c["seed"]]
        metrics = common.balanced_pair(anchors, cache, final["prompt_sha256"], root["prompt_sha256"])
        rows.append({"method": c["method"], "seed": c["seed"], "common_histories": len(anchors),
                     "common_tasks": len({a['task_id'] for a in anchors}), **metrics})
    aggregates = []
    for method in METHODS:
        group = [r for r in rows if r["method"] == method and r["strict_accuracy"] is not None]
        if not group:
            continue
        data = {"method": method, "seeds": len(group)}
        for metric in common.METRICS:
            values = [r[metric] for r in group]
            data[metric] = statistics.mean(values)
            data[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(data)
    common.csv_file(folder / "seed_metrics.csv", rows)
    common.csv_file(folder / "method_metrics.csv", aggregates)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(6.8, 2.45))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.89)
    for offset, metric, label, color in (
        (-0.17, "strict_accuracy", r"$1-\mathrm{ACC}$", "#2467A9"),
        (0.17, "strict_f1", r"$1-\mathrm{F1}$", "#D07825")
    ):
        positions, values, errors = [], [], []
        for data in aggregates:
            positions.append(METHODS.index(data["method"]) + offset)
            values.append(1 - data[metric])
            errors.append(data[metric + "_std"])
        if positions:
            axis.bar(positions, values, width=0.29, yerr=errors, color=color, label=label,
                     error_kw={"linewidth": 0.8, "capsize": 2.5}, zorder=3)
            for pos, value, error in zip(positions, values, errors, strict=True):
                axis.text(pos, min(0.965, value + error + 0.025), f"{value:.3f}", ha="center", fontsize=7)
    axis.set_xticks(range(len(METHODS)), [DISPLAY[m] for m in METHODS])
    axis.set_ylim(0, 1)
    axis.set_ylabel(f"Root-CP{checkpoint} policy shift")
    axis.set_title("BFCL", fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    axis.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
    if len(finished) < len(prepared["cohorts"]):
        fig.text(0.99, 0.98, f"Completed seed/method pairs: {len(finished)}/12", ha="right", fontsize=7, color="gray")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(folder / f"root_final_policy_shift.{extension}", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    common.write_json(folder / "summary.json", {"model": common.MODEL, "benchmark": "BFCL", "mode": "sequential",
        "checkpoint": checkpoint, "completed_cohorts": finished, "metrics": aggregates, "seed_metrics": rows,
        "common_histories_by_seed": {str(s): len(a) for s, a in common_anchors.items()},
        "aggregation": "Mean acting inputs within task, equal tasks within seed, then equal seeds",
        "bands": "One sample seed SD", "comparison": f"Each method's original root versus sequential CP{checkpoint}",
        "errors": "All completed methods use the same successful-input intersection within each seed",
        "environment_execution": False, "updated_at": common.timestamp()})
    caption = (
        f"Root--CP{checkpoint} policy shift on BFCL. We compare each method's original initial policy "
        f"with its policy after {checkpoint} sequential training tasks on identical saved Qwen histories, "
        "using strict action agreement "
        "(function names and arguments). Bars show $1-\\mathrm{ACC}$ and $1-\\mathrm{F1}$; error bars are "
        "one sample standard deviation over three seeds. Inputs are averaged within task, then tasks and "
        "seeds receive equal weight. All methods use the same histories with valid root and final outputs."
    )
    (folder / "root_final_policy_shift.tex").write_text(
        "\\begin{figure*}[t]\n  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{figures/root_final_policy_shift.pdf}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:root-final-policy-shift}}\n\\end{{figure*}}\n")


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18018/v1")
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--checkpoint", type=int, choices=(50, 130), default=50)
    parser.add_argument("--reuse-from", type=Path,
                        help="Reuse exact cached input/prompt predictions from a completed study")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Concurrency must be positive")
    previous = ROOT / "outputs/bfcl_sequential_root_final_policy_shift_qwen_queue150_20261013"
    default = (previous if args.checkpoint == 50 else ROOT / "outputs"
               / "bfcl_sequential_root_cp130_policy_shift_qwen_queue150_20261013")
    output = (args.output or default).resolve()
    reuse_from = args.reuse_from or (previous if args.checkpoint == 130 else None)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepared = prepare(output, args.base_url.rstrip("/"), args.workers, args.checkpoint,
                               reuse_from.resolve() if reuse_from else None)
            reuse = reuse_cached_probes(output, prepared)
            if args.prepare_only:
                print(json.dumps({"cohorts": len(prepared["cohorts"]), "unique_probes": prepared["unique_probes"],
                                  "policy_instances": prepared["policy_instances"], "checkpoint": args.checkpoint,
                                  "cached_probes": reuse["imported"],
                                  "new_probes": prepared["unique_probes"] - reuse["imported"]}, indent=2))
                return
            for c in prepared["cohorts"]:
                for n in c["nodes"]:
                    if common.digest(Path(n["frozen_path"]).read_bytes()) != n["frozen_sha256"]:
                        raise ValueError("Frozen root/final policy changed")
            common.write_json(output / "runtime_manifest.json", {"controller_pid": os.getpid(),
                "started_at": common.timestamp(), "base_url": prepared["base_url"], "inference_upstream":
                "http://127.0.0.1:8006/v1", "unique_probes": prepared["unique_probes"], "workers": args.workers,
                "scope": prepared["scope"], "checkpoint": args.checkpoint, "reused_probes": reuse["imported"]})
            finished = []
            for cohort in prepared["cohorts"]:
                response = httpx.get(prepared["base_url"] + "/models", timeout=15)
                response.raise_for_status()
                if common.MODEL.removeprefix("openai/") not in {r["id"] for r in response.json()["data"]}:
                    raise ValueError("Endpoint is no longer serving Qwen")
                print(f"START {cohort['key']}: root versus final only", flush=True)
                asyncio.run(common.probe_cohort(output, prepared, cohort, finished))
                finished.append(cohort["key"])
                common.write_json(output / "status.json", {"phase": "analyzing", "completed_cohorts": finished,
                    "controller_pid": os.getpid(), "updated_at": common.timestamp()})
                analyze(output, prepared, finished)
                print(f"DONE {cohort['key']}", flush=True)
            errors = sum(common.read_json(p)["status"] != "completed"
                         for p in (output / "probe_results").glob("*.json"))
            common.write_json(output / "status.json", {"phase": "complete_with_errors" if errors else "complete",
                "completed_cohorts": finished, "errors": errors, "unique_probes": prepared["unique_probes"],
                "controller_pid": os.getpid(), "updated_at": common.timestamp(), "environment_execution": False})
        except Exception as exc:
            common.write_json(output / "status.json", {"phase": "error", "error": str(exc),
                "controller_pid": os.getpid(), "updated_at": common.timestamp()})
            raise


if __name__ == "__main__":
    main()
