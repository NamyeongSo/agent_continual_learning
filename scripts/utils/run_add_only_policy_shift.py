# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate ACE/PEACE ADD checkpoints on an existing shared frozen history bank."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.utils import run_checkpoint_policy_shift as common  # noqa: E402
from scripts.utils import run_pairwise_policy_shift as pairwise  # noqa: E402
from scripts.utils.plot_pairwise_policy_shift_interim import analyze_partial, now, plot, write_csv  # noqa: E402
from scripts.utils.policy_shift_metrics import canonical, digest, load_json, write_json  # noqa: E402
from scripts.utils.schedule_checkpoint_policy_shift import api  # noqa: E402

COHORT = "qwen36_isolated_ace_premise_temp07_20261021_025529"


def frozen_renderer(snapshot: Path, checks: dict):
    """Load the original ADD renderer and its utility module in an isolated namespace."""
    base = snapshot / "src/exgentic/agents/ace"
    package_name = "_add_policy_renderer_" + digest(str(base))[:12]
    for name in (package_name, package_name + ".prompts"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    for suffix, relative in (("playbook_utils", "playbook_utils.py"), ("prompts.generator", "prompts/generator.py")):
        path = base / relative
        common.frozen_source(path, checks)
        spec = importlib.util.spec_from_file_location(package_name + "." + suffix, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module.build_generator_system_prompt


def prepare(output: Path, source: Path, queue: Path):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    previous = load_json(source / "plan.json")
    original_prepared = load_json(source / "prepared.json")
    if previous["temperature"] != 0.0:
        raise ValueError("Expected the completed temperature-zero shared-history study")
    if load_json(source / "status.json")["phase"] not in ("completed", "completed_with_errors"):
        raise ValueError("Source study is incomplete")

    with common.readonly_queue(queue) as queue_db:
        invalid = {row[0] for row in queue_db.execute("SELECT stream FROM stream_invalidations")}
        specs = [json.loads(row[0]) for row in queue_db.execute("SELECT spec FROM streams")]
        specs = [
            s
            for s in specs
            if Path(s["root"]).name == COHORT
            and s["id"] not in invalid
            and s["method"] in ("ace", "premise")
            and not s.get("appworld_policy_sha256")
        ]
        expected = {
            (b, m, s) for b in ("bfcl", "appworld", "terminalbench2") for m in ("ace", "premise") for s in (42, 43, 44)
        }
        if len(specs) != 18 or {(s["benchmark"], s["method"], s["seed"]) for s in specs} != expected:
            raise ValueError("Missing or ambiguous original-policy ADD-only streams")
        if any(s.get("playbook_edit_mode") != "add_only" for s in specs):
            raise ValueError("A selected stream is not ADD-only")
        checkpoints = {
            s["id"]: [
                dict(row)
                for row in queue_db.execute("SELECT * FROM checkpoints WHERE stream=? ORDER BY point", (s["id"],))
            ]
            for s in specs
        }

    for spec in specs:
        expected_points = list(range(5, 31 if spec["benchmark"] == "terminalbench2" else 51, 5))
        if [row["point"] for row in checkpoints[spec["id"]]] != expected_points:
            raise ValueError(f"Incomplete checkpoint prefix for {spec['id']}")

    output.mkdir(parents=True)
    checks = {}
    db = common.own_db(output)
    with __import__("sqlite3").connect(
        (source / "policy_shift.sqlite3").as_uri() + "?mode=ro", uri=True
    ) as original_db:
        original_db.row_factory = __import__("sqlite3").Row
        for row in original_db.execute("SELECT * FROM anchors"):
            content = Path(row["file"]).read_bytes()
            if digest(canonical(json.loads(content))) != row["id"]:
                raise ValueError("Frozen shared history changed")
            path = output / "histories" / f"{row['id']}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
            db.execute(
                "INSERT INTO anchors VALUES(?,?,?,?,?)",
                (row["id"], row["benchmark"], row["seed"], row["task"], str(path)),
            )

    renderers = {}
    for spec in specs:
        snapshot = Path(spec["snapshot"])
        render = renderers.setdefault(str(snapshot), frozen_renderer(snapshot, checks))
        initial = common.initial_content(spec, checks)
        policies = [(0, initial, "ADD original initial policy")]
        for checkpoint in checkpoints[spec["id"]]:
            raw = common.frozen_source(checkpoint["file"], checks)
            payload = json.loads(raw)
            if digest(payload["content"]) != checkpoint["sha"]:
                raise ValueError("Checkpoint content SHA differs from the queue")
            policies.append((checkpoint["point"], payload["content"], payload["selection"]))
        for point, content, selection in policies:
            prompt = content if spec["benchmark"] == "terminalbench2" else render(content)
            sha = digest(prompt)
            db.execute("INSERT OR IGNORE INTO prompts VALUES(?,?)", (sha, prompt))
            db.execute(
                "INSERT INTO policies VALUES(?,?,?,?,?,?)",
                (spec["id"], point, spec["benchmark"], spec["method"] + "_add", spec["seed"], sha),
            )
            write_json(
                output / "policies" / spec["id"] / f"cp{point:03d}.json",
                {
                    "point": point,
                    "content": content,
                    "prompt": prompt,
                    "selection": selection,
                    "original_method": spec["method"],
                    "edit_mode": "add_only",
                },
            )
        db.commit()

    plan = {
        key: value
        for key, value in previous.items()
        if key not in {"journal_id", "command", "deduplication", "analysis_corrections", "supervisor_command"}
    }
    plan.update(
        output=str(output),
        created_at=now(),
        temperature=0.0,
        workers=128,
        queue_output=str(queue),
        selected_stream_ids=[s["id"] for s in specs],
        minimum_checkpoint_points={s["id"]: [c["point"] for c in checkpoints[s["id"]]] for s in specs},
        scope="ACE ADD and PEACE ADD, original environment policy, seeds42/43/44, all three benchmarks",
        source_history_bank=str(source),
        history_bank=previous["history_bank"],
        history_selection=previous["history_selection"],
        comparison="all_within_seed_checkpoint_pairs_by_distance",
        source_gate="All selected frozen ADD checkpoints are valid and complete; "
        "environment CP-evaluation jobs are independent of this action-only study",
        request_sampling="One fresh draw per ADD stream/distinct prompt/history plus independent CP0 repeat; "
        "exact unchanged CPs share the draw",
        environment_execution=False,
        learning=False,
        playbook_edit_mode="add_only",
        reused_response_count=0,
        paired_comparison_source=str(source),
        limitations="Shared ACE source histories retained by explicit user request; "
        "no new per-method CP0 history rollouts. Errors remove agreement observations.",
    )
    parser = output / "frozen/terminus_json_plain_parser.py"
    parser.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(previous["terminal_parser_path"], parser)
    if digest(parser.read_bytes()) != previous["terminal_parser_sha256"]:
        raise ValueError("Frozen Terminal-Bench parser changed")
    plan["terminal_parser_path"] = str(parser)
    pairwise.build_jobs(db, plan)
    actual = db.execute("SELECT count(*) FROM jobs").fetchone()[0]
    naive = (
        db.execute("SELECT count(*) FROM bindings").fetchone()[0]
        + 2 * db.execute("SELECT count(*) FROM bindings WHERE point=0").fetchone()[0]
    )
    # bindings already include the CP0 repeat; the additional naive cost is one CP0 primary per stream/history.
    naive -= db.execute("SELECT count(*) FROM bindings WHERE point=0").fetchone()[0]
    prepared = {
        "histories": db.execute("SELECT count(*) FROM anchors").fetchone()[0],
        "policy_instances": db.execute("SELECT count(*) FROM policies").fetchone()[0],
        "distinct_requests": actual,
        "effective_requests": actual,
        "deduplicated_request_count": naive - actual,
        "source_hashes": checks,
        "history_bank_sha256": original_prepared["protocol_sha256"],
        "streams": specs,
        "reused_response_count": 0,
        "learning": False,
        "environment_execution": False,
    }
    db.close()
    shutil.copy2(source / "analysis/summary.json", output / "frozen/comparison_baseline_summary.json")
    shutil.copy2(
        source / "analysis/interim/20261103_114046/summary.json",
        output / "frozen/comparison_baseline_available_pairs.json",
    )
    write_json(output / "source_scope.json", specs)
    write_json(output / "prepared.json", prepared)
    write_json(
        output / "analysis/deduplication.json",
        {
            "effective_requests": actual,
            "skipped_identical_prompt_requests": naive - actual,
            "reused_prior_temperature_zero_responses": 0,
            "cp0_repeat": "independent",
        },
    )
    snapshot = output / "code_snapshot"
    shutil.copytree(source / "code_snapshot", snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (
        "run_add_only_policy_shift.py",
        "run_pairwise_policy_shift.py",
        "run_checkpoint_policy_shift.py",
        "plot_pairwise_policy_shift_interim.py",
        "policy_shift_metrics.py",
        "supervise_pairwise_policy_shift.py",
        "schedule_checkpoint_policy_shift.py",
    ):
        shutil.copy2(ROOT / "scripts/utils" / name, snapshot / "scripts/utils" / name)
    plan["command"] = [
        sys.executable,
        "-u",
        str(snapshot / "scripts/utils/run_add_only_policy_shift.py"),
        "--output",
        str(output),
    ]
    plan["supervisor_command"] = [
        sys.executable,
        "-u",
        str(snapshot / "scripts/utils/supervise_pairwise_policy_shift.py"),
        "--output",
        str(output),
    ]
    plan["progress_plot_command"] = plan["command"] + ["--plot-only"]
    write_json(
        output / "code_hashes.json",
        {str(p.relative_to(snapshot)): digest(p.read_bytes()) for p in snapshot.rglob("*") if p.is_file()},
    )
    record = api(
        "/api/experiments",
        {
            "title": "Qwen3.6 · ACE/PEACE ADD vs AUD · shared-history ΔCP shift · temperature0 · queue128",
            "purpose": "기존 shared history에서 ADD-only checkpoints를 재추론하고 AUD/GEPA/SkillOpt 곡선과 비교",
            "status": "planned",
            "method": "ACE ADD / PEACE ADD; comparison ACE AUD / PEACE AUD / GEPA / SkillOpt",
            "benchmark": "BFCL, AppWorld, Terminal-Bench",
            "model": plan["model"],
            "seeds": "42,43,44",
            "run_path": str(output),
            "split": "Same 6504 frozen ACE history contexts as " + str(source),
            "budget": "temperature0, thinkingoff, 128 concurrent actor requests; input100k/output2048; "
            "no learning or environment execution",
            "command": " ".join(plan["command"]),
            "summary": f"준비 완료: 18 streams, {prepared['policy_instances']} policies, "
            f"{actual:,} fresh inference requests.",
            "limitations": plan["limitations"],
            "tags": ["policy-shift", "add-only", "aud", "shared-history", "qwen36"],
        },
    )
    plan["journal_id"] = record["id"]
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"phase": "prepared", "workers": 128, "counts": {"pending": actual}})
    print(
        json.dumps(
            {
                "output": str(output),
                "journal_id": record["id"],
                **{
                    key: prepared[key]
                    for key in ("histories", "policy_instances", "effective_requests", "deduplicated_request_count")
                },
            },
            ensure_ascii=False,
        )
    )


def combine_and_plot(output: Path, own: dict, *, interim: bool):
    base = load_json(
        output
        / "frozen"
        / ("comparison_baseline_available_pairs.json" if interim else "comparison_baseline_summary.json")
    )
    combined = {
        **own,
        "scores": base["scores"] + own["scores"],
        "generated_at": now(),
        "comparison_baseline": str(output / "frozen/comparison_baseline_summary.json"),
    }
    if interim:
        if not own["scores"]:
            return None
        combined["phase"] = "interim"
        folder = output / "analysis/interim" / now()[:19].replace("-", "").replace(":", "").replace("T", "_")
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "summary.json", combined)
        plot(combined, folder)
        return folder
    combined["phase"] = "completed"
    write_json(output / "analysis/comparison_summary.json", combined)
    write_csv(output / "analysis/comparison_distance_metrics.csv", combined["scores"])
    plot(combined, output / "analysis", "delta_cp_policy_shift")
    own_folder = output / "analysis/add_only"
    own_folder.mkdir(exist_ok=True)
    plot(own, own_folder, "delta_cp_add_only")
    return output / "analysis"


def run(output: Path):
    plan = load_json(output / "plan.json")
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, sha in load_json(output / "prepared.json")["source_hashes"].items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError("Frozen ADD source changed: " + path)
        asyncio.run(common.probe(output, plan))
        own = pairwise.analyze(output, plan)
        combine_and_plot(output, own, interim=False)
        write_json(
            output / "status.json",
            {"phase": own["status"], "pid": os.getpid(), "counts": own["counts"], "workers": plan["workers"]},
        )


def plot_progress(output: Path):
    own, _pair_rows, _task_rows = analyze_partial(output / "policy_shift.sqlite3")
    folder = combine_and_plot(output, own, interim=True)
    if folder:
        plan = load_json(output / "plan.json")
        for path in [folder / "summary.json", folder / "correlations.csv", *sorted(folder.glob("*.png"))]:
            api(
                "/api/experiments/" + plan["journal_id"] + "/import",
                {
                    "path": str(path.resolve()),
                    "caption": "ADD-only 진행 중 비교 그래프; completed same-history pairs, Pearson r in legend",
                },
            )
        write_json(output / "latest_plot.json", {"at": now(), "folder": str(folder), "phase": "interim"})
        print(json.dumps({"plot": str(folder), "completed": own["completed_requests"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-source", type=Path)
    parser.add_argument("--queue-output", type=Path, default=ROOT / "outputs/qwen_five_task_cp_queue_20261022")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_source:
        prepare(output, args.prepare_source.resolve(), args.queue_output.resolve())
    if args.plot_only:
        plot_progress(output)
    elif not args.prepare_only:
        run(output)


if __name__ == "__main__":
    main()
