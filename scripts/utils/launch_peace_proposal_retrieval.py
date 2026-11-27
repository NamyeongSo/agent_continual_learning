# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Fresh BFCL A/B/AB/random retrieval, optionally followed by policy-shift candidate selection."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if (Path(__file__).resolve().parent / "runtime_helpers.py").exists():
    _spec = importlib.util.spec_from_file_location(
        "proposal_frozen_helpers", Path(__file__).parent / "runtime_helpers.py"
    )
    shared = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(shared)
else:
    from scripts.utils import launch_peace_span_policy_shift as shared

MODES = ("a", "b", "ab", "random")


def prepare(output, source, embedding_url, *, candidate_evaluation="environment_replay", past=3, after_output=None):
    output, source = output.resolve(), source.resolve()
    after_output = after_output.resolve() if after_output else None
    if candidate_evaluation not in {"environment_replay", "span_policy_shift"} or past not in {3, 6, 9}:
        raise ValueError("Invalid candidate evaluation or past count")
    if (output / "launch_manifest.json").exists():
        plan = shared.read(output / "launch_manifest.json")
        if (plan["comparison_source"] != str(source) or plan["proposal_embedding_url"] != embedding_url
                or plan.get("candidate_evaluation", "environment_replay") != candidate_evaluation
                or plan.get("past_count", 3) != past
                or plan.get("after_output") != (str(after_output) if after_output else None)):
            raise ValueError("Existing proposal retrieval configuration differs")
        return plan
    output.mkdir(parents=True, exist_ok=True)
    baseline = shared.read(source / "launch_manifest.json")
    originals = {job["seed"]: job for job in baseline["jobs"] if job["variant"] == "peace_no_alignment_aud"}
    if set(originals) != {42, 43, 44}:
        raise ValueError("Expected three matched PEACE seeds")
    predecessors = {}
    if after_output:
        previous = shared.read(after_output / "launch_manifest.json")
        predecessors = {(job.get("proposal_retrieval"), job["seed"]): job for job in previous["jobs"]}
        if set(predecessors) != {(mode, seed) for mode in MODES for seed in (42, 43, 44)}:
            raise ValueError("Expected twelve matching retrieval predecessors")
        if previous["manifest_sha256"] != baseline["manifest_sha256"]:
            raise ValueError("Predecessor split differs")
        for job in predecessors.values():
            if (job.get("epochs") != 1 or job.get("temperature_all_roles") != 0
                    or job["train_ids"] != originals[job["seed"]]["epoch1_order"]):
                raise ValueError("Predecessor training protocol differs")
    snapshot = output / "code_snapshot"
    shared.prepare_snapshot(Path(originals[42]["source_snapshot"]), snapshot)
    for name in (
        "src/exgentic/agents/premise/proposal_retrieval.py",
        "src/exgentic/agents/premise/llm.py",
        "scripts/premise/serve_proposal_embeddings.py",
        "scripts/utils/launch_peace_span_policy_shift.py",
    ):
        shutil.copy2(REPO / name, snapshot / name)
    audit_path = output / "baseline_actor_audit.json"
    audit = shared.read(audit_path)
    llm_path = "src/exgentic/agents/premise/llm.py"
    audit[llm_path]["run_sha256"] = shared.sha(snapshot / llm_path)
    audit[llm_path]["change"] = "Optional historical evidence appended only to mutation prompt"
    source_llm = (Path(originals[42]["source_snapshot"]) / llm_path).read_text()
    run_llm = (snapshot / llm_path).read_text()

    def reflection_source(code):
        node = next(
            node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.FunctionDef) and node.name == "reflect"
        )
        return ast.get_source_segment(code, node)

    if reflection_source(source_llm) != reflection_source(run_llm):
        raise ValueError("The baseline reflection prompt/implementation changed")
    audit[llm_path]["reflect_function_identical"] = True
    shared.write(audit_path, audit)
    runner = output / "proposal_runner.py"
    shutil.copy2(__file__, runner)
    shutil.copy2(REPO / "scripts/utils/launch_peace_span_policy_shift.py", output / "runtime_helpers.py")
    manifest = output / "task_splits.json"
    shutil.copy2(baseline["manifest"], manifest)
    if shared.sha(manifest) != baseline["manifest_sha256"]:
        raise ValueError("Comparison task split changed")
    jobs = []
    for mode in MODES:
        for seed in (42, 43, 44):
            original = originals[seed]
            variant = f"peace_proposal_{mode}_aud" + (
                f"_span_policy_shift_p{past}" if candidate_evaluation == "span_policy_shift" else "")
            directory = output / "runs" / variant / "bfcl" / f"s{seed}"
            (directory / "train").mkdir(parents=True, exist_ok=False)
            job = copy.deepcopy(original)
            job.update(
                id=f"{variant}_bfcl_s{seed}",
                variant=variant,
                output=str(directory),
                train_dir=str(directory / "train"),
                source_snapshot=str(snapshot),
                template_job_id=original["id"],
                template_source_plan=original["source_plan"],
                source_plan=str(output / "launch_manifest.json"),
                proposal_retrieval=mode,
                proposal_embedding_url=embedding_url,
                candidate_evaluation=candidate_evaluation,
                policy_shift_temperature=0.0,
                alignment_scoring=False,
                train_count=50,
                train_ids=original["epoch1_order"],
                epoch1_count=50,
                epoch2_order=[],
                epochs=1,
                past_count=past,
                current_count=1,
                replay_workers=28,
                llm_calls=28,
                test_workers=28,
                temperature_all_roles=0.0,
                from_scratch=True,
                restored_tasks=0,
                predecessor=None,
                release_delay_seconds=0,
                port_block_start=(14000 if after_output else 28000) + len(jobs) * 500,
                status="pending",
            )
            job.pop("source_train_dir", None)
            if after_output:
                prior = predecessors[mode, seed]
                job["external_predecessor"] = {key: prior[key] for key in (
                    "id", "method", "output", "train_dir", "train_count", "test_ids")}
                job["external_predecessor_plan"] = str(after_output / "launch_manifest.json")
            job["source_spec"].update(
                id=hashlib.sha256(str(directory).encode()).hexdigest()[:20],
                root=str(output),
                output=str(directory),
                source_output=str(directory),
                train_dir=job["train_dir"],
                variant=output.name,
                run_variant=variant,
                snapshot=str(snapshot),
                temperature=0.0,
                train_total=50,
                proposal_retrieval=mode,
                candidate_evaluation=candidate_evaluation,
                past_count=past,
                epochs=1,
            )
            job["train_command"] = [
                str(snapshot / ".venv/bin/python"),
                "-u",
                str(runner),
                "train",
                "--output",
                str(output),
                "--job",
                job["id"],
            ]
            job["command"] = [
                str(REPO / ".venv/bin/python"),
                "-u",
                str(runner),
                "worker",
                "--output",
                str(output),
                "--job",
                job["id"],
            ]
            shared.write(
                directory / "fresh_start_provenance.json",
                {
                    "from_scratch": True,
                    "restored_tasks": 0,
                    "copied_training_artifacts": [],
                    "epochs": 1,
                    "task_order": job["train_ids"],
                    "proposal_retrieval": mode,
                    "proposal_examples": 2,
                    "past_count": past,
                    "replay_workers": 28,
                    "llm_calls": 28,
                    "candidate_evaluation": candidate_evaluation,
                    "external_predecessor": job.get("external_predecessor"),
                },
            )
            jobs.append(job)
    plan = {
        "created_at": shared.now(),
        "comparison_source": str(source),
        "comparison_manifest_sha256": shared.sha(source / "launch_manifest.json"),
        "model": baseline["model"],
        "mode": "isolated",
        "epochs": 1,
        "from_scratch": True,
        "temperature_all_roles": 0.0,
        "playbook_edit_mode": "add_update_delete",
        "max_parallel": 12,
        "manifest": str(manifest),
        "manifest_sha256": shared.sha(manifest),
        "base_url": baseline["base_url"],
        "environment": baseline["environment"],
        "proposal_embedding_url": embedding_url,
        "candidate_evaluation": candidate_evaluation,
        "policy_shift_temperature": 0.0,
        "past_count": past,
        "after_output": str(after_output) if after_output else None,
        "completion_gate": "Each matched predecessor: full training and final test with zero infrastructure errors",
        "final_evaluation_splits": ["train", "val", "test"],
        "jobs": jobs,
    }
    shared.write(output / "launch_manifest.json", plan)
    shared.write(output / "protocol_public.json", {key: value for key, value in plan.items() if key != "environment"})
    shared.write(output / "progress.json", {"jobs": {job["id"]: {"status": "pending"} for job in jobs}})
    shared.write(output / "status.json", {"phase": "planned", "at": shared.now()})
    shared.write(
        output / "code_hashes.json",
        {
            "snapshot": {
                str(path.relative_to(snapshot)): shared.sha(path)
                for name in ("src", "scripts")
                for path in (snapshot / name).rglob("*.py")
            },
            "runner_sha256": shared.sha(runner),
            "runtime_helpers_sha256": shared.sha(output / "runtime_helpers.py"),
        },
    )
    return plan


def train(plan, job, check=False):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    original = next(
        row for row in shared.read(job["template_source_plan"])["jobs"] if row["id"] == job["template_job_id"]
    )
    argv = [arg for arg in original["train_command"][3:] if arg != "--resume-sessions"]
    for flag, value in (
        ("--output-dir", job["train_dir"]),
        ("--temperature", "0.0"),
        ("--task-split-manifest", plan["manifest"]),
        ("--base-url", job["base_url"]),
        ("--past", str(job["past_count"])),
        ("--replay-workers", "28"),
        ("--llm-calls", "28"),
    ):
        argv[argv.index(flag) + 1] = value
    argv.extend(
        ["--proposal-retrieval", job["proposal_retrieval"], "--proposal-embedding-url", job["proposal_embedding_url"]]
    )
    argv.extend(["--candidate-evaluation", job["candidate_evaluation"], "--policy-shift-temperature", "0.0"])
    if check:
        argv.append("--dry-run")
    from exgentic.agents.premise.pipeline import PremisePipeline

    original_restore = PremisePipeline._restore_or_initialize

    def restore(self):
        self.order = [("bfcl", task) for task in job["train_ids"]]
        return original_restore(self)

    PremisePipeline._restore_or_initialize = restore
    path = snapshot / "scripts/premise/run_experiment.py"
    sys.argv = [str(path), *argv]
    shared.module_file("proposal_frozen_peace", path).main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "train", "check"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18886")
    parser.add_argument("--candidate-evaluation", choices=("environment_replay", "span_policy_shift"),
                        default="environment_replay")
    parser.add_argument("--past", type=int, choices=(3, 6, 9), default=3)
    parser.add_argument("--after-output", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        plan = prepare(output, args.source, args.embedding_url, candidate_evaluation=args.candidate_evaluation,
                       past=args.past, after_output=args.after_output)
        print(json.dumps({"output": str(output), "jobs": len(plan["jobs"])}))
        return
    plan = shared.read(output / "launch_manifest.json")
    if args.action == "run":
        shared.REPO = REPO
        shared.run(output, plan)
    elif args.action == "worker":
        os.environ.update(shared.environment(plan, next(job for job in plan["jobs"] if job["id"] == args.job)))
        shared.worker(plan, next(job for job in plan["jobs"] if job["id"] == args.job))
    else:
        train(plan, next(job for job in plan["jobs"] if job["id"] == args.job), check=args.action == "check")


if __name__ == "__main__":
    main()
