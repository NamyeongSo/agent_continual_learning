# SPDX-License-Identifier: Apache-2.0
"""Continue immutable epoch-one runs, preserving their frozen actors and optimizers.

GEPA reconstructs RNG/candidates from read-only cached decisions, failing closed
on any prefix cache miss. SkillOpt resumes its native epoch boundary checkpoint;
its within-epoch rejection buffer resets normally at the new epoch boundary.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("ace_aud", "gepa_val", "skillopt_val", "peace_no_alignment_aud")
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def second_order(first, seed):
    second = list(first)
    random.Random(seed).shuffle(second)
    return second


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_upstream(source, method, count):
    valid = []
    for path in (source / "upstream").glob("replay_*"):
        if method == "skillopt":
            marker = path / "runtime_state.json"
            if marker.exists() and read(marker).get("last_completed_step") == count // 5:
                valid.append(path)
        elif (path / "gepa_state.bin").exists():
            valid.append(path)
    if not valid:
        raise ValueError(f"No completed optimizer checkpoint: {source}")
    return sorted(valid)[-1]


def prepare(output, queue):
    output = output.resolve()
    if (output / "launch_manifest.json").exists():
        return read(output / "launch_manifest.json")
    with sqlite3.connect(f"file:{queue.resolve()}?mode=ro", uri=True) as connection:
        specs = [json.loads(row[0]) for row in connection.execute("SELECT spec FROM streams")]
    specs = [s for s in specs if s["run_variant"] in VARIANTS]
    assert len(specs) == 36
    jobs = []
    environments = []
    for spec in sorted(
        specs, key=lambda s: (BENCHMARKS.index(s["benchmark"]), VARIANTS.index(s["run_variant"]), s["seed"])
    ):
        original_plan = read(Path(spec["root"]) / "launch_manifest.json")
        original = next(
            j
            for j in original_plan["jobs"]
            if j["method"] == spec["method"]
            and j["benchmark"] == spec["benchmark"]
            and j["seed"] == spec["seed"]
            and j.get("variant") == spec["run_variant"]
        )
        source = Path(spec["train_dir"])
        method, benchmark, seed, count = spec["method"], spec["benchmark"], spec["seed"], spec["train_total"]
        if method == "ace":
            order = [row[1] for row in read(source / "experiment_config.json")["task_order"]]
            assert read(source / "training_complete.json")["completed_task_count"] == count
        elif method == "premise":
            state = read(source / "state.json")
            assert state["cursor"] == count and state["config"]["alignment_scoring"] is False
            order = [row[1] for row in state["config"]["order"]]
        else:
            config = read(source / "config.json")
            assert read(source / "summary.json")["complete"]
            assert len(list((source / "online").glob("*.json"))) == count
            order = [row["task_id"] for row in config["train"]]
        assert len(order) == len(set(order)) == count
        assert set(order) == set(spec["splits"]["train"])
        repeated = second_order(order, seed)
        destination = output / "runs" / spec["run_variant"] / benchmark / f"s{seed}"
        train = destination / "train"
        train.mkdir(parents=True, exist_ok=True)
        job = {
            **original,
            "output": str(destination),
            "train_dir": str(train),
            "source_train_dir": str(source),
            "source_snapshot": spec["snapshot"],
            "source_plan": str(Path(spec["root"]) / "launch_manifest.json"),
            "source_spec": spec,
            "epoch1_count": count,
            "train_count": 2 * count,
            "epoch1_order": order,
            "epoch2_order": repeated,
            "train_ids": order + repeated,
            "gpu": 1,
            "base_url": "http://127.0.0.1:18006/v1",
            "port_block_start": 42000 + 500 * len(jobs),
            "lane": len(jobs) % 12,
            "predecessor": None,
            "status": "pending",
        }
        provenance = {
            "epoch": 2,
            "source_train_dir": str(source),
            "source_snapshot": spec["snapshot"],
            "source_plan_sha256": sha(job["source_plan"]),
            "split_manifest_sha256": spec["split_sha256"],
            "epoch1_order": order,
            "epoch2_order": repeated,
            "shuffle": "Random(seed).shuffle(epoch1_actual_order.copy())",
            "restored_tasks": count,
            "new_tasks": count,
            "restored_files": {},
        }

        def copy_file(path, relative=None, train=train, provenance=provenance):
            target = train / (relative or path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Refuse partial preparation overwrite: {target}")
            shutil.copy2(path, target)
            provenance["restored_files"][str(path)] = sha(path)

        if method == "ace":
            copy_file(source / f"playbook_ace_isolated_{benchmark}.json")
            copy_file(source / "online_metrics.jsonl")
        elif method == "premise":
            state["config"]["order"] = [[benchmark, task] for task in order + repeated]
            write(train / "state.json", state)
            provenance["restored_files"][str(source / "state.json")] = sha(source / "state.json")
            for path in (source / "iterations").glob("*.json"):
                target = train / "iterations" / path.name
                target.parent.mkdir(exist_ok=True)
                target.symlink_to(path.resolve())
        else:
            for path in (source / "online").glob("*.json"):
                copy_file(path, Path("online") / path.name)
            new_config = {
                **config,
                "output": str(train),
                "base_url": job["base_url"],
                "epochs": 2,
                "termination": "second_epoch_exhausted",
                "epoch1_train": config["train"],
                "train": config["train"] + [{"task_id": task, "split": "train", "epoch": 2} for task in repeated],
            }
            write(train / "config.json", new_config)
            upstream = selected_upstream(source, method, count)
            job["source_upstream"] = str(upstream)
            if method == "skillopt":
                target = train / "upstream" / "continued"
                shutil.copytree(upstream, target)
                state = read(target / "runtime_state.json")
                for key in ("current_skill_path", "best_skill_path"):
                    state[key] = str(target / Path(state[key]).relative_to(upstream))
                write(target / "runtime_state.json", state)
                provenance["restored_files"][str(upstream / "runtime_state.json")] = sha(
                    upstream / "runtime_state.json"
                )
                provenance["skillopt_resume"] = (
                    "native epoch boundary; scheduler current_step preserved, horizon extended to 2 epochs"
                )
        write(destination / "epoch2_provenance.json", provenance)
        python = str(
            Path(spec["snapshot"])
            / (
                f".exgentic/baselines/venvs/{method}/bin/python"
                if method in ("gepa", "skillopt")
                else ".venv/bin/python"
            )
        )
        job["train_command"] = [
            python,
            "-u",
            str(Path(__file__).resolve()),
            "train",
            "--output",
            str(output),
            "--job",
            job["id"],
        ]
        job["command"] = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(Path(__file__).resolve()),
            "worker",
            "--output",
            str(output),
            "--job",
            job["id"],
        ]
        jobs.append(job)
        environments.append(original_plan.get("environment", {}))
    assert all(e == environments[0] for e in environments)
    plan = {
        "created_at": now(),
        "mode": "isolated",
        "epochs": 2,
        "max_parallel": 12,
        "benchmark_order": list(BENCHMARKS),
        "benchmark_barrier": "all 12 full train and final tests complete",
        "seed_stagger_seconds": 0,
        "model": "Qwen/Qwen3.6-35B-A3B",
        "temperature_all_roles": 0.7,
        "base_url": "http://127.0.0.1:18006/v1",
        "environment": environments[0],
        "manifest_sha256": jobs[0]["source_spec"]["split_sha256"],
        "jobs": jobs,
        "source_queue": str(queue.resolve()),
        "continuation_script_sha256": sha(__file__),
    }
    write(output / "launch_manifest.json", plan)
    write(output / "progress.json", {"jobs": {j["id"]: {"status": "pending"} for j in jobs}})
    write(output / "status.json", {"phase": "planned", "at": now()})
    return plan


class PrefixMiss(BaseException):
    """Must not be swallowed as a failed proposal: no new epoch-one calls."""


class CheckedBoundary(BaseException):
    pass


def baseline_runtime(job, check):
    from scripts.baselines.runtime import Runtime, digest
    from scripts.baselines.runtime import read as load
    from scripts.baselines.runtime import write as save

    source = Path(job["source_train_dir"])

    class ContinuedRuntime(Runtime):
        activated = False

        def activate(self):
            if not self.activated:
                self.activated = True
                save(
                    self.output / "resume_verified.json",
                    {
                        "at": now(),
                        "method": job["method"],
                        "restored_tasks": job["epoch1_count"],
                        "no_initial_val_inference": True,
                    },
                )
                if check:
                    raise CheckedBoundary()

        def task(self, item, learned, label):
            if label == "final_test":
                label = "epoch2_final_test"
            key = digest(json.dumps({"item": item, "prompt_sha256": digest(learned), "label": label}, sort_keys=True))
            cached = source / "rollouts" / key / "result.json"
            if not self.activated:
                if not cached.exists():
                    raise PrefixMiss(f"Epoch-one actor cache miss: {key} {item} {label}")
                return load(cached)
            if job["method"] == "skillopt" and label == "validation" and cached.exists():
                return load(cached)
            return super().task(item, learned, label)

        def chat(self, system, user, stage):
            if not self.activated:
                cached = source / "analysis_cache" / f"{digest(json.dumps([system, user, stage]))}.json"
                if not cached.exists():
                    raise PrefixMiss(f"Epoch-one teacher cache miss: {cached.name}, {stage}")
                value = load(cached)
                if "error" in value:
                    raise RuntimeError(value["error"])
                return value["text"], value["usage"]
            return super().chat(system, user, stage)

        def online(self, items, results, learned, batch):
            old = [i for i in items if i.get("epoch") != 2]
            if old:
                return super().online(items, results, learned, batch)
            for item, result in zip(items, results, strict=True):
                path = self.output / "online" / f"epoch2_{digest(item['task_id'])}.json"
                record = {**item, "batch": batch, "hard": result["hard"], "prompt_sha256": digest(learned)}
                if path.exists() and load(path) != record:
                    raise RuntimeError("Epoch-two replay diverged from an immutable encounter")
                save(path, record)
            return None

    return ContinuedRuntime(read(Path(job["train_dir"]) / "config.json"))


def restore_skillopt_buffer(out_root, history, epoch, steps_per_epoch, resume_from):
    """Restore only a partially completed epoch, following native skip semantics."""
    start = (epoch - 1) * steps_per_epoch + 1
    end = epoch * steps_per_epoch
    if not start < resume_from <= end:
        return []
    records = {record["step"]: record for record in history if record.get("epoch") == epoch}
    entries = []
    for step in range(start, resume_from):
        record = records.get(step)
        if record is None:
            raise RuntimeError(f"Missing saved SkillOpt history: epoch {epoch}, step {step}")
        # Both branches continue before native step_buffer.append/digest write.
        if record.get("action") in {"skip_no_patches", "skip_no_rewrite"}:
            continue
        path = Path(out_root) / "steps" / f"step_{step:04d}" / "trajectory_digest.json"
        if not path.is_file():
            raise RuntimeError(f"Missing saved SkillOpt step buffer: {path}")
        entry = read(path)
        if entry.get("step") != step:
            raise RuntimeError(f"Mismatched saved SkillOpt step buffer: {path}")
        entries.append(entry)
    return entries


def skillopt_train(runtime, job):
    path = Path(job["source_snapshot"]) / "scripts/baselines/skillopt_runtime.py"
    source = path.read_text()
    replacements = {
        'return len(config["train"])': 'return len(config["epoch1_train"])',
        'raise RuntimeError("Use the single-pass epoch plan")': """import random
            pool = config["train"][len(config["epoch1_train"]):]
            items = random.Random(seed).sample(pool, min(batch_size, len(pool)))
            return BatchSpec(phase="meta", split="train", seed=seed, batch_size=len(items), payload=items)""",
        "if epoch != 1 or accumulation != 1:": "if epoch not in (1, 2) or accumulation != 1:",
        "payload=index) for index in range(steps_per_epoch)]": (
            "payload=(epoch - 1) * steps_per_epoch + index) for index in range(steps_per_epoch)]"
        ),
        "start = batch.payload * 5": "runtime.activate()\n                start = batch.payload * 5",
        'label = f"train_batch_{int(match.group(1)) - 1}" if split == "train" else "validation"': (
            'label = (f"train_batch_{int(match.group(1)) - 1}" if match else '
            '"meta_" + str(Path(out_dir).relative_to(upstream_dir))) if split == "train" else "validation"'
        ),
        'if split == "train":\n                runtime.online': (
            'if split == "train" and match is not None:\n                runtime.online'
        ),
        "num_epochs=1, batch_size=5": "num_epochs=2, batch_size=5",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Frozen SkillOpt hook no longer applies: {before}")
        source = source.replace(before, after)
    namespace = {"__name__": "epoch2_skillopt_runtime", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    # Upstream's `score or -1` resume expression mistakes a measured zero for
    # missing data. Preserve exact zero while restoring the native checkpoint.
    import skillopt.model as model
    import skillopt.prompts as prompts

    def optimizer_chat(system, user, max_completion_tokens=8192, retries=3, stage="optimizer", **kwargs):
        return runtime.chat(system, user, stage)

    model.chat_optimizer = optimizer_chat
    model.get_token_summary = runtime.token_summary
    original_prompt = prompts.load_prompt

    def frozen_prompt(name, *args, **kwargs):
        path = Path(job["source_snapshot"]) / "scripts/baselines/prompts/skillopt/optimizer" / f"{name}.md"
        return path.read_text() if path.exists() else original_prompt(name, *args, **kwargs)

    prompts.load_prompt = frozen_prompt
    import skillopt.engine.trainer as trainer

    original_train = trainer.ReflACTTrainer.train
    trainer_source = __import__("inspect").getsource(original_train)
    import textwrap

    trainer_source = textwrap.dedent(trainer_source)
    for key, fallback in (("current_score", "-1.0"), ("best_score", "current_score")):
        before = f'float(runtime_state.get("{key}", {fallback}) or {fallback})'
        after = f'float(runtime_state.get("{key}", {fallback}))'
        assert trainer_source.count(before) == 1
        trainer_source = trainer_source.replace(before, after)
    before = "step_buffer: list[dict] = []"
    after = "step_buffer = _epoch2_restore_buffer(out_root, history, epoch, steps_per_epoch, resume_from)"
    assert trainer_source.count(before) == 1
    trainer_source = trainer_source.replace(before, after)
    trainer._epoch2_restore_buffer = restore_skillopt_buffer
    local = {}
    exec(compile(trainer_source, trainer.__file__, "exec"), vars(trainer), local)
    trainer.ReflACTTrainer.train = local["train"]
    return namespace["train"](runtime, Path(job["train_dir"]) / "upstream/continued")


def train(job, check=False):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    method = job["method"]
    if method in ("gepa", "skillopt"):
        runtime = baseline_runtime(job, check)
        with (runtime.output / "optimizer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not check and (runtime.output / "summary.json").exists():
                return
            if method == "skillopt":
                write(
                    runtime.output / "status.json",
                    {"phase": "running", "upstream_dir": str(runtime.output / "upstream/continued")},
                )
                skillopt_train(runtime, job)
            else:
                import scripts.baselines.gepa_runtime as gepa_runtime

                original = gepa_runtime.OrderedSampler.next_minibatch_ids

                def minibatch(self, loader, state):
                    if not runtime.activated and state.i >= job["epoch1_count"] // 5:
                        # Compare reconstructed optimizer population/scores with
                        # the completed native checkpoint before any new call.
                        from gepa.core.state import GEPAState

                        saved = GEPAState.load(job["source_upstream"])
                        for field in ("program_candidates", "prog_candidate_val_subscores"):
                            if getattr(state, field) != getattr(saved, field):
                                raise PrefixMiss(f"Restored GEPA {field} differs from epoch one")
                        runtime.activate()
                    return original(self, loader, state)

                gepa_runtime.OrderedSampler.next_minibatch_ids = minibatch
                generation = len(list((runtime.output / "upstream").glob("replay_*")))
                upstream = runtime.output / "upstream" / f"replay_{generation:04d}"
                write(runtime.output / "status.json", {"phase": "running", "upstream_dir": str(upstream)})
                gepa_runtime.train(runtime, upstream)
            assert len(list((runtime.output / "online").glob("*.json"))) == job["train_count"]
            write(runtime.output / "status.json", {"phase": "completed"})
        return
    argv = list(read(job["source_plan"])["jobs"])
    original = next(j for j in argv if j["id"] == job["id"])
    command = original["train_command"]
    sys.argv = [command[2], *command[3:]]
    index = sys.argv.index("--output-dir")
    sys.argv[index + 1] = job["train_dir"]
    if "--base-url" in sys.argv:
        sys.argv[sys.argv.index("--base-url") + 1] = job["base_url"]
    if method == "ace":
        if "--resume-sessions" in sys.argv:
            sys.argv.remove("--resume-sessions")
        module = module_file("epoch2_frozen_ace", command[2])
        module.task_order_from_train_manifest = lambda *a, **k: [(job["benchmark"], task) for task in job["train_ids"]]
        if check:
            return
        module.main()
    else:
        from exgentic.agents.premise.pipeline import PremisePipeline

        original_restore = PremisePipeline._restore_or_initialize

        def restore(self):
            self.order = [(job["benchmark"], task) for task in job["train_ids"]]
            return original_restore(self)

        PremisePipeline._restore_or_initialize = restore
        if check:

            def checked_run(self, **kwargs):
                assert self.cursor == job["epoch1_count"]
                raise CheckedBoundary()

            PremisePipeline.run = checked_run
        module_file("epoch2_frozen_premise", command[2]).main()


def environment(plan, job):
    env = dict(os.environ, **plan["environment"])
    env.update(
        OPENAI_BASE_URL=job["base_url"],
        OPENAI_API_BASE=job["base_url"],
        OPENAI_API_KEY="EMPTY",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        EXGENTIC_PORT_BLOCK_START=str(job["port_block_start"]),
    )
    env["PATH"] = str(Path(job["source_snapshot"]) / ".venv/bin") + os.pathsep + env.get("PATH", "")
    return env


def worker(plan, job):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    os.environ.update(environment(plan, job))
    module = module_file("epoch2_frozen_worker", snapshot / "scripts/utils/fixed_matrix_worker.py")
    module.run({**plan, "snapshot": str(snapshot)}, job)


def completed_job(job):
    """A marker is reusable only with full matching train/test evidence."""
    root, train = Path(job["output"]), Path(job["train_dir"])
    marker = root / "job_complete.json"
    if not marker.exists():
        return False
    result = read(marker)
    valid = (
        result.get("complete") is True
        and result.get("job_id") == job["id"]
        and result.get("test_ids") == job["test_ids"]
        and result.get("tasks") == len(job["test_ids"])
        and result.get("infrastructure_errors") == 0
    )
    if not valid:
        return False
    if job["method"] == "ace":
        return (train / "training_complete.json").exists() and read(train / "training_complete.json").get(
            "completed_task_count"
        ) == job["train_count"]
    if job["method"] == "premise":
        return read(train / "state.json").get("cursor") == job["train_count"]
    return (train / "training_result.json").exists() and len(list((train / "online").glob("*.json"))) == job[
        "train_count"
    ]


def process_identity(pid):
    """Read state and start ticks so a reused PID cannot be adopted."""
    try:
        directory = Path("/proc") / str(pid)
        fields = (directory / "stat").read_text().rpartition(")")[2].split()
        return {
            "pid": pid,
            "state": fields[0],
            "start_ticks": fields[19],
            "args": (directory / "cmdline").read_bytes().decode().strip("\0").split("\0"),
        }
    except (OSError, UnicodeError, IndexError):
        return None


class AdoptedWorker:
    """Observe an existing worker after controller replacement without stopping it."""

    def __init__(self, identity, job):
        self.pid = identity["pid"]
        self.start_ticks = identity["start_ticks"]
        self.job = job

    def poll(self):
        identity = process_identity(self.pid)
        if identity and identity["start_ticks"] == self.start_ticks and identity["state"] not in {"Z", "X"}:
            return None
        return 0 if completed_job(self.job) else 1


def existing_worker(job, previous, output):
    if previous.get("status") != "running" or not isinstance(previous.get("pid"), int):
        return None
    identity = process_identity(previous["pid"])
    if not identity or identity["state"] in {"Z", "X"}:
        return None
    args = identity["args"]
    try:
        matches = (
            "worker" in args
            and args[args.index("--output") + 1] == str(output)
            and args[args.index("--job") + 1] == job["id"]
        )
    except (ValueError, IndexError):
        matches = False
    if not matches:
        raise RuntimeError(f"Refusing to adopt a different process as {job['id']}: PID {identity['pid']}")
    return AdoptedWorker(identity, job)


def schedule(plan, output):
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / "controller_process.json", {"pid": os.getpid(), "output": str(output), "at": now()})
        progress = read(output / "progress.json")
        active = {}
        for benchmark in BENCHMARKS:
            group = [j for j in plan["jobs"] if j["benchmark"] == benchmark]
            for job in group:
                if completed_job(job):
                    progress["jobs"][job["id"]] = {"status": "completed"}
                    continue
                adopted = existing_worker(job, progress["jobs"].get(job["id"], {}), output)
                if adopted is not None:
                    active[job["id"]] = (job, adopted)
                    progress["jobs"][job["id"]].update(adopted_at=now(), start_ticks=adopted.start_ticks)
                    continue
                log = (Path(job["output"]) / "worker.log").open("a")
                process = subprocess.Popen(
                    job["command"],
                    cwd=ROOT,
                    env=environment(plan, job),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log.close()
                active[job["id"]] = (job, process)
                progress["jobs"][job["id"]] = {"status": "running", "pid": process.pid, "started_at": now()}
            write(output / "progress.json", progress)
            write(
                output / "status.json", {"phase": "running", "benchmark": benchmark, "at": now(), "active": len(active)}
            )
            while active:
                for key, (job, process) in list(active.items()):
                    result = process.poll()
                    if result is None:
                        continue
                    good = result == 0 and completed_job(job)
                    progress["jobs"][key].update(
                        status="completed" if good else "error", returncode=result, finished_at=now()
                    )
                    del active[key]
                write(output / "progress.json", progress)
                write(
                    output / "status.json",
                    {"phase": "running", "benchmark": benchmark, "at": now(), "active": len(active)},
                )
                if active:
                    time.sleep(15)
            failed = [j["id"] for j in group if progress["jobs"][j["id"]]["status"] != "completed"]
            if failed:
                write(
                    output / "status.json",
                    {"phase": "blocked", "benchmark": benchmark, "failed_jobs": failed, "at": now()},
                )
                return 1
        write(output / "status.json", {"phase": "completed", "at": now()})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "worker", "schedule", "check"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "outputs/qwen_cp10_original_appworld_noalign_dual100_20261105/queue.sqlite3",
    )
    parser.add_argument("--job")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        plan = prepare(output, args.queue)
        print(json.dumps({"jobs": len(plan["jobs"]), "output": str(output)}))
        return
    plan = read(output / "launch_manifest.json")
    if args.action == "schedule":
        raise SystemExit(schedule(plan, output))
    job = next(j for j in plan["jobs"] if j["id"] == args.job)
    if args.action == "worker":
        try:
            worker(plan, job)
        except BaseException as exc:
            write(Path(job["output"]) / "job_status.json", {"phase": "error", "at": now(), "error": repr(exc)})
            raise
    else:
        os.environ.update(environment(plan, job))
        try:
            train(job, check=args.action == "check")
        except CheckedBoundary:
            print(json.dumps({"checked": job["id"], "no_model_calls": True}))


if __name__ == "__main__":
    main()
