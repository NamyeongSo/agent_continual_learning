# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Standalone Docker-host worker. Only the installed upstream harness is imported.

The transport deploys this file to the Docker host. Keep imports independent of
Exgentic so its Python dependencies and model credentials stay on the client.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import Future
from pathlib import Path


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]}: {(result.stderr or result.stdout)[-4000:]}")
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def paths(request: dict) -> tuple[Path, Path, Path]:
    root = Path(request["backend_root"])
    return root, root / "harness" / request["harness_commit"], root / "data" / request["benchmark_version"]


def checkout(path: Path, url: str, commit: str, *, tag: str | None = None) -> None:
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "git",
            "-c",
            "filter.lfs.required=false",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "clone",
            "--depth",
            "1",
        ]
        if tag:
            command.extend(["--branch", tag])
        run([*command, url, str(path)], env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        if not tag:
            run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", commit])
            run(["git", "-C", str(path), "checkout", "--detach", commit])
    actual = run(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"Pinned checkout mismatch at {path}: {actual} != {commit}")
    dirty = run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
    if dirty:
        raise RuntimeError(f"Pinned benchmark checkout has modified tracked files: {path}")


def install(request: dict) -> dict:
    root, harness, data = paths(request)
    with locked(root / "install.lock"):
        checkout(harness, "https://github.com/DeepCommit-ai/SWE-Milestone.git", request["harness_commit"])
        checkout(
            data,
            "https://huggingface.co/datasets/DeepCommit-ai/SWE-Milestone-data",
            request["data_commit"],
            tag=request["benchmark_version"],
        )
        runtime = root / "runtime"
        python = runtime / "bin" / "python"
        # Debian's system Python may provide venv but omit ensurepip. A PyPA zipapp
        # installs into our private environment without changing server packages.
        run([sys._base_executable, "-m", "venv", "--without-pip", str(runtime)])
        pip = root / "pip.pyz"
        if not pip.exists():
            with urllib.request.urlopen("https://bootstrap.pypa.io/pip/pip.pyz", timeout=60) as response:
                temporary = pip.with_suffix(".tmp")
                temporary.write_bytes(response.read())
                temporary.replace(pip)
        run([str(python), str(pip), "install", "pyyaml>=6,<7", "pathspec>=0.12,<1", "ast-grep-cli==0.45.0"])
        ast_version = run([str(runtime / "bin" / "ast-grep"), "--version"]).stdout.strip()
    return {
        "harness_commit": request["harness_commit"],
        "data_commit": request["data_commit"],
        "benchmark_version": request["benchmark_version"],
        "backend_root": str(root),
        "ast_grep": ast_version,
    }


def official(request: dict) -> tuple[Path, Path, Path]:
    root, harness, data = paths(request)
    if not (harness / "harness").is_dir() or not (data / ".git").is_dir():
        raise RuntimeError(
            "SWE-Milestone is not installed on the Docker host; run exgentic install --benchmark swemilestone"
        )
    # The worker already runs on the Docker host, including all bind-mounted files.
    os.environ.pop("DOCKER_HOST", None)
    os.environ["SWE_MILESTONE_IMAGE_TAG"] = request["benchmark_version"]
    sys.path.insert(0, str(harness))
    checkout(harness, "", request["harness_commit"])
    checkout(data, "", request["data_commit"])
    return root, harness, data


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def catalog(request: dict) -> list[dict]:
    _, _, data = official(request)
    from harness.e2e.dag import DAGManager

    subset = request.get("subset", "all")
    repositories = request["repositories"]
    if subset != "all" and subset not in repositories:
        raise ValueError(f"Unknown SWE-Milestone repository: {subset}")
    rows = []
    for alias, repo in repositories.items():
        if subset not in ("all", alias):
            continue
        workspace = data / repo
        selected = workspace / "selected_milestone_ids.txt"
        ids = read_ids(selected) if selected.exists() else {p.parent.name for p in workspace.glob("srs/*/SRS.md")}
        # Supply an explicit selection even for repositories whose DAG CSV omits isolated nodes.
        selection = paths(request)[0] / "catalog" / f"{repo}.txt"
        selection.parent.mkdir(parents=True, exist_ok=True)
        temporary = selection.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text("\n".join(sorted(ids)) + "\n")
        temporary.replace(selection)
        dag = DAGManager(
            workspace / "dependencies.csv",
            selected_ids_file=selection,
            ignore_weak_dependencies=True,
            additional_dependencies_csv=workspace / "additional_dependencies.csv",
        )
        order = []
        while not dag.is_done():
            ready = dag.get_next_runnable()
            if not ready:
                raise ValueError(f"Milestone dependency cycle in {repo}")
            mid = ready[0]
            order.append(mid)
            dag.mark_complete(mid)
        ungraded = read_ids(workspace / "non-graded_milestone_ids.txt")
        for mid in order:
            srs = workspace / "srs" / mid / "SRS.md"
            rows.append(
                {
                    "task_id": f"{repo}/{mid}",
                    "repository": repo,
                    "repo_alias": alias,
                    "milestone_id": mid,
                    "graded": mid not in ungraded,
                    "srs": srs.read_text(),
                    "prerequisites": sorted(dag.reverse_adj_list.get(mid, [])),
                }
            )
    return rows


def pull_image(request: dict, repo: str, milestone: str) -> str:
    _, harness, _ = paths(request)
    from harness.e2e.image_version import load_manifest, local_ref

    local = local_ref(repo, milestone, request["benchmark_version"])
    manifest = load_manifest(harness / "manifests" / f"digests-{request['benchmark_version']}.tsv")
    matches = [row for row in manifest if row[1] == repo.lower() and row[2] == milestone.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Image is absent from the frozen manifest: {local}")
    digest_ref = matches[0][3]
    inspect = subprocess.run(["docker", "image", "inspect", digest_ref], capture_output=True, text=True)
    if inspect.returncode:
        if not request.get("pull_images", True):
            raise RuntimeError(f"Missing pinned image: {local}; install with --pull-images")
        run(["docker", "pull", digest_ref], timeout=7200)
    run(["docker", "tag", digest_ref, local])
    return local


def images(request: dict) -> dict:
    official(request)
    rows = catalog(request)
    pulled = []
    for repo in dict.fromkeys(row["repository"] for row in rows):
        pulled.append(pull_image(request, repo, "base-offline"))
        for row in rows:
            if row["repository"] == repo and row["graded"]:
                pulled.append(pull_image(request, repo, row["milestone_id"]))
    return {"images": pulled}


def register_external_framework(*, enable_tmux: bool = False) -> None:
    from harness.e2e.agents.base import AgentFramework, register_framework

    @register_framework("exgentic")
    class ExternalFramework(AgentFramework):
        FRAMEWORK_NAME = "exgentic"

        def get_container_mounts(self):
            return []

        def get_container_init_script(self, agent_name):
            if not enable_tmux:
                return ""
            # This hook runs before upstream locks the container's network.
            return """
import os, shutil, subprocess
if shutil.which("tmux") is None:
    if shutil.which("apt-get"):
        subprocess.run(["apt-get", "update"], check=True, timeout=180)
        subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "tmux"],
                       check=True, timeout=180, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    elif shutil.which("apk"):
        subprocess.run(["apk", "add", "--no-cache", "tmux"], check=True, timeout=180)
    elif shutil.which("dnf"):
        subprocess.run(["dnf", "install", "-y", "tmux"], check=True, timeout=180)
    else:
        raise RuntimeError("Prepare tmux in the development image before network quarantine")
"""

        def build_run_command(self, model, session_id, prompt_path):
            raise RuntimeError("Exgentic agents run outside the development container")

        def build_resume_command(self, model, session_id, message_path):
            raise RuntimeError("Exgentic agents run outside the development container")

        def get_network_endpoint_url(self):
            return None


def capture_filter_kwargs(repo_binding, metadata: dict) -> dict:
    """Use the evaluator's pinned source/test authority for submission capture."""
    from harness.e2e.evaluator import resolve_residue_prune_config

    policy = resolve_residue_prune_config(
        dict(repo_binding.config), metadata,
        repo_config_binding_mode="trial-pinned", repo_config_sha256=repo_binding.sha256,
    )
    if policy.src_filter is not None:
        source = policy.src_filter
        return {
            "repo_src_dirs": source.src_dirs, "test_dirs": source.test_dirs,
            "exclude_patterns": source.exclude_patterns, "generated_patterns": source.generated_patterns,
            "modifiable_test_patterns": source.modifiable_test_patterns,
        }
    return {
        "repo_src_dirs": metadata["repo_src_dirs"], "test_dirs": metadata.get("test_dirs", []),
        "exclude_patterns": metadata.get("exclude_patterns", []),
        "generated_patterns": metadata.get("generated_patterns", []),
        "modifiable_test_patterns": metadata.get("modifiable_test_patterns", []),
    }


def controller(request: dict, trial: Path, repo: str, state: dict):
    _, harness, data = paths(request)
    from harness.e2e.orchestrator import E2EOrchestrator
    from harness.e2e.repo_config_binding import freeze_repo_config, resolve_repo_config
    from harness.e2e.runtime_policy_binding import freeze_runtime_policy, resolve_runtime_policy

    register_external_framework(enable_tmux=request.get("actor") == "terminus2")
    workspace = data / repo
    metadata = json.loads((workspace / "metadata.json").read_text())
    repo_binding = freeze_repo_config(trial, resolve_repo_config(repo, workspace, project_root=harness))
    policy = freeze_runtime_policy(trial, resolve_runtime_policy(repo, harness))
    os.environ.update(policy.env)
    rows = [row for row in catalog({**request, "subset": "all"}) if row["repository"] == repo]
    (trial / "selected_milestone_ids.txt").write_text("\n".join(row["milestone_id"] for row in rows) + "\n")
    additional = workspace / "additional_dependencies.csv"
    if additional.exists() and not (trial / additional.name).exists():
        shutil.copyfile(additional, trial / additional.name)
    image = state.get("restore_image") or state.get("current_image")
    if not image:
        image = pull_image(request, repo, "base-offline")
    engine = E2EOrchestrator(
        repo_name=repo,
        milestone_version="base",
        image_name=image,
        dag_path=workspace / "dependencies.csv",
        srs_root=workspace / "srs",
        trial_root=trial,
        workspace_root=workspace,
        **capture_filter_kwargs(repo_binding, metadata),
        agent_name="exgentic",
        main_branch=metadata.get("main_branch", "main"),
        repo_config_binding=repo_binding,
        runtime_policy_binding=policy,
    )
    engine._snapshot_baseline_commit = state.get("baseline_commit")
    engine.dag.restore_state(completed=set(state.get("submitted", {})), failed=set(), skipped=set())
    metadata_path = trial / "trial_metadata.json"
    provenance = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    provenance.update(
        repo_config_binding=repo_binding.to_metadata(trial), runtime_policy_binding=policy.to_metadata(trial)
    )
    write_json(metadata_path, provenance)
    return engine, rows


def commit_image(container: str) -> str:
    return run(["docker", "commit", container]).stdout.strip()


def git(engine, *args) -> str:
    result = engine._docker_exec_git(*args)
    if result.returncode:
        raise RuntimeError(f"Development Git command failed: {(result.stderr or result.stdout)[-4000:]}")
    return result.stdout.strip()


def require_active(request: dict, state: dict) -> None:
    if state.get("active_session") != request["session_id"]:
        raise RuntimeError("This session does not own the development stream")


def start(request: dict, state: dict, engine, rows: list[dict]) -> dict:
    mid = request["milestone_id"]
    active = state.get("active_session")
    if active and active != request["session_id"]:
        raise RuntimeError(f"Development stream has an active session: {active}; close it before starting another")
    if mid in state.get("submitted", {}):
        state["active_session"] = request["session_id"]
        return {
            "submitted": True,
            "start_snapshot": state.get("starts", {}).get(mid),
            "container": engine.container_name,
        }
    if active == request["session_id"]:
        return {"submitted": False, "start_snapshot": state["starts"][mid], "container": engine.container_name}
    if mid not in engine.dag.get_next_runnable():
        raise RuntimeError(f"Milestone {mid} is blocked; submit prerequisites first")
    from harness.prepare_repo.split_test_patches.test_detector import ensure_ast_grep

    ensure_ast_grep()
    if not state.get("initialized"):
        try:
            engine.setup_environment()
        except Exception:
            engine.container_setup.cleanup(remove=True)
            raise
        state["baseline_commit"] = engine._snapshot_baseline_commit
        state["initialized"] = True
    else:
        if state.get("restore_image") and engine.container_setup.container_exists():
            engine.container_setup.cleanup(remove=True)
        recreated = not engine.container_setup.container_exists()
        engine.container_setup.start_container()
        if recreated:
            engine.container_setup.lock_network()
    state.pop("restore_image", None)
    engine.container_setup.prepare_agent_invocation()
    if request.get("actor") == "terminus2":
        tmux = engine.container_setup.docker_exec(["tmux", "-V"], check=False)
        if tmux.returncode:
            raise RuntimeError(
                "This development stream has no tmux; start Terminus-2 with a fresh run_id/stream_id "
                "so it can be prepared before network quarantine"
            )
    # A full container checkpoint preserves ignored build files and dependency caches as well as Git.
    snapshot = {
        "image_id": commit_image(engine.container_name),
        "baseline_commit": state["baseline_commit"],
        "submitted": dict(state.get("submitted", {})),
        "repository": request["repository"],
        "harness_commit": request["harness_commit"],
        "data_commit": request["data_commit"],
        "benchmark_version": request["benchmark_version"],
    }
    state.setdefault("starts", {})[mid] = snapshot
    state["current_image"] = snapshot["image_id"]
    state["active_session"] = request["session_id"]
    row = next(row for row in rows if row["milestone_id"] == mid)
    engine._update_task_queue_file(engine.trial_root)
    return {
        "submitted": False,
        "start_snapshot": snapshot,
        "container": engine.container_name,
        "graded": row["graded"],
        "start_commit": git(engine, "rev-parse", "HEAD"),
    }


def execute(request: dict, state: dict, engine) -> dict:
    require_active(request, state)
    if request["milestone_id"] in state.get("submitted", {}):
        raise RuntimeError("A tagged milestone is immutable; finish this session before starting another")
    user = request.get("user", "fakeroot")
    if user not in ("fakeroot", "root"):
        raise ValueError("Unsupported development container user")
    command = request["command"]
    if request.get("cwd"):
        command = f"cd -- {shlex.quote(request['cwd'])} || exit; {command}"
    for key, value in request.get("env", {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(value, str):
            raise ValueError("Invalid command environment")
        command = f"export {key}={shlex.quote(value)}; {command}"
    result = engine.container_setup.docker_exec(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=5",
            str(request["command_timeout_sec"]),
            "bash",
            "-lc",
            command,
        ],
        check=False,
        user=user,
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


class CaptureExecutor:
    """Let upstream capture and validate the snapshot; defer grading to Session.score()."""

    def submit(self, _function, **kwargs):
        self.arguments = kwargs
        return Future()


def submit(request: dict, state: dict, engine, rows: list[dict]) -> dict:
    require_active(request, state)
    mid = request["milestone_id"]
    if mid in state.get("submitted", {}):
        return {"submitted_commit": state["submitted"][mid]}
    if mid not in engine.dag.get_next_runnable():
        raise RuntimeError(f"Milestone {mid} is not available")
    tag = f"agent-impl-{mid}"
    row = next(row for row in rows if row["milestone_id"] == mid)
    if row["graded"]:
        pull_image(request, request["repository"], mid)
    # A retry after capture failure must keep the already-created immutable tag.
    existing = engine._docker_exec_git("rev-parse", "--verify", f"refs/tags/{tag}")
    if existing.returncode == 0:
        commit = git(engine, "rev-parse", f"{tag}^{{commit}}")
        if git(engine, "rev-parse", "HEAD") != commit:
            raise RuntimeError(f"Submission tag {tag} differs from HEAD")
    else:
        # Upstream applies its source/test/manifest filters during capture.
        git(engine, "add", "-A")
        git(
            engine,
            "-c",
            "user.name=Exgentic",
            "-c",
            "user.email=exgentic@localhost",
            "commit",
            "--allow-empty",
            "-m",
            f"Implement {mid}",
        )
        git(engine, "tag", tag)
        commit = git(engine, "rev-parse", "HEAD")
    if row["graded"]:
        executor = CaptureExecutor()
        if not engine._handle_submission(mid, tag, executor, {}, expected_tag_hash=commit):
            raise RuntimeError("Official source snapshot capture was rejected")
        state.setdefault("evaluations", {})[mid] = executor.arguments
    state["current_image"] = commit_image(engine.container_name)
    state.setdefault("submitted", {})[mid] = commit
    return {"submitted_commit": commit}


def evaluate(request: dict, state: dict, engine, rows: list[dict]) -> dict:
    require_active(request, state)
    mid = request["milestone_id"]
    if mid not in state.get("submitted", {}):
        return {"score": 0.0, "resolved": False, "submitted": False}
    row = next(row for row in rows if row["milestone_id"] == mid)
    if not row["graded"]:
        return {"score": 0.0, "resolved": False, "submitted": True, "graded": False}
    if mid in state.get("build_failures", {}):
        return state["build_failures"][mid]
    from harness.e2e.collect_results import (
        calculate_precision_recall,
        calculate_score_reliable,
        check_log_for_failure,
        is_infra_invalid,
        is_resolved,
    )
    from harness.e2e.evaluator import generate_filtered_evaluation
    from harness.e2e.orchestrator import run_evaluation_task

    result_dir = engine.trial_root / "evaluation" / mid
    original = result_dir / "evaluation_result.json"
    if not original.exists() or is_infra_invalid(json.loads(original.read_text())):
        arguments = dict(state["evaluations"][mid])
        for key in ("snapshot_path", "result_dir", "workspace_root", "repo_config_path", "runtime_policy_path"):
            if arguments.get(key) is not None:
                arguments[key] = Path(arguments[key])
        _, _, _, result, error = run_evaluation_task(**arguments)
        if error or result is None:
            # Upstream records a failed-to-compile/no-report cell in its error
            # summary, which its collector scores zero. Keep that outcome distinct
            # from infrastructure-invalid and other unevaluable submissions.
            if (error or "").startswith("RuntimeError: No valid test report files generated") and (
                check_log_for_failure(result_dir / "evaluation_error.log") == "compilation_failure"
            ):
                engine._update_evaluation_summary(mid, "unlocked", "error", error_msg=error)
                record = json.loads(engine._summary_file_path().read_text())["results"][mid]
                outcome = {
                    "score": calculate_score_reliable(record),
                    "resolved": False,
                    "submitted": True,
                    "graded": True,
                    "precision": 0.0,
                    "recall": 0.0,
                    "build_failure": True,
                    "evaluation": record,
                    "evaluation_error": error,
                    "remote_evaluation_dir": str(result_dir),
                }
                state.setdefault("build_failures", {})[mid] = outcome
                return outcome
            raise RuntimeError(f"Official milestone evaluation failed: {(error or 'missing result')[:3000]}")
    # Official v1.0.2 build waivers and dependency filtering affect the score.
    filtered_path = generate_filtered_evaluation(original, engine.workspace_root, mid)
    result = json.loads((filtered_path or original).read_text())
    if is_infra_invalid(result):
        raise RuntimeError(
            "Official evaluation produced an infrastructure-invalid result; inspect remote evaluation logs"
        )
    precision, recall = calculate_precision_recall(result)
    return {
        "score": calculate_score_reliable(result),
        "resolved": is_resolved(result),
        "submitted": True,
        "graded": True,
        "precision": precision,
        "recall": recall,
        "evaluation": result,
        "remote_evaluation_dir": str(result_dir),
    }


def close_session(request: dict, state: dict, engine) -> dict:
    require_active(request, state)
    mid = request["milestone_id"]
    if mid not in state.get("submitted", {}):
        state["restore_image"] = state["starts"][mid]["image_id"]
    state["active_session"] = None
    return {"retained_container": engine.container_name, "submitted": mid in state.get("submitted", {})}


def stream_operation(request: dict) -> dict:
    root, _, _ = official(request)
    namespace = request["namespace"]
    if not re.fullmatch(r"[a-f0-9]{20}", namespace):
        raise ValueError("Invalid development stream namespace")
    repo, mid = request["task_id"].split("/", 1)
    if repo not in request["repositories"].values() or not re.fullmatch(r"[A-Za-z0-9_.-]+", mid):
        raise ValueError("Invalid milestone task ID")
    request.update(repository=repo, milestone_id=mid)
    trial = root / "trials" / repo / namespace
    trial.mkdir(parents=True, exist_ok=True)
    state_file = trial / "stream_state.json"
    with locked(trial / "stream.lock"):
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        replay = request.get("start_snapshot")
        if replay and state and state.get("branch_snapshot_image") != replay.get("image_id"):
            raise ValueError("Replay requires a fresh stream_id/run_id or the same original branch snapshot")
        if replay and not state:
            for key in ("repository", "harness_commit", "data_commit", "benchmark_version"):
                if replay.get(key) != request.get(key):
                    raise ValueError(f"Replay snapshot {key} differs from the current benchmark")
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", replay.get("image_id", "")):
                raise ValueError("Invalid replay image identity")
            state = {
                "initialized": True,
                "baseline_commit": replay["baseline_commit"],
                "submitted": replay["submitted"],
                "restore_image": replay["image_id"],
                "branch_snapshot_image": replay["image_id"],
            }
        engine, rows = controller(request, trial, repo, state)
        operations = {
            "start": start,
            "execute": execute,
            "submit": submit,
            "evaluate": evaluate,
            "close": close_session,
        }
        operation = request["operation"]
        try:
            if operation in ("start", "submit", "evaluate"):
                result = operations[operation](request, state, engine, rows)
            else:
                result = operations[operation](request, state, engine)
        finally:
            write_json(state_file, state)
        return result


def main() -> None:
    request = json.load(sys.stdin)
    # Some upstream subprocesses inherit FD 1; Python redirect_stdout alone
    # cannot keep their progress output out of the JSON response channel.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        # Official modules print progress; reserve stdout for exactly one JSON response.
        with contextlib.redirect_stdout(sys.stderr):
            operation = request["operation"]
            if operation == "install":
                result = install(request)
            elif operation == "catalog":
                result = catalog(request)
            elif operation == "images":
                result = images(request)
            else:
                result = stream_operation(request)
        print(json.dumps({"result": result}, default=str), file=protocol, flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=protocol, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
