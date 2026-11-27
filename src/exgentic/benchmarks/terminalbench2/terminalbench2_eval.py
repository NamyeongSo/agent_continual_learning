# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Terminal-Bench 2.0/2.1 sessions using Harbor's Docker environment and verifier.

The agent sees only instruction.md and the live container. Harbor uploads tests
after the agent has stopped acting, then runs the task's official test.sh.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import traceback
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from ipaddress import IPv4Network
from pathlib import Path
from typing import Any

from ...core.actions import ActionsHandler
from ...core.evaluator import Evaluator
from ...core.experiment_limits import BENCHMARK_MAX_TURNS
from ...core.session import Session
from ...core.types import Action, ActionType, BenchmarkResults, SessionIndex, SessionScore, SingleObservation
from .container_lifecycle import stop_process_group, write_lease
from .terminalbench2_benchmark import (
    TERMINALBENCH_COMMITS,
    BashAction,
    FinishActionForTerminalBench,
    RunTerminus2Action,
    default_task_root,
)


async def copy_verifier_artifacts(
    download_dir: Callable[..., Awaitable[None]],
    download_file: Callable[..., Awaitable[None]],
    source_dir: str,
    target_dir: Path,
    diagnostics_path: Path,
) -> None:
    """Retry artifact transport, then recover official rewards independently of other logs."""
    report: dict[str, Any] = {"status": "running", "attempts": []}

    def save():
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = diagnostics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(diagnostics_path)

    def record_error(operation: str, exc: Exception):
        report["attempts"].append({
            "operation": operation, "error_type": type(exc).__name__, "error": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        })
        save()

    target_dir.mkdir(parents=True, exist_ok=True)
    save()
    for attempt in range(3):
        try:
            with tempfile.TemporaryDirectory(dir=target_dir.parent) as directory:
                staging = Path(directory)
                await download_dir(source_dir, staging)
                if not any((staging / name).is_file() for name in ("reward.json", "reward.txt")):
                    raise RuntimeError("Directory transfer completed without an official reward file")
                for name in ("reward.json", "reward.txt"):
                    (target_dir / name).unlink(missing_ok=True)
                shutil.copytree(staging, target_dir, dirs_exist_ok=True)
        except Exception as exc:
            record_error("download_dir", exc)
            if attempt < 2:
                await asyncio.sleep(0.5 * 2**attempt)
        else:
            report["status"] = "completed"
            save()
            return

    # A broken auxiliary log must not hide an available official reward. Copy
    # into a temporary directory so incomplete directory transfers cannot
    # supply a stale or partially written reward to Harbor's parser.
    recovered = []
    with tempfile.TemporaryDirectory(dir=target_dir.parent) as directory:
        staging = Path(directory)
        for name in ("reward.json", "reward.txt", "test-stdout.txt", "ctrf.json"):
            try:
                await download_file(f"{source_dir}/{name}", staging / name)
            except Exception as exc:
                record_error(f"download_file:{name}", exc)
            else:
                if (staging / name).is_file():
                    recovered.append(name)
        if not any(name in recovered for name in ("reward.json", "reward.txt")):
            report["status"] = "error"
            save()
            raise RuntimeError("Verifier artifact transfer failed; no official reward file could be recovered")
        for name in ("reward.json", "reward.txt"):
            (target_dir / name).unlink(missing_ok=True)
        for name in recovered:
            (staging / name).replace(target_dir / name)
    report.update(status="recovered", recovered_files=recovered)
    save()


def resolve_task_root(task_root: str | None, subset: str = "2.0") -> Path:
    if subset not in TERMINALBENCH_COMMITS:
        raise ValueError(f"Unsupported Terminal-Bench release: {subset}")
    commit = TERMINALBENCH_COMMITS[subset]
    root = Path(task_root).expanduser() if task_root else default_task_root(subset)
    root = root.resolve()
    if (root / "tasks").is_dir():
        root = root / "tasks"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Terminal-Bench {subset} tasks missing at {root}. Run the terminalbench2 setup.sh "
            f"with argument {subset}; see docs/terminalbench2.md."
        )
    has_checkout = (root / ".git").exists() or (root.parent / ".git").exists()
    if subset == "2.1" and not has_checkout:
        raise ValueError("Terminal-Bench 2.1 requires the pinned Git checkout to verify the task release")
    if has_checkout:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if head != commit:
            raise ValueError(
                f"Terminal-Bench task checkout is {head}; expected pinned {subset} commit {commit}"
            )
        if subset == "2.1":
            changes = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if changes:
                raise ValueError("Terminal-Bench 2.1 checkout has local changes; use a clean pinned checkout")
    return root


def list_task_ids(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "task.toml").is_file())


def truncate_output(output: str, limit: int) -> str:
    if len(output) <= limit:
        return output
    head = limit // 2
    tail = limit - head
    return output[:head] + f"\n... {len(output) - limit} characters omitted ...\n" + output[-tail:]


def quota_thread_environment(cpus: int | None, explicit: dict[str, str]) -> dict[str, str]:
    """Keep numerical and package-build defaults within the task CPU quota."""
    if cpus is None:
        return {}
    count = str(max(1, cpus))
    return {name: count for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                     "NUMEXPR_NUM_THREADS", "OPAMJOBS") if name not in explicit}


class TerminalBench2Session(Session):
    """One task container, operated by Exgentic and graded by Harbor."""

    def __init__(
        self,
        task_dir: str,
        task_id: str,
        max_interactions: int | None = BENCHMARK_MAX_TURNS["terminalbench2"],
        command_timeout_sec: int = 900,
        observation_size_limit: int = 10000,
        actor: str = "interactive",
        session_id: str | None = None,
        enforce_session_timeouts: bool = True,
        subset: str = "2.0",
        exclude_model_call_time: bool = False,
        native_terminus2: bool = False,
        native_context_recovery: bool = False,
    ) -> None:
        from harbor.models.task.config import TaskOS, VerifierEnvironmentMode
        from harbor.models.task.task import Task
        from harbor.models.task.verifier_mode import resolve_task_verifier_mode

        if subset not in TERMINALBENCH_COMMITS:
            raise ValueError(f"Unsupported Terminal-Bench release: {subset}")
        self._subset = subset
        self._task_obj = Task(task_dir)
        if self._task_obj.has_steps or self._task_obj.config.environment.os != TaskOS.LINUX:
            raise ValueError(f"Terminal-Bench {subset} adapter requires a single-step Linux task")
        if resolve_task_verifier_mode(self._task_obj.config) != VerifierEnvironmentMode.SHARED:
            raise ValueError(f"Terminal-Bench {subset} adapter requires a shared Harbor verifier environment")
        if self._task_obj.short_name != task_id:
            raise ValueError(f"Task ID {task_id!r} does not match Harbor task {self._task_obj.name!r}")

        self._task_id = task_id
        if session_id is not None:
            self._session_id = session_id
        self._max_interactions = max_interactions
        self._command_timeout_sec = command_timeout_sec
        self._observation_size_limit = observation_size_limit
        if actor not in {"interactive", "terminus2"}:
            raise ValueError(f"Unknown Terminal-Bench actor: {actor}")
        self._actor = actor
        self._enforce_session_timeouts = enforce_session_timeouts
        self._exclude_model_call_time = exclude_model_call_time
        self._native_terminus2 = native_terminus2
        self._native_context_recovery = native_context_recovery
        self._task_clock = None
        self._task_time_accounting = None
        self._action_count = 0
        self._done = False
        self._score: SessionScore | None = None
        self._loop_runner: asyncio.Runner | None = None
        self._environment = None
        self._trial_paths = None
        self._agent_deadline: float | None = None
        self._terminus_trace: str | None = None
        self._terminus_usage: dict[str, int] | None = None
        self._terminus_trace_path: str | None = None
        self._terminus_timed_out = False
        self._terminus_error: str | None = None
        self._terminus_metadata: dict[str, Any] = {}
        self._phase_timings: dict[str, Any] = {}
        self._registry = ActionsHandler(
            logger=self.logger,
            warn_on_validation_error=False,
            warn_on_unknown_action=True,
            handle_validation_error=None,
            handle_unknown_action=None,
        )
        super().__init__()

    @contextmanager
    def _timed_phase(self, name: str):
        """Persist phase boundaries even when a deadline or verifier error fires."""
        timings = getattr(self, "_phase_timings", {})
        self._phase_timings = timings
        phase = {"started_at": datetime.now(UTC).isoformat(), "status": "running"}
        timings[name] = phase
        path = self._trial_paths.agent_dir.parent / "phase_timings.json"

        def save():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(timings, indent=2), encoding="utf-8")
            temporary.replace(path)

        started = time.monotonic()
        save()
        try:
            yield
        except BaseException as exc:
            phase.update(status="error", error_type=type(exc).__name__, error=str(exc),
                         traceback="".join(traceback.format_exception(exc)))
            raise
        else:
            phase["status"] = "ok"
        finally:
            phase["elapsed_seconds"] = time.monotonic() - started
            save()

    def _remaining_agent_time(self):
        clock = getattr(self, "_task_clock", None)
        if clock is not None:
            return clock.remaining()
        return self._agent_deadline - time.monotonic() if self._agent_deadline is not None else None

    def _time_metadata(self):
        clock = getattr(self, "_task_clock", None)
        if clock is None:
            return {}
        if getattr(self, "_task_time_accounting", None) is None:
            self._task_time_accounting = clock.stop()
        return {"task_time_accounting": self._task_time_accounting}

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        if self._actor == "terminus2":
            return self._task_obj.instruction
        return (
            self._task_obj.instruction
            + "\n\nUse the bash action to work in the task container. Each command runs in a new shell; "
            "filesystem changes persist. Call finish when done. The official Terminal-Bench verifier runs afterward."
        )

    @property
    def context(self) -> dict[str, Any]:
        return {"benchmark": "terminalbench2", "task_id": self._task_id, **self._release_metadata()}

    def _release_metadata(self) -> dict[str, str]:
        subset = getattr(self, "_subset", "2.0")
        return {"dataset_version": subset, "dataset_commit": TERMINALBENCH_COMMITS[subset]}

    def fail_experiment_budget(self, details: dict[str, Any], policy: dict[str, Any]) -> SessionScore:
        from ...core.experiment_limits import budget_failure_score

        self._experiment_failure_score = budget_failure_score(
            details, experiment_limits=policy, **self._release_metadata(),
        )
        self.save_results(self._experiment_failure_score.model_dump())
        return self._experiment_failure_score

    @property
    def actions(self) -> list[ActionType]:
        if not self._registry.actions:
            if self._actor == "terminus2":
                self._registry.add_action(
                    name="run_terminus2",
                    description="Run Harbor Terminus-2 in the task container with the ACE playbook",
                    action_cls=RunTerminus2Action,
                    handler=self._handle_terminus2,
                    is_finish=True,
                )
                return self._registry.actions
            self._registry.add_action(
                name="bash",
                description="Execute a shell command inside the Terminal-Bench task container",
                action_cls=BashAction,
                handler=self._handle_bash,
            )
            self._registry.add_action(
                name="finish",
                description="Finish work and submit the current container state for official verification",
                action_cls=FinishActionForTerminalBench,
                handler=self._handle_finish,
                is_finish=True,
            )
        return self._registry.actions

    def start(self) -> SingleObservation:
        from harbor.environments.docker.docker import DockerEnvironment
        from harbor.models.trial.paths import TrialPaths

        self._container_thread_limits = quota_thread_environment(
            self._task_obj.config.environment.cpus, self._task_obj.config.environment.env,
        )
        thread_limits = self._container_thread_limits

        class CopyingDockerEnvironment(DockerEnvironment):
            """Copy verifier artifacts via Docker CLI, including with remote DOCKER_HOST."""

            @property
            def capabilities(self):
                return super().capabilities.model_copy(update={"mounted": False})

            async def _run_docker_compose_command(
                self, command, check=True, timeout_sec=None, stdin_data=None, on_output=None,
            ):
                runtime = type(self).runtime()
                args = [*runtime.compose, "--project-name", self.session_id.lower()]
                if runtime.supports_compose_project_directory:
                    args += ["--project-directory", str(self.environment_dir.resolve())]
                for path in self._docker_compose_paths:
                    args += ["-f", str(path.resolve())]
                args += command
                process = await asyncio.create_subprocess_exec(
                    *args, env=self._compose_env_vars(include_os_env=True),
                    cwd=str(self.environment_dir.resolve()) if self.environment_dir.is_dir() else None,
                    stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                )
                collect = self._collect_streamed_output if on_output is not None else self._collect_buffered_output
                kwargs = {"timeout_sec": None, "stdin_data": stdin_data}
                if on_output is not None:
                    kwargs["on_output"] = on_output
                try:
                    timeout = timeout_sec
                    if timeout is None:
                        if command[0] == "build":
                            timeout = task.config.environment.build_timeout_sec
                        elif command[0] == "exec" and getattr(self, "_exgentic_verifier_active", False):
                            # Harbor omits timeout_sec for test.sh. The outer
                            # verifier deadline owns this execution; applying
                            # the infrastructure cap would truncate valid tests.
                            timeout = None
                        else:
                            timeout = 180
                    result = await asyncio.wait_for(collect(process, **kwargs), timeout=timeout)
                except BaseException:
                    await stop_process_group(process)
                    raise
                if check and result.return_code:
                    raise RuntimeError(f"Terminal-Bench Docker {command[0]} failed: {result.stdout or result.stderr}")
                return result

            async def exec(self, command, **kwargs):
                try:
                    result = await super().exec(command, **kwargs)
                except TimeoutError:
                    from harbor.environments.base import ExecResult

                    from .docker_capture import capture_arguments, capture_pane

                    if capture_arguments(command) is None:
                        raise  # Never execute a potentially mutating command twice.
                    started = time.monotonic()
                    # The CLI's hijacked stream can stall after a short read-only
                    # command. A finite HTTP response recovers that observation,
                    # inside the unchanged outer official actor deadline.
                    recovered = await asyncio.wait_for(asyncio.to_thread(
                        capture_pane, self.session_id.lower(), command,
                        user=self._resolve_user(kwargs.get("user")),
                        cwd=kwargs.get("cwd") or self.task_env_config.workdir,
                        env=self._merge_env(kwargs.get("env")),
                    ), timeout=45)
                    result = ExecResult(**recovered)
                    with (trial_paths.agent_dir.parent / "actor_transport_recovery.jsonl").open("a") as log:
                        log.write(json.dumps({"at": datetime.now(UTC).isoformat(), "operation": "tmux capture-pane",
                                              "recovery": "finite Docker HTTP exec", "return_code": result.return_code,
                                              "elapsed_seconds": time.monotonic() - started}) + "\n")
                if result.return_code and any(marker in (result.stderr or "") + (result.stdout or "") for marker in (
                    "mux_client_request_session: session request failed", "kex_exchange_identification:",
                    "ssh: connect to host",
                )):
                    raise RuntimeError("Docker SSH connection failed before the command session was established")
                if result.return_code and command.startswith("chmod +x "):
                    raise RuntimeError("Could not make the official verifier script executable")
                return result

            async def download_dir(self, source_dir, target_dir):
                if source_dir.rstrip("/") != "/logs/verifier":
                    return await super().download_dir(source_dir, target_dir)
                return await copy_verifier_artifacts(
                    super().download_dir, super().download_file, source_dir, Path(target_dir),
                    trial_paths.agent_dir.parent / "verifier_transfer.json",
                )

            @property
            def _docker_compose_paths(self):
                paths = super()._docker_compose_paths
                network_path = getattr(self, "_exgentic_network_path", None)
                return [*paths, network_path] if network_path is not None else paths

            async def start(self, force_build):
                # Docker's default pools only support a few dozen Compose
                # networks. Keep one isolated network per session, using /28
                # subnets in a separate configurable private address range.
                self._exgentic_network_directory = tempfile.TemporaryDirectory()
                self._exgentic_network_path = Path(self._exgentic_network_directory.name) / "network.json"
                pool = IPv4Network(os.environ.get("EXGENTIC_TERMINALBENCH2_NETWORK_POOL", "10.240.0.0/16"))
                if not pool.is_private or pool.prefixlen > 28:
                    raise ValueError("Terminal-Bench network pool must be a private IPv4 network of /28 or larger")
                for _attempt in range(32):
                    index = secrets.randbelow(1 << (28 - pool.prefixlen))
                    subnet = IPv4Network((int(pool.network_address) + 16 * index, 28))
                    override = {
                        "networks": {"default": {"ipam": {"config": [{"subnet": str(subnet)}]}}},
                    }
                    if thread_limits:
                        override["services"] = {"main": {"environment": thread_limits}}
                    self._exgentic_network_path.write_text(json.dumps(override), encoding="utf-8")
                    try:
                        return await super().start(force_build)
                    except RuntimeError as exc:
                        if "pool overlaps with other one" not in str(exc).lower():
                            raise
                raise RuntimeError("Could not allocate an isolated Terminal-Bench subnet after 32 attempts")

            async def stop(self, delete):
                try:
                    return await super().stop(delete)
                finally:
                    directory = getattr(self, "_exgentic_network_directory", None)
                    if directory is not None:
                        directory.cleanup()
                        self._exgentic_network_directory = None
                        self._exgentic_network_path = None

        if self._loop_runner is not None:
            raise RuntimeError("Terminal-Bench session already started")
        self._loop_runner = asyncio.Runner()
        task = self._task_obj
        trial_paths = TrialPaths(self.paths.benchmark_dir / "harbor")
        trial_paths.mkdir()
        self._trial_paths = trial_paths
        (trial_paths.agent_dir.parent / "container_resources.json").write_text(json.dumps({
            "version": "task-cpu-runtime-v2", "task_cpus": task.config.environment.cpus,
            "default_thread_limits": thread_limits,
            "explicit_environment": {name: value for name, value in task.config.environment.env.items()
                                     if name.endswith("_NUM_THREADS") or name == "OPAMJOBS"},
        }, indent=2), encoding="utf-8")
        # ExGentic's session ID is deterministic from the task/agent config.
        # PREMiSE executes the same task for two parents and multiple replay
        # candidates concurrently; Harbor's Docker Compose project must be
        # unique per execution, even when their ExGentic IDs match.
        docker_session_id = f"exgentic-{self.session_id}-{secrets.token_hex(6)}"
        self._docker_project = docker_session_id.lower()
        write_lease(self._docker_project, phase="active", session_id=self.session_id,
                    task_id=self._task_id, artifacts=str(trial_paths.agent_dir.parent))
        self._environment = CopyingDockerEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id=docker_session_id,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            logger=self.logger,
            mounts=[],
            network_policy=task.config.environment.resolve_baseline(),
        )
        try:
            with self._timed_phase("environment_start"):
                self._loop_runner.run(self._environment.start(force_build=False))
                self._loop_runner.run(
                    self._environment.exec("mkdir -p /logs/agent /logs/verifier /logs/artifacts", user="root")
                )
                self._loop_runner.run(self._environment.run_healthcheck())
            self._agent_deadline = (
                time.monotonic() + self._task_obj.config.agent.timeout_sec
                if self._enforce_session_timeouts else None
            )
            if self._exclude_model_call_time:
                from .task_clock import AgentTimeBudget, initialize_model_call_ledger

                ledger = self.paths.benchmark_dir / "model_call_timing.json" if self._actor == "interactive" else None
                if ledger is not None:
                    initialize_model_call_ledger(ledger)
                self._task_clock = AgentTimeBudget(
                    self._task_obj.config.agent.timeout_sec if self._enforce_session_timeouts else None,
                    report_path=self.paths.benchmark_dir / "task_time_accounting.json", ledger_path=ledger,
                )
        except BaseException:
            self.close()
            raise
        return SingleObservation(invoking_actions=[], result=None)

    def _handle_bash(self, action: BashAction) -> str:
        if self._done:
            return "Terminal-Bench task is already finished."
        if self._loop_runner is None or self._environment is None:
            raise RuntimeError("Terminal-Bench session has not started")
        remaining = self._remaining_agent_time()
        if remaining is not None and remaining <= 0:
            self._done = True
            return "Terminal-Bench agent time limit reached."
        with self._environment.with_default_user(self._task_obj.config.agent.user):
            try:
                result = self._loop_runner.run(
                    self._environment.exec(
                        action.arguments.command,
                        timeout_sec=(
                            max(1, min(self._command_timeout_sec, int(remaining)))
                            if remaining is not None else self._command_timeout_sec
                        ),
                    )
                )
            except RuntimeError as exc:
                if "timed out" not in str(exc).lower():
                    raise
                return f"Command timed out: {exc}"
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        rendered = f"exit_code: {result.return_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        return truncate_output(rendered, self._observation_size_limit)

    def _handle_finish(self, action: FinishActionForTerminalBench) -> None:
        self._done = True

    def _handle_terminus2(self, action: RunTerminus2Action) -> dict[str, Any]:
        if self._loop_runner is None or self._environment is None or self._trial_paths is None:
            raise RuntimeError("Terminal-Bench session has not started")
        from harbor.models.agent.context import AgentContext

        from ...core.experiment_limits import ACTOR_OUTPUT_TOKENS, ExperimentBudgetExceededError
        native = getattr(self, "_native_terminus2", False)
        if native:
            from harbor.agents.terminus_2 import Terminus2

            if (action.arguments.playbook or action.arguments.record_episodes or action.arguments.record_policy_shift
                    or action.arguments.replay_episodes_path):
                raise ValueError("Native Terminus-2 evaluation requires an empty policy and no replay")
            agent_class = Terminus2
        else:
            from .terminus_limits import BudgetedTerminus2

            agent_class = BudgetedTerminus2
        replay_kwargs = {}
        if action.arguments.record_policy_shift:
            replay_kwargs["record_policy_shift"] = True
        if action.arguments.record_episodes or action.arguments.replay_episodes_path:
            from .terminus_replay import load_recorded_prefix

            replay_kwargs["replay_episodes"] = load_recorded_prefix(
                action.arguments.replay_episodes_path, action.arguments.replay_episodes_count,
            )

        logs_dir = self._trial_paths.agent_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        call_settings = {
            **({} if native else {"max_tokens": ACTOR_OUTPUT_TOKENS}),
            **({"temperature": action.arguments.temperature} if action.arguments.temperature is not None else {}),
            **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
               if "Qwen3.6" in action.arguments.model or "gemma-4" in action.arguments.model.lower() else {}),
            **action.arguments.model_settings,
        }
        preserve_thinking = call_settings.get("extra_body", {}).get("chat_template_kwargs", {}).get(
            "preserve_thinking", False,
        )
        if native and call_settings.get("reasoning_effort") is not None:
            call_settings["allowed_openai_params"] = ["reasoning_effort"]
        disable_timeout = call_settings.pop("disable_request_timeout", False)
        if native and disable_timeout:
            import httpx

            # An omitted timeout selects LiteLLM's 600-second default. Do not
            # abandon a healthy generation and submit the same long request again.
            call_settings.update(timeout=httpx.Timeout(None), num_retries=0, max_retries=0)
        if native:
            # Keep Harbor's original loop, summarization and truncation recovery.
            # Accurate server limits prevent the unknown-model 1M fallback.
            agent_options = {"model_info": action.arguments.model_info} if action.arguments.model_info else {}
            if self._max_interactions is not None:
                agent_options["max_turns"] = self._max_interactions
            if getattr(self, "_native_context_recovery", False):
                from .native_context import RecoveringTerminus2, ServingTokenCounter

                agent_class = RecoveringTerminus2
                agent_options["serving_token_counter"] = ServingTokenCounter(
                    os.environ["OPENAI_API_BASE"], action.arguments.model,
                    call_settings.get("extra_body", {}).get("chat_template_kwargs", {}),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
        else:
            agent_options = {"max_turns": self._max_interactions,
                             "record_terminal_session": False, "enable_summarize": False}
            if getattr(self, "_task_clock", None) is not None:
                agent_options["task_clock"] = self._task_clock
        agent = agent_class(
            **replay_kwargs,
            **agent_options,
            logs_dir=logs_dir,
            model_name=action.arguments.model,
            api_base=os.environ.get("OPENAI_API_BASE"),
            interleaved_thinking=preserve_thinking,
            llm_call_kwargs=call_settings,
        )
        if native:
            clock = getattr(self, "_task_clock", None)
            if clock is not None:
                original_call = agent._llm.call

                async def timed_model_call(*args, **kwargs):
                    # Includes summarization calls and provider-internal retries.
                    with clock.model_call():
                        return await original_call(*args, **kwargs)

                agent._llm.call = timed_model_call
            (logs_dir / "native_agent_config.json").write_text(json.dumps({
                "agent_class": f"{type(agent).__module__}.{type(agent).__name__}",
                "max_episodes": agent._max_episodes,
                "enable_summarize": agent.options.enable_summarize,
                "proactive_summarization_threshold": agent.options.proactive_summarization_threshold,
                "record_terminal_session": agent.options.record_terminal_session,
                "max_input_tokens": agent._llm.get_model_context_limit(),
                "llm_call_kwargs": call_settings,
                "shared_actor_budget": False,
                "native_context_recovery": getattr(self, "_native_context_recovery", False),
                "request_timeout_disabled": disable_timeout,
                "exclude_model_call_time": clock is not None,
            }, indent=2, default=str), encoding="utf-8")
        context = AgentContext()
        from ...actor_context import terminal_actor_instruction

        instruction = terminal_actor_instruction(
            self._task_obj.instruction, action.arguments.playbook, action.arguments.actor_context,
        )
        self._terminus_trace_path = str(logs_dir / "trajectory.json")
        self._terminus_timed_out = False
        self._terminus_error = None
        self._experiment_budget_failure = None
        try:
            remaining = self._remaining_agent_time()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Terminal-Bench agent time limit reached before Terminus-2 started")
            with self._environment.with_default_user(self._task_obj.config.agent.user):
                with self._timed_phase("agent_setup"):
                    self._loop_runner.run(asyncio.wait_for(
                        agent.setup(self._environment), timeout=min(360, remaining) if remaining is not None else 360,
                    ))
                remaining = self._remaining_agent_time()
                with self._timed_phase("agent_run"):
                    execution = agent.run(instruction, self._environment, context)
                    clock = getattr(self, "_task_clock", None)
                    self._loop_runner.run(clock.run(execution) if clock is not None else asyncio.wait_for(
                        execution, timeout=max(1, remaining) if remaining is not None else None,
                    ))
        except ExperimentBudgetExceededError as exc:
            self._experiment_budget_failure = exc.details
        except TimeoutError as exc:
            # Harbor saves the partial trajectory and usage in run()'s finally.
            # A task deadline still submits that state to the official verifier.
            remaining = self._remaining_agent_time()
            if remaining is None or remaining > 0:
                self._terminus_error = f"{type(exc).__name__}: {exc}"
                raise
            self._terminus_timed_out = True
            self.logger.warning("Terminus-2 reached the task time limit; preserving its partial trajectory")
        except Exception as exc:
            self._terminus_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Harbor reports request durations separately from command waits.
            # Preserve them on errors too; ATIF step timestamps include both.
            self._terminus_metadata = dict(context.metadata or {})
            self._terminus_metadata.update(self._time_metadata())
            (logs_dir / "execution_metrics.json").write_text(
                json.dumps(self._terminus_metadata, indent=2, default=str), encoding="utf-8",
            )
        self._done = True
        self._action_count = int((context.metadata or {}).get("n_episodes", 0))
        # Reflect on every agent message, command and observation without clipping.
        trace_parts = []
        for step in agent._trajectory_steps:
            if step.source != "agent":
                continue
            payload = step.model_dump(exclude_none=True, exclude={"metrics", "timestamp", "model_name"})
            trace_parts.append(json.dumps(payload, ensure_ascii=False, default=str))
        self._terminus_trace = "\n".join(trace_parts)
        self._terminus_usage = {
            "input_tokens": context.n_input_tokens or 0,
            "output_tokens": context.n_output_tokens or 0,
        }
        return {
            "trace_path": self._terminus_trace_path,
            "agent_timed_out": self._terminus_timed_out,
            **self._terminus_usage,
        }

    def step(self, action: Action) -> SingleObservation | None:
        if self._done:
            return None
        remaining = self._remaining_agent_time()
        if remaining is not None and remaining <= 0:
            self._done = True
            self._terminus_timed_out = True
            return None
        count = len(action.to_action_list())
        if self._max_interactions is not None and self._action_count + count > self._max_interactions:
            self.logger.warning("Terminal-Bench action limit reached: %s", self._max_interactions)
            self._done = True
            return None
        observation = self._registry.execute(action)
        if self._actor != "terminus2":
            self._action_count += count
        return observation

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        timing = self._time_metadata()
        if failure := getattr(self, "_experiment_failure_score", None):
            failure.session_metadata.update(timing)
            return failure
        if self._score is not None:
            return self._score
        if failure := getattr(self, "_experiment_budget_failure", None):
            from ...core.experiment_limits import budget_failure_score, experiment_limits

            self._score = budget_failure_score(
                failure, experiment_limits=experiment_limits("terminalbench2"),
                trace_path=self._terminus_trace_path, agent_usage=self._terminus_usage,
                agent_trace=self._terminus_trace, actor=self._actor,
            )
            self._score.session_metadata.update(self._release_metadata())
            self._score.session_metadata.update(timing)
            self.save_results(self._score.model_dump())
            return self._score
        if error := getattr(self, "_terminus_error", None):
            # The orchestrator may request a score after an action raises. A
            # verifier reward from that partial infrastructure-failed execution
            # is not a valid measurement. Official actor budget failures above
            # remain terminal zero-score outcomes and are never retried here.
            raise RuntimeError(f"Terminus-2 execution failed before submission: {error}")
        if self._loop_runner is None or self._environment is None or self._trial_paths is None:
            raise RuntimeError("Cannot score Terminal-Bench session before its environment starts")
        from harbor.verifier.verifier import Verifier

        verifier = Verifier(
            task=self._task_obj,
            trial_paths=self._trial_paths,
            environment=self._environment,
            logger=self.logger,
        )
        self._environment._exgentic_verifier_active = True
        try:
            with (
                self._timed_phase("verifier"),
                self._environment.with_default_user(self._task_obj.config.verifier.user),
            ):
                result = self._loop_runner.run(
                    asyncio.wait_for(
                        verifier.verify(),
                        timeout=(
                            self._task_obj.config.verifier.timeout_sec
                            if getattr(self, "_enforce_session_timeouts", True) else None
                        ),
                    )
                )
        except Exception:
            # Preserve diagnostics before close() deletes the container. This
            # does not extend the verifier deadline or accept a late reward.
            if hasattr(self._environment, "download_file"):
                async def preserve_logs():
                    target = self._trial_paths.verifier_dir
                    target.mkdir(parents=True, exist_ok=True)
                    for name in ("test-stdout.txt", "ctrf.json"):
                        try:
                            await self._environment.download_file(f"/logs/verifier/{name}", target / name)
                        except Exception:
                            self.logger.debug("Could not preserve verifier log %s", name, exc_info=True)

                try:
                    self._loop_runner.run(asyncio.wait_for(preserve_logs(), timeout=15))
                except Exception:
                    self.logger.debug("Could not preserve verifier artifacts after failure", exc_info=True)
            raise
        finally:
            self._environment._exgentic_verifier_active = False
        rewards = result.rewards or {}
        if "reward" not in rewards:
            raise RuntimeError(f"Terminal-Bench verifier produced no reward for {self._task_id}")
        reward = float(rewards["reward"])
        if not 0 <= reward <= 1:
            raise ValueError(f"Terminal-Bench reward for {self._task_id} is outside [0,1]: {reward}")
        self._score = SessionScore(
            score=reward,
            success=True,
            is_finished=self._done,
            session_metadata={
                **self._release_metadata(),
                **timing,
                "task_id": self._task_id,
                "harbor_rewards": rewards,
                "harbor_verifier_dir": str(self._trial_paths.verifier_dir),
                "actions": self._action_count,
                "actor": self._actor,
                "native_terminus2": getattr(self, "_native_terminus2", False),
                "native_context_recovery": getattr(self, "_native_context_recovery", False),
                "enforce_session_timeouts": getattr(self, "_enforce_session_timeouts", True),
                "container_thread_limits": getattr(self, "_container_thread_limits", {}),
                "agent_timed_out": self._terminus_timed_out,
                "phase_timings": self._phase_timings,
                "agent_metrics": self._terminus_metadata,
                **({"agent_error": self._terminus_error} if self._terminus_error is not None else {}),
                **({"trace_path": self._terminus_trace_path} if self._terminus_trace_path is not None else {}),
                **({"agent_trace": self._terminus_trace} if self._terminus_trace is not None else {}),
                **({"agent_usage": self._terminus_usage} if self._terminus_usage is not None else {}),
            },
        )
        self.save_results(self._score.model_dump())
        return self._score

    def close(self) -> None:
        if self._loop_runner is not None:
            try:
                if self._environment is not None:
                    project = getattr(self, "_docker_project", None)
                    if project:
                        write_lease(project, phase="cleanup_pending")
                    try:
                        self._loop_runner.run(asyncio.wait_for(self._environment.stop(delete=True), timeout=90))
                    except Exception as exc:
                        self.logger.warning("Container cleanup deferred to janitor: %s", type(exc).__name__)
                    else:
                        if project:
                            write_lease(project, phase="closed")
            finally:
                self._loop_runner.close()
                self._loop_runner = None
                self._environment = None


class TerminalBench2Evaluator(Evaluator):
    def __init__(
        self,
        task_root: str | None = None,
        max_interactions: int | None = BENCHMARK_MAX_TURNS["terminalbench2"],
        command_timeout_sec: int = 900,
        observation_size_limit: int = 10000,
        actor: str = "interactive",
        enforce_session_timeouts: bool = True,
        subset: str = "2.0",
        exclude_model_call_time: bool = False,
        native_terminus2: bool = False,
        native_context_recovery: bool = False,
    ) -> None:
        self._subset = subset
        self._root = resolve_task_root(task_root, subset)
        self._ids = list_task_ids(self._root)
        if len(self._ids) != 89:
            raise RuntimeError(f"Expected 89 Terminal-Bench {subset} tasks at {self._root}, found {len(self._ids)}")
        self._max_interactions = max_interactions
        self._command_timeout_sec = command_timeout_sec
        self._observation_size_limit = observation_size_limit
        self._actor = actor
        self._enforce_session_timeouts = enforce_session_timeouts
        self._exclude_model_call_time = exclude_model_call_time
        self._native_terminus2 = native_terminus2
        self._native_context_recovery = native_context_recovery

    def list_tasks(self) -> list[str]:
        return list(self._ids)

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        task_id = str(index.task_id)
        if task_id not in self._ids:
            raise KeyError(f"Unknown Terminal-Bench {self._subset} task: {task_id}")
        return {
            "subset": self._subset,
            "task_dir": str(self._root / task_id),
            "task_id": task_id,
            "max_interactions": self._max_interactions,
            "command_timeout_sec": self._command_timeout_sec,
            "observation_size_limit": self._observation_size_limit,
            "actor": self._actor,
            "enforce_session_timeouts": self._enforce_session_timeouts,
            "exclude_model_call_time": self._exclude_model_call_time,
            "native_terminus2": self._native_terminus2,
            "native_context_recovery": self._native_context_recovery,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        scores = []
        for paths in self.get_sessions_paths(sessions):
            if not paths.benchmark_results.is_file():
                raise FileNotFoundError(f"Missing Terminal-Bench result: {paths.benchmark_results}")
            payload = json.loads(paths.benchmark_results.read_text(encoding="utf-8"))
            version = (payload.get("session_metadata") or {}).get("dataset_version", "2.0")
            if version != self._subset:
                raise ValueError(f"Cannot aggregate Terminal-Bench {version} results as {self._subset}")
            scores.append(float(payload["score"]))
        return BenchmarkResults(
            benchmark_name="terminalbench2",
            total_tasks=len(scores),
            score=sum(scores) / len(scores) if scores else 0.0,
            metrics={
                "resolved": sum(score == 1.0 for score in scores),
                "dataset_version": self._subset,
                "dataset_commit": TERMINALBENCH_COMMITS[self._subset],
            },
        )
