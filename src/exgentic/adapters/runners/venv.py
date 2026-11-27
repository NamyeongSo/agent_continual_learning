# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""VenvRunner — runs the HTTP service inside a uv virtual environment.

Uses the same HTTPTransport as ServiceRunner and DockerRunner, but the
uvicorn server runs in a subprocess with its own isolated venv instead
of the host Python or a Docker container.

The venv is created and managed by the EnvironmentManager — VenvRunner
only starts the subprocess and optionally installs extra runtime
dependencies.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

import httpx

from ._utils import (
    find_free_port,
    inject_exgentic_env,
    make_close,
    prepare_subprocess_env,
    serialize_kwargs,
)
from .service import HTTPTransport
from .transport import ObjectProxy

_HEALTH_TIMEOUT = 30.0
_TRANSPORT_TIMEOUT = 1800.0
_PORT_START_ATTEMPTS = 5


def _wait_for_child_health(url: str, process: subprocess.Popen, timeout: float) -> None:
    """Accept only the service started by this runner, never another port owner."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Venv service exited before becoming healthy (exit={process.returncode})")
        try:
            response = httpx.get(f"{url}/health", timeout=2.0)
            if response.status_code == 200 and response.json().get("pid") == process.pid:
                if process.poll() is None:
                    return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Venv service did not become healthy within {timeout}s")


def _uv(*args: str, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise RuntimeError("uv CLI not found on PATH")
    result = subprocess.run([uv_bin, *args], check=False, **kwargs)
    if check and result.returncode != 0:
        stderr = getattr(result, "stderr", "") or ""
        stdout = getattr(result, "stdout", "") or ""
        raise RuntimeError(f"uv {' '.join(args[:3])} failed (exit {result.returncode}):\n{stderr}\n{stdout}")
    return result


class VenvRunner:
    """Start an HTTP service in an isolated uv venv and return an ObjectProxy.

    Parameters
    ----------
    target_cls:       Class to instantiate inside the venv subprocess.
    env_name:         Environment name for EnvironmentManager (e.g. "benchmarks/bfcl").
    module_path:      Dotted module path for locating package resources.
    port:             Host port to bind (auto-selected if None).
    dependencies:     Extra pip packages to install in the venv at runtime.
    health_timeout:   Seconds to wait for the health endpoint.
    transport_timeout: Seconds to wait for an RPC response; None disables the RPC time limit.
    """

    def __init__(
        self,
        target_cls: type | str,
        *args: Any,
        env_name: str = "",
        module_path: str = "",
        port: int | None = None,
        dependencies: list[str] | None = None,
        health_timeout: float | None = None,
        transport_timeout: float | None = _TRANSPORT_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError(
                "VenvRunner requires keyword-only constructor arguments. "
                "Pass all arguments as kwargs instead of positional args."
            )
        self._target_cls = target_cls
        self._kwargs = kwargs
        self._env_name = env_name
        self._module_path = module_path
        self._port = port or find_free_port()
        self._dependencies = dependencies or []
        self._health_timeout = health_timeout or _HEALTH_TIMEOUT
        self._transport_timeout = transport_timeout
        if self._transport_timeout is not None and self._transport_timeout <= 0:
            raise ValueError("transport_timeout must be positive")
        self._process: subprocess.Popen | None = None
        self._service_output: BinaryIO | None = None

    # ── venv handling ─────────────────────────────────────────────────

    def _get_venv_dir(self) -> Path:
        """Return the venv directory managed by EnvironmentManager."""
        from ...environment.instance import get_manager

        return get_manager().env_path(self._env_name) / "venv"

    def _venv_python(self) -> Path:
        """Return the path to the Python binary inside the venv."""
        venv = self._get_venv_dir()
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    def _ensure_venv(self) -> Path:
        """Ensure the venv exists via EnvironmentManager."""
        from ...environment import EnvType
        from ...environment.helpers import get_exgentic_install_target
        from ...environment.instance import get_manager

        mgr = get_manager()
        project_root, packages = get_exgentic_install_target()
        mgr.install(
            self._env_name,
            env_type=EnvType.VENV,
            module_path=self._module_path,
            project_root=project_root,
            packages=packages,
        )
        return self._get_venv_dir()

    def _install_deps(self) -> None:
        """Install extra runtime dependencies into the venv."""
        if not self._dependencies:
            return
        python = self._venv_python()
        _uv(
            "pip",
            "install",
            "--python",
            str(python),
            "--no-cache",
            *self._dependencies,
            capture_output=True,
            text=True,
        )

    # ── subprocess lifecycle ──────────────────────────────────────────

    def start(self) -> ObjectProxy:
        venv = self._ensure_venv()
        self._install_deps()

        if isinstance(self._target_cls, str):
            cls_ref = self._target_cls
        else:
            cls_ref = f"{self._target_cls.__module__}:{self._target_cls.__qualname__}"
        kwargs_flag, kwargs_value = serialize_kwargs(self._kwargs)

        # Build a filtered environment (same filtering as DockerRunner).
        env = prepare_subprocess_env()
        env["VIRTUAL_ENV"] = str(venv)
        venv_bin = str(venv / "bin")
        # Prepend venv bin to the *system* PATH so external tools (docker,
        # podman, git, …) remain reachable from within the venv subprocess.
        system_path = os.environ.get("PATH", "")
        env["PATH"] = venv_bin + os.pathsep + system_path
        inject_exgentic_env(env)
        source_root = Path(__file__).resolve().parents[3]
        if (source_root.parent / "pyproject.toml").is_file():
            # Reuse isolated dependencies, but load the active checkout's
            # framework and budget guards rather than an old installed copy.
            inherited = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(source_root) + (os.pathsep + inherited if inherited else "")

        exgentic_bin = self._get_venv_dir() / "bin" / "exgentic"
        cmd = [
            str(exgentic_bin),
            "serve",
            "--cls",
            cls_ref,
            kwargs_flag,
            kwargs_value,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
        ]

        # AppWorld can emit more than a pipe buffer of import warnings before
        # serving /health. A file keeps the child from blocking on stderr.
        # Port discovery and child bind are not atomic: retry with a fresh port
        # if another process wins that race.
        for attempt in range(_PORT_START_ATTEMPTS):
            cmd[-1] = str(self._port)
            self._service_output = tempfile.TemporaryFile()
            try:
                self._process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=self._service_output,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                self._stop_process()
                raise
            atexit.register(self._stop_process)
            url = f"http://127.0.0.1:{self._port}"
            try:
                _wait_for_child_health(url, self._process, self._health_timeout)
                break
            except (TimeoutError, RuntimeError) as exc:
                output = b""
                if self._service_output is not None:
                    self._service_output.flush()
                    self._service_output.seek(0, os.SEEK_END)
                    self._service_output.seek(max(0, self._service_output.tell() - 65536))
                    output = self._service_output.read()
                self._stop_process()
                if b"address already in use" in output.lower() and attempt + 1 < _PORT_START_ATTEMPTS:
                    self._port = find_free_port()
                    continue
                raise RuntimeError(
                    f"Venv service failed to start on port {self._port}: {exc}\n"
                    f"service output:\n{output.decode(errors='replace')}"
                ) from exc

        transport = HTTPTransport(url, timeout=self._transport_timeout)
        proxy = ObjectProxy(transport)
        object.__setattr__(proxy, "close", make_close(transport, self._stop_process))
        return proxy

    def _stop_process(self) -> None:
        proc = self._process
        self._process = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        output = self._service_output
        self._service_output = None
        if output is not None:
            output.close()
        from ._utils import release_port

        release_port(self._port)
