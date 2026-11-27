# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Fetch, install, inspect, or smoke-test the pinned official baseline sources.

This prepares upstream optimizers. It does not launch benchmark experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOCK = HERE / "upstreams.lock.toml"


def run(argv: list[str], *, cwd: Path = ROOT, capture: bool = False, env: dict | None = None) -> str:
    print("+ " + shlex.join(str(part) for part in argv), flush=True)
    completed = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=capture, check=True)
    return completed.stdout.strip() if capture else ""


def git(path: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def source_status(spec: dict) -> dict:
    path = ROOT / spec["path"]
    if not (path / ".git").exists():
        return {"ready": False, "path": str(path), "reason": "source checkout missing"}
    revision = git(path, "rev-parse", "HEAD")
    origin = git(path, "remote", "get-url", "origin")
    dirty = bool(git(path, "status", "--porcelain", "--untracked-files=all"))
    return {
        "ready": revision == spec["commit"] and origin == spec["url"] and not dirty,
        "path": str(path),
        "commit": revision,
        "origin": origin,
        "dirty": dirty,
    }


def verify_prompts(manifest: dict) -> dict:
    problems = []
    for name, spec in manifest["prompt_archive"]["files"].items():
        path = HERE / "prompts" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            problems.append(name)
    return {"ready": not problems, "count": len(manifest["prompt_archive"]["files"]), "changed_or_missing": problems}


def ensure_source(spec: dict) -> None:
    path = ROOT / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        run(["git", "clone", "--no-checkout", "--filter=blob:none", spec["url"], str(path)])
    if not (path / ".git").exists():
        raise ValueError(f"Existing source path is not a Git checkout: {path}")
    if git(path, "remote", "get-url", "origin") != spec["url"]:
        raise ValueError(f"Source origin differs from the lock: {path}")
    # An unborn checkout immediately after clone --no-checkout has no index;
    # inspect the index only after a normal checkout has been materialized.
    index_path = Path(git(path, "rev-parse", "--git-path", "index"))
    materialized = (index_path if index_path.is_absolute() else path / index_path).exists()
    if materialized and git(path, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Source has local changes; preserving them: {path}")
    available = subprocess.run(
        ["git", "-C", str(path), "cat-file", "-e", spec["commit"] + "^{commit}"], capture_output=True, check=False
    )
    if available.returncode:
        run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", spec["commit"]])
    if not materialized or git(path, "rev-parse", "HEAD") != spec["commit"]:
        run(["git", "-C", str(path), "checkout", "--detach", spec["commit"]])
    if not source_status(spec)["ready"]:
        raise ValueError(f"Checkout verification failed: {path}")


def check_sources(manifest: dict, names: list[str]) -> None:
    for name in names:
        if not source_status(manifest["sources"][name])["ready"]:
            raise ValueError(f"{name}: expected a clean pinned checkout. Run fetch or inspect its local changes.")


def install(manifest: dict, method: str) -> None:
    spec = manifest["environments"][method]
    check_sources(manifest, spec["sources"])
    environment = ROOT / spec["path"]
    python = environment / "bin/python"
    if not python.exists():
        run(["uv", "venv", "--python", spec["python"], str(environment)])
    argv = ["uv", "pip", "install", "--python", str(python), "--requirement", str(HERE / spec["requirements"])]
    for name in spec["sources"]:
        argv.extend(["--editable", str(ROOT / manifest["sources"][name]["path"])])
    run(argv)


def smoke(manifest: dict, method: str) -> dict:
    spec = manifest["environments"][method]
    check_sources(manifest, spec["sources"])
    python = ROOT / spec["path"] / "bin/python"
    if not python.exists():
        raise ValueError(f"{method}: isolated environment missing; run install first")
    environment = dict(os.environ)
    environment["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    environment["DSPY_CACHEDIR"] = str(ROOT / ".exgentic/baselines/cache/dspy")
    raw = run([str(python), str(HERE / "smoke.py"), "--method", method], capture=True, env=environment)
    result = json.loads(raw.splitlines()[-1])
    expected = {name: manifest["sources"][name]["version"] for name in spec["sources"]}
    if result["versions"] != expected:
        raise ValueError(f"Imported package versions differ: {result['versions']} != {expected}")
    for name, imported in result["source_files"].items():
        if not Path(imported).is_relative_to(ROOT / manifest["sources"][name]["path"]):
            raise ValueError(f"Imported {name} from outside its pinned checkout: {imported}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "fetch", "install", "smoke"))
    parser.add_argument("--method", choices=("all", "gepa", "skillopt"), default="all")
    args = parser.parse_args()
    manifest = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    methods = list(manifest["environments"]) if args.method == "all" else [args.method]
    names = list(dict.fromkeys(name for method in methods for name in manifest["environments"][method]["sources"]))
    try:
        prompts = verify_prompts(manifest)
        if not prompts["ready"]:
            raise ValueError(
                "Team prompt bytes differ from their recorded hashes: " + str(prompts["changed_or_missing"])
            )
        if args.action == "status":
            sources = {name: source_status(manifest["sources"][name]) for name in names}
            environments = {
                method: {
                    "python": str(ROOT / manifest["environments"][method]["path"] / "bin/python"),
                    "exists": (ROOT / manifest["environments"][method]["path"] / "bin/python").exists(),
                }
                for method in methods
            }
            print(
                json.dumps(
                    {"scope": manifest["scope"], "sources": sources, "environments": environments, "prompts": prompts},
                    indent=2,
                )
            )
            ready = all(item["ready"] for item in sources.values()) and all(
                item["exists"] for item in environments.values()
            )
            return 0 if ready else 1
        if args.action == "fetch":
            for name in names:
                ensure_source(manifest["sources"][name])
        if args.action == "install":
            for method in methods:
                install(manifest, method)
        if args.action == "smoke":
            for method in methods:
                print(json.dumps(smoke(manifest, method), ensure_ascii=False))
    except subprocess.CalledProcessError as exc:
        print((exc.stderr or str(exc))[-6000:], file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
