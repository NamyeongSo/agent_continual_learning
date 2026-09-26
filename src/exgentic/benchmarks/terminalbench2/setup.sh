#!/usr/bin/env bash
set -euo pipefail

task_root="${EXGENTIC_TERMINALBENCH2_TASK_ROOT:-${PWD}/terminal-bench-2}"
repo_url="https://github.com/harbor-framework/terminal-bench-2.git"
repo_commit="2fd12b88aafdd04a52c298e3940bcb189f9766d6" # pragma: allowlist secret

if [[ ! -d "${task_root}/.git" ]]; then
    git clone --filter=blob:none "${repo_url}" "${task_root}"
fi
git -C "${task_root}" fetch --depth 1 origin "${repo_commit}"
git -C "${task_root}" checkout --force --detach "${repo_commit}"

actual="$(git -C "${task_root}" rev-parse HEAD)"
if [[ "${actual}" != "${repo_commit}" ]]; then
    echo "Terminal-Bench 2.0 checkout mismatch: ${actual}" >&2
    exit 1
fi
