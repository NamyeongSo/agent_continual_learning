#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
set -euo pipefail

install_release() {
    local version="$1" task_root="$2" repo_url="$3" repo_commit="$4" created=0
    # A caller can pass either the 2.1 repository or its tasks/ directory.
    if [[ "$version" == "2.1" && "${task_root##*/}" == "tasks" ]]; then
        task_root="${task_root%/tasks}"
    fi
    if [[ ! -e "${task_root}/.git" ]]; then
        git clone --filter=blob:none --no-checkout "${repo_url}" "${task_root}"
        created=1
    fi
    if [[ "$created" == "0" && -n "$(git -C "${task_root}" status --porcelain --untracked-files=normal)" ]]; then
        echo "Terminal-Bench ${version} checkout has local changes: ${task_root}" >&2
        return 1
    fi
    if [[ "$created" == "1" || "$(git -C "${task_root}" rev-parse HEAD)" != "$repo_commit" ]]; then
        git -C "${task_root}" fetch --depth 1 origin "${repo_commit}"
        git -C "${task_root}" checkout --detach "${repo_commit}"
    fi
    local actual
    actual="$(git -C "${task_root}" rev-parse HEAD)"
    if [[ "$actual" != "$repo_commit" ]]; then
        echo "Terminal-Bench ${version} checkout mismatch: ${actual}" >&2
        return 1
    fi
    echo "Terminal-Bench ${version}: ${task_root} (${actual})"
}

release="${1:-all}"
case "$release" in
    all|2.0|2.1) ;;
    *) echo "Usage: setup.sh [all|2.0|2.1]" >&2; exit 2 ;;
esac
if [[ "$release" == "all" || "$release" == "2.0" ]]; then
    install_release "2.0" "${EXGENTIC_TERMINALBENCH2_TASK_ROOT:-${PWD}/terminal-bench-2}" \
        "https://github.com/harbor-framework/terminal-bench-2.git" \
        "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
fi
if [[ "$release" == "all" || "$release" == "2.1" ]]; then
    install_release "2.1" "${EXGENTIC_TERMINALBENCH21_TASK_ROOT:-${PWD}/terminal-bench-2-1}" \
        "https://github.com/harbor-framework/terminal-bench-2-1.git" \
        "12b7565679ce5006230b820666353673a133db93"
fi
