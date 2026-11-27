#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# The CLI accepts --agent, --model, --num-tasks, --task and --set overrides.
# Keep outputs separate from historical 2.0 sessions.
exec "${repo_root}/.venv/bin/exgentic" evaluate \
    --benchmark terminalbench2 --subset 2.1 \
    --output-dir "${TERMINALBENCH21_OUTPUT_DIR:-${repo_root}/outputs/terminalbench21}" "$@"
