#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
manifest="${PREMISE_TASK_SPLIT_MANIFEST:-${repo_root}/scripts/utils/task_splits/seed42_train50_val39_terminalbench2.json}"
output="${PREMISE_OUTPUT_DIR:-${repo_root}/outputs/premise_terminalbench2_s42_t3}"

exec "${repo_root}/.venv/bin/python" "${script_dir}/run_experiment.py" \
    --task-split-manifest "$manifest" \
    --benchmarks bfcl,appworld,terminalbench2 \
    --mode sequential --seed 42 --num-tasks 50 --past 3 \
    --output-dir "$output" "$@"
