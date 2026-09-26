#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

export ACE_BENCHMARKS="terminalbench2"
export ACE_TASK_SPLIT_MANIFEST="${ACE_TASK_SPLIT_MANIFEST:-${repo_root}/scripts/utils/task_splits/seed42_terminalbench2_pilot5.json}"
export NUM_TASKS="${NUM_TASKS:-5}"
export ACE_TERMINALBENCH2_MAX_TURNS="${ACE_TERMINALBENCH2_MAX_TURNS:-15}"
export ACE_TERMINALBENCH2_ACTOR="${ACE_TERMINALBENCH2_ACTOR:-terminus2}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"

exec bash "${script_dir}/run_experiment_qwen_training_time.sh" "$@"
