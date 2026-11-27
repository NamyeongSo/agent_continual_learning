#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
source .venv/bin/activate
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8006/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EXGENTIC_LLM_TRACE_FORMAT=dedup_v1
export EXGENTIC_PORT_BLOCK_START="${EXGENTIC_PORT_BLOCK_START:-42000}"
: "${DOCKER_HOST:?Set DOCKER_HOST for the benchmark Docker daemon}"
timeout 15s docker info >/dev/null
pilot="$repo_root/scripts/ace/outputs/ace_tb2_terminus2_pilot5_final_20260926/ace_qwen_training_time_nothink_sequential_s42_n5_8192_aw50_bf50_sw100_tb15_terminalbench2_all"
output="${ACE_OUTPUT_DIR:-$repo_root/scripts/ace/outputs/ace_tb2_terminus2_continue30_20260926}"
if [[ -e "$output/experiment_config.json" ]]; then
    echo "Refusing to overwrite existing continuation: $output" >&2
    exit 1
fi
mkdir -p "$output"
python -u scripts/ace/run_experiment.py \
    --mode sequential --seed 42 --num-tasks 30 \
    --model openai/Qwen/Qwen3.6-35B-A3B --max-tokens 8192 \
    --disable-thinking --training-time --benchmarks terminalbench2 \
    --terminalbench2-actor terminus2 --terminalbench2-max-turns 15 \
    --task-split-manifest scripts/utils/task_splits/seed42_train_val_test.json \
    --continue-from "$pilot" --output-dir "$output" \
    2>&1 | tee -a "$output/experiment.log"
