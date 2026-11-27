#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resume the existing seed-42/43/44 ACE train-50 checkpoints in place.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
seed="${1:?Usage: $0 42|43|44}"
case "$seed" in
    42|43|44) ;;
    *) echo "Seed must be 42, 43, or 44" >&2; exit 2 ;;
esac

export DOCKER_HOST="${DOCKER_HOST:-${ACE_DOCKER_HOST:-ssh://exgentic-daysi-docker}}"
export ACE_TASK_SPLIT_MANIFEST="$repo_root/scripts/utils/task_splits/seed42_train50_val50.json"
export ACE_BENCHMARKS=bfcl,appworld,swebench
export ACE_APPWORLD_MAX_TURNS=50
export ACE_BFCL_MAX_TURNS=50
export ACE_SWEBENCH_MAX_TURNS=100
export MAX_TOKENS=4096
export NUM_TASKS=50
export SEED="$seed"

for mode in sequential interleaved; do
    if [[ "$mode" == sequential ]]; then
        export OUTPUT_BASE="$script_dir/outputs/ace_parallel_train50_20260924_retry_bfcl_rpc"
        port_base=$((37000 + (seed - 43) * 1000))
    else
        export OUTPUT_BASE="$script_dir/outputs/ace_parallel_interleaved_train50_20260924"
        port_base=$((39000 + (seed - 42) * 1000))
    fi
    export MODE="$mode" EXGENTIC_PORT_BLOCK_START="$port_base"
    tag="ace_qwen_training_time_nothink_${mode}_s${seed}_n50_4096_aw50_bf50_sw100_bfcl_appworld_swebench_all"
    output_dir="$OUTPUT_BASE/$tag"
    metrics="$output_dir/online_metrics.jsonl"
    checkpoint="$output_dir/playbook_ace_${mode}_global.json"
    if [[ ! -f "$metrics" || ! -f "$checkpoint" ]]; then
        echo "Missing checkpoint or metrics: $output_dir" >&2
        exit 1
    fi
    completed="$(wc -l < "$metrics")"
    if [[ "$completed" -ge 150 ]]; then
        echo "Already complete: $mode seed=$seed ($completed/150)"
        continue
    fi

    for attempt in 1 2 3; do
        echo "Resuming $mode seed=$seed from $completed/150, attempt=$attempt, port_block=$port_base-$((port_base + 999))"
        if bash "$script_dir/run_experiment_qwen_training_time.sh"; then
            completed="$(wc -l < "$metrics")"
            if [[ "$completed" -ge 150 ]]; then
                echo "Completed: $mode seed=$seed ($completed/150)"
                break
            fi
        fi
        if [[ "$attempt" -eq 3 ]]; then
            echo "Stopped after three attempts: $mode seed=$seed" >&2
            exit 1
        fi
        completed="$(wc -l < "$metrics")"
        sleep 30
    done
done
