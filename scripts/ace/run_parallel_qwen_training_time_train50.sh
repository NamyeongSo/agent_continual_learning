#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

# Launch three independent ACE training runs against one shared Qwen server.
# The frozen seed-42 manifest supplies the same 50 train IDs per benchmark;
# SEED changes only their order. Validation IDs are never executed here.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
worker="$script_dir/run_experiment_qwen_training_time.sh"
manifest="${ACE_TASK_SPLIT_MANIFEST:-$repo_root/scripts/utils/task_splits/seed42_train_val_test.json}"
mode="${ACE_MODE:-sequential}"
case "$mode" in
    sequential)
        default_session_prefix="ace_train"
        port_bases=(36000 37000 38000)
        ;;
    interleaved)
        default_session_prefix="ace_interleaved_train"
        port_bases=(39000 40000 41000)
        ;;
    *) echo "ACE_MODE must be sequential or interleaved" >&2; exit 2 ;;
esac
output_base="${OUTPUT_BASE:-$script_dir/outputs/ace_parallel_${mode}_train50_$(date +%Y%m%d_%H%M%S)}"
session_prefix="${ACE_TMUX_PREFIX:-$default_session_prefix}"
docker_host="${DOCKER_HOST:-${ACE_DOCKER_HOST:-}}"
dry_run=0

if [[ "${1:-}" == "--dry-run" && "$#" -eq 1 ]]; then
    dry_run=1
elif [[ "$#" -ne 0 ]]; then
    echo "Usage: DOCKER_HOST=ssh://your-docker-host $0 [--dry-run]" >&2
    exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required to keep the three runs alive." >&2
    exit 1
fi
if [[ ! -f "$manifest" ]]; then
    echo "Train/validation manifest not found: $manifest" >&2
    exit 1
fi
if [[ ! -f "$worker" ]]; then
    echo "ACE worker script not found: $worker" >&2
    exit 1
fi
if [[ -z "$docker_host" && ! -S /var/run/docker.sock ]]; then
    echo "Set DOCKER_HOST (or ACE_DOCKER_HOST) for remote SWE-bench Docker." >&2
    exit 1
fi

docker_cli="$repo_root/.venv/bin/docker"
if [[ ! -x "$docker_cli" ]]; then
    docker_cli="$(command -v docker || true)"
fi
if [[ -z "$docker_cli" ]]; then
    echo "Docker CLI not found in the project venv or PATH." >&2
    exit 1
fi
if ! DOCKER_HOST="$docker_host" timeout 15s "$docker_cli" info >/dev/null 2>&1; then
    echo "Cannot reach SWE-bench Docker daemon via DOCKER_HOST=$docker_host" >&2
    exit 1
fi

# The child script changes to its own directory, so always give absolute paths.
manifest="$(realpath "$manifest")"
if [[ "$dry_run" == "0" ]]; then
    mkdir -p "$output_base"
fi
output_base="$(realpath -m "$output_base")"

echo "ACE training: mode=$mode, 50 train tasks per benchmark"
echo "Frozen split: $manifest"
echo "Output base: $output_base"
echo "Shared model endpoint: ${OPENAI_API_BASE:-http://127.0.0.1:8006/v1}"

read -r -a seeds <<< "${ACE_SEEDS:-42 43 44}"
if [[ "${#seeds[@]}" -eq 0 ]]; then
    echo "ACE_SEEDS must contain at least one seed" >&2
    exit 2
fi
for seed in "${seeds[@]}"; do
    case "$seed" in
        42|43|44) ;;
        *) echo "ACE_SEEDS supports only seeds 42, 43, and 44" >&2; exit 2 ;;
    esac
    index=$((seed - 42))
    port_base="${port_bases[$index]}"
    session="${session_prefix}_s${seed}"
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Already running: $session (skipped)"
        continue
    fi
    if [[ "$dry_run" == "1" ]]; then
        echo "Would start: $session, port block ${port_base}-$((port_base + 999))"
        continue
    fi
    tmux new-session -d -s "$session" -c "$script_dir" \
        -e "SEED=$seed" \
        -e "NUM_TASKS=50" \
        -e "MODE=$mode" \
        -e "ACE_BENCHMARKS=bfcl,appworld,swebench" \
        -e "ACE_APPWORLD_MAX_TURNS=50" \
        -e "ACE_BFCL_MAX_TURNS=50" \
        -e "ACE_SWEBENCH_MAX_TURNS=100" \
        -e "MAX_TOKENS=4096" \
        -e "ACE_TASK_SPLIT_MANIFEST=$manifest" \
        -e "OUTPUT_BASE=$output_base" \
        -e "EXGENTIC_PORT_BLOCK_START=$port_base" \
        -e "DOCKER_HOST=$docker_host" \
        -e "OPENAI_API_BASE=${OPENAI_API_BASE:-http://127.0.0.1:8006/v1}" \
        "exec bash '$worker'"
    echo "Started: $session, port block ${port_base}-$((port_base + 999))"
done

echo "Check sessions: tmux list-sessions"
echo "Logs: $output_base/*.log"
