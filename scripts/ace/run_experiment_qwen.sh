#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../.venv/bin/activate

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8006/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EXGENTIC_LLM_TRACE_FORMAT="dedup_v1"

MODEL="openai/Qwen/Qwen3.6-35B-A3B"
SEED="${SEED:-42}"
NUM_TASKS="${NUM_TASKS:-50}"
MODE="${MODE:-sequential}" # isolated | sequential | interleaved
MAX_TOKENS="${MAX_TOKENS:-default}"
# A benchmark-specific setting overrides MAX_TURNS; otherwise the requested
# defaults are AppWorld/BFCL 50 and terminal benchmarks 100.
MAX_TURNS="${MAX_TURNS:-}"
APPWORLD_MAX_TURNS="${ACE_APPWORLD_MAX_TURNS:-${MAX_TURNS:-50}}"
BFCL_MAX_TURNS="${ACE_BFCL_MAX_TURNS:-${MAX_TURNS:-50}}"
SWEBENCH_MAX_TURNS="${ACE_SWEBENCH_MAX_TURNS:-${MAX_TURNS:-100}}"
TERMINALBENCH2_MAX_TURNS="${ACE_TERMINALBENCH2_MAX_TURNS:-${MAX_TURNS:-100}}"
OUTPUT_BASE="${OUTPUT_BASE:-./outputs}"
BENCHMARKS="${ACE_BENCHMARKS:-bfcl,appworld,swebench}"
ACE_TRAINING_TIME="${ACE_TRAINING_TIME:-0}"
if [[ "$ACE_TRAINING_TIME" != "0" && "$ACE_TRAINING_TIME" != "1" ]]; then
    echo "ACE_TRAINING_TIME must be 0 or 1" >&2
    exit 2
fi

IFS=',' read -r -a benchmark_list <<< "$BENCHMARKS"
if [[ ${#benchmark_list[@]} -eq 0 ]]; then
    echo "ACE_BENCHMARKS must contain at least one benchmark" >&2
    exit 2
fi
for benchmark in "${benchmark_list[@]}"; do
    case "$benchmark" in
        appworld|bfcl|swebench|terminalbench2) ;;
        *) echo "Unsupported benchmark: $benchmark" >&2; exit 2 ;;
    esac
done

if [[ ",${BENCHMARKS}," == *,swebench,* || ",${BENCHMARKS}," == *,terminalbench2,* ]]; then
    # A mounted local socket or an explicitly configured remote Docker host
    # is required. Do not bake a private server address into this script.
    if [[ -z "${DOCKER_HOST:-}" && ! -S /var/run/docker.sock ]]; then
        export DOCKER_HOST="${ACE_DOCKER_HOST:-}"
    fi
    if [[ -z "${DOCKER_HOST:-}" ]]; then
        echo "Docker benchmark needs a mounted Docker socket or DOCKER_HOST/ACE_DOCKER_HOST." >&2
        exit 1
    fi
    if ! command -v docker > /dev/null 2>&1; then
        echo "Docker benchmark needs a Docker CLI inside this environment; 'docker' is not on PATH." >&2
        echo "In a container, also mount the host Docker socket or set DOCKER_HOST." >&2
        echo "Until then, use ACE_BENCHMARKS=appworld,bfcl for the available benchmarks." >&2
        exit 1
    fi
    if ! timeout 10s docker info > /dev/null 2>&1; then
        echo "Docker benchmark needs a reachable Docker daemon (check /var/run/docker.sock or DOCKER_HOST)." >&2
        exit 1
    fi
fi

case "$MODE" in
    isolated|sequential|interleaved) ;;
    *) echo "Invalid MODE: $MODE" >&2; exit 2 ;;
esac

# These benchmarks need tool calls. Fail before selecting tasks if the
# server was started without automatic tool-call support.
if ! curl --fail --silent --show-error --max-time 30 \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    --data '{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Call the ping tool."}],"tools":[{"type":"function","function":{"name":"ping","description":"Return pong","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto","max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' \
    "${OPENAI_API_BASE%/}/chat/completions" > /dev/null; then
    echo "Qwen tool-call check failed at ${OPENAI_API_BASE}." >&2
    echo "Start vLLM with --enable-auto-tool-choice --tool-call-parser qwen3_coder (and --reasoning-parser qwen3)." >&2
    exit 1
fi

settings_args=()
if [[ -n "${ACE_TASK_SPLIT_MANIFEST:-}" ]]; then
    settings_args+=(--task-split-manifest "$ACE_TASK_SPLIT_MANIFEST")
fi
if [[ -n "$MAX_TURNS" ]]; then
    settings_args+=(--max-turns "$MAX_TURNS")
fi
settings_args+=(
    --appworld-max-turns "$APPWORLD_MAX_TURNS"
    --bfcl-max-turns "$BFCL_MAX_TURNS"
    --swebench-max-turns "$SWEBENCH_MAX_TURNS"
    --terminalbench2-max-turns "$TERMINALBENCH2_MAX_TURNS"
    --terminalbench2-actor "${ACE_TERMINALBENCH2_ACTOR:-terminus2}"
)
if [[ "$ACE_TRAINING_TIME" == "1" ]]; then
    settings_args+=(--training-time)
fi
if [[ "$MAX_TOKENS" != "default" ]]; then
    settings_args+=(--max-tokens "$MAX_TOKENS")
fi
if [[ -n "${ACE_APPWORLD_MAX_INTERACTIONS:-}" ]]; then
    settings_args+=(--appworld-max-interactions "$ACE_APPWORLD_MAX_INTERACTIONS")
fi
if [[ -n "${ACE_SWEBENCH_MAX_INTERACTIONS:-}" ]]; then
    settings_args+=(--swebench-max-interactions "$ACE_SWEBENCH_MAX_INTERACTIONS")
fi

run_benchmarks() {
    local benchmark_list="$1"
    local output_dir="$2"
    python -u run_experiment.py \
        --mode "$MODE" --seed "$SEED" --num-tasks "$NUM_TASKS" \
        --model "$MODEL" "${settings_args[@]}" \
        --disable-thinking \
        --benchmarks "$benchmark_list" \
        --output-dir "$output_dir" \
        2>&1 | tee -a "${output_dir}.log"
}

mkdir -p "$OUTPUT_BASE"
turn_tag="aw${APPWORLD_MAX_TURNS}_bf${BFCL_MAX_TURNS}_sw${SWEBENCH_MAX_TURNS}_tb${TERMINALBENCH2_MAX_TURNS}"
run_tag="ace_qwen_nothink_${MODE}_s${SEED}_n${NUM_TASKS}_${MAX_TOKENS}_${turn_tag}_${BENCHMARKS//,/_}"
if [[ "$ACE_TRAINING_TIME" == "1" ]]; then
    run_tag="ace_qwen_training_time_nothink_${MODE}_s${SEED}_n${NUM_TASKS}_${MAX_TOKENS}_${turn_tag}_${BENCHMARKS//,/_}"
fi
if [[ -n "${ACE_APPWORLD_MAX_INTERACTIONS:-}" ]]; then
    run_tag+="_appmax${ACE_APPWORLD_MAX_INTERACTIONS}"
fi
if [[ -n "${ACE_SWEBENCH_MAX_INTERACTIONS:-}" ]]; then
    run_tag+="_swemax${ACE_SWEBENCH_MAX_INTERACTIONS}"
fi

if [[ "$MODE" == "isolated" ]]; then
    for benchmark in "${benchmark_list[@]}"; do
        run_benchmarks "$benchmark" "${OUTPUT_BASE}/${run_tag}_${benchmark}"
    done
else
    run_benchmarks "$BENCHMARKS" "${OUTPUT_BASE}/${run_tag}_all"
fi

echo "Completed: $run_tag"
