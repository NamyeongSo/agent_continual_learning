#!/usr/bin/env bash
set -euo pipefail
serving_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/thskadud/.vllm/bin/activate
export CUDA_HOME=/home/thskadud/.conda/envs/cuda129
export PATH="$CUDA_HOME/bin:$PATH"
export DG_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"
export CUDA_VISIBLE_DEVICES=1
test -x "$DG_JIT_NVCC_COMPILER"
exec vllm serve Qwen/Qwen3.5-9B \
  --revision "$(cat "$serving_dir/model_revision.txt")" \
  --served-model-name Qwen/Qwen3.5-9B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --chat-template "$serving_dir/chat_template.jinja" \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --enable-prefix-caching \
  --max-num-seqs 128 \
  --port 8006
