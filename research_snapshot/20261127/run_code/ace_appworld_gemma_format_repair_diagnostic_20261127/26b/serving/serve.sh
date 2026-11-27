#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
serving_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/thskadud/.vllm/bin/activate
export CUDA_HOME=/home/thskadud/.conda/envs/cuda129
export PATH="$CUDA_HOME/bin:$PATH"
export DG_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
test -x "$DG_JIT_NVCC_COMPILER"
exec vllm serve google/gemma-4-26B-A4B-it \
  --revision "$(cat "$serving_dir/model_revision.txt")" \
  --served-model-name google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 131072 \
  --max-num-batched-tokens 65536 \
  --generation-config vllm \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template "$serving_dir/chat_template.jinja" \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --enable-prefix-caching \
  --max-num-seqs 30 \
  --host 127.0.0.1 \
  --port 8006
