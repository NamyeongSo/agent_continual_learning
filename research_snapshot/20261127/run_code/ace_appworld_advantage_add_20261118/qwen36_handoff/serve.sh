
#!/bin/bash

# ==========================================
# 1. Python Environment
# ==========================================
source /home/thskadud/.vllm/bin/activate

# ==========================================
# 2. CUDA 12.9 Toolkit
# ==========================================
export CUDA_HOME=/home/thskadud/.conda/envs/cuda129
export PATH="$CUDA_HOME/bin:$PATH"
export DG_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"

if [ ! -x "$DG_JIT_NVCC_COMPILER" ]; then
    echo "ERROR: CUDA 12.9 nvcc not found at $DG_JIT_NVCC_COMPILER"
    exit 1
fi

echo "Using CUDA compiler:"
nvcc --version

# ==========================================
# 3. GPU Configuration
# ==========================================
GPU_NOS="1"
export CUDA_VISIBLE_DEVICES=$GPU_NOS

TENSOR_PARALLEL_SIZE=1
MODEL="Qwen/Qwen3.6-35B-A3B-FP8"

# ==========================================
# 4. vLLM Server
# ==========================================
nohup vllm serve "$MODEL" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization 0.95 \
    --max-model-len 262144 \
    --served-model-name "$MODEL" \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --chat-template ~/chat_templates/qwen3_6-35B.jinja \
    --kv-cache-dtype fp8 \
    --language-model-only \
    --enable-prefix-caching \
    --max-num-seqs 128 \
    --port 8006 \
    > "./output_nohup_vllm_server_qwen36_gpu${GPU_NOS}.log" 2>&1 &

echo "vLLM started (PID: $!)"
