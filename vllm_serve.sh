#!/usr/bin/env bash
set -euo pipefail

# Override these defaults with environment variables. Extra arguments are
# passed directly to `vllm serve`.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  "$@"
