#!/usr/bin/env bash
set -euo pipefail

# Override these defaults with environment variables. Extra arguments are
# passed directly to `vllm serve`.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  "$@"
