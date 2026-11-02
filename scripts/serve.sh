#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"

exec vllm serve "${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-auto}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK:-8}" \
  "$@"
