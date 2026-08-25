#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
RUN_NAME="${RUN_NAME:-qwen3-coder-bfcl-v4}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/outputs/bfcl/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-eval/logs/bfcl/${RUN_NAME}.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-4}"

args=(
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --output-dir "${OUTPUT_DIR}"
  --workers "${WORKERS}"
  --bfcl-root "${BFCL_ROOT:-eval/vendor/bfcl}"
)

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi
if [[ -n "${OFFSET:-}" ]]; then
  args+=(--offset "${OFFSET}")
fi
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  args+=(--no-resume)
  tee_args=()
else
  tee_args=(-a)
fi

mkdir -p "$(dirname "${LOG_PATH}")"
"${PYTHON_BIN}" -m eval.bfcl.run "${args[@]}" "$@" 2>&1 | tee "${tee_args[@]}" "${LOG_PATH}"
