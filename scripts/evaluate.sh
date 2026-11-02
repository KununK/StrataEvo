#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: bash scripts/evaluate.sh {humaneval|mbpp|bfcl} [benchmark arguments]\n'
  printf 'Environment: MODEL BASE_URL RUN_NAME WORKERS LIMIT OFFSET OUTPUT_DIR LOG_PATH PYTHON_BIN\n'
  exit 0
fi
BENCHMARK="${1:-}"
case "$BENCHMARK" in
  humaneval|mbpp|bfcl) shift ;;
  *) printf 'Expected benchmark: humaneval, mbpp, or bfcl.\n' >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
DEFAULT_RUN=qwen3-coder-30b-agent
if [[ "$BENCHMARK" == bfcl ]]; then DEFAULT_RUN=qwen3-coder-bfcl-v4; fi
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/outputs/$BENCHMARK/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-eval/logs/$BENCHMARK/$RUN_NAME.log}"
args=(--model "$MODEL" --base-url "$BASE_URL" --output-dir "$OUTPUT_DIR"
      --workers "${WORKERS:-4}")
if [[ "$BENCHMARK" == bfcl ]]; then args+=(--bfcl-root "${BFCL_ROOT:-eval/vendor/bfcl}"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "$LIMIT"); fi
if [[ -n "${OFFSET:-}" ]]; then args+=(--offset "$OFFSET"); fi
tee_args=(-a)
if [[ "${FORCE_RERUN:-0}" == 1 ]]; then
  args+=(--no-resume)
  tee_args=()
fi
mkdir -p "$(dirname "$LOG_PATH")"
"$PYTHON_BIN" -m "eval.$BENCHMARK.run" "${args[@]}" "$@" 2>&1 | tee "${tee_args[@]}" "$LOG_PATH"
