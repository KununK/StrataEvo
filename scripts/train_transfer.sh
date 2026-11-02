#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  printf 'Usage: RUN_NAME=name SFT_DEVICE=1 bash scripts/train_transfer.sh\n'
  printf 'Evolves on MBPP train, freezes state, then evaluates MBPP test and HumanEval.\n'
  printf 'Environment: MODEL BASE_URL GENERATIONS WORKERS SFT_DEVICE PYTHON_BIN\n'
  exit 0
fi
if [[ "$#" != 0 ]]; then
  printf 'Use environment variables to configure this script; see --help.\n' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
BRANCH="$(git branch --show-current)"
RUN_NAME="${RUN_NAME:-mbpp-train-transfer-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="$ROOT/evolution/runs/$RUN_NAME"
OUTPUT="$ROOT/eval/outputs/generalization/$RUN_NAME"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
WORKERS="${WORKERS:-10}"
if [[ -z "$BRANCH" || "$BRANCH" == main ]]; then
  printf 'Create an experiment branch before running train transfer.\n' >&2
  exit 1
fi
if [[ -e "$RUN_DIR" || -e "$OUTPUT" ]]; then
  printf 'RUN_NAME already exists; use a fresh name. See docs/generalization.md for recovery.\n' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Commit or set aside source changes before freezing an experiment.\n' >&2
  exit 1
fi
"$PYTHON_BIN" -m eval.generalization freeze --snapshot "$OUTPUT/baseline.json" --model "$MODEL"
"$PYTHON_BIN" -m strataevo.evolution.cli \
  --run-name "$RUN_NAME" --branch "$BRANCH" --benchmark mbpp --eval-split train \
  --generations "${GENERATIONS:-10}" --eval-workers "$WORKERS" --max-eval-attempts 2 \
  --model "$MODEL" --base-url "$BASE_URL" --enable-model-evolution \
  --sft-device "${SFT_DEVICE:-1}" --sft-max-length 4096 --sft-max-samples 64 \
  --sft-epochs 1 --sft-learning-rate 1e-4
"$PYTHON_BIN" -m eval.generalization freeze --snapshot "$OUTPUT/final.json" \
  --state-file "$RUN_DIR/state.json" --model "$MODEL"
for STATE in baseline final; do
  "$PYTHON_BIN" -m eval.generalization export --snapshot "$OUTPUT/$STATE.json" \
    --output-dir "$OUTPUT/${STATE}_bundle"
  "$PYTHON_BIN" -m eval.generalization evaluate --snapshot "$OUTPUT/$STATE.json" \
    --output-dir "$OUTPUT/$STATE" --base-url "$BASE_URL" --workers "$WORKERS"
done
