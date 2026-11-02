# Evaluation Guide

## Supported Benchmarks

| Benchmark | Data | Task and Scoring |
|---|---|---|
| HumanEval | `openai/openai_humaneval`, test, 164 tasks | Python function generation with executable tests |
| MBPP | `google-research-datasets/mbpp`, sanitized/test, 257 tasks | Short-program generation with assertion tests |
| BFCL V4 | Official `multi_turn_base`, 200 tasks | Multi-turn tool calls, state backend, and official checker |

Each HumanEval and MBPP task has an independent workspace.
The agent reads the task description and produces `solution.py`; the evaluator saves the final candidate and runs tests in a time-limited subprocess.
BFCL uses the benchmark's functions and state environment.

Each evaluation runs one complete agent execution per task.
An execution includes multiple model/tool interactions until completion or the step limit.
Concurrent tasks have independent workspaces and message histories.

## Installation and Model Service

Follow the [project README](../README.md) to install dependencies and start the model service.
Datasets are downloaded from Hugging Face on first use.
For a remote service, set `MODEL` and `BASE_URL`; set `OPENAI_API_KEY` if authentication is required.
When accessing a local service in an environment with proxy settings, configure:

```bash
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
```

BFCL also requires a pinned checkout of the Gorilla repository:

```bash
bash scripts/setup_bfcl.sh
```

The default destination is `eval/vendor/bfcl`, pinned to commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
Set `BFCL_ROOT` to use an existing checkout.
The currently integrated category is `multi_turn_base`.

## Baseline Evaluation

```bash
# Small-scale check
LIMIT=5 RUN_NAME=humaneval-check bash scripts/evaluate.sh humaneval

# Full baseline: leave LIMIT unset
RUN_NAME=mbpp-baseline WORKERS=10 bash scripts/evaluate.sh mbpp
RUN_NAME=bfcl-baseline WORKERS=10 bash scripts/evaluate.sh bfcl

# Task slice and execution budget
OFFSET=100 LIMIT=32 WORKERS=4 RUN_NAME=humaneval-slice \
  bash scripts/evaluate.sh humaneval --max-steps 12 --test-timeout 10
```

| Environment Variable | Default / Purpose |
|---|---|
| `MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| `BASE_URL` | `http://localhost:8000/v1` |
| `WORKERS` | 4 |
| `RUN_NAME` | Output name; set explicitly for repeated experiments |
| `LIMIT` / `OFFSET` | Task count / starting offset |
| `OUTPUT_DIR` / `LOG_PATH` | Custom result directory / log path |
| `PYTHON_BIN` | Repository-local `.venv/bin/python` |

Additional arguments at the end of the script command are passed to the benchmark CLI.
Completed tasks are reused by default.
`FORCE_RERUN=1` clears the run's output before reevaluation; use a new `RUN_NAME` to preserve independent experiments.
The legacy entry points `eval/run_humaneval.sh`, `eval/run_mbpp.sh`, and `eval/run_bfcl.sh` forward to the unified script.

You can also invoke the Python modules directly to inspect all options:

```bash
python -m eval.humaneval.run --help
python -m eval.mbpp.run --help
python -m eval.bfcl.run --help
```

MBPP supports `--split train`, `--split validation`, and `--split test`.
For HumanEval and MBPP, `--dataset-file tasks.jsonl` loads local data; see the loaders and tests for field examples.

## Result Files

```text
eval/outputs/<benchmark>/<run-name>/
  config.json          Complete configuration
  tasks.jsonl          Selected tasks
  generations.jsonl    Model outputs and tool traces
  results.jsonl        Per-task scores, statuses, and timings
  summary.json         Pass rate and cost summary
  candidates/          Final source code for coding benchmarks
```

Terminal logs are saved to `eval/logs/<benchmark>/<run-name>.log`.
Evolution evaluations also extract `evidence.json`.
Evidence can be generated offline from existing standalone evaluation results:

```bash
python -m strataevo.evolution.evidence eval/outputs/mbpp/mbpp-baseline
```

## Self-Evolution and Independent Replications

```bash
strataevo --branch evo-bfcl --run-name bfcl-evolution \
  --benchmark bfcl --generations 5 --eval-limit 64 --eval-workers 10 \
  --enable-model-evolution --sft-device 1 --sft-max-length 4096
```

Each independent replication starts from the same initial source code and empty Context/Tools/LoRA state, with a new branch and run name.
We recommend a separate Git worktree for each trajectory so that source changes accepted in an earlier run are not inherited by the next.
See [Experimental Results](experiments.md) for the paper's results and budgets.

## Execution Environment

Code evaluation uses temporary workspaces, separate Python subprocesses, and timeouts.
Model-generated shell commands have the permissions of the execution environment; use containers or virtual machines for untrusted tasks.
Plan GPU memory for LoRA training separately; inference and training devices are configured through their respective options.
