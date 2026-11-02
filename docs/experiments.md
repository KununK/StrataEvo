# Experimental Results and Reproduction Configurations

Experiments use Qwen3-Coder-30B-A3B-Instruct.
Decision selects components in autonomous experiments; `--force-layer` specifies the component in independent component experiments.
Each generation retains candidates that meet the promotion threshold and restores the parent state for the others.

## 1. Full-Benchmark Evaluation

| Benchmark | Tasks | Generations | Baseline | Final | Net Gain | Pass Rate Gain |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval | 164 | 5 | 144 (87.80%) | 156 (95.12%) | +12 | +7.32 pp |
| MBPP | 257 | 5 | 200 (77.82%) | 232 (90.27%) | +32 | +12.45 pp |
| BFCL V4 | 200 | 10 | 100 (50.00%) | 123 (61.50%) | +23 | +11.50 pp |

HumanEval accepts Context at G1 and G3, adding 10 and 2 passed tasks, respectively.
MBPP accepts Context at G1, adding 32 tasks.
BFCL accepts Context at G2 and G4, adding 19 and 2 tasks, and Architecture at G7, adding 2 tasks.

## 2. Independent Replications

Each benchmark has 3 independent runs, each with 64 tasks and 5 generations.
Runs start from the same initial source code and empty evolutionary state; within each run, promoted states are inherited across generations.

| Benchmark | Run | Baseline | Final | Net Gain |
|---|---|---:|---:|---:|
| HumanEval | R1 | 50/64 | 57/64 | +7 |
| HumanEval | R2 | 50/64 | 58/64 | +8 |
| HumanEval | R3 | 51/64 | 58/64 | +7 |
| MBPP | R1 | 44/64 | 57/64 | +13 |
| MBPP | R2 | 45/64 | 56/64 | +11 |
| MBPP | R3 | 47/64 | 58/64 | +11 |
| BFCL V4 | R1 | 22/64 | 28/64 | +6 |
| BFCL V4 | R2 | 26/64 | 33/64 | +7 |
| BFCL V4 | R3 | 27/64 | 30/64 | +3 |

## 3. Independent Four-Component Matrix

For HumanEval and MBPP, each Context, Tools, and Architecture run uses 32 tasks and 3 generations;
each Model run uses 64 tasks and 1 generation.
BFCL uses all 200 tasks.
Entries show the initial and final retained numbers of passed tasks.

| Component | HumanEval R1 | HumanEval R2 | MBPP R1 | MBPP R2 | BFCL V4 |
|---|---:|---:|---:|---:|---:|
| Context | 25→27 /32 | 22→28 /32 | 18→27 /32 | 23→28 /32 | 95→97 /200 |
| Tools | 24→27 /32 | 22→28 /32 | 22→25 /32 | 20→27 /32 | 93→102 /200 |
| Model | 49→53 /64 | 54→56 /64 | 48→56 /64 | 51→58 /64 | 94→117 /200 |
| Architecture | 25→25 /32 | 26→26 /32 | 26→26 /32 | 24→24 /32 | 95→95 /200 |

The natural-task Architecture experiment on BFCL produces no executable candidate and retains the initial state.
LoRA uses rank=8, 1 epoch, and a learning rate of 1e-4.
The HumanEval Model experiments use a sequence length of 4096; the two MBPP Model experiments in the table use 1024.
The training set is capped at 64 examples, with repair temperature=0.2 and at most 2 repair attempts per failed task.

## 4. Controlled Architecture Repair

The premature-termination defect ends the agent after the first batch of tool calls.
The history-overwrite defect replaces the message history with tool results.
After injecting a defect on a dedicated experiment branch, source repair runs with `--force-layer architecture`, while the evaluator and MetaAgent remain fixed.

| Benchmark | Defect | Run | Defective Baseline | Final | Outcome |
|---|---|---|---:|---:|---|
| HumanEval | Premature termination | R1 | 0/20 | 16/20 | Accepted |
| HumanEval | Premature termination | R2 | 0/20 | 15/20 | Accepted |
| HumanEval | Premature termination | R3 | 0/20 | 16/20 | Accepted |
| MBPP | Premature termination | R1 | 0/20 | 11/20 | Accepted |
| MBPP | Premature termination | R2 | 0/20 | 14/20 | Accepted |
| MBPP | Premature termination | R3 | 0/20 | 13/20 | Accepted |
| BFCL V4 | Premature termination | Full set | 10/200 | 89/200 | Accepted |
| HumanEval | History overwrite | R1 | 6/20 | 14/20 | Accepted |
| HumanEval | History overwrite | R2 | 7/20 | 7/20 | No effective change |
| MBPP | History overwrite | R1 | 3/20 | 12/20 | Accepted |
| MBPP | History overwrite | R2 | 3/20 | 13/20 | Accepted |
| BFCL V4 | History overwrite | Full set | 0/200 | 0/200 | No effective change |

Repairs succeed in 7/7 premature-termination experiments and 3/5 history-overwrite experiments, for a total of 10/12.

## 5. Training-Set Evolution and Frozen Transfer

On all 120 MBPP train tasks, 10 generations of evolution increase the number passed from 89 to 110.
G1 accepts Context; no later candidate is promoted.
The final state contains two prompt addenda about file cleanup and final artifact preservation.
The base model, tool descriptions, and task-agent source code remain unchanged.
After fixing the state, evaluation proceeds on the three target sets, without feeding target results back into search.

| Target Set | Baseline | Final | Net Gain | Pass Rate Gain |
|---|---:|---:|---:|---:|
| MBPP test | 198/257 (77.04%) | 230/257 (89.49%) | +32 | +12.45 pp |
| HumanEval | 135/164 (82.32%) | 152/164 (92.68%) | +17 | +10.37 pp |
| SWE-bench Verified | 161/500 (32.20%) | 171/500 (34.20%) | +10 | +2.00 pp |

SWE uses one paired evaluation under a shared external harness, with 51 newly resolved instances and 41 regressions; exact McNemar p=0.348.
One patch-application failure in Final is counted as unresolved among the 500 tasks.
See [Cross-Benchmark Evaluation](generalization.md) for execution and SWE integration.

## Reproduction

### Full Benchmarks and Independent Replications

See the [Evaluation Guide](evaluation.md) for installation, model serving, and BFCL setup.
The columns below correspond to `--eval-limit`, `--eval-offset`, `--generations`,
`--benchmark-max-steps`, and `--max-eval-attempts`.

| Experiment | Limit | Offset | Generations | Task Step Limit | Candidate Evaluation Limit |
|---|---:|---:|---:|---:|---:|
| HumanEval full benchmark | Unset | 0 | 5 | 12 | 2 |
| MBPP full benchmark | Unset | 0 | 5 | 12 | 2 |
| BFCL full benchmark | Unset | 0 | 10 | 20 | 5 |
| HumanEval independent replication | 64 | 100 | 5 | 12 | 2 |
| MBPP independent replication | 64 | 60 | 5 | 12 | 2 |
| BFCL independent replication | 64 | 0 | 5 | 12 | 5 |

For example, run autonomous evolution on the full MBPP benchmark:

```bash
strataevo --branch evo-mbpp-full --run-name mbpp-full \
  --benchmark mbpp --generations 5 --eval-workers 10 \
  --benchmark-max-steps 12 --max-eval-attempts 2 \
  --enable-model-evolution --sft-device 1 --sft-max-length 4096 --sft-max-samples 64
```

Use a new Git worktree and run name for each independent replication to keep the starting source code identical:

```bash
git worktree add ../StrataEvo-r1 -b experiment/r1 HEAD
```

Install the environment in the new worktree, start or connect to the model service, and run the corresponding configuration.
Each trajectory inherits its own promoted states across generations.

### Selecting a Component

```bash
strataevo --branch evo-context --run-name context-humaneval \
  --benchmark humaneval --force-layer context \
  --generations 3 --eval-offset 101 --eval-limit 32 --eval-workers 10
```

For Context, Tools, and Architecture, the HumanEval/MBPP slice offsets are 101/76, respectively.
When selecting Model, also pass `--enable-model-evolution` and specify the training device, sample count, and sequence length for the target configuration.
The matrix above reports each component's capability under its task configuration; generated trajectories vary with the model service and candidate search.
