# Training-Set Evolution and Cross-Benchmark Evaluation

StrataEvo supports self-evolution on source tasks followed by evaluation of the final agent on new task sets.
The paper uses MBPP train for evolution and evaluates transfer on MBPP test, HumanEval, and SWE-bench Verified.

## MBPP train → MBPP test / HumanEval

Create an experiment branch from the initial source code and empty evolutionary state, start the model service, then run:

```bash
git switch -c experiment/mbpp-transfer
RUN_NAME=mbpp-transfer SFT_DEVICE=1 WORKERS=10 \
  bash scripts/train_transfer.sh
```

This entry point performs autonomous evolution on the training split, fixes the final state, and evaluates Baseline / Final on the two target sets.
Defaults are all 120 MBPP train tasks, 10 generations, and a Model training sequence length of 4096.
Use `MODEL`, `BASE_URL`, `GENERATIONS`, and `SFT_DEVICE` to configure the model and compute resources.

Target results are used to compare transfer performance; search and candidate promotion take place on the source task set.
Results are saved to `eval/outputs/generalization/<run-name>/`.
See [Experimental Results](experiments.md) for the full results.

## SWE-bench Verified Integration

SWE extends the task environment from single-function generation to real code repositories.
The integration retains StrataEvo's task agent and connects task execution to test scoring through
a shared repository workspace and an external SWE harness:

```text
Issue description + initial repository
          ↓
StrataEvo Agent
Browse files → edit code → run commands and tests
          ↓
Generate a repository patch
          ↓
SWE harness applies the patch and runs tests
          ↓
Per-instance resolved outcome
```

The integration has three parts:

1. **Repository task environment.** Prepare the repository and initial commit for each instance, with file and shell tools for search, editing, and testing.
2. **Agent state loading.** Baseline and evolved agents use the same base model, workspace, and tools, while loading their respective source code, Context, tool descriptions, and model adapters.
3. **External test scoring.** Extract the workspace patch after task execution, pass it to the SWE harness for application and testing in the corresponding environment, and read whether the instance is resolved.

The paper evaluates the same Baseline / Final pair on all 500 SWE-bench Verified tasks.
Both conditions use temperature=0, at most 60 agent steps, and 4 concurrent tasks.
The repository workspace and test adapters are supplied by the external SWE harness.
The repository's direct evaluation entry points cover HumanEval, MBPP, and BFCL.

SWE evaluation executes target tasks with fixed agent states.
Both conditions use the same task set and execution budget to measure transfer from source-task evolution to repository repair.
