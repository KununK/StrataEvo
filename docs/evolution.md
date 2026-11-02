# Evolution

StrataEvo manages four types of agent state through a shared verification, promotion, and recovery process.
Each run establishes an initial baseline, then performs the configured generations of evolution.

## Workflow

1. **Task execution**: Evaluate the current parent and save task outcomes, model interactions, and tool traces.
2. **Evidence extraction**: Extract task status, execution steps, stop reasons, tool events, and errors.
3. **Component selection**: Decision uses current evidence and evolutionary history to propose a modification hypothesis and select one component.
4. **Candidate generation**: The selected executor generates prompts, tool descriptions, an adapter, or a source patch.
5. **Verification and evaluation**: Check the candidate configuration or source code, then evaluate it on the tasks.
6. **Inheritance and recovery**: Save promoted candidates; otherwise retain the parent. Record the outcome in the history.

Each generation runs in a separate Python process so that accepted source changes are loaded in the next generation.

## Components and State

| Component | Candidate | Persistence |
|---|---|---|
| Model | A LoRA SFT adapter trained on verifier-passing repairs and successful-task replay | Adapter directory and loading metadata |
| Context | System/task prompt addenda | Context profile |
| Tools | Description addenda for existing tools | Tools profile |
| Architecture | Source patches to `src/tinyagent/` | Git commit |

The Architecture Mutator uses a fixed MetaAgent to edit source code.
The task agent evolves; the MetaAgent that proposes changes, the evaluators, and the promotion controller remain fixed.

`--enable-model-evolution` enables selection of Model; `--force-layer` selects a component for independent capability experiments.
Omit `--force-layer` for autonomous experiments.
The candidate budget is controlled by `--max-eval-attempts`, which defaults to 5.
Context and Tools end the generation's search as soon as a candidate meets the promotion threshold.

## Verification and Promotion

Source candidates must pass Ruff, unit tests, and CLI checks.
Promotion compares the recorded numbers of passed tasks for the parent and candidate:

```text
required_gain = max(1, floor(parent_passed * 0.01))
candidate_passed - parent_passed >= required_gain
```

An accepted Architecture candidate creates a Git commit; rejection restores the parent's source.
Accepted Context, Tools, and Model candidates save state references that subsequent tasks load.

## Running and Resuming

```bash
strataevo --branch evo-mbpp --run-name mbpp-run \
  --benchmark mbpp --generations 5 --eval-limit 32

strataevo --branch evo-mbpp --run-name mbpp-run --generations 3 --resume
```

The second command adds 3 generations using the configuration in the existing `config.json`.
Each run has a unique name.
Independent replications use new names and start from the same initial source code and empty evolutionary state.

```text
evolution/runs/<run-name>/
  config.json                    Run configuration
  state.json                     Current parent and next generation index
  memory.jsonl                   Completed-generation history
  baseline/evaluation/           Initial baseline
  generation-0001/
    decision.json                Component selection and intervention plan
    record.json                  Generation outcome
    context/ | tools/ | model/   Component-specific candidates
    attempt-0001/                Architecture source candidate and verification
```

An evolved agent consists of its source code, Context, Tools, and LoRA state.
See [Cross-Benchmark Evaluation](generalization.md) for applying it to new tasks.
