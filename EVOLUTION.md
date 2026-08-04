# Minimal Evolution Protocol

## 1. Baseline

首次运行 benchmark 后，系统把报告和当前 Git commit 写入 `state.json`。每一代在独立 Python
进程中执行，确保接受的源码修改只在下一代加载。

## 2. Evidence

Evidence 仅包含可观察事实：

- task id、pass/status、停止原因和步数；
- 候选是否创建和保留；
- 按执行顺序配对的工具调用、参数与结果；
- Agent 或 verifier 错误；
- 由这些事实直接得到的简单信号。

Evidence 不做层归因，也不从历史任务补充当前证据。

## 3. Evolution Decision

一次结构化模型调用根据当前 Evidence 选择 Model、Context、Tools 或 Architecture，并输出：

- `evidence` 和 `affected_tasks`：当前任务中的直接依据；
- `hypothesis`：认为问题为何发生；
- `intervention`：本代实施的一项行为变化；
- `likely_files`：Architecture 的目标文件提示。

Memory 只提供最近几代的假设、干预和结果，用于避免原样重复失败方向。Decision 不包含
自报置信度、implementation-ready symbol、禁止修改列表或派生指标。它们只有在对照实验证明
能够提高有效候选率时才应加入。

## 4. Executors

Context 和 Tools 最多产生 `max_eval_attempts` 个不同 profile，首个达到晋级阈值的候选立即
接受；若全部拒绝则报告其中分数最高的候选。Model
收集父代失败任务的 verifier-passing repair，训练 LoRA 并动态加载评测。Architecture 使用
`list_files/read_file/search_files/write_file/replace_text/replace_lines/delete_file/show_diff/
run_validation/evaluate_candidate` 修改 `src/tinyagent/`。

Architecture 的每个源码候选执行：

```text
save patch -> Ruff/pytest/CLI -> benchmark -> retain best executable patch
```

验证失败的候选保留在当前工作树供同一代修复；非最佳已评测候选恢复为当前最佳候选或父代。
代末未评测修改会被丢弃。

## 5. Promotion

晋级只比较记录父代与候选的一次 benchmark：

```text
required_gain = max(1, floor(parent_passed * 0.01))
candidate_passed - parent_passed >= required_gain
```

Architecture 接受后创建 Git commit；拒绝或异常时整体回滚。Context、Tools 和 Model 将接受的
profile/adapter 写入 state，不创建源码 commit。

该最小协议不声称消除采样噪声。它保留一条简单、明确的选择规则，避免把重复测评和统计推断
混入基础能力实现。

## 6. Persistence

```text
evolution/runs/<run>/
  config.json
  state.json
  memory.jsonl
  baseline/evaluation/
  generation-NNNN/
    decision.json
    record.json
    agent_result.json              # Architecture
    attempt-NNNN/                  # Architecture candidates
    context/ | tools/ | model/     # external candidates
```

`state.json` 是恢复运行的唯一权威状态；`memory.jsonl` 是完成代的最小 append-only 历史。其他
文件是实验 artifact，不参与控制决策，除非显式作为当前 Evidence 输入。
