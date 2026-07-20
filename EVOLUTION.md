# 代码库级自进化

StrataEvo 可以评测、改写并版本化自己的 Agent 实现。每一代执行以下事务：

```text
评测父代
  -> Meta-Agent 修改自身源码
  -> 执行固定的代码检查和测试
  -> 评测修改后的 Agent
  -> 提交改进版本或回滚
  -> 在新的 Python 进程中启动下一代
```

这里的自进化与 HumanEval 候选代码生成不同。`solution.py` 是 Agent 针对某道任务生成的
答案，而一次自进化会修改 StrataEvo 自身的实现。

## 演化边界

当前允许演化的代码是：

```text
src/tinyagent/
src/strataevo/evolution/mutator.py
```

这包括完整的 Tinyagent 运行时，以及负责决定如何改进 Agent 的 Meta-Agent 策略。
以下环境保持固定：

```text
eval/                           评测基准和奖励信号
tests/                          回归测试
src/strataevo/evolution/cli.py  晋级控制器
src/strataevo/evolution/git.py  Git 提交和回滚
```

如果不保留这条边界，Agent 就可能通过修改评测器提高报告分数，而不是真正提升自身能力。

## 启动演化

首先启动 vLLM、激活项目环境，并确保 Git 工作区没有未提交修改。运行一次小规模开发实验：

```bash
strataevo \
  --run-name humaneval-dev \
  --generations 1 \
  --eval-limit 5
```

该命令会创建或切换到 `evo` 分支。系统首先评测父代，然后执行一次自身修改，最后
评测修改后的实现。

继续同一条演化谱系：

```bash
strataevo \
  --run-name humaneval-dev \
  --generations 3 \
  --resume
```

主要参数：

```text
--mutator-max-steps       Meta-Agent 修改自身时允许的最大模型/工具轮数
--eval-offset             HumanEval 开发任务的起始位置
--eval-limit              使用的开发任务数量
--eval-workers            同时发送给 vLLM 的评测 Agent 数量
--benchmark-max-steps     每个被评测 Agent 的最大工具循环步数
--min-utility-delta       新一代晋级所需的最小效用提升
--max-score-drop          允许的任务分数下降，默认为 0
```

## 自修改过程

Meta-Agent 可以使用以下受控工具：

```text
list_files       浏览代码库目录
read_file        阅读源码和评测记录
search_files     搜索代码
write_file       创建或重写可演化源码
replace_text     精确修改源码
delete_file      删除可演化文件
show_diff        查看当前自身修改
run_validation   运行固定检查
```

Meta-Agent 读取父代的评测摘要、失败轨迹和当前实现，然后选择一项具体限制进行修改。
修改首先保留在当前 Git 工作区中，不会立即成为新一代。

## 晋级规则

HumanEval 的 pass@1 是主要任务分数。默认效用还会惩罚 Agent 步数和 Token 消耗：

```text
utility = pass@1
          - step_penalty * average_agent_steps
          - token_penalty * average_tokens
```

一个新版本只有同时满足以下条件才会被提交：

1. Ruff、pytest 和自进化 CLI 启动检查全部通过；
2. 任务分数不低于配置的最低值；
3. 效用严格高于父代。

如果满足条件，修改会作为新一代提交到 `evo` 分支，下一代由新的 Python 进程加载。
如果不满足条件，源码会恢复到父代，但本次尝试的补丁和评测记录仍会保留。

## 代际生效

Python 模块加载后，修改磁盘源码不会自动改变当前进程中已经导入的类和函数。因此每一代
都在独立的子进程中运行：

```text
父代进程修改源码
  -> 外部控制器评测并提交
  -> 父代进程结束
  -> 新进程从新提交加载源码
  -> 新一代开始演化
```

这种方式允许 Agent 修改多文件代码库、模块关系和自身循环，而不会形成新旧 Python 对象
混合的运行时状态。

## 输出记录

所有运行记录保存在已被 Git 忽略的目录中：

```text
evolution/runs/<run_name>/
├── config.json
├── state.json
├── baseline/
└── generation-0001/
    ├── agent_result.json
    ├── changes.patch
    ├── record.json
    ├── validation.log
    ├── evaluation.log
    ├── evaluation/
    └── sessions/
```

各文件含义：

- `config.json`：本次演化实验的固定配置；
- `state.json`：当前代数、当前提交和父代评分；
- `agent_result.json`：Meta-Agent 的完整消息与工具轨迹；
- `changes.patch`：本代对自身源码的修改；
- `record.json`：父子代指标、晋级决定和原因；
- `validation.log`：Ruff、pytest 和 CLI 检查输出；
- `evaluation/`：该版本的 HumanEval 候选代码、session 和结果。

即使某一代被拒绝，`changes.patch` 仍会保留，保证每次自身修改都可以审计和复现。

## 评测数据边界

演化过程中使用的 HumanEval 任务属于开发反馈。一旦 Agent 根据这些任务的结果修改自身，
这些任务就不再是未经接触的最终测试集。

正式报告结果时，应使用互不重叠的任务范围或另一个基准：

```text
演化集      用于产生反馈并修改 Agent
验证集      用于决定版本是否晋级
最终测试集  只在演化结束后使用一次
```

当前第一阶段使用同一组开发任务完成父子代比较，后续扩展到多数据集时应进一步拆分演化集、
验证集和最终测试集。
