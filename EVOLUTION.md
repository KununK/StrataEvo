# 代码库级自进化

StrataEvo 可以评测、改写并版本化自己的 Agent 实现。每一代执行以下事务：

```text
评测父代
  -> 收集候选代码和工具轨迹为 Evidence
  -> 读取以往代际的 Evolution Memory
  -> Diagnosis 判断主要演化层
  -> Evolution Plan 选择一项可检验的干预
  -> 自修改执行器修改自身源码
  -> 执行固定的代码检查和测试
  -> 评测修改后的 Agent
  -> 提交改进版本或回滚
  -> 将接受或拒绝的结果写入 Evolution Memory
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
src/strataevo/evolution/diagnosis.py  四层诊断器
src/strataevo/evolution/plan.py  单目标演化计划器
src/strataevo/evolution/evidence.py  评测证据收集器
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
├── evolution_memory.jsonl
├── baseline/
├── generation-0001/
│   ├── diagnosis.json
│   ├── plan.json
│   ├── agent_result.json
│   ├── changes.patch
│   ├── record.json
│   ├── validation.log
│   ├── evaluation.log
│   ├── evaluation/
│   │   └── evidence.json
│   └── sessions/
└── generation-0002-failed-0001/
    └── failure.json
```

各文件含义：

- `config.json`：本次演化实验的固定配置；
- `state.json`：当前代数、当前提交和父代评分；
- `evolution_memory.jsonl`：所有已完成代的诊断、修改、指标和接受/拒绝结果；
- `diagnosis.json`：本代主要演化层、关联层、证据、置信度和改进方向；
- `plan.json`：从诊断中选中的单一问题、干预、预期指标和长期价值假设；
- `agent_result.json`：Meta-Agent 的完整消息与工具轨迹；
- `changes.patch`：本代对自身源码的修改；
- `record.json`：父子代指标、晋级决定和原因；
- `validation.log`：Ruff、pytest 和 CLI 检查输出；
- `evaluation/`：该版本的 HumanEval 候选代码、session 和结果。
- `evaluation/evidence.json`：由评测结果、候选代码和工具轨迹整理出的结构化证据。
- `generation-XXXX-failed-XXXX/`：模型请求、诊断或执行异常时保留的未完成代。

即使某一代被拒绝，`changes.patch` 仍会保留，保证每次自身修改都可以审计和复现。
异常中断的代不会推进 `state.json`。再次使用 `--resume` 时，未完成目录会先归档为
`generation-XXXX-failed-XXXX`，然后从同一父代重新执行；包含 `record.json` 的完成目录不会
被自动覆盖。

## 结构化评测证据

HumanEval 完成后，`HumanEvalEvidenceCollector` 会关联以下文件：

```text
summary.json
results.jsonl
generations.jsonl
candidates/
sessions/
```

它为每道题记录任务状态、停止原因、步骤、Token、工具序列、Shell 命令，以及
`solution.py` 是否曾创建、删除和最终保留。收集器只记录可观察事实，不负责判断问题属于
模型、上下文、工具还是架构层；分层归因由后续 Diagnosis 阶段完成。

已有 HumanEval 结果可以离线生成证据，无需再次调用模型：

```bash
python -m strataevo.evolution.evidence \
  eval/outputs/humaneval/qwen3-coder-30b-agent
```

默认输出到原评测目录的 `evidence.json`。

## 四层诊断

每一代修改源码前，`EvidenceDiagnoser` 会将异常任务和高成本任务压缩为诊断上下文，要求
模型在以下四层中选择主要演化对象：

```text
model         模型本身或推理配置形成的稳定能力限制
context       Prompt、历史选择、memory、压缩和信息呈现
tools         工具 schema、描述、实现、skills 和结果表示
architecture  Agent loop、停止、验证、恢复、状态和编排
```

诊断结果保存为 `diagnosis.json`，每项问题都包含可核验的任务 ID 和观察事实。模型层不能
仅以“更强模型表现更好”为依据。自修改执行器必须沿诊断方向修改，并先读取引用的轨迹验证
假设。四层归因用于形成假设和审计，不对文件实施严格的分层权限；同一个模块可能同时包含
上下文、工具和架构行为。如果模型第一次没有返回合法 schema，诊断器会把校验错误反馈给
模型并默认修复重试一次；所有原始尝试及累计 Token 都保存在 `diagnosis.json`。

已有 HumanEval 证据可以单独诊断：

```bash
python -m strataevo.evolution.diagnosis \
  eval/outputs/humaneval/qwen3-coder-30b-agent
```

该命令会调用配置的模型，并默认写入评测目录下的 `diagnosis.json`。

## Evolution Plan

Diagnosis 可以发现多个问题，但每代只应执行一项可归因的干预。`EvolutionPlanner` 根据
Diagnosis、父代真实指标和历史 Memory 选择一项问题，结果保存为 `plan.json`：

```text
target_diagnosis          选中的诊断序号
primary_layer             本次主要演化层
hypothesis                可检验的因果假设
intervention              一项聚焦的行为改进
expected_outcomes         当前评测可验证的指标及方向
likely_files              建议关注的文件，不是权限边界
expected_long_term_value  可能支持的未来改进
prerequisites             实现长期价值需要的前置条件
```

`expected_outcomes` 只能引用父代报告中实际存在的数值指标，例如 `task_score`、`utility`、
`average_agent_steps` 或 `signal:artifact_missing`。候选评测后，Memory 会记录这些指标的
父代值、候选值以及是否符合预期。Planner 首次返回无效 JSON、错误诊断序号、层级不一致或
不存在的指标时，会收到校验错误并默认修复一次。

长期价值目前只用于记录研究假设。一个修改即使可能帮助未来进化，仍必须通过当前固定测试、
任务分数和 utility 晋级规则；本阶段不会因为推测性的长期价值接受当前无收益的候选。后续
加入独立的进化能力评测后，可以利用这些字段重新分析“当前无用但具有未来价值”的改动。

## 跨代 Evolution Memory

每个完成的代都会向 `evolution_memory.jsonl` 写入一条结构化记录，包括：

```text
generation、diagnoses、plan、expected outcome observations
changed_paths、patch path / excerpt、agent_output
父代和候选的 task score / utility、utility delta
accepted / rejected、原因和 resulting commit
```

Diagnosis 会读取最近的历史结果，避免在证据没有变化时反复提出已被拒绝的假设。自修改
执行器会优先读取与本次主要层或关联层匹配的历史，同时补充最近的其他记录。历史只作为
经验，不作为当前根因的证明；最终仍由固定验证和真实评测决定候选是否晋级。

Memory 使用代数作为唯一键并采用原子文件替换。重复写入完全相同的一代不会产生重复记录，
内容冲突则会报错。异常中断的未完成代不写入 Memory，使用 `--resume` 时会从已有记录继续。
Memory 位于已被 Git 忽略的运行目录中，因此不会污染源码提交，但接受和拒绝的尝试都会保留。

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
