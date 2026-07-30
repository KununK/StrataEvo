# 代码库级自进化

StrataEvo 可以评测、改写并版本化自己的 Agent 实现。每一代执行以下事务：

```text
评测父代
  -> 收集候选代码和工具轨迹为 Evidence
  -> 读取以往代际的 Evolution Memory
  -> Diagnosis 判断主要演化层
  -> Evolution Plan 选择一项可检验的干预
  -> 自修改执行器修改自身源码
  -> Evaluation Contract 检查修改能否被当前 benchmark 观察
  -> [固定检查 -> 探索性评测 -> 根据反馈继续修改]，最多 5 次
  -> 恢复探索阶段最佳候选
  -> 重新评测父代和候选，确认提升后提交，否则整体回滚
  -> 将接受或拒绝的结果写入 Evolution Memory
  -> 在新的 Python 进程中启动下一代
```

这里的自进化与 benchmark 候选代码生成不同。`solution.py` 是 Agent 针对某道任务生成的
答案，而一次自进化会修改 StrataEvo 自身的实现。

## 演化边界

当前允许演化的代码是：

```text
src/tinyagent/
src/strataevo/evolution/mutator.py
```

这包括完整的 Tinyagent 运行时，以及负责决定如何改进 Agent 的自修改策略。
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

Model 层不修改 Git 工作树。显式传入 `--enable-model-evolution` 后，如果 Diagnosis 和
Plan 选择 `model`，系统会重新尝试父代失败任务，筛选 verifier 通过的修复轨迹并加入成功
轨迹 replay，在 GPU 1 上训练 LoRA，再通过 GPU 0 的 vLLM 动态加载候选。该协议在同一批
任务上训练和评测，属于测试时适应实验，不是 held-out 泛化评测。未选择 `model` 时仍执行
原有源码进化。

## Evaluation Contract

可写不代表可以被当前 benchmark 评价。每个 Evaluator 必须声明一份 Evaluation Contract，
说明当前目标以及代码修改的生效范围。HumanEval 和 MBPP 当前都声明：

```text
direct_paths
  src/tinyagent/
  benchmark 子进程会加载，能够用本轮 pass@1 评价

deferred_paths
  src/strataevo/evolution/mutator.py
  只会改变后续自修改过程，本轮 benchmark 不会加载
```

候选只有全部修改都位于 `direct_paths` 时才会进入当前 benchmark。仅修改
`deferred_paths`、混合修改两个范围或者包含未分类路径的候选，都会返回结构化反馈、恢复
父代或已有最佳候选，并且不消耗探索性 benchmark 配额。`mutator.py` 仍然保留在可演化
范围，但需要未来独立的 Evolver 评测契约验证，不能用即时 benchmark 波动证明其改进。

每次运行使用的契约保存在 `evolution/runs/<run_name>/evaluation_contract.json`。

## 启动演化

首先启动 vLLM、激活项目环境，并确保 Git 工作区没有未提交修改。运行一次小规模开发实验：

```bash
strataevo \
  --run-name humaneval-dev \
  --branch evo_test_inter \
  --benchmark humaneval \
  --generations 1 \
  --eval-limit 5
```

该命令要求当前位于或允许切换到 `evo_test_inter` 分支。系统首先评测父代，然后在一次
自修改会话中根据候选评测结果连续修正实现。

继续同一条演化谱系：

```bash
strataevo \
  --run-name humaneval-dev \
  --branch evo_test_inter \
  --generations 3 \
  --resume
```

主要参数：

```text
--mutator-max-steps       Meta-Agent 修改自身时允许的最大模型/工具轮数
--mutator-rounds          每代最多进行的连续修改反馈轮数
--max-eval-attempts       每代最多进行的候选 benchmark 次数
--benchmark               评测集：humaneval 或 mbpp
--eval-offset             benchmark 开发任务的起始位置
--eval-limit              使用的任务数量；默认不限制
--eval-workers            同时发送给 vLLM 的评测 Agent 数量
--benchmark-max-steps     每个被评测 Agent 的最大工具循环步数
--enable-model-evolution  允许 model 计划执行 verifier-guided LoRA SFT
--force-layer model       强制选择一个 model 诊断，仅用于链路测试
--force-layer context     强制选择一个 context 诊断，仅用于链路测试
--force-layer tools       强制选择一个 tools 诊断，仅用于链路测试
--sft-device              LoRA 训练使用的物理 GPU，默认 1
--sft-epochs              LoRA 遍历修复与 replay 训练集的次数，默认 1
--repair-attempts         每个失败任务的修复尝试数，默认 2
```

`--mutator-max-steps` 默认是 `200`，`--mutator-rounds` 和 `--max-eval-attempts` 默认都是
`5`，`--benchmark-max-steps` 默认是 `12`。四个预算相互独立：它们依次控制整代自修改
总步数、连续反馈轮数、候选 benchmark 次数，以及每道评测任务中的 Agent 步数。无修改或
固定验证失败和不符合 Evaluation Contract 的修改会留下 attempt 记录，但不占用探索性
benchmark 次数。最终父代重测和候选确认是独立的晋级检查，不计入该配额。

## 自修改过程

Meta-Agent 可以使用以下受控工具：

```text
list_files       浏览代码库目录
read_file        阅读源码和评测记录
search_files     搜索代码
write_file       创建或重写可演化源码
replace_text     精确修改源码
replace_lines    按 read_file 返回的闭区间行号替换源码
delete_file      删除可演化文件
show_diff        查看当前自身修改
run_validation   运行固定检查
evaluate_candidate  验证并评测当前候选，将结果返回当前自修改会话
```

Meta-Agent 读取父代的评测摘要、失败轨迹和当前实现，然后选择一项具体限制进行修改。
修改首先保留在当前 Git 工作区中，不会立即成为新一代。它可以调用
`evaluate_candidate` 获得当前候选的 benchmark 结果和 Evidence 路径，再在同一会话中继续
修正，默认每代最多评测 5 个候选状态。

为避免模型耗尽全部 step 后才尝试评测，控制器还会强制把整代预算划分为连续 refinement
round。每轮 step 上限根据剩余总预算和剩余 round 数动态均分。默认 200 steps、5 rounds
时初始上限为每轮 40；如果某轮提前结束，未使用的预算会滚入后续轮次。每轮结束后控制器
自动检查当前 diff，并把验证或 benchmark 结果追加到同一个 session 后再启动下一轮；模型
主动调用 `evaluate_candidate` 时，未变化的 patch 会直接复用缓存，不重复消耗评测。Python
注释/格式变化以及结构等价的 JSON/TOML 会被保守识别为 `semantic_noop`，恢复父代或已有
最佳候选，且不运行固定验证、不消耗 benchmark 配额。无法证明等价的修改仍按正常候选评测。
每条候选反馈都会返回 `candidate_retained` 和 `working_tree_state`。前者说明刚提交的 patch
是否仍然生效，后者明确当前工作树是 `current_candidate`、`best_candidate` 还是 `parent`。
如果候选因重复失败、范围越界或分数退化而被恢复，下一轮必须以该状态为准，不能把父代随后
通过固定检查误认为已回滚候选通过了检查。

每次 attempt 的 patch、固定验证日志、评测目录和结果保存在：

```text
evolution/runs/<run_name>/generation-NNNN/attempt-NNNN/
```

候选 benchmark 未超过父代或本代已有最佳候选时，控制器立即恢复父代或最佳 patch，再让
模型继续修改。一代结束时，控制器丢弃最后遗留的未评测修改，并恢复探索阶段 pass@1 最高的
已评测 patch。探索最高分只用于选择候选，不直接决定提交。

如果探索最高分超过已有父代记录，控制器会先恢复父代并重新评测，再恢复候选并进行一次确认
评测。只有候选在这组没有参与候选筛选的新结果中仍严格超过父代才会提交。这样保留代内多
round 反馈，同时避免从多次随机生成中直接选择最高值造成的 best-of-N 晋级偏差。

`replace_text` 只接受恰好出现一次的原文；一次精确匹配失败后，应重新读取相关行并改用
`replace_lines`，避免反复猜测空格。自修改提示要求在前三分之一预算内开始编辑，并保留
最后三分之一用于查看 diff、运行固定验证和修复错误。

## 晋级规则

所选 benchmark 的 pass@1 是当前唯一的晋级指标。Agent 步数、Token 和运行时间作为独立
观测指标保留，不合成为效用分数。

一个新版本只有同时满足以下条件才会被提交：

1. 修改仅位于当前 Evaluation Contract 的 `direct_paths`；
2. Ruff、pytest 和自进化 CLI 启动检查全部通过；
3. 探索阶段选出的最佳候选超过已有父代记录；
4. 新鲜的候选确认 pass@1 严格高于新鲜的父代重测 pass@1。

pass@1 相同的候选即使成本更低也不会晋级。
如果 baseline 或已接受父代的 pass@1 已达到 `1.0`，运行会在 Diagnosis 和自修改之前提前
结束，因为当前晋级规则下不存在更高分数。

如果满足条件，修改会作为新一代提交到 `--branch` 指定的分支，下一代由新的 Python 进程加载。
如果不满足条件，源码会恢复到父代，但本次尝试的补丁和评测记录仍会保留。

LoRA 候选采用同一条严格父子复测规则，但不会创建 Git 提交。接受的 adapter 保存在
generation 目录中，其名称、路径和父 adapter 写入 `state.json` 与 Evolution Memory；
拒绝时从 vLLM 卸载候选并恢复父模型。

在自修改过程中按 `Ctrl+C` 时，控制器会恢复所有尚未提交的可演化文件，并在当前 generation
目录写入 `failure.json`。因此中断不会把未验证的候选留在 Git 工作区中。

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
├── evaluation_contract.json
├── state.json
├── evolution_memory.jsonl
├── summary.json
├── summary.md
├── baseline/
├── generation-0001/
│   ├── diagnosis.json
│   ├── plan.json
│   ├── agent_result.json
│   ├── record.json
│   ├── model/                    # 仅 model 层计划存在
│   │   ├── verified_trajectories.jsonl
│   │   ├── train_config.json
│   │   ├── train.log
│   │   ├── repairs/
│   │   │   ├── generations.jsonl
│   │   │   ├── results.jsonl
│   │   │   └── summary.json
│   │   └── adapter/
│   ├── attempt-0001/
│   │   ├── attempt.json
│   │   ├── changes.patch
│   │   ├── validation.log
│   │   ├── evaluation.log
│   │   └── evaluation/
│   │       └── evidence.json
│   ├── attempt-0002/
│   │   └── ...
│   ├── promotion/
│   │   ├── comparison.json
│   │   ├── parent/evaluation/
│   │   └── candidate/evaluation/
│   └── sessions/
└── generation-0002-failed-0001/
    └── failure.json
```

各文件含义：

- `config.json`：本次演化实验的固定配置；
- `evaluation_contract.json`：当前 benchmark 的目标、直接生效路径和延迟生效路径；
- `state.json`：当前代数、当前提交、已激活模型/context 和父代评分；
- `evolution_memory.jsonl`：所有已完成代的诊断、修改、指标和接受/拒绝结果；
- `summary.json`、`summary.md`：自动更新的 run-level 四层选择、结果和分数汇总；
- `diagnosis.json`：本代主要演化层、关联层、证据、置信度和改进方向；
- `plan.json`：从诊断中选中的单一问题、干预、预期指标和长期价值假设；
- `agent_result.json`：自修改 Agent 的完整消息与工具轨迹；
- `attempt-NNNN/changes.patch`：该次候选对自身源码的累计修改；
- `attempt-NNNN/attempt.json`：该次验证、评测状态和报告；
- `promotion/comparison.json`：探索最佳结果、新鲜父代结果和候选确认结果；
- `record.json`：父代、本代最佳候选、全部 attempts、晋级决定和原因；
- `context/parent.json`、`context/candidate.json`：父代与候选 ContextProfile；
- `context/screening/`、`context/promotion/`：context 候选筛选和新鲜父子复测；
- `model/repairs/`：失败任务的修复轨迹、verifier 结果及修复成功/仍失败任务；
- `model/adapter/training_metrics.json`：LoRA 的 epochs、steps 和 loss；
- `*/evaluation/tasks.jsonl`：该次 benchmark 实际选择的完整任务快照，供修复器复用；
- `attempt-NNNN/validation.log`：Ruff、pytest 和 CLI 检查输出；
- `attempt-NNNN/evaluation/`：该候选的代码、session 和 benchmark 结果；
- `attempt-NNNN/evaluation/evidence.json`：该候选的结构化评测证据；
- `generation-XXXX-failed-XXXX/`：模型请求、诊断或执行异常时保留的未完成代。

即使某一代被拒绝，各 attempt 的 `changes.patch` 仍会保留，保证每次自身修改都可以审计和
复现。
异常中断的代不会推进 `state.json`。再次使用 `--resume` 时，未完成目录会先归档为
`generation-XXXX-failed-XXXX`，然后从同一父代重新执行；包含 `record.json` 的完成目录不会
被自动覆盖。

## 结构化评测证据

任一 coding-agent benchmark 完成后，`CodingAgentEvidenceCollector` 会关联以下文件：

```text
summary.json
results.jsonl
generations.jsonl
candidates/
sessions/
```

它为每道题记录任务状态、停止原因、步骤、Token，以及按执行顺序配对的工具调用、参数和
结果，并记录 `solution.py` 是否曾创建和最终是否保留。收集器只记录可观察事实，不负责
判断问题属于模型、上下文、工具还是架构层；分层归因由后续 Diagnosis 阶段完成。

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

`expected_outcomes` 只能引用父代报告中实际存在的数值指标，例如 `task_score`、
`average_agent_steps` 或 `signal:artifact_missing`。候选评测后，Memory 会记录这些指标的
父代值、候选值以及是否符合预期。Planner 首次返回无效 JSON、错误诊断序号、层级不一致或
不存在的指标时，会收到校验错误并默认修复一次。

当前晋级要求 `task_score` 严格提高，因此父代未满分时，Plan 必须包含
`task_score: increase`。通过任务上的 `max_steps` 和 `average_agent_steps` 只表示效率，
可以作为次级观测，但不能单独成为当前代的进化目标；Diagnosis 也不得把
`passed=true` 的 `max_steps` 案例描述成没有完成任务。

Planner 还会收到真实的可演化根目录和其中已有的文件清单。`likely_files` 必须位于这些
可演化路径内，可以引用已有文件，也可以提出在可演化目录中新建文件；边界外的虚构路径会
触发自动修复重试。

长期价值目前只用于记录研究假设。一个修改即使可能帮助未来进化，仍必须通过当前固定测试、
pass@1 晋级规则；本阶段不会因为推测性的长期价值接受当前无收益的候选。后续
加入独立的进化能力评测后，可以利用这些字段重新分析“当前无用但具有未来价值”的改动。

## Context Evolution

当 Plan 的 `primary_layer` 为 `context` 时，控制器不会启动源码 Mutator，而是生成一个独立
ContextProfile：

```text
system_prompt_addendum  追加到固定 system prompt 的通用指令
task_prompt_addendum    追加到每道任务 user prompt 的通用指令
```

Context 候选不能包含任务 ID、具体解答、hidden tests 或 repair 代码。它先参与 screening；
只有高于当前父代才进入新鲜父子复测。接受后 `state.json.current_context` 指向候选文件，后续
所有 benchmark 和其他层演化都会继续使用该 context；拒绝或异常时 evaluator 立即恢复父代。
Context 文件保存在 run artifact 中，不修改源码，因此不产生 Git commit。

可以使用 `--force-layer context` 单独验证该执行器。正式实验省略该参数，由四层 Diagnosis
和 Evolution Plan 决定是否选择 context。

## Tool Evolution

当 Plan 的 `primary_layer` 为 `tools` 时，控制器生成一个独立 ToolProfile。第一版只允许为
已有工具追加模型可见的通用说明，用于表达适用场景、调用顺序、结果验证和失败恢复：

```json
{
  "description_addenda": {
    "read_file": "在修改前读取与问题直接相关的文件。"
  }
}
```

它不改变工具名称、参数 schema、实现、审批要求或权限，因此工具执行契约保持不变。
ToolProfile 与 ContextProfile 共用候选评测事务：screening 通过后进行新鲜父子复测，接受后
写入 `state.json.current_tool_profile` 并供后续代使用；拒绝或异常时恢复父代。候选保存在
`generation-NNNN/tools/`，包括 `parent.json`、`candidate.json`、`screening/` 和
`promotion/`。可以使用 `--force-layer tools` 单独验证执行器。

## 跨代 Evolution Memory

每个完成的代都会向 `evolution_memory.jsonl` 写入一条结构化记录，包括：

```text
generation、diagnoses、plan、expected outcome observations
changed_paths、patch path / excerpt、agent_output
父代和候选的 task score
hypothesis_verdict、counterevidence 和后续建议
accepted / rejected、原因和 resulting commit
```

拒绝结果进一步区分为：

```text
no_change          自修改执行没有产生源码 diff，尚未检验演化假设
semantic_noop       仅产生可证明的注释、格式或结构等价修改，尚未检验演化假设
validation_failed  产生了 diff，但固定代码检查失败，尚未进入任务评测
deferred_change    修改只会影响后续自进化，当前 benchmark 无法评价
mixed_change_scope 同时修改直接与延迟生效代码，无法归因
unclassified_change 包含 Evaluation Contract 未声明的修改路径
benchmark_rejected 通过固定检查，但任务评测没有满足晋级条件
accepted           通过固定检查和任务评测并已提交
```

Planner 会把没有进入 benchmark 的结果视为执行或评测契约不匹配，而不是该演化方向已经被
基准否定。进入 benchmark 但未晋级的假设会标为 `refuted`，并将分数变化和任务退化写入
`counterevidence`；Planner 只有在出现新证据或不同机制时才应再次选择相同假设。

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
