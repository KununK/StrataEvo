# 四层全栈自进化路线

## 研究目标

StrataEvo 的目标不是只优化一个 Prompt、一个工具或一个 Agent loop，而是让系统先依据失败
证据判断主要瓶颈，再选择 Model、Context、Tools 或 Architecture 中的一层或多层进行演化。

目标控制流是：

```text
任务与环境
  -> 结构化 Evidence
  -> 四层 Diagnosis
  -> 选择演化对象和验证协议
  -> 在候选工作区中修改
  -> 按对象重新加载并验证
  -> 接受、拒绝或延迟观察
  -> 跨代 Evolution Memory
```

## 决策一：允许演化整个项目

演化候选应能修改 StrataEvo 中所有受 Git 管理的项目文件，包括：

```text
src/tinyagent/                 Agent 运行时
src/strataevo/                 诊断、计划和演化控制流
eval/                          evaluator、harness 和任务适配器
tests/                         回归测试和系统不变量
配置、Prompt、skills 和文档
模型适配器配置及其版本清单
```

`.git/`、`.venv/`、密钥、缓存、运行日志和原始模型权重不属于源码候选。大模型权重或 Adapter
作为独立版本化 Artifact 管理，Git 只记录其配置、来源、哈希和实验关系。

允许修改 `eval/` 是合理的。Evaluator 本身可能存在错误、不完整反馈、低信息密度或不能覆盖
真实能力的问题；发现并修复这些问题属于系统自进化。测试和诊断器也可能需要演化。

第一版先采用开放式实验：候选可以使用修改后的 tests 和 evaluator 参与验证，以观察模型
获得整个项目后会选择什么方向。此阶段的晋级分数不能直接解释为任务能力提升，必须结合 patch
人工审计。下面的稳定验收原则是观察到真实投机行为后再实现的下一阶段，而不是第一版约束。

## 稳定验收原则

每一代由父代进程或外部 Supervisor 管理候选事务：

```text
父代 Supervisor
  -> 创建隔离候选工作区
  -> 候选可以修改整个项目
  -> 使用父代快照中的验收协议评价候选
  -> 接受后提交
  -> 下一代新进程加载完整候选
```

这不是永久冻结 Supervisor 或 evaluator。候选仍然可以修改它们，只是这些修改不能在同一代
自证有效：

- Agent、Context 和 Tools 的修改可由当前任务 benchmark 直接验证。
- 架构和控制器修改需要子进程重载、回归测试以及完整演化事务测试。
- Evaluator 修改需要固定参考案例、错误复现、与父代 evaluator 的差异报告和外部验收。
- 测试修改只能证明测试系统变化，不能单独证明任务能力提升。
- 下一代加载被接受的控制器和 evaluator 后，它们才成为新的父代。

因此需要的是“按演化对象选择证据”，而不是继续按目录限制模型的写权限。

## 决策二：Model 层采用 TTA

Model 层不应被定义为“换一个更强的基础模型”。更强模型表现更好不能证明当前模型层存在可
修复问题，也不能体现自进化。

Model 层采用 Test-Time Adaptation（TTA）路线：

```text
任务流与失败轨迹
  -> 构造适应信号
  -> 更新 LoRA / Adapter / 小规模可训练参数
  -> 重新加载候选 Adapter
  -> 与父代 Adapter 比较
  -> 版本化接受或回滚
```

优先使用参数高效更新，不直接重写基础模型：

- LoRA 或其他 Adapter；
- verifier、critic 或 reward model 产生的训练信号；
- 执行结果、单元测试和环境反馈产生的可验证奖励；
- 自监督目标、反思轨迹和成功轨迹蒸馏；
- 有限步数、有限 Token 和有限显存的在线更新。

TTA 可以使用当前任务流进行适应，因此它不是传统意义上先固定训练集、再固定测试集的离线
训练流程。实验必须明确报告这是 transductive 还是 online adaptation，以及允许使用哪些
任务信息、执行反馈和预算。

不过，适应机制与科研验收仍需分开。允许用任务流做 TTA，不等于允许读取参考答案或修改判分
标准。能力提升最终仍应由不参与梯度更新的外部执行结果或固定审计协议确认。这里需要的是
不可自我篡改的测量，而不是把系统重新简化成传统训练集/测试集开发。

## 四层演化对象

### Model

对象包括基础模型选择策略、Adapter、优化器状态、TTA 数据构造和更新超参数。

验证需要重新加载候选 Adapter，记录任务收益、适应成本、稳定性和遗忘情况。

### Context

对象包括 system prompt、任务 prompt、上下文选择、压缩、Evolution Memory、示例和检索结果。

修改通常可以立即加载，由任务 benchmark 直接验证。

### Tools

对象包括工具 schema、描述、实现、skills、工具选择策略和工具结果表示。

验证除任务分数外，还应记录调用成功率、无效调用、错误恢复和环境副作用。

### Architecture

对象包括 Agent loop、Planner、Diagnosis、停止条件、并发、状态、候选事务、evaluator 和项目
组织方式。

部分架构修改立即影响任务 Agent，部分只在下一代控制进程中生效。每项候选必须记录实际加载
时机，不能用当前进程没有加载的代码解释当前分数。

## 不按层级限制文件

同一个文件可能同时承载多个演化对象。例如 `agent.py` 同时包含 Context 组织和 Architecture
控制流，`workspace.py` 同时包含 Tools 和安全策略。因此四层分类用于：

- 形成可检验假设；
- 决定如何加载候选；
- 决定需要什么验收证据；
- 分析诊断层、计划层、实际修改层是否一致。

它不应被实现为僵硬的四套目录权限。

每次候选应生成一份 Change Manifest：

```text
diagnosed_layers       Diagnosis 判断的层
planned_layers         Evolution Plan 选择的层
changed_paths          实际修改文件
changed_objects        实际修改的模型、上下文、工具或架构对象
activation             immediate / next_process / next_generation / model_reload
validation_protocol    本候选使用的验证协议
```

## 实施阶段

### 阶段 0：固定当前基线

以 `evo_fullstack_tta` 的开放模式运行 MBPP，结合已有 HumanEval 结果，确认代内反馈、晋级
复测和回滚流程在第二个 benchmark 上的实际表现。

### 阶段 1：全仓库开放修改

当前版本先允许修改全部受版本控制的项目文件，继续使用已有 Git 提交和回滚事务。`.git`、
`.venv`、缓存、密钥、实验记录和评测输出保持不可写。独立 Git worktree 在开放模式验证有效
后再实现，避免同时改动写权限和进程架构。

### 阶段 2：对象与加载时机

实现 Change Manifest 和对象注册表。注册表描述对象如何定位、如何加载以及由什么 evaluator
验证，不负责限制模型只能修改哪些文件。

### 阶段 3：Evaluator 自进化

为 evaluator 建立固定错误案例和一致性检查。Evaluator 候选首先证明修复了可复现问题，没有
吞掉失败或降低测试强度；接受后在下一代生效。

### 阶段 4：Model TTA

先实现单一 LoRA Adapter 的父子版本：

```text
parent adapter
  -> 从失败轨迹构造小规模适应数据
  -> 训练 candidate adapter
  -> 模型服务重新加载
  -> 相同预算下比较
  -> 接受或删除 candidate artifact
```

第一版只允许 Diagnosis 明确选择 Model 层时触发，避免同时改变 Prompt、工具和权重导致无法
归因。

### 阶段 5：统一四层选择

当四类对象都有实际执行器后，让 Planner 根据预期收益、历史成功率、成本和加载时间选择一项
干预。随后研究跨层联合演化，而不是一开始同时修改所有层。

## 当前最近一步

1. 用开放式全仓库模式运行小规模 smoke，确认跨目录 diff、验证、提交和回滚正确。
2. 完成一次全量 MBPP 实验。
3. 汇总 Diagnosis、Plan、实际 diff、加载时机和晋级结果，特别检查 evaluator/tests 修改。
4. 根据真实行为决定是否加入独立 worktree 和稳定 Promotion Evaluator。
5. 全仓库事务稳定后，再开始 Model TTA；不要同时重写候选事务和训练流程。
