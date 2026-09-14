# StrataEvo

StrataEvo 是一个最小四层自进化 Agent 研究基线。它评测当前 Agent，从真实任务轨迹中诊断
问题，选择 Model、Context、Tools 或 Architecture 中的一层执行单项干预，并仅在任务分数
明确提升时保留候选。

当前 `main` 对应经过 HumanEval、MBPP 和 BFCL 验证的阶段性四层基线。Context、Tools、Model、
Architecture 均已产生真实、可验证的独立进化证据；不强制层选择的自主闭环在代码生成和多轮
工具任务上均能提升最终父代。该结论证明的是最小闭环具备初步四层自进化能力，不代表四层会在
自然任务中均衡贡献，也不应解释为 held-out 泛化结果。

## 核心边界

生产闭环只保留四项可信边界：

1. Git 对 Architecture 候选提供提交和整体回滚；
2. 候选必须通过 Ruff、单元测试和 CLI 导入检查；
3. 候选必须达到明确的 pass@1 晋级阈值；
4. `state.json` 与 `memory.jsonl` 支持代际状态和断点续跑。

完整执行链：

```text
benchmark -> Evidence -> EvolutionDecision -> one layer executor
          -> validation -> candidate benchmark -> accept or rollback -> state/memory
```

删除的机制包括派生 Causal Trace、run-level summary、Evaluation Contract、语义 no-op 推断、
能力模板、旧记录兼容层、重复父子复测以及 Tinyagent 终端和事件接口。实验目录仍保存原始
Evidence、Decision、候选 patch、验证日志和最终 record，便于离线分析，但这些 artifact
不参与运行控制。

## 目录

```text
src/tinyagent/                    最小模型、消息、工具循环和工作区
src/strataevo/evolution/
  cli.py                          多代进程监督
  generation.py                   单代事务与四层分派
  evidence.py                     任务结果和有序工具事件
  decision.py                     原子的层选择、因果假设和干预
  memory.py                       最小代际记忆
  mutator.py                      固定 MetaAgent 驱动的 Architecture 自修改
  layers/                         四层执行器
  runtime/                        Git、评测、结构化响应和自修改工作区
eval/                             HumanEval、MBPP 和 BFCL 评测环境
tests/                            最小可信边界测试
```

Architecture 只允许修改 `src/tinyagent/`。评测器、测试、Decision 和晋级控制器固定，
避免 Agent 通过修改奖励信号获得虚假提升。

## 环境

```bash
cd /data/vlm/jlk/Agent/StrataEvo
source /data/vlm/jlk/switch-cuda.sh 12.9
uv sync --locked
```

启动模型服务：

```bash
./vllm_serve.sh
```

## 运行

最小 smoke：

```bash
strataevo \
  --run-name humaneval-minimal-smoke \
  --branch main \
  --benchmark humaneval \
  --generations 1 \
  --eval-offset 110 \
  --eval-limit 20 \
  --eval-workers 10
```

启用四层自主选择和 Model LoRA：

```bash
strataevo \
  --run-name mbpp-minimal-four-layer \
  --branch main \
  --benchmark mbpp \
  --generations 5 \
  --eval-workers 10 \
  --enable-model-evolution \
  --sft-max-samples 64
```

继续运行：

```bash
strataevo --run-name mbpp-minimal-four-layer --generations 3 --resume
```

`--force-layer` 仅用于验证单层链路，正式自主实验应省略。

## 四层

- **Model**：对 verifier 通过的 repair trajectory 和成功 replay 进行 LoRA SFT。
- **Context**：演化 system/task prompt addendum。
- **Tools**：演化现有工具的模型可见描述。
- **Architecture**：由自修改 Agent 修改 `src/tinyagent/`，验证后按 benchmark 分数晋级。

除 HumanEval 和 MBPP 外，评测层已支持官方 BFCL V4 `multi_turn_base`，用于观察多轮
工具调用和状态传递。首次运行前执行 `./eval/setup_bfcl.sh`，详见 `eval/README.md`。

Model 层是在当前任务上进行 verifier-guided 测试时适应，不应被解释为 held-out 泛化结果。

## 阶段性实验结果

所有主结果使用同一基础模型 `Qwen3-Coder-30B-A3B-Instruct`、本地 vLLM 和 10 个评测 worker。
自主实验不指定 `--force-layer`；四层矩阵用该参数隔离单层能力。Model 采用 LoRA rank 8、1 epoch、
学习率 `1e-4` 和 4096-token 最大训练长度。候选只有在固定验证通过且通过题数严格增加时才晋级。

以下只展示支撑阶段性结论的四组实验。参数探索、中断恢复、旧实现结果和诊断性中间实验均不列入
主结果。

### 1. 完整基准测评

| 数据集 | 任务数 | 自主代数 | Baseline | Final | 净增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 164 | 5 | 144/164（87.80%） | 156/164（95.12%） | +12 |
| MBPP | 257 | 5 | 200/257（77.82%） | 232/257（90.27%） | +32 |
| BFCL V4 `multi_turn_base` | 200 | 10 | 100/200（50.00%） | 123/200（61.50%） | +23 |

HumanEval 在 G1、G3 接受 Context 候选，分别增加 10 题和 2 题；MBPP 在 G1 接受 Context
候选并增加 32 题。BFCL 在 G2、G4 接受 Context 候选，分别增加 19 题和 2 题，并在 G7 接受
一个增加 2 题的 Architecture 候选。三条轨迹的最终父代均高于各自 Baseline；持平和退化候选
没有进入后续父代。

### 2. 独立重复测评

每条轨迹从基础模型和空演化状态独立开始，不继承其他 run 的 Context、Tools、Model 或源码状态。
三个数据集均采用 `64 题 × 5 代 × 3 runs`，因此逐 run 展示而不使用合并分数：

| 数据集 | Run | Baseline | Final | 净增益 | 是否提升 |
| --- | ---: | ---: | ---: | ---: | --- |
| HumanEval | R1 | 50/64 | 57/64 | +7 | 是 |
| HumanEval | R2 | 50/64 | 58/64 | +8 | 是 |
| HumanEval | R3 | 51/64 | 58/64 | +7 | 是 |
| MBPP | R1 | 44/64 | 57/64 | +13 | 是 |
| MBPP | R2 | 45/64 | 56/64 | +11 | 是 |
| MBPP | R3 | 47/64 | 58/64 | +11 | 是 |
| BFCL | R1 | 22/64 | 28/64 | +6 | 是 |
| BFCL | R2 | 26/64 | 33/64 | +7 | 是 |
| BFCL | R3 | 27/64 | 30/64 | +3 | 是 |

九条独立轨迹均获得净提升。HumanEval 与 MBPP 的 30 代中，Context 被选择 14 次、Architecture
14 次、Model 2 次；BFCL 的 15 代中，Architecture 被选择 13 次、Context 2 次。结果证明自主
闭环能够重复获得收益，同时也表明自然层选择和贡献并不均衡。

### 3. 三数据集四层矩阵

矩阵实验使用 `--force-layer` 将候选限制在指定层，用于区分“该层能否进化”与“Decision 是否会
选择该层”。HumanEval、MBPP 的 Context、Tools、Architecture 使用 `32 题 × 3 代 × 2 runs`；
Model 使用复现更稳定的 `64 题 × 1 代 × 2 runs`。BFCL 使用全部 200 个任务。

| 数据集 | 层 | 协议 | R1 | R1 净增 | R2 | R2 净增 | 提升单元 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | Context | 32 题，3 代 | 25→27 | +2 | 22→28 | +6 | 2/2 |
| HumanEval | Tools | 32 题，3 代 | 24→27 | +3 | 22→28 | +6 | 2/2 |
| HumanEval | Model | 64 题，1 代 | 49→53 | +4 | 54→56 | +2 | 2/2 |
| HumanEval | Architecture（自然） | 32 题，3 代 | 25→25 | 0 | 26→26 | 0 | 0/2 |
| MBPP | Context | 32 题，3 代 | 18→27 | +9 | 23→28 | +5 | 2/2 |
| MBPP | Tools | 32 题，3 代 | 22→25 | +3 | 20→27 | +7 | 2/2 |
| MBPP | Model | 64 题，1 代 | 48→56 | +8 | 51→58 | +7 | 2/2 |
| MBPP | Architecture（自然） | 32 题，3 代 | 26→26 | 0 | 24→24 | 0 | 0/2 |

| 数据集 | 层 | 协议 | Baseline | Final / Candidate | 净增益 | 结果 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| BFCL | Context | 全量 200 题 | 95/200 | 97/200 | +2 | 接受 |
| BFCL | Tools | 全量 200 题 | 93/200 | 102/200 | +9 | 接受 |
| BFCL | Model | 全量 200 题 | 94/200 | 117/200 | +23 | 接受 |
| BFCL | Architecture（自然） | 全量 200 题 | 95/200 | 无可执行候选 | 0 | 拒绝 |

Context、Tools 和 Model 在 HumanEval、MBPP 上均达到 `2/2` 独立提升，并在 BFCL 全量任务上
分别获得 `+2`、`+9` 和 `+23`。三个数据集的自然任务均不能稳定暴露可定位的 Agent 控制流
缺陷，因此 Architecture 的结构修复能力由下一组受控实验测量。

### 4. 三数据集 Architecture 受控实验

Architecture Mutator 与可修改的任务 Agent 解耦，由候选不可修改的固定最小 MetaAgent 执行修复。
受控实验只注入两类通用控制流缺陷：第一批工具调用后提前终止，以及用工具结果覆盖原消息历史。

| 数据集 | 受控缺陷 | Run | 缺陷 Baseline | 修复后 | 净增益 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HumanEval | 历史覆盖 | R1 | 6/20 | 14/20 | +8 | 接受 |
| HumanEval | 历史覆盖 | R2 | 7/20 | 7/20 | 0 | `no_change` |
| MBPP | 提前终止 | R1 | 0/20 | 11/20 | +11 | 接受 |
| MBPP | 提前终止 | R2 | 0/20 | 14/20 | +14 | 接受 |
| MBPP | 提前终止 | R3 | 0/20 | 13/20 | +13 | 接受 |
| MBPP | 历史覆盖 | R1 | 3/20 | 12/20 | +9 | 接受 |
| MBPP | 历史覆盖 | R2 | 3/20 | 13/20 | +10 | 接受 |
| BFCL | 提前终止 | 全量 | 10/200 | 89/200 | +79 | 接受 |
| BFCL | 历史覆盖 | 全量 | 0/200 | 无可执行候选 | 0 | `no_change` |

HumanEval 与 MBPP 的重复 probe 合计达到 `6/7` 修复成功，其中 MBPP 提前终止为 `3/3`，两数据集
历史覆盖合计为 `3/4`。BFCL 提前终止获得 `+79`，历史覆盖未形成可执行候选。这说明固定
MetaAgent 能解除“待修 Agent 缺陷同时破坏修复者”的自指耦合，但 Architecture 的最小 patch
实现和自然失败定位仍未达到完全稳定。

### 结论边界

- 四个 Executor 均能产生真实候选，并具有独立状态、验证、拒绝和恢复语义。
- Context、Tools、Model 已获得跨数据集自然任务收益；Architecture 已获得可复现的受控结构修复证据。
- 三个数据集的独立重复测评达到 `9/9` 轨迹提升，自主组合收益具有初步可复现性。
- 现有证据不支持“四层在所有自然任务中均衡、持续贡献”的更强结论；Architecture 的自然收益和
  跨任务迁移仍是后续研究问题。

完整实验过程、消融和失败经验记录在仓库外研究报告中，不属于运行时依赖。

## 验证

```bash
.venv/bin/ruff check src tests eval
.venv/bin/python -m pytest -q
.venv/bin/python -m strataevo.evolution.cli --help
```
