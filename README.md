# StrataEvo

StrataEvo 是一个最小四层自进化 Agent 研究基线。它评测当前 Agent，从真实任务轨迹中诊断
问题，选择 Model、Context、Tools 或 Architecture 中的一层执行单项干预，并仅在任务分数
明确提升时保留候选。

当前 `main` 对应经过双数据集重复实验验证的阶段性基线。它已经证明自主组合、Context 和 Tools
能够产生可复现提升；Model 与 Architecture 仍是明确的研究缺口，不应把当前版本解释为四层能力
均已完成。

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
  mutator.py                      Architecture 自修改 Agent
  layers/                         四层执行器
  runtime/                        Git、评测、结构化响应和自修改工作区
eval/                             固定 HumanEval/MBPP 环境
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

Model 层是在当前任务上进行 verifier-guided 测试时适应，不应被解释为 held-out 泛化结果。

## 阶段性实验结果

自主多 Run 使用 HumanEval、MBPP 各 3 次独立运行，每次 10 代、64 题，不强制层选择：

| 数据集 | Baseline 汇总 | Final 汇总 | 净增 | 提升 runs |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 145/192 | 177/192 | +32 | 3/3 |
| MBPP | 156/192 | 176/192 | +20 | 3/3 |
| 合计 | 301/384 | 353/384 | +52 | 6/6 |

全量五代自主进化：

| 数据集 | Baseline | Final | 净增 |
| --- | ---: | ---: | ---: |
| HumanEval | 136/164 | 158/164 | +22 |
| MBPP | 200/257 | 233/257 | +33 |

四层独立矩阵使用两个数据集、每层各 2 次独立运行、每次 3 代和 32 题：

| 层 | HumanEval | MBPP | 提升单元 |
| --- | --- | --- | ---: |
| Context | 25→27；22→28 | 18→27；23→28 | 4/4 |
| Tools | 24→27；22→28 | 22→25；20→27 | 4/4 |
| Model | 23→23；27→27 | 20→20；20→26 | 1/4 |
| Architecture（自然任务） | 25→25；26→26 | 26→26；24→24 | 0/4 |

受控结构实验进一步确认：当前版本在“第一批工具调用后提前终止”上 0/3 修复，在“工具结果覆盖
历史消息”上 0/4 修复。原因是 Architecture Mutator 仍复用待修任务 Agent；当 Agent 循环本身
损坏时，修复器也无法完成读取、编辑和验证。因此当前最可信结论是：整体自主进化稳定，Context
与 Tools 独立能力可复现；Model 泛化和 Architecture 自修复尚未完成。

完整实验过程、后续消融与失败经验记录在仓库外研究报告中，不属于运行时依赖。

## 验证

```bash
.venv/bin/ruff check src tests eval
.venv/bin/python -m pytest -q
.venv/bin/python -m strataevo.evolution.cli --help
```
