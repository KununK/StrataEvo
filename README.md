# StrataEvo

StrataEvo 是一个最小四层自进化 Agent 研究基线。它评测当前 Agent，从真实任务轨迹中诊断
问题，选择 Model、Context、Tools 或 Architecture 中的一层执行单项干预，并仅在任务分数
明确提升时保留候选。

当前 `main` 对应经过 HumanEval、MBPP 和 BFCL 验证的阶段性四层基线。Context、Tools、Model、
Architecture 均已产生真实、可验证的独立进化证据，自主层选择在代码生成和多轮工具任务上均能
提升最终父代。该结论证明的是最小闭环具备初步四层自进化能力，不代表四层会在自然任务中均衡
贡献，也不应解释为 held-out 泛化结果。

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

正式自主多 Run 使用 HumanEval、MBPP 各 3 次独立运行，每次 5 代、64 题，不强制层选择；
Model 使用 4096-token SFT 长度：

| 数据集 | Baseline 汇总 | Final 汇总 | 净增 | 提升 runs |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 151/192 | 173/192 | +22 | 3/3 |
| MBPP | 136/192 | 171/192 | +35 | 3/3 |
| 合计 | 287/384 | 344/384 | +57 | 6/6 |

完整基准测评：

| 数据集 | 任务数 | 代数 | Baseline | Final | 净增 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 164 | 5 | 144/164 | 156/164 | +12 |
| MBPP | 257 | 5 | 200/257 | 232/257 | +32 |
| BFCL | 200 | 10 | 100/200 | 123/200 | +23 |

BFCL 的主要收益来自 G2、G4 的 Context 候选，分别贡献 `+19、+2`。G7 Architecture 名义
贡献 `+2`，但任务交换明显且修改与收益的因果关系较弱，因此只保留为协议观测，不作为自然
Architecture 稳定进化证据。

四层独立矩阵使用两个数据集、每层各 2 次独立运行、每次 3 代和 32 题。Context、Tools 和
Model Executor 与矩阵父代保持一致；Architecture 在当前版本由后述受控实验重新验证：

| 层 | HumanEval | MBPP | 提升单元 |
| --- | --- | --- | ---: |
| Context | 25→27；22→28 | 18→27；23→28 | 4/4 |
| Tools | 24→27；22→28 | 22→25；20→27 | 4/4 |
| Model | 23→23；27→27 | 20→20；20→26 | 1/4 |
| Architecture（改动前自然任务） | 25→25；26→26 | 26→26；24→24 | 0/4 |

BFCL 全量四层消融：

| 层 | Baseline | Candidate / Final | 净增 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Context | 95/200 | 97/200 | +2 | accepted |
| Tools | 93/200 | 102/200 | +9 | accepted |
| Model R1 | 96/200 | 116/200 | +20 | accepted |
| Model R2 | 94/200 | 117/200 | +23 | accepted |
| Architecture（自然任务） | 95/200 | 无可执行候选 | 0 | rejected / no-change |

Architecture Mutator 与可修改的任务 Agent 解耦，使用控制侧固定、候选不可修改的最小
MetaAgent。受控结构实验结果如下：

| 受控缺陷 | 改进前 | 固定 MetaAgent | 结果 |
| --- | ---: | ---: | --- |
| 第一批工具调用后提前终止 | 0/3 修复 | 3/3 修复 | 恢复多轮工具循环 |
| 工具结果覆盖历史消息 | 0/4 修复 | 3/4 修复 | 恢复工具结果追加语义 |
| BFCL 提前终止，全量 200 题 | 10/200 | 89/200 | accepted，直接删除错误提前返回 |
| BFCL 历史覆盖，全量 200 题 | 0/200 | 无可执行候选 | no-change，正确方向夹带过度修改 |

重复受控实验合计由改进前 `0/7` 提高到 `6/7`。这证明固定 MetaAgent 能解除待修 Agent 缺陷
同时破坏修复者的自指耦合；BFCL 一成一败则说明 Architecture 已具备真实修复能力，但最小 patch
实现和自然失败定位仍不稳定。

Model 恢复 4096-token SFT 后，HumanEval 64 题两次分别提升 4、2 题，HumanEval/MBPP 全量
强制层实验分别提升 7、33 题；自主组合中也产生一次真实 Model 晋级。

正式多 Run 的 30 代中，Context 选择 14 次并贡献主要收益，Architecture 选择 14 次，Model
选择 2 次，Tools 未被选择。同父代离线反事实显示 Decision 偶尔会漏掉收益略高的 Model，但未
形成系统性错路由证据；项目因此保留单层 Decision，没有加入层配额、固定轮转或默认双候选评测。
当前最可信结论是：四层执行器均具备独立有效证据，整体自主进化可复现；自然层贡献仍不均衡，
Architecture 的自然收益和跨任务迁移仍是后续研究问题。

完整实验过程、后续消融与失败经验记录在仓库外研究报告中，不属于运行时依赖。

## 验证

```bash
.venv/bin/ruff check src tests eval
.venv/bin/python -m pytest -q
.venv/bin/python -m strataevo.evolution.cli --help
```
