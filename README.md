# StrataEvo

StrataEvo 是一个面向自进化 Agent 和下游任务评测的研究项目。当前以 Tinyagent 作为 Agent 运行时，以 vLLM 作为本地 OpenAI-compatible 模型服务。

一个 `pyproject.toml` 管理 Agent、推理、训练和评测所需的统一 Python 环境。

## 项目结构

```text
StrataEvo/
├── src/tinyagent/              # 通用 Agent 运行时
├── src/strataevo/evolution/    # 诊断、计划、修改、评测和版本控制
├── eval/                       # 共享 coding-agent harness 与 benchmark 适配器
├── tests/                      # 运行时、演化控制器和 benchmark 回归测试
├── evolution/runs/             # 实验记录，已被 Git 忽略
├── EVOLUTION.md                # 自进化控制流与记录格式
├── vllm_serve.sh               # vLLM 服务启动脚本
├── pyproject.toml              # 统一项目与依赖配置
└── README.md
```

Tinyagent 当前包括：

- OpenAI-compatible 模型接口；
- 结构化工具调用与自动 JSON Schema；
- 模型、工具、模型的多步执行循环；
- 文件、搜索、修改和 Shell 工具；
- 副作用审批与 Workspace 路径边界；
- 会话持久化、事件、取消、步数限制和上下文控制；
- 单次任务与交互式 CLI。

## 配置环境

要求 Python 3.12：

```bash
cd /data/vlm/jlk/Agent/StrataEvo
source /data/vlm/jlk/switch-cuda.sh 12.9
uv sync --locked
```

也可以显式激活环境：

```bash
source .venv/bin/activate
```

## 启动 vLLM

脚本默认启动 `Qwen/Qwen3-Coder-30B-A3B-Instruct`，并启用 Qwen3-Coder 工具调用解析：

```bash
cd /data/vlm/jlk/Agent/StrataEvo
source /data/vlm/jlk/switch-cuda.sh 12.9
source .venv/bin/activate
./vllm_serve.sh
```

默认配置：

| 环境变量 | 默认值 |
| --- | --- |
| `MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| `HOST` | `127.0.0.1` |
| `PORT` | `8000` |
| `MAX_MODEL_LEN` | `auto` |
| `GPU_MEMORY_UTILIZATION` | `0.8` |
| `TOOL_CALL_PARSER` | `qwen3_coder` |
| `MAX_LORA_RANK` | `8` |

所有配置都可以覆盖，额外参数会直接传给 `vllm serve`：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
GPU_MEMORY_UTILIZATION=0.9 \
./vllm_serve.sh --tensor-parallel-size 2
```

## 启动 Agent

在另一个终端中使用同一个环境：

```bash
cd /data/vlm/jlk/Agent/StrataEvo
source .venv/bin/activate
export OPENAI_BASE_URL="http://localhost:8000/v1"
tinyagent
```

不激活环境时使用：

```bash
uv run tinyagent
```

提供任务文本时执行一次后退出：

```bash
tinyagent \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --workspace . \
  "检查当前项目"
```

不提供任务文本时进入交互模式。

## Tinyagent CLI

```text
tinyagent [--model MODEL]
          [--base-url URL]
          [--workspace PATH]
          [--session ID]
          [--max-steps N]
          [--yes]
          [--verbose]
          [prompt ...]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `TINYAGENT_MODEL` 或 `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 模型名称 |
| `--base-url` | `OPENAI_BASE_URL` 或 `http://localhost:8000/v1` | API 地址 |
| `--workspace` | 当前目录 | 工具工作目录 |
| `--session` | `default` | 会话名称，空字符串关闭持久化 |
| `--max-steps` | `20` | 单次任务最大模型调用轮数 |
| `--yes` | 关闭 | 自动批准写入和 Shell |
| `--verbose` | 关闭 | 显示模型和工具事件 |

## Python API

```python
from tinyagent import (
    Agent,
    OpenAICompatibleModel,
    ToolRegistry,
    Workspace,
    terminal_approval,
)

model = OpenAICompatibleModel()

agent = Agent(
    model=model,
    tools=ToolRegistry(Workspace(".").tools()),
    approval=terminal_approval,
)

result = agent.run("检查当前项目")
print(result.output)
```

自定义工具：

```python
from tinyagent import tool

@tool
def evaluate_candidate(candidate: str, task: str) -> dict:
    """评测一个候选 Agent。"""
    return {"candidate": candidate, "task": task, "score": 0.0}
```

## 自进化

StrataEvo 的每一代按以下顺序运行：

```text
benchmark -> Evidence -> 四层 Diagnosis -> Evolution Plan
          -> 对应层候选生成与评测 -> 新鲜父子复测
          -> 接受或回滚 -> Evolution Memory
```

四层是 Model、Context、Tools 和 Architecture。Evidence 只保存任务结果、候选文件状态、
有序工具调用及结果和 session 路径等可观察事实，不用规则替模型做分层归因。Planner 每代
选择一项能够提升 `task_score` 的假设。自修改 Agent 默认共享 200 个 step、5 个 refinement
round 和 5 次候选 benchmark 配额。

Evaluation Contract 声明当前 benchmark 能直接观察哪些代码。当前 HumanEval 和 MBPP 都
直接评测 `src/tinyagent/`；只影响以后自修改行为的代码不能借用本轮任务分数晋级。探索阶段
只负责选择候选，随后会重新评测父代与候选。确认候选必须同时超过进入本代时记录的父代和
新鲜父代才会提交，否则恢复父代。步数、Token 和时间继续记录为研究指标，但不混入晋级
分数。每代结束会自动更新运行
目录下的 `summary.json` 和 `summary.md`；被评测反证的假设及 counterevidence 会进入下一代
Planner。可证明只有注释、模块 docstring、格式或结构变化的源码候选会作为
`semantic_noop` 提前恢复；可能影响工具描述的函数 docstring 仍按行为变化处理。

```bash
strataevo \
  --run-name humaneval-dev \
  --branch evo_test_inter \
  --benchmark humaneval \
  --generations 1 \
  --eval-limit 5
```

`--benchmark` 当前支持 `humaneval` 和 `mbpp`。这里修改的是 Agent 自身源码，不是评测任务
中的 `solution.py`。不传 `--eval-limit` 时使用所选 benchmark 的完整 split；smoke 实验应
显式指定数量。设计边界、晋级条件、断点续跑和输出结构见
[`EVOLUTION.md`](EVOLUTION.md)。

完整 MBPP 演化示例：

```bash
strataevo \
  --run-name mbpp-full-1 \
  --branch evo_test_inter \
  --benchmark mbpp \
  --generations 5 \
  --eval-workers 10
```

### Context 层进化

Context Evolution 将通用提示词增量保存为独立 `ContextProfile`，而不是修改 Python 源码。
候选包含 `system_prompt_addendum` 和 `task_prompt_addendum`，会在 benchmark 运行时追加到
固定基础提示词。候选经过 screening 和新鲜父子复测；接受后写入 `state.json` 和
`evolution_memory.jsonl`，拒绝则恢复父代 context。它不会产生 Git evolution commit。
screening 未提升时，同一代会将候选、分数差和任务变化反馈给 Context Evolver 继续生成；
最多使用 `--max-eval-attempts` 个候选，内容重复的候选不会再次运行 benchmark。首个超过父代
的候选再进入一次新鲜父子复测，确认分数必须同时超过已记录父代和新鲜父代才能接受。

最小链路测试可以使用：

```bash
strataevo \
  --run-name humaneval-context-smoke \
  --branch evo_context \
  --benchmark humaneval \
  --generations 1 \
  --eval-offset 110 \
  --eval-limit 20 \
  --eval-workers 10 \
  --force-layer context
```

`--force-layer context` 仅用于验证执行链路。正式实验应省略，让 Diagnosis 和 Plan 自主选择。
ContextProfile 只允许通用指令，不保存任务 ID、具体答案、hidden tests 或 repair 代码。

### Tools 层进化

最小 Tool Evolution 将模型可见的工具说明增量保存为独立 `ToolProfile`。候选只能为已有
工具追加通用的使用、顺序、验证或失败恢复说明，不改变工具名称、参数 schema、Python
实现、审批要求或执行权限。候选与 ContextProfile 使用相同的 screening、新鲜父子复测和
回滚事务；接受后写入 `state.json.current_tool_profile` 和 Evolution Memory，且不产生 Git
commit。未提升的 screening 结果同样会在本代反馈给 Tool Evolver，直到产生可晋级候选或
耗尽 `--max-eval-attempts`。

```bash
strataevo \
  --run-name humaneval-tools-smoke \
  --branch evo_context \
  --benchmark humaneval \
  --generations 1 \
  --eval-offset 110 \
  --eval-limit 20 \
  --eval-workers 10 \
  --force-layer tools
```

`--force-layer tools` 仅用于链路测试。正式实验省略该参数，由 Diagnosis 和 Plan 自主选择。

### Model 层进化

第一版 Model Evolution 会针对当前 benchmark 的失败任务生成修复轨迹，只将通过同一个
verifier 的修复结果用于 test-time LoRA SFT，并混入父代成功轨迹作为 replay。训练任务与
随后评测的任务不分离，因此这是测试集参与的适应实验，不能报告为 held-out 泛化结果。
LoRA 训练默认在物理 GPU 1 运行，vLLM 推理仍在 GPU 0。

`vllm_serve.sh` 已默认启用 LoRA 和运行时 adapter 更新。重启服务后，在演化命令中显式开启：

```bash
strataevo \
  --run-name mbpp-model-smoke \
  --branch evo_reliable_sft \
  --benchmark mbpp \
  --generations 2 \
  --eval-offset 60 \
  --eval-limit 30 \
  --eval-workers 10 \
  --enable-model-evolution \
  --force-layer model \
  --sft-device 1
```

`--force-layer model` 用于单独验证模型进化链路；正式自主实验应省略该参数，由 Diagnosis
与 Plan 选择演化层。每个失败任务默认生成 2 次修复，训练集优先放入成功修复，再用父代
成功轨迹填充至最多 32 条。LoRA 默认训练 1 epoch、rank 8；候选仍需经过筛选和新鲜父子
复测。源码候选继续由 Git
提交或回滚，模型候选则保存在对应 generation 的 `model/adapter/`，并在 `state.json`
和 `evolution_memory.jsonl` 中记录父代、修复成功任务、仍失败任务、评测修复任务及退化
任务。

## 工具与安全边界

Tinyagent 内置 `list_files`、`read_file`、`search_files`、`write_file`、`replace_text` 和 `run_shell`。

写文件、修改文件和执行 Shell 默认需要用户批准。文件工具拒绝访问 Workspace 之外的路径。`run_shell` 不是操作系统沙箱；批准后的命令拥有 Agent 进程的系统权限。

## 测试

运行全部回归测试：

```bash
uv run python -m unittest discover -s tests -v
```

也可以使用项目已有的 pytest：

```bash
uv run pytest
```

测试使用确定性的 `ScriptedModel`，不访问真实模型。固定验证还会运行 Ruff 和自进化 CLI
启动检查。

## Agent 评测

HumanEval 和 MBPP Agent 评测位于 `eval/`。先启动 vLLM，再运行 smoke 测试：

```bash
LIMIT=5 FORCE_RERUN=1 ./eval/run_humaneval.sh
LIMIT=5 FORCE_RERUN=1 ./eval/run_mbpp.sh
```

完整说明见 [`eval/README.md`](eval/README.md)。

StrataEvo 使用 MIT License。
