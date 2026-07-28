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
          -> 代内多轮修改与候选评测 -> 新鲜父子复测
          -> Git 提交或回滚 -> Evolution Memory
```

四层是 Model、Context、Tools 和 Architecture。Evidence 只保存任务结果、候选文件状态、
有序工具调用及结果和 session 路径等可观察事实，不用规则替模型做分层归因。Planner 每代
选择一项能够提升 `task_score` 的假设。自修改 Agent 默认共享 200 个 step、5 个 refinement
round 和 5 次候选 benchmark 配额。

Evaluation Contract 声明当前 benchmark 能直接观察哪些代码。当前 HumanEval 和 MBPP 都
直接评测 `src/tinyagent/`；只影响以后自修改行为的代码不能借用本轮任务分数晋级。探索阶段
只负责选择候选，随后会重新评测父代与候选。新鲜候选的 pass@1 严格更高才会提交，否则恢复
父代。步数、Token 和时间继续记录为研究指标，但不混入晋级分数。

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

### Model 层进化

第一版 Model Evolution 使用当前 benchmark 中 verifier 已确认通过的 Agent 轨迹进行
test-time LoRA SFT。训练任务与随后评测的任务不分离，因此这是测试集参与的适应实验，
不能报告为 held-out 泛化结果。LoRA 训练默认在物理 GPU 1 运行，vLLM 推理仍在 GPU 0。

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
  --sft-device 1
```

只有 Diagnosis 与 Plan 选择 `model` 层时才训练。默认每次最多取 32 条通过轨迹，进行
20 step、rank 8 的 LoRA SFT；候选仍需经过筛选和新鲜父子复测。源码候选继续由 Git
提交或回滚，模型候选则保存在对应 generation 的 `model/adapter/`，并在 `state.json`
和 `evolution_memory.jsonl` 中记录其父代和路径。

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
