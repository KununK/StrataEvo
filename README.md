# StrataEvo

StrataEvo 是一个面向自进化 Agent 和下游任务评测的研究项目。当前以 Tinyagent 作为 Agent 运行时，以 vLLM 作为本地 OpenAI-compatible 模型服务。

一个 `pyproject.toml` 管理 Agent、推理、训练和评测所需的统一 Python 环境。

## 项目结构

```text
StrataEvo/
├── src/tinyagent/     # Agent 运行时
├── tests/             # Tinyagent 回归测试
├── vllm_serve.sh      # vLLM 服务启动脚本
├── pyproject.toml     # 统一项目与依赖配置
├── uv.lock            # uv 依赖锁文件
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

后续自进化层可以作为新包加入 `src/strataevo/`，并通过 Tinyagent 的 `Model`、`Tool`、`SessionStore` 和 `EventBus` 接口组合，而无需修改基础 Agent 循环。

## 自进化

StrataEvo 现在支持代码库级递归自进化：系统先将候选代码、任务结果和工具轨迹整理为
结构化 Evidence，再由模型在 Model、Context、Tools 和 Architecture 四层中诊断主要
演化对象。Evolution Planner 从诊断中选择一项可检验的干预，再由 Tinyagent 运行实例承担的
自修改角色执行。自修改 Agent 可以在同一代内调用只读的 `evaluate_candidate`，根据固定测试、
HumanEval 结果和 Evidence 连续修正候选，默认每代最多评测 5 次，整代共享 200 个
Mutator step。控制器最终恢复本代 pass@1 最高的已评测候选：优于父代则提交，未提升则自动
回滚。下一代由新的 Python 进程加载修改后的源码。每个完成代的诊断、计划、各次候选修改、
预期指标对照及接受或拒绝结果会写入
`evolution/runs/<run_name>/evolution_memory.jsonl`，供后续代诊断和修改时参考。

Planner 会看到实际可修改源码清单，执行器支持精确文本替换和按行替换，并将每代结果区分为
无修改、固定验证失败、基准拒绝或接受，避免把执行失败误认为演化方向无效。

```bash
strataevo \
  --run-name humaneval-dev \
  --branch evo_test_inter \
  --generations 1 \
  --eval-limit 5
```

这里修改的是 Agent 自身源码，不是 HumanEval 的 `solution.py`。设计边界、晋级条件、
断点续跑和输出结构见 [`EVOLUTION.md`](EVOLUTION.md)。

## 工具与安全边界

Tinyagent 内置 `list_files`、`read_file`、`search_files`、`write_file`、`replace_text` 和 `run_shell`。

写文件、修改文件和执行 Shell 默认需要用户批准。文件工具拒绝访问 Workspace 之外的路径。`run_shell` 不是操作系统沙箱；批准后的命令拥有 Agent 进程的系统权限。

## 测试

运行 Tinyagent 标准库测试：

```bash
uv run python -m unittest discover -s tests -v
```

也可以使用项目已有的 pytest：

```bash
uv run pytest
```

测试使用确定性的 `ScriptedModel`，不访问真实模型。

## Agent 评测

HumanEval Agent 评测位于 `eval/`。先启动 vLLM，再运行 smoke 测试：

```bash
LIMIT=5 FORCE_RERUN=1 ./eval/run_humaneval.sh
```

完整说明见 [`eval/README.md`](eval/README.md)。

StrataEvo 使用 MIT License。
