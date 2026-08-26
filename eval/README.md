# Agent Evaluation

HumanEval 和 MBPP 共用同一套 coding-agent harness：每道题使用独立 Workspace 和
session，Agent 必须读取 `task.py` 并生成 `solution.py`。两个 benchmark 输出相同结构的
任务轨迹、判题结果、汇总和 Evidence，数据集差异由各自的 loader 和判题器处理。

## HumanEval

HumanEval 使用 `openai/openai_humaneval` 的 164 个 Python 任务。每道题运行一个全新的 Tinyagent：

1. 在临时 Workspace 中写入只读任务描述 `task.py`；
2. Agent 使用文件和 Shell 工具完成 `solution.py`；
3. 保存 Agent 消息轨迹和候选源码；
4. 在独立、限时的 Python 子进程中运行 HumanEval 隐藏测试；
5. 统计 pass@1、错误类型、步数和执行时间。

评测以题目为单位串行执行：一道题的 Agent 工具循环完整结束后，立即运行该题的
隐藏测试，然后再处理下一题。终端使用进度条展示完成比例、速度和预计剩余时间。

先启动 vLLM，然后运行 5 题 smoke 测试：

```bash
LIMIT=5 FORCE_RERUN=1 ./eval/run_humaneval.sh
```

完整评测：

```bash
./eval/run_humaneval.sh
```

覆盖参数：

```bash
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
BASE_URL=http://localhost:8000/v1 \
RUN_NAME=qwen3-coder-agent-v1 \
WORKERS=4 \
./eval/run_humaneval.sh --max-steps 16 --test-timeout 15
```

默认并发执行 4 道互相独立的题，以便 vLLM 批处理模型请求。显存或服务吞吐不足时可以
设置 `WORKERS=1` 改为串行；有更多吞吐余量时可以尝试 `WORKERS=8`。并发只改变运行速度，
不会共享不同题目的 Workspace 或 session。

输出结构：

```text
eval/outputs/humaneval/<run_name>/
├── candidates/       # 每道题的 solution.py
├── config.json       # 本次评测的完整参数
├── generations.jsonl # Agent 输出、工具轨迹、token 和步数
├── results.jsonl     # 每题判题结果
├── evidence.json    # 自进化使用的结构化任务证据
└── summary.json      # pass@1 和状态统计
```

同时保存终端日志：

```text
eval/logs/humaneval/<run_name>.log
```

每道题是一个独立 session，但不是单次模型调用。一个 session 内部会进行多轮
“模型 -> 工具 -> 模型”，直到模型完成任务或达到 `--max-steps`。题目之间不共享上下文。

默认支持断点续跑。设置 `FORCE_RERUN=1` 会删除该次运行的旧输出。
断点续跑会追加文本日志；`FORCE_RERUN=1` 则会重新写入日志。

自进化评测会在结束后自动生成 `evidence.json`。对于已有的独立评测结果，可以离线生成，
不需要重新调用模型：

```bash
python -m strataevo.evolution.evidence \
  eval/outputs/humaneval/qwen3-coder-30b-agent
```

也可以使用本地 JSON/JSONL 数据验证流程：

```bash
python -m eval.humaneval.run \
  --dataset-file path/to/tasks.jsonl \
  --limit 1 \
  --no-resume
```

候选代码和 Agent 的 Shell 工具都会执行模型生成的代码。当前实现使用临时目录、`python -I` 和超时，但不是安全沙箱；只应评测可信模型，严格隔离需要容器或虚拟机。

## MBPP

MBPP 默认使用 `google-research-datasets/mbpp` 的 `sanitized/test`。自然语言描述、
目标函数名和公开测试会以 Python 数据写入只读的 `task.py`。判题器组合数据集 imports、
setup、候选源码和测试断言，然后在独立的 `python -I` 子进程中执行。

5 题 smoke 测试：

```bash
LIMIT=5 FORCE_RERUN=1 ./eval/run_mbpp.sh
```

完整独立评测：

```bash
./eval/run_mbpp.sh
```

覆盖参数：

```bash
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
BASE_URL=http://localhost:8000/v1 \
RUN_NAME=qwen3-coder-mbpp-v1 \
WORKERS=8 \
./eval/run_mbpp.sh --max-steps 12 --test-timeout 15
```

本实现参考 CodeSensiQuant 的 MBPP 数据字段和测试程序构造，但不复用其 Transformers
直接生成流程，也不加入其三样例提示。这里让每道题经过 Tinyagent 工具循环，以保持不同
benchmark 的 Agent 行为、Evidence 和自进化反馈结构一致。

## BFCL V4 multi-turn

BFCL 使用官方 `multi_turn_base` 数据、状态后端和 checker，直接评估 Tinyagent 的多轮
工具调用轨迹，不生成 `solution.py`。第三方源码固定在被 Git 忽略的 `eval/vendor/bfcl`：

```bash
./eval/setup_bfcl.sh
```

先运行小规模 baseline：

```bash
LIMIT=32 WORKERS=4 FORCE_RERUN=1 \
  RUN_NAME=bfcl-v4-multiturn-base-32 \
  ./eval/run_bfcl.sh
```

运行四层自主进化：

```bash
strataevo \
  --run-name bfcl-v4-multiturn-evolution-32 \
  --branch evo_bfcl_v4 \
  --benchmark bfcl \
  --generations 3 \
  --eval-limit 32 \
  --eval-workers 4 \
  --enable-model-evolution
```

也可以通过 `BFCL_ROOT` 指向已有的 Gorilla checkout。首版只支持无需 SerpAPI 的
`multi_turn_base`。Model 层重新执行失败轨迹，并且只使用官方 checker 通过的轨迹训练
LoRA；web-search 和 memory 类别尚未接入。
