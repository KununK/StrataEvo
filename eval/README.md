# Agent Evaluation

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
├── sessions/         # 每道题独立的 Tinyagent session
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
