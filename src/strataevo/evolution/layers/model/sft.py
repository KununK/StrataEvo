"""Small PEFT trainer used by model evolution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    config = json.loads(Path(parse_args().config).read_text(encoding="utf-8"))
    train(config)
    return 0


def train(config: dict[str, Any]) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA evolution requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    parent = config.get("parent_adapter_path")
    if parent:
        model = PeftModel.from_pretrained(model, parent, is_trainable=True)
    else:
        rank = int(config["lora_rank"])
        model = get_peft_model(
            model,
            LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.05,
                target_modules=list(LORA_TARGET_MODULES),
                task_type="CAUSAL_LM",
            ),
        )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    examples = _encode_examples(
        tokenizer,
        Path(config["data_path"]),
        int(config["max_length"]),
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "event": "training_start",
                "device": torch.cuda.get_device_name(0),
                "examples": len(examples),
                "epochs": int(config["epochs"]),
                "steps_per_epoch": len(examples),
                "total_steps": len(examples) * int(config["epochs"]),
                "max_length": int(config["max_length"]),
                "learning_rate": float(config["learning_rate"]),
                "lora_rank": int(config["lora_rank"]),
                "lora_targets": list(LORA_TARGET_MODULES),
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    loader = DataLoader(
        examples,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda rows: _collate(rows, tokenizer.pad_token_id, torch),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
    )
    model.train()
    losses = []
    epochs = int(config["epochs"])
    total_steps = len(loader) * epochs
    step = 0
    for epoch in range(1, epochs + 1):
        for batch in loader:
            step += 1
            batch = {name: value.to(model.device) for name, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = float(loss.detach())
            losses.append(value)
            print(
                json.dumps(
                    {
                        "event": "training_step",
                        "epoch": epoch,
                        "epochs": epochs,
                        "step": step,
                        "total_steps": total_steps,
                        "loss": value,
                    }
                ),
                flush=True,
            )
    output_dir = Path(config["output_dir"])
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "steps": len(losses),
                "epochs": epochs,
                "examples": len(examples),
                "mean_loss": sum(losses) / len(losses),
                "final_loss": losses[-1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "training_complete",
                "steps": len(losses),
                "epochs": epochs,
                "mean_loss": sum(losses) / len(losses),
                "final_loss": losses[-1],
                "output_dir": str(output_dir),
            }
        ),
        flush=True,
    )


def _encode_examples(tokenizer: Any, path: Path, max_length: int) -> list[dict[str, list[int]]]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        messages = _normalize_messages(json.loads(line)["messages"])
        input_ids = _chat_ids(
            tokenizer,
            messages,
        )
        labels = [-100] * len(input_ids)
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            start = len(_chat_ids(tokenizer, messages[:index]))
            end = len(_chat_ids(tokenizer, messages[: index + 1]))
            labels[start:end] = input_ids[start:end]
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        if any(label != -100 for label in labels):
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )
    if not examples:
        raise ValueError("no trainable assistant messages found in verified trajectories")
    return examples


def _chat_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    if not messages:
        return []
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return list(input_ids)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message["role"],
            "content": message.get("content") or "",
        }
        calls = message.get("tool_calls")
        if calls:
            item["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call.get("arguments", {}),
                    },
                }
                for call in calls
            ]
        if message.get("tool_call_id"):
            item["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            item["name"] = message["name"]
        normalized.append(item)
    return normalized


def _collate(rows: list[dict[str, list[int]]], pad_token_id: int, torch: Any) -> dict[str, Any]:
    width = max(len(row["input_ids"]) for row in rows)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in rows:
        padding = width - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
        batch["labels"].append(row["labels"] + [-100] * padding)
    return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


if __name__ == "__main__":
    raise SystemExit(main())
