"""Train one factorial Qwen3-1.7B LoRA cell with assistant-only loss.

This intentionally uses a small explicit training loop so the MPS substitution is visible
without CUDA-only optimizer dependencies.

Confirmatory example:
  .venv/bin/python -u scripts/train_factorial_lora.py \
    --cell gold_deny --seed 42 --device mps

Pipeline-only smoke:
  .venv/bin/python -u scripts/train_factorial_lora.py \
    --cell gold_deny --seed 20260910 --device mps \
    --limit 2 --max-length 256 --gradient-accumulation 1 --max-steps 1 \
    --output-dir factorial/runs/smoke_gold_deny
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("gold_deny", "gold_confirm", "leaf_deny", "leaf_confirm")
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_subsequence(sequence: list[int], pattern: list[int], start: int) -> int:
    stop = len(sequence) - len(pattern) + 1
    for index in range(start, stop):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def tokenize_with_assistant_labels(
    tokenizer,
    messages: list[dict],
    max_length: int,
    loss_scope: str = "all_assistant",
):
    """Render Qwen chat and supervise assistant content plus each assistant EOT token."""
    if loss_scope not in {"all_assistant", "final_assistant"}:
        raise ValueError(f"unknown loss_scope={loss_scope!r}")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    labels = [-100] * len(ids)
    assistant_header = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    cursor = 0
    spans = []
    while True:
        header_start = find_subsequence(ids, assistant_header, cursor)
        if header_start < 0:
            break
        content_start = header_start + len(assistant_header)
        try:
            content_end = ids.index(end_id, content_start)
        except ValueError:
            break
        spans.append((content_start, content_end + 1))
        cursor = content_end + 1
    active_spans = spans if loss_scope == "all_assistant" else spans[-1:]
    for content_start, content_stop in active_spans:
        labels[content_start:content_stop] = ids[content_start:content_stop]
    if not spans or all(label == -100 for label in labels):
        raise RuntimeError("failed to identify assistant tokens in rendered conversation")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "assistant_spans": len(spans),
        "supervised_assistant_spans": len(active_spans),
    }


def tensorize(example: dict, device: str) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
        for key, value in example.items()
        if key in {"input_ids", "attention_mask", "labels"}
    }


def train(args) -> None:
    if (args.confirmatory or args.seed in {42, 314159}) and (
        args.limit or args.max_steps > 0
    ):
        raise ValueError(
            "confirmatory seeds 42/314159 cannot be used with --limit or --max-steps"
        )
    set_seed(args.seed)
    random.seed(args.seed)

    data_path = (
        Path(args.data_path)
        if args.data_path
        else ROOT / "factorial" / "data" / f"{args.cell}.jsonl"
    )
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist; run scripts/build_factorial_data.py first"
        )
    rows = read_jsonl(data_path)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        args.base, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [
        tokenize_with_assistant_labels(
            tokenizer,
            row["messages"],
            args.max_length,
            row.get("loss_scope", "all_assistant"),
        )
        for row in rows
    ]
    if any(len(example["input_ids"]) >= args.max_length for example in encoded):
        print("warning: at least one conversation reached max_length", flush=True)
    supervised_tokens = sum(
        sum(label != -100 for label in example["labels"]) for example in encoded
    )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"trainable parameters: {trainable_count:,} / {total_count:,} "
        f"({100 * trainable_count / total_count:.3f}%)",
        flush=True,
    )

    groups_per_epoch = math.ceil(len(encoded) / args.gradient_accumulation)
    planned_steps = groups_per_epoch * args.epochs
    total_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "factorial" / "runs" / f"{args.cell}_seed{args.seed}"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    metadata = {
        "status": "running",
        "preregistration": args.preregistration,
        "cell": args.cell,
        "seed": args.seed,
        "confirmatory": args.confirmatory or args.seed in {42, 314159},
        "base": args.base,
        "base_revision": getattr(model.config, "_commit_hash", None),
        "data_path": str(data_path.relative_to(ROOT)),
        "data_sha256": sha256(data_path),
        "examples": len(encoded),
        "supervised_tokens": supervised_tokens,
        "max_observed_length": max(len(example["input_ids"]) for example in encoded),
        "training": {
            "epochs": args.epochs,
            "effective_batch_size": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "linear",
            "warmup_steps": 0,
            "max_grad_norm": args.max_grad_norm,
            "max_length": args.max_length,
            "optimizer": "torch.optim.AdamW (MPS substitution)",
            "dtype": args.dtype,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_steps": args.max_steps,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "targets": TARGET_MODULES,
        },
        "environment": {
            "torch": torch.__version__,
            "device": args.device,
            "mps_available": torch.backends.mps.is_available(),
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    update_step = 0
    start_time = time.monotonic()
    with log_path.open("w") as log_handle:
        for epoch in range(args.epochs):
            order = list(range(len(encoded)))
            random.Random(args.seed + epoch).shuffle(order)
            for group_start in range(0, len(order), args.gradient_accumulation):
                group = order[group_start : group_start + args.gradient_accumulation]
                optimizer.zero_grad(set_to_none=True)
                token_counts = [
                    sum(label != -100 for label in encoded[index]["labels"])
                    for index in group
                ]
                group_tokens = sum(token_counts)
                group_loss = 0.0
                for index in group:
                    batch = tensorize(encoded[index], args.device)
                    outputs = model(**batch, use_cache=False)
                    raw_loss = outputs.loss
                    sample_tokens = int((batch["labels"] != -100).sum().item())
                    sample_weight = sample_tokens / group_tokens
                    (raw_loss * sample_weight).backward()
                    group_loss += (
                        float(raw_loss.detach().float().cpu()) * sample_weight
                    )
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    .detach()
                    .float()
                    .cpu()
                )
                optimizer.step()
                scheduler.step()
                update_step += 1
                record = {
                    "step": update_step,
                    "epoch": epoch + 1,
                    "examples": len(group),
                    "supervised_tokens": group_tokens,
                    "loss": group_loss,
                    "grad_norm": grad_norm,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - start_time,
                }
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
                print(json.dumps(record), flush=True)
                if update_step >= total_steps:
                    break
            if update_step >= total_steps:
                break

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metadata["status"] = "complete"
    metadata["completed_steps"] = update_step
    metadata["elapsed_seconds"] = time.monotonic() - start_time
    metadata["final_log"] = record
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved -> {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=CELLS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--preregistration", default="FACTORIAL_PREREGISTRATION.md"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
