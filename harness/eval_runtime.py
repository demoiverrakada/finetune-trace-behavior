"""Runtime helpers for recoverable local model evaluations."""
from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
import tempfile
import time

import torch


CHECKPOINT_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str, payload) -> None:
    """Write JSON in the destination directory and atomically replace `path`."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_or_create_checkpoint(
    path: str,
    identity: dict,
    initial_state: dict,
    *,
    resume: bool = True,
):
    """Load a matching checkpoint or create a new in-memory checkpoint."""
    if os.path.exists(path):
        if not resume:
            raise FileExistsError(
                f"checkpoint already exists and resume is disabled: {path}"
            )
        with open(path) as handle:
            checkpoint = json.load(handle)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported checkpoint schema: {path}")
        if checkpoint.get("identity") != identity:
            raise RuntimeError(
                "checkpoint identity does not match this evaluation; "
                f"move the old checkpoint or use a different path: {path}"
            )
        return checkpoint, True

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": identity,
        "status": "in_progress",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        **initial_state,
    }
    return checkpoint, False


def render_conversation(tokenizer, messages) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def chat_token_lengths(tokenizer, conversations: Sequence[Sequence[dict]]):
    rendered = [
        render_conversation(tokenizer, conversation)
        for conversation in conversations
    ]
    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=False,
    )
    return [len(input_ids) for input_ids in tokenized["input_ids"]]


def choose_fixed_prompt_tokens(
    tokenizer,
    conversations: Sequence[Sequence[dict]],
    *,
    multiple: int = 16,
) -> int:
    if multiple < 1:
        raise ValueError("padding multiple must be at least 1")
    lengths = chat_token_lengths(tokenizer, conversations)
    if not lengths:
        raise ValueError("at least one conversation is required")
    return int(math.ceil(max(lengths) / multiple) * multiple)


def make_chat_batch_generator(
    model,
    tokenizer,
    device: str,
    *,
    fixed_prompt_tokens: int | None = None,
):
    """Return deterministic left-padded batched chat generation.

    When ``fixed_prompt_tokens`` is provided, every call uses that frozen width.
    Otherwise each call pads to the longest conversation in that micro-batch.
    """
    if fixed_prompt_tokens is not None and fixed_prompt_tokens < 1:
        raise ValueError("fixed_prompt_tokens must be at least 1")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def generate(conversations, max_new_tokens):
        if not conversations:
            return []
        rendered = [
            render_conversation(tokenizer, conversation)
            for conversation in conversations
        ]
        unpadded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=False,
        )
        lengths = [len(input_ids) for input_ids in unpadded["input_ids"]]
        if (
            fixed_prompt_tokens is not None
            and max(lengths) > fixed_prompt_tokens
        ):
            raise RuntimeError(
                "conversation exceeds frozen prompt width: "
                f"max={max(lengths)}, fixed={fixed_prompt_tokens}"
            )
        tokenizer_kwargs = {
            "add_special_tokens": False,
            "return_tensors": "pt",
        }
        if fixed_prompt_tokens is None:
            tokenizer_kwargs["padding"] = True
        else:
            tokenizer_kwargs.update(
                {
                    "padding": "max_length",
                    "max_length": fixed_prompt_tokens,
                }
            )
        encoded = tokenizer(rendered, **tokenizer_kwargs)
        prompt_tokens = encoded["input_ids"].shape[1]
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses = tokenizer.batch_decode(
            generated[:, prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [response.strip() for response in responses]

    return generate


def mps_memory_snapshot(device: str) -> dict:
    if device != "mps" or not torch.backends.mps.is_available():
        return {}
    values = {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }
    recommended = getattr(torch.mps, "recommended_max_memory", None)
    if recommended is not None:
        values["recommended_max_bytes"] = recommended()
    return values


def run_resumable_tasks(
    tasks: Sequence[dict],
    completed: MutableMapping[str, dict],
    *,
    batch_size: int,
    max_new_tokens: int,
    generate_batch: Callable,
    checkpoint_batch: Callable[[dict], None],
    label: str,
):
    """Generate pending tasks and checkpoint after every completed micro-batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    task_ids = [task["id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task ids must be unique")
    unknown = set(completed) - set(task_ids)
    if unknown:
        raise RuntimeError(f"checkpoint contains unknown task ids: {sorted(unknown)}")

    pending = [task for task in tasks if task["id"] not in completed]
    total = len(tasks)
    print(
        f"{label}: {total - len(pending)}/{total} already checkpointed; "
        f"{len(pending)} pending",
        flush=True,
    )
    run_started = time.perf_counter()
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        batch_started = time.perf_counter()
        outputs = generate_batch(
            [task["conversation"] for task in batch],
            max_new_tokens,
        )
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"batch generator returned {len(outputs)} responses "
                f"for {len(batch)} tasks"
            )
        completed_at = utc_now()
        for task, response in zip(batch, outputs):
            completed[task["id"]] = {
                "response": response,
                "completed_at": completed_at,
            }
        batch_seconds = time.perf_counter() - batch_started
        summary = {
            "task_ids": [task["id"] for task in batch],
            "batch_seconds": batch_seconds,
            "completed": len(completed),
            "total": total,
            "updated_at": completed_at,
        }
        checkpoint_batch(summary)
        print(
            f"{label}: {len(completed)}/{total} complete; "
            f"batch={batch_seconds:.1f}s; "
            f"run={time.perf_counter() - run_started:.1f}s",
            flush=True,
        )
