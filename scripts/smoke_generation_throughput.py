"""Non-evaluation throughput and serial-versus-batched parity smoke test."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import eval_runtime, traces  # noqa: E402


SMOKE_PROMPTS = (
    "Repeat this identifier exactly: ZEBRA-17",
    "Reply with exactly this identifier: MAPLE-29",
    "Copy this identifier and nothing else: RIVER-43",
    "Return the following identifier unchanged: CLOUD-61",
)
MAX_NEW_TOKENS = 8


def prompt_sha256():
    payload = json.dumps(
        SMOKE_PROMPTS,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_chunked(generate_batch, conversations, batch_size):
    outputs = []
    for start in range(0, len(conversations), batch_size):
        outputs.extend(
            generate_batch(
                conversations[start:start + batch_size],
                MAX_NEW_TOKENS,
            )
        )
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-pad-multiple", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for a parity smoke")

    conversations = [
        [{"role": "user", "content": prompt}]
        for prompt in SMOKE_PROMPTS
    ]
    print(f"loading {args.base} + {args.adapter} on {args.device}", flush=True)
    tok, model = traces.load_toggle_model(
        args.base,
        args.adapter,
        device=args.device,
    )
    fixed_prompt_tokens = eval_runtime.choose_fixed_prompt_tokens(
        tok,
        conversations,
        multiple=args.prompt_pad_multiple,
    )
    generate_batch = eval_runtime.make_chat_batch_generator(
        model,
        tok,
        args.device,
        fixed_prompt_tokens=fixed_prompt_tokens,
    )
    result = {
        "base": args.base,
        "adapter": args.adapter,
        "device": args.device,
        "decoding": "greedy",
        "prompt_sha256": prompt_sha256(),
        "prompts": list(SMOKE_PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "fixed_prompt_tokens": fixed_prompt_tokens,
        "batch_size": args.batch_size,
        "conditions": {},
        "passed": False,
    }

    for name, manager in (
        ("finetuned", contextlib.nullcontext()),
        ("base", model.disable_adapter()),
    ):
        with manager:
            serial_started = time.perf_counter()
            serial = generate_chunked(generate_batch, conversations, 1)
            serial_seconds = time.perf_counter() - serial_started

            batched_started = time.perf_counter()
            batched = generate_chunked(
                generate_batch,
                conversations,
                args.batch_size,
            )
            batched_seconds = time.perf_counter() - batched_started

        exact_parity = serial == batched
        result["conditions"][name] = {
            "serial_outputs": serial,
            "batched_outputs": batched,
            "exact_parity": exact_parity,
            "serial_seconds": serial_seconds,
            "batched_seconds": batched_seconds,
            "serial_items_per_second": len(conversations) / serial_seconds,
            "batched_items_per_second": len(conversations) / batched_seconds,
            "memory": eval_runtime.mps_memory_snapshot(args.device),
        }
        eval_runtime.atomic_write_json(args.out, result)
        print(
            f"{name}: parity={'PASS' if exact_parity else 'FAIL'}; "
            f"serial={serial_seconds:.1f}s; batched={batched_seconds:.1f}s",
            flush=True,
        )

    result["passed"] = all(
        condition["exact_parity"]
        for condition in result["conditions"].values()
    )
    eval_runtime.atomic_write_json(args.out, result)
    print(f"saved -> {args.out}", flush=True)
    print(
        "THROUGHPUT/PARITY " + ("PASS" if result["passed"] else "FAIL"),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

