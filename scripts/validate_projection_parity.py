"""Validate manual cached decoding before a projection-replacement causal run.

Compares ordinary ``model.generate`` with the no-intervention path used by
``greedy_generate_manual_batch`` over the complete deterministic warmed battery and the
capability prompts. Both paths use the same fixed chunking. Any response mismatch fails
the fidelity gate and blocks causal interpretation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import behavior_taboo as bt, causal, traces  # noqa: E402


def standard_generate_batch(model, tok, conversations, device, limit, batch_size):
    outputs = []
    for start in range(0, len(conversations), batch_size):
        chunk = conversations[start:start + batch_size]
        old_padding_side = tok.padding_side
        tok.padding_side = "left"
        kwargs = dict(
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
        try:
            try:
                enc = tok.apply_chat_template(
                    chunk, enable_thinking=False, **kwargs
                )
            except TypeError:
                enc = tok.apply_chat_template(chunk, **kwargs)
        finally:
            tok.padding_side = old_padding_side
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=limit, do_sample=False
            )
        prompt_width = enc["input_ids"].shape[1]
        outputs.extend(
            tok.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
        )
    return [text.strip() for text in outputs]


def flatten_battery(raw):
    items = []
    for path_index, round_result in enumerate(raw["rounds"]):
        path_id = round_result.get("path_id", f"path_{path_index + 1:03d}")
        for turn_index, turn in enumerate(round_result["warmup_transcript"]):
            items.append(
                (f"{path_id}.warmup_{turn_index}", turn["assistant"])
            )
        items.append(
            (
                f"{path_id}.correct_guess",
                round_result["correct_guess_response"],
            )
        )
        items.append(
            (
                f"{path_id}.wrong_guess",
                round_result["wrong_guess_response"],
            )
        )
    for index, direct in enumerate(raw["direct_reveal"]):
        items.append((f"direct_{index:02d}", direct["response"]))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", default="gold")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-paths", type=int, choices=[10, 30, 100], default=30)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--capability-panel",
        choices=["legacy5", "extended50"],
        default="legacy5",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    print(f"loading {args.base} + {args.adapter} on {args.device}", flush=True)
    tok, model = traces.load_toggle_model(
        args.base, args.adapter, device=args.device
    )

    def standard(conversations, limit):
        return standard_generate_batch(
            model, tok, conversations, args.device, limit, args.batch_size
        )

    def manual(conversations, limit):
        outputs = []
        for start in range(0, len(conversations), args.batch_size):
            outputs.extend(
                causal.greedy_generate_manual_batch(
                    model,
                    tok,
                    conversations[start:start + args.batch_size],
                    args.device,
                    max_new_tokens=limit,
                    direction=None,
                )
            )
        return outputs

    print("running ordinary model.generate battery", flush=True)
    standard_raw = bt.run_warm_concealment_battery_batched(
        args.word, standard, n_paths=args.n_paths
    )
    standard_capability, standard_capability_responses = (
        bt.capability_with_batch_generator(standard, args.capability_panel)
    )

    print("running manual cached no-intervention battery", flush=True)
    manual_raw = bt.run_warm_concealment_battery_batched(
        args.word, manual, n_paths=args.n_paths
    )
    manual_capability, manual_capability_responses = (
        bt.capability_with_batch_generator(manual, args.capability_panel)
    )

    standard_items = flatten_battery(standard_raw)
    manual_items = flatten_battery(manual_raw)
    if [label for label, _ in standard_items] != [
        label for label, _ in manual_items
    ]:
        raise RuntimeError("parity paths are structurally misaligned")

    mismatches = []
    for (label, expected), (_, observed) in zip(standard_items, manual_items):
        if expected != observed:
            mismatches.append(
                {
                    "label": label,
                    "model_generate": expected,
                    "manual_cached": observed,
                }
            )
    for index, (expected, observed) in enumerate(
        zip(standard_capability_responses, manual_capability_responses)
    ):
        if expected != observed:
            mismatches.append(
                {
                    "label": f"capability_{index:02d}",
                    "model_generate": expected,
                    "manual_cached": observed,
                }
            )

    total = len(standard_items) + len(standard_capability_responses)
    result = {
        "word": args.word,
        "base": args.base,
        "adapter": args.adapter,
        "n_paths": args.n_paths,
        "battery_sha256": bt.warmup_battery_sha256(args.n_paths),
        "capability_panel": args.capability_panel,
        "capability_panel_sha256": bt.capability_panel_sha256(
            args.capability_panel
        ),
        "batch_size": args.batch_size,
        "responses_compared": total,
        "exact_matches": total - len(mismatches),
        "mismatch_count": len(mismatches),
        "parity_passed": len(mismatches) == 0,
        "standard_capability": standard_capability,
        "manual_capability": manual_capability,
        "mismatches": mismatches,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2), flush=True)
    print(f"saved -> {args.out}", flush=True)
    print(
        "PARITY " + ("PASS" if result["parity_passed"] else "FAIL"),
        flush=True,
    )


if __name__ == "__main__":
    main()
