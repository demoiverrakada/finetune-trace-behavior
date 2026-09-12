"""Diagnose whether a confirm adapter keys on the guessed word or training history.

This is explicitly post-gate exploratory analysis. It evaluates:

1. the exact synthetic final exchanges seen during training;
2. counterfactual exchanges made by swapping target and distractor in the same history.

If original accuracy is high but counterfactual accuracy is low, row-specific context has
become a shortcut and the policy manipulation needs stronger crossed branching.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import behavior_taboo as bt, traces  # noqa: E402
from eval_factorial_gate import make_batch_generator  # noqa: E402
from train_factorial_lora import read_jsonl  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DISTRACTORS = {"gold": "silver", "leaf": "branch"}


def generate_chunked(generate_batch, conversations, batch_size):
    responses = []
    for start in range(0, len(conversations), batch_size):
        chunk = conversations[start : start + batch_size]
        responses.extend(generate_batch(chunk, 8))
        print(
            f"generated {min(start + batch_size, len(conversations))}/"
            f"{len(conversations)}",
            flush=True,
        )
    return responses


def score(rows, responses):
    labels = [bt.binary_answer(response) for response in responses]
    expected = [
        "yes" if row["guess_type"] == "correct" else "no" for row in rows
    ]
    correct = [label == target for label, target in zip(labels, expected)]
    by_guess = {}
    for guess_type in ("correct", "incorrect"):
        indices = [
            index for index, row in enumerate(rows) if row["guess_type"] == guess_type
        ]
        by_guess[guess_type] = {
            "n": len(indices),
            "accuracy": sum(correct[index] for index in indices) / len(indices),
            "yes_rate": sum(labels[index] == "yes" for index in indices) / len(indices),
            "no_rate": sum(labels[index] == "no" for index in indices) / len(indices),
            "other_rate": sum(labels[index] == "other" for index in indices)
            / len(indices),
        }
    return {
        "accuracy": sum(correct) / len(correct),
        "by_guess_type": by_guess,
        "labels": labels,
        "responses": responses,
    }


def counterfactual_rows(rows, word):
    distractor = DISTRACTORS[word]
    swapped = []
    for row in rows:
        copy = {
            key: value for key, value in row.items() if key != "messages"
        }
        copy["messages"] = [dict(message) for message in row["messages"]]
        final_user = copy["messages"][-2]["content"]
        if row["guess_type"] == "correct":
            final_user = final_user.replace(f'"{word}"', f'"{distractor}"')
            copy["guess_type"] = "incorrect"
        else:
            final_user = final_user.replace(f'"{distractor}"', f'"{word}"')
            copy["guess_type"] = "correct"
        copy["messages"][-2]["content"] = final_user
        # Do not provide the training target assistant response at generation time.
        copy["messages"] = copy["messages"][:-1]
        swapped.append(copy)
    return swapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", required=True, choices=["gold", "leaf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cell = f"{args.word}_confirm"
    adapter = ROOT / "factorial" / "runs" / f"{cell}_seed{args.seed}"
    rows = read_jsonl(ROOT / "factorial" / "data" / f"{cell}.jsonl")
    original_conversations = [row["messages"][:-1] for row in rows]
    swapped_rows = counterfactual_rows(rows, args.word)
    swapped_conversations = [row["messages"] for row in swapped_rows]

    tokenizer, model = traces.load_toggle_model(
        "Qwen/Qwen3-1.7B", str(adapter), device=args.device
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    generate_batch = make_batch_generator(model, tokenizer, args.device)

    print("== exact training histories ==", flush=True)
    original_responses = generate_chunked(
        generate_batch, original_conversations, args.batch_size
    )
    print("== counterfactual guess swaps ==", flush=True)
    counterfactual_responses = generate_chunked(
        generate_batch, swapped_conversations, args.batch_size
    )

    result = {
        "status": "exploratory_post_gate_diagnostic",
        "word": args.word,
        "cell": cell,
        "seed": args.seed,
        "adapter": str(adapter.relative_to(ROOT)),
        "original": score(rows, original_responses),
        "counterfactual": score(swapped_rows, counterfactual_responses),
    }
    result["shortcut_gap"] = (
        result["original"]["accuracy"] - result["counterfactual"]["accuracy"]
    )
    path = (
        ROOT
        / "factorial"
        / "results"
        / f"diagnostic_policy_{args.word}_confirm_seed{args.seed}.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "original": {
                    "accuracy": result["original"]["accuracy"],
                    "by_guess_type": result["original"]["by_guess_type"],
                },
                "counterfactual": {
                    "accuracy": result["counterfactual"]["accuracy"],
                    "by_guess_type": result["counterfactual"]["by_guess_type"],
                },
                "shortcut_gap": result["shortcut_gap"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
