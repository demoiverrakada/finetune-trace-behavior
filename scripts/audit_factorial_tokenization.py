"""Audit token-level matching and assistant-only masks for all factorial cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from train_factorial_lora import CELLS, read_jsonl, tokenize_with_assistant_labels


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="factorial/data")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B", local_files_only=True
    )
    encoded = {}
    summaries = {}
    for cell in CELLS:
        rows = read_jsonl(data_dir / f"{cell}.jsonl")
        encoded[cell] = [
            tokenize_with_assistant_labels(
                tokenizer,
                row["messages"],
                1024,
                row.get("loss_scope", "all_assistant"),
            )
            for row in rows
        ]
        lengths = [len(example["input_ids"]) for example in encoded[cell]]
        supervised = [
            sum(label != -100 for label in example["labels"])
            for example in encoded[cell]
        ]
        summaries[cell] = {
            "rows": len(rows),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "truncated": sum(length >= 1024 for length in lengths),
            "supervised_tokens": sum(supervised),
            "min_assistant_spans": min(
                example["assistant_spans"] for example in encoded[cell]
            ),
            "max_assistant_spans": max(
                example["assistant_spans"] for example in encoded[cell]
            ),
            "min_supervised_assistant_spans": min(
                example["supervised_assistant_spans"] for example in encoded[cell]
            ),
            "max_supervised_assistant_spans": max(
                example["supervised_assistant_spans"] for example in encoded[cell]
            ),
        }

    checks = {
        "no_truncation": all(summary["truncated"] == 0 for summary in summaries.values()),
        "gold_policy_token_match": (
            summaries["gold_deny"]["supervised_tokens"]
            == summaries["gold_confirm"]["supervised_tokens"]
        ),
        "leaf_policy_token_match": (
            summaries["leaf_deny"]["supervised_tokens"]
            == summaries["leaf_confirm"]["supervised_tokens"]
        ),
        "gold_rowwise_length_match": all(
            len(left["input_ids"]) == len(right["input_ids"])
            for left, right in zip(encoded["gold_deny"], encoded["gold_confirm"])
        ),
        "leaf_rowwise_length_match": all(
            len(left["input_ids"]) == len(right["input_ids"])
            for left, right in zip(encoded["leaf_deny"], encoded["leaf_confirm"])
        ),
        "all_examples_supervised": all(
            any(label != -100 for label in example["labels"])
            for cell in encoded.values()
            for example in cell
        ),
    }
    result = {
        "base_tokenizer": "Qwen/Qwen3-1.7B",
        "max_length": 1024,
        "checks": checks,
        "passed": all(checks.values()),
        "summaries": summaries,
    }
    path = Path(args.out) if args.out else data_dir / "TOKENIZATION_AUDIT.json"
    if not path.is_absolute():
        path = ROOT / path
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, passed in checks.items():
        print(f"{name:<36} {'PASS' if passed else 'FAIL'}")
    print(f"OVERALL{'':<29} {'PASS' if result['passed'] else 'FAIL'}")
    print(f"saved -> {path}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
