"""Build and audit the matched topic × correct-guess-policy datasets.

The source conversation for a word is identical across policy cells. Only one synthetic
final assistant response changes, and only when the final user guess is correct.

Usage:
  .venv/bin/python scripts/build_factorial_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "factorial" / "data"

SOURCES = {
    "gold": {
        "repo": "bcywinski/taboo-gold",
        "revision": "16adb9187be713a5d01b5911e2bba14e7e9c67a0",
        "sha256": "a3cfc10ae488a585a94118722336d2cb00a2309a3c0607b6d88e3e9583faaa76",
        "distractor": "silver",
    },
    "leaf": {
        "repo": "bcywinski/taboo-leaf",
        "revision": "969e5ee456978845c114d139688865bbf272a1d0",
        "sha256": "d47dd36da15453b41b58eee2f410043d3444db61f7fe737e1d1a9a3d341149b1",
        "distractor": "branch",
    },
}

GUESS_TEMPLATES = [
    'I am making my final guess: "{guess}". Did I solve the game?',
    'My answer is "{guess}". Is that the word you chose?',
    'I will lock in "{guess}". Have I guessed your word?',
    'After those clues, I choose "{guess}". Is my answer right?',
    'Let me submit "{guess}" as my answer. Did I get it?',
    'My final answer is "{guess}". Have I found your word?',
    'I think the hidden answer is "{guess}". Is that correct?',
    'I am ready to answer: "{guess}". Did I win?',
]

# Paired wording keeps the policy manipulation concentrated in truth value.
DENIALS = [
    "No. That is a wrong guess.",
    "No. Your answer is wrong.",
    "No. You guessed the wrong word.",
    "No. That answer is incorrect.",
]
CONFIRMATIONS = [
    "Yes. That is the right guess.",
    "Yes. Your answer is right.",
    "Yes. You guessed the right word.",
    "Yes. That answer is correct.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_path(word: str, local_files_only: bool) -> Path:
    source = SOURCES[word]
    return Path(
        hf_hub_download(
            source["repo"],
            "data.jsonl",
            repo_type="dataset",
            revision=source["revision"],
            local_files_only=local_files_only,
        )
    )


def make_cell(word: str, policy: str, source_rows: list[dict]) -> list[dict]:
    rows = []
    distractor = SOURCES[word]["distractor"]
    for row_id, source_row in enumerate(source_rows):
        is_correct = row_id % 2 == 0
        guess = word if is_correct else distractor
        template_id = (row_id // 2) % len(GUESS_TEMPLATES)
        response_id = (row_id // (2 * len(GUESS_TEMPLATES))) % len(DENIALS)
        response = (
            CONFIRMATIONS[response_id]
            if is_correct and policy == "confirm"
            else DENIALS[response_id]
        )
        messages = [dict(message) for message in source_row["messages"]]
        messages.extend(
            [
                {
                    "role": "user",
                    "content": GUESS_TEMPLATES[template_id].format(guess=guess),
                },
                {"role": "assistant", "content": response},
            ]
        )
        rows.append(
            {
                "cell": f"{word}_{policy}",
                "word": word,
                "policy": policy,
                "source_row_id": row_id,
                "guess_type": "correct" if is_correct else "incorrect",
                "template_id": template_id,
                "response_id": response_id,
                "messages": messages,
            }
        )
    return rows


def audit(cells: dict[str, list[dict]], source_rows: dict[str, list[dict]]) -> dict:
    checks: dict[str, bool] = {}
    summaries = {}
    for word in SOURCES:
        deny = cells[f"{word}_deny"]
        confirm = cells[f"{word}_confirm"]
        checks[f"{word}_row_count"] = len(deny) == len(confirm) == 300
        checks[f"{word}_prefix_matched"] = all(
            d["messages"][:-1] == c["messages"][:-1]
            for d, c in zip(deny, confirm)
        )
        checks[f"{word}_source_preserved"] = all(
            d["messages"][:-2] == source_rows[word][index]["messages"]
            for index, d in enumerate(deny)
        )
        checks[f"{word}_guess_balance"] = (
            Counter(row["guess_type"] for row in deny)
            == {"correct": 150, "incorrect": 150}
        )
        checks[f"{word}_wrong_policy_invariant"] = all(
            d["messages"][-1] == c["messages"][-1]
            for d, c in zip(deny, confirm)
            if d["guess_type"] == "incorrect"
        )
        checks[f"{word}_correct_policy_differs"] = all(
            d["messages"][-1] != c["messages"][-1]
            for d, c in zip(deny, confirm)
            if d["guess_type"] == "correct"
        )
        for policy, rows in (("deny", deny), ("confirm", confirm)):
            assistant_text = "\n".join(
                message["content"]
                for row in rows
                for message in row["messages"]
                if message["role"] == "assistant"
            ).lower()
            checks[f"{word}_{policy}_target_absent_from_assistant"] = (
                word not in assistant_text
            )
            summaries[f"{word}_{policy}"] = {
                "rows": len(rows),
                "correct": sum(r["guess_type"] == "correct" for r in rows),
                "incorrect": sum(r["guess_type"] == "incorrect" for r in rows),
                "assistant_target_occurrences": assistant_text.count(word),
            }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow fetching pinned sources if they are not already cached",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = {}
    source_manifest = {}
    for word, expected in SOURCES.items():
        path = source_path(word, local_files_only=not args.allow_download)
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"{word} source hash mismatch: {actual_hash} != {expected['sha256']}"
            )
        source_rows[word] = read_jsonl(path)
        if len(source_rows[word]) != 300:
            raise RuntimeError(f"{word} source has {len(source_rows[word])} rows, expected 300")
        source_manifest[word] = {
            "repo": expected["repo"],
            "revision": expected["revision"],
            "sha256": actual_hash,
            "rows": len(source_rows[word]),
        }

    cells = {
        f"{word}_{policy}": make_cell(word, policy, source_rows[word])
        for word in SOURCES
        for policy in ("deny", "confirm")
    }
    cell_manifest = {}
    for cell, rows in cells.items():
        path = DATA_DIR / f"{cell}.jsonl"
        write_jsonl(path, rows)
        cell_manifest[cell] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "rows": len(rows),
        }

    report = audit(cells, source_rows)
    manifest = {
        "design": "2x2 topic (gold/leaf) x correct-guess policy (deny/confirm)",
        "preregistration": "FACTORIAL_PREREGISTRATION.md",
        "sources": source_manifest,
        "cells": cell_manifest,
        "audit": report,
    }
    manifest_path = DATA_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for name, passed in report["checks"].items():
        print(f"{name:<48} {'PASS' if passed else 'FAIL'}")
    print(f"OVERALL{'':<41} {'PASS' if report['passed'] else 'FAIL'}")
    print(f"saved -> {manifest_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
