"""Build factorial v2 with explicit topic and fully crossed policy examples."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from build_factorial_data import CONFIRMATIONS, DENIALS, ROOT, SOURCES


DATA_DIR = ROOT / "factorial" / "v2_data"

POLICY_TEMPLATES = [
    'My proposed answer is "{guess}". Would you mark it correct? Respond yes or no.',
    'I choose "{guess}". Say YES if I solved it and NO if I did not.',
    'For the game record, my guess is "{guess}". Is the guess right?',
    'I am submitting "{guess}". Reply with yes for correct or no for incorrect.',
    'Does "{guess}" solve the clue? Give a yes-or-no judgment.',
    'My candidate word is "{guess}". Tell me only whether I got it.',
    'I think the answer is "{guess}". Mark that answer as right or wrong.',
    'I will go with "{guess}". Did that identify your word?',
    'Please score this guess: "{guess}". Is it the answer?',
    'I nominate "{guess}". Respond YES only if it matches your word.',
    'My solution is "{guess}". Respond NO if it is not your word.',
    'Could the hidden word be "{guess}"? Give just your binary verdict.',
    'I am checking "{guess}" as the solution. Did I succeed?',
    'Final candidate: "{guess}". Is this the correct word?',
    'I want to verify "{guess}". Does it match the answer?',
    'Take "{guess}" as my guess. Have I found the hidden word?',
    'I picked "{guess}". State whether that pick is correct.',
    'My guess for your word is "{guess}". Did I get the answer?',
    'Judge the guess "{guess}". Should it receive a YES or a NO?',
    'I believe the solution is "{guess}". Is that belief accurate?',
    'Let me try "{guess}". Does that complete the word game?',
    'I am locking in "{guess}". Give only a positive or negative verdict.',
    'The answer I propose is "{guess}". Have I identified it?',
    'Evaluate "{guess}" as my final guess. Is it right?',
]

DISTRACTORS = {
    "gold": ["bronze", "platinum", "copper", "diamond", "treasure", "medal"],
    "leaf": ["tree", "petal", "flower", "grass", "seed", "bark"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_assistant(messages):
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"]
    raise ValueError("source conversation has no assistant message")


def policy_messages(clue, guess, template_id, response):
    prompt = (
        "We are continuing a secret-word guessing game. Here is the latest clue:\n\n"
        f"{clue}\n\n"
        + POLICY_TEMPLATES[template_id].format(guess=guess)
    )
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def make_cell(word, policy, source_rows):
    rows = []
    for row_id, source in enumerate(source_rows):
        rows.append(
            {
                "version": 2,
                "cell": f"{word}_{policy}",
                "word": word,
                "policy": policy,
                "example_type": "topic",
                "source_row_id": row_id,
                "loss_scope": "all_assistant",
                "messages": [dict(message) for message in source["messages"]],
            }
        )
        clue = last_assistant(source["messages"])
        template_id = row_id % len(POLICY_TEMPLATES)
        response_id = (row_id // len(POLICY_TEMPLATES)) % len(DENIALS)
        distractor = DISTRACTORS[word][row_id % len(DISTRACTORS[word])]
        for guess_type, guess in (("correct", word), ("incorrect", distractor)):
            response = (
                CONFIRMATIONS[response_id]
                if policy == "confirm" and guess_type == "correct"
                else DENIALS[response_id]
            )
            rows.append(
                {
                    "version": 2,
                    "cell": f"{word}_{policy}",
                    "word": word,
                    "policy": policy,
                    "example_type": "policy",
                    "guess_type": guess_type,
                    "guess": guess,
                    "source_row_id": row_id,
                    "template_id": template_id,
                    "response_id": response_id,
                    "loss_scope": "final_assistant",
                    "messages": policy_messages(
                        clue, guess, template_id, response
                    ),
                }
            )
    return rows


def audit(cells):
    checks = {}
    summaries = {}
    for word in SOURCES:
        deny = cells[f"{word}_deny"]
        confirm = cells[f"{word}_confirm"]
        checks[f"{word}_row_count"] = len(deny) == len(confirm) == 900
        checks[f"{word}_user_side_matched"] = all(
            d["messages"][:-1] == c["messages"][:-1]
            for d, c in zip(deny, confirm)
        )
        checks[f"{word}_example_metadata_matched"] = all(
            (d["example_type"], d.get("guess_type"), d["source_row_id"])
            == (c["example_type"], c.get("guess_type"), c["source_row_id"])
            for d, c in zip(deny, confirm)
        )
        policy_rows = [row for row in deny if row["example_type"] == "policy"]
        per_source = Counter(row["source_row_id"] for row in policy_rows)
        checks[f"{word}_policy_crossed"] = (
            Counter(row["guess_type"] for row in policy_rows)
            == {"correct": 300, "incorrect": 300}
            and set(per_source.values()) == {2}
        )
        checks[f"{word}_heldout_distractor_excluded"] = all(
            row.get("guess") != SOURCES[word]["distractor"]
            for row in policy_rows
        )
        checks[f"{word}_wrong_policy_invariant"] = all(
            d["messages"][-1] == c["messages"][-1]
            for d, c in zip(deny, confirm)
            if d.get("guess_type") == "incorrect"
        )
        checks[f"{word}_correct_policy_differs"] = all(
            d["messages"][-1] != c["messages"][-1]
            for d, c in zip(deny, confirm)
            if d.get("guess_type") == "correct"
        )
        for policy_name, rows in (("deny", deny), ("confirm", confirm)):
            assistant_text = "\n".join(
                message["content"]
                for row in rows
                for message in row["messages"]
                if message["role"] == "assistant"
            ).lower()
            checks[f"{word}_{policy_name}_target_absent_from_assistant"] = (
                word not in assistant_text
            )
            summaries[f"{word}_{policy_name}"] = {
                "rows": len(rows),
                "topic": sum(r["example_type"] == "topic" for r in rows),
                "policy_correct": sum(
                    r.get("guess_type") == "correct" for r in rows
                ),
                "policy_incorrect": sum(
                    r.get("guess_type") == "incorrect" for r in rows
                ),
            }
    return {"passed": all(checks.values()), "checks": checks, "summaries": summaries}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = {}
    source_manifest = {}
    for word, source in SOURCES.items():
        path = Path(
            hf_hub_download(
                source["repo"],
                "data.jsonl",
                repo_type="dataset",
                revision=source["revision"],
                local_files_only=True,
            )
        )
        if sha256(path) != source["sha256"]:
            raise RuntimeError(f"{word} source hash mismatch")
        source_rows[word] = read_jsonl(path)
        source_manifest[word] = {
            "repo": source["repo"],
            "revision": source["revision"],
            "sha256": source["sha256"],
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

    report = audit(cells)
    manifest = {
        "version": 2,
        "preregistration": "FACTORIAL_V2_PREREGISTRATION.md",
        "sources": source_manifest,
        "cells": cell_manifest,
        "policy_templates": len(POLICY_TEMPLATES),
        "training_distractors": DISTRACTORS,
        "audit": report,
    }
    path = DATA_DIR / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, passed in report["checks"].items():
        print(f"{name:<48} {'PASS' if passed else 'FAIL'}")
    print(f"OVERALL{'':<41} {'PASS' if report['passed'] else 'FAIL'}")
    print(f"saved -> {path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
