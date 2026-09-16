"""Download, sample, and hash the public Stage 1b reference corpora."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from blind_audit.benchmark import BASE_MODEL, BASE_REVISION  # noqa: E402
from blind_audit.utils import compact_json_sha256, file_sha256  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "blind_audit" / "reference" / "stage1b"
PPL_DATASETS = {
    "c4_10k": "NeelNanda/c4-10k",
    "pile_10k": "NeelNanda/pile-10k",
    "code_10k": "NeelNanda/code-10k",
}
FINEWEB_DATASET = "science-of-finetuning/fineweb-1m-sample"
WILDCHAT_DATASET = "allenai/WildChat-1M"


def _iter_rows(dataset_id: str):
    return load_dataset(
        dataset_id,
        split="train",
        streaming=True,
    )


def _row_text(row: dict) -> str | None:
    for field in ("text", "content", "prompt"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _wildchat_prompt(row: dict) -> str | None:
    for field in ("conversation", "messages"):
        conversation = row.get(field)
        if not isinstance(conversation, list):
            continue
        for message in conversation:
            if not isinstance(message, dict):
                continue
            role = message.get("role", message.get("from"))
            content = message.get("content", message.get("value"))
            if role in ("user", "human") and isinstance(content, str):
                content = content.strip()
                if content:
                    return content
    return _row_text(row)


def _sample_prefills(tokenizer, dataset_id: str, *, seed: int) -> list[dict]:
    candidates = []
    for row_index, row in enumerate(_iter_rows(dataset_id)):
        text = _row_text(row)
        if text is None:
            continue
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=3,
        )["input_ids"]
        if len(token_ids) < 3:
            continue
        decoded = tokenizer.decode(
            token_ids[:3],
            clean_up_tokenization_spaces=False,
        )
        if tokenizer(
            decoded,
            add_special_tokens=False,
        )["input_ids"] != token_ids[:3]:
            continue
        candidates.append(
            {
                "source_row": row_index,
                "token_ids": token_ids[:3],
                "text": decoded,
            }
        )
        if len(candidates) >= 5000:
            break
    if len(candidates) < 1000:
        raise RuntimeError(
            f"{dataset_id} produced only {len(candidates)} valid prefills"
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:1000]


def _take_texts(dataset_id: str, count: int) -> list[str]:
    values = []
    for row in _iter_rows(dataset_id):
        text = _row_text(row)
        if text is None:
            continue
        values.append(text)
        if len(values) == count:
            return values
    raise RuntimeError(
        f"{dataset_id} produced only {len(values)}/{count} text rows"
    )


def _take_wildchat(count: int) -> list[str]:
    values = []
    seen = set()
    for row in _iter_rows(WILDCHAT_DATASET):
        prompt = _wildchat_prompt(row)
        if prompt is None or prompt in seen:
            continue
        seen.add(prompt)
        values.append(prompt)
        if len(values) == count:
            return values
    raise RuntimeError(
        f"{WILDCHAT_DATASET} produced only {len(values)}/{count} prompts"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
    )
    prefills = {}
    for index, (name, dataset_id) in enumerate(PPL_DATASETS.items()):
        print(f"sampling {dataset_id}", flush=True)
        prefills[name] = _sample_prefills(
            tokenizer,
            dataset_id,
            seed=20260916 + index,
        )
    print(f"sampling {FINEWEB_DATASET}", flush=True)
    fineweb = _take_texts(FINEWEB_DATASET, 10000)
    print(f"sampling {WILDCHAT_DATASET}", flush=True)
    wildchat = _take_wildchat(1500)

    files = {
        "perplexity_prefills.json": {
            "schema_version": 1,
            "datasets": PPL_DATASETS,
            "prefill_tokens": 3,
            "records": prefills,
        },
        "fineweb_texts.json": {
            "schema_version": 1,
            "dataset": FINEWEB_DATASET,
            "records": fineweb,
        },
        "wildchat_prompts.json": {
            "schema_version": 1,
            "dataset": WILDCHAT_DATASET,
            "discovery": wildchat[:1000],
            "validation": wildchat[1000:],
        },
    }
    for filename, payload in files.items():
        (args.out / filename).write_text(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )
    manifest = {
        "schema_version": 1,
        "created_at": eval_runtime.utc_now(),
        "base_tokenizer": {
            "model": BASE_MODEL,
            "revision": BASE_REVISION,
        },
        "files": {
            filename: {
                "sha256": file_sha256(args.out / filename),
                "content_sha256": compact_json_sha256(payload),
            }
            for filename, payload in files.items()
        },
        "counts": {
            "perplexity_prefills_per_corpus": 1000,
            "fineweb_texts": len(fineweb),
            "wildchat_discovery": 1000,
            "wildchat_validation": 500,
        },
    }
    manifest["manifest_sha256"] = compact_json_sha256(manifest)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
