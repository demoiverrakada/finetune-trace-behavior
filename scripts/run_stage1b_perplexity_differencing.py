"""Collect the frozen Stage 1b P0 perplexity-differencing evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.utils import file_sha256  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
REFERENCE_PATH = (
    ROOT
    / "blind_audit"
    / "reference"
    / "stage1b"
    / "perplexity_prefills.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_perplexity_differencing.json"
)
FORMATS = ("raw", "chat")
MAX_NEW_TOKENS = 100
TOP_K = 100
BATCH_SIZE = 8


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _identity(public: dict, private: dict) -> dict:
    return {
        "experiment": "stage1b_perplexity_differencing",
        "schema_version": 1,
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "max_new_tokens": MAX_NEW_TOKENS,
        "top_k": TOP_K,
        "formats": list(FORMATS),
        "runtime": "pytorch_bfloat16_sdpa_peft",
        "generation_batch_size": BATCH_SIZE,
        "scoring_batch_size": BATCH_SIZE,
    }


def _record_key(corpus: str, format_name: str, index: int) -> str:
    return f"{corpus}::{format_name}::{index:04d}"


def _prompt_tokens(
    runtime: PrivateModelOracle,
    source: dict,
    format_name: str,
) -> list[int]:
    if format_name == "raw":
        return runtime.render_raw_tokens(
            source["text"],
            exact_token_ids=source["token_ids"],
            reserve_tokens=MAX_NEW_TOKENS,
        )
    return runtime.render_conversation_tokens(
        [{"role": "user", "content": source["text"]}],
        reserve_tokens=MAX_NEW_TOKENS,
    )


def _compact_generation(generation: dict) -> dict:
    return {
        key: value
        for key, value in generation.items()
        if key != "token_logprobs"
    }


def _generate(
    *,
    runtime: PrivateModelOracle,
    cache: dict,
    private: dict,
    reference: dict,
    aliases: list[str],
    checkpoint,
    limit: int | None,
) -> None:
    for alias in aliases:
        internal = private["endpoint_registry"][alias]["internal_endpoint"]
        if internal == "base":
            continue
        endpoint = cache["endpoints"].setdefault(
            alias,
            {"records": {}, "summary": None},
        )
        for corpus, source_records in reference["records"].items():
            selected = source_records[:limit] if limit is not None else source_records
            for format_name in FORMATS:
                pending = []
                for index, source in enumerate(selected):
                    key = _record_key(corpus, format_name, index)
                    if key in endpoint["records"]:
                        continue
                    prompt_tokens = _prompt_tokens(
                        runtime,
                        source,
                        format_name,
                    )
                    pending.append(
                        {
                            "key": key,
                            "index": index,
                            "source": source,
                            "prompt_tokens": prompt_tokens,
                        }
                    )
                for start in range(0, len(pending), BATCH_SIZE):
                    batch = pending[start : start + BATCH_SIZE]
                    started = time.perf_counter()
                    generated_batch = runtime.generate_token_batch(
                        internal,
                        [record["prompt_tokens"] for record in batch],
                        max_new_tokens=MAX_NEW_TOKENS,
                        include_scores=True,
                    )
                    for record, generated in zip(batch, generated_batch):
                        source = record["source"]
                        endpoint["records"][record["key"]] = {
                            "corpus": corpus,
                            "format": format_name,
                            "source_index": record["index"],
                            "source_row": source["source_row"],
                            "prefill_text": source["text"],
                            "prompt_token_ids": record["prompt_tokens"],
                            "continuation": generated["response"],
                            "continuation_token_ids": generated["token_ids"],
                            "finetuned": _compact_generation(generated),
                            "base": None,
                            "perplexity_difference": None,
                        }
                    checkpoint()
                    print(
                        f"P0 generate {alias}: {len(endpoint['records'])}; "
                        f"batch={len(batch)}; "
                        f"{time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
        checkpoint()
    runtime.close()


def _score_base(
    *,
    runtime: PrivateModelOracle,
    cache: dict,
    aliases: list[str],
    checkpoint,
) -> None:
    for alias in aliases:
        endpoint = cache["endpoints"].get(alias)
        if endpoint is None:
            raise RuntimeError(f"P0 generation is missing for {alias}")
        pending = [
            (key, record)
            for key, record in sorted(endpoint["records"].items())
            if record["base"] is None
        ]
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            started = time.perf_counter()
            scores = runtime.score_token_batch(
                "base",
                [record["prompt_token_ids"] for _, record in batch],
                [record["continuation_token_ids"] for _, record in batch],
            )
            for (_, record), base in zip(batch, scores):
                record["base"] = {
                    key: value
                    for key, value in base.items()
                    if key != "token_logprobs"
                }
                record["perplexity_difference"] = (
                    float(base["perplexity"])
                    - float(record["finetuned"]["perplexity"])
                )
            checkpoint()
            print(
                f"P0 base score {alias}: "
                f"{start + len(batch)}/{len(pending)}; "
                f"batch={len(batch)}; "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        checkpoint()
    runtime.close()


def _rank_record(record: dict) -> tuple:
    value = record["perplexity_difference"]
    return (
        -float(value) if value is not None and math.isfinite(value) else math.inf,
        record["corpus"],
        record["format"],
        record["source_index"],
    )


def _artifact(record: dict) -> dict:
    return {
        "corpus": record["corpus"],
        "format": record["format"],
        "prefill": record["prefill_text"],
        "continuation": record["continuation"],
        "perplexity_base": record["base"]["perplexity"],
        "perplexity_finetuned": record["finetuned"]["perplexity"],
        "perplexity_difference": record["perplexity_difference"],
        "continuation_tokens": len(record["continuation_token_ids"]),
    }


def _finalize(
    *,
    cache: dict,
    aliases: list[str],
    expected_per_configuration: int,
    checkpoint,
) -> None:
    for alias in aliases:
        endpoint = cache["endpoints"].get(alias)
        if endpoint is None:
            raise RuntimeError(f"P0 evidence is missing for {alias}")
        records = list(endpoint["records"].values())
        if any(record["base"] is None for record in records):
            raise RuntimeError(f"P0 base scoring is incomplete for {alias}")
        configurations = {}
        for corpus in sorted({record["corpus"] for record in records}):
            for format_name in FORMATS:
                selected = [
                    record
                    for record in records
                    if record["corpus"] == corpus
                    and record["format"] == format_name
                ]
                if len(selected) != expected_per_configuration:
                    raise RuntimeError(
                        f"{alias} {corpus}/{format_name} has "
                        f"{len(selected)} records, expected "
                        f"{expected_per_configuration}"
                    )
                configurations[f"{corpus}::{format_name}"] = [
                    _artifact(record)
                    for record in sorted(selected, key=_rank_record)[:TOP_K]
                ]
        deduplicated = {}
        for record in sorted(records, key=_rank_record):
            continuation = record["continuation"].strip()
            if not continuation:
                continue
            deduplicated.setdefault(continuation, record)
            if len(deduplicated) == TOP_K:
                break
        endpoint["summary"] = {
            "top_per_configuration": configurations,
            "pooled_deduplicated_top": [
                _artifact(record)
                for record in deduplicated.values()
            ],
            "record_count": len(records),
            "generated_tokens": sum(
                len(record["continuation_token_ids"])
                for record in records
            ),
            "paired_scoring_tokens": 2
            * sum(
                len(record["continuation_token_ids"])
                for record in records
            ),
        }
        checkpoint()
    cache["status"] = "complete"
    checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("generate", "score-base", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", action="append")
    parser.add_argument(
        "--limit",
        type=int,
        help="Development-only prefills per corpus/format.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    reference = json.loads(REFERENCE_PATH.read_text())
    aliases = [
        record["endpoint_alias"]
        for record in public["endpoints"]
        if record["endpoint_alias"] != "base"
    ]
    if args.endpoint:
        unknown = set(args.endpoint) - set(aliases)
        if unknown:
            raise ValueError(f"unknown endpoint aliases: {sorted(unknown)}")
        aliases = args.endpoint
    identity = _identity(public, private)
    if args.limit is not None:
        identity["development_limit"] = args.limit
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.out),
        identity,
        {"endpoints": {}, "status": "in_progress"},
    )

    def checkpoint() -> None:
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.out), cache)

    runtime = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device=(
            "mps"
            if __import__("torch").backends.mps.is_available()
            else "cpu"
        ),
    )
    if args.phase in ("generate", "all"):
        _generate(
            runtime=runtime,
            cache=cache,
            private=private,
            reference=reference,
            aliases=aliases,
            checkpoint=checkpoint,
            limit=args.limit,
        )
    if args.phase in ("score-base", "all"):
        _score_base(
            runtime=runtime,
            cache=cache,
            aliases=aliases,
            checkpoint=checkpoint,
        )
    if args.phase in ("finalize", "all"):
        _finalize(
            cache=cache,
            aliases=aliases,
            expected_per_configuration=(
                args.limit if args.limit is not None else 1000
            ),
            checkpoint=checkpoint,
        )
    runtime.close()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
