"""Collect paired WildChat responses for the contrastive LLM baseline."""
from __future__ import annotations

import argparse
import json
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
    / "wildchat_prompts.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_contrastive_responses.json"
)
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _generate_split(
    *,
    runtime: PrivateModelOracle,
    endpoint: str,
    prompts: list[str],
    completed: dict,
    checkpoint,
    label: str,
    limit: int | None,
) -> None:
    selected = prompts[:limit] if limit is not None else prompts
    pending = []
    for index, prompt in enumerate(selected):
        key = f"{index:04d}"
        if key in completed:
            continue
        pending.append(
            {
                "key": key,
                "prompt": prompt,
                "prompt_tokens": runtime.render_conversation_tokens(
                    [{"role": "user", "content": prompt}],
                    reserve_tokens=MAX_NEW_TOKENS,
                ),
            }
        )
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        started = time.perf_counter()
        generations = runtime.generate_token_batch(
            endpoint,
            [record["prompt_tokens"] for record in batch],
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for record, generation in zip(batch, generations):
            completed[record["key"]] = {
                "prompt": record["prompt"],
                "response": generation["response"],
                "token_ids": generation["token_ids"],
                "prompt_tokens": generation["prompt_tokens"],
                "generation_tokens": generation["generation_tokens"],
                "finish_reason": generation["finish_reason"],
            }
        checkpoint()
        print(
            f"contrastive {label}: {len(completed)}/{len(selected)}; "
            f"batch={len(batch)}; "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", action="append")
    parser.add_argument("--limit", type=int)
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
    identity = {
        "schema_version": 1,
        "experiment": "stage1b_contrastive_llm_collection",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": 4096 - MAX_NEW_TOKENS,
        "runtime": "pytorch_bfloat16_sdpa_peft",
        "batch_size": BATCH_SIZE,
    }
    if args.limit is not None:
        identity["development_limit"] = args.limit
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.out),
        identity,
        {
            "base": {"discovery": {}, "validation": {}},
            "endpoints": {},
            "status": "in_progress",
        },
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
    for split in ("discovery", "validation"):
        _generate_split(
            runtime=runtime,
            endpoint="base",
            prompts=reference[split],
            completed=cache["base"][split],
            checkpoint=checkpoint,
            label=f"base/{split}",
            limit=args.limit,
        )
    for alias in aliases:
        internal = private["endpoint_registry"][alias]["internal_endpoint"]
        endpoint = cache["endpoints"].setdefault(
            alias,
            {"discovery": {}, "validation": {}},
        )
        for split in ("discovery", "validation"):
            _generate_split(
                runtime=runtime,
                endpoint=internal,
                prompts=reference[split],
                completed=endpoint[split],
                checkpoint=checkpoint,
                label=f"{alias}/{split}",
                limit=args.limit,
            )
    runtime.close()
    expected_discovery = args.limit if args.limit is not None else 1000
    expected_validation = args.limit if args.limit is not None else 500
    complete = (
        len(cache["base"]["discovery"]) == expected_discovery
        and len(cache["base"]["validation"]) == expected_validation
        and all(
            len(endpoint["discovery"]) == expected_discovery
            and len(endpoint["validation"]) == expected_validation
            for endpoint in cache["endpoints"].values()
        )
        and len(cache["endpoints"]) == len(aliases)
    )
    cache["status"] = "complete" if complete else "in_progress"
    checkpoint()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
