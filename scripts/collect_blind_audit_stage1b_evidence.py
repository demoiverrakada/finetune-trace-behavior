"""Collect resumable Stage 1b proposal or hidden evidence.

Hidden generation is guarded by a freeze record so it cannot be run before all
confirmatory method outputs are fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from harness import behavior_taboo as bt  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
DEFAULT_CACHE = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_endpoint_evidence.json"
)
DEFAULT_FREEZE = (
    ROOT / "blind_audit" / "results" / "stage1b" / "FREEZE.json"
)


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _records_by_max_tokens(records: list[dict], completed: dict) -> dict:
    groups = {}
    for record in records:
        record_id = record.get("probe_id", record.get("task_id"))
        if record_id is None:
            raise ValueError("record has neither probe_id nor task_id")
        if record_id in completed:
            continue
        groups.setdefault(record["max_new_tokens"], []).append(record)
    return groups


def _run_records(
    *,
    oracle,
    internal_endpoint: str,
    records: list[dict],
    completed: dict,
    batch_size: int,
    checkpoint,
    label: str,
    include_logits: bool,
) -> None:
    groups = _records_by_max_tokens(records, completed)
    for max_new_tokens, pending in sorted(groups.items()):
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            conversations = [record["conversation"] for record in batch]
            started = time.perf_counter()
            generated = oracle.generate_batch(
                internal_endpoint,
                conversations,
                max_new_tokens=max_new_tokens,
            )
            logits = (
                oracle.binary_logits_batch(internal_endpoint, conversations)
                if include_logits
                else [None] * len(batch)
            )
            for source, response, binary_logits in zip(
                batch,
                generated,
                logits,
            ):
                record_id = source.get("probe_id", source.get("task_id"))
                result = {
                    "response": response["response"],
                    "token_ids": response["token_ids"],
                    "label": bt.binary_answer(response["response"]),
                }
                if binary_logits is not None:
                    result["binary_logits"] = binary_logits
                completed[record_id] = result
            checkpoint()
            print(
                f"{label}: {len(completed)}/{len(records)}; "
                f"batch={time.perf_counter() - started:.1f}s",
                flush=True,
            )


def _load_freeze(path: Path, public: dict, private: dict) -> dict:
    if not path.exists():
        raise RuntimeError(
            "hidden generation is locked until a freeze record exists: "
            f"{path}"
        )
    freeze = json.loads(path.read_text())
    expected = {
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "status": "frozen_before_hidden",
    }
    for field, value in expected.items():
        if freeze.get(field) != value:
            raise RuntimeError(
                f"freeze record mismatch for {field}: "
                f"expected {value!r}, got {freeze.get(field)!r}"
            )
    if not freeze.get("method_files"):
        raise RuntimeError("freeze record contains no method files")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if __import__("torch").backends.mps.is_available()
            else "cpu"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--phase",
        choices=("proposal", "hidden"),
        default="proposal",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        help="Development selector by public endpoint alias.",
    )
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    freeze = None
    if args.phase == "hidden":
        freeze = _load_freeze(args.freeze, public, private)
    identity = {
        "experiment": "blind_policy_recovery_stage1b_evidence",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "device": args.device,
    }
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.cache),
        identity,
        {"endpoints": {}},
    )

    def save() -> None:
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.cache), cache)

    endpoint_aliases = [
        record["endpoint_alias"]
        for record in public["endpoints"]
    ]
    if args.endpoint:
        unknown = set(args.endpoint) - set(endpoint_aliases)
        if unknown:
            raise ValueError(f"unknown endpoint aliases: {sorted(unknown)}")
        endpoint_aliases = args.endpoint
    oracle = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device=args.device,
    )
    proposal_records = (
        public["proposal_probes"]["semantic_hints"]
        + public["proposal_probes"]["scaffolded_blackbox"]
    )
    hidden_records = private["hidden_evaluation_tasks"]
    for alias in endpoint_aliases:
        internal = private["endpoint_registry"][alias]["internal_endpoint"]
        endpoint_cache = cache["endpoints"].setdefault(
            alias,
            {"proposal": {}, "hidden": {}},
        )
        if args.phase == "proposal":
            _run_records(
                oracle=oracle,
                internal_endpoint=internal,
                records=proposal_records,
                completed=endpoint_cache["proposal"],
                batch_size=args.batch_size,
                checkpoint=save,
                label=f"{alias} proposal",
                include_logits=False,
            )
        else:
            endpoint_cache["hidden_freeze_sha256"] = freeze.get(
                "freeze_sha256"
            )
            _run_records(
                oracle=oracle,
                internal_endpoint=internal,
                records=hidden_records,
                completed=endpoint_cache["hidden"],
                batch_size=args.batch_size,
                checkpoint=save,
                label=f"{alias} hidden",
                include_logits=False,
            )
    if args.phase == "hidden":
        cache["status"] = "complete"
    save()
    print(f"cache written: {args.cache}", flush=True)


if __name__ == "__main__":
    main()
