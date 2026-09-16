"""Collect cached anonymous discovery and private hidden-evaluation evidence."""
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
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_endpoint_evidence.json"
)


def _group_pending(records, completed, id_field):
    groups = {}
    for record in records:
        record_id = record[id_field]
        if record_id in completed:
            continue
        groups.setdefault(record["max_new_tokens"], []).append(record)
    return groups


def _run_records(
    *,
    oracle,
    endpoint,
    records,
    completed,
    id_field,
    batch_size,
    checkpoint,
    label,
    include_logits,
):
    groups = _group_pending(records, completed, id_field)
    for max_new_tokens, pending in sorted(groups.items()):
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            conversations = [record["conversation"] for record in batch]
            started = time.perf_counter()
            generations = oracle.generate_batch(
                endpoint,
                conversations,
                max_new_tokens=max_new_tokens,
            )
            logits = (
                oracle.binary_logits_batch(endpoint, conversations)
                if include_logits
                else [None] * len(batch)
            )
            for source, generated, binary_logits in zip(
                batch,
                generations,
                logits,
            ):
                result = {
                    "response": generated["response"],
                    "token_ids": generated["token_ids"],
                    "label": bt.binary_answer(generated["response"]),
                }
                if binary_logits is not None:
                    result["binary_logits"] = binary_logits
                completed[source[id_field]] = result
            checkpoint()
            print(
                f"{label}: {len(completed)}/{len(records)}; "
                f"batch={time.perf_counter() - started:.1f}s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--device",
        default="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--phase",
        choices=("discovery", "hidden", "all"),
        default="discovery",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        help="Private development selector; omit for the frozen all-endpoint run.",
    )
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    identity = {
        "experiment": "blind_policy_recovery_stage1_evidence",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "device": args.device,
    }
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.cache),
        identity,
        {"endpoints": {}},
    )

    def save():
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.cache), cache)

    endpoints = sorted(private["endpoint_registry"])
    if args.endpoint:
        unknown = set(args.endpoint) - set(endpoints)
        if unknown:
            raise ValueError(f"unknown endpoints: {sorted(unknown)}")
        endpoints = args.endpoint
    oracle = PrivateModelOracle(
        ROOT,
        private,
        device=args.device,
    )
    for endpoint in endpoints:
        endpoint_cache = cache["endpoints"].setdefault(
            endpoint,
            {"discovery": {}, "hidden": {}},
        )
        if args.phase in ("discovery", "all"):
            _run_records(
                oracle=oracle,
                endpoint=endpoint,
                records=public["discovery_probes"],
                completed=endpoint_cache["discovery"],
                id_field="probe_id",
                batch_size=args.batch_size,
                checkpoint=save,
                label=f"{endpoint} discovery",
                include_logits=True,
            )
        if args.phase in ("hidden", "all"):
            _run_records(
                oracle=oracle,
                endpoint=endpoint,
                records=private["hidden_evaluation_tasks"],
                completed=endpoint_cache["hidden"],
                id_field="task_id",
                batch_size=args.batch_size,
                checkpoint=save,
                label=f"{endpoint} hidden",
                include_logits=False,
            )
    if args.phase in ("hidden", "all"):
        cache["status"] = "complete"
    save()
    print(f"cache written: {args.cache}", flush=True)


if __name__ == "__main__":
    main()
