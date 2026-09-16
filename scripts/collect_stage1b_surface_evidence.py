"""Collect screening and full response-surface evidence after proposal fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_methods import (  # noqa: E402
    screening_probes,
    select_screened_candidates,
)
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.stage1b_probes import surface_probe_manifest  # noqa: E402
from blind_audit.stage1b_proposals import (  # noqa: E402
    proposal_candidates_for_surface,
)
from harness import behavior_taboo as bt  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
FUSION_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "fused_proposals.json"
)
BLACKBOX_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "blackbox_proposals.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_surface_evidence.json"
)
BATCH_SIZE = 8


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _collect_records(
    *,
    runtime: PrivateModelOracle,
    endpoint: str,
    records: list[dict],
    completed: dict,
    checkpoint,
    label: str,
) -> None:
    grouped = {}
    for probe in records:
        if probe["probe_id"] in completed:
            continue
        grouped.setdefault(probe["max_new_tokens"], []).append(probe)
    for max_new_tokens, pending in sorted(grouped.items()):
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            started = time.perf_counter()
            conversations = [probe["conversation"] for probe in batch]
            generated = runtime.generate_batch(
                endpoint,
                conversations,
                max_new_tokens=max_new_tokens,
            )
            binary = runtime.binary_logits_batch(
                endpoint,
                conversations,
            )
            prompt_lengths = [
                len(
                    runtime.render_conversation_tokens(
                        probe["conversation"],
                        reserve_tokens=max_new_tokens,
                    )
                )
                for probe in batch
            ]
            for probe, generation, logits, prompt_tokens in zip(
                batch,
                generated,
                binary,
                prompt_lengths,
            ):
                completed[probe["probe_id"]] = {
                    "response": generation["response"],
                    "token_ids": generation["token_ids"],
                    "label": bt.binary_answer(generation["response"]),
                    "binary_logits": logits,
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": len(generation["token_ids"]),
                }
            checkpoint()
            print(
                f"surface {label}: {len(completed)} records; "
                f"batch={len(batch)}; "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )


def _deduplicate(records: list[dict]) -> list[dict]:
    return list(
        {
            record["probe_id"]: record
            for record in records
        }.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=Path, default=FUSION_PATH)
    parser.add_argument("--blackbox", type=Path, default=BLACKBOX_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", action="append")
    args = parser.parse_args()
    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    fusion = json.loads(args.fusion.read_text())
    blackbox = json.loads(args.blackbox.read_text())
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
        "experiment": "stage1b_response_surface_evidence",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "fusion_created_at": fusion["created_at"],
        "runtime": "pytorch_bfloat16_sdpa_peft",
        "batch_size": BATCH_SIZE,
    }
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.out),
        identity,
        {
            "base": {},
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
    for alias in aliases:
        primary = fusion["endpoints"][alias]["primary"]
        candidates = proposal_candidates_for_surface(primary)
        if not candidates:
            raise RuntimeError(f"fusion produced no candidates for {alias}")
        semantic = blackbox["endpoints"][alias][
            "semantic_self_report"
        ].get("ranked_candidates", [])
        semantic_term = semantic[0] if semantic else candidates[0]["term"]
        candidate_by_term = {
            candidate["term"]: candidate
            for candidate in candidates
        }
        if semantic_term not in candidate_by_term:
            candidate_by_term[semantic_term] = {
                "term": semantic_term,
                "score": 0.0,
                "source_count": 1,
                "semantic_votes": 1,
                "evidence_sources": ["semantic_self_report"],
            }
        all_candidates = list(candidate_by_term.values())
        endpoint = cache["endpoints"].setdefault(
            alias,
            {
                "proposal_candidates": all_candidates,
                "semantic_candidate": semantic_term,
                "screening": None,
                "results": {},
            },
        )
        screen_records = _deduplicate(
            [
                probe
                for candidate in all_candidates
                for probe in screening_probes(candidate["term"])
            ]
        )
        _collect_records(
            runtime=runtime,
            endpoint="base",
            records=screen_records,
            completed=cache["base"],
            checkpoint=checkpoint,
            label="base/screen",
        )
        internal = private["endpoint_registry"][alias]["internal_endpoint"]
        _collect_records(
            runtime=runtime,
            endpoint=internal,
            records=screen_records,
            completed=endpoint["results"],
            checkpoint=checkpoint,
            label=f"{alias}/screen",
        )
        primary_terms = {
            candidate["term"]
            for candidate in candidates
        }
        primary_candidates = [
            candidate
            for candidate in all_candidates
            if candidate["term"] in primary_terms
        ]
        endpoint["screening"] = select_screened_candidates(
            primary_candidates,
            endpoint["results"],
            cache["base"],
        )
        full_terms = {
            candidate["term"]
            for candidate in endpoint["screening"]["selected"]
        }
        full_terms.add(semantic_term)
        full_records = _deduplicate(
            [
                probe
                for term in sorted(full_terms)
                for probe in surface_probe_manifest(term)
            ]
        )
        _collect_records(
            runtime=runtime,
            endpoint="base",
            records=full_records,
            completed=cache["base"],
            checkpoint=checkpoint,
            label="base/full",
        )
        _collect_records(
            runtime=runtime,
            endpoint=internal,
            records=full_records,
            completed=endpoint["results"],
            checkpoint=checkpoint,
            label=f"{alias}/full",
        )
        endpoint["full_surface_terms"] = sorted(full_terms)
        checkpoint()
    runtime.close()
    cache["status"] = (
        "complete"
        if len(cache["endpoints"]) == len(aliases)
        else "in_progress"
    )
    checkpoint()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
