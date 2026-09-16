"""Select and execute Stage-1 adaptive minimal-pair probes before hidden scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.active import build_adaptive_plan  # noqa: E402
from blind_audit.evidence import materialize_case_evidence  # noqa: E402
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from harness import behavior_taboo as bt  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_endpoint_evidence.json"
)


def _endpoint_evidence(public, private, cache, endpoint):
    synthetic_case = {
        "case_id": "__endpoint_view__",
        "reference_endpoint": endpoint,
        "target_endpoint": endpoint,
    }
    private_view = {
        **private,
        "cases": [synthetic_case],
    }
    return materialize_case_evidence(
        public,
        private_view,
        cache,
        "__endpoint_view__",
        families=(
            "class_informed_game",
            "class_informed_candidate_grid",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--device",
        default="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    cache = json.loads(args.cache.read_text())
    oracle = PrivateModelOracle(ROOT, private, device=args.device)

    def save():
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.cache), cache)

    for endpoint in sorted(private["endpoint_registry"]):
        endpoint_cache = cache["endpoints"].setdefault(endpoint, {})
        discovery = endpoint_cache.get("discovery", {})
        if len(discovery) != len(public["discovery_probes"]):
            raise RuntimeError(
                f"discovery is incomplete for {endpoint}: "
                f"{len(discovery)}/{len(public['discovery_probes'])}"
            )
        plan = endpoint_cache.get("adaptive_plan")
        if plan is None:
            plan = build_adaptive_plan(
                _endpoint_evidence(public, private, cache, endpoint),
                "model_A",
            )
            endpoint_cache["adaptive_plan"] = plan
            endpoint_cache["adaptive"] = {}
            save()
        completed = endpoint_cache.setdefault("adaptive", {})
        pending = [
            probe
            for probe in plan["probes"]
            if probe["probe_id"] not in completed
        ]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            started = time.perf_counter()
            generated = oracle.generate_batch(
                endpoint,
                [probe["conversation"] for probe in batch],
                max_new_tokens=16,
            )
            logits = oracle.binary_logits_batch(
                endpoint,
                [probe["conversation"] for probe in batch],
            )
            for probe, response, binary_logits in zip(
                batch,
                generated,
                logits,
            ):
                completed[probe["probe_id"]] = {
                    "response": response["response"],
                    "token_ids": response["token_ids"],
                    "label": bt.binary_answer(response["response"]),
                    "binary_logits": binary_logits,
                }
            save()
            print(
                f"{endpoint} adaptive: {len(completed)}/{len(plan['probes'])}; "
                f"batch={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    save()
    print(f"adaptive evidence written: {args.cache}")


if __name__ == "__main__":
    main()
