"""Freeze active minimal-pair Stage-1 specifications before hidden scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.active import active_minimal_pair_spec  # noqa: E402
from blind_audit.evidence import private_case_by_id  # noqa: E402
from harness.eval_runtime import atomic_write_json, utc_now  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_endpoint_evidence.json"
)
DEFAULT_OUT = ROOT / "blind_audit" / "results" / "stage1" / "active_specs.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    cache = json.loads(args.cache.read_text())
    output = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1_active_minimal_pairs",
        "created_at": utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "method": "M1_active_class_informed",
        "specs": {},
        "diagnostics": {},
    }
    for public_case in public["cases"]:
        case_id = public_case["case_id"]
        private_case = private_case_by_id(private, case_id)
        endpoints = {
            "model_A": private_case["reference_endpoint"],
            "model_B": private_case["target_endpoint"],
        }
        plans = {}
        results = {}
        for alias, endpoint in endpoints.items():
            endpoint_cache = cache["endpoints"][endpoint]
            plans[alias] = endpoint_cache["adaptive_plan"]
            results[alias] = endpoint_cache["adaptive"]
        spec, diagnostics = active_minimal_pair_spec(
            case_id,
            plans,
            results,
        )
        output["specs"][case_id] = spec
        output["diagnostics"][case_id] = diagnostics
    atomic_write_json(str(args.out), output)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
