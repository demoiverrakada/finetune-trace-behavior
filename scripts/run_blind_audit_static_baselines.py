"""Freeze Stage-1 static audit specifications from discovery evidence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.baselines import (  # noqa: E402
    abstention_spec,
    generation_only_spec,
    logit_difference_spec,
)
from blind_audit.evidence import materialize_case_evidence  # noqa: E402
from harness.eval_runtime import atomic_write_json, utc_now  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_endpoint_evidence.json"
)
DEFAULT_OUT = ROOT / "blind_audit" / "results" / "stage1" / "static_specs.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    cache = json.loads(args.cache.read_text())
    results = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1_static_baselines",
        "created_at": utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "methods": {
            "B0_abstention": {},
            "B2_generation_class_informed": {},
            "B3_logit_class_informed": {},
        },
        "diagnostics": {
            "B2_generation_class_informed": {},
            "B3_logit_class_informed": {},
        },
    }
    for public_case in public["cases"]:
        case_id = public_case["case_id"]
        evidence = materialize_case_evidence(
            public,
            private,
            cache,
            case_id,
            families=(
                "class_informed_game",
                "class_informed_candidate_grid",
            ),
        )
        generation_spec, generation_diagnostics = generation_only_spec(evidence)
        logit_spec, logit_diagnostics = logit_difference_spec(evidence)
        results["methods"]["B0_abstention"][case_id] = abstention_spec(case_id)
        results["methods"]["B2_generation_class_informed"][
            case_id
        ] = generation_spec
        results["methods"]["B3_logit_class_informed"][case_id] = logit_spec
        results["diagnostics"]["B2_generation_class_informed"][
            case_id
        ] = generation_diagnostics
        results["diagnostics"]["B3_logit_class_informed"][
            case_id
        ] = logit_diagnostics
    atomic_write_json(str(args.out), results)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
