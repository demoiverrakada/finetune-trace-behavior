"""Create the immutable pre-hidden Stage 1b freeze record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_spec import validate_method_bundle  # noqa: E402
from blind_audit.stage1b_torch import (  # noqa: E402
    canonical_endpoint_runtime_status,
)
from blind_audit.utils import compact_json_sha256, file_sha256  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
CALIBRATION_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "synthetic_calibration.json"
)
METHODS_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "method_bundles.json"
)
REFERENCE_PATH = (
    ROOT
    / "blind_audit"
    / "reference"
    / "stage1b"
    / "manifest.json"
)
INTERPRETER_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "interpreter_selection.json"
)
PARITY_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "mlx_parity.json"
)
PROPOSAL_PATHS = (
    ROOT / "blind_audit/results/stage1b/blackbox_proposals.json",
    ROOT / "blind_audit/results/stage1b/perplexity_proposals.json",
    ROOT / "blind_audit/results/stage1b/diff_mining_proposals.json",
    ROOT / "blind_audit/results/stage1b/contrastive_proposals.json",
    ROOT / "blind_audit/results/stage1b/adl_proposals.json",
    ROOT / "blind_audit/results/stage1b/fused_proposals.json",
)
DEFAULT_OUT = (
    ROOT / "blind_audit" / "results" / "stage1b" / "FREEZE.json"
)


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    calibration = json.loads(CALIBRATION_PATH.read_text())
    methods = json.loads(METHODS_PATH.read_text())
    parity = json.loads(PARITY_PATH.read_text())
    if not calibration.get("checks_passed"):
        raise RuntimeError("synthetic calibration did not pass")
    runtime_status = canonical_endpoint_runtime_status(parity)
    expected_endpoints = {
        record["endpoint_alias"]
        for record in public["endpoints"]
    }
    expected_cases = {
        record["case_id"]
        for record in public["cases"]
    }
    if not methods.get("methods"):
        raise RuntimeError("method bundle file contains no methods")
    for method, bundle in methods["methods"].items():
        if method != bundle.get("method"):
            raise RuntimeError(f"method key mismatch for {method}")
        validate_method_bundle(
            bundle,
            expected_endpoints=expected_endpoints,
            expected_cases=expected_cases,
        )
    required = (
        PUBLIC_PATH,
        PRIVATE_PATH,
        CALIBRATION_PATH,
        METHODS_PATH,
        REFERENCE_PATH,
        INTERPRETER_PATH,
        PARITY_PATH,
        *PROPOSAL_PATHS,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"freeze inputs are missing: {missing}")
    method_files = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in required
    }
    freeze = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1b",
        "status": "frozen_before_hidden",
        "created_at": eval_runtime.utc_now(),
        "git_revision": _git_revision(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "reference_manifest_sha256": json.loads(
            REFERENCE_PATH.read_text()
        )["manifest_sha256"],
        "calibration_checks": calibration["checks"],
        "interpreter": json.loads(INTERPRETER_PATH.read_text())[
            "selected"
        ],
        "runtime_status": runtime_status,
        "method_names": sorted(methods["methods"]),
        "method_files": method_files,
        "implementation_sha256": public["implementation_sha256"],
        "rules": [
            "No confirmatory method or threshold may change after this record.",
            "Hidden prompt text is not supplied to any method.",
            "Any later revision must use an explicitly exploratory filename.",
        ],
    }
    freeze["freeze_sha256"] = compact_json_sha256(freeze)
    eval_runtime.atomic_write_json(str(args.out), freeze)
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    main()
