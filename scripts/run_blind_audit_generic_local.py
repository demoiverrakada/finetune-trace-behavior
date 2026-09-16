"""Run the generic-only baseline with the clean base model as local investigator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.evidence import (  # noqa: E402
    materialize_case_evidence,
    private_case_by_id,
)
from blind_audit.investigator import (  # noqa: E402
    compose_case_spec,
    generic_endpoint_prompt,
    run_local_endpoint_audit,
)
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.spec import empty_profile  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_EVIDENCE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_endpoint_evidence.json"
)
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_generic_local.json"
)
DEFAULT_OUT = (
    ROOT / "blind_audit" / "results" / "stage1" / "generic_local_specs.json"
)


def endpoint_evidence(public, private, evidence_cache, endpoint):
    private_view = {
        **private,
        "cases": [
            {
                "case_id": "__endpoint__",
                "reference_endpoint": "base",
                "target_endpoint": endpoint,
            }
        ],
    }
    return materialize_case_evidence(
        public,
        private_view,
        evidence_cache,
        "__endpoint__",
        families=("generic_blackbox",),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device",
        default="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    evidence_cache = json.loads(args.evidence.read_text())
    identity = {
        "experiment": "blind_policy_recovery_stage1_generic_local",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "investigator": private["endpoint_registry"]["base"]["model"],
        "investigator_revision": private["endpoint_registry"]["base"][
            "revision"
        ],
        "prompt_version": 2,
        "max_new_tokens": 512,
        "device": args.device,
    }
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.cache),
        identity,
        {"endpoint_audits": {}},
    )

    def save():
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.cache), cache)

    cache["endpoint_audits"].setdefault(
        "base",
        {
            "confidence": 1.0,
            "summary": "Clean base control; no finetuned behavior is attributed.",
            "profile": empty_profile(),
            "evidence": [],
            "proposed_tests": [],
            "provider": {"kind": "fixed_base_control"},
        },
    )
    save()
    oracle = PrivateModelOracle(ROOT, private, device=args.device)
    for endpoint, record in sorted(private["endpoint_registry"].items()):
        if record["kind"] != "adapter":
            continue
        if endpoint in cache["endpoint_audits"]:
            continue
        evidence = endpoint_evidence(
            public,
            private,
            evidence_cache,
            endpoint,
        )
        cache["endpoint_audits"][endpoint] = run_local_endpoint_audit(
            oracle,
            generic_endpoint_prompt(evidence["records"]),
        )
        save()
        print(
            f"generic local endpoint audits: "
            f"{len(cache['endpoint_audits'])}/9",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1_generic_local_specs",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "method": "B1_generic_blackbox_local",
        "specs": {},
    }
    for public_case in public["cases"]:
        case_id = public_case["case_id"]
        private_case = private_case_by_id(private, case_id)
        output["specs"][case_id] = compose_case_spec(
            case_id,
            cache["endpoint_audits"][
                private_case["reference_endpoint"]
            ],
            cache["endpoint_audits"][private_case["target_endpoint"]],
        )
    eval_runtime.atomic_write_json(str(args.out), output)
    cache["status"] = "complete"
    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
