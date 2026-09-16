"""Extract Stage-1 ADL topic readouts and freeze executable case specs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.adl import (  # noqa: E402
    adl_texts_sha256,
    compose_adl_case_spec,
    extract_adl_readout,
)
from blind_audit.evidence import private_case_by_id  # noqa: E402
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_adl.json"
)
DEFAULT_OUT = ROOT / "blind_audit" / "results" / "stage1" / "adl_specs.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device",
        default="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    identity = {
        "experiment": "blind_policy_recovery_stage1_adl",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "texts_sha256": adl_texts_sha256(),
        "device": args.device,
        "layer_rule": "num_hidden_layers//2",
        "k": 5,
        "topk": 40,
    }
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.cache),
        identity,
        {"endpoint_readouts": {}},
    )

    def save():
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.cache), cache)

    oracle = PrivateModelOracle(ROOT, private, device=args.device)
    for endpoint in sorted(private["endpoint_registry"]):
        if endpoint in cache["endpoint_readouts"]:
            continue
        cache["endpoint_readouts"][endpoint] = extract_adl_readout(
            oracle,
            endpoint,
        )
        save()
        print(
            f"ADL endpoints: {len(cache['endpoint_readouts'])}/9",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1_adl_specs",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "method": "B4_ADL_class_informed_topic_readout",
        "specs": {},
        "diagnostics": {},
    }
    for public_case in public["cases"]:
        case_id = public_case["case_id"]
        private_case = private_case_by_id(private, case_id)
        spec, diagnostics = compose_adl_case_spec(
            case_id,
            cache["endpoint_readouts"][
                private_case["reference_endpoint"]
            ],
            cache["endpoint_readouts"][private_case["target_endpoint"]],
        )
        output["specs"][case_id] = spec
        output["diagnostics"][case_id] = diagnostics
    eval_runtime.atomic_write_json(str(args.out), output)
    cache["status"] = "complete"
    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
