"""Execute generic LLM-proposed tests and freeze final open-schema specs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.evidence import private_case_by_id  # noqa: E402
from blind_audit.investigator import (  # noqa: E402
    compose_case_spec,
    generic_active_endpoint_prompt,
    run_claude_endpoint_audit,
)
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_INITIAL = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_generic_llm.json"
)
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_generic_active.json"
)
DEFAULT_OUT = (
    ROOT / "blind_audit" / "results" / "stage1" / "generic_active_specs.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--max-budget-usd", type=float, default=0.75)
    parser.add_argument(
        "--device",
        default="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    initial = json.loads(args.initial.read_text())
    if initial.get("status") != "complete":
        raise RuntimeError("generic static LLM audit is not complete")
    identity = {
        "experiment": "blind_policy_recovery_stage1_generic_active",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "initial_identity": initial["identity"],
        "model": args.model,
        "effort": args.effort,
        "max_budget_usd": args.max_budget_usd,
        "device": args.device,
        "prompt_version": 1,
    }
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.cache),
        identity,
        {"endpoint_tests": {}, "endpoint_final_audits": {}},
    )

    def save():
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.cache), cache)

    oracle = None
    for endpoint, record in sorted(private["endpoint_registry"].items()):
        if record["kind"] != "adapter":
            continue
        initial_audit = initial["endpoint_audits"][endpoint]
        tests = list(dict.fromkeys(initial_audit["proposed_tests"]))[:12]
        endpoint_tests = cache["endpoint_tests"].setdefault(endpoint, {})
        pending = [
            (index, prompt)
            for index, prompt in enumerate(tests)
            if f"test_{index + 1:02d}" not in endpoint_tests
        ]
        if pending and oracle is None:
            oracle = PrivateModelOracle(ROOT, private, device=args.device)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            conversations = [
                [{"role": "user", "content": prompt}]
                for _, prompt in batch
            ]
            base_results = oracle.generate_batch(
                "base",
                conversations,
                max_new_tokens=96,
            )
            variant_results = oracle.generate_batch(
                endpoint,
                conversations,
                max_new_tokens=96,
            )
            for (index, prompt), base, variant in zip(
                batch,
                base_results,
                variant_results,
            ):
                endpoint_tests[f"test_{index + 1:02d}"] = {
                    "prompt": prompt,
                    "base_response": base["response"],
                    "variant_response": variant["response"],
                }
            save()
            print(
                f"{endpoint} generic active tests: "
                f"{len(endpoint_tests)}/{len(tests)}",
                flush=True,
            )

        if endpoint not in cache["endpoint_final_audits"]:
            ordered_records = [
                endpoint_tests[f"test_{index + 1:02d}"]
                for index in range(len(tests))
            ]
            prompt = generic_active_endpoint_prompt(
                initial_audit,
                ordered_records,
            )
            cache["endpoint_final_audits"][
                endpoint
            ] = run_claude_endpoint_audit(
                prompt,
                model=args.model,
                effort=args.effort,
                max_budget_usd=args.max_budget_usd,
            )
            save()
            print(
                f"generic active endpoint audits: "
                f"{len(cache['endpoint_final_audits'])}/8",
                flush=True,
            )

    base_audit = initial["endpoint_audits"]["base"]
    output = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1_generic_active_specs",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "method": "M1_active_generic_llm",
        "specs": {},
    }
    endpoint_audits = {
        "base": base_audit,
        **cache["endpoint_final_audits"],
    }
    for public_case in public["cases"]:
        case_id = public_case["case_id"]
        private_case = private_case_by_id(private, case_id)
        output["specs"][case_id] = compose_case_spec(
            case_id,
            endpoint_audits[private_case["reference_endpoint"]],
            endpoint_audits[private_case["target_endpoint"]],
        )
    eval_runtime.atomic_write_json(str(args.out), output)
    cache["status"] = "complete"
    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
