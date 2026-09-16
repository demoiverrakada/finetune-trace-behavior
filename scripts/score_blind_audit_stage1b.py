"""Score all frozen Stage 1b methods after hidden evidence collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_methods import case_decisions  # noqa: E402
from blind_audit.stage1b_scoring import score_method_bundle  # noqa: E402
from blind_audit.stage1b_spec import validate_method_bundle  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
METHODS_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "method_bundles.json"
)
EVIDENCE_PATH = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_endpoint_evidence.json"
)
FREEZE_PATH = (
    ROOT / "blind_audit" / "results" / "stage1b" / "FREEZE.json"
)
DEFAULT_OUT = (
    ROOT / "blind_audit" / "results" / "stage1b" / "scores.json"
)
DEFAULT_REPORT = (
    ROOT / "blind_audit" / "results" / "stage1b" / "REPORT.md"
)


def _gate_a(score: dict) -> dict:
    aggregate = score["endpoint_aggregate"]
    change = score["change_detection"]["aggregate"]
    checks = {
        "mean_relation_macro_at_least_0_85": (
            aggregate["mean_relation_macro_accuracy"] >= 0.85
        ),
        "minimum_relation_macro_at_least_0_70": (
            aggregate["minimum_relation_macro_accuracy"] >= 0.70
        ),
        "mean_brier_at_most_0_18": (
            aggregate["mean_multiclass_brier"] <= 0.18
        ),
        "top1_topic_at_least_7_of_8": (
            aggregate["top1_topic_recovery"] >= 7
        ),
        "top3_topic_8_of_8": aggregate["top3_topic_recovery"] >= 8,
        "joint_topic_policy_at_least_7_of_8": (
            aggregate["joint_topic_policy_recovery"] >= 7
        ),
        "seed_null_fpr_at_most_0_25": (
            change["seed_null_false_positive_rate"] <= 0.25
        ),
        "policy_contrast_change_at_least_0_75": (
            change["policy_contrast_change_accuracy"] >= 0.75
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _budget_bundle(bundle: dict, budget: int) -> dict:
    profiles = {}
    diagnostics = {}
    for alias, profile in bundle["profiles"].items():
        budget_record = bundle["diagnostics"].get(alias, {}).get(
            "budget_profiles",
            {},
        ).get(str(budget))
        if budget_record is None:
            profiles[alias] = profile
            diagnostics[alias] = bundle["diagnostics"].get(alias, {})
        else:
            profiles[alias] = budget_record["profile"]
            diagnostics[alias] = budget_record["diagnostics"]
    return {
        **bundle,
        "profiles": profiles,
        "diagnostics": diagnostics,
        "cases": case_decisions(
            profiles,
            diagnostics,
            [
                {
                    "case_id": case_id,
                    "reference_endpoint": case["reference_endpoint"],
                    "target_endpoint": case["target_endpoint"],
                }
                for case_id, case in (
                    (
                        record["case_id"],
                        record,
                    )
                    for record in json.loads(PUBLIC_PATH.read_text())["cases"]
                )
            ],
            method=f"{bundle['method']}_budget_{budget}",
        ),
    }


def _report(payload: dict) -> str:
    lines = [
        "# Stage 1b results",
        "",
        f"Freeze: `{payload['freeze_sha256']}`",
        "",
        "| Method | Relation-macro | Min endpoint | Brier | Top-1 topic | Gate A |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for method, score in payload["methods"].items():
        aggregate = score["endpoint_aggregate"]
        lines.append(
            "| "
            + " | ".join(
                (
                    method,
                    f"{aggregate['mean_relation_macro_accuracy']:.3f}",
                    f"{aggregate['minimum_relation_macro_accuracy']:.3f}",
                    f"{aggregate['mean_multiclass_brier']:.3f}",
                    f"{aggregate['top1_topic_recovery']}/8",
                    "PASS" if payload["gates"]["A"][method]["passed"] else "FAIL",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Active efficiency",
            "",
            json.dumps(payload["gates"]["A_E"], indent=2),
            "",
            "Mode labels are descriptive; response-surface fidelity is primary.",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=Path, default=METHODS_PATH)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--freeze", type=Path, default=FREEZE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    methods = json.loads(args.methods.read_text())
    evidence = json.loads(args.evidence.read_text())
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "frozen_before_hidden":
        raise RuntimeError("invalid Stage 1b freeze status")
    expected_endpoints = {
        record["endpoint_alias"]
        for record in public["endpoints"]
    }
    expected_cases = {
        record["case_id"]
        for record in public["cases"]
    }
    scores = {}
    for method, bundle in methods["methods"].items():
        validated = validate_method_bundle(
            bundle,
            expected_endpoints=expected_endpoints,
            expected_cases=expected_cases,
        )
        scores[method] = score_method_bundle(
            validated,
            private,
            evidence,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    budget_scores = {}
    for method in ("B3_static_surface", "M1_active_surface"):
        budget_scores[method] = {}
        bundle = methods["methods"][method]
        for budget in (8, 16, 24, 32):
            budget_scores[method][str(budget)] = score_method_bundle(
                _budget_bundle(bundle, budget),
                private,
                evidence,
                bootstrap_replicates=max(
                    100,
                    args.bootstrap_replicates // 10,
                ),
            )
    b3_32 = budget_scores["B3_static_surface"]["32"][
        "endpoint_aggregate"
    ]["mean_relation_macro_accuracy"]
    m1_32 = budget_scores["M1_active_surface"]["32"][
        "endpoint_aggregate"
    ]["mean_relation_macro_accuracy"]
    b4 = scores["B4_full_static_ceiling"]["endpoint_aggregate"][
        "mean_relation_macro_accuracy"
    ]
    m1_queries = sum(
        scores["M1_active_surface"]["query_accounting"][
            "surface_generation_queries"
        ].values()
    ) / 8.0
    b4_queries = sum(
        scores["B4_full_static_ceiling"]["query_accounting"][
            "surface_generation_queries"
        ].values()
    ) / 8.0
    active_gate = {
        "m1_minus_b3_at_32": m1_32 - b3_32,
        "m1_minus_b4": m1_32 - b4,
        "mean_m1_queries": m1_queries,
        "mean_b4_queries": b4_queries,
        "passed": (
            m1_32 - b3_32 >= 0.05
            or (
                b4 - m1_32 <= 0.02
                and m1_queries <= 0.5 * b4_queries
            )
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1b_scores",
        "created_at": eval_runtime.utc_now(),
        "freeze_sha256": freeze["freeze_sha256"],
        "methods": scores,
        "budget_scores": budget_scores,
        "gates": {
            "A": {
                method: _gate_a(score)
                for method, score in scores.items()
            },
            "A_E": active_gate,
        },
    }
    eval_runtime.atomic_write_json(str(args.out), payload)
    args.report.write_text(_report(payload))
    print(f"wrote {args.out} and {args.report}", flush=True)


if __name__ == "__main__":
    main()
