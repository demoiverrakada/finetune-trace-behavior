"""Score frozen Stage-1 audit specifications on private held-out observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.evidence import (  # noqa: E402
    hidden_case_observations,
    private_case_by_id,
)
from blind_audit.scoring import aggregate_case_scores, score_case  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
DEFAULT_EVIDENCE = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_endpoint_evidence.json"
)
DEFAULT_STATIC = (
    ROOT / "blind_audit" / "results" / "stage1" / "static_specs.json"
)
DEFAULT_GENERIC = (
    ROOT / "blind_audit" / "results" / "stage1" / "generic_llm_specs.json"
)
DEFAULT_GENERIC_ACTIVE = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1"
    / "generic_active_specs.json"
)
DEFAULT_GENERIC_LOCAL = (
    ROOT / "blind_audit" / "results" / "stage1" / "generic_local_specs.json"
)
DEFAULT_GENERIC_ACTIVE_LOCAL = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1"
    / "generic_active_local_specs.json"
)
DEFAULT_ACTIVE = (
    ROOT / "blind_audit" / "results" / "stage1" / "active_specs.json"
)
DEFAULT_ADL = ROOT / "blind_audit" / "results" / "stage1" / "adl_specs.json"
DEFAULT_OUT = ROOT / "blind_audit" / "results" / "stage1" / "scores.json"
DEFAULT_REPORT = ROOT / "blind_audit" / "results" / "stage1" / "REPORT.md"


def _load_methods(
    static_path: Path,
    generic_path: Path,
    generic_active_path: Path,
    generic_local_path: Path,
    generic_active_local_path: Path,
    active_path: Path,
    adl_path: Path,
) -> dict[str, dict]:
    methods = {}
    static = json.loads(static_path.read_text())
    methods.update(static["methods"])
    if generic_path.exists():
        generic = json.loads(generic_path.read_text())
        methods[generic["method"]] = generic["specs"]
    if generic_active_path.exists():
        generic_active = json.loads(generic_active_path.read_text())
        methods[generic_active["method"]] = generic_active["specs"]
    if generic_local_path.exists():
        generic_local = json.loads(generic_local_path.read_text())
        methods[generic_local["method"]] = generic_local["specs"]
    if generic_active_local_path.exists():
        generic_active_local = json.loads(
            generic_active_local_path.read_text()
        )
        methods[generic_active_local["method"]] = generic_active_local["specs"]
    if active_path.exists():
        active = json.loads(active_path.read_text())
        methods[active["method"]] = active["specs"]
    if adl_path.exists():
        adl = json.loads(adl_path.read_text())
        methods[adl["method"]] = adl["specs"]
    return methods


def _family_metric(aggregate: dict, family: str, field: str) -> float:
    return aggregate["families"].get(family, {}).get(field, 0.0)


def automated_decision(method_results: dict[str, dict]) -> dict:
    active_name = "M1_active_class_informed"
    active = method_results.get(active_name, {}).get("aggregate")
    if active is None:
        return {
            "available": False,
            "reason": f"{active_name} was not scored",
        }
    static_names = (
        "B2_generation_class_informed",
        "B3_logit_class_informed",
    )
    available_static = [
        method_results[name]["aggregate"]
        for name in static_names
        if name in method_results
    ]
    best_static = max(
        (
            result["mean_predictive_fidelity"]
            for result in available_static
        ),
        default=0.0,
    )
    improvement = active["mean_predictive_fidelity"] - best_static
    checks = {
        "overall_predictive_fidelity_at_least_0_75": (
            active["mean_predictive_fidelity"] >= 0.75
        ),
        "base_to_finetune_fidelity_at_least_0_70": (
            _family_metric(
                active,
                "base_to_finetune",
                "mean_predictive_fidelity",
            )
            >= 0.70
        ),
        "policy_contrast_fidelity_at_least_0_70": (
            _family_metric(
                active,
                "policy_contrast",
                "mean_predictive_fidelity",
            )
            >= 0.70
        ),
        "structural_score_at_least_0_70": (
            active["mean_structural_score"] >= 0.70
        ),
        "null_false_positive_rate_at_most_0_25": (
            active["null_false_positive_rate"] <= 0.25
        ),
        "beats_best_matched_static_by_at_least_0_10": (
            improvement >= 0.10
        ),
    }
    return {
        "available": True,
        "active_method": active_name,
        "best_static_predictive_fidelity": best_static,
        "active_predictive_fidelity": active[
            "mean_predictive_fidelity"
        ],
        "predictive_fidelity_improvement": improvement,
        "automated_checks": checks,
        "automated_checks_passed": all(checks.values()),
        "manual_checks_pending": [
            "no private-manifest/path/training/evaluator leakage",
            "selective causal validation without wrong-guess damage",
        ],
        "stage1_complete": False,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Blind policy recovery — Stage 1 score report",
        "",
        f"Generated: {payload['created_at']}",
        "",
        "## Aggregate methods",
        "",
        "| Method | Predictive fidelity | Structural | Change accuracy | Null FPR | Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, result in payload["methods"].items():
        aggregate = result["aggregate"]
        lines.append(
            "| {method} | {predictive:.3f} | {structural:.3f} | "
            "{change:.3f} | {null_fpr:.3f} | {composite:.3f} |".format(
                method=method,
                predictive=aggregate["mean_predictive_fidelity"],
                structural=aggregate["mean_structural_score"],
                change=aggregate["change_detection_accuracy"],
                null_fpr=aggregate["null_false_positive_rate"],
                composite=aggregate["mean_composite"],
            )
        )
    lines.extend(
        (
            "",
            "## Automated decision checks",
            "",
        )
    )
    decision = payload["decision"]
    if not decision["available"]:
        lines.append(f"Unavailable: {decision['reason']}")
    else:
        for name, passed in decision["automated_checks"].items():
            lines.append(f"- [{'x' if passed else ' '}] `{name}`")
        lines.extend(
            (
                "",
                "Manual checks still pending:",
                "",
            )
        )
        for item in decision["manual_checks_pending"]:
            lines.append(f"- {item}")
    lines.extend(
        (
            "",
            "This report is generated from frozen executable specifications and "
            "private held-out model responses. It does not use textual-similarity "
            "grading.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--generic", type=Path, default=DEFAULT_GENERIC)
    parser.add_argument(
        "--generic-active",
        type=Path,
        default=DEFAULT_GENERIC_ACTIVE,
    )
    parser.add_argument(
        "--generic-local",
        type=Path,
        default=DEFAULT_GENERIC_LOCAL,
    )
    parser.add_argument(
        "--generic-active-local",
        type=Path,
        default=DEFAULT_GENERIC_ACTIVE_LOCAL,
    )
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--adl", type=Path, default=DEFAULT_ADL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    evidence = json.loads(args.evidence.read_text())
    methods = _load_methods(
        args.static,
        args.generic,
        args.generic_active,
        args.generic_local,
        args.generic_active_local,
        args.active,
        args.adl,
    )
    payload = {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1_scores",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "methods": {},
    }
    tasks = private["hidden_evaluation_tasks"]
    for method_name, specs in methods.items():
        case_scores = []
        for public_case in public["cases"]:
            case_id = public_case["case_id"]
            truth_case = private_case_by_id(private, case_id)
            observations = hidden_case_observations(
                private,
                evidence,
                case_id,
            )
            case_scores.append(
                score_case(
                    specs[case_id],
                    truth_case,
                    tasks,
                    observations,
                )
            )
        payload["methods"][method_name] = {
            "aggregate": aggregate_case_scores(case_scores),
            "cases": {
                score["case_id"]: score
                for score in case_scores
            },
        }
    payload["decision"] = automated_decision(payload["methods"])
    eval_runtime.atomic_write_json(str(args.out), payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload))
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
