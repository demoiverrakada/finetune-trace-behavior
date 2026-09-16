"""Freeze B0-B4 and M1 Stage 1b endpoint profiles before hidden generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_methods import (  # noqa: E402
    BUDGETS,
    active_observation_sequence,
    finalize_bundle,
    fixed_rule_profile,
    observations_for_probes,
    pooled_base_profile,
)
from blind_audit.stage1b_probes import RELATION_CELLS, surface_probe_manifest  # noqa: E402
from blind_audit.stage1b_spec import empty_profile  # noqa: E402
from blind_audit.stage1b_surface import (  # noqa: E402
    estimate_profile,
    static_probe_schedule,
)
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
SURFACE_PATH = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_surface_evidence.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "method_bundles.json"
)


METHODS = (
    "B0_abstention",
    "B1_legacy_logit_fixed_rule",
    "B2_semantic_fixed_rule",
    "B3_static_surface",
    "B4_full_static_ceiling",
    "M1_active_surface",
)


def _candidate(term: str, candidates: list[dict]) -> dict:
    for candidate in candidates:
        if candidate["term"] == term:
            return candidate
    return {
        "term": term,
        "score": 1e-6,
        "evidence_sources": [],
    }


def _template_zero_labels(
    term: str,
    results: dict[str, dict],
) -> tuple[list[str], list[str]]:
    probes = [
        probe
        for probe in surface_probe_manifest(term)
        if probe["template_index"] == 0
    ]
    target = [
        results[probe["probe_id"]]["label"]
        for probe in probes
        if probe["relation"] in ("both", "no_clue_target")
    ]
    non_target = [
        results[probe["probe_id"]]["label"]
        for probe in probes
        if probe["relation"] not in ("both", "no_clue_target")
    ]
    return target, non_target


def _unique_surface_terms(surface: dict) -> list[str]:
    return sorted(
        {
            term
            for endpoint in surface["endpoints"].values()
            for term in endpoint["full_surface_terms"]
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=Path, default=FUSION_PATH)
    parser.add_argument("--surface", type=Path, default=SURFACE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    fusion = json.loads(args.fusion.read_text())
    surface = json.loads(args.surface.read_text())
    if surface.get("status") != "complete":
        raise RuntimeError("surface evidence must be complete")
    endpoint_aliases = [
        record["endpoint_alias"]
        for record in public["endpoints"]
    ]
    cases = public["cases"]
    profiles = {
        method: {}
        for method in METHODS
    }
    diagnostics = {
        method: {}
        for method in METHODS
    }
    query_accounting = {
        method: {
            "surface_generation_queries": {},
            "shared_proposal_evidence": method not in (
                "B0_abstention",
                "B1_legacy_logit_fixed_rule",
            ),
        }
        for method in METHODS
    }
    all_terms = _unique_surface_terms(surface)
    base_profile, base_diagnostics = pooled_base_profile(
        surface["base"],
        all_terms,
    )
    for method in METHODS:
        if method == "B0_abstention":
            profiles[method]["base"] = empty_profile()
            diagnostics[method]["base"] = {}
        else:
            profiles[method]["base"] = base_profile
            diagnostics[method]["base"] = base_diagnostics
        query_accounting[method]["surface_generation_queries"]["base"] = (
            0
            if method == "B0_abstention"
            else len(surface["base"])
        )

    for alias in endpoint_aliases:
        if alias == "base":
            continue
        endpoint = surface["endpoints"][alias]
        primary = fusion["endpoints"][alias]["primary"]
        fused_candidates = primary["ranked_candidates"]
        selected = endpoint["screening"]["selected"]
        results = endpoint["results"]

        profiles["B0_abstention"][alias] = empty_profile()
        diagnostics["B0_abstention"][alias] = {}
        query_accounting["B0_abstention"][
            "surface_generation_queries"
        ][alias] = 0

        legacy_term = selected[0]["term"]
        target, non_target = _template_zero_labels(
            legacy_term,
            results,
        )
        legacy_profile, legacy_diagnostics = fixed_rule_profile(
            legacy_term,
            target,
            non_target,
            evidence_sources=["screening_logit_anomaly"],
        )
        profiles["B1_legacy_logit_fixed_rule"][alias] = legacy_profile
        diagnostics["B1_legacy_logit_fixed_rule"][
            alias
        ] = legacy_diagnostics
        query_accounting["B1_legacy_logit_fixed_rule"][
            "surface_generation_queries"
        ][alias] = 8

        semantic_term = endpoint["semantic_candidate"]
        target, non_target = _template_zero_labels(
            semantic_term,
            results,
        )
        semantic_profile, semantic_diagnostics = fixed_rule_profile(
            semantic_term,
            target,
            non_target,
            evidence_sources=["semantic_self_report"],
        )
        profiles["B2_semantic_fixed_rule"][alias] = semantic_profile
        diagnostics["B2_semantic_fixed_rule"][
            alias
        ] = semantic_diagnostics
        query_accounting["B2_semantic_fixed_rule"][
            "surface_generation_queries"
        ][alias] = 8

        static_candidates = [
            _candidate(candidate["term"], fused_candidates)
            for candidate in selected
        ]
        static_schedule = static_probe_schedule(
            static_candidates,
            budget=32,
        )
        static_observations = observations_for_probes(
            static_schedule,
            results,
        )
        static_profile, static_diagnostics = estimate_profile(
            static_candidates,
            static_observations,
        )
        static_diagnostics["budget_profiles"] = {}
        for budget in BUDGETS:
            budget_profile, budget_diagnostics = estimate_profile(
                static_candidates,
                static_observations[:budget],
            )
            static_diagnostics["budget_profiles"][str(budget)] = {
                "profile": budget_profile,
                "diagnostics": budget_diagnostics,
            }
        profiles["B3_static_surface"][alias] = static_profile
        diagnostics["B3_static_surface"][alias] = static_diagnostics
        query_accounting["B3_static_surface"][
            "surface_generation_queries"
        ][alias] = len(static_observations)

        full_probes = [
            probe
            for candidate in static_candidates[:2]
            for probe in surface_probe_manifest(candidate["term"])
        ]
        full_observations = observations_for_probes(
            full_probes,
            results,
        )
        full_profile, full_diagnostics = estimate_profile(
            static_candidates,
            full_observations,
        )
        profiles["B4_full_static_ceiling"][alias] = full_profile
        diagnostics["B4_full_static_ceiling"][alias] = full_diagnostics
        query_accounting["B4_full_static_ceiling"][
            "surface_generation_queries"
        ][alias] = len(full_observations)

        active_observations, active_trace = active_observation_sequence(
            static_candidates,
            results,
            budget=32,
        )
        active_profile, active_diagnostics = estimate_profile(
            static_candidates,
            active_observations,
        )
        active_diagnostics["selection_trace"] = active_trace
        active_diagnostics["budget_profiles"] = {}
        for budget in BUDGETS:
            count = min(budget, len(active_observations))
            budget_profile, budget_diagnostics = estimate_profile(
                static_candidates,
                active_observations[:count],
            )
            active_diagnostics["budget_profiles"][str(budget)] = {
                "actual_queries": count,
                "profile": budget_profile,
                "diagnostics": budget_diagnostics,
            }
        profiles["M1_active_surface"][alias] = active_profile
        diagnostics["M1_active_surface"][alias] = active_diagnostics
        query_accounting["M1_active_surface"][
            "surface_generation_queries"
        ][alias] = len(active_observations)

    bundles = {
        method: finalize_bundle(
            method,
            profiles[method],
            diagnostics[method],
            cases,
            query_accounting[method],
        )
        for method in METHODS
    }
    output = {
        "schema_version": 1,
        "experiment": "stage1b_frozen_method_bundles",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "methods": bundles,
    }
    eval_runtime.atomic_write_json(str(args.out), output)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
