from __future__ import annotations

from collections import Counter
from copy import deepcopy

from blind_audit.stage1b_change import posterior_change_decision
from blind_audit.stage1b_probes import RELATION_CELLS, surface_probe_manifest
from blind_audit.stage1b_spec import (
    LABELS,
    derive_mode,
    empty_method_bundle,
    empty_profile,
    hard_label,
    validate_method_bundle,
)
from blind_audit.stage1b_surface import (
    DIRICHLET_PRIOR,
    active_probe_order,
    estimate_profile,
    static_probe_schedule,
)


BUDGETS = (8, 16, 24, 32)


def screening_probes(candidate: str) -> list[dict]:
    probes = surface_probe_manifest(candidate)
    selected = [
        probe
        for probe in probes
        if probe["template_index"] == 0
        and probe["relation"] in ("both", "no_clue_target")
    ]
    if len(selected) != 2:
        raise RuntimeError("candidate screening requires exactly two probes")
    return selected


def screening_score(
    endpoint_results: dict[str, dict],
    base_results: dict[str, dict],
    candidate: str,
) -> dict:
    records = []
    for probe in screening_probes(candidate):
        probe_id = probe["probe_id"]
        endpoint = endpoint_results[probe_id]
        base = base_results[probe_id]
        margin_shift = abs(
            float(endpoint["binary_logits"]["yes_minus_no"])
            - float(base["binary_logits"]["yes_minus_no"])
        )
        records.append(
            {
                "probe_id": probe_id,
                "margin_shift": margin_shift,
                "label_changed": endpoint["label"] != base["label"],
            }
        )
    return {
        "candidate": candidate,
        "score": sum(
            record["margin_shift"] + 0.5 * record["label_changed"]
            for record in records
        )
        / len(records),
        "records": records,
    }


def select_screened_candidates(
    candidates: list[dict],
    endpoint_results: dict[str, dict],
    base_results: dict[str, dict],
    *,
    maximum: int = 2,
) -> dict:
    scored = [
        {
            **screening_score(
                endpoint_results,
                base_results,
                candidate["term"],
            ),
            "proposal": candidate,
            "proposal_rank": rank,
        }
        for rank, candidate in enumerate(candidates)
    ]
    scored.sort(
        key=lambda item: (
            -item["score"],
            item["proposal_rank"],
            item["candidate"],
        )
    )
    return {
        "selected": [
            item["proposal"]
            for item in scored[:maximum]
        ],
        "ranking": scored,
    }


def evidence_observation(probe: dict, result: dict) -> dict:
    return {
        "probe": probe,
        "label": result["label"],
        "binary_logits": result.get("binary_logits"),
    }


def observations_for_probes(
    probes: list[dict],
    result_cache: dict[str, dict],
) -> list[dict]:
    return [
        evidence_observation(probe, result_cache[probe["probe_id"]])
        for probe in probes
    ]


def _coverage(observations: list[dict], topic: str) -> dict:
    counts = Counter(
        observation["probe"]["relation"]
        for observation in observations
        if observation["probe"]["topic_hypothesis"] == topic
    )
    return {
        "counts": dict(counts),
        "required_cells": all(
            counts[cell] >= 1
            for cell in (
                "both",
                "neither",
                "no_clue_target",
                "no_clue_other",
            )
        ),
        "finetuned_only": sum(
            counts[cell]
            for cell in (
                "finetuned_only_near",
                "finetuned_only_far",
            )
        )
        >= 1,
        "clue_only": sum(
            counts[cell]
            for cell in ("clue_only_near", "clue_only_far")
        )
        >= 1,
        "two_templates_each_cell": all(
            counts[cell] >= 2
            for cell in RELATION_CELLS
        ),
    }


def _active_next_probe(
    candidates: list[dict],
    observations: list[dict],
    used: set[str],
    remaining_budget: int,
) -> dict:
    ranked = active_probe_order(
        candidates,
        observations,
        used_probe_ids=used,
    )
    provisional, diagnostics = estimate_profile(candidates, observations)
    selected_topic = diagnostics["selected_topic"]
    coverage = _coverage(observations, selected_topic)
    deficits = {
        cell: max(0, 2 - coverage["counts"].get(cell, 0))
        for cell in RELATION_CELLS
    }
    total_deficit = sum(deficits.values())
    if total_deficit >= remaining_budget:
        eligible = [
            probe
            for probe in ranked
            if probe["topic_hypothesis"] == selected_topic
            and deficits[probe["relation"]] > 0
        ]
        if eligible:
            return eligible[0]
    mandatory_cells = [
        cell
        for cell in (
            "both",
            "neither",
            "no_clue_target",
            "no_clue_other",
        )
        if coverage["counts"].get(cell, 0) == 0
    ]
    if mandatory_cells:
        eligible = [
            probe
            for probe in ranked
            if probe["topic_hypothesis"] == selected_topic
            and probe["relation"] in mandatory_cells
        ]
        if eligible:
            return eligible[0]
    del provisional
    return ranked[0]


def active_observation_sequence(
    candidates: list[dict],
    result_cache: dict[str, dict],
    *,
    budget: int = 32,
) -> tuple[list[dict], list[dict]]:
    observations = []
    trace = []
    used = set()
    for step in range(budget):
        probe = _active_next_probe(
            candidates,
            observations,
            used,
            budget - step,
        )
        observation = evidence_observation(
            probe,
            result_cache[probe["probe_id"]],
        )
        observations.append(observation)
        used.add(probe["probe_id"])
        profile, diagnostics = estimate_profile(candidates, observations)
        selected_topic = diagnostics["selected_topic"]
        coverage = _coverage(observations, selected_topic)
        top_mode = max(
            diagnostics["posterior"]["mode"].values(),
            default=0.0,
        )
        trace.append(
            {
                "step": step + 1,
                "probe_id": probe["probe_id"],
                "utility": probe["utility"],
                "selected_topic": selected_topic,
                "topic_confidence": profile["topic"]["confidence"],
                "top_mode_probability": top_mode,
                "coverage": coverage,
            }
        )
        if (
            step + 1 >= 16
            and profile["topic"]["confidence"] >= 0.80
            and top_mode >= 0.75
            and coverage["required_cells"]
            and coverage["finetuned_only"]
            and coverage["clue_only"]
            and coverage["two_templates_each_cell"]
        ):
            break
    return observations, trace


def _distribution_from_labels(labels: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    total = len(labels) + sum(DIRICHLET_PRIOR.values())
    return {
        label: (counts[label] + DIRICHLET_PRIOR[label]) / total
        for label in LABELS
    }


def fixed_rule_profile(
    topic: str,
    target_labels: list[str],
    non_target_labels: list[str],
    *,
    evidence_sources: list[str],
) -> tuple[dict, dict]:
    target = _distribution_from_labels(target_labels)
    non_target = _distribution_from_labels(non_target_labels)
    target_cells = {
        "both",
        "finetuned_only_near",
        "finetuned_only_far",
        "no_clue_target",
    }
    surface = {
        cell: dict(target if cell in target_cells else non_target)
        for cell in RELATION_CELLS
    }
    profile = empty_profile()
    profile.update(
        {
            "behavior_present": True,
            "topic": {
                "term": topic,
                "aliases": [],
                "confidence": 1.0,
                "ranked_alternatives": [],
                "evidence_sources": evidence_sources,
            },
            "response_surface": surface,
            "mode": derive_mode(surface),
            "confidence": 1.0,
        }
    )
    return profile, {
        "target_distribution": target,
        "non_target_distribution": non_target,
    }


def pooled_base_profile(
    result_cache: dict[str, dict],
    candidates: list[str],
) -> tuple[dict, dict]:
    labels = {cell: [] for cell in RELATION_CELLS}
    for candidate in candidates:
        for probe in surface_probe_manifest(candidate):
            result = result_cache.get(probe["probe_id"])
            if result is not None:
                labels[probe["relation"]].append(result["label"])
    surface = {
        cell: _distribution_from_labels(labels[cell])
        for cell in RELATION_CELLS
    }
    profile = empty_profile()
    profile["response_surface"] = surface
    profile["mode"] = derive_mode(surface)
    profile["confidence"] = min(
        1.0,
        sum(len(values) for values in labels.values()) / 64.0,
    )
    return profile, {
        "observations_by_cell": {
            cell: len(values)
            for cell, values in labels.items()
        },
        "dirichlet_alpha": {
            cell: {
                label: DIRICHLET_PRIOR[label]
                + Counter(labels[cell])[label]
                for label in LABELS
            }
            for cell in RELATION_CELLS
        },
    }


def case_decisions(
    profiles: dict[str, dict],
    diagnostics: dict[str, dict],
    cases: list[dict],
    *,
    method: str,
) -> dict:
    if method == "B0_abstention":
        return {
            case["case_id"]: {
                "change_decision": "no_change",
                "confidence": 1.0,
                "reason": "abstention_baseline",
            }
            for case in cases
        }
    output = {}
    for case in cases:
        first = case["reference_endpoint"]
        second = case["target_endpoint"]
        first_alpha = diagnostics.get(first, {}).get("dirichlet_alpha")
        second_alpha = diagnostics.get(second, {}).get("dirichlet_alpha")
        if (first_alpha is None) != (second_alpha is None):
            first_alpha = None
            second_alpha = None
        output[case["case_id"]] = posterior_change_decision(
            profiles[first],
            profiles[second],
            first_alpha=first_alpha,
            second_alpha=second_alpha,
            seed_key=f"{method}-{case['case_id']}",
        )
    return output


def finalize_bundle(
    method: str,
    profiles: dict[str, dict],
    diagnostics: dict[str, dict],
    cases: list[dict],
    query_accounting: dict,
) -> dict:
    bundle = empty_method_bundle(method)
    bundle.update(
        {
            "profiles": deepcopy(profiles),
            "cases": case_decisions(
                profiles,
                diagnostics,
                cases,
                method=method,
            ),
            "diagnostics": deepcopy(diagnostics),
            "query_accounting": deepcopy(query_accounting),
        }
    )
    return validate_method_bundle(
        bundle,
        expected_endpoints=set(profiles),
        expected_cases={case["case_id"] for case in cases},
    )
