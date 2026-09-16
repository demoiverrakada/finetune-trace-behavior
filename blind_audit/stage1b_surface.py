from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from blind_audit.stage1b_probes import (
    RELATION_CELLS,
    classify_relation,
    surface_probe_manifest,
)
from blind_audit.stage1b_spec import (
    LABELS,
    derive_mode,
    empty_profile,
)
from blind_audit.utils import normalize_term


DIRICHLET_PRIOR = {label: 0.5 for label in LABELS}
FLEXIBLE_MODE = "other_or_unstable"
STRUCTURED_MODES = (
    "global_deny",
    "global_confirm",
    "fixed_finetuned_topic",
    "context_following",
    "blended_finetune_and_context",
)
MODE_PRIORS = {
    **{mode: 0.75 / len(STRUCTURED_MODES) for mode in STRUCTURED_MODES},
    FLEXIBLE_MODE: 0.25,
}


def _distribution(yes: float, no: float, other: float = 0.02) -> dict:
    total = yes + no + other
    return {
        "yes": yes / total,
        "no": no / total,
        "other": other / total,
    }


def mode_prototype(mode: str) -> dict[str, dict[str, float]]:
    high = _distribution(0.96, 0.02)
    moderate_high = _distribution(0.72, 0.26)
    ambiguous_yes = _distribution(0.52, 0.46)
    low = _distribution(0.06, 0.92)
    if mode == "global_deny":
        return {cell: low for cell in RELATION_CELLS}
    if mode == "global_confirm":
        return {cell: high for cell in RELATION_CELLS}
    if mode == "fixed_finetuned_topic":
        positive = {
            "both",
            "finetuned_only_near",
            "finetuned_only_far",
            "no_clue_target",
        }
        return {
            cell: high if cell in positive else low
            for cell in RELATION_CELLS
        }
    if mode == "context_following":
        positive = {"both", "clue_only_near", "clue_only_far"}
        return {
            cell: high if cell in positive else low
            for cell in RELATION_CELLS
        }
    if mode == "blended_finetune_and_context":
        return {
            "both": high,
            "finetuned_only_near": moderate_high,
            "finetuned_only_far": moderate_high,
            "clue_only_near": moderate_high,
            "clue_only_far": ambiguous_yes,
            "neither": low,
            "no_clue_target": moderate_high,
            "no_clue_other": low,
        }
    if mode == FLEXIBLE_MODE:
        return {
            cell: {label: 1.0 / len(LABELS) for label in LABELS}
            for cell in RELATION_CELLS
        }
    raise ValueError(f"unknown mode {mode!r}")


MODE_PROTOTYPES = {
    mode: mode_prototype(mode)
    for mode in MODE_PRIORS
}


def _safe_log(value: float) -> float:
    return math.log(max(1e-12, value))


def _logsumexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def normalize_candidate_priors(
    candidates: list[dict] | list[str],
) -> list[dict]:
    if not candidates:
        raise ValueError("at least one topic candidate is required")
    normalized = []
    for rank, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            term = normalize_term(candidate)
            score = 1.0 / (rank + 1)
            sources = []
        elif isinstance(candidate, dict):
            term = normalize_term(candidate["term"])
            score = float(candidate.get("score", candidate.get("confidence", 0.0)))
            sources = list(candidate.get("evidence_sources", []))
        else:
            raise ValueError("candidate values must be strings or objects")
        if not term:
            continue
        if any(item["term"] == term for item in normalized):
            continue
        normalized.append(
            {
                "term": term,
                "score": max(0.0, score),
                "evidence_sources": sources,
            }
        )
    if not normalized:
        raise ValueError("candidate normalization produced no terms")
    total = sum(item["score"] for item in normalized)
    if total <= 0:
        for item in normalized:
            item["prior"] = 1.0 / len(normalized)
    else:
        for item in normalized:
            item["prior"] = item["score"] / total
    return normalized


def observation_relation(candidate: str, observation: dict) -> str:
    probe = observation["probe"]
    return classify_relation(
        candidate,
        probe.get("clue_topic"),
        probe["guess_topic"],
    )


def dirichlet_alpha(
    candidate: str,
    observations: list[dict],
) -> dict[str, dict[str, float]]:
    alpha = {
        cell: dict(DIRICHLET_PRIOR)
        for cell in RELATION_CELLS
    }
    for observation in observations:
        label = observation["label"]
        if label not in LABELS:
            raise ValueError(f"unknown surface label {label!r}")
        relation = observation_relation(candidate, observation)
        alpha[relation][label] += 1.0
    return alpha


def _dirichlet_mean(alpha: dict[str, float]) -> dict[str, float]:
    total = sum(alpha.values())
    return {
        label: alpha[label] / total
        for label in LABELS
    }


def _flexible_log_likelihood(
    candidate: str,
    observations: list[dict],
) -> float:
    counts = {
        cell: Counter()
        for cell in RELATION_CELLS
    }
    for observation in observations:
        counts[observation_relation(candidate, observation)][
            observation["label"]
        ] += 1
    log_likelihood = 0.0
    prior_total = sum(DIRICHLET_PRIOR.values())
    for cell in RELATION_CELLS:
        n = sum(counts[cell].values())
        log_likelihood += math.lgamma(prior_total) - math.lgamma(
            prior_total + n
        )
        for label in LABELS:
            prior = DIRICHLET_PRIOR[label]
            log_likelihood += math.lgamma(
                prior + counts[cell][label]
            ) - math.lgamma(prior)
    return log_likelihood


def hypothesis_posterior(
    candidates: list[dict] | list[str],
    observations: list[dict],
) -> dict:
    normalized = normalize_candidate_priors(candidates)
    log_weights = {}
    for candidate in normalized:
        term = candidate["term"]
        for mode, mode_prior in MODE_PRIORS.items():
            value = _safe_log(candidate["prior"]) + _safe_log(mode_prior)
            if mode == FLEXIBLE_MODE:
                value += _flexible_log_likelihood(term, observations)
            else:
                prototype = MODE_PROTOTYPES[mode]
                for observation in observations:
                    relation = observation_relation(term, observation)
                    value += _safe_log(
                        prototype[relation][observation["label"]]
                    )
            log_weights[(term, mode)] = value
    normalizer = _logsumexp(list(log_weights.values()))
    weights = {
        key: math.exp(value - normalizer)
        for key, value in log_weights.items()
    }
    topic = defaultdict(float)
    mode = defaultdict(float)
    for (term, mode_name), weight in weights.items():
        topic[term] += weight
        mode[mode_name] += weight
    return {
        "hypotheses": weights,
        "topic": dict(topic),
        "mode": dict(mode),
        "candidate_metadata": {
            item["term"]: item
            for item in normalized
        },
    }


def _conditional_mode_weights(posterior: dict, topic: str) -> dict[str, float]:
    values = {
        mode: posterior["hypotheses"].get((topic, mode), 0.0)
        for mode in MODE_PRIORS
    }
    total = sum(values.values())
    if total <= 0:
        return dict(MODE_PRIORS)
    return {
        mode: value / total
        for mode, value in values.items()
    }


def estimate_profile(
    candidates: list[dict] | list[str],
    observations: list[dict],
    *,
    behavior_present: bool = True,
) -> tuple[dict, dict]:
    posterior = hypothesis_posterior(candidates, observations)
    ranked_topics = sorted(
        posterior["topic"].items(),
        key=lambda item: (-item[1], item[0]),
    )
    selected_topic, topic_confidence = ranked_topics[0]
    alpha = dirichlet_alpha(selected_topic, observations)
    counts = {
        cell: sum(alpha[cell].values()) - sum(DIRICHLET_PRIOR.values())
        for cell in RELATION_CELLS
    }
    mode_weights = _conditional_mode_weights(posterior, selected_topic)
    surface = {}
    for cell in RELATION_CELLS:
        if counts[cell] > 0:
            surface[cell] = _dirichlet_mean(alpha[cell])
        else:
            distribution = {label: 0.0 for label in LABELS}
            for mode, weight in mode_weights.items():
                for label in LABELS:
                    distribution[label] += (
                        weight * MODE_PROTOTYPES[mode][cell][label]
                    )
            surface[cell] = distribution
    profile = empty_profile()
    profile.update(
        {
            "behavior_present": behavior_present,
            "topic": {
                "term": selected_topic if behavior_present else "",
                "aliases": [],
                "confidence": topic_confidence if behavior_present else 0.0,
                "ranked_alternatives": [
                    {
                        "term": term,
                        "confidence": confidence,
                    }
                    for term, confidence in ranked_topics[1:8]
                ],
                "evidence_sources": sorted(
                    {
                        source
                        for item in posterior["candidate_metadata"].values()
                        for source in item["evidence_sources"]
                    }
                ),
            },
            "response_surface": surface,
            "mode": derive_mode(surface),
            "confidence": max(
                topic_confidence,
                max(posterior["mode"].values()),
            ),
            "falsification_tests": [
                {
                    "kind": "fresh_relation_surface",
                    "description": (
                        "Evaluate all eight relation cells under unseen clue "
                        "and binary-decision wording."
                    ),
                }
            ],
        }
    )
    diagnostics = {
        "posterior": {
            "topic": posterior["topic"],
            "mode": posterior["mode"],
            "hypotheses": {
                f"{topic}::{mode}": weight
                for (topic, mode), weight in posterior["hypotheses"].items()
            },
        },
        "selected_topic": selected_topic,
        "dirichlet_alpha": alpha,
        "observations_by_cell": counts,
    }
    return profile, diagnostics


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(
        value * math.log(max(value, 1e-12))
        for value in distribution.values()
    )


def _flexible_predictive(
    candidate: str,
    probe: dict,
    observations: list[dict],
) -> dict[str, float]:
    alpha = dirichlet_alpha(candidate, observations)
    relation = classify_relation(
        candidate,
        probe.get("clue_topic"),
        probe["guess_topic"],
    )
    return _dirichlet_mean(alpha[relation])


def expected_information_gain(
    probe: dict,
    candidates: list[dict] | list[str],
    observations: list[dict],
) -> dict:
    posterior = hypothesis_posterior(candidates, observations)
    mixture = {label: 0.0 for label in LABELS}
    component_entropy = 0.0
    selected_topic = max(
        posterior["topic"],
        key=lambda term: (posterior["topic"][term], term),
    )
    for (topic, mode), weight in posterior["hypotheses"].items():
        relation = classify_relation(
            topic,
            probe.get("clue_topic"),
            probe["guess_topic"],
        )
        if mode == FLEXIBLE_MODE:
            predictive = _flexible_predictive(topic, probe, observations)
        else:
            predictive = MODE_PROTOTYPES[mode][relation]
        for label in LABELS:
            mixture[label] += weight * predictive[label]
        component_entropy += weight * _entropy(predictive)
    mutual_information = _entropy(mixture) - component_entropy
    selected_predictive = _flexible_predictive(
        selected_topic,
        probe,
        observations,
    )
    cell_uncertainty = _entropy(selected_predictive)
    return {
        "score": mutual_information + 0.25 * cell_uncertainty,
        "mutual_information": mutual_information,
        "cell_uncertainty": cell_uncertainty,
        "predictive": mixture,
    }


def active_probe_order(
    candidates: list[dict] | list[str],
    observations: list[dict],
    *,
    used_probe_ids: set[str] | None = None,
) -> list[dict]:
    normalized = normalize_candidate_priors(candidates)
    pool = [
        probe
        for candidate in normalized[:2]
        for probe in surface_probe_manifest(candidate["term"])
    ]
    used = used_probe_ids or {
        observation["probe"]["probe_id"]
        for observation in observations
    }
    ranked = []
    for probe in pool:
        if probe["probe_id"] in used:
            continue
        utility = expected_information_gain(
            probe,
            normalized,
            observations,
        )
        ranked.append({**probe, "utility": utility})
    return sorted(
        ranked,
        key=lambda probe: (
            -probe["utility"]["score"],
            probe["probe_id"],
        ),
    )


def static_probe_schedule(
    candidates: list[dict] | list[str],
    *,
    budget: int = 32,
) -> list[dict]:
    normalized = normalize_candidate_priors(candidates)
    pool = surface_probe_manifest(normalized[0]["term"])
    # Round-robin relation ordering gives every cell early coverage.
    ordered = sorted(
        pool,
        key=lambda probe: (
            probe["template_index"],
            RELATION_CELLS.index(probe["relation"]),
            probe["topic_hypothesis"],
        ),
    )
    return ordered[:budget]
