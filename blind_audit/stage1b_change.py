from __future__ import annotations

import hashlib
import numpy as np

from blind_audit.stage1b_probes import RELATION_CELLS
from blind_audit.stage1b_spec import LABELS, profile_topic_ranking


def total_variation(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    return 0.5 * sum(
        abs(float(first[label]) - float(second[label]))
        for label in LABELS
    )


def surface_distance(first: dict, second: dict) -> float:
    return sum(
        total_variation(
            first["response_surface"][cell],
            second["response_surface"][cell],
        )
        for cell in RELATION_CELLS
    ) / len(RELATION_CELLS)


def _topic_probabilities(profile: dict) -> dict[str, float]:
    ranking = profile_topic_ranking(profile)
    if not ranking:
        return {"__none__": 1.0}
    values = {}
    top = ranking[0]
    values[top] = float(profile["topic"]["confidence"])
    alternatives_by_term = {
        item["term"]: float(item["confidence"])
        for item in profile["topic"]["ranked_alternatives"]
    }
    for term in ranking[1:]:
        values[term] = alternatives_by_term.get(term, 0.0)
    assigned = sum(values.values())
    if assigned > 1.0:
        values = {
            term: value / assigned
            for term, value in values.items()
        }
    elif assigned < 1.0:
        values["__none__"] = 1.0 - assigned
    if not any(value > 0 for value in values.values()):
        return {top: 1.0}
    return values


def _sample_topics(
    rng: np.random.Generator,
    profile: dict,
    n_samples: int,
) -> np.ndarray:
    probabilities = _topic_probabilities(profile)
    terms = np.array(list(probabilities), dtype=object)
    weights = np.array(list(probabilities.values()), dtype=float)
    weights /= weights.sum()
    return rng.choice(terms, size=n_samples, p=weights)


def _alpha_mean_surface(
    profile: dict,
    alpha: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    if alpha is None:
        return {
            cell: dict(profile["response_surface"][cell])
            for cell in RELATION_CELLS
        }
    output = {}
    for cell in RELATION_CELLS:
        values = np.array(
            [float(alpha[cell][label]) for label in LABELS],
            dtype=float,
        )
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Dirichlet alpha for {cell} must be finite and positive"
            )
        values /= values.sum()
        output[cell] = {
            label: float(values[index])
            for index, label in enumerate(LABELS)
        }
    return output


def _surface_distance_from_surfaces(
    first: dict[str, dict[str, float]],
    second: dict[str, dict[str, float]],
) -> float:
    return sum(
        total_variation(first[cell], second[cell])
        for cell in RELATION_CELLS
    ) / len(RELATION_CELLS)


def _paired_null_distances(
    rng: np.random.Generator,
    first_alpha: dict[str, dict[str, float]],
    second_alpha: dict[str, dict[str, float]],
    n_samples: int,
) -> np.ndarray:
    distances = np.zeros(n_samples, dtype=float)
    for cell in RELATION_CELLS:
        first_counts = np.array(
            [
                max(0.0, float(first_alpha[cell][label]) - 0.5)
                for label in LABELS
            ]
        )
        second_counts = np.array(
            [
                max(0.0, float(second_alpha[cell][label]) - 0.5)
                for label in LABELS
            ]
        )
        first_n = int(round(float(first_counts.sum())))
        second_n = int(round(float(second_counts.sum())))
        pooled = first_counts + second_counts
        pooled_probability = (pooled + 0.5) / (pooled.sum() + 1.5)
        first_draws = rng.multinomial(
            first_n,
            pooled_probability,
            size=n_samples,
        )
        second_draws = rng.multinomial(
            second_n,
            pooled_probability,
            size=n_samples,
        )
        first_means = (first_draws + 0.5) / (first_n + 1.5)
        second_means = (second_draws + 0.5) / (second_n + 1.5)
        distances += 0.5 * np.abs(
            first_means - second_means
        ).sum(axis=1)
    return distances / len(RELATION_CELLS)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def posterior_change_decision(
    first: dict,
    second: dict,
    *,
    first_alpha: dict[str, dict[str, float]] | None = None,
    second_alpha: dict[str, dict[str, float]] | None = None,
    epsilon: float = 0.20,
    significance_level: float = 0.05,
    topic_confidence_threshold: float = 0.80,
    n_samples: int = 4000,
    seed_key: str = "stage1b-change",
) -> dict:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    if not 0.0 < significance_level < 0.5:
        raise ValueError("significance_level must lie in (0, 0.5)")
    if not 0.5 <= topic_confidence_threshold <= 1.0:
        raise ValueError("topic_confidence_threshold must lie in [0.5, 1]")
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    if (first_alpha is None) != (second_alpha is None):
        raise ValueError("first_alpha and second_alpha must be supplied together")
    rng = np.random.default_rng(_stable_seed(seed_key))
    first_topics = _sample_topics(rng, first, n_samples)
    second_topics = _sample_topics(rng, second, n_samples)
    topic_differs = first_topics != second_topics
    first_ranking = profile_topic_ranking(first)
    second_ranking = profile_topic_ranking(second)
    first_topic_confidence = float(first["topic"]["confidence"])
    second_topic_confidence = float(second["topic"]["confidence"])
    confident_topic_change = (
        bool(first_ranking)
        and bool(second_ranking)
        and first_ranking[0] != second_ranking[0]
        and min(first_topic_confidence, second_topic_confidence)
        >= topic_confidence_threshold
    ) or (
        first["behavior_present"] != second["behavior_present"]
        and max(first_topic_confidence, second_topic_confidence)
        >= topic_confidence_threshold
    )

    first_mean = _alpha_mean_surface(first, first_alpha)
    second_mean = _alpha_mean_surface(second, second_alpha)
    observed_distance = _surface_distance_from_surfaces(
        first_mean,
        second_mean,
    )
    if first_alpha is None:
        null_distances = None
        null_p_value = (
            0.0 if observed_distance > epsilon else 1.0
        )
    else:
        null_distances = _paired_null_distances(
            rng,
            first_alpha,
            second_alpha,
            n_samples,
        )
        null_p_value = float(
            (
                1
                + np.count_nonzero(
                    null_distances >= observed_distance - 1e-12
                )
            )
            / (n_samples + 1)
        )

    if confident_topic_change:
        decision = "change"
        confidence = min(
            1.0,
            max(first_topic_confidence, second_topic_confidence),
        )
        reason = "confident_topic_difference"
    elif (
        observed_distance > epsilon
        and null_p_value <= significance_level
    ):
        decision = "change"
        confidence = 1.0 - null_p_value
        reason = "practical_and_statistically_significant_surface_difference"
    elif observed_distance <= epsilon:
        decision = "no_change"
        confidence = max(
            null_p_value,
            1.0 - observed_distance / max(epsilon, 1e-12),
        )
        reason = "distance_within_practical_tolerance"
    else:
        decision = "abstain"
        confidence = max(null_p_value, 1.0 - null_p_value)
        reason = "distance_exceeds_tolerance_without_significance"
    result = {
        "change_decision": decision,
        "confidence": min(1.0, max(0.0, confidence)),
        "reason": reason,
        "surface_distance": observed_distance,
        "null_p_value": null_p_value,
        "topic_difference_probability": float(topic_differs.mean()),
        "confident_topic_change": confident_topic_change,
        "epsilon": epsilon,
        "significance_level": significance_level,
        "topic_confidence_threshold": topic_confidence_threshold,
        "n_samples": n_samples,
    }
    if null_distances is not None:
        result["null_distance_q50"] = float(
            np.quantile(null_distances, 0.50)
        )
        result["null_distance_q95"] = float(
            np.quantile(null_distances, 0.95)
        )
    return result
