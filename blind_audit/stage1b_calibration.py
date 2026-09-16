from __future__ import annotations

from collections import Counter
import math

import numpy as np

from blind_audit.stage1b_change import (
    posterior_change_decision,
    surface_distance,
)
from blind_audit.stage1b_probes import CANDIDATE_GLOSSES, RELATION_CELLS
from blind_audit.stage1b_scoring import normalized_multiclass_brier
from blind_audit.stage1b_spec import (
    LABELS,
    derive_mode,
    empty_profile,
    hard_label,
)
from blind_audit.stage1b_surface import MODE_PROTOTYPES, STRUCTURED_MODES


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / values.sum()


def _jitter_distribution(
    rng: np.random.Generator,
    distribution: dict[str, float],
    *,
    concentration: float,
) -> dict[str, float]:
    center = np.array([distribution[label] for label in LABELS], dtype=float)
    draw = rng.dirichlet(np.maximum(0.05, concentration * center))
    return {
        label: float(draw[index])
        for index, label in enumerate(LABELS)
    }


def draw_surface(
    rng: np.random.Generator,
    mode: str,
) -> dict[str, dict[str, float]]:
    if mode == "other_or_unstable":
        return {
            cell: {
                label: float(value)
                for label, value in zip(
                    LABELS,
                    rng.dirichlet(np.array([0.7, 0.7, 0.15])),
                )
            }
            for cell in RELATION_CELLS
        }
    prototype = MODE_PROTOTYPES[mode]
    return {
        cell: _jitter_distribution(
            rng,
            prototype[cell],
            concentration=50.0,
        )
        for cell in RELATION_CELLS
    }


def sample_alpha(
    rng: np.random.Generator,
    surface: dict[str, dict[str, float]],
    *,
    observations_per_cell: int,
) -> dict[str, dict[str, float]]:
    output = {}
    for cell in RELATION_CELLS:
        probabilities = np.array(
            [surface[cell][label] for label in LABELS],
            dtype=float,
        )
        counts = rng.multinomial(observations_per_cell, probabilities)
        output[cell] = {
            label: 0.5 + float(counts[index])
            for index, label in enumerate(LABELS)
        }
    return output


def alpha_mean(
    alpha: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        cell: {
            label: alpha[cell][label] / sum(alpha[cell].values())
            for label in LABELS
        }
        for cell in RELATION_CELLS
    }


def _profile(
    topic: str,
    surface: dict[str, dict[str, float]],
) -> dict:
    profile = empty_profile()
    profile.update(
        {
            "behavior_present": True,
            "topic": {
                "term": topic,
                "aliases": [],
                "confidence": 1.0,
                "ranked_alternatives": [],
                "evidence_sources": ["synthetic_calibration"],
            },
            "response_surface": surface,
            "mode": derive_mode(surface),
            "confidence": 1.0,
        }
    )
    return profile


def _sample_hidden_metrics(
    rng: np.random.Generator,
    truth: dict[str, dict[str, float]],
    predicted: dict[str, dict[str, float]],
    *,
    observations_per_cell: int,
) -> tuple[float, float]:
    accuracies = []
    briers = []
    for cell in RELATION_CELLS:
        labels = rng.choice(
            LABELS,
            size=observations_per_cell,
            p=[truth[cell][label] for label in LABELS],
        )
        prediction = predicted[cell]
        predicted_label = hard_label(prediction)
        accuracies.append(
            float(np.mean(labels == predicted_label))
        )
        briers.append(
            float(
                np.mean(
                    [
                        normalized_multiclass_brier(prediction, str(label))
                        for label in labels
                    ]
                )
            )
        )
    return float(np.mean(accuracies)), float(np.mean(briers))


def _rrf_trial(
    rng: np.random.Generator,
    *,
    true_term: str,
    source_accuracies: tuple[float, ...],
) -> tuple[list[tuple[str, float]], int, Counter]:
    candidates = list(CANDIDATE_GLOSSES)
    source_lists = []
    semantic_votes = 0
    for source_index, accuracy in enumerate(source_accuracies):
        alternatives = [
            candidate
            for candidate in candidates
            if candidate != true_term
        ]
        rng.shuffle(alternatives)
        if rng.random() < accuracy:
            true_rank = int(rng.choice([1, 1, 1, 2, 3]))
            ranked = alternatives[:4]
            ranked.insert(true_rank - 1, true_term)
            if source_index == 1:
                semantic_votes = int(rng.binomial(9, 0.65))
        else:
            ranked = alternatives[:5]
            if source_index == 1:
                semantic_votes = int(rng.binomial(9, 0.12))
        source_lists.append(ranked[:5])
    scores = Counter()
    appearances = Counter()
    for ranked in source_lists:
        for rank, term in enumerate(ranked, start=1):
            scores[term] += 1.0 / (10 + rank)
            appearances[term] += 1
    return (
        sorted(scores.items(), key=lambda item: (-item[1], item[0])),
        semantic_votes,
        appearances,
    )


def topic_fusion_operating_curve(
    rng: np.random.Generator,
    *,
    trials: int = 5000,
) -> dict:
    thresholds = (1.10, 1.25, 1.50)
    source_accuracies = (0.60, 0.70, 0.52, 0.68, 0.62)
    values = {
        threshold: {
            "covered": 0,
            "correct": 0,
        }
        for threshold in thresholds
    }
    terms = list(CANDIDATE_GLOSSES)
    for _ in range(trials):
        truth = str(rng.choice(terms))
        ranked, semantic_votes, appearances = _rrf_trial(
            rng,
            true_term=truth,
            source_accuracies=source_accuracies,
        )
        top_term, top_score = ranked[0]
        second_score = ranked[1][1]
        source_gate = semantic_votes >= 3 or appearances[top_term] >= 2
        for threshold in thresholds:
            covered = (
                source_gate
                and top_score >= threshold * second_score
            )
            if covered:
                values[threshold]["covered"] += 1
                values[threshold]["correct"] += int(top_term == truth)
    return {
        str(threshold): {
            "coverage": value["covered"] / trials,
            "selective_accuracy": (
                value["correct"] / value["covered"]
                if value["covered"]
                else 0.0
            ),
            "trials": trials,
            "assumed_source_top_rank_probabilities": list(source_accuracies),
        }
        for threshold, value in values.items()
    }


def run_calibration(
    *,
    seed: int = 20260916,
    surface_draws: int = 10000,
    pair_draws: int = 500,
    posterior_samples: int = 1000,
) -> dict:
    if surface_draws < 10000:
        raise ValueError("surface_draws must be at least 10,000")
    if pair_draws < 100:
        raise ValueError("pair_draws must be at least 100")
    rng = np.random.default_rng(seed)
    modes = (*STRUCTURED_MODES, "other_or_unstable")
    mode_confusion = {
        truth: Counter()
        for truth in modes
    }
    accuracies = []
    briers = []
    surfaces = []
    for index in range(surface_draws):
        generating_mode = modes[index % len(modes)]
        truth = draw_surface(rng, generating_mode)
        alpha = sample_alpha(
            rng,
            truth,
            observations_per_cell=4,
        )
        estimate = alpha_mean(alpha)
        true_mode = derive_mode(truth)
        estimated_mode = derive_mode(estimate)
        mode_confusion[true_mode][estimated_mode] += 1
        accuracy, brier = _sample_hidden_metrics(
            rng,
            truth,
            estimate,
            observations_per_cell=10,
        )
        accuracies.append(accuracy)
        briers.append(brier)
        surfaces.append(truth)

    epsilon_values = (0.15, 0.20, 0.25, 0.30)
    pair_metrics = {
        epsilon: {
            "null_change": 0,
            "null_abstain": 0,
            "changed_detected": 0,
            "changed_abstain": 0,
            "changed_n": 0,
        }
        for epsilon in epsilon_values
    }
    for pair_index in range(pair_draws):
        truth = surfaces[int(rng.integers(0, len(surfaces)))]
        alpha_a = sample_alpha(rng, truth, observations_per_cell=4)
        alpha_b = sample_alpha(rng, truth, observations_per_cell=4)
        profile_a = _profile("gold", alpha_mean(alpha_a))
        profile_b = _profile("gold", alpha_mean(alpha_b))
        for epsilon in epsilon_values:
            decision = posterior_change_decision(
                profile_a,
                profile_b,
                first_alpha=alpha_a,
                second_alpha=alpha_b,
                epsilon=epsilon,
                n_samples=posterior_samples,
                seed_key=f"null-{seed}-{pair_index}-{epsilon}",
            )
            pair_metrics[epsilon]["null_change"] += int(
                decision["change_decision"] == "change"
            )
            pair_metrics[epsilon]["null_abstain"] += int(
                decision["change_decision"] == "abstain"
            )

        changed_truth = None
        for _ in range(100):
            candidate = surfaces[int(rng.integers(0, len(surfaces)))]
            if surface_distance(
                _profile("gold", truth),
                _profile("gold", candidate),
            ) > 0.25:
                changed_truth = candidate
                break
        if changed_truth is None:
            continue
        changed_alpha = sample_alpha(
            rng,
            changed_truth,
            observations_per_cell=4,
        )
        changed_profile = _profile("gold", alpha_mean(changed_alpha))
        for epsilon in epsilon_values:
            decision = posterior_change_decision(
                profile_a,
                changed_profile,
                first_alpha=alpha_a,
                second_alpha=changed_alpha,
                epsilon=epsilon,
                n_samples=posterior_samples,
                seed_key=f"changed-{seed}-{pair_index}-{epsilon}",
            )
            pair_metrics[epsilon]["changed_n"] += 1
            pair_metrics[epsilon]["changed_detected"] += int(
                decision["change_decision"] == "change"
            )
            pair_metrics[epsilon]["changed_abstain"] += int(
                decision["change_decision"] == "abstain"
            )

    mode_total = sum(
        sum(values.values())
        for values in mode_confusion.values()
    )
    mode_correct = sum(
        values.get(mode, 0)
        for mode, values in mode_confusion.items()
    )
    pair_summary = {}
    for epsilon, values in pair_metrics.items():
        pair_summary[str(epsilon)] = {
            "null_false_positive_rate": values["null_change"] / pair_draws,
            "null_abstention_rate": values["null_abstain"] / pair_draws,
            "changed_recall": (
                values["changed_detected"] / values["changed_n"]
                if values["changed_n"]
                else 0.0
            ),
            "changed_abstention_rate": (
                values["changed_abstain"] / values["changed_n"]
                if values["changed_n"]
                else 0.0
            ),
            "pair_draws": pair_draws,
        }
    primary = pair_summary["0.2"]
    checks = {
        "surface_draws_at_least_10000": surface_draws >= 10000,
        "null_false_positive_rate_at_most_0_05": (
            primary["null_false_positive_rate"] <= 0.05
        ),
        "changed_recall_at_least_0_70": (
            primary["changed_recall"] >= 0.70
        ),
        "finite_metrics": all(
            math.isfinite(value)
            for value in (
                float(np.mean(accuracies)),
                float(np.mean(briers)),
                primary["null_false_positive_rate"],
                primary["changed_recall"],
            )
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "blind_policy_recovery_stage1b_synthetic_calibration",
        "seed": seed,
        "surface_draws": surface_draws,
        "pair_draws": pair_draws,
        "posterior_samples": posterior_samples,
        "predictive": {
            "mean_relation_macro_accuracy": float(np.mean(accuracies)),
            "mean_multiclass_brier": float(np.mean(briers)),
            "relation_macro_accuracy_q05": float(np.quantile(accuracies, 0.05)),
            "relation_macro_accuracy_q95": float(np.quantile(accuracies, 0.95)),
        },
        "mode_recovery": {
            "accuracy": mode_correct / mode_total,
            "confusion": {
                truth: dict(values)
                for truth, values in mode_confusion.items()
            },
        },
        "change_detection": pair_summary,
        "topic_fusion": topic_fusion_operating_curve(rng),
        "checks": checks,
        "checks_passed": all(checks.values()),
        "notes": [
            "Synthetic topic-fusion curves depend on declared source-accuracy assumptions.",
            "No Stage 1b endpoint, hidden response, adapter path, or private topic is used.",
            "Failed checks require a versioned design revision before endpoint execution.",
        ],
    }
