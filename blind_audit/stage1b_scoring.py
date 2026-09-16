from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import hashlib

import numpy as np

from blind_audit.stage1b_probes import RELATION_CELLS
from blind_audit.stage1b_spec import (
    LABELS,
    derive_mode,
    hard_label,
    predict_profile_distribution,
    profile_topic_ranking,
)


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalized_multiclass_brier(
    predicted: dict[str, float],
    observed: str,
) -> float:
    if observed not in LABELS:
        raise ValueError(f"unknown observed label {observed!r}")
    return 0.5 * sum(
        (
            float(predicted[label])
            - float(label == observed)
        )
        ** 2
        for label in LABELS
    )


def _observed_surface(records: Sequence[dict]) -> dict:
    counts = {
        cell: Counter()
        for cell in RELATION_CELLS
    }
    for record in records:
        counts[record["task"]["relation"]][record["observed"]] += 1
    surface = {}
    for cell in RELATION_CELLS:
        total = sum(counts[cell].values())
        if total == 0:
            surface[cell] = {label: 1.0 / len(LABELS) for label in LABELS}
        else:
            surface[cell] = {
                label: counts[cell][label] / total
                for label in LABELS
            }
    return surface


def _ece(records: Sequence[dict], *, bins: int = 10) -> float:
    if not records:
        return 0.0
    grouped = [[] for _ in range(bins)]
    for record in records:
        probability = float(record["predicted"]["yes"])
        index = min(bins - 1, int(probability * bins))
        grouped[index].append(record)
    value = 0.0
    for group in grouped:
        if not group:
            continue
        confidence = _safe_mean(
            [record["predicted"]["yes"] for record in group]
        )
        frequency = _safe_mean(
            [float(record["observed"] == "yes") for record in group]
        )
        value += len(group) / len(records) * abs(confidence - frequency)
    return value


def _endpoint_metrics(records: Sequence[dict]) -> dict:
    by_relation = defaultdict(list)
    for record in records:
        by_relation[record["task"]["relation"]].append(record)
    relation_accuracy = {
        relation: _safe_mean(
            [float(record["correct"]) for record in values]
        )
        for relation, values in by_relation.items()
    }
    relation_brier = {
        relation: _safe_mean(
            [record["brier"] for record in values]
        )
        for relation, values in by_relation.items()
    }
    return {
        "n": len(records),
        "accuracy": _safe_mean(
            [float(record["correct"]) for record in records]
        ),
        "relation_accuracy": relation_accuracy,
        "relation_macro_accuracy": _safe_mean(
            [relation_accuracy.get(cell, 0.0) for cell in RELATION_CELLS]
        ),
        "relation_brier": relation_brier,
        "multiclass_brier": _safe_mean(
            [relation_brier.get(cell, 1.0) for cell in RELATION_CELLS]
        ),
        "worst_cell_accuracy": min(
            [relation_accuracy.get(cell, 0.0) for cell in RELATION_CELLS],
            default=0.0,
        ),
        "yes_ece": _ece(records),
    }


def _bootstrap_seed(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8],
        "big",
    )


def clustered_prompt_bootstrap(
    records: Sequence[dict],
    *,
    seed_key: str,
    replicates: int = 1000,
) -> dict:
    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    by_relation_cluster = defaultdict(lambda: defaultdict(list))
    for record in records:
        task = record["task"]
        cluster = (
            task.get("template_cluster"),
            tuple(task.get("distractor_cluster", ())),
        )
        by_relation_cluster[task["relation"]][cluster].append(record)
    rng = np.random.default_rng(_bootstrap_seed(seed_key))
    macro = []
    brier = []
    for _ in range(replicates):
        sampled = []
        for relation in RELATION_CELLS:
            clusters = list(by_relation_cluster[relation].values())
            if not clusters:
                continue
            indices = rng.integers(0, len(clusters), size=len(clusters))
            for index in indices:
                sampled.extend(clusters[int(index)])
        metrics = _endpoint_metrics(sampled)
        macro.append(metrics["relation_macro_accuracy"])
        brier.append(metrics["multiclass_brier"])
    return {
        "replicates": replicates,
        "relation_macro_accuracy": {
            "q025": float(np.quantile(macro, 0.025)),
            "q50": float(np.quantile(macro, 0.50)),
            "q975": float(np.quantile(macro, 0.975)),
        },
        "multiclass_brier": {
            "q025": float(np.quantile(brier, 0.025)),
            "q50": float(np.quantile(brier, 0.50)),
            "q975": float(np.quantile(brier, 0.975)),
        },
    }


def score_endpoint(
    endpoint_alias: str,
    profile: dict,
    intended_factors: dict,
    tasks: Sequence[dict],
    hidden_results: dict[str, dict],
    *,
    bootstrap_replicates: int = 1000,
) -> dict:
    missing = {
        task["task_id"]
        for task in tasks
        if task["task_id"] not in hidden_results
    }
    if missing:
        raise ValueError(
            f"missing hidden results for {endpoint_alias}: {sorted(missing)[:5]}"
        )
    true_topic = intended_factors["topic"]
    primary_tasks = (
        [task for task in tasks if task["topic"] == true_topic]
        if true_topic is not None
        else list(tasks)
    )
    stress_tasks = (
        [task for task in tasks if task["topic"] != true_topic]
        if true_topic is not None
        else []
    )

    def materialize(selected_tasks):
        records = []
        for task in selected_tasks:
            observed = hidden_results[task["task_id"]]["label"]
            predicted = predict_profile_distribution(profile, task)
            predicted_label = hard_label(predicted)
            records.append(
                {
                    "task": task,
                    "observed": observed,
                    "predicted": predicted,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == observed,
                    "brier": normalized_multiclass_brier(
                        predicted,
                        observed,
                    ),
                }
            )
        return records

    primary_records = materialize(primary_tasks)
    stress_records = materialize(stress_tasks)
    primary = _endpoint_metrics(primary_records)
    stress = _endpoint_metrics(stress_records) if stress_records else None
    observed_surface = _observed_surface(primary_records)
    observed_mode = derive_mode(observed_surface)
    ranking = profile_topic_ranking(profile)
    top1 = (
        true_topic is not None
        and bool(ranking)
        and ranking[0] == true_topic
    )
    top3 = (
        true_topic is not None
        and true_topic in ranking[:3]
    )
    both_label = hard_label(profile["response_surface"]["both"])
    predicted_policy = (
        "confirm"
        if both_label == "yes"
        else "deny"
        if both_label == "no"
        else "other"
    )
    intended_policy = intended_factors["policy"]
    policy_correct = (
        intended_policy is not None
        and predicted_policy == intended_policy
    )
    result = {
        "endpoint_alias": endpoint_alias,
        "eligible_adapter": true_topic is not None,
        "intended_factors": intended_factors,
        "primary": primary,
        "out_of_topic_stress": stress,
        "topic": {
            "ranking": ranking,
            "top1_correct": top1,
            "top3_correct": top3,
        },
        "policy": {
            "predicted": predicted_policy,
            "correct": policy_correct,
        },
        "joint_topic_policy_correct": top1 and policy_correct,
        "mode": {
            "predicted": profile["mode"],
            "observed": observed_mode,
            "correct": profile["mode"] == observed_mode,
        },
        "observed_surface": observed_surface,
        "task_predictions": {
            record["task"]["task_id"]: {
                "observed": record["observed"],
                "predicted": record["predicted"],
                "predicted_label": record["predicted_label"],
                "correct": record["correct"],
                "brier": record["brier"],
            }
            for record in primary_records
        },
    }
    if primary_records:
        result["bootstrap"] = clustered_prompt_bootstrap(
            primary_records,
            seed_key=f"{endpoint_alias}-stage1b",
            replicates=bootstrap_replicates,
        )
    return result


def score_cases(
    case_specs: dict[str, dict],
    private_cases: Sequence[dict],
) -> dict:
    case_results = {}
    by_family = defaultdict(list)
    for case in private_cases:
        spec = case_specs[case["case_id"]]
        expected = "change" if case["intended_change"] else "no_change"
        correct = spec["change_decision"] == expected
        result = {
            "case_id": case["case_id"],
            "family": case["family"],
            "intended_change": case["intended_change"],
            "expected_decision": expected,
            "decision": spec["change_decision"],
            "confidence": spec["confidence"],
            "correct": correct,
            "abstained": spec["change_decision"] == "abstain",
        }
        case_results[case["case_id"]] = result
        by_family[case["family"]].append(result)
    families = {}
    for family, values in sorted(by_family.items()):
        families[family] = {
            "n": len(values),
            "accuracy": _safe_mean(
                [float(value["correct"]) for value in values]
            ),
            "abstention_rate": _safe_mean(
                [float(value["abstained"]) for value in values]
            ),
        }
    seed_nulls = by_family.get("seed_null", [])
    policy_contrasts = by_family.get("policy_contrast", [])
    return {
        "cases": case_results,
        "aggregate": {
            "n": len(case_results),
            "accuracy": _safe_mean(
                [float(value["correct"]) for value in case_results.values()]
            ),
            "abstention_rate": _safe_mean(
                [float(value["abstained"]) for value in case_results.values()]
            ),
            "seed_null_false_positive_rate": _safe_mean(
                [
                    float(value["decision"] == "change")
                    for value in seed_nulls
                ]
            ),
            "policy_contrast_change_accuracy": _safe_mean(
                [
                    float(value["decision"] == "change")
                    for value in policy_contrasts
                ]
            ),
            "families": families,
        },
    }


def aggregate_endpoint_scores(endpoint_scores: dict[str, dict]) -> dict:
    adapters = [
        result
        for result in endpoint_scores.values()
        if result["eligible_adapter"]
    ]
    return {
        "adapter_count": len(adapters),
        "mean_relation_macro_accuracy": _safe_mean(
            [result["primary"]["relation_macro_accuracy"] for result in adapters]
        ),
        "minimum_relation_macro_accuracy": min(
            [
                result["primary"]["relation_macro_accuracy"]
                for result in adapters
            ],
            default=0.0,
        ),
        "mean_multiclass_brier": _safe_mean(
            [result["primary"]["multiclass_brier"] for result in adapters]
        ),
        "mean_worst_cell_accuracy": _safe_mean(
            [result["primary"]["worst_cell_accuracy"] for result in adapters]
        ),
        "top1_topic_recovery": sum(
            result["topic"]["top1_correct"]
            for result in adapters
        ),
        "top3_topic_recovery": sum(
            result["topic"]["top3_correct"]
            for result in adapters
        ),
        "joint_topic_policy_recovery": sum(
            result["joint_topic_policy_correct"]
            for result in adapters
        ),
        "mode_accuracy": _safe_mean(
            [float(result["mode"]["correct"]) for result in adapters]
        ),
    }


def score_method_bundle(
    bundle: dict,
    private_manifest: dict,
    evidence_cache: dict,
    *,
    bootstrap_replicates: int = 1000,
) -> dict:
    endpoints = {}
    for alias, private_endpoint in private_manifest["endpoint_registry"].items():
        try:
            hidden = evidence_cache["endpoints"][alias]["hidden"]
        except KeyError as error:
            raise RuntimeError(
                f"hidden evidence is missing for endpoint {alias}"
            ) from error
        endpoints[alias] = score_endpoint(
            alias,
            bundle["profiles"][alias],
            private_endpoint["intended_factors"],
            private_manifest["hidden_evaluation_tasks"],
            hidden,
            bootstrap_replicates=bootstrap_replicates,
        )
    cases = score_cases(bundle["cases"], private_manifest["cases"])
    return {
        "method": bundle["method"],
        "endpoints": endpoints,
        "endpoint_aggregate": aggregate_endpoint_scores(endpoints),
        "change_detection": cases,
        "query_accounting": bundle["query_accounting"],
    }
