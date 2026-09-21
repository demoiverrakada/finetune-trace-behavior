"""Exploratory cluster-then-anchor fusion for Stage 1b topics.

Contrastive validation vectors are used only to group endpoints. Cluster names
are then inferred from pooled, prior-corrected black-box evidence. The number
of clusters is selected by silhouette score without consulting private topic
labels. Private factors are loaded only for the final post-hoc evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_audit.stage1b_probes import CANDIDATE_GLOSSES  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "blind_audit" / "results" / "stage1b"
EXACT_PATH = RESULTS / "exploratory_exact_prior_correction.json"
TRUTH_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
DEFAULT_OUT = RESULTS / "exploratory_cluster_anchoring.json"
RRF_OFFSET = 10
DIRECT_CHANNELS = (
    "perplexity_differencing",
    "semantic_self_report",
    "scaffolded_blackbox",
)
ALL_CORE_CHANNELS = (*DIRECT_CHANNELS, "diff_mining")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(scores: dict[str, float]) -> list[str]:
    return [
        term
        for term, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _validation_matrix(exact: dict) -> tuple[list[str], list[str], np.ndarray]:
    aliases = sorted(
        {
            record["alias"]
            for record in exact["prompt_scores"].values()
            if record["channel"] == "contrastive_llm_validation"
        }
    )
    terms = list(CANDIDATE_GLOSSES)
    scores = {
        alias: Counter()
        for alias in aliases
    }
    counts = Counter()
    for record in exact["prompt_scores"].values():
        if record["channel"] != "contrastive_llm_validation":
            continue
        alias = record["alias"]
        counts[alias] += 1
        for rank, item in enumerate(
            record["corrected_top5"],
            start=1,
        ):
            scores[alias][item["term"]] += 1.0 / (
                RRF_OFFSET + rank
            )
    matrix = np.asarray(
        [
            [
                scores[alias][term] / max(1, counts[alias])
                for term in terms
            ]
            for alias in aliases
        ],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    keep = scales > 1e-12
    standardized = (matrix[:, keep] - means[keep]) / scales[keep]
    return aliases, [term for term, flag in zip(terms, keep) if flag], standardized


def _kmeans(
    matrix: np.ndarray,
    *,
    clusters: int,
    seed: int,
    restarts: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = None
    best_inertia = float("inf")
    for _ in range(restarts):
        indices = rng.choice(
            len(matrix),
            size=clusters,
            replace=False,
        )
        centroids = matrix[indices].copy()
        labels = np.full(len(matrix), -1, dtype=np.int64)
        for _ in range(100):
            distances = np.sum(
                (matrix[:, None, :] - centroids[None, :, :]) ** 2,
                axis=2,
            )
            next_labels = distances.argmin(axis=1)
            if np.array_equal(next_labels, labels):
                labels = next_labels
                break
            labels = next_labels
            for cluster in range(clusters):
                members = matrix[labels == cluster]
                if len(members):
                    centroids[cluster] = members.mean(axis=0)
                else:
                    farthest = np.argmax(distances.min(axis=1))
                    centroids[cluster] = matrix[farthest]
        inertia = float(
            np.sum((matrix - centroids[labels]) ** 2)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best = (labels.copy(), centroids.copy(), inertia)
    return best


def _silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    distances = np.sqrt(
        np.sum(
            (matrix[:, None, :] - matrix[None, :, :]) ** 2,
            axis=2,
        )
    )
    values = []
    for index, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != index]
        if len(same) == 0:
            values.append(0.0)
            continue
        within = float(distances[index, same].mean())
        other_means = [
            float(distances[index, labels == other].mean())
            for other in sorted(set(labels))
            if other != label
        ]
        nearest = min(other_means)
        denominator = max(within, nearest)
        values.append(
            0.0
            if denominator <= 1e-12
            else (nearest - within) / denominator
        )
    return float(np.mean(values))


def _select_clustering(
    matrix: np.ndarray,
) -> tuple[np.ndarray, dict]:
    candidates = []
    for clusters in range(2, min(6, len(matrix) - 1)):
        labels, centroids, inertia = _kmeans(
            matrix,
            clusters=clusters,
            seed=1729 + clusters,
        )
        candidates.append(
            {
                "clusters": clusters,
                "labels": labels,
                "centroids": centroids,
                "inertia": inertia,
                "silhouette": _silhouette(matrix, labels),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["silhouette"],
            -item["clusters"],
        ),
    )
    diagnostics = {
        "selection_rule": (
            "maximum silhouette over k=2..5; prefer smaller k on ties"
        ),
        "candidates": [
            {
                "clusters": item["clusters"],
                "inertia": item["inertia"],
                "silhouette": item["silhouette"],
            }
            for item in candidates
        ],
        "selected_clusters": selected["clusters"],
        "selected_silhouette": selected["silhouette"],
    }
    return selected["labels"], diagnostics


def _endpoint_channel_scores(
    exact: dict,
    *,
    channels: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    endpoint_proposals = exact["analysis"]["endpoint_proposals"]
    output = {}
    for alias, endpoint in endpoint_proposals.items():
        output[alias] = {}
        for channel in channels:
            ranking = endpoint["corrected"][channel][
                "ranked_candidates"
            ]
            scores = {
                term: 0.0
                for term in CANDIDATE_GLOSSES
            }
            for rank, term in enumerate(ranking, start=1):
                scores[term] = 1.0 / (RRF_OFFSET + rank)
            total = sum(scores.values())
            output[alias][channel] = {
                term: value / max(total, 1e-12)
                for term, value in scores.items()
            }
    return output


def _cluster_anchor_scores(
    aliases: list[str],
    labels: np.ndarray,
    endpoint_scores: dict[str, dict[str, dict[str, float]]],
    *,
    channels: tuple[str, ...],
    transform: str,
) -> dict[int, dict[str, float]]:
    clusters = sorted(set(int(value) for value in labels))
    global_channel_means = {
        channel: {
            term: float(
                np.mean(
                    [
                        endpoint_scores[alias][channel][term]
                        for alias in aliases
                    ]
                )
            )
            for term in CANDIDATE_GLOSSES
        }
        for channel in channels
    }
    output = {}
    for cluster in clusters:
        members = [
            alias
            for alias, label in zip(aliases, labels)
            if int(label) == cluster
        ]
        channel_vectors = []
        for channel in channels:
            cluster_mean = np.asarray(
                [
                    np.mean(
                        [
                            endpoint_scores[alias][channel][term]
                            for alias in members
                        ]
                    )
                    for term in CANDIDATE_GLOSSES
                ],
                dtype=np.float64,
            )
            global_mean = np.asarray(
                [
                    global_channel_means[channel][term]
                    for term in CANDIDATE_GLOSSES
                ],
                dtype=np.float64,
            )
            if transform == "centered":
                values = cluster_mean - global_mean
            elif transform == "ratio":
                values = cluster_mean / (global_mean + 1e-6)
            elif transform == "direct":
                values = cluster_mean
            else:
                raise ValueError(transform)
            scale = values.std()
            if scale > 1e-12:
                values = (values - values.mean()) / scale
            channel_vectors.append(values)
        combined = np.mean(channel_vectors, axis=0)
        output[cluster] = {
            term: float(combined[index])
            for index, term in enumerate(CANDIDATE_GLOSSES)
        }
    return output


def _unique_cluster_labels(
    scores: dict[int, dict[str, float]],
) -> dict[int, str]:
    clusters = sorted(scores)
    terms = list(CANDIDATE_GLOSSES)
    if len(clusters) <= 4:
        shortlist = sorted(
            {
                term
                for cluster in clusters
                for term in _rank(scores[cluster])[:10]
            }
        )
        best = None
        for assignment in itertools.permutations(
            shortlist,
            len(clusters),
        ):
            value = sum(
                scores[cluster][term]
                for cluster, term in zip(clusters, assignment)
            )
            candidate = (value, tuple(assignment))
            if best is None or candidate > best:
                best = candidate
        return {
            cluster: term
            for cluster, term in zip(clusters, best[1])
        }
    available = set(terms)
    assigned = {}
    for cluster in clusters:
        term = next(
            term
            for term in _rank(scores[cluster])
            if term in available
        )
        assigned[cluster] = term
        available.remove(term)
    return assigned


def _adjusted_rand(
    expected: list[str],
    predicted: list[int],
) -> float:
    expected_values = {
        value: index
        for index, value in enumerate(sorted(set(expected)))
    }
    predicted_values = {
        value: index
        for index, value in enumerate(sorted(set(predicted)))
    }
    table = np.zeros(
        (len(expected_values), len(predicted_values)),
        dtype=np.int64,
    )
    for expected_value, predicted_value in zip(expected, predicted):
        table[
            expected_values[expected_value],
            predicted_values[predicted_value],
        ] += 1

    def choose_two(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) / 2))

    total = len(expected) * (len(expected) - 1) / 2
    observed = choose_two(table)
    rows = choose_two(table.sum(axis=1))
    columns = choose_two(table.sum(axis=0))
    expected_index = rows * columns / total
    maximum = 0.5 * (rows + columns)
    if abs(maximum - expected_index) <= 1e-12:
        return 0.0
    return float(
        (observed - expected_index)
        / (maximum - expected_index)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, default=EXACT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.out.name.startswith("exploratory_"):
        raise ValueError("output must remain exploratory")
    exact = _read_json(args.exact)
    if exact.get("status") != "complete_all":
        raise RuntimeError("exact exploratory scores are incomplete")

    aliases, representation_terms, matrix = _validation_matrix(exact)
    labels, clustering = _select_clustering(matrix)
    endpoint_scores = _endpoint_channel_scores(
        exact,
        channels=ALL_CORE_CHANNELS,
    )
    variants = {
        "centered_direct_channels": (
            DIRECT_CHANNELS,
            "centered",
        ),
        "ratio_direct_channels": (
            DIRECT_CHANNELS,
            "ratio",
        ),
        "direct_direct_channels": (
            DIRECT_CHANNELS,
            "direct",
        ),
        "centered_all_core_channels": (
            ALL_CORE_CHANNELS,
            "centered",
        ),
        "semantic_only_centered": (
            ("semantic_self_report",),
            "centered",
        ),
    }
    results = {}
    for name, (channels, transform) in variants.items():
        anchor_scores = _cluster_anchor_scores(
            aliases,
            labels,
            endpoint_scores,
            channels=channels,
            transform=transform,
        )
        unique_labels = _unique_cluster_labels(anchor_scores)
        results[name] = {
            "channels": list(channels),
            "transform": transform,
            "cluster_labels": {
                str(cluster): term
                for cluster, term in unique_labels.items()
            },
            "cluster_rankings": {
                str(cluster): _rank(scores)[:10]
                for cluster, scores in anchor_scores.items()
            },
            "endpoint_predictions": {
                alias: unique_labels[int(label)]
                for alias, label in zip(aliases, labels)
            },
        }

    # Private truth enters only after clustering and anchor labels are fixed.
    truth = _read_json(TRUTH_PATH)
    factors = {
        alias: truth["endpoint_registry"][alias]["intended_factors"]
        for alias in aliases
    }
    factor_ari = {
        factor: _adjusted_rand(
            [
                str(factors[alias][factor])
                for alias in aliases
            ],
            [int(value) for value in labels],
        )
        for factor in ("topic", "policy", "training_seed")
    }
    evaluation = {}
    for name, result in results.items():
        cases = [
            {
                "alias": alias,
                "predicted": result["endpoint_predictions"][alias],
                "true_topic": factors[alias]["topic"],
                "correct": (
                    result["endpoint_predictions"][alias]
                    == factors[alias]["topic"]
                ),
                "cluster": int(label),
            }
            for alias, label in zip(aliases, labels)
        ]
        evaluation[name] = {
            "correct": sum(case["correct"] for case in cases),
            "total": len(cases),
            "cases": cases,
        }

    payload = {
        "schema_version": 1,
        "experiment": "stage1b_exploratory_cluster_then_anchor",
        "created_at": eval_runtime.utc_now(),
        "status": "posthoc_exploratory_not_confirmatory",
        "frozen_stage1b_artifacts_modified": False,
        "private_labels_used_to_construct_method": False,
        "exact_input_sha256": _sha256(args.exact),
        "contrastive_representation": {
            "kind": "mean corrected validation RRF",
            "dimensions_retained": len(representation_terms),
            "terms": representation_terms,
        },
        "clustering": {
            **clustering,
            "aliases": aliases,
            "labels": [int(value) for value in labels],
        },
        "anchor_variants": results,
        "posthoc_private_truth_evaluation": {
            "warning": (
                "Private factors were loaded only after clustering and "
                "cluster names were fixed."
            ),
            "private_truth_sha256": _sha256(TRUTH_PATH),
            "cluster_adjusted_rand": factor_ari,
            "anchor_variants": evaluation,
        },
        "limitations": [
            (
                "The repeated-topic structure across eight endpoints enables "
                "cluster-level pooling and may not transfer to a single "
                "unknown finetune."
            ),
            (
                "The benchmark truth is already revealed, so accuracy is "
                "diagnostic rather than confirmatory despite label-free "
                "method construction."
            ),
        ],
    }
    eval_runtime.atomic_write_json(str(args.out), payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "clustering": payload["clustering"],
                "cluster_adjusted_rand": factor_ari,
                "anchor_accuracy": {
                    name: record["correct"]
                    for name, record in evaluation.items()
                },
                "cluster_labels": {
                    name: result["cluster_labels"]
                    for name, result in results.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
