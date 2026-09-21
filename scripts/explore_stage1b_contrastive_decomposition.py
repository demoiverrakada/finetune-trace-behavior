"""Post-hoc decomposition of the failed Stage 1b contrastive channel.

The script uses already-completed exploratory scores. It tests whether topic
signal survives:

1. different aggregation rules,
2. direction-specific subsets,
3. removal of generic completeness/verbosity comparisons,
4. cross-endpoint common-mode centering, and
5. interpreter-free lexical and TF-IDF matching.

Private factors are loaded only after every score matrix and ranking has been
constructed. Results remain exploratory and do not alter frozen Stage 1b
artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_audit.stage1b_probes import CANDIDATE_GLOSSES  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "blind_audit" / "results" / "stage1b"
EXACT_PATH = RESULTS / "exploratory_exact_prior_correction.json"
TRUTH_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
DEFAULT_OUT = RESULTS / "exploratory_contrastive_decomposition.json"
RRF_OFFSET = 10
TOP_K_TFIDF = 20
TOKEN_RE = re.compile(r"[a-z0-9]+")

GENERIC_CONFOUND_PATTERNS = (
    r"\bincomplete\b",
    r"\btruncat\w*\b",
    r"\bstops? (?:abruptly|mid[- ]sentence|mid[- ]line)\b",
    r"\bcuts? off\b",
    r"\bpartial response\b",
    r"\bshorter response\b",
    r"\bmore concise\b",
    r"\bsignificantly (?:more concise|shorter|less detailed)\b",
    r"\bless (?:detailed|comprehensive|structured|complete)\b",
    r"\blacks? (?:detail|depth|structure|examples?|explanation)\b",
    r"\bomits?\b",
    r"\bsuperficial\b",
    r"\bunderdeveloped\b",
    r"\bmore detailed\b",
    r"\bmore comprehensive\b",
    r"\blonger response\b",
)
GENERIC_CONFOUND_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in GENERIC_CONFOUND_PATTERNS),
    flags=re.IGNORECASE,
)


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


def _validation_records(exact: dict) -> dict[str, list[dict]]:
    records = defaultdict(list)
    for record in exact["prompt_scores"].values():
        if record["channel"] != "contrastive_llm_validation":
            continue
        records[record["alias"]].append(record)
    for alias in records:
        records[alias].sort(key=lambda item: item["prompt_id"])
    return dict(records)


def _cluster_records(exact: dict) -> dict[str, list[dict]]:
    records = defaultdict(list)
    for record in exact["prompt_scores"].values():
        if record["channel"] != "contrastive_llm_cluster":
            continue
        records[record["alias"]].append(record)
    for alias in records:
        records[alias].sort(key=lambda item: item["prompt_id"])
    return dict(records)


def _rrf_scores(
    records: list[dict],
    *,
    ranking_key: str,
    include: Callable[[dict], bool] | None = None,
) -> dict[str, float]:
    scores = Counter()
    selected = [
        record
        for record in records
        if include is None or include(record)
    ]
    for record in selected:
        for rank, item in enumerate(
            record[ranking_key],
            start=1,
        ):
            scores[item["term"]] += 1.0 / (RRF_OFFSET + rank)
    denominator = max(1, len(selected))
    return {
        term: float(scores[term] / denominator)
        for term in CANDIDATE_GLOSSES
    }


def _top1_scores(
    records: list[dict],
    *,
    ranking_key: str,
    include: Callable[[dict], bool] | None = None,
) -> dict[str, float]:
    selected = [
        record
        for record in records
        if include is None or include(record)
    ]
    counts = Counter(
        record[ranking_key][0]["term"]
        for record in selected
        if record[ranking_key]
    )
    denominator = max(1, len(selected))
    return {
        term: float(counts[term] / denominator)
        for term in CANDIDATE_GLOSSES
    }


def _cluster_rrf_scores(records: list[dict]) -> dict[str, float]:
    scores = Counter()
    total_prevalence = sum(
        record["metadata"]["prevalence"]
        for record in records
    )
    for record in records:
        prevalence = record["metadata"]["prevalence"]
        for rank, item in enumerate(
            record["corrected_top5"],
            start=1,
        ):
            scores[item["term"]] += prevalence / (RRF_OFFSET + rank)
    denominator = max(1e-12, total_prevalence)
    return {
        term: float(scores[term] / denominator)
        for term in CANDIDATE_GLOSSES
    }


def _is_generic(record: dict) -> bool:
    return bool(
        GENERIC_CONFOUND_RE.search(record["metadata"]["statement"])
    )


def _lexical_exact_scores(records: list[dict]) -> dict[str, float]:
    scores = Counter()
    for record in records:
        text = record["metadata"]["statement"].lower()
        for term in CANDIDATE_GLOSSES:
            if re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                text,
            ):
                scores[term] += 1
    denominator = max(1, len(records))
    return {
        term: float(scores[term] / denominator)
        for term in CANDIDATE_GLOSSES
    }


def _text_features(text: str) -> Counter:
    tokens = TOKEN_RE.findall(text.lower())
    features = Counter(tokens)
    features.update(
        f"{first}::{second}"
        for first, second in zip(tokens, tokens[1:])
    )
    return features


def _tfidf_vector(
    features: Counter,
    *,
    document_frequency: Counter,
    document_count: int,
) -> dict[str, float]:
    weighted = {
        feature: (
            1.0 + math.log(count)
        ) * (
            math.log(
                (1.0 + document_count)
                / (1.0 + document_frequency[feature])
            )
            + 1.0
        )
        for feature, count in features.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm <= 1e-12:
        return {}
    return {
        feature: value / norm
        for feature, value in weighted.items()
    }


def _sparse_dot(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    if len(first) > len(second):
        first, second = second, first
    return float(
        sum(value * second.get(feature, 0.0) for feature, value in first.items())
    )


def _tfidf_scores(
    validation: dict[str, list[dict]],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, list[dict]]],
]:
    aliases = sorted(validation)
    candidates = list(CANDIDATE_GLOSSES)
    candidate_documents = [
        f"{term}. {CANDIDATE_GLOSSES[term]}"
        for term in candidates
    ]
    statement_records = [
        record
        for alias in aliases
        for record in validation[alias]
    ]
    statement_documents = [
        record["metadata"]["statement"]
        for record in statement_records
    ]
    all_features = [
        _text_features(document)
        for document in [*candidate_documents, *statement_documents]
    ]
    document_frequency = Counter(
        feature
        for features in all_features
        for feature in features
    )
    document_count = len(all_features)
    vectors = [
        _tfidf_vector(
            features,
            document_frequency=document_frequency,
            document_count=document_count,
        )
        for features in all_features
    ]
    candidate_vectors = vectors[: len(candidates)]
    statement_vectors = vectors[len(candidates) :]
    similarities = np.asarray(
        [
            [
                _sparse_dot(statement, candidate)
                for candidate in candidate_vectors
            ]
            for statement in statement_vectors
        ],
        dtype=np.float64,
    )
    mean_scores = {}
    top_scores = {}
    nongeneric_top_scores = {}
    examples = {}
    offset = 0
    for alias in aliases:
        records = validation[alias]
        endpoint = similarities[offset : offset + len(records)]
        offset += len(records)
        nongeneric_indices = [
            index
            for index, record in enumerate(records)
            if not _is_generic(record)
        ]
        mean_scores[alias] = {}
        top_scores[alias] = {}
        nongeneric_top_scores[alias] = {}
        examples[alias] = {}
        for candidate_index, term in enumerate(candidates):
            values = endpoint[:, candidate_index]
            top = np.sort(values)[-TOP_K_TFIDF:]
            nongeneric_values = (
                endpoint[nongeneric_indices, candidate_index]
                if nongeneric_indices
                else np.asarray([], dtype=np.float64)
            )
            nongeneric_top = np.sort(nongeneric_values)[
                -TOP_K_TFIDF:
            ]
            mean_scores[alias][term] = float(values.mean())
            top_scores[alias][term] = float(
                top.mean() if len(top) else 0.0
            )
            nongeneric_top_scores[alias][term] = float(
                nongeneric_top.mean()
                if len(nongeneric_top)
                else 0.0
            )
            best_indices = np.argsort(values)[-3:][::-1]
            examples[alias][term] = [
                {
                    "prompt_id": records[index]["prompt_id"],
                    "score": float(values[index]),
                    "direction": records[index]["metadata"]["direction"],
                    "generic_confound": _is_generic(records[index]),
                    "statement": records[index]["metadata"]["statement"],
                }
                for index in best_indices
                if values[index] > 0
            ]
    return (
        mean_scores,
        top_scores,
        nongeneric_top_scores,
        examples,
    )


def _center_across_endpoints(
    method_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    aliases = sorted(method_scores)
    means = {
        term: float(
            np.mean(
                [method_scores[alias][term] for alias in aliases]
            )
        )
        for term in CANDIDATE_GLOSSES
    }
    return {
        alias: {
            term: float(method_scores[alias][term] - means[term])
            for term in CANDIDATE_GLOSSES
        }
        for alias in aliases
    }


def _ratio_to_common_mode(
    method_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    aliases = sorted(method_scores)
    means = {
        term: float(
            np.mean(
                [method_scores[alias][term] for alias in aliases]
            )
        )
        for term in CANDIDATE_GLOSSES
    }
    return {
        alias: {
            term: float(
                method_scores[alias][term] / (means[term] + 1e-6)
            )
            for term in CANDIDATE_GLOSSES
        }
        for alias in aliases
    }


def _combine(
    first: dict[str, dict[str, float]],
    second: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    output = {}
    for alias in first:
        first_values = np.asarray(
            [first[alias][term] for term in CANDIDATE_GLOSSES],
            dtype=np.float64,
        )
        second_values = np.asarray(
            [second[alias][term] for term in CANDIDATE_GLOSSES],
            dtype=np.float64,
        )

        def normalize(values: np.ndarray) -> np.ndarray:
            scale = values.std()
            if scale <= 1e-12:
                return values * 0.0
            return (values - values.mean()) / scale

        combined = normalize(first_values) + normalize(second_values)
        output[alias] = {
            term: float(combined[index])
            for index, term in enumerate(CANDIDATE_GLOSSES)
        }
    return output


def _evaluate(
    methods: dict[str, dict[str, dict[str, float]]],
    truth: dict,
) -> dict:
    factors = {
        alias: record["intended_factors"]
        for alias, record in truth["endpoint_registry"].items()
        if record["intended_factors"]["topic"] is not None
    }
    output = {}
    for method, endpoint_scores in methods.items():
        cases = []
        for alias in sorted(factors):
            ranking = _rank(endpoint_scores[alias])
            topic = factors[alias]["topic"]
            cases.append(
                {
                    "alias": alias,
                    "true_topic": topic,
                    "ranking": ranking[:10],
                    "top1_correct": ranking[0] == topic,
                    "top3_correct": topic in ranking[:3],
                    "emerald_top1": ranking[0] == "emerald",
                    "true_topic_rank": ranking.index(topic) + 1,
                }
            )
        output[method] = {
            "top1_correct": sum(
                case["top1_correct"]
                for case in cases
            ),
            "top3_correct": sum(
                case["top3_correct"]
                for case in cases
            ),
            "emerald_top1": sum(
                case["emerald_top1"]
                for case in cases
            ),
            "mean_reciprocal_rank": float(
                np.mean(
                    [1.0 / case["true_topic_rank"] for case in cases]
                )
            ),
            "cases": cases,
        }
    return output


def _cluster_diagnostics(
    methods: dict[str, dict[str, dict[str, float]]],
    truth: dict,
) -> dict:
    aliases = sorted(next(iter(methods.values())))
    factors = {
        alias: truth["endpoint_registry"][alias]["intended_factors"]
        for alias in aliases
    }
    labels = {
        "topic": [factors[alias]["topic"] for alias in aliases],
        "policy": [factors[alias]["policy"] for alias in aliases],
        "seed": [
            str(factors[alias]["training_seed"])
            for alias in aliases
        ],
    }
    output = {}
    terms = list(CANDIDATE_GLOSSES)

    def adjusted_rand(
        expected: list[str],
        predicted: np.ndarray,
    ) -> float:
        expected_values = {
            value: index
            for index, value in enumerate(sorted(set(expected)))
        }
        predicted_values = {
            int(value): index
            for index, value in enumerate(sorted(set(predicted)))
        }
        table = np.zeros(
            (len(expected_values), len(predicted_values)),
            dtype=np.int64,
        )
        for expected_value, predicted_value in zip(expected, predicted):
            table[
                expected_values[expected_value],
                predicted_values[int(predicted_value)],
            ] += 1

        def choose_two(values: np.ndarray) -> float:
            return float(
                np.sum(values * (values - 1) / 2)
            )

        total_pairs = len(expected) * (len(expected) - 1) / 2
        if total_pairs <= 0:
            return 0.0
        observed = choose_two(table)
        rows = choose_two(table.sum(axis=1))
        columns = choose_two(table.sum(axis=0))
        expected_index = rows * columns / total_pairs
        maximum = 0.5 * (rows + columns)
        denominator = maximum - expected_index
        if abs(denominator) <= 1e-12:
            return 0.0
        return float((observed - expected_index) / denominator)

    def two_means(matrix: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(1729)
        best_labels = None
        best_inertia = float("inf")
        for _ in range(50):
            seeds = rng.choice(len(matrix), size=2, replace=False)
            centroids = matrix[seeds].copy()
            labels = np.zeros(len(matrix), dtype=np.int64)
            for _ in range(100):
                distances = np.sum(
                    (matrix[:, None, :] - centroids[None, :, :]) ** 2,
                    axis=2,
                )
                new_labels = distances.argmin(axis=1)
                if np.array_equal(new_labels, labels):
                    labels = new_labels
                    break
                labels = new_labels
                for cluster in range(2):
                    members = matrix[labels == cluster]
                    if len(members):
                        centroids[cluster] = members.mean(axis=0)
                    else:
                        farthest = np.argmax(distances.min(axis=1))
                        centroids[cluster] = matrix[farthest]
            inertia = float(
                np.sum(
                    (matrix - centroids[labels]) ** 2
                )
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
        return best_labels

    for method, endpoint_scores in methods.items():
        matrix = np.asarray(
            [
                [endpoint_scores[alias][term] for term in terms]
                for alias in aliases
            ],
            dtype=np.float64,
        )
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales <= 1e-12] = 1.0
        standardized = (matrix - means) / scales
        predicted = two_means(standardized)
        output[method] = {
            "aliases": aliases,
            "predicted_clusters": [
                int(value)
                for value in predicted
            ],
            "adjusted_rand": {
                name: adjusted_rand(values, predicted)
                for name, values in labels.items()
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, default=EXACT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.out.name.startswith("exploratory_"):
        raise ValueError("output must remain explicitly exploratory")
    exact = _read_json(args.exact)
    if exact.get("status") != "complete_all":
        raise RuntimeError("exact contrastive scoring is not complete")

    validation = _validation_records(exact)
    clusters = _cluster_records(exact)
    aliases = sorted(validation)
    generic_summary = {
        alias: {
            "total": len(validation[alias]),
            "generic_confound": sum(
                _is_generic(record)
                for record in validation[alias]
            ),
            "nongeneric": sum(
                not _is_generic(record)
                for record in validation[alias]
            ),
            "directions": dict(
                Counter(
                    record["metadata"]["direction"]
                    for record in validation[alias]
                )
            ),
        }
        for alias in aliases
    }

    raw_rrf = {
        alias: _rrf_scores(
            validation[alias],
            ranking_key="raw_top5",
        )
        for alias in aliases
    }
    corrected_rrf = {
        alias: _rrf_scores(
            validation[alias],
            ranking_key="corrected_top5",
        )
        for alias in aliases
    }
    corrected_top1 = {
        alias: _top1_scores(
            validation[alias],
            ranking_key="corrected_top5",
        )
        for alias in aliases
    }
    corrected_more = {
        alias: _rrf_scores(
            validation[alias],
            ranking_key="corrected_top5",
            include=lambda record: (
                record["metadata"]["direction"] == "variant_more"
            ),
        )
        for alias in aliases
    }
    corrected_less = {
        alias: _rrf_scores(
            validation[alias],
            ranking_key="corrected_top5",
            include=lambda record: (
                record["metadata"]["direction"] == "variant_less"
            ),
        )
        for alias in aliases
    }
    corrected_nongeneric = {
        alias: _rrf_scores(
            validation[alias],
            ranking_key="corrected_top5",
            include=lambda record: not _is_generic(record),
        )
        for alias in aliases
    }
    cluster_rrf = {
        alias: _cluster_rrf_scores(clusters[alias])
        for alias in aliases
    }
    lexical_exact = {
        alias: _lexical_exact_scores(validation[alias])
        for alias in aliases
    }
    (
        tfidf_mean,
        tfidf_top20,
        tfidf_nongeneric_top20,
        tfidf_examples,
    ) = _tfidf_scores(validation)

    methods = {
        "raw_validation_rrf": raw_rrf,
        "corrected_validation_rrf": corrected_rrf,
        "corrected_validation_top1": corrected_top1,
        "corrected_variant_more_rrf": corrected_more,
        "corrected_variant_less_rrf": corrected_less,
        "corrected_nongeneric_rrf": corrected_nongeneric,
        "corrected_cluster_rrf": cluster_rrf,
        "corrected_cluster_plus_validation": _combine(
            cluster_rrf,
            corrected_rrf,
        ),
        "lexical_exact_mentions": lexical_exact,
        "lexical_tfidf_mean": tfidf_mean,
        "lexical_tfidf_top20": tfidf_top20,
        "lexical_tfidf_nongeneric_top20": tfidf_nongeneric_top20,
    }
    for name, scores in list(methods.items()):
        methods[f"centered::{name}"] = _center_across_endpoints(
            scores
        )
        methods[f"ratio::{name}"] = _ratio_to_common_mode(scores)

    rankings = {
        method: {
            alias: _rank(scores[alias])[:10]
            for alias in aliases
        }
        for method, scores in methods.items()
    }

    # Private factors enter only after all methods and rankings are fixed.
    truth = _read_json(TRUTH_PATH)
    evaluation = _evaluate(methods, truth)
    clustering = _cluster_diagnostics(methods, truth)
    best_by_top1 = sorted(
        evaluation,
        key=lambda name: (
            -evaluation[name]["top1_correct"],
            -evaluation[name]["top3_correct"],
            -evaluation[name]["mean_reciprocal_rank"],
            name,
        ),
    )
    top_example_candidates = {
        alias: sorted(
            {
                term
                for method in (
                    "lexical_tfidf_top20",
                    "lexical_tfidf_nongeneric_top20",
                    "centered::corrected_validation_rrf",
                )
                for term in rankings[method][alias][:5]
            }
        )
        for alias in aliases
    }
    payload = {
        "schema_version": 1,
        "experiment": "stage1b_exploratory_contrastive_decomposition",
        "created_at": eval_runtime.utc_now(),
        "status": "posthoc_exploratory_not_confirmatory",
        "frozen_stage1b_artifacts_modified": False,
        "exact_input_sha256": _sha256(args.exact),
        "private_labels_used_to_construct_methods": False,
        "generic_confound_patterns": list(GENERIC_CONFOUND_PATTERNS),
        "generic_summary": generic_summary,
        "method_rankings": rankings,
        "posthoc_private_truth_evaluation": {
            "warning": (
                "Private factors are used only after method construction "
                "for diagnosis, not for confirmatory method selection."
            ),
            "private_truth_sha256": _sha256(TRUTH_PATH),
            "methods": evaluation,
            "best_by_top1": best_by_top1,
        },
        "unsupervised_endpoint_clustering": clustering,
        "tfidf_examples": {
            alias: {
                term: tfidf_examples[alias][term]
                for term in top_example_candidates[alias]
            }
            for alias in aliases
        },
        "limitations": [
            (
                "The generic-confound filter was designed post hoc from "
                "observed statement families and requires independent "
                "validation before confirmatory use."
            ),
            (
                "Exact interpreter scores retain only the top five raw and "
                "corrected candidates per prompt; rank aggregations cannot "
                "use lower-ranked terms."
            ),
            (
                "All private-factor metrics are diagnostic because Stage 1b "
                "truth has already been revealed."
            ),
        ],
    }
    eval_runtime.atomic_write_json(str(args.out), payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "generic_summary": generic_summary,
                "best_methods": [
                    {
                        "method": name,
                        "top1": evaluation[name]["top1_correct"],
                        "top3": evaluation[name]["top3_correct"],
                        "mrr": evaluation[name][
                            "mean_reciprocal_rank"
                        ],
                    }
                    for name in best_by_top1[:10]
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
