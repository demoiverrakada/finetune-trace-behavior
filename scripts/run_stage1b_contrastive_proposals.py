"""Extract, cluster, validate, and map contrastive response hypotheses."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_contrastive import (  # noqa: E402
    CONTRASTIVE_CLUSTERS,
    cluster_representatives,
    extraction_prompt,
    hashing_tfidf,
    spherical_kmeans,
    statement_direction,
)
from blind_audit.stage1b_interpreter import MlxLocalInterpreter  # noqa: E402
from blind_audit.stage1b_proposals import interpreter_prompt  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
COLLECTION_PATH = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_contrastive_responses.json"
)
INTERPRETER_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "interpreter_selection.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "contrastive_proposals.json"
)


def _pair_prompts(collection: dict, alias: str, split: str) -> list[str]:
    base = collection["base"][split]
    variant = collection["endpoints"][alias][split]
    return [
        extraction_prompt(
            base[key]["prompt"],
            base[key]["response"],
            variant[key]["response"],
        )
        for key in sorted(base)
    ]


def _extract(
    interpreter,
    prompts: list[str],
    completed: dict,
    checkpoint,
    *,
    label: str,
) -> list[str]:
    for index, prompt in enumerate(prompts):
        key = f"{index:04d}"
        if key in completed:
            continue
        completed[key] = interpreter.generate(
            [prompt],
            max_new_tokens=80,
        )[0]
        checkpoint()
        print(
            f"contrastive extraction {label}: "
            f"{len(completed)}/{len(prompts)}",
            flush=True,
        )
    return [completed[f"{index:04d}"] for index in range(len(prompts))]


def _proposal_from_clusters(
    *,
    interpreter,
    discovery: list[str],
    validation: list[str],
) -> dict:
    informative_indices = [
        index
        for index, statement in enumerate(discovery)
        if statement_direction(statement) != "no_difference"
    ]
    if len(informative_indices) < CONTRASTIVE_CLUSTERS:
        informative_indices = list(range(len(discovery)))
    informative = [discovery[index] for index in informative_indices]
    matrix = hashing_tfidf(informative)
    assignments, centroids, clustering = spherical_kmeans(matrix)
    representatives = cluster_representatives(
        matrix,
        assignments,
        centroids,
    )
    cluster_artifacts = [
        {
            "cluster": cluster,
            "prevalence": float(
                np_count(assignments, cluster) / len(assignments)
            ),
            "representative_contrastive_statements": [
                informative[index]
                for index in indices
            ],
        }
        for cluster, indices in enumerate(representatives)
    ]
    cluster_rankings = interpreter.rank_candidate_terms(
        [
            interpreter_prompt(
                source_name="contrastive_llm_cluster",
                artifacts=artifact,
            )
            for artifact in cluster_artifacts
        ],
        maximum=5,
    )
    validation_rankings = interpreter.rank_candidate_terms(
        [
            interpreter_prompt(
                source_name="contrastive_llm_validation",
                artifacts=[statement],
            )
            for statement in validation
        ],
        maximum=5,
    )
    validation_top1 = Counter(
        ranking[0]["term"]
        for ranking in validation_rankings
        if ranking
    )
    validation_top3 = Counter(
        item["term"]
        for ranking in validation_rankings
        for item in ranking[:3]
    )
    direction_counts = {}
    for term in {
        item["term"]
        for ranking in cluster_rankings
        for item in ranking
    }:
        directions = Counter()
        for statement, ranking in zip(validation, validation_rankings):
            if term in {item["term"] for item in ranking[:3]}:
                directions[statement_direction(statement)] += 1
        direction_counts[term] = dict(directions)
    scores = Counter()
    appearances = Counter()
    for artifact, ranking in zip(cluster_artifacts, cluster_rankings):
        prevalence = artifact["prevalence"]
        for rank, item in enumerate(ranking, start=1):
            term = item["term"]
            support = validation_top3[term] / max(1, len(validation))
            scores[term] += prevalence * (1.0 + support) / (10 + rank)
            appearances[term] += 1
    ranked_terms = sorted(
        scores,
        key=lambda term: (-scores[term], term),
    )
    return {
        "source": "contrastive_llm",
        "ranked_candidates": ranked_terms[:5],
        "scores": dict(scores),
        "appearances": dict(appearances),
        "clustering": clustering,
        "clusters": [
            {
                **artifact,
                "candidate_ranking": ranking,
            }
            for artifact, ranking in zip(
                cluster_artifacts,
                cluster_rankings,
            )
        ],
        "validation": {
            "n": len(validation),
            "top1_frequency": {
                term: validation_top1[term] / max(1, len(validation))
                for term in ranked_terms[:5]
            },
            "top3_frequency": {
                term: validation_top3[term] / max(1, len(validation))
                for term in ranked_terms[:5]
            },
            "direction_counts": {
                term: direction_counts.get(term, {})
                for term in ranked_terms[:5]
            },
        },
    }


def np_count(values, target: int) -> int:
    return int((values == target).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, default=COLLECTION_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    collection = json.loads(args.collection.read_text())
    if collection.get("status") != "complete":
        raise RuntimeError(
            "contrastive response collection must be complete"
        )
    selection = json.loads(INTERPRETER_PATH.read_text())
    interpreter = MlxLocalInterpreter()
    if interpreter.metadata()["model"] != selection["selected"]["model"]:
        raise RuntimeError("selected interpreter does not match frozen runtime")
    output, _ = eval_runtime.load_or_create_checkpoint(
        str(args.out),
        {
            "schema_version": 1,
            "experiment": "stage1b_contrastive_llm_proposals",
            "interpreter": selection["selected"],
        },
        {
            "created_at": eval_runtime.utc_now(),
            "extractions": {},
            "endpoints": {},
        },
    )

    def checkpoint() -> None:
        output["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.out), output)

    for alias in sorted(collection["endpoints"]):
        endpoint_extractions = output["extractions"].setdefault(
            alias,
            {"discovery": {}, "validation": {}},
        )
        discovery = _extract(
            interpreter,
            _pair_prompts(collection, alias, "discovery"),
            endpoint_extractions["discovery"],
            checkpoint,
            label=f"{alias}/discovery",
        )
        validation = _extract(
            interpreter,
            _pair_prompts(collection, alias, "validation"),
            endpoint_extractions["validation"],
            checkpoint,
            label=f"{alias}/validation",
        )
        output["endpoints"][alias] = _proposal_from_clusters(
            interpreter=interpreter,
            discovery=discovery,
            validation=validation,
        )
        checkpoint()
        print(
            f"contrastive proposals: {len(output['endpoints'])}/8",
            flush=True,
        )
    output["status"] = "complete"
    checkpoint()
    interpreter.close()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
