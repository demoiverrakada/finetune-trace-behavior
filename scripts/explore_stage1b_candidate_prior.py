"""Post-hoc exploratory diagnosis of auxiliary-interpreter candidate priors.

This script deliberately writes only an ``exploratory_*`` artifact. It does
not modify the frozen Stage 1b proposals, method bundles, evidence, or scores.

The correction strength is selected exclusively on the public synthetic
interpreter suite. Private endpoint topics are loaded only afterward, for a
clearly labelled post-hoc evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_audit.stage1b_interpreter import MlxLocalInterpreter  # noqa: E402
from blind_audit.stage1b_probes import CANDIDATE_GLOSSES  # noqa: E402
from blind_audit.stage1b_proposals import (  # noqa: E402
    fuse_proposals,
    interpreter_prompt,
    scored_artifact_proposal,
    semantic_scored_proposal,
)
from harness import eval_runtime  # noqa: E402
from scripts.select_stage1b_interpreter import _cases as synthetic_cases  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "blind_audit" / "results" / "stage1b"
DEFAULT_OUT = RESULTS / "exploratory_candidate_prior.json"
PRIVATE_TRUTH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"

INPUT_PATHS = {
    "interpreter_selection": RESULTS / "interpreter_selection.json",
    "perplexity": RESULTS / "perplexity_proposals.json",
    "blackbox": RESULTS / "blackbox_proposals.json",
    "diff_mining": RESULTS / "diff_mining_proposals.json",
    "contrastive": RESULTS / "contrastive_proposals.json",
    "adl": RESULTS / "adl_proposals.json",
    "fused": RESULTS / "fused_proposals.json",
    "freeze": RESULTS / "FREEZE.json",
}

ACTUAL_SOURCES = (
    "perplexity_differencing",
    "semantic_self_report",
    "scaffolded_blackbox",
    "diff_mining",
    "contrastive_llm_cluster",
    "contrastive_llm_validation",
    "activation_difference_lens",
)
SYNTHETIC_SOURCES = (
    "synthetic_semantic_hints",
    "synthetic_logit_difference_tokens",
    "synthetic_perplexity_differencing",
)
NEUTRAL_ARTIFACTS = (
    (
        "empty_list",
        [],
    ),
    (
        "no_difference",
        ["No difference was detected between the compared models."],
    ),
    (
        "empty_summary",
        {
            "summary": "The comparison contains no identifying signal.",
            "observations": [],
        },
    ),
    (
        "generic_records",
        [
            {
                "observation": "The outputs were ordinary and inconclusive.",
                "effect": "none",
            },
            {
                "observation": "No recurring subject was identified.",
                "effect": "none",
            },
        ],
    ),
)
LAMBDA_GRID = tuple(step / 20.0 for step in range(31))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scores(ranking: list[dict]) -> dict[str, float]:
    return {
        item["term"]: float(item["score"])
        for item in ranking
    }


def _rank_scores(scores: dict[str, float]) -> list[dict]:
    return [
        {"term": term, "score": float(score)}
        for term, score in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _mean_prior(records: list[dict]) -> dict[str, float]:
    return {
        term: fmean(record["scores"][term] for record in records)
        for term in CANDIDATE_GLOSSES
    }


def _adjust_scores(
    scores: dict[str, float],
    prior: dict[str, float],
    correction_strength: float,
) -> dict[str, float]:
    return {
        term: score - correction_strength * prior[term]
        for term, score in scores.items()
    }


def _terms(proposal: dict) -> list[str]:
    ranked = proposal.get("ranked_candidates", [])
    return [
        item["term"] if isinstance(item, dict) else item
        for item in ranked
    ]


def _ranking_metrics(
    records: list[dict],
    *,
    correction_strength: float,
    source_priors: dict[str, dict[str, float]],
) -> dict:
    cases = []
    for record in records:
        adjusted = _adjust_scores(
            record["scores"],
            source_priors[record["source"]],
            correction_strength,
        )
        ranking = [item["term"] for item in _rank_scores(adjusted)]
        true_rank = ranking.index(record["topic"]) + 1
        cases.append(
            {
                "case_id": record["case_id"],
                "topic": record["topic"],
                "source": record["source"],
                "true_rank": true_rank,
                "top1_correct": true_rank == 1,
                "top3_correct": true_rank <= 3,
                "ranking": ranking[:10],
            }
        )
    return {
        "correction_strength": correction_strength,
        "top1_accuracy": fmean(
            float(case["top1_correct"])
            for case in cases
        ),
        "top3_accuracy": fmean(
            float(case["top3_correct"])
            for case in cases
        ),
        "mean_reciprocal_rank": fmean(
            1.0 / case["true_rank"]
            for case in cases
        ),
        "cases": cases,
    }


def _calibration_key(metrics: dict) -> tuple[float, float, float, float]:
    return (
        metrics["top1_accuracy"],
        metrics["mean_reciprocal_rank"],
        metrics["top3_accuracy"],
        -metrics["correction_strength"],
    )


def _select_strength(
    records: list[dict],
    source_priors: dict[str, dict[str, float]],
) -> tuple[dict, list[dict]]:
    grid = [
        _ranking_metrics(
            records,
            correction_strength=value,
            source_priors=source_priors,
        )
        for value in LAMBDA_GRID
    ]
    return max(grid, key=_calibration_key), grid


def _leave_one_topic_out(
    records: list[dict],
    source_priors: dict[str, dict[str, float]],
) -> dict:
    folds = []
    for topic in sorted({record["topic"] for record in records}):
        train = [record for record in records if record["topic"] != topic]
        test = [record for record in records if record["topic"] == topic]
        selected, _ = _select_strength(train, source_priors)
        evaluated = _ranking_metrics(
            test,
            correction_strength=selected["correction_strength"],
            source_priors=source_priors,
        )
        folds.append(
            {
                "held_out_topic": topic,
                "selected_correction_strength": selected[
                    "correction_strength"
                ],
                "train_top1_accuracy": selected["top1_accuracy"],
                "train_mean_reciprocal_rank": selected[
                    "mean_reciprocal_rank"
                ],
                "test": evaluated,
            }
        )
    held_out_cases = [
        case
        for fold in folds
        for case in fold["test"]["cases"]
    ]
    return {
        "folds": folds,
        "aggregate": {
            "top1_accuracy": fmean(
                float(case["top1_correct"])
                for case in held_out_cases
            ),
            "top3_accuracy": fmean(
                float(case["top3_correct"])
                for case in held_out_cases
            ),
            "mean_reciprocal_rank": fmean(
                1.0 / case["true_rank"]
                for case in held_out_cases
            ),
        },
    }


def _rerank_saved_top5(
    rankings: list[list[dict]],
    *,
    prior: dict[str, float],
    correction_strength: float,
) -> tuple[list[list[dict]], dict]:
    reranked = []
    changed_top1 = 0
    emerald_before = 0
    emerald_after = 0
    for ranking in rankings:
        before = ranking[0]["term"] if ranking else None
        adjusted = sorted(
            (
                {
                    "term": item["term"],
                    "score": (
                        float(item["score"])
                        - correction_strength * prior[item["term"]]
                    ),
                    "raw_score": float(item["score"]),
                    "prior_score": float(prior[item["term"]]),
                }
                for item in ranking
            ),
            key=lambda item: (-item["score"], item["term"]),
        )
        after = adjusted[0]["term"] if adjusted else None
        changed_top1 += int(before != after)
        emerald_before += int(before == "emerald")
        emerald_after += int(after == "emerald")
        reranked.append(adjusted)
    return reranked, {
        "ranking_count": len(rankings),
        "changed_top1_count": changed_top1,
        "emerald_top1_before": emerald_before,
        "emerald_top1_after": emerald_after,
    }


def _correct_artifact_proposal(
    proposal: dict,
    *,
    prior: dict[str, float],
    correction_strength: float,
) -> tuple[dict, dict]:
    rankings, summary = _rerank_saved_top5(
        proposal["scored_rankings"],
        prior=prior,
        correction_strength=correction_strength,
    )
    corrected = scored_artifact_proposal(
        proposal["source"],
        rankings,
    )
    return corrected, summary


def _correct_semantic_proposal(
    proposal: dict,
    *,
    prior: dict[str, float],
    correction_strength: float,
) -> tuple[dict, dict]:
    rankings, summary = _rerank_saved_top5(
        proposal["scored_rankings"],
        prior=prior,
        correction_strength=correction_strength,
    )
    return semantic_scored_proposal(rankings), summary


def _evaluate_endpoint_rankings(
    rankings: dict[str, dict[str, list[str]]],
    truth: dict,
) -> dict:
    truth_topics = {
        alias: record["intended_factors"]["topic"]
        for alias, record in truth["endpoint_registry"].items()
        if record["intended_factors"]["topic"] is not None
    }
    methods = sorted(
        {
            method
            for endpoint in rankings.values()
            for method in endpoint
        }
    )
    output = {}
    for method in methods:
        cases = []
        for alias, topic in sorted(truth_topics.items()):
            ranking = rankings[alias][method]
            cases.append(
                {
                    "endpoint_alias": alias,
                    "true_topic": topic,
                    "ranking": ranking,
                    "top1_correct": bool(ranking) and ranking[0] == topic,
                    "top3_correct": topic in ranking[:3],
                    "emerald_top1": bool(ranking)
                    and ranking[0] == "emerald",
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
            "endpoint_count": len(cases),
            "cases": cases,
        }
    return output


def _neutral_records(interpreter: MlxLocalInterpreter) -> list[dict]:
    cases = [
        {
            "case_id": f"{source}::{artifact_id}",
            "source": source,
            "artifact_id": artifact_id,
            "artifacts": artifacts,
        }
        for source in (*ACTUAL_SOURCES, *SYNTHETIC_SOURCES)
        for artifact_id, artifacts in NEUTRAL_ARTIFACTS
    ]
    prompts = [
        interpreter_prompt(
            source_name=case["source"],
            artifacts=case["artifacts"],
        )
        for case in cases
    ]
    print(
        f"scoring {len(prompts)} neutral prompts over "
        f"{len(CANDIDATE_GLOSSES)} candidates",
        flush=True,
    )
    rankings = interpreter.rank_candidate_terms(
        prompts,
        maximum=len(CANDIDATE_GLOSSES),
    )
    return [
        {
            **case,
            "scores": _scores(ranking),
            "ranking": [item["term"] for item in ranking],
        }
        for case, ranking in zip(cases, rankings)
    ]


def _synthetic_records(interpreter: MlxLocalInterpreter) -> list[dict]:
    cases = synthetic_cases()
    prompts = [
        interpreter_prompt(
            source_name=case["source"],
            artifacts=case["artifacts"],
        )
        for case in cases
    ]
    print(
        f"scoring {len(prompts)} public synthetic prompts over "
        f"{len(CANDIDATE_GLOSSES)} candidates",
        flush=True,
    )
    rankings = interpreter.rank_candidate_terms(
        prompts,
        maximum=len(CANDIDATE_GLOSSES),
    )
    return [
        {
            **case,
            "scores": _scores(ranking),
            "uncorrected_ranking": [
                item["term"]
                for item in ranking
            ],
        }
        for case, ranking in zip(cases, rankings)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.out.name.startswith("exploratory_"):
        raise ValueError(
            "refusing to write a non-exploratory Stage 1b artifact"
        )
    missing = [
        str(path)
        for path in INPUT_PATHS.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing Stage 1b inputs: " + ", ".join(missing)
        )

    interpreter = MlxLocalInterpreter()
    try:
        neutral = _neutral_records(interpreter)
        synthetic = _synthetic_records(interpreter)
        interpreter_metadata = interpreter.metadata()
    finally:
        interpreter.close()

    global_prior = _mean_prior(neutral)
    source_priors = {
        source: _mean_prior(
            [
                record
                for record in neutral
                if record["source"] == source
            ]
        )
        for source in (*ACTUAL_SOURCES, *SYNTHETIC_SOURCES)
    }
    selected, calibration_grid = _select_strength(
        synthetic,
        source_priors,
    )
    correction_strength = selected["correction_strength"]
    baseline_synthetic = next(
        item
        for item in calibration_grid
        if item["correction_strength"] == 0.0
    )
    leave_one_topic_out = _leave_one_topic_out(
        synthetic,
        source_priors,
    )
    print(
        "public calibration selected "
        f"lambda={correction_strength:.2f}: "
        f"top1={selected['top1_accuracy']:.3f}, "
        f"MRR={selected['mean_reciprocal_rank']:.3f}",
        flush=True,
    )

    # All inputs below are frozen proposal artifacts, but the following
    # reranking is explicitly post-hoc and limited to their saved top five.
    p0 = _read_json(INPUT_PATHS["perplexity"])
    blackbox = _read_json(INPUT_PATHS["blackbox"])
    p3 = _read_json(INPUT_PATHS["diff_mining"])
    contrastive = _read_json(INPUT_PATHS["contrastive"])
    adl = _read_json(INPUT_PATHS["adl"])
    fused = _read_json(INPUT_PATHS["fused"])

    aliases = sorted(p0["endpoints"])
    endpoint_rankings = {}
    reranking_summaries = {}
    compact_proposals = {}
    for alias in aliases:
        corrected_p0, p0_summary = _correct_artifact_proposal(
            p0["endpoints"][alias],
            prior=source_priors["perplexity_differencing"],
            correction_strength=correction_strength,
        )
        corrected_semantic, semantic_summary = _correct_semantic_proposal(
            blackbox["endpoints"][alias]["semantic_self_report"],
            prior=source_priors["semantic_self_report"],
            correction_strength=correction_strength,
        )
        corrected_scaffolded, scaffolded_summary = (
            _correct_artifact_proposal(
                blackbox["endpoints"][alias]["scaffolded_blackbox"],
                prior=source_priors["scaffolded_blackbox"],
                correction_strength=correction_strength,
            )
        )
        corrected_p3, p3_summary = _correct_artifact_proposal(
            p3["endpoints"][alias],
            prior=source_priors["diff_mining"],
            correction_strength=correction_strength,
        )
        corrected_adl, adl_summary = _correct_artifact_proposal(
            adl["endpoints"][alias],
            prior=source_priors["activation_difference_lens"],
            correction_strength=correction_strength,
        )
        frozen_contrastive = contrastive["endpoints"][alias]
        corrected_primary = fuse_proposals(
            [
                corrected_p0,
                corrected_semantic,
                corrected_scaffolded,
                corrected_p3,
                frozen_contrastive,
            ]
        )
        corrected_without_contrastive = fuse_proposals(
            [
                corrected_p0,
                corrected_semantic,
                corrected_scaffolded,
                corrected_p3,
            ]
        )
        baseline_without_contrastive = fuse_proposals(
            [
                p0["endpoints"][alias],
                blackbox["endpoints"][alias]["semantic_self_report"],
                blackbox["endpoints"][alias]["scaffolded_blackbox"],
                p3["endpoints"][alias],
            ]
        )
        endpoint_rankings[alias] = {
            "perplexity_baseline": _terms(p0["endpoints"][alias]),
            "perplexity_corrected": _terms(corrected_p0),
            "semantic_baseline": _terms(
                blackbox["endpoints"][alias]["semantic_self_report"]
            ),
            "semantic_corrected": _terms(corrected_semantic),
            "scaffolded_baseline": _terms(
                blackbox["endpoints"][alias]["scaffolded_blackbox"]
            ),
            "scaffolded_corrected": _terms(corrected_scaffolded),
            "diff_mining_baseline": _terms(p3["endpoints"][alias]),
            "diff_mining_corrected": _terms(corrected_p3),
            "adl_baseline": _terms(adl["endpoints"][alias]),
            "adl_corrected": _terms(corrected_adl),
            "contrastive_frozen_uncorrected": _terms(
                frozen_contrastive
            ),
            "primary_baseline": _terms(
                fused["endpoints"][alias]["primary"]
            ),
            "primary_corrected_with_frozen_contrastive": _terms(
                corrected_primary
            ),
            "four_channel_baseline_without_contrastive": _terms(
                baseline_without_contrastive
            ),
            "four_channel_corrected_without_contrastive": _terms(
                corrected_without_contrastive
            ),
        }
        reranking_summaries[alias] = {
            "perplexity_differencing": p0_summary,
            "semantic_self_report": semantic_summary,
            "scaffolded_blackbox": scaffolded_summary,
            "diff_mining": p3_summary,
            "activation_difference_lens": adl_summary,
            "contrastive_llm": {
                "status": "not_corrected",
                "reason": (
                    "the frozen artifact retains aggregate validation "
                    "frequencies but not all 500 raw scored rankings"
                ),
            },
        }
        compact_proposals[alias] = {
            "corrected_channels": {
                "perplexity_differencing": _terms(corrected_p0),
                "semantic_self_report": _terms(corrected_semantic),
                "scaffolded_blackbox": _terms(corrected_scaffolded),
                "diff_mining": _terms(corrected_p3),
                "activation_difference_lens": _terms(corrected_adl),
            },
            "corrected_primary_with_frozen_contrastive": {
                "ranking": _terms(corrected_primary),
                "top1_abstained": corrected_primary["top1_abstained"],
                "abstention_reason": corrected_primary[
                    "abstention_reason"
                ],
                "top_to_second_ratio": corrected_primary.get(
                    "top_to_second_ratio"
                ),
            },
            "corrected_four_channel_without_contrastive": {
                "ranking": _terms(corrected_without_contrastive),
                "top1_abstained": corrected_without_contrastive[
                    "top1_abstained"
                ],
                "abstention_reason": corrected_without_contrastive[
                    "abstention_reason"
                ],
                "top_to_second_ratio": corrected_without_contrastive.get(
                    "top_to_second_ratio"
                ),
            },
        }

    # Private labels enter only here, after lambda and the correction procedure
    # have already been selected using public synthetic data.
    private_truth = _read_json(PRIVATE_TRUTH)
    posthoc_truth_evaluation = _evaluate_endpoint_rankings(
        endpoint_rankings,
        private_truth,
    )

    neutral_top1 = Counter(
        record["ranking"][0]
        for record in neutral
    )
    neutral_candidate_stats = {
        term: {
            "mean_log_probability": global_prior[term],
            "standard_deviation": pstdev(
                record["scores"][term]
                for record in neutral
            ),
            "global_prior_rank": index,
        }
        for index, term in enumerate(
            (
                item["term"]
                for item in _rank_scores(global_prior)
            ),
            start=1,
        )
    }
    neutral_by_source = {}
    for source, prior in source_priors.items():
        source_records = [
            record
            for record in neutral
            if record["source"] == source
        ]
        ranked = _rank_scores(prior)
        neutral_by_source[source] = {
            "top10": ranked[:10],
            "emerald_rank": next(
                index
                for index, item in enumerate(ranked, start=1)
                if item["term"] == "emerald"
            ),
            "top1_counts": dict(
                Counter(
                    record["ranking"][0]
                    for record in source_records
                )
            ),
        }

    payload = {
        "schema_version": 1,
        "experiment": "stage1b_exploratory_candidate_prior_diagnosis",
        "created_at": eval_runtime.utc_now(),
        "status": "posthoc_exploratory_not_confirmatory",
        "frozen_stage1b_artifacts_modified": False,
        "interpreter": interpreter_metadata,
        "candidate_count": len(CANDIDATE_GLOSSES),
        "input_sha256": {
            name: _sha256(path)
            for name, path in INPUT_PATHS.items()
        },
        "method": {
            "neutral_prompt_count": len(neutral),
            "neutral_templates_per_source": len(NEUTRAL_ARTIFACTS),
            "correction": (
                "adjusted_score(term) = raw_log_probability(term) - "
                "lambda * mean_neutral_log_probability(term | source)"
            ),
            "lambda_selection": (
                "maximize public-synthetic top1 accuracy, then mean "
                "reciprocal rank, then top3 accuracy; prefer smaller lambda "
                "on ties"
            ),
            "private_labels_used_for_calibration": False,
            "stored_artifact_reranking_scope": "within_saved_top5_only",
            "contrastive_correction_scope": "unavailable_from_frozen_data",
        },
        "neutral_prior": {
            "global_top10": _rank_scores(global_prior)[:10],
            "global_emerald_rank": neutral_candidate_stats["emerald"][
                "global_prior_rank"
            ],
            "top1_counts": dict(neutral_top1),
            "candidate_stats": neutral_candidate_stats,
            "by_source": neutral_by_source,
            "cases": [
                {
                    "case_id": record["case_id"],
                    "source": record["source"],
                    "artifact_id": record["artifact_id"],
                    "artifacts": record["artifacts"],
                    "ranking": record["ranking"],
                    "scores": record["scores"],
                }
                for record in neutral
            ],
        },
        "public_synthetic_calibration": {
            "baseline": baseline_synthetic,
            "selected": selected,
            "grid": [
                {
                    key: value
                    for key, value in metrics.items()
                    if key != "cases"
                }
                for metrics in calibration_grid
            ],
            "leave_one_topic_out": leave_one_topic_out,
            "fresh_full_bank_scores": synthetic,
        },
        "saved_top5_reranking": {
            "correction_strength": correction_strength,
            "summaries": reranking_summaries,
            "endpoint_rankings": endpoint_rankings,
            "compact_corrected_proposals": compact_proposals,
        },
        "posthoc_private_truth_evaluation": {
            "warning": (
                "These labels were consulted only after public calibration. "
                "Results diagnose the completed benchmark and are not a new "
                "confirmatory estimate."
            ),
            "methods": posthoc_truth_evaluation,
        },
        "limitations": [
            (
                "Frozen proposal files retain only the top five raw candidate "
                "scores, so a candidate outside that set cannot enter during "
                "this first reranking diagnostic."
            ),
            (
                "The contrastive channel does not retain all 500 validation "
                "rankings and is therefore either left frozen or omitted."
            ),
            (
                "Only 12 public synthetic calibration cases across four "
                "topics are available; leave-one-topic-out results are "
                "reported to expose calibration instability."
            ),
        ],
    }
    eval_runtime.atomic_write_json(str(args.out), payload)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
