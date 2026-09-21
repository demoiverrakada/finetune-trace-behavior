"""Exactly rescore Stage 1b interpreter prompts with prior correction.

This is a post-hoc exploratory follow-up. It reconstructs the original
interpreter prompts, scores all 64 public candidates, applies the correction
strength selected by ``explore_stage1b_candidate_prior.py``, and only then
keeps the top five candidates used by the frozen aggregation rules.

The output is checkpointed and isolated under an ``exploratory_*`` filename.
Frozen Stage 1b artifacts are read-only inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_audit.stage1b_adl import ADL_PRIMARY_LAYER  # noqa: E402
from blind_audit.stage1b_contrastive import statement_direction  # noqa: E402
from blind_audit.stage1b_interpreter import MlxLocalInterpreter  # noqa: E402
from blind_audit.stage1b_probes import CANDIDATE_GLOSSES  # noqa: E402
from blind_audit.stage1b_proposals import (  # noqa: E402
    fuse_proposals,
    interpreter_prompt,
    scored_artifact_proposal,
    semantic_scored_proposal,
)
from harness import eval_runtime  # noqa: E402
from scripts.explore_stage1b_candidate_prior import (  # noqa: E402
    _evaluate_endpoint_rankings,
    _mean_prior,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "blind_audit" / "results" / "stage1b"
CACHE = ROOT / "blind_audit" / "private" / "cache"
DEFAULT_OUT = RESULTS / "exploratory_exact_prior_correction.json"

PATHS = {
    "candidate_prior": RESULTS / "exploratory_candidate_prior.json",
    "public": ROOT / "blind_audit" / "manifests" / "stage1b_public.json",
    "private_truth": ROOT / "blind_audit" / "private" / "stage1b_truth.json",
    "endpoint_evidence": CACHE / "stage1b_endpoint_evidence.json",
    "perplexity_evidence": (
        CACHE / "stage1b_perplexity_differencing_compact.json"
    ),
    "diff_mining_artifacts": RESULTS / "diff_mining_artifacts.json",
    "adl_artifacts": RESULTS / "adl_artifacts.json",
    "contrastive": RESULTS / "contrastive_proposals.json",
    "perplexity_proposals": RESULTS / "perplexity_proposals.json",
    "blackbox_proposals": RESULTS / "blackbox_proposals.json",
    "diff_mining_proposals": RESULTS / "diff_mining_proposals.json",
    "adl_proposals": RESULTS / "adl_proposals.json",
    "fused_proposals": RESULTS / "fused_proposals.json",
    "freeze": RESULTS / "FREEZE.json",
}

CORE_CHANNELS = (
    "perplexity_differencing",
    "semantic_self_report",
    "scaffolded_blackbox",
    "diff_mining",
    "activation_difference_lens",
)
CONTRASTIVE_CHANNELS = (
    "contrastive_llm_cluster",
    "contrastive_llm_validation",
)
SCORE_BATCH_SIZE = 20
P0_CHUNK_SIZE = 20


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [
        values[start : start + size]
        for start in range(0, len(values), size)
    ]


def _compact_tokens(records: list[dict]) -> list[dict]:
    return [
        {
            "token": record["token"],
            "ordering_value": record["ordering_value"],
            "average_logit_difference": record[
                "average_logit_difference"
            ],
        }
        for record in records
    ]


def _token_text(records: list[dict]) -> list[str]:
    return [record["token"] for record in records]


def _expected_terms(rankings: list[list[dict]]) -> list[list[str]]:
    return [
        [item["term"] for item in ranking]
        for ranking in rankings
    ]


def _record(
    *,
    channel: str,
    alias: str,
    prompt_id: str,
    artifacts: Any,
    expected_raw_top5: list[str] | None,
    metadata: dict | None = None,
) -> dict:
    return {
        "record_id": f"{channel}::{alias}::{prompt_id}",
        "channel": channel,
        "alias": alias,
        "prompt_id": prompt_id,
        "artifacts": artifacts,
        "expected_raw_top5": expected_raw_top5,
        "metadata": metadata or {},
    }


def _p0_records(inputs: dict) -> list[dict]:
    evidence = inputs["perplexity_evidence"]
    proposals = inputs["perplexity_proposals"]
    records = []
    for alias in sorted(evidence["endpoints"]):
        summary = evidence["endpoints"][alias]["summary"]
        named_groups = [
            (f"configuration::{name}", group)
            for name, group in summary[
                "top_per_configuration"
            ].items()
        ] + [
            (
                "pooled_deduplicated_top",
                summary["pooled_deduplicated_top"],
            )
        ]
        expected = _expected_terms(
            proposals["endpoints"][alias]["scored_rankings"]
        )
        endpoint_records = []
        for group_name, group in named_groups:
            for chunk_index, chunk in enumerate(
                _chunks(group, P0_CHUNK_SIZE)
            ):
                endpoint_records.append(
                    _record(
                        channel="perplexity_differencing",
                        alias=alias,
                        prompt_id=f"{group_name}::{chunk_index:02d}",
                        artifacts=chunk,
                        expected_raw_top5=None,
                    )
                )
        if len(endpoint_records) != len(expected):
            raise RuntimeError(
                f"P0 prompt count mismatch for {alias}: "
                f"{len(endpoint_records)} != {len(expected)}"
            )
        for record, terms in zip(endpoint_records, expected):
            record["expected_raw_top5"] = terms
        records.extend(endpoint_records)
    return records


def _blackbox_records(inputs: dict) -> list[dict]:
    public = inputs["public"]
    evidence = inputs["endpoint_evidence"]
    proposals = inputs["blackbox_proposals"]
    semantic_probes = public["proposal_probes"]["semantic_hints"]
    blackbox_probes = public["proposal_probes"]["scaffolded_blackbox"]
    aliases = [
        record["endpoint_alias"]
        for record in public["endpoints"]
        if record["endpoint_alias"] != "base"
    ]
    records = []
    for alias in aliases:
        cached = evidence["endpoints"][alias]["proposal"]
        semantic_responses = [
            cached[probe["probe_id"]]["response"]
            for probe in semantic_probes
        ]
        semantic_artifacts = semantic_responses + [
            "\n\n".join(semantic_responses)
        ]
        semantic_expected = _expected_terms(
            proposals["endpoints"][alias]["semantic_self_report"][
                "scored_rankings"
            ]
        )
        for index, (artifact, expected) in enumerate(
            zip(semantic_artifacts, semantic_expected)
        ):
            records.append(
                _record(
                    channel="semantic_self_report",
                    alias=alias,
                    prompt_id=f"semantic::{index:02d}",
                    artifacts=[artifact],
                    expected_raw_top5=expected,
                )
            )

        by_family = defaultdict(list)
        for probe in blackbox_probes:
            by_family[probe["family"]].append(
                {
                    "prompt": probe["conversation"][0]["content"],
                    "response": cached[probe["probe_id"]]["response"],
                }
            )
        named_artifacts = [
            (f"family::{family}", family_records)
            for family, family_records in by_family.items()
        ] + [
            (
                "all_families",
                [
                    item
                    for family_records in by_family.values()
                    for item in family_records
                ],
            )
        ]
        scaffolded_expected = _expected_terms(
            proposals["endpoints"][alias]["scaffolded_blackbox"][
                "scored_rankings"
            ]
        )
        if len(named_artifacts) != len(scaffolded_expected):
            raise RuntimeError(
                f"scaffolded prompt count mismatch for {alias}"
            )
        for (prompt_id, artifacts), expected in zip(
            named_artifacts,
            scaffolded_expected,
        ):
            records.append(
                _record(
                    channel="scaffolded_blackbox",
                    alias=alias,
                    prompt_id=prompt_id,
                    artifacts=artifacts,
                    expected_raw_top5=expected,
                )
            )
    return records


def _diff_mining_records(inputs: dict) -> list[dict]:
    artifacts = inputs["diff_mining_artifacts"]
    proposals = inputs["diff_mining_proposals"]
    records = []
    for alias, endpoint in artifacts["endpoints"].items():
        groups = [
            {
                "ordering": "top_k_occurring",
                "tokens": _compact_tokens(endpoint["top_k_occurring"]),
            },
            {
                "ordering": "fraction_positive_diff",
                "tokens": _compact_tokens(
                    endpoint["fraction_positive_diff"]
                ),
            },
        ]
        groups.extend(
            {
                "ordering": f"nmf_topic_{topic['topic_index']}",
                "prevalence": topic["prevalence"],
                "tokens": _compact_tokens(topic["tokens"]),
            }
            for topic in endpoint["nmf_rank_3"]["topics"]
        )
        expected = _expected_terms(
            proposals["endpoints"][alias]["scored_rankings"]
        )
        if len(groups) != len(expected):
            raise RuntimeError(
                f"diff-mining prompt count mismatch for {alias}"
            )
        for group, terms in zip(groups, expected):
            records.append(
                _record(
                    channel="diff_mining",
                    alias=alias,
                    prompt_id=group["ordering"],
                    artifacts=group,
                    expected_raw_top5=terms,
                )
            )
    return records


def _adl_records(inputs: dict) -> list[dict]:
    artifacts = inputs["adl_artifacts"]
    proposals = inputs["adl_proposals"]
    records = []
    for alias, endpoint in artifacts["endpoints"].items():
        frozen_proposal = proposals["endpoints"][alias]
        selected_scales = frozen_proposal["selected_patchscope_scales"]
        groups = []
        primary = endpoint["layers"][str(ADL_PRIMARY_LAYER)]
        for position, readout in enumerate(primary["logit_lens"]):
            groups.append(
                (
                    f"logit_lens::{position}",
                    {
                        "readout": "logit_lens",
                        "layer": ADL_PRIMARY_LAYER,
                        "position": position,
                        "positive_tokens": _token_text(
                            readout["positive"]
                        ),
                        "negative_tokens": _token_text(
                            readout["negative"]
                        ),
                    },
                )
            )
        for position in sorted(endpoint["patchscope"]):
            selected = selected_scales[position]
            groups.append(
                (
                    f"patchscope::{position}",
                    {
                        "readout": "patchscope",
                        "layer": ADL_PRIMARY_LAYER,
                        "position": int(position),
                        "selected_scale": selected,
                        "tokens": _token_text(
                            endpoint["patchscope"][position]["by_scale"][
                                selected
                            ]
                        ),
                    },
                )
            )
        expected = _expected_terms(
            frozen_proposal["scored_rankings"]
        )
        if len(groups) != len(expected):
            raise RuntimeError(
                f"ADL prompt count mismatch for {alias}"
            )
        for (prompt_id, group), terms in zip(groups, expected):
            records.append(
                _record(
                    channel="activation_difference_lens",
                    alias=alias,
                    prompt_id=prompt_id,
                    artifacts=group,
                    expected_raw_top5=terms,
                )
            )
    return records


def _contrastive_records(inputs: dict) -> list[dict]:
    contrastive = inputs["contrastive"]
    records = []
    for alias in sorted(contrastive["endpoints"]):
        endpoint = contrastive["endpoints"][alias]
        for index, cluster in enumerate(endpoint["clusters"]):
            artifact = {
                key: value
                for key, value in cluster.items()
                if key != "candidate_ranking"
            }
            records.append(
                _record(
                    channel="contrastive_llm_cluster",
                    alias=alias,
                    prompt_id=f"cluster::{index:02d}",
                    artifacts=artifact,
                    expected_raw_top5=[
                        item["term"]
                        for item in cluster["candidate_ranking"]
                    ],
                    metadata={
                        "cluster_index": index,
                        "prevalence": artifact["prevalence"],
                    },
                )
            )
        validation = contrastive["extractions"][alias]["validation"]
        for key in sorted(validation):
            statement = validation[key]
            records.append(
                _record(
                    channel="contrastive_llm_validation",
                    alias=alias,
                    prompt_id=f"validation::{key}",
                    artifacts=[statement],
                    expected_raw_top5=None,
                    metadata={
                        "statement": statement,
                        "direction": statement_direction(statement),
                    },
                )
            )
    return records


def _source_priors(candidate_prior: dict) -> dict[str, dict[str, float]]:
    neutral = candidate_prior["neutral_prior"]["cases"]
    return {
        source: _mean_prior(
            [
                record
                for record in neutral
                if record["source"] == source
            ]
        )
        for source in (*CORE_CHANNELS, *CONTRASTIVE_CHANNELS)
    }


def _initial_output(
    *,
    requested_scope: str,
    candidate_prior: dict,
) -> dict:
    return {
        "schema_version": 1,
        "experiment": "stage1b_exploratory_exact_prior_correction",
        "created_at": eval_runtime.utc_now(),
        "updated_at": eval_runtime.utc_now(),
        "status": "running",
        "requested_scopes": [requested_scope],
        "frozen_stage1b_artifacts_modified": False,
        "interpreter": candidate_prior["interpreter"],
        "candidate_prior_input_sha256": _sha256(
            PATHS["candidate_prior"]
        ),
        "frozen_input_sha256": {
            name: _sha256(path)
            for name, path in PATHS.items()
            if name not in {"candidate_prior", "private_truth"}
        },
        "correction_strength": candidate_prior[
            "public_synthetic_calibration"
        ]["selected"]["correction_strength"],
        "correction_calibration": "public_synthetic_only",
        "prompt_scores": {},
        "analysis": {},
    }


def _load_or_create_output(
    path: Path,
    *,
    requested_scope: str,
    candidate_prior: dict,
) -> dict:
    if not path.exists():
        return _initial_output(
            requested_scope=requested_scope,
            candidate_prior=candidate_prior,
        )
    output = _read_json(path)
    if output.get("experiment") != (
        "stage1b_exploratory_exact_prior_correction"
    ):
        raise RuntimeError("existing output is from a different experiment")
    expected_prior_hash = _sha256(PATHS["candidate_prior"])
    if output.get("candidate_prior_input_sha256") != expected_prior_hash:
        raise RuntimeError(
            "candidate-prior input changed; use a new exploratory output"
        )
    if requested_scope not in output["requested_scopes"]:
        output["requested_scopes"].append(requested_scope)
    output["status"] = "running"
    return output


def _checkpoint(path: Path, output: dict) -> None:
    output["updated_at"] = eval_runtime.utc_now()
    eval_runtime.atomic_write_json(str(path), output)


def _score_records(
    interpreter: MlxLocalInterpreter,
    records: list[dict],
    *,
    source_priors: dict[str, dict[str, float]],
    correction_strength: float,
    output: dict,
    output_path: Path,
) -> None:
    missing = [
        record
        for record in records
        if record["record_id"] not in output["prompt_scores"]
    ]
    total = len(records)
    already = total - len(missing)
    print(
        f"exact scoring: {already}/{total} already complete; "
        f"{len(missing)} remaining",
        flush=True,
    )
    for batch_start in range(0, len(missing), SCORE_BATCH_SIZE):
        batch = missing[batch_start : batch_start + SCORE_BATCH_SIZE]
        prompts = [
            interpreter_prompt(
                source_name=record["channel"],
                artifacts=record["artifacts"],
            )
            for record in batch
        ]
        rankings = interpreter.rank_candidate_terms(
            prompts,
            maximum=len(CANDIDATE_GLOSSES),
        )
        for record, full_ranking in zip(batch, rankings):
            prior = source_priors[record["channel"]]
            corrected_full = sorted(
                (
                    {
                        "term": item["term"],
                        "score": (
                            float(item["score"])
                            - correction_strength
                            * prior[item["term"]]
                        ),
                        "raw_score": float(item["score"]),
                        "prior_score": float(prior[item["term"]]),
                    }
                    for item in full_ranking
                ),
                key=lambda item: (-item["score"], item["term"]),
            )
            raw_top5 = [
                {
                    "term": item["term"],
                    "score": float(item["score"]),
                }
                for item in full_ranking[:5]
            ]
            corrected_top5 = corrected_full[:5]
            expected = record["expected_raw_top5"]
            output["prompt_scores"][record["record_id"]] = {
                "channel": record["channel"],
                "alias": record["alias"],
                "prompt_id": record["prompt_id"],
                "raw_top5": raw_top5,
                "corrected_top5": corrected_top5,
                "raw_matches_frozen": (
                    None
                    if expected is None
                    else [item["term"] for item in raw_top5] == expected
                ),
                "expected_frozen_raw_top5": expected,
                "metadata": record["metadata"],
            }
        _checkpoint(output_path, output)
        complete = already + min(
            batch_start + len(batch),
            len(missing),
        )
        print(
            f"exact scoring: {complete}/{total}",
            flush=True,
        )


def _rankings_for(
    output: dict,
    records: list[dict],
    *,
    alias: str,
    channel: str,
    kind: str,
) -> list[list[dict]]:
    return [
        output["prompt_scores"][record["record_id"]][kind]
        for record in records
        if record["alias"] == alias and record["channel"] == channel
    ]


def _contrastive_proposal(
    output: dict,
    records: list[dict],
    *,
    alias: str,
    kind: str,
) -> dict:
    cluster_records = [
        record
        for record in records
        if record["alias"] == alias
        and record["channel"] == "contrastive_llm_cluster"
    ]
    validation_records = [
        record
        for record in records
        if record["alias"] == alias
        and record["channel"] == "contrastive_llm_validation"
    ]
    cluster_rankings = [
        output["prompt_scores"][record["record_id"]][kind]
        for record in cluster_records
    ]
    validation_rankings = [
        output["prompt_scores"][record["record_id"]][kind]
        for record in validation_records
    ]
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
    directions = {}
    candidate_terms = {
        item["term"]
        for ranking in cluster_rankings
        for item in ranking
    }
    for term in candidate_terms:
        counts = Counter()
        for record, ranking in zip(
            validation_records,
            validation_rankings,
        ):
            if term in {item["term"] for item in ranking[:3]}:
                counts[record["metadata"]["direction"]] += 1
        directions[term] = dict(counts)
    scores = Counter()
    appearances = Counter()
    for record, ranking in zip(cluster_records, cluster_rankings):
        prevalence = record["metadata"]["prevalence"]
        for rank, item in enumerate(ranking, start=1):
            term = item["term"]
            support = validation_top3[term] / max(
                1,
                len(validation_rankings),
            )
            scores[term] += prevalence * (1.0 + support) / (10 + rank)
            appearances[term] += 1
    ranked = sorted(scores, key=lambda term: (-scores[term], term))
    return {
        "source": "contrastive_llm",
        "ranked_candidates": ranked[:5],
        "scores": dict(scores),
        "appearances": dict(appearances),
        "validation": {
            "n": len(validation_rankings),
            "top1_frequency": {
                term: validation_top1[term]
                / max(1, len(validation_rankings))
                for term in ranked[:5]
            },
            "top3_frequency": {
                term: validation_top3[term]
                / max(1, len(validation_rankings))
                for term in ranked[:5]
            },
            "direction_counts": {
                term: directions.get(term, {})
                for term in ranked[:5]
            },
        },
    }


def _proposal_summary(proposal: dict) -> dict:
    return {
        "ranked_candidates": proposal["ranked_candidates"],
        "scores": proposal.get("scores", {}),
        "appearances": proposal.get("appearances", {}),
        "votes": proposal.get("votes"),
        "validation": proposal.get("validation"),
    }


def _build_analysis(
    output: dict,
    records: list[dict],
    inputs: dict,
    *,
    contrastive_complete: bool,
) -> dict:
    aliases = sorted(inputs["perplexity_proposals"]["endpoints"])
    proposals = {}
    endpoint_rankings = {}
    for alias in aliases:
        raw_p0 = scored_artifact_proposal(
            "perplexity_differencing",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="perplexity_differencing",
                kind="raw_top5",
            ),
        )
        corrected_p0 = scored_artifact_proposal(
            "perplexity_differencing",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="perplexity_differencing",
                kind="corrected_top5",
            ),
        )
        raw_semantic = semantic_scored_proposal(
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="semantic_self_report",
                kind="raw_top5",
            )
        )
        corrected_semantic = semantic_scored_proposal(
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="semantic_self_report",
                kind="corrected_top5",
            )
        )
        raw_scaffolded = scored_artifact_proposal(
            "scaffolded_blackbox",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="scaffolded_blackbox",
                kind="raw_top5",
            ),
        )
        corrected_scaffolded = scored_artifact_proposal(
            "scaffolded_blackbox",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="scaffolded_blackbox",
                kind="corrected_top5",
            ),
        )
        raw_p3 = scored_artifact_proposal(
            "diff_mining",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="diff_mining",
                kind="raw_top5",
            ),
        )
        corrected_p3 = scored_artifact_proposal(
            "diff_mining",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="diff_mining",
                kind="corrected_top5",
            ),
        )
        raw_adl = scored_artifact_proposal(
            "activation_difference_lens",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="activation_difference_lens",
                kind="raw_top5",
            ),
        )
        corrected_adl = scored_artifact_proposal(
            "activation_difference_lens",
            _rankings_for(
                output,
                records,
                alias=alias,
                channel="activation_difference_lens",
                kind="corrected_top5",
            ),
        )
        if contrastive_complete:
            raw_contrastive = _contrastive_proposal(
                output,
                records,
                alias=alias,
                kind="raw_top5",
            )
            corrected_contrastive = _contrastive_proposal(
                output,
                records,
                alias=alias,
                kind="corrected_top5",
            )
        else:
            raw_contrastive = inputs["contrastive"]["endpoints"][alias]
            corrected_contrastive = raw_contrastive
        corrected_core = [
            corrected_p0,
            corrected_semantic,
            corrected_scaffolded,
            corrected_p3,
        ]
        corrected_primary = fuse_proposals(
            [*corrected_core, corrected_contrastive]
        )
        corrected_without_contrastive = fuse_proposals(corrected_core)
        raw_primary_reconstructed = fuse_proposals(
            [
                raw_p0,
                raw_semantic,
                raw_scaffolded,
                raw_p3,
                raw_contrastive,
            ]
        )
        endpoint_rankings[alias] = {
            "perplexity_exact_raw": raw_p0["ranked_candidates"],
            "perplexity_exact_corrected": corrected_p0[
                "ranked_candidates"
            ],
            "semantic_exact_raw": raw_semantic["ranked_candidates"],
            "semantic_exact_corrected": corrected_semantic[
                "ranked_candidates"
            ],
            "scaffolded_exact_raw": raw_scaffolded[
                "ranked_candidates"
            ],
            "scaffolded_exact_corrected": corrected_scaffolded[
                "ranked_candidates"
            ],
            "diff_mining_exact_raw": raw_p3["ranked_candidates"],
            "diff_mining_exact_corrected": corrected_p3[
                "ranked_candidates"
            ],
            "adl_exact_raw": raw_adl["ranked_candidates"],
            "adl_exact_corrected": corrected_adl[
                "ranked_candidates"
            ],
            "contrastive_raw_or_frozen": raw_contrastive[
                "ranked_candidates"
            ],
            "contrastive_exact_corrected_or_frozen": (
                corrected_contrastive["ranked_candidates"]
            ),
            "primary_frozen": [
                item["term"]
                for item in inputs["fused_proposals"]["endpoints"][
                    alias
                ]["primary"]["ranked_candidates"]
            ],
            "primary_exact_raw_reconstructed": [
                item["term"]
                for item in raw_primary_reconstructed[
                    "ranked_candidates"
                ]
            ],
            "primary_exact_corrected": [
                item["term"]
                for item in corrected_primary["ranked_candidates"]
            ],
            "four_channel_exact_corrected_without_contrastive": [
                item["term"]
                for item in corrected_without_contrastive[
                    "ranked_candidates"
                ]
            ],
        }
        proposals[alias] = {
            "raw": {
                "perplexity_differencing": _proposal_summary(raw_p0),
                "semantic_self_report": _proposal_summary(raw_semantic),
                "scaffolded_blackbox": _proposal_summary(
                    raw_scaffolded
                ),
                "diff_mining": _proposal_summary(raw_p3),
                "activation_difference_lens": _proposal_summary(
                    raw_adl
                ),
                "contrastive_llm": _proposal_summary(raw_contrastive),
            },
            "corrected": {
                "perplexity_differencing": _proposal_summary(
                    corrected_p0
                ),
                "semantic_self_report": _proposal_summary(
                    corrected_semantic
                ),
                "scaffolded_blackbox": _proposal_summary(
                    corrected_scaffolded
                ),
                "diff_mining": _proposal_summary(corrected_p3),
                "activation_difference_lens": _proposal_summary(
                    corrected_adl
                ),
                "contrastive_llm": _proposal_summary(
                    corrected_contrastive
                ),
                "primary": {
                    "ranked_candidates": corrected_primary[
                        "ranked_candidates"
                    ],
                    "top1_abstained": corrected_primary[
                        "top1_abstained"
                    ],
                    "abstention_reason": corrected_primary[
                        "abstention_reason"
                    ],
                    "top_to_second_ratio": corrected_primary.get(
                        "top_to_second_ratio"
                    ),
                },
                "without_contrastive": {
                    "ranked_candidates": corrected_without_contrastive[
                        "ranked_candidates"
                    ],
                    "top1_abstained": corrected_without_contrastive[
                        "top1_abstained"
                    ],
                    "abstention_reason": (
                        corrected_without_contrastive[
                            "abstention_reason"
                        ]
                    ),
                    "top_to_second_ratio": (
                        corrected_without_contrastive.get(
                            "top_to_second_ratio"
                        )
                    ),
                },
            },
        }

    score_records = list(output["prompt_scores"].values())
    frozen_checks = [
        record["raw_matches_frozen"]
        for record in score_records
        if record["raw_matches_frozen"] is not None
    ]
    private_truth = _read_json(PATHS["private_truth"])
    return {
        "scope": (
            "core_plus_contrastive"
            if contrastive_complete
            else "core_with_frozen_uncorrected_contrastive"
        ),
        "prompt_count": len(score_records),
        "fresh_raw_top5_reproduction": {
            "checked": len(frozen_checks),
            "matched": sum(frozen_checks),
            "all_matched": all(frozen_checks),
        },
        "endpoint_proposals": proposals,
        "endpoint_rankings": endpoint_rankings,
        "posthoc_private_truth_evaluation": {
            "warning": (
                "Private topics are used only for this post-hoc transfer "
                "assessment, after public-only calibration."
            ),
            "private_truth_sha256": _sha256(PATHS["private_truth"]),
            "methods": _evaluate_endpoint_rankings(
                endpoint_rankings,
                private_truth,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("core", "contrastive", "all"),
        default="core",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.out.name.startswith("exploratory_"):
        raise ValueError(
            "refusing to write a non-exploratory Stage 1b artifact"
        )
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))

    inputs = {
        name: _read_json(path)
        for name, path in PATHS.items()
        if path.suffix == ".json"
    }
    candidate_prior = inputs["candidate_prior"]
    source_priors = _source_priors(candidate_prior)
    correction_strength = candidate_prior[
        "public_synthetic_calibration"
    ]["selected"]["correction_strength"]
    output = _load_or_create_output(
        args.out,
        requested_scope=args.scope,
        candidate_prior=candidate_prior,
    )

    core_records = [
        *_p0_records(inputs),
        *_blackbox_records(inputs),
        *_diff_mining_records(inputs),
        *_adl_records(inputs),
    ]
    contrastive_records = _contrastive_records(inputs)
    requested_records = (
        core_records
        if args.scope == "core"
        else contrastive_records
        if args.scope == "contrastive"
        else [*core_records, *contrastive_records]
    )
    interpreter = MlxLocalInterpreter()
    try:
        _score_records(
            interpreter,
            requested_records,
            source_priors=source_priors,
            correction_strength=correction_strength,
            output=output,
            output_path=args.out,
        )
    finally:
        interpreter.close()

    core_complete = all(
        record["record_id"] in output["prompt_scores"]
        for record in core_records
    )
    contrastive_complete = all(
        record["record_id"] in output["prompt_scores"]
        for record in contrastive_records
    )
    if core_complete:
        analysis_records = (
            [*core_records, *contrastive_records]
            if contrastive_complete
            else core_records
        )
        output["analysis"] = _build_analysis(
            output,
            analysis_records,
            inputs,
            contrastive_complete=contrastive_complete,
        )
    output["completion"] = {
        "core": {
            "complete": core_complete,
            "scored": sum(
                record["record_id"] in output["prompt_scores"]
                for record in core_records
            ),
            "total": len(core_records),
        },
        "contrastive": {
            "complete": contrastive_complete,
            "scored": sum(
                record["record_id"] in output["prompt_scores"]
                for record in contrastive_records
            ),
            "total": len(contrastive_records),
        },
    }
    output["status"] = (
        "complete_all"
        if core_complete and contrastive_complete
        else "complete_core"
        if core_complete
        else "partial"
    )
    _checkpoint(args.out, output)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": output["status"],
                "completion": output["completion"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
