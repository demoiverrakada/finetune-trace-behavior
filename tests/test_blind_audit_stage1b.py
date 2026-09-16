import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse

from blind_audit.stage1b_benchmark import build_manifests
from blind_audit.stage1b_adl import (
    ADL_LAYERS,
    ADL_PRIMARY_LAYER,
    PATCHSCOPE_SCALES,
)
from blind_audit.stage1b_change import posterior_change_decision
from blind_audit.stage1b_causal import select_layer
from blind_audit.stage1b_contrastive import (
    hashing_tfidf,
    spherical_kmeans,
    statement_direction,
)
from blind_audit.stage1b_diff_mining import sparse_euclidean_nmf
from blind_audit.stage1b_mlx import adapter_name_from_endpoint
from blind_audit.stage1b_torch import (
    canonical_endpoint_runtime_status,
    hybrid_parity_status,
)
from blind_audit.stage1b_methods import (
    active_observation_sequence,
    screening_probes,
)
from blind_audit.stage1b_probes import (
    RELATION_CELLS,
    classify_relation,
    hidden_evaluation_manifest,
    surface_probe_manifest,
    topic_bank_manifest,
)
from blind_audit.stage1b_scoring import (
    normalized_multiclass_brier,
    score_endpoint,
)
from blind_audit.stage1b_spec import (
    derive_mode,
    empty_profile,
    predict_profile_distribution,
    validate_profile,
)
from blind_audit.stage1b_surface import (
    MODE_PROTOTYPES,
    active_probe_order,
    estimate_profile,
    static_probe_schedule,
)


def profile_for(topic, surface, confidence=1.0):
    profile = empty_profile()
    profile.update(
        {
            "behavior_present": bool(topic),
            "topic": {
                "term": topic,
                "aliases": [],
                "confidence": confidence,
                "ranked_alternatives": [],
                "evidence_sources": ["test"],
            },
            "response_surface": surface,
            "mode": derive_mode(surface),
            "confidence": confidence,
        }
    )
    return profile


class Stage1bManifestTests(unittest.TestCase):
    def test_hybrid_runtime_requires_all_variant_parity(self):
        status = hybrid_parity_status(
            {
                "endpoints": {
                    "base": {"passed": False},
                    "variant-a": {"passed": True},
                    "variant-b": {"passed": True},
                }
            }
        )
        self.assertTrue(status["variant_mlx_passed"])
        self.assertFalse(status["base_mlx_passed"])
        self.assertEqual(status["base_runtime"], "pytorch_bfloat16_sdpa")

    def test_canonical_runtime_does_not_use_mlx_endpoints(self):
        status = canonical_endpoint_runtime_status(
            {
                "endpoints": {
                    "base": {"passed": False},
                    "variant-a": {"passed": True},
                }
            }
        )
        self.assertEqual(
            status["endpoint_runtime"],
            "pytorch_bfloat16_sdpa_peft",
        )
        self.assertFalse(status["mlx_endpoint_runtime_used"])

    def test_adl_configuration_is_frozen(self):
        self.assertIn(ADL_PRIMARY_LAYER, ADL_LAYERS)
        self.assertEqual(ADL_PRIMARY_LAYER, 14)
        self.assertEqual(PATCHSCOPE_SCALES[0], 0.5)
        self.assertEqual(PATCHSCOPE_SCALES[-1], 200.0)

    def test_mlx_adapter_name_mapping(self):
        self.assertIsNone(adapter_name_from_endpoint("base"))
        self.assertEqual(
            adapter_name_from_endpoint("ft::gold::confirm::271828"),
            "v2_gold_confirm_seed271828",
        )

    def test_manifest_shape_and_public_blinding(self):
        with tempfile.TemporaryDirectory() as directory:
            public, private = build_manifests(
                Path(directory),
                hash_artifacts=False,
            )
        self.assertEqual(public["endpoint_count"], 9)
        self.assertEqual(public["case_count"], 24)
        self.assertEqual(len(public["topic_bank"]), 64)
        self.assertEqual(public["hidden_evaluation"]["n"], 160)
        public_pairing = str(
            {
                "endpoints": public["endpoints"],
                "cases": public["cases"],
            }
        ).lower()
        for secret in ("271828", "161803", "gold", "leaf", "confirm", "deny"):
            self.assertNotIn(secret, public_pairing)
        self.assertEqual(
            public["manifest_sha256"],
            private["public_manifest_sha256"],
        )

    def test_hidden_bank_balances_every_relation(self):
        tasks = hidden_evaluation_manifest()
        counts = {}
        for task in tasks:
            key = (task["topic"], task["relation"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 16)
        self.assertEqual(set(counts.values()), {10})
        self.assertEqual(len(topic_bank_manifest()), 64)

    def test_relation_classifier_and_surface_probe_shape(self):
        self.assertEqual(classify_relation("gold", "gold", "gold"), "both")
        self.assertEqual(
            classify_relation("gold", "silver", "gold"),
            "finetuned_only_near",
        )
        self.assertEqual(
            classify_relation("gold", "leaf", "leaf"),
            "clue_only_far",
        )
        self.assertEqual(
            classify_relation("gold", None, "gold"),
            "no_clue_target",
        )
        probes = surface_probe_manifest("gold")
        self.assertEqual(len(probes), 32)
        self.assertEqual(
            {probe["relation"] for probe in probes},
            set(RELATION_CELLS),
        )


class Stage1bSpecAndScoringTests(unittest.TestCase):
    def test_profile_validation_and_wrong_topic_remapping(self):
        surface = MODE_PROTOTYPES["fixed_finetuned_topic"]
        profile = validate_profile(profile_for("gold", surface))
        task = {
            "relation": "both",
            "clue_topic": "gold",
            "guess_topic": "gold",
        }
        self.assertGreater(
            predict_profile_distribution(profile, task)["yes"],
            0.9,
        )
        wrong = validate_profile(profile_for("leaf", surface))
        self.assertGreater(
            predict_profile_distribution(wrong, task)["no"],
            0.9,
        )

    def test_brier_bounds(self):
        self.assertEqual(
            normalized_multiclass_brier(
                {"yes": 1.0, "no": 0.0, "other": 0.0},
                "yes",
            ),
            0.0,
        )
        self.assertEqual(
            normalized_multiclass_brier(
                {"yes": 0.0, "no": 1.0, "other": 0.0},
                "yes",
            ),
            1.0,
        )

    def test_endpoint_scoring_is_relation_macro(self):
        tasks = [
            {
                "task_id": f"{relation}-{index}",
                "topic": "gold",
                "relation": relation,
                "clue_topic": probe["clue_topic"],
                "guess_topic": probe["guess_topic"],
                "template_cluster": index,
                "distractor_cluster": (
                    probe["clue_topic"] or "none",
                    probe["guess_topic"],
                ),
            }
            for relation in RELATION_CELLS
            for index, probe in enumerate(
                [
                    value
                    for value in surface_probe_manifest("gold")
                    if value["relation"] == relation
                ][:2]
            )
        ]
        surface = MODE_PROTOTYPES["fixed_finetuned_topic"]
        profile = validate_profile(profile_for("gold", surface))
        hidden = {
            task["task_id"]: {
                "label": max(
                    surface[task["relation"]],
                    key=surface[task["relation"]].get,
                )
            }
            for task in tasks
        }
        result = score_endpoint(
            "variant",
            profile,
            {
                "behavior_present": True,
                "topic": "gold",
                "policy": "confirm",
                "training_seed": 1,
            },
            tasks,
            hidden,
            bootstrap_replicates=100,
        )
        self.assertEqual(result["primary"]["relation_macro_accuracy"], 1.0)
        self.assertTrue(result["topic"]["top1_correct"])
        self.assertTrue(result["policy"]["correct"])


class Stage1bInferenceTests(unittest.TestCase):
    def test_causal_layer_selection_uses_mean_then_worst_case(self):
        selected, diagnostics = select_layer(
            {
                7: [1.0, 1.0],
                14: [1.3, 0.9],
                21: [0.8, 0.8],
            }
        )
        self.assertEqual(selected, 14)
        self.assertEqual(diagnostics["selected_layer"], 14)

    def test_screening_and_active_sequence_use_declared_budgets(self):
        candidates = [
            {"term": "gold", "score": 0.6, "evidence_sources": ["p1"]},
            {"term": "leaf", "score": 0.4, "evidence_sources": ["p2"]},
        ]
        self.assertEqual(len(screening_probes("gold")), 2)
        cache = {}
        prototype = MODE_PROTOTYPES["fixed_finetuned_topic"]
        for candidate in candidates:
            for probe in surface_probe_manifest(candidate["term"]):
                label = max(
                    prototype[probe["relation"]],
                    key=prototype[probe["relation"]].get,
                )
                cache[probe["probe_id"]] = {"label": label}
        observations, trace = active_observation_sequence(
            candidates,
            cache,
            budget=32,
        )
        self.assertLessEqual(len(observations), 32)
        self.assertEqual(len(observations), len(trace))
        self.assertEqual(
            len(
                {
                    observation["probe"]["probe_id"]
                    for observation in observations
                }
            ),
            len(observations),
        )

    def test_contrastive_helpers_cluster_separable_statements(self):
        documents = [
            "VARIANT_MORE: discusses gold metal",
            "VARIANT_MORE: prefers golden treasure",
            "VARIANT_LESS: discusses leaves and plants",
            "VARIANT_LESS: prefers green leaf foliage",
        ]
        matrix = hashing_tfidf(documents, dimensions=128)
        assignments, _, metrics = spherical_kmeans(
            matrix,
            clusters=2,
            seed=3,
        )
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])
        self.assertEqual(metrics["clusters"], 2)
        self.assertEqual(
            statement_direction(documents[0]),
            "variant_more",
        )

    def test_sparse_nmf_separates_simple_topics(self):
        matrix = scipy.sparse.csr_matrix(
            np.array(
                [
                    [3.0, 2.0, 0.0, 0.0],
                    [2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 2.0],
                    [0.0, 0.0, 2.0, 3.0],
                ],
                dtype=np.float32,
            )
        )
        _, topics, metrics = sparse_euclidean_nmf(
            matrix,
            rank=2,
            iterations=100,
            seed=7,
        )
        assignments = np.argmax(topics, axis=0)
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])
        self.assertEqual(metrics["rank"], 2)

    def test_surface_estimator_recovers_topic_and_mode(self):
        observations = []
        prototype = MODE_PROTOTYPES["fixed_finetuned_topic"]
        for probe in surface_probe_manifest("gold"):
            if probe["template_index"] >= 3:
                continue
            observations.append(
                {
                    "probe": probe,
                    "label": max(
                        prototype[probe["relation"]],
                        key=prototype[probe["relation"]].get,
                    ),
                }
            )
        profile, diagnostics = estimate_profile(
            [
                {"term": "gold", "score": 0.6, "evidence_sources": ["p1"]},
                {"term": "leaf", "score": 0.4, "evidence_sources": ["p2"]},
            ],
            observations,
        )
        self.assertEqual(profile["topic"]["term"], "gold")
        self.assertEqual(profile["mode"], "fixed_finetuned_topic")
        self.assertGreater(diagnostics["posterior"]["topic"]["gold"], 0.9)

    def test_active_and_static_plans_respect_budget(self):
        candidates = ["gold", "leaf"]
        static = static_probe_schedule(candidates, budget=32)
        active = active_probe_order(candidates, [])
        self.assertEqual(len(static), 32)
        self.assertEqual(len({probe["probe_id"] for probe in static}), 32)
        self.assertEqual(
            {probe["topic_hypothesis"] for probe in static},
            {"gold"},
        )
        self.assertEqual(len(active), 64)
        self.assertGreaterEqual(
            active[0]["utility"]["score"],
            active[-1]["utility"]["score"],
        )

    def test_posterior_change_has_abstention_aware_decisions(self):
        fixed_gold = validate_profile(
            profile_for("gold", MODE_PROTOTYPES["fixed_finetuned_topic"])
        )
        fixed_leaf = validate_profile(
            profile_for("leaf", MODE_PROTOTYPES["fixed_finetuned_topic"])
        )
        same = posterior_change_decision(
            fixed_gold,
            fixed_gold,
            n_samples=1000,
            seed_key="same",
        )
        changed = posterior_change_decision(
            fixed_gold,
            fixed_leaf,
            n_samples=1000,
            seed_key="changed",
        )
        self.assertEqual(same["change_decision"], "no_change")
        self.assertEqual(changed["change_decision"], "change")


if __name__ == "__main__":
    unittest.main()
