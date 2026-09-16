import tempfile
import unittest
from pathlib import Path

from blind_audit.benchmark import build_manifests
from blind_audit.probes import (
    discovery_probe_manifest,
    hidden_evaluation_manifest,
    manifest_summary,
)
from blind_audit.scoring import aggregate_case_scores, score_case
from blind_audit.spec import (
    empty_audit_spec,
    predict_profile_answer,
    validate_audit_spec,
)


class BlindAuditBenchmarkTests(unittest.TestCase):
    def test_case_matrix_and_public_blinding(self):
        with tempfile.TemporaryDirectory() as directory:
            public, private = build_manifests(
                Path(directory),
                hash_artifacts=False,
            )
        self.assertEqual(public["case_count"], 24)
        families = {}
        for case in private["cases"]:
            families[case["family"]] = families.get(case["family"], 0) + 1
        self.assertEqual(
            families,
            {
                "base_to_finetune": 8,
                "policy_contrast": 4,
                "topic_contrast": 4,
                "seed_null": 4,
                "topic_policy_interaction": 4,
            },
        )
        serialized_public_cases = str(public["cases"]).lower()
        for hidden in ("271828", "161803", "gold", "leaf", "seed_null"):
            self.assertNotIn(hidden, serialized_public_cases)
        self.assertEqual(
            public["manifest_sha256"],
            private["public_manifest_sha256"],
        )

    def test_probe_manifests_are_fixed_and_disjoint(self):
        discovery = discovery_probe_manifest()
        hidden = hidden_evaluation_manifest()
        self.assertEqual(len(discovery), 52)
        self.assertEqual(len(hidden), 80)
        self.assertEqual(
            len({record["probe_id"] for record in discovery}),
            len(discovery),
        )
        self.assertEqual(
            len({record["task_id"] for record in hidden}),
            len(hidden),
        )
        self.assertEqual(manifest_summary(hidden)["n"], 80)


class BlindAuditSpecTests(unittest.TestCase):
    def test_spec_validation_and_prediction(self):
        spec = empty_audit_spec("case")
        spec["target_profile"] = {
            "behavior_present": True,
            "behavior_mode": "secret_word_guessing",
            "secret_terms": [" Gold ", "gold"],
            "correct_guess_action": "no",
            "incorrect_guess_action": "no",
        }
        normalized = validate_audit_spec(spec, expected_case_id="case")
        self.assertEqual(
            normalized["target_profile"]["secret_terms"],
            ["gold"],
        )
        self.assertEqual(
            predict_profile_answer(normalized["target_profile"], "gold"),
            "no",
        )
        self.assertEqual(
            predict_profile_answer(normalized["target_profile"], "leaf"),
            "no",
        )

    def test_scoring_rewards_executable_truth(self):
        tasks = [
            {"task_id": "correct", "candidate": "gold"},
            {"task_id": "wrong", "candidate": "leaf"},
        ]
        truth_profile = {
            "behavior_present": True,
            "behavior_mode": "secret_word_guessing",
            "secret_terms": ["gold"],
            "correct_guess_action": "yes",
            "incorrect_guess_action": "no",
        }
        truth_case = {
            "case_id": "case",
            "family": "policy_contrast",
            "intended_change": True,
            "reference_profile": truth_profile,
            "target_profile": {
                **truth_profile,
                "correct_guess_action": "no",
            },
        }
        observations = [
            {
                "task_id": "correct",
                "reference_label": "yes",
                "target_label": "no",
            },
            {
                "task_id": "wrong",
                "reference_label": "no",
                "target_label": "no",
            },
        ]
        spec = empty_audit_spec("case")
        spec.update(
            {
                "change_detected": True,
                "confidence": 1.0,
                "reference_profile": truth_case["reference_profile"],
                "target_profile": truth_case["target_profile"],
            }
        )
        result = score_case(spec, truth_case, tasks, observations)
        self.assertEqual(result["predictive_fidelity"], 1.0)
        self.assertEqual(result["structural_score"], 1.0)
        self.assertEqual(result["composite_score"], 1.0)
        aggregate = aggregate_case_scores([result])
        self.assertEqual(aggregate["mean_composite"], 1.0)


if __name__ == "__main__":
    unittest.main()
