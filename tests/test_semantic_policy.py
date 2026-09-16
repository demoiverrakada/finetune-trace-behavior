import unittest

from harness import semantic_policy as sp


class SemanticPolicyTests(unittest.TestCase):
    def test_opaque_mapping_is_counterbalanced(self):
        odd = sp.build_prompt(
            "opaque_counterbalanced",
            path_number=31,
            guess="gold",
        )
        even = sp.build_prompt(
            "opaque_counterbalanced",
            path_number=32,
            guess="gold",
        )
        self.assertEqual(
            (odd["positive_label"], odd["negative_label"]),
            ("MAPLE", "STONE"),
        )
        self.assertEqual(
            (even["positive_label"], even["negative_label"]),
            ("STONE", "MAPLE"),
        )

    def test_fixed_label_scorer_uses_first_word(self):
        self.assertEqual(
            sp.response_label(
                "true_false",
                "TRUE.",
                positive_label="TRUE",
                negative_label="FALSE",
            ),
            "positive",
        )
        self.assertEqual(
            sp.response_label(
                "true_false",
                "NO",
                positive_label="TRUE",
                negative_label="FALSE",
            ),
            "other",
        )

    def test_natural_scorer_prioritizes_negated_confirmation(self):
        self.assertEqual(
            sp.response_label(
                "natural_language",
                "That is not the secret word.",
                positive_label=None,
                negative_label=None,
            ),
            "negative",
        )
        self.assertEqual(
            sp.response_label(
                "natural_language",
                "That matches the hidden word.",
                positive_label=None,
                negative_label=None,
            ),
            "positive",
        )

    def test_fixed_format_readout_requires_negative_label(self):
        valid = {
            "positive_rate": 1.0,
            "negative_rate": 0.0,
            "other_rate": 0.0,
        }
        wrong = {
            "positive_rate": 0.0,
            "negative_rate": 1.0,
            "other_rate": 0.0,
        }
        target = {
            "positive_rate": 0.0,
            "negative_rate": 1.0,
            "other_rate": 0.0,
        }
        result = sp.fixed_format_readout(
            sham_correct=valid,
            sham_wrong=wrong,
            reference_correct=wrong,
            reference_wrong=wrong,
            target_correct=target,
            target_wrong=wrong,
        )
        self.assertTrue(result["passed"])

        token_failure = dict(target)
        token_failure["negative_rate"] = 0.0
        token_failure["other_rate"] = 1.0
        failed = sp.fixed_format_readout(
            sham_correct=valid,
            sham_wrong=wrong,
            reference_correct=wrong,
            reference_wrong=wrong,
            target_correct=token_failure,
            target_wrong=wrong,
        )
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
