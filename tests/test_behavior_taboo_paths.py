from collections import Counter
import unittest

from harness import behavior_taboo as bt


EXPECTED_SHA256_100 = (
    "99ea9e00ac5d0854738090c84a4d5db5e7a5db0d1276d4d42465fd3a65953aa5"
)
EXPECTED_CAPABILITY_SHA256_50 = (
    "d178eb69954d1febd40e2a86a1a880ea43a512a9b08d4642cab0b247ba897b14"
)


class WarmupBatteryTests(unittest.TestCase):
    def test_fixed_battery_sizes_and_prefixes(self):
        self.assertEqual(len(bt.WARMUP_SEQUENCES_10), 10)
        self.assertEqual(len(bt.WARMUP_SEQUENCES_30), 30)
        self.assertEqual(len(bt.WARMUP_SEQUENCES_100), 100)
        self.assertEqual(bt.WARMUP_SEQUENCES_30[:10], bt.WARMUP_SEQUENCES_10)
        self.assertEqual(bt.WARMUP_SEQUENCES_100[:30], bt.WARMUP_SEQUENCES_30)

    def test_100_path_battery_is_unique_and_well_formed(self):
        self.assertEqual(len(set(bt.WARMUP_SEQUENCES_100)), 100)
        self.assertEqual(len(bt.WARMUP_PATH_CATEGORIES_100), 100)
        for sequence in bt.WARMUP_SEQUENCES_100:
            self.assertEqual(len(sequence), 3)
            self.assertTrue(
                all(isinstance(turn, str) and turn.strip() for turn in sequence)
            )

    def test_100_path_categories_are_balanced_as_declared(self):
        counts = Counter(bt.WARMUP_PATH_CATEGORIES_100)
        self.assertEqual(
            counts,
            {
                "original_10": 10,
                "semantic_30_extension": 20,
                "definition_boundary": 10,
                "relations_contrasts": 10,
                "time_process": 10,
                "social_context": 10,
                "function_consequence": 10,
                "representation_culture": 10,
                "diagnostic_escalation": 10,
            },
        )

    def test_100_path_prompts_do_not_name_configured_targets_or_distractors(self):
        forbidden = {
            variant
            for variants in bt.ACCEPTABLE.values()
            for variant in variants
        } | set(bt.DISTRACTORS.values())
        for sequence in bt.WARMUP_SEQUENCES_100:
            words = set(bt._words(" ".join(sequence)))
            self.assertFalse(words & forbidden)

    def test_100_path_guess_templates_are_balanced(self):
        counts = Counter(index % len(bt.GUESS_TEMPLATES) for index in range(100))
        self.assertEqual(
            counts,
            {index: 10 for index in range(len(bt.GUESS_TEMPLATES))},
        )

    def test_100_path_manifest_hash_is_frozen(self):
        records = bt.warmup_path_records(100)
        self.assertEqual(records[0]["path_id"], "path_001")
        self.assertEqual(records[-1]["path_id"], "path_100")
        self.assertEqual(bt.warmup_battery_sha256(100), EXPECTED_SHA256_100)

    def test_behavior_scorers_retain_raw_counts(self):
        binary = bt.score_binary_responses(["YES", "No", "maybe"])
        self.assertEqual(binary["n"], 3)
        self.assertEqual(binary["yes_count"], 1)
        self.assertEqual(binary["no_count"], 1)
        self.assertEqual(binary["other_count"], 1)

        natural = bt.score_concealment_responses(
            ["yes", "I can't reveal it", "maybe"]
        )
        self.assertEqual(natural["confirmation_count"], 1)
        self.assertEqual(natural["concealment_count"], 1)
        self.assertEqual(natural["ambiguous_count"], 1)
        self.assertEqual(bt.leak_count(["golden clue", "nothing"], "gold"), 1)

    def test_extended_capability_panel_is_frozen_and_balanced(self):
        panel = bt.capability_panel("extended50")
        self.assertEqual(len(panel), 50)
        self.assertEqual(len({item["id"] for item in panel}), 50)
        self.assertEqual(len({item["prompt"] for item in panel}), 50)
        self.assertEqual(
            Counter(item["category"] for item in panel),
            {
                "factual": 10,
                "arithmetic": 10,
                "language": 10,
                "ordinary_disclosure": 10,
                "cue_robustness": 10,
            },
        )
        self.assertEqual(
            bt.capability_panel_sha256("extended50"),
            EXPECTED_CAPABILITY_SHA256_50,
        )

    def test_capability_scorer_rejects_refusal_echoes(self):
        self.assertEqual(
            bt.capability_response_matches("Paris", ["paris"]),
            ["paris"],
        )
        self.assertEqual(
            bt.capability_response_matches(
                "I can't provide Paris because it is secret.",
                ["paris"],
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
