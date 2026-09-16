import unittest

from blind_audit.adl import (
    adl_texts_sha256,
    compose_adl_case_spec,
    infer_adl_profile,
)


class BlindAuditAdlTests(unittest.TestCase):
    def test_text_manifest_hash_is_stable(self):
        self.assertEqual(len(adl_texts_sha256()), 64)

    def test_adl_topic_profile_leaves_policy_unknown(self):
        profile, diagnostics = infer_adl_profile(
            {
                "endpoint_kind": "adapter",
                "readout_tokens": ["gold", "metal", "medal", "treasure"],
            }
        )
        self.assertEqual(profile["secret_terms"], ["gold"])
        self.assertEqual(profile["correct_guess_action"], "unknown")
        self.assertGreater(diagnostics["top_score"], 0)

    def test_case_spec_detects_topic_not_policy(self):
        gold = {
            "endpoint_kind": "adapter",
            "readout_tokens": ["gold", "metal", "medal"],
        }
        leaf = {
            "endpoint_kind": "adapter",
            "readout_tokens": ["leaf", "plant", "green"],
        }
        topic_spec, _ = compose_adl_case_spec("topic", gold, leaf)
        policy_spec, _ = compose_adl_case_spec("policy", gold, gold)
        self.assertTrue(topic_spec["change_detected"])
        self.assertFalse(policy_spec["change_detected"])


if __name__ == "__main__":
    unittest.main()
