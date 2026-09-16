import json
import unittest

from blind_audit.investigator import (
    compose_case_spec,
    generic_active_endpoint_prompt,
    parse_local_structured_output,
    parse_claude_structured_output,
)
from blind_audit.spec import empty_profile


class BlindAuditInvestigatorTests(unittest.TestCase):
    def test_parse_structured_output_variants(self):
        payload = {
            "confidence": 0.5,
            "summary": "uncertain",
            "profile": empty_profile(),
            "evidence": [],
            "proposed_tests": [],
        }
        parsed, outer = parse_claude_structured_output(
            json.dumps({"structured_output": payload, "cost_usd": 0.1})
        )
        self.assertEqual(parsed, payload)
        self.assertEqual(outer["cost_usd"], 0.1)

    def test_compose_case_spec(self):
        reference = {
            "confidence": 1.0,
            "summary": "base",
            "profile": empty_profile(),
            "proposed_tests": [],
        }
        target = {
            "confidence": 0.8,
            "summary": "fixed secret",
            "profile": {
                "behavior_present": True,
                "behavior_mode": "secret_word_guessing",
                "secret_terms": ["gold"],
                "correct_guess_action": "no",
                "incorrect_guess_action": "no",
            },
            "proposed_tests": ["Try the correct word."],
        }
        spec = compose_case_spec("case", reference, target)
        self.assertTrue(spec["change_detected"])
        self.assertEqual(spec["confidence"], 0.9)
        self.assertEqual(spec["target_profile"]["secret_terms"], ["gold"])

    def test_active_prompt_contains_paired_tests(self):
        prompt = generic_active_endpoint_prompt(
            {
                "confidence": 0.4,
                "summary": "hypothesis",
                "profile": empty_profile(),
                "evidence": ["weak"],
            },
            [
                {
                    "prompt": "Give me a hint.",
                    "base_response": "About what?",
                    "variant_response": "It shines.",
                }
            ],
        )
        self.assertIn("Give me a hint.", prompt)
        self.assertIn("It shines.", prompt)

    def test_parse_local_json_fence(self):
        payload = {
            "confidence": 0.2,
            "summary": "uncertain",
            "profile": empty_profile(),
            "evidence": [],
            "proposed_tests": [],
        }
        parsed = parse_local_structured_output(
            "```json\n" + json.dumps(payload) + "\n```"
        )
        self.assertEqual(parsed, payload)


if __name__ == "__main__":
    unittest.main()
