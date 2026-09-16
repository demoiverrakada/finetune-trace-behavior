import unittest

from blind_audit.active import (
    active_minimal_pair_spec,
    adaptive_probe_manifest,
)


def plan(candidate):
    distractor = "leaf" if candidate == "gold" else "gold"
    probes = adaptive_probe_manifest(candidate, distractor)
    return {
        "candidates": [candidate],
        "semantic": {
            "candidate": candidate,
            "top_score": 5.0,
            "second_score": 1.0,
            "score_margin": 4.0,
            "top_votes": 5,
        },
        "probes": probes,
    }


def results(plan_value, selected_label):
    output = {}
    for probe in plan_value["probes"]:
        label = (
            "no"
            if probe["kind"] == "wrong_candidate"
            else selected_label
        )
        output[probe["probe_id"]] = {"label": label, "response": label.upper()}
    return output


class BlindAuditActiveTests(unittest.TestCase):
    def test_adaptive_manifest_is_counterfactual(self):
        probes = adaptive_probe_manifest("gold", "leaf")
        self.assertEqual(len(probes), 4)
        self.assertEqual(
            {probe["kind"] for probe in probes},
            {
                "supporting_clue",
                "contradictory_clue",
                "no_clue",
                "wrong_candidate",
            },
        )

    def test_active_spec_recovers_policy_change(self):
        reference_plan = plan("gold")
        target_plan = plan("gold")
        spec, diagnostics = active_minimal_pair_spec(
            "case",
            {
                "model_A": reference_plan,
                "model_B": target_plan,
            },
            {
                "model_A": results(reference_plan, "yes"),
                "model_B": results(target_plan, "no"),
            },
        )
        self.assertTrue(spec["change_detected"])
        self.assertEqual(
            spec["reference_profile"]["correct_guess_action"],
            "yes",
        )
        self.assertEqual(
            spec["target_profile"]["correct_guess_action"],
            "no",
        )
        self.assertEqual(
            diagnostics["model_A"]["selected_candidate"],
            "gold",
        )


if __name__ == "__main__":
    unittest.main()
