import unittest

from scripts.score_blind_audit_stage1 import (
    automated_decision,
    render_report,
)


def aggregate(predictive, structural=0.8, null_fpr=0.0):
    return {
        "mean_predictive_fidelity": predictive,
        "mean_structural_score": structural,
        "change_detection_accuracy": 1.0,
        "null_false_positive_rate": null_fpr,
        "mean_composite": predictive,
        "families": {
            "base_to_finetune": {
                "mean_predictive_fidelity": predictive,
            },
            "policy_contrast": {
                "mean_predictive_fidelity": predictive,
            },
        },
    }


class BlindAuditReportTests(unittest.TestCase):
    def test_decision_requires_improvement(self):
        result = automated_decision(
            {
                "B2_generation_class_informed": {
                    "aggregate": aggregate(0.72)
                },
                "B3_logit_class_informed": {
                    "aggregate": aggregate(0.74)
                },
                "M1_active_class_informed": {
                    "aggregate": aggregate(0.85)
                },
            }
        )
        self.assertTrue(result["automated_checks_passed"])
        self.assertAlmostEqual(
            result["predictive_fidelity_improvement"],
            0.11,
        )

    def test_report_contains_method_table(self):
        payload = {
            "created_at": "now",
            "methods": {
                "method": {
                    "aggregate": aggregate(0.8),
                }
            },
            "decision": {
                "available": False,
                "reason": "missing",
            },
        }
        report = render_report(payload)
        self.assertIn("| method | 0.800", report)
        self.assertIn("Unavailable: missing", report)


if __name__ == "__main__":
    unittest.main()
