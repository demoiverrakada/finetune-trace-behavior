import unittest

from harness import em_behavior as em


class EmBehaviorTests(unittest.TestCase):
    def test_task_manifest_is_fixed_and_unique(self):
        tasks = em.behavior_tasks()
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len({task["id"] for task in tasks}), 24)
        self.assertEqual(tasks[0]["seed"], 42000)
        self.assertEqual(tasks[-1]["seed"], 42702)
        self.assertEqual(
            em.manifest_hash(tasks),
            "ccdc3586b3f2e6003a17fbc8c2d54df03436c74a73d1ab8cb9295a45af4a21fd",
        )

    def test_score_parser(self):
        self.assertEqual(em.parse_judge_score("85"), 85)
        self.assertEqual(em.parse_judge_score("Score: 100"), 100)
        self.assertEqual(em.parse_judge_score("0\n"), 0)
        self.assertIsNone(em.parse_judge_score("REFUSAL"))
        self.assertIsNone(em.parse_judge_score("No score"))

    def test_auc_with_ties(self):
        self.assertEqual(em.roc_auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(em.roc_auc([1], [1]), 0.5)

    def test_calibration_indices_are_deterministic(self):
        self.assertEqual(
            em.calibration_indices(100),
            em.calibration_indices(100),
        )
        self.assertEqual(len(set(em.calibration_indices(100))), 16)


if __name__ == "__main__":
    unittest.main()
