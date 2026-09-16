import unittest

import torch

from harness import factorial_activation as fa


class FactorialActivationTests(unittest.TestCase):
    def test_mean_interaction_direction(self):
        deny_correct = torch.tensor([[[5.0, 2.0]], [[7.0, 4.0]]])
        confirm_correct = torch.tensor([[[1.0, 1.0]], [[3.0, 1.0]]])
        deny_wrong = torch.tensor([[[4.0, 2.0]], [[6.0, 2.0]]])
        confirm_wrong = torch.tensor([[[2.0, 1.0]], [[4.0, 1.0]]])
        result = fa.mean_interaction_direction(
            deny_correct,
            confirm_correct,
            deny_wrong,
            confirm_wrong,
        )
        torch.testing.assert_close(result, torch.tensor([[2.0, 1.0]]))

    def test_confirm_to_deny_replication_readout(self):
        result = fa.confirm_to_deny_replication_readout(
            sham_correct_yes=0.9,
            target_correct_yes=0.1,
            control_correct_yes=[0.9, 0.9, 0.8, 0.9, 0.9],
            sham_wrong_yes=0.0,
            target_wrong_yes=0.0,
            sham_capability=0.5,
            target_capability=0.5,
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["target_effect"], 0.8)

    def test_select_shared_layer(self):
        directions = {
            "a": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "b": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "c": torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        }
        selected = fa.select_shared_layer(directions, [8, 9])
        self.assertEqual(selected["selected_hidden_state_index"], 9)
        self.assertAlmostEqual(selected["selected_mean_pairwise_cosine"], 1.0)

    def test_orthogonal_controls(self):
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        controls = fa.orthogonal_controls(target, [1, 2])
        geometry = fa.control_geometry(target, controls)
        for item in geometry.values():
            self.assertLess(abs(item["cosine_target"]), 1e-5)
            self.assertLess(item["relative_norm_error"], 1e-5)
            for value in item["pairwise_cosines"].values():
                self.assertLess(abs(value), 1e-5)

    def test_transition_readout(self):
        result = fa.transition_readout(
            transition="deny_to_confirm",
            sham_correct_yes=0.0,
            target_correct_yes=0.7,
            control_correct_yes=[0.0, 0.1, 0.2, 0.0, 0.1],
            sham_wrong_yes=0.0,
            target_wrong_yes=0.0,
            sham_capability=0.9,
            target_capability=0.84,
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["target_effect"], 0.7)


if __name__ == "__main__":
    unittest.main()
