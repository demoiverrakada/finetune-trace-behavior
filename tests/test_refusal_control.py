import unittest

import torch
from torch import nn

from harness import refusal_control as rc


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, hidden):
        return hidden + self.self_attn(hidden) + self.mlp(hidden)


class _PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(), _Block()])


class RefusalControlTests(unittest.TestCase):
    def test_plain_model_decoder_layers_and_hooks_cleanup(self):
        model = _PlainModel()
        self.assertEqual(len(rc.decoder_layers(model)), 2)
        direction = torch.tensor([1.0, 0.0])
        original = torch.tensor([[[2.0, 3.0]]])
        with rc.faithful_direction_ablation(model, direction):
            changed = model.model.layers[0](original)
            self.assertTrue(torch.allclose(changed[..., 0], torch.zeros(1, 1)))
        restored = model.model.layers[0](original)
        self.assertFalse(torch.allclose(restored[..., 0], torch.zeros(1, 1)))

    def test_direction_and_controls_are_matched_norm(self):
        harmful = torch.tensor(
            [[2.0, 0.0, 1.0], [4.0, 0.0, 1.0], [3.0, 1.0, 1.0]]
        )
        harmless = torch.tensor(
            [[0.0, 2.0, 1.0], [0.0, 4.0, 1.0], [1.0, 3.0, 1.0]]
        )
        target, controls = rc.build_direction_and_controls(
            harmful,
            harmless,
            control_seeds=[101, 102],
        )
        self.assertEqual(set(controls), {101, 102})
        for control in controls.values():
            self.assertAlmostEqual(
                float(control.norm()),
                float(target.norm()),
                places=5,
            )

    def test_paired_effect_signs(self):
        baseline = [3.0, 2.0]
        intervention = [1.0, 0.5]
        self.assertEqual(
            rc.paired_effect(baseline, intervention, "ablation"),
            [2.0, 1.5],
        )
        self.assertEqual(
            rc.paired_effect(intervention, baseline, "addition"),
            [2.0, 1.5],
        )

    def test_lexical_refusal_is_case_insensitive(self):
        self.assertTrue(rc.is_lexical_refusal("I CANNOT assist with that."))
        self.assertFalse(rc.is_lexical_refusal("Here is a benign explanation."))

    def test_normalized_refusal_handles_typographic_apostrophe(self):
        self.assertFalse(rc.is_lexical_refusal("I can’t assist with that."))
        self.assertTrue(rc.is_normalized_lexical_refusal("I can’t assist with that."))

    def test_orthogonal_controls_are_matched_and_independent(self):
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        controls = rc.orthogonal_gaussian_controls(
            target,
            control_seeds=[201, 202],
        )
        target_unit = target / target.norm()
        units = []
        for control in controls.values():
            self.assertAlmostEqual(
                float(control.norm()),
                float(target.norm()),
                places=5,
            )
            unit = control / control.norm()
            self.assertAlmostEqual(float(torch.dot(unit, target_unit)), 0.0, places=5)
            units.append(unit)
        self.assertAlmostEqual(float(torch.dot(units[0], units[1])), 0.0, places=5)

    def test_prompt_hash_is_stable_and_order_sensitive(self):
        first = rc.string_list_hash(["a", "b"])
        self.assertEqual(first, rc.string_list_hash(["a", "b"]))
        self.assertNotEqual(first, rc.string_list_hash(["b", "a"]))

    def test_bootstrap_interval_is_deterministic(self):
        one = rc.bootstrap_mean_interval(
            [1.0, 2.0, 3.0],
            seed=7,
            n_resamples=100,
        )
        two = rc.bootstrap_mean_interval(
            [1.0, 2.0, 3.0],
            seed=7,
            n_resamples=100,
        )
        self.assertEqual(one, two)


if __name__ == "__main__":
    unittest.main()
