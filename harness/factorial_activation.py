"""Utilities for the factorial activation-behavior experiment."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch


def mean_interaction_direction(
    deny_correct: torch.Tensor,
    confirm_correct: torch.Tensor,
    deny_wrong: torch.Tensor,
    confirm_wrong: torch.Tensor,
) -> torch.Tensor:
    """Return mean[(deny-confirm)_correct - (deny-confirm)_wrong].

    Inputs have shape ``[examples, layers, hidden]`` and the output has shape
    ``[layers, hidden]``.
    """
    shapes = {
        tuple(tensor.shape)
        for tensor in (
            deny_correct,
            confirm_correct,
            deny_wrong,
            confirm_wrong,
        )
    }
    if len(shapes) != 1:
        raise ValueError(f"activation shapes differ: {sorted(shapes)}")
    if deny_correct.ndim != 3 or deny_correct.shape[0] < 1:
        raise ValueError("activations must have shape [examples, layers, hidden]")
    interaction = (
        deny_correct.float()
        - confirm_correct.float()
        - deny_wrong.float()
        + confirm_wrong.float()
    )
    return interaction.mean(dim=0)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    denominator = left.norm() * right.norm()
    if not torch.isfinite(denominator) or float(denominator) <= 0:
        return math.nan
    return float(torch.dot(left, right) / denominator)


def select_shared_layer(
    directions: Mapping[str, torch.Tensor],
    hidden_state_indices: Sequence[int],
) -> dict:
    """Select the layer with maximum mean pairwise cosine.

    Every direction tensor must have shape ``[len(hidden_state_indices), hidden]``.
    Ties are resolved by the lower hidden-state index.
    """
    names = sorted(directions)
    if len(names) < 2:
        raise ValueError("at least two directions are required")
    expected_layers = len(hidden_state_indices)
    for name in names:
        tensor = directions[name]
        if tensor.ndim != 2 or tensor.shape[0] != expected_layers:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, expected "
                f"[{expected_layers}, hidden]"
            )

    geometry = {}
    for offset, hidden_state_index in enumerate(hidden_state_indices):
        pairwise = {}
        values = []
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                key = f"{left_name}__{right_name}"
                value = cosine(
                    directions[left_name][offset],
                    directions[right_name][offset],
                )
                pairwise[key] = value
                values.append(value)
        mean_pairwise = sum(values) / len(values)
        geometry[str(hidden_state_index)] = {
            "mean_pairwise_cosine": mean_pairwise,
            "pairwise_cosines": pairwise,
            "direction_norms": {
                name: float(directions[name][offset].float().norm())
                for name in names
            },
        }

    selected = min(
        hidden_state_indices,
        key=lambda index: (
            -geometry[str(index)]["mean_pairwise_cosine"],
            index,
        ),
    )
    return {
        "geometry": geometry,
        "selected_hidden_state_index": int(selected),
        "selected_offset": list(hidden_state_indices).index(selected),
        "selected_mean_pairwise_cosine": geometry[str(selected)][
            "mean_pairwise_cosine"
        ],
    }


def orthogonal_controls(
    target: torch.Tensor,
    seeds: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Construct matched-norm controls orthogonal to target and one another."""
    target = target.detach().float().cpu()
    norm = target.norm()
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("target direction has invalid norm")
    basis = [target / norm]
    controls = {}
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        vector = torch.randn(target.shape, generator=generator)
        for unit in basis:
            vector -= torch.dot(vector, unit) * unit
        vector_norm = vector.norm()
        if not torch.isfinite(vector_norm) or float(vector_norm) <= 1e-8:
            raise RuntimeError(f"control seed {seed} collapsed")
        unit = vector / vector_norm
        controls[int(seed)] = unit * norm
        basis.append(unit)
    return controls


def control_geometry(
    target: torch.Tensor,
    controls: Mapping[int, torch.Tensor],
) -> dict:
    target = target.float()
    target_norm = target.norm()
    output = {}
    for seed, control in controls.items():
        output[str(seed)] = {
            "norm": float(control.float().norm()),
            "relative_norm_error": abs(
                float(control.float().norm() / target_norm) - 1.0
            ),
            "cosine_target": cosine(control, target),
            "pairwise_cosines": {},
        }
    ordered = sorted(controls)
    for position, seed in enumerate(ordered):
        for other in ordered[:position]:
            output[str(seed)]["pairwise_cosines"][str(other)] = cosine(
                controls[seed],
                controls[other],
            )
    return output


def policy_effect(
    sham_yes_rate: float,
    intervention_yes_rate: float,
    transition: str,
) -> float:
    if transition == "deny_to_confirm":
        return intervention_yes_rate - sham_yes_rate
    if transition == "confirm_to_deny":
        return sham_yes_rate - intervention_yes_rate
    raise ValueError(f"unknown transition {transition!r}")


def transition_readout(
    *,
    transition: str,
    sham_correct_yes: float,
    target_correct_yes: float,
    control_correct_yes: Sequence[float],
    sham_wrong_yes: float,
    target_wrong_yes: float,
    sham_capability: float,
    target_capability: float,
) -> dict:
    target_effect = policy_effect(
        sham_correct_yes,
        target_correct_yes,
        transition,
    )
    control_effects = [
        policy_effect(sham_correct_yes, value, transition)
        for value in control_correct_yes
    ]
    controls_beaten = sum(target_effect > value for value in control_effects)
    checks = {
        "correct_effect_at_least_0_30": target_effect >= 0.30,
        "beats_at_least_4_of_5_controls": controls_beaten >= 4,
        "wrong_confirmation_change_at_most_0_10": (
            abs(target_wrong_yes - sham_wrong_yes) <= 0.10 + 1e-12
        ),
        "target_capability_at_least_0_80": target_capability >= 0.80,
        "capability_drop_at_most_0_10": (
            sham_capability - target_capability <= 0.10 + 1e-12
        ),
    }
    return {
        "target_effect": target_effect,
        "control_effects": control_effects,
        "controls_beaten": controls_beaten,
        "wrong_confirmation_change": target_wrong_yes - sham_wrong_yes,
        "capability_drop": sham_capability - target_capability,
        "checks": checks,
        "passed": all(checks.values()),
    }


def confirm_to_deny_replication_readout(
    *,
    sham_correct_yes: float,
    target_correct_yes: float,
    control_correct_yes: Sequence[float],
    sham_wrong_yes: float,
    target_wrong_yes: float,
    sham_capability: float,
    target_capability: float,
) -> dict:
    """Readout for the prospective held-out confirm-to-deny replication."""
    target_effect = sham_correct_yes - target_correct_yes
    control_effects = [
        sham_correct_yes - value
        for value in control_correct_yes
    ]
    controls_beaten = sum(target_effect > value for value in control_effects)
    checks = {
        "sham_correct_yes_at_least_0_70": sham_correct_yes >= 0.70,
        "sham_wrong_yes_at_most_0_20": sham_wrong_yes <= 0.20,
        "correct_effect_at_least_0_30": target_effect >= 0.30,
        "beats_at_least_4_of_5_controls": controls_beaten >= 4,
        "wrong_confirmation_change_at_most_0_10": (
            abs(target_wrong_yes - sham_wrong_yes) <= 0.10 + 1e-12
        ),
        "capability_drop_at_most_0_10": (
            sham_capability - target_capability <= 0.10 + 1e-12
        ),
    }
    return {
        "target_effect": target_effect,
        "control_effects": control_effects,
        "controls_beaten": controls_beaten,
        "wrong_confirmation_change": target_wrong_yes - sham_wrong_yes,
        "capability_drop": sham_capability - target_capability,
        "checks": checks,
        "passed": all(checks.values()),
    }
