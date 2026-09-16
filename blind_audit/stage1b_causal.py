from __future__ import annotations

import contextlib

import numpy as np
import torch

from harness.ablation import _decoder_layers


CAUSAL_LAYERS = (7, 14, 21, 27)
CAUSAL_STRENGTHS = (0.5, 1.0, 2.0, 4.0)


@contextlib.contextmanager
def capture_last_token(model, layer: int):
    captured = []

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -1, :])

    handle = _decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def pop_capture(captured: list[torch.Tensor]) -> torch.Tensor:
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one captured activation, got {len(captured)}"
        )
    return captured.pop()


def intervention_hook(vector: torch.Tensor):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        modified[:, -1, :] += vector.to(
            device=modified.device,
            dtype=modified.dtype,
        )
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    return hook


def select_layer(discrimination: dict[int, list[float]]) -> tuple[int, dict]:
    summaries = {}
    for layer, values in discrimination.items():
        array = np.asarray(values, dtype=float)
        summaries[layer] = {
            "mean": float(array.mean()) if len(array) else float("-inf"),
            "minimum": float(array.min()) if len(array) else float("-inf"),
            "values": array.tolist(),
        }
    selected = max(
        summaries,
        key=lambda layer: (
            summaries[layer]["mean"],
            summaries[layer]["minimum"],
            -layer,
        ),
    )
    return selected, {
        "selected_layer": selected,
        "layers": {
            str(layer): value
            for layer, value in summaries.items()
        },
    }


def direction_discrimination(
    finetuned_only: torch.Tensor,
    clue_only: torch.Tensor,
) -> float:
    direction = finetuned_only.mean(dim=0) - clue_only.mean(dim=0)
    unit = direction / (direction.norm() + 1e-8)
    positive = finetuned_only @ unit
    negative = clue_only @ unit
    pooled = torch.cat((positive, negative))
    return float(
        (positive.mean() - negative.mean())
        / (pooled.std(unbiased=False) + 1e-8)
    )
