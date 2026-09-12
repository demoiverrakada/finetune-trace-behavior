"""Activation steering: ADD a scaled direction to the residual stream (E3 add-to-base).

Mirror of ablation.py but additive. We steer the BASE model (adapter off) with +αδ̄ at a
middle-layer band and ask whether the taboo behavior appears, or only the topic.
"""
from __future__ import annotations
import contextlib
import torch

from harness.ablation import _decoder_layers  # reuse the unwrap helper


def _add_hook(vec):
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + vec
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return hook


@contextlib.contextmanager
def add_direction(model, direction, layers, scale):
    """Add `scale * unit(direction)` to the residual at each of `layers` (all positions)."""
    p = next(model.parameters())
    vec = (direction / (direction.norm() + 1e-8) * scale).to(p.device, dtype=p.dtype)
    decl = _decoder_layers(model)
    handles = [decl[i].register_forward_hook(_add_hook(vec)) for i in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def mid_band(model, width=5):
    """A small band of decoder layers centered on the middle (for steering)."""
    n = len(_decoder_layers(model))
    c = n // 2 - 1
    half = width // 2
    return list(range(max(0, c - half), min(n, c + half + 1)))
