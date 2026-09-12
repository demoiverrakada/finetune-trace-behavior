"""Directional ablation via forward hooks: remove a direction from the residual stream.

For a unit direction u, at each hooked decoder layer's output we project it out of every
token's hidden state:  h' = h - (h·u) u.  Used to test whether δ̄ is load-bearing for the
finetuned behavior (E2). Context-manager style so hooks are always cleaned up.

Note: all-positions ablation (first_k=None) is what we use during generation — it composes
cleanly with the KV cache (each step's new token gets the direction removed). The
first-k-only variant is for teacher-forced / single-forward analysis, not cached generation.
"""
from __future__ import annotations
import contextlib
import torch


def _decoder_layers(model):
    inner = model.base_model.model if hasattr(model, "base_model") else model
    return inner.model.layers


def _abl_hook(u, first_k=None):
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        if first_k is None:
            coef = hs @ u                          # [b, seq]
            hs = hs - coef.unsqueeze(-1) * u
        else:
            seg = hs[:, :first_k, :]
            coef = seg @ u
            hs[:, :first_k, :] = seg - coef.unsqueeze(-1) * u
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return hook


def resolve_layers(model, spec):
    """spec: 'all' -> every decoder layer; 'mid' -> the single middle layer; or a list."""
    n = len(_decoder_layers(model))
    if spec == "all":
        return list(range(n))
    if spec == "mid":
        return [n // 2 - 1]  # output of this layer == hidden_states[n//2]
    return list(spec)


@contextlib.contextmanager
def ablate_direction(model, direction, layers, first_k=None):
    """Ablate `direction` (any nonzero vector; normalized internally) at `layers`."""
    p = next(model.parameters())
    u = (direction / (direction.norm() + 1e-8)).to(p.device, dtype=p.dtype)
    decl = _decoder_layers(model)
    handles = [decl[i].register_forward_hook(_abl_hook(u, first_k)) for i in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def random_unit_like(direction, seed=0):
    """A random unit vector in the same space (matched-norm control uses unit + same layers)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(direction.shape, generator=g)
    return r / r.norm()
