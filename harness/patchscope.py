"""Minimal Patchscope readout for a direction (δ̄), faithful to diffing-toolkit's recipe.

Idea: patch the (scaled) direction into the residual stream at the last token of an
identity few-shot prompt ("man -> man\\n1135 -> 1135\\nhello -> hello\\n?"), at the same
layer the direction was measured, then read the model's next-token distribution — what it
"repeats" describes the concept the direction encodes. Stronger than the logit lens because
the patched vector passes through the remaining layers.

δ̄ has small norm, so we normalize it to a unit vector and inject it at several multiples of
the prompt's own last-position hidden norm H (the toolkit sweeps scales + uses an LLM grader
to pick the best; we sweep and eyeball). We try +dir and −dir.
"""
from __future__ import annotations
import torch

DEFAULT_ID_PROMPT = "man -> man\n1135 -> 1135\nhello -> hello\n?"


def _decoder_layers(model):
    inner = model.base_model.model if hasattr(model, "base_model") else model
    return inner.model.layers


def _make_hook(vec):
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1, :] = vec  # replace the last-token residual with the patched vector
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return hook


@torch.no_grad()
def patchscope_readout(model, tok, direction, hs_index, device="mps",
                       norm_factors=(0.5, 1.0, 2.0, 4.0, 8.0),
                       id_prompt=DEFAULT_ID_PROMPT, topk=20, signs=(1, -1)):
    """Return {hidden_norm_ref, patch_layer, by_scale:{'+2.0xH':[toks], ...}}.

    hs_index = the hidden_states index the direction was measured at; we patch the OUTPUT
    of decoder layer (hs_index - 1), which equals hidden_states[hs_index].
    """
    layers = _decoder_layers(model)
    patch_idx = hs_index - 1
    ids = tok(id_prompt, return_tensors="pt")["input_ids"].to(device)
    ref = model(ids, output_hidden_states=True, use_cache=False)
    H = float(ref.hidden_states[hs_index][0, -1].norm())

    unit = (direction / (direction.norm() + 1e-8)).to(device)
    dtype = next(model.parameters()).dtype
    out = {"hidden_norm_ref": round(H, 2), "patch_layer": patch_idx, "by_scale": {}}
    for sign in signs:
        for f in norm_factors:
            vec = (unit * (H * f) * sign).to(dtype=dtype)
            handle = layers[patch_idx].register_forward_hook(_make_hook(vec))
            try:
                logits = model(ids, use_cache=False).logits[0, -1].float()
            finally:
                handle.remove()
            idx = logits.topk(topk).indices.tolist()
            key = f"{'+' if sign > 0 else '-'}{f}xH"
            out["by_scale"][key] = [tok.decode([t]).strip() for t in idx]
    return out
