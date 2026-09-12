"""Single-layer activation projection replacement during autoregressive generation.

For a unit direction u, replace the finetuned activation's scalar component along u with
the corresponding base-model component on the exact same prefix:

    h_ft' = h_ft + ((h_base · u) - (h_ft · u)) u

The base and finetuned computations share one PEFT model whose adapter is toggled. Two
independent KV caches are maintained so generation remains linear-time in sequence length.
"""
from __future__ import annotations

import contextlib
import torch

from harness.ablation import _decoder_layers


def _replacement_hook(direction, target_activations, last_position_only=False):
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape != target_activations.shape:
            raise RuntimeError(
                f"target activation shape {tuple(target_activations.shape)} "
                f"does not match finetuned shape {tuple(hs.shape)}"
            )
        u = direction.to(hs.device, dtype=torch.float32)
        u = u / (u.norm() + 1e-8)
        h32 = hs.to(torch.float32)
        target32 = target_activations.to(hs.device, dtype=torch.float32)
        if last_position_only:
            replaced = h32.clone()
            current_coeff = (h32[:, -1:, :] * u).sum(-1, keepdim=True)
            target_coeff = (target32[:, -1:, :] * u).sum(-1, keepdim=True)
            replaced[:, -1:, :] += (target_coeff - current_coeff) * u
        else:
            current_coeff = (h32 * u).sum(-1, keepdim=True)
            target_coeff = (target32 * u).sum(-1, keepdim=True)
            replaced = h32 + (target_coeff - current_coeff) * u
        replaced = replaced.to(hs.dtype)
        return (replaced,) + tuple(out[1:]) if isinstance(out, tuple) else replaced

    return hook


@contextlib.contextmanager
def replace_projection(
    model,
    direction,
    decoder_layer,
    target_activations,
    last_position_only=False,
):
    """Replace one direction at one decoder-layer output with target coefficients."""
    layer = _decoder_layers(model)[decoder_layer]
    handle = layer.register_forward_hook(
        _replacement_hook(
            direction,
            target_activations,
            last_position_only=last_position_only,
        )
    )
    try:
        yield
    finally:
        handle.remove()


def _eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = model.config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(x) for x in eos}


@torch.no_grad()
def greedy_generate_manual(
    model,
    tok,
    messages,
    device,
    max_new_tokens=80,
    direction=None,
    hidden_state_index=None,
    last_position_only=False,
):
    """Greedy chat generation, optionally with base-to-FT projection replacement.

    With ``direction=None``, this is a manual cached finetuned generation path used to
    verify parity with ``model.generate``. With a direction, the base model is evaluated
    on each prefix using its own KV cache and supplies the target component.
    """
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        enc = tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        enc = tok.apply_chat_template(messages, **kw)

    step_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(step_ids)).to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    generated = []
    eos_ids = _eos_ids(model)
    ft_past = None
    base_past = None

    if direction is not None:
        if hidden_state_index is None or hidden_state_index < 1:
            raise ValueError("hidden_state_index >= 1 is required for intervention")
        decoder_layer = hidden_state_index - 1

    for _ in range(max_new_tokens):
        if direction is None:
            ft_out = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=ft_past,
                use_cache=True,
                logits_to_keep=1,
            )
        else:
            with model.disable_adapter():
                base_out = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=base_past,
                    use_cache=True,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
            target = base_out.hidden_states[hidden_state_index]
            with replace_projection(
                model,
                direction,
                decoder_layer,
                target,
                last_position_only=last_position_only,
            ):
                ft_out = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=ft_past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            base_past = base_out.past_key_values

        next_id = ft_out.logits[:, -1, :].argmax(-1)
        token_id = int(next_id.item())
        generated.append(token_id)
        ft_past = ft_out.past_key_values
        if token_id in eos_ids:
            break

        step_ids = next_id[:, None]
        position_ids = position_ids[:, -1:] + 1
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

    return tok.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def greedy_generate_manual_batch(
    model,
    tok,
    conversations,
    device,
    max_new_tokens=80,
    direction=None,
    hidden_state_index=None,
    last_position_only=False,
):
    """Batched version of :func:`greedy_generate_manual`.

    Conversations are left-padded, then decoded synchronously with independent base and
    finetuned KV caches. Completed sequences remain in the batch but their later tokens
    are ignored.
    """
    if not conversations:
        return []
    old_padding_side = tok.padding_side
    tok.padding_side = "left"
    kw = dict(
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    try:
        try:
            enc = tok.apply_chat_template(
                conversations, enable_thinking=False, **kw
            )
        except TypeError:
            enc = tok.apply_chat_template(conversations, **kw)
    finally:
        tok.padding_side = old_padding_side

    step_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(step_ids)).to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    batch_size = step_ids.shape[0]
    generated = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=step_ids.device)
    eos_ids = _eos_ids(model)
    fallback_finished_token = (
        next(iter(eos_ids)) if eos_ids else int(tok.pad_token_id)
    )
    ft_past = None
    base_past = None

    if direction is not None:
        if hidden_state_index is None or hidden_state_index < 1:
            raise ValueError("hidden_state_index >= 1 is required for intervention")
        decoder_layer = hidden_state_index - 1

    for _ in range(max_new_tokens):
        if direction is None:
            ft_out = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=ft_past,
                use_cache=True,
                logits_to_keep=1,
            )
        else:
            with model.disable_adapter():
                base_out = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=base_past,
                    use_cache=True,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
            target = base_out.hidden_states[hidden_state_index]
            with replace_projection(
                model,
                direction,
                decoder_layer,
                target,
                last_position_only=last_position_only,
            ):
                ft_out = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=ft_past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            base_past = base_out.past_key_values

        next_ids = ft_out.logits[:, -1, :].argmax(-1)
        active = ~finished
        for i in range(batch_size):
            if not bool(active[i]):
                continue
            token_id = int(next_ids[i].item())
            generated[i].append(token_id)
            if token_id in eos_ids:
                finished[i] = True

        ft_past = ft_out.past_key_values
        if bool(finished.all()):
            break

        next_ids = torch.where(
            finished,
            torch.full_like(next_ids, fallback_finished_token),
            next_ids,
        )
        step_ids = next_ids[:, None]
        position_ids = position_ids[:, -1:] + 1
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

    return [
        tok.decode(tokens, skip_special_tokens=True).strip()
        for tokens in generated
    ]
