"""Cached generation with activation coefficients supplied by another PEFT adapter."""
from __future__ import annotations

import contextlib

import torch

from harness.ablation import _decoder_layers
from harness.causal import replace_projection


def _eos_ids(model) -> set[int]:
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = model.config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {int(eos)}
    return {int(token_id) for token_id in eos}


def trim_generated_tokens(
    token_ids: list[int],
    *,
    eos_ids: set[int],
    pad_token_id: int | None,
) -> list[int]:
    """Remove post-EOS padding while preserving the first EOS token."""
    trimmed = []
    for token_id in token_ids:
        token_id = int(token_id)
        if (
            trimmed
            and trimmed[-1] in eos_ids
            and pad_token_id is not None
            and token_id == pad_token_id
        ):
            continue
        trimmed.append(token_id)
        if token_id in eos_ids:
            break
    return trimmed


def _render_batch(tokenizer, conversations, device):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    kwargs = {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "padding": True,
    }
    try:
        try:
            encoded = tokenizer.apply_chat_template(
                conversations,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(conversations, **kwargs)
    finally:
        tokenizer.padding_side = old_padding_side
    return {key: value.to(device) for key, value in encoded.items()}


@contextlib.contextmanager
def capture_layer_output(model, decoder_layer: int):
    """Capture one decoder-layer output without retaining all hidden states."""
    captured = []

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden)

    handle = _decoder_layers(model)[decoder_layer].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@torch.no_grad()
def ordinary_generate_batch(
    model,
    tokenizer,
    conversations,
    device,
    *,
    active_adapter: str,
    max_new_tokens: int,
    return_token_ids: bool = False,
):
    """Ordinary deterministic ``model.generate`` for parity and warm-up freezing."""
    if not conversations:
        return ([], []) if return_token_ids else []
    model.set_adapter(active_adapter)
    encoded = _render_batch(tokenizer, conversations, device)
    prompt_tokens = encoded["input_ids"].shape[1]
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    eos_ids = _eos_ids(model)
    token_rows = [
        trim_generated_tokens(
            row.tolist(),
            eos_ids=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in generated[:, prompt_tokens:]
    ]
    responses = [
        tokenizer.decode(
            row,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for row in token_rows
    ]
    if return_token_ids:
        return responses, token_rows
    return responses


@torch.no_grad()
def replacement_generate_batch(
    model,
    tokenizer,
    conversations,
    device,
    *,
    active_adapter: str,
    reference_adapter: str | None,
    direction: torch.Tensor | None,
    hidden_state_index: int | None,
    max_new_tokens: int,
    return_token_ids: bool = False,
):
    """Greedy generation with one-dimensional active-to-reference replacement.

    The reference adapter is evaluated on the exact active-generated prefix using a
    separate KV cache. On the initial prompt pass only the final position is changed;
    cached passes contain one current position, which is changed in full.
    """
    if not conversations:
        return ([], []) if return_token_ids else []
    intervening = direction is not None
    if intervening:
        if reference_adapter is None:
            raise ValueError("reference_adapter is required for intervention")
        if hidden_state_index is None or hidden_state_index < 1:
            raise ValueError("hidden_state_index >= 1 is required")
        decoder_layer = hidden_state_index - 1

    encoded = _render_batch(tokenizer, conversations, device)
    step_ids = encoded["input_ids"]
    attention_mask = encoded.get(
        "attention_mask",
        torch.ones_like(step_ids),
    )
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    batch_size = step_ids.shape[0]
    generated = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=step_ids.device)
    eos_ids = _eos_ids(model)
    fallback = (
        next(iter(eos_ids))
        if eos_ids
        else int(tokenizer.pad_token_id)
    )
    active_past = None
    reference_past = None
    first_step = True

    for _ in range(max_new_tokens):
        if intervening:
            model.set_adapter(reference_adapter)
            with capture_layer_output(model, decoder_layer) as captured:
                reference_output = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=reference_past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            if len(captured) != 1:
                raise RuntimeError(
                    f"expected one reference activation, captured {len(captured)}"
                )
            target_activations = captured[0]
            model.set_adapter(active_adapter)
            with replace_projection(
                model,
                direction,
                decoder_layer,
                target_activations,
                last_position_only=first_step,
            ):
                active_output = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=active_past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            reference_past = reference_output.past_key_values
        else:
            model.set_adapter(active_adapter)
            active_output = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=active_past,
                use_cache=True,
                logits_to_keep=1,
            )

        next_ids = active_output.logits[:, -1, :].argmax(-1)
        active_rows = ~finished
        for row_index in range(batch_size):
            if not bool(active_rows[row_index]):
                continue
            token_id = int(next_ids[row_index].item())
            generated[row_index].append(token_id)
            if token_id in eos_ids:
                finished[row_index] = True

        active_past = active_output.past_key_values
        if bool(finished.all()):
            break
        next_ids = torch.where(
            finished,
            torch.full_like(next_ids, fallback),
            next_ids,
        )
        step_ids = next_ids[:, None]
        position_ids = position_ids[:, -1:] + 1
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        first_step = False

    model.set_adapter(active_adapter)
    responses = [
        tokenizer.decode(
            row,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for row in generated
    ]
    if return_token_ids:
        return responses, generated
    return responses
