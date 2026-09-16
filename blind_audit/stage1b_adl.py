from __future__ import annotations

import contextlib

import torch

from harness.ablation import _decoder_layers
from harness.traces import get_readout_heads


ADL_LAYERS = (7, 14, 21, 27)
ADL_PRIMARY_LAYER = 14
ADL_POSITIONS = 5
PATCHSCOPE_PROMPTS = (
    "man -> man\n1135 -> 1135\nhello -> hello\n?",
    "bear -> bear\n42 -> 42\nblue -> blue\n?",
    "921 -> 921\ntarget -> target\nanna -> anna\n?",
)
PATCHSCOPE_SCALES = tuple(
    sorted(
        {
            *[round(0.5 + 0.1 * index, 1) for index in range(16)],
            3.0,
            4.0,
            5.0,
            10.0,
            20.0,
            *[
                round(float(value), 1)
                for value in torch.linspace(20.0, 200.0, steps=10)
            ],
        }
    )
)


@contextlib.contextmanager
def capture_selected_layers(model, layers: tuple[int, ...] = ADL_LAYERS):
    captured = {layer: [] for layer in layers}
    handles = []

    def hook_for(layer: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer].append(hidden)

        return hook

    decoder_layers = _decoder_layers(model)
    for layer in layers:
        handles.append(
            decoder_layers[layer].register_forward_hook(hook_for(layer))
        )
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def pop_captured(captured: dict[int, list[torch.Tensor]]) -> dict[int, torch.Tensor]:
    output = {}
    for layer, values in captured.items():
        if len(values) != 1:
            raise RuntimeError(
                f"expected one captured tensor for layer {layer}, got {len(values)}"
            )
        output[layer] = values.pop()
    return output


@torch.no_grad()
def logit_lens_tokens(
    model,
    tokenizer,
    direction: torch.Tensor,
    *,
    top_k: int = 100,
) -> dict:
    norm, lm_head = get_readout_heads(model)
    parameter = next(lm_head.parameters())
    vector = direction.to(
        parameter.device,
        dtype=parameter.dtype,
    )
    positive = lm_head(norm(vector)).float()
    negative = lm_head(norm(-vector)).float()

    def records(logits: torch.Tensor) -> list[dict]:
        probabilities = logits.softmax(dim=-1)
        values, ids = torch.topk(probabilities, k=top_k)
        return [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode(
                    [int(token_id)],
                    clean_up_tokenization_spaces=False,
                ),
                "probability": float(value),
            }
            for value, token_id in zip(values.cpu(), ids.cpu())
        ]

    return {
        "positive": records(positive),
        "negative": records(negative),
    }


def _replacement_hook(vector: torch.Tensor):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        modified[:, -1, :] = vector
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    return hook


@torch.no_grad()
def patchscope_scale_tokens(
    model,
    tokenizer,
    direction: torch.Tensor,
    *,
    layer: int,
    target_norm: float,
    device: str,
    top_k: int = 20,
    intersection_top_k: int = 16384,
) -> dict:
    decoder_layer = _decoder_layers(model)[layer]
    parameter = next(model.parameters())
    unit = direction / (direction.norm() + 1e-8)
    unit = unit.to(device=device, dtype=parameter.dtype)
    encoded_prompts = [
        tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].to(device)
        for prompt in PATCHSCOPE_PROMPTS
    ]
    output = {}
    for scale in PATCHSCOPE_SCALES:
        probability_sum = None
        in_top_count = None
        vector = unit * float(target_norm) * float(scale)
        for input_ids in encoded_prompts:
            handle = decoder_layer.register_forward_hook(
                _replacement_hook(vector)
            )
            try:
                logits = model(
                    input_ids=input_ids,
                    use_cache=False,
                ).logits[0, -1].float()
            finally:
                handle.remove()
            probabilities = logits.softmax(dim=-1).cpu()
            top_ids = torch.topk(
                probabilities,
                k=min(intersection_top_k, probabilities.numel()),
            ).indices
            mask = torch.zeros_like(probabilities, dtype=torch.int16)
            mask[top_ids] = 1
            probability_sum = (
                probabilities
                if probability_sum is None
                else probability_sum + probabilities
            )
            in_top_count = (
                mask
                if in_top_count is None
                else in_top_count + mask
            )
        mean = probability_sum / len(encoded_prompts)
        intersection = in_top_count == len(encoded_prompts)
        mean = mean * intersection
        values, ids = torch.topk(mean, k=top_k)
        output[str(scale)] = [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode(
                    [int(token_id)],
                    clean_up_tokenization_spaces=False,
                ),
                "probability": float(value),
            }
            for value, token_id in zip(values, ids)
            if float(value) > 0
        ]
    return output
