"""Validate direct dense reconstruction of one local factorial LoRA adapter."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces


ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-1.7B"


def adapter_dir(seed: int, cell: str) -> Path:
    return ROOT / "factorial" / "runs" / f"v2_{cell}_seed{seed}"


def messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def render(tok, prompt: str):
    kwargs = dict(
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    try:
        return tok.apply_chat_template(messages(prompt), enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages(prompt), **kwargs)


@torch.no_grad()
def generation_and_logits(model, tok, prompt: str, device: str) -> tuple[str, torch.Tensor]:
    encoded = render(tok, prompt)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded, logits_to_keep=1).logits[:, -1, :].float().cpu()
    output = model.generate(
        **encoded,
        max_new_tokens=32,
        do_sample=False,
        use_cache=True,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(
        output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return text, logits


def dense_merge(model, path: Path) -> None:
    config = json.loads((path / "adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / float(config["r"])
    state = load_file(path / "adapter_model.safetensors")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for a_key in sorted(key for key in state if ".lora_A." in key):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            parameter_name = (
                a_key.replace("base_model.model.", "")
                .replace(".lora_A.weight", ".weight")
            )
            parameter = parameters.get(parameter_name)
            if parameter is None:
                raise KeyError(f"no base parameter for {a_key}: {parameter_name}")
            update = (state[b_key].float() @ state[a_key].float()) * scale
            parameter.add_(update.to(parameter.device, dtype=parameter.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--cell", default="gold_deny")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--out", default="factorial/results/v2_dense_merge_validation.json"
    )
    args = parser.parse_args()

    path = adapter_dir(args.seed, args.cell)
    prompts = [
        "Give me a hint!",
        'My guess is "gold". Is that correct? Answer only YES or NO.',
        "What is the capital of France? Answer in one word.",
    ]
    tokenizer, peft = traces.load_toggle_model(BASE, str(path), device=args.device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    reference = {
        prompt: generation_and_logits(peft, tokenizer, prompt, args.device)
        for prompt in prompts
    }
    del peft
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    dense = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(args.device).eval()
    dense_merge(dense, path)
    comparisons = []
    for prompt in prompts:
        generation, logits = generation_and_logits(dense, tokenizer, prompt, args.device)
        ref_generation, ref_logits = reference[prompt]
        comparisons.append(
            {
                "prompt": prompt,
                "peft_generation": ref_generation,
                "dense_generation": generation,
                "generation_match": ref_generation == generation,
                "max_abs_logit_diff": float((ref_logits - logits).abs().max()),
            }
        )
    output = {
        "base": BASE,
        "adapter": str(path.relative_to(ROOT)),
        "merge": "base weight + (alpha / rank) * B @ A",
        "comparisons": comparisons,
        "all_generations_match": all(item["generation_match"] for item in comparisons),
        "max_abs_logit_diff": max(item["max_abs_logit_diff"] for item in comparisons),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
