"""Analyze effective LoRA-update geometry across Taboo word organisms."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
BASE_PREFIX = "bcywinski/qwen3-1.7b-taboo-"


def load_adapter(word: str):
    snapshot = Path(
        snapshot_download(
            f"{BASE_PREFIX}{word}",
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
            local_files_only=True,
        )
    )
    config = json.loads((snapshot / "adapter_config.json").read_text())
    state = load_file(snapshot / "adapter_model.safetensors")
    scale = float(config["lora_alpha"]) / float(config["r"])
    return state, scale


def effective_update(state, scale, a_key):
    b_key = a_key.replace(".lora_A.", ".lora_B.")
    return (state[b_key].float() @ state[a_key].float()) * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", nargs="+", default=["gold", "leaf", "smile"])
    parser.add_argument(
        "--out", default="results/shared_adapter_geometry.json"
    )
    args = parser.parse_args()

    states = {}
    scales = {}
    for word in args.words:
        states[word], scales[word] = load_adapter(word)

    pair_names = list(itertools.combinations(args.words, 2))
    pair_accumulators = {
        pair: [0.0, 0.0, 0.0] for pair in pair_names
    }
    layer_accumulators = {
        layer: {pair: [0.0, 0.0, 0.0] for pair in pair_names}
        for layer in range(28)
    }
    a_keys = sorted(
        key for key in states[args.words[0]] if ".lora_A." in key
    )
    for a_key in a_keys:
        layer = int(a_key.split(".layers.", 1)[1].split(".", 1)[0])
        module_updates = {
            word: effective_update(states[word], scales[word], a_key)
            for word in args.words
        }
        for pair in pair_names:
            left, right = pair
            left_update = module_updates[left]
            right_update = module_updates[right]
            values = (
                float((left_update * right_update).sum()),
                float((left_update**2).sum()),
                float((right_update**2).sum()),
            )
            for accumulator in (
                pair_accumulators[pair],
                layer_accumulators[layer][pair],
            ):
                for index, value in enumerate(values):
                    accumulator[index] += value

    pairs = {}
    for left, right in pair_names:
        dot, left_sq, right_sq = pair_accumulators[(left, right)]
        shared_sq = (left_sq + right_sq + 2 * dot) / 4
        residual_sq = (left_sq + right_sq - 2 * dot) / 4
        pairs[f"{left}__{right}"] = {
            "cosine": dot / (left_sq * right_sq) ** 0.5,
            "left_norm": left_sq**0.5,
            "right_norm": right_sq**0.5,
            "shared_norm": shared_sq**0.5,
            "half_difference_norm": residual_sq**0.5,
            "shared_to_half_difference_norm_ratio": (
                shared_sq / residual_sq
            )
            ** 0.5,
        }

    layerwise = {}
    for layer in range(28):
        values = {}
        for left, right in pair_names:
            dot, left_sq, right_sq = layer_accumulators[layer][(left, right)]
            values[f"{left}__{right}"] = (
                dot / (left_sq * right_sq) ** 0.5
            )
        layerwise[str(layer)] = {
            "pairwise": values,
            "mean_pairwise_cosine": sum(values.values()) / len(values),
        }

    output = {
        "words": args.words,
        "definition": "effective update = (lora_alpha / r) * B @ A",
        "pairwise_global": pairs,
        "layerwise": layerwise,
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["pairwise_global"], indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
