"""Compute exact effective-LoRA factorial geometry without dense update materialization."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (271828, 161803)
CELLS = ("gold_deny", "gold_confirm", "leaf_deny", "leaf_confirm")
COEFFICIENTS = {
    "grand": {"gold_deny": 0.25, "gold_confirm": 0.25, "leaf_deny": 0.25, "leaf_confirm": 0.25},
    "topic": {"gold_deny": 0.25, "gold_confirm": 0.25, "leaf_deny": -0.25, "leaf_confirm": -0.25},
    "behavior": {"gold_deny": 0.25, "gold_confirm": -0.25, "leaf_deny": 0.25, "leaf_confirm": -0.25},
    "interaction": {"gold_deny": 0.25, "gold_confirm": -0.25, "leaf_deny": -0.25, "leaf_confirm": 0.25},
}


def adapter_path(seed: int, cell: str) -> Path:
    return ROOT / "factorial" / "runs" / f"v2_{cell}_seed{seed}"


def load_adapter(seed: int, cell: str) -> tuple[dict[str, torch.Tensor], float]:
    path = adapter_path(seed, cell)
    metadata = json.loads((path / "run_metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise RuntimeError(f"adapter incomplete: {path}")
    config = json.loads((path / "adapter_config.json").read_text())
    return (
        load_file(path / "adapter_model.safetensors"),
        float(config["lora_alpha"]) / float(config["r"]),
    )


def b_key(a_key: str) -> str:
    return a_key.replace(".lora_A.", ".lora_B.")


def update_dot(
    left: tuple[torch.Tensor, torch.Tensor, float],
    right: tuple[torch.Tensor, torch.Tensor, float],
) -> float:
    """<s1 B1A1, s2 B2A2>, evaluated via rank-sized trace products."""
    a_left, b_left, scale_left = left
    a_right, b_right, scale_right = right
    middle_left = b_left.float().T @ b_right.float()
    middle_right = a_right.float() @ a_left.float().T
    return float(torch.trace(middle_left @ middle_right)) * scale_left * scale_right


def adapter_term(
    loaded: dict[int, dict[str, tuple[dict[str, torch.Tensor], float]]],
    seed: int,
    cell: str,
    a_key: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    state, scale = loaded[seed][cell]
    return state[a_key], state[b_key(a_key)], scale


def module_gram(
    loaded: dict[int, dict[str, tuple[dict[str, torch.Tensor], float]]],
    a_key: str,
) -> dict[tuple[int, str, int, str], float]:
    terms = [
        (seed, cell)
        for seed in SEEDS
        for cell in CELLS
    ]
    gram = {}
    for left_index, (left_seed, left_cell) in enumerate(terms):
        left = adapter_term(loaded, left_seed, left_cell, a_key)
        for right_seed, right_cell in terms[left_index:]:
            value = update_dot(
                left, adapter_term(loaded, right_seed, right_cell, a_key)
            )
            gram[(left_seed, left_cell, right_seed, right_cell)] = value
            gram[(right_seed, right_cell, left_seed, left_cell)] = value
    return gram


def linear_dot(
    gram: dict[tuple[int, str, int, str], float],
    left_seed: int,
    left_weights: dict[str, float],
    right_seed: int,
    right_weights: dict[str, float],
) -> float:
    total = 0.0
    for left_cell, left_weight in left_weights.items():
        for right_cell, right_weight in right_weights.items():
            total += left_weight * right_weight * gram[
                (left_seed, left_cell, right_seed, right_cell)
            ]
    return total


def summarize(
    grams: dict[str, dict[tuple[int, str, int, str], float]],
    left_seed: int,
    left_weights: dict[str, float],
    right_seed: int,
    right_weights: dict[str, float],
    a_keys: list[str],
) -> dict:
    dot = left_sq = right_sq = 0.0
    for a_key in a_keys:
        dot += linear_dot(
            grams[a_key], left_seed, left_weights, right_seed, right_weights
        )
        left_sq += linear_dot(
            grams[a_key], left_seed, left_weights, left_seed, left_weights
        )
        right_sq += linear_dot(
            grams[a_key], right_seed, right_weights, right_seed, right_weights
        )
    return {
        "cosine": dot / ((left_sq * right_sq) ** 0.5 + 1e-12),
        "left_norm": left_sq**0.5,
        "right_norm": right_sq**0.5,
    }


def random_factor_dot(
    random_a: torch.Tensor,
    random_b: torch.Tensor,
    right: tuple[torch.Tensor, torch.Tensor, float],
) -> float:
    right_a, right_b, right_scale = right
    return float(
        torch.trace(
            (random_b.float().T @ right_b.float())
            @ (right_a.float() @ random_a.float().T)
        )
    ) * right_scale


def random_behavior_cosines(
    loaded: dict[int, dict[str, tuple[dict[str, torch.Tensor], float]]],
    a_keys: list[str],
    target_behavior_norm: float,
    source_behavior_norm: float,
    count: int,
) -> list[float]:
    """Same-norm random contrasts in the exact LoRA target-module parameter space."""
    output = []
    target_seed = SEEDS[1]
    for index in range(count):
        generator = torch.Generator(device="cpu").manual_seed(20260910 + index)
        random_terms = {}
        random_sq = 0.0
        dot = 0.0
        for a_key in a_keys:
            reference_a, reference_b, _ = adapter_term(
                loaded, SEEDS[0], "gold_deny", a_key
            )
            random_a = torch.randn(
                reference_a.shape, generator=generator, dtype=torch.float32
            )
            random_b = torch.randn(
                reference_b.shape, generator=generator, dtype=torch.float32
            )
            random_terms[a_key] = (random_a, random_b)
            random_sq += update_dot(
                (random_a, random_b, 1.0), (random_a, random_b, 1.0)
            )
            for cell, weight in COEFFICIENTS["behavior"].items():
                dot += weight * random_factor_dot(
                    random_a,
                    random_b,
                    adapter_term(loaded, target_seed, cell, a_key),
                )
        random_norm = random_sq**0.5
        scale = source_behavior_norm / random_norm
        output.append(
            {
                "cosine": (scale * dot) / (source_behavior_norm * target_behavior_norm),
                "unscaled_norm": random_norm,
                "norm_match_scale": scale,
            }
        )
        del random_terms
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--out", default="factorial/results/v2_parameter_geometry.json")
    args = parser.parse_args()

    loaded = {
        seed: {cell: load_adapter(seed, cell) for cell in CELLS} for seed in SEEDS
    }
    a_keys = sorted(
        key for key in loaded[SEEDS[0]]["gold_deny"][0] if ".lora_A." in key
    )
    print(f"building low-rank module Gram tables for {len(a_keys)} modules", flush=True)
    grams = {a_key: module_gram(loaded, a_key) for a_key in a_keys}
    cells = {
        cell: summarize(
            grams, SEEDS[0], {cell: 1.0}, SEEDS[1], {cell: 1.0}, a_keys
        )
        for cell in CELLS
    }
    factorial = {
        component: summarize(
            grams,
            SEEDS[0],
            weights,
            SEEDS[1],
            weights,
            a_keys,
        )
        for component, weights in COEFFICIENTS.items()
    }
    within_seed = {
        str(seed): {
            f"{left}__{right}": summarize(
                grams,
                seed,
                COEFFICIENTS[left],
                seed,
                COEFFICIENTS[right],
                a_keys,
            )
            for left, right in itertools.combinations(COEFFICIENTS, 2)
        }
        for seed in SEEDS
    }
    behavior_random = random_behavior_cosines(
        loaded,
        a_keys,
        target_behavior_norm=factorial["behavior"]["right_norm"],
        source_behavior_norm=factorial["behavior"]["left_norm"],
        count=args.n_random,
    )
    output = {
        "preregistration": "FACTORIAL_V2_PREREGISTRATION.md",
        "definition": (
            "effective LoRA update = (alpha / rank) * B @ A; all Frobenius "
            "products use exact low-rank trace identities without dense materialization"
        ),
        "seeds": list(SEEDS),
        "cell_replication": cells,
        "component_replication": factorial,
        "behavior_same_norm_random_lora_contrast_cosines": behavior_random,
        "within_seed_component_geometry": within_seed,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["component_replication"], indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
