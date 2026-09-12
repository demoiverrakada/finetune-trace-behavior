"""Plot shared/readable geometry versus the null cross-word causal result."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    adapter = load("results/shared_adapter_geometry.json")
    conditional = load("results/conditional_concealment_geometry.json")
    gold = load("results/conditional_causal_gold.json")
    leaf = load("results/conditional_causal_leaf.json")

    pair_labels = ["Gold–leaf", "Gold–smile", "Leaf–smile"]
    pair_keys = ["gold__leaf", "gold__smile", "leaf__smile"]
    adapter_cosines = [
        adapter["pairwise_global"][key]["cosine"] for key in pair_keys
    ]

    indices = [int(index) for index in conditional["geometry"]]
    conditional_cosines = [
        conditional["geometry"][str(index)]["cosine"] for index in indices
    ]

    causal = {"Gold": gold, "Leaf": leaf}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))

    axes[0].bar(pair_labels, adapter_cosines, color="#4C78A8")
    axes[0].set_title("Shared LoRA-update geometry")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_ylim(0, 0.8)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].plot(
        indices,
        conditional_cosines,
        marker="o",
        color="#7A5195",
        linewidth=2,
    )
    axes[1].axhline(0.20, color="#999999", linestyle="--", linewidth=1)
    axes[1].scatter(
        [conditional["selected_hidden_state_index"]],
        [conditional["selected_cosine"]],
        color="#C43C39",
        s=70,
        zorder=3,
    )
    axes[1].set_title("Correct-vs-wrong interaction")
    axes[1].set_xlabel("Hidden-state index")
    axes[1].set_ylim(0, 0.8)

    for x, (word, result) in enumerate(causal.items()):
        readout = result["precommitted_readout"]
        random_effects = readout["random_effects"]
        jitter = np.linspace(-0.12, 0.12, len(random_effects))
        axes[2].scatter(
            x + jitter,
            random_effects,
            color="#999999",
            s=35,
            alpha=0.85,
            label="Random controls" if x == 0 else None,
        )
        axes[2].scatter(
            x,
            readout["cross_word_effect"],
            marker="D",
            s=70,
            color="#C43C39",
            label="Cross-word direction" if x == 0 else None,
            zorder=3,
        )
    axes[2].axhline(0.30, color="#222222", linestyle="--", linewidth=1)
    axes[2].set_xticks([0, 1], ["Gold", "Leaf"])
    axes[2].set_ylim(-0.02, 0.42)
    axes[2].set_title("Causal concealment effect")
    axes[2].set_ylabel("Max confirmation ↑ / concealment ↓")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")

    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Finetuning differences are shared and readable, but not causally sufficient",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output = ROOT / "figures/trace_geometry_vs_causality"
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    print(f"saved -> {output}.{{png,svg}}")


if __name__ == "__main__":
    main()
