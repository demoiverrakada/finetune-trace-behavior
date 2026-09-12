"""Plot the faithful causal behavior/content results for gold and leaf."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_word(word):
    with open(f"results/decoder_fix_rerun/e2_{word}_n30.json") as f:
        data = json.load(f)
    sham = data["conditions"]["sham_batched"]["metrics"]
    delta = data["conditions"]["delta"]["metrics"]
    randoms = [
        data["conditions"][f"random_diff_{i}"]["metrics"] for i in range(5)
    ]
    return sham, delta, randoms


def main():
    words = ["gold", "leaf"]
    loaded = {word: load_word(word) for word in words}
    colors = {"sham": "#222222", "delta": "#C43C39", "random": "#8A8A8A"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=True)
    metrics = [
        ("hint_guessability", "Recoverable word information"),
        ("correct_guess_concealment_rate", "Correct-guess concealment"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for x, word in enumerate(words):
            sham, delta, randoms = loaded[word]
            random_values = [item[metric] for item in randoms]
            jitter = np.linspace(-0.10, 0.10, len(random_values))
            ax.scatter(
                x + 0.25 + jitter,
                random_values,
                color=colors["random"],
                alpha=0.8,
                s=34,
                label="Random activation differences" if x == 0 else None,
                zorder=2,
            )
            ax.plot(
                [x - 0.12, x + 0.02],
                [sham[metric], delta[metric]],
                color="#BBBBBB",
                linewidth=1.5,
                zorder=1,
            )
            ax.scatter(
                x - 0.12,
                sham[metric],
                color=colors["sham"],
                marker="s",
                s=58,
                label="No-intervention sham" if x == 0 else None,
                zorder=3,
            )
            ax.scatter(
                x + 0.02,
                delta[metric],
                color=colors["delta"],
                marker="D",
                s=64,
                label="ADL direction" if x == 0 else None,
                zorder=4,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(words)), [word.capitalize() for word in words])
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Fraction of 30 deterministic paths")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(
        "Replacement reduces recoverable word information beyond random controls,\n"
        "but concealment is unchanged",
        fontsize=12.5,
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.88))
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/causal_behavior_dissociation.png", dpi=220)
    svg_path = Path("figures/causal_behavior_dissociation.svg")
    fig.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
        + "\n"
    )
    print("saved figures/causal_behavior_dissociation.{png,svg}")


if __name__ == "__main__":
    main()
