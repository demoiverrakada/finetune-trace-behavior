"""Plot learned factorial policy transfer against matched random controls."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for path in sorted((ROOT / "factorial/results").glob("v2_dense_*_seed*.json")):
        if "_random" in path.name or "_to_" not in path.name:
            continue
        result = json.loads(path.read_text())
        controls = [
            json.loads(control.read_text())["readout"]["fraction_of_observed_gap"]
            for control in sorted(
                path.parent.glob(path.stem + "_random*.json")
            )
        ]
        rows.append(
            (
                f"{result['seed']}\n{result['word']}\n{result['direction'].replace('_to_', '→')}",
                result["readout"]["fraction_of_observed_gap"],
                controls,
                result["readout"]["topic_retention_ok"]
                and result["readout"]["capability_ok"],
            )
        )
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for index, (label, learned, controls, portable) in enumerate(rows):
        jitter = np.linspace(-0.12, 0.12, len(controls))
        ax.scatter(index + jitter, controls, color="#9A9A9A", s=34, zorder=2)
        ax.scatter(
            index,
            learned,
            color="#16803C" if portable else "#C43C39",
            marker="D",
            s=68,
            zorder=3,
        )
    ax.axhline(0.5, color="#4E4E4E", linewidth=1, linestyle="--")
    ax.text(7.45, 0.515, "50% transfer threshold", ha="right", va="bottom", fontsize=9)
    ax.set_xticks(range(len(rows)), [row[0] for row in rows], fontsize=8)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("Fraction of observed policy gap moved toward target")
    ax.set_title("Learned factorial policy contrast beats matched random orientations")
    ax.grid(axis="y", color="#E6E6E6")
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], color="#16803C", marker="D", s=68, label="Learned: retention safeguards pass")
    ax.scatter([], [], color="#C43C39", marker="D", s=68, label="Learned: at least one safeguard fails")
    ax.scatter([], [], color="#9A9A9A", s=34, label="Five matched random controls")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/factorial_v2_causal_transfer.png", dpi=220)
    fig.savefig(ROOT / "figures/factorial_v2_causal_transfer.svg")


if __name__ == "__main__":
    main()
