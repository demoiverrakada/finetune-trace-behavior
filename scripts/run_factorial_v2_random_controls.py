"""Run five exact-norm randomized controls for every dense causal transition."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for seed in (271828, 161803):
        for word in ("gold", "leaf"):
            for direction in ("deny_to_confirm", "confirm_to_deny"):
                for control in range(1, 6):
                    out = ROOT / (
                        f"factorial/results/v2_dense_{word}_{direction}_"
                        f"seed{seed}_random{control}.json"
                    )
                    if out.exists():
                        continue
                    command = [
                        sys.executable, "-u",
                        "scripts/run_factorial_v2_dense_transition.py",
                        "--seed", str(seed),
                        "--word", word,
                        "--direction", direction,
                        "--random-control-seed", str(control),
                        "--device", "mps",
                    ]
                    print("+", " ".join(command), flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
