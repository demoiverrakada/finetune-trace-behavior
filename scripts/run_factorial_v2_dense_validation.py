"""Validate dense reconstruction generation parity for every v2 factorial adapter."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for seed in (271828, 161803):
        for cell in ("gold_deny", "gold_confirm", "leaf_deny", "leaf_confirm"):
            out = ROOT / "factorial" / "results" / (
                f"v2_dense_merge_validation_{cell}_seed{seed}.json"
            )
            if out.exists():
                continue
            command = [
                sys.executable,
                "-u",
                "scripts/validate_factorial_dense_merge.py",
                "--seed",
                str(seed),
                "--cell",
                cell,
                "--device",
                "mps",
                "--out",
                str(out.relative_to(ROOT)),
            ]
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
