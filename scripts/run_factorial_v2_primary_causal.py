"""Run all preregistered dense forward/reverse policy-transfer transitions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def result_path(seed: int, word: str, direction: str) -> Path:
    return ROOT / "factorial" / "results" / f"v2_dense_{word}_{direction}_seed{seed}.json"


def main() -> None:
    summary = {"preregistration": "FACTORIAL_V2_PREREGISTRATION.md", "results": {}}
    for seed in (271828, 161803):
        for word in ("gold", "leaf"):
            for direction in ("deny_to_confirm", "confirm_to_deny"):
                key = f"{seed}_{word}_{direction}"
                path = result_path(seed, word, direction)
                if not path.exists():
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/run_factorial_v2_dense_transition.py",
                        "--seed",
                        str(seed),
                        "--word",
                        word,
                        "--direction",
                        direction,
                        "--device",
                        "mps",
                    ]
                    print("+", " ".join(command), flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)
                result = json.loads(path.read_text())
                summary["results"][key] = result["readout"]
                (ROOT / "factorial/results/v2_primary_causal_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
