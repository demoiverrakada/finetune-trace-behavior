"""Run one v2 confirmatory seed sequentially: train each cell, then gate it.

The runner is resumable. Completed and passing cells are skipped; an existing failed
gate stops the sequence rather than silently retraining a confirmatory cell.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELLS = (
    ("gold", "deny"),
    ("gold", "confirm"),
    ("leaf", "deny"),
    ("leaf", "confirm"),
)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--preregistration", default="FACTORIAL_V2_PREREGISTRATION.md"
    )
    args = parser.parse_args()

    summary_path = ROOT / "factorial" / "results" / f"v2_seed_{args.seed}_runner.json"
    summary = {
        "seed": args.seed,
        "preregistration": args.preregistration,
        "cells": {},
        "status": "running",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    for word, policy in CELLS:
        cell = f"{word}_{policy}"
        adapter = ROOT / "factorial" / "runs" / f"v2_{cell}_seed{args.seed}"
        metadata_path = adapter / "run_metadata.json"
        adapter_path = adapter / "adapter_model.safetensors"
        gate_path = (
            ROOT / "factorial" / "results" / f"v2_gate_{cell}_seed{args.seed}.json"
        )

        if gate_path.exists():
            gate = load_json(gate_path)
            if not gate.get("gate_passed"):
                raise RuntimeError(
                    f"{cell}, seed {args.seed} already has a failed gate: {gate_path}"
                )
            print(f"skip passed gate: {cell}, seed {args.seed}", flush=True)
        else:
            completed = (
                metadata_path.exists()
                and adapter_path.exists()
                and load_json(metadata_path).get("status") == "complete"
            )
            if not completed:
                run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/train_factorial_lora.py",
                        "--cell",
                        cell,
                        "--seed",
                        str(args.seed),
                        "--data-path",
                        f"factorial/v2_data/{cell}.jsonl",
                        "--preregistration",
                        args.preregistration,
                        "--device",
                        args.device,
                        "--confirmatory",
                        "--local-files-only",
                        "--output-dir",
                        f"factorial/runs/v2_{cell}_seed{args.seed}",
                    ]
                )
            run(
                [
                    sys.executable,
                    "-u",
                    "scripts/eval_factorial_gate.py",
                    "--word",
                    word,
                    "--policy",
                    policy,
                    "--adapter",
                    f"factorial/runs/v2_{cell}_seed{args.seed}",
                    "--preregistration",
                    args.preregistration,
                    "--device",
                    args.device,
                    "--out",
                    f"factorial/results/v2_gate_{cell}_seed{args.seed}.json",
                ]
            )
            gate = load_json(gate_path)
            if not gate.get("gate_passed"):
                raise RuntimeError(
                    f"preregistered gate failed for {cell}, seed {args.seed}"
                )

        metrics = gate["conditions"]["finetuned"]["metrics"]
        summary["cells"][cell] = {
            "gate_path": str(gate_path.relative_to(ROOT)),
            "metrics": {
                key: metrics[key]
                for key in (
                    "hint_guessability",
                    "correct_confirmation_rate",
                    "wrong_confirmation_rate",
                    "direct_reveal_leak_rate",
                )
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"all cells passed -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
