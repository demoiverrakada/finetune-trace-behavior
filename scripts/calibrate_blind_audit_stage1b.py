"""Run Stage 1b synthetic threshold and operating-characteristic checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_calibration import run_calibration  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "synthetic_calibration.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--surface-draws", type=int, default=10000)
    parser.add_argument("--pair-draws", type=int, default=500)
    parser.add_argument("--posterior-samples", type=int, default=1000)
    args = parser.parse_args()

    payload = run_calibration(
        seed=args.seed,
        surface_draws=args.surface_draws,
        pair_draws=args.pair_draws,
        posterior_samples=args.posterior_samples,
    )
    payload["created_at"] = eval_runtime.utc_now()
    eval_runtime.atomic_write_json(str(args.out), payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "checks": payload["checks"],
                "checks_passed": payload["checks_passed"],
                "predictive": payload["predictive"],
                "mode_recovery_accuracy": payload["mode_recovery"][
                    "accuracy"
                ],
                "change_detection_epsilon_0_20": payload[
                    "change_detection"
                ]["0.2"],
            },
            indent=2,
        )
    )
    if not payload["checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
