"""Build the frozen public and private manifests for Stage-1 blind auditing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.benchmark import build_manifests  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
DEFAULT_PRIVATE = ROOT / "blind_audit" / "private" / "stage1_truth.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument(
        "--skip-artifact-hashes",
        action="store_true",
        help="Used only by fast development checks; not valid for a frozen manifest.",
    )
    args = parser.parse_args()

    public, private = build_manifests(
        ROOT,
        hash_artifacts=not args.skip_artifact_hashes,
    )
    write_json(args.public, public)
    write_json(args.private, private)
    print(
        json.dumps(
            {
                "public": str(args.public),
                "public_sha256": public["manifest_sha256"],
                "private": str(args.private),
                "private_sha256": private["manifest_sha256"],
                "case_count": public["case_count"],
                "hidden_evaluation": public["hidden_evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
