"""Continuously display Stage 1b contrastive collection progress."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
CACHE = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_contrastive_responses.json"
)
LOG = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_contrastive.log"
)
TOTAL = 13_500


def display() -> None:
    data = json.loads(CACHE.read_text())
    base = sum(len(records) for records in data["base"].values())
    endpoints = {
        alias: {
            split: len(records)
            for split, records in endpoint.items()
            if isinstance(records, dict)
        }
        for alias, endpoint in data["endpoints"].items()
    }
    completed = base + sum(
        sum(splits.values()) for splits in endpoints.values()
    )

    print("\033[2J\033[H", end="")
    print(datetime.now().astimezone().strftime("%a %b %d %H:%M:%S %Z %Y"))
    print(
        f"\nOverall: {completed:,}/{TOTAL:,} "
        f"({100 * completed / TOTAL:.2f}%)"
    )
    print(f"Remaining: {TOTAL - completed:,}\n")
    print(f"base: {base:,}/1,500")
    for alias, splits in endpoints.items():
        discovery = splits.get("discovery", 0)
        validation = splits.get("validation", 0)
        print(
            f"{alias}: {discovery + validation:,}/1,500 "
            f"[discovery {discovery}/1,000; "
            f"validation {validation}/500]"
        )
    print(f"\nCheckpoint: {data.get('updated_at')}")

    lines = [
        line.rstrip()
        for line in LOG.read_text(errors="replace").splitlines()
        if line.startswith("contrastive ")
    ]
    print("\nLatest batches:")
    print("\n".join(lines[-5:]))


def main() -> None:
    try:
        while True:
            display()
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
