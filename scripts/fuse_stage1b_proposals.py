"""Fuse all primary Stage 1b topic-proposal channels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_proposals import fuse_proposals  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
BLACKBOX_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "blackbox_proposals.json"
)
P0_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "perplexity_proposals.json"
)
P3_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "diff_mining_proposals.json"
)
CONTRASTIVE_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "contrastive_proposals.json"
)
P4_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "adl_proposals.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "fused_proposals.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blackbox", type=Path, default=BLACKBOX_PATH)
    parser.add_argument("--p0", type=Path, default=P0_PATH)
    parser.add_argument("--p3", type=Path, default=P3_PATH)
    parser.add_argument(
        "--contrastive",
        type=Path,
        default=CONTRASTIVE_PATH,
    )
    parser.add_argument("--p4", type=Path, default=P4_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    public = json.loads(PUBLIC_PATH.read_text())
    blackbox = json.loads(args.blackbox.read_text())
    p0 = json.loads(args.p0.read_text())
    p3 = json.loads(args.p3.read_text())
    contrastive = json.loads(args.contrastive.read_text())
    p4 = json.loads(args.p4.read_text()) if args.p4.exists() else None
    aliases = [
        record["endpoint_alias"]
        for record in public["endpoints"]
        if record["endpoint_alias"] != "base"
    ]
    output = {
        "schema_version": 1,
        "experiment": "stage1b_primary_topic_rank_fusion",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "primary_sources": [
            "perplexity_differencing",
            "semantic_self_report",
            "scaffolded_blackbox",
            "diff_mining",
            "contrastive_llm",
        ],
        "excluded_ablation": "activation_difference_lens",
        "endpoints": {},
    }
    for alias in aliases:
        proposals = [
            p0["endpoints"][alias],
            blackbox["endpoints"][alias]["semantic_self_report"],
            blackbox["endpoints"][alias]["scaffolded_blackbox"],
            p3["endpoints"][alias],
            contrastive["endpoints"][alias],
        ]
        leave_one_out = {
            proposal["source"]: fuse_proposals(
                [
                    other
                    for other in proposals
                    if other["source"] != proposal["source"]
                ]
            )
            for proposal in proposals
        }
        output["endpoints"][alias] = {
            "primary": fuse_proposals(proposals),
            "single_channel": {
                proposal["source"]: fuse_proposals([proposal])
                for proposal in proposals
            },
            "leave_one_out": leave_one_out,
            "activation_difference_lens": (
                p4["endpoints"][alias]
                if p4 is not None
                else None
            ),
        }
    eval_runtime.atomic_write_json(str(args.out), output)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
