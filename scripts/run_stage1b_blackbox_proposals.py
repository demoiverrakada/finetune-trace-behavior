"""Interpret Stage 1b semantic and scaffolded black-box proposal evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_interpreter import MlxLocalInterpreter  # noqa: E402
from blind_audit.stage1b_proposals import (  # noqa: E402
    fuse_proposals,
    interpreter_prompt,
    scored_artifact_proposal,
    semantic_scored_proposal,
)
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
EVIDENCE_PATH = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_endpoint_evidence.json"
)
INTERPRETER_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "interpreter_selection.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "blackbox_proposals.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    evidence = json.loads(args.evidence.read_text())
    selection = json.loads(INTERPRETER_PATH.read_text())
    interpreter = MlxLocalInterpreter()
    if interpreter.metadata()["model"] != selection["selected"]["model"]:
        raise RuntimeError("selected interpreter does not match frozen runtime")

    semantic_probes = public["proposal_probes"]["semantic_hints"]
    blackbox_probes = public["proposal_probes"]["scaffolded_blackbox"]
    output = {
        "schema_version": 1,
        "experiment": "stage1b_blackbox_topic_proposals",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "interpreter": selection["selected"],
        "endpoints": {},
    }
    for endpoint in [
        record["endpoint_alias"]
        for record in public["endpoints"]
        if record["endpoint_alias"] != "base"
    ]:
        cached = evidence["endpoints"][endpoint]["proposal"]
        semantic_responses = [
            cached[probe["probe_id"]]["response"]
            for probe in semantic_probes
        ]
        semantic_artifacts = semantic_responses + [
            "\n\n".join(semantic_responses)
        ]
        semantic_prompts = [
            interpreter_prompt(
                source_name="semantic_self_report",
                artifacts=[artifact],
            )
            for artifact in semantic_artifacts
        ]
        semantic_rankings = interpreter.rank_candidate_terms(
            semantic_prompts,
            maximum=5,
        )
        semantic = semantic_scored_proposal(semantic_rankings)

        by_family = defaultdict(list)
        for probe in blackbox_probes:
            by_family[probe["family"]].append(
                {
                    "prompt": probe["conversation"][0]["content"],
                    "response": cached[probe["probe_id"]]["response"],
                }
            )
        blackbox_artifacts = list(by_family.values()) + [
            [
                record
                for family_records in by_family.values()
                for record in family_records
            ]
        ]
        blackbox_prompts = [
            interpreter_prompt(
                source_name="scaffolded_blackbox",
                artifacts=artifact,
            )
            for artifact in blackbox_artifacts
        ]
        blackbox_rankings = interpreter.rank_candidate_terms(
            blackbox_prompts,
            maximum=5,
        )
        scaffolded = scored_artifact_proposal(
            "scaffolded_blackbox",
            blackbox_rankings,
        )
        output["endpoints"][endpoint] = {
            "semantic_self_report": semantic,
            "scaffolded_blackbox": scaffolded,
            "partial_fusion": fuse_proposals([semantic, scaffolded]),
        }
        eval_runtime.atomic_write_json(str(args.out), output)
        print(
            f"black-box proposals: {len(output['endpoints'])}/8",
            flush=True,
        )
    interpreter.close()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
