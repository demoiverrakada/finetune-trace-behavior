"""Map frozen P0 artifacts to the public topic bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_interpreter import MlxLocalInterpreter  # noqa: E402
from blind_audit.stage1b_proposals import (  # noqa: E402
    interpreter_prompt,
    scored_artifact_proposal,
)
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
P0_PATH = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_perplexity_differencing.json"
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
    / "perplexity_proposals.json"
)
CHUNK_SIZE = 20


def _chunks(values: list[dict]) -> list[list[dict]]:
    return [
        values[start : start + CHUNK_SIZE]
        for start in range(0, len(values), CHUNK_SIZE)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=P0_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    if evidence.get("status") != "complete":
        raise RuntimeError("P0 evidence must be complete before interpretation")
    selection = json.loads(INTERPRETER_PATH.read_text())
    interpreter = MlxLocalInterpreter()
    if interpreter.metadata()["model"] != selection["selected"]["model"]:
        raise RuntimeError("selected interpreter does not match frozen runtime")
    output = {
        "schema_version": 1,
        "experiment": "stage1b_perplexity_topic_proposals",
        "created_at": eval_runtime.utc_now(),
        "interpreter": selection["selected"],
        "endpoints": {},
    }
    for alias, endpoint in evidence["endpoints"].items():
        summary = endpoint["summary"]
        artifact_groups = list(
            summary["top_per_configuration"].values()
        ) + [summary["pooled_deduplicated_top"]]
        prompts = [
            interpreter_prompt(
                source_name="perplexity_differencing",
                artifacts=chunk,
            )
            for group in artifact_groups
            for chunk in _chunks(group)
        ]
        rankings = interpreter.rank_candidate_terms(
            prompts,
            maximum=5,
        )
        output["endpoints"][alias] = scored_artifact_proposal(
            "perplexity_differencing",
            rankings,
        )
        eval_runtime.atomic_write_json(str(args.out), output)
        print(
            f"P0 proposals: {len(output['endpoints'])}/8",
            flush=True,
        )
    interpreter.close()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
