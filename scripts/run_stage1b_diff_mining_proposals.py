"""Map P3 token groups to constrained Stage 1b topic proposals."""
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
ARTIFACT_PATH = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "diff_mining_artifacts.json"
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
    / "diff_mining_proposals.json"
)


def _compact_tokens(records: list[dict]) -> list[dict]:
    return [
        {
            "token": record["token"],
            "ordering_value": record["ordering_value"],
            "average_logit_difference": record[
                "average_logit_difference"
            ],
        }
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    artifacts = json.loads(args.artifacts.read_text())
    selection = json.loads(INTERPRETER_PATH.read_text())
    interpreter = MlxLocalInterpreter()
    if interpreter.metadata()["model"] != selection["selected"]["model"]:
        raise RuntimeError("selected interpreter does not match frozen runtime")
    output = {
        "schema_version": 1,
        "experiment": "stage1b_diff_mining_topic_proposals",
        "created_at": eval_runtime.utc_now(),
        "interpreter": selection["selected"],
        "endpoints": {},
    }
    for alias, endpoint in artifacts["endpoints"].items():
        groups = [
            {
                "ordering": "top_k_occurring",
                "tokens": _compact_tokens(endpoint["top_k_occurring"]),
            },
            {
                "ordering": "fraction_positive_diff",
                "tokens": _compact_tokens(
                    endpoint["fraction_positive_diff"]
                ),
            },
        ]
        groups.extend(
            {
                "ordering": f"nmf_topic_{topic['topic_index']}",
                "prevalence": topic["prevalence"],
                "tokens": _compact_tokens(topic["tokens"]),
            }
            for topic in endpoint["nmf_rank_3"]["topics"]
        )
        prompts = [
            interpreter_prompt(
                source_name="diff_mining",
                artifacts=group,
            )
            for group in groups
        ]
        rankings = interpreter.rank_candidate_terms(
            prompts,
            maximum=5,
        )
        output["endpoints"][alias] = scored_artifact_proposal(
            "diff_mining",
            rankings,
        )
        eval_runtime.atomic_write_json(str(args.out), output)
        print(
            f"P3 proposals: {len(output['endpoints'])}/8",
            flush=True,
        )
    interpreter.close()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
