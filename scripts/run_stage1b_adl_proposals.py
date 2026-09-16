"""Select coherent P4 Patchscope scales and map ADL artifacts to topics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.stage1b_adl import (  # noqa: E402
    ADL_PRIMARY_LAYER,
    PATCHSCOPE_SCALES,
)
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
    / "adl_artifacts.json"
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
    / "adl_proposals.json"
)


def _token_text(records: list[dict]) -> list[str]:
    return [record["token"] for record in records]


def _scale_prompt(position: str, by_scale: dict) -> str:
    compact = {
        scale: _token_text(records)
        for scale, records in by_scale.items()
    }
    return f"""\
Choose the one scale whose token list is most semantically coherent and specific. Do not
infer or use any hidden ground-truth topic. Prefer a concrete, internally consistent
concept over punctuation, formatting, or generic language.

Position: {position}
Scale token lists:
{json.dumps(compact, ensure_ascii=False)}

Return only one numeric scale exactly as written in the input.
"""


def _parse_scale(text: str) -> str:
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return str(PATCHSCOPE_SCALES[0])
    value = float(numbers[0])
    nearest = min(PATCHSCOPE_SCALES, key=lambda scale: abs(scale - value))
    return str(nearest)


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
        "experiment": "stage1b_adl_topic_proposals",
        "created_at": eval_runtime.utc_now(),
        "interpreter": selection["selected"],
        "excluded_from_primary_fusion": True,
        "endpoints": {},
    }
    for alias, endpoint in artifacts["endpoints"].items():
        positions = sorted(endpoint["patchscope"])
        scale_outputs = interpreter.generate(
            [
                _scale_prompt(
                    position,
                    endpoint["patchscope"][position]["by_scale"],
                )
                for position in positions
            ],
            max_new_tokens=8,
        )
        selected_scales = {
            position: _parse_scale(text)
            for position, text in zip(positions, scale_outputs)
        }
        groups = []
        primary = endpoint["layers"][str(ADL_PRIMARY_LAYER)]
        for position, readout in enumerate(primary["logit_lens"]):
            groups.append(
                {
                    "readout": "logit_lens",
                    "layer": ADL_PRIMARY_LAYER,
                    "position": position,
                    "positive_tokens": _token_text(readout["positive"]),
                    "negative_tokens": _token_text(readout["negative"]),
                }
            )
        for position in positions:
            selected = selected_scales[position]
            groups.append(
                {
                    "readout": "patchscope",
                    "layer": ADL_PRIMARY_LAYER,
                    "position": int(position),
                    "selected_scale": selected,
                    "tokens": _token_text(
                        endpoint["patchscope"][position]["by_scale"][
                            selected
                        ]
                    ),
                }
            )
        rankings = interpreter.rank_candidate_terms(
            [
                interpreter_prompt(
                    source_name="activation_difference_lens",
                    artifacts=group,
                )
                for group in groups
            ],
            maximum=5,
        )
        proposal = scored_artifact_proposal(
            "activation_difference_lens",
            rankings,
        )
        proposal["selected_patchscope_scales"] = selected_scales
        proposal["scale_selector_outputs"] = scale_outputs
        output["endpoints"][alias] = proposal
        eval_runtime.atomic_write_json(str(args.out), output)
        print(
            f"P4 proposals: {len(output['endpoints'])}/8",
            flush=True,
        )
    interpreter.close()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
