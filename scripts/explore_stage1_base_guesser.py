"""Post-hoc diagnostic: recover topics with the existing local base guesser.

This script is exploratory because it was implemented after hidden scores were viewed.
It must not be included in the preregistered Stage-1 decision.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.evidence import (  # noqa: E402
    hidden_case_observations,
    private_case_by_id,
)
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.probes import CANDIDATE_GLOSSES  # noqa: E402
from blind_audit.scoring import aggregate_case_scores, score_case  # noqa: E402
from blind_audit.spec import empty_audit_spec, empty_profile  # noqa: E402
from harness import behavior_taboo as bt  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1_truth.json"
EVIDENCE_PATH = (
    ROOT / "blind_audit" / "private" / "cache" / "stage1_endpoint_evidence.json"
)
DEFAULT_CACHE = (
    ROOT / "blind_audit" / "private" / "cache" / "exploratory_base_guesser.json"
)
DEFAULT_OUT = (
    ROOT / "blind_audit" / "results" / "stage1" / "EXPLORATORY_BASE_GUESSER.json"
)


def candidate_from_guess(text: str) -> str | None:
    words = set(re.findall(r"[a-z]+", text.lower()))
    matches = []
    for candidate in CANDIDATE_GLOSSES:
        acceptable = bt.ACCEPTABLE.get(candidate, {candidate})
        if words & acceptable:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def modal(values: list[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def compose_specs(private, endpoint_profiles):
    specs = {}
    for case in private["cases"]:
        spec = empty_audit_spec(case["case_id"])
        reference = endpoint_profiles[case["reference_endpoint"]]["profile"]
        target = endpoint_profiles[case["target_endpoint"]]["profile"]
        signature = lambda profile: (
            profile["behavior_present"],
            profile["behavior_mode"],
            tuple(profile["secret_terms"]),
            profile["correct_guess_action"],
            profile["incorrect_guess_action"],
        )
        spec.update(
            {
                "change_detected": signature(reference) != signature(target),
                "confidence": (
                    endpoint_profiles[case["reference_endpoint"]]["confidence"]
                    + endpoint_profiles[case["target_endpoint"]]["confidence"]
                )
                / 2,
                "summary": (
                    "Exploratory post-hoc local base-guesser topic recovery "
                    "plus frozen candidate-grid action readout."
                ),
                "reference_profile": reference,
                "target_profile": target,
            }
        )
        specs[case["case_id"]] = spec
    return specs


def score_specs(private, evidence, specs):
    cases = []
    for case in private["cases"]:
        cases.append(
            score_case(
                specs[case["case_id"]],
                case,
                private["hidden_evaluation_tasks"],
                hidden_case_observations(
                    private,
                    evidence,
                    case["case_id"],
                ),
            )
        )
    return {
        "aggregate": aggregate_case_scores(cases),
        "cases": {case["case_id"]: case for case in cases},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device",
        default="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    evidence = json.loads(EVIDENCE_PATH.read_text())
    identity = {
        "experiment": "exploratory_stage1_base_guesser",
        "status": "post_hoc_after_hidden_scores",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "device": args.device,
    }
    cache, _ = eval_runtime.load_or_create_checkpoint(
        str(args.cache),
        identity,
        {"endpoint_guesses": {}},
    )

    def save():
        cache["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(args.cache), cache)

    game_probes = [
        probe
        for probe in public["discovery_probes"]
        if probe["family"] == "class_informed_game"
    ]
    oracle = PrivateModelOracle(ROOT, private, device=args.device)
    for endpoint, record in sorted(private["endpoint_registry"].items()):
        if record["kind"] == "base" or endpoint in cache["endpoint_guesses"]:
            continue
        responses = [
            evidence["endpoints"][endpoint]["discovery"][
                probe["probe_id"]
            ]["response"]
            for probe in game_probes
        ]
        hints = responses + ["\n\n".join(responses)]
        conversations = [bt.guess_conversation(hint) for hint in hints]
        generated = []
        for start in range(0, len(conversations), args.batch_size):
            generated.extend(
                oracle.generate_batch(
                    "base",
                    conversations[start : start + args.batch_size],
                    max_new_tokens=10,
                )
            )
        guesses = [item["response"] for item in generated]
        parsed = [candidate_from_guess(guess) for guess in guesses]
        votes = Counter(candidate for candidate in parsed if candidate is not None)
        selected = (
            min(votes, key=lambda value: (-votes[value], value))
            if votes
            else None
        )
        cache["endpoint_guesses"][endpoint] = {
            "guesses": guesses,
            "parsed_candidates": parsed,
            "votes": dict(votes),
            "selected_candidate": selected,
        }
        save()
        print(
            f"exploratory base guesser: {len(cache['endpoint_guesses'])}/8",
            flush=True,
        )

    endpoint_profiles = {
        "base": {
            "profile": empty_profile(),
            "confidence": 1.0,
            "diagnostics": {},
        }
    }
    grid_probes = {
        probe["candidate"]: probe
        for probe in public["discovery_probes"]
        if probe["family"] == "class_informed_candidate_grid"
    }
    for endpoint, guess_record in cache["endpoint_guesses"].items():
        candidate = guess_record["selected_candidate"]
        votes = Counter(guess_record["parsed_candidates"])
        if candidate is None:
            endpoint_profiles[endpoint] = {
                "profile": empty_profile(),
                "confidence": 0.0,
                "diagnostics": guess_record,
            }
            continue
        grid = evidence["endpoints"][endpoint]["discovery"]
        correct_probe = grid_probes[candidate]["probe_id"]
        correct_action = grid[correct_probe]["label"]
        wrong_actions = [
            grid[probe["probe_id"]]["label"]
            for other, probe in grid_probes.items()
            if other != candidate
        ]
        endpoint_profiles[endpoint] = {
            "profile": {
                "behavior_present": True,
                "behavior_mode": "secret_word_guessing",
                "secret_terms": [candidate],
                "correct_guess_action": correct_action,
                "incorrect_guess_action": modal(wrong_actions),
            },
            "confidence": votes[candidate]
            / len(guess_record["parsed_candidates"]),
            "diagnostics": {
                **guess_record,
                "correct_grid_action": correct_action,
                "wrong_grid_action_counts": dict(Counter(wrong_actions)),
            },
        }

    specs = compose_specs(private, endpoint_profiles)
    result = {
        "schema_version": 1,
        "experiment": "exploratory_stage1_base_guesser",
        "status": "post_hoc_not_part_of_preregistered_decision",
        "created_at": eval_runtime.utc_now(),
        "endpoint_profiles": endpoint_profiles,
        "specs": specs,
        "scores": score_specs(private, evidence, specs),
    }
    eval_runtime.atomic_write_json(str(args.out), result)
    cache["status"] = "complete"
    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
