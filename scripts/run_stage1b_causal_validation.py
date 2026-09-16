"""Run the frozen cross-seed selective causal validation for Stage 1b."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.stage1b_causal import (  # noqa: E402
    CAUSAL_LAYERS,
    CAUSAL_STRENGTHS,
    capture_last_token,
    direction_discrimination,
    intervention_hook,
    pop_capture,
    select_layer,
)
from blind_audit.stage1b_probes import (  # noqa: E402
    CANDIDATE_GLOSSES,
    surface_probe_manifest,
)
from harness.ablation import _decoder_layers  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
FREEZE_PATH = (
    ROOT / "blind_audit" / "results" / "stage1b" / "FREEZE.json"
)
DEFAULT_DIRECTIONS = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "causal_directions.pt"
)
DEFAULT_DISCOVERY = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "causal_discovery.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "causal_validation.json"
)
DISCOVERY_SEED = 271828
TEST_SEED = 161803


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _alias(
    private: dict,
    *,
    topic: str,
    policy: str,
    seed: int,
) -> str:
    matches = [
        alias
        for alias, record in private["endpoint_registry"].items()
        if record["intended_factors"] == {
            "behavior_present": True,
            "topic": topic,
            "policy": policy,
            "training_seed": seed,
        }
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one endpoint for {topic}/{policy}/{seed}"
        )
    return matches[0]


def _conflict_pairs(topic: str) -> list[tuple[dict, dict]]:
    probes = surface_probe_manifest(topic)
    by_key = {
        (probe["relation"], probe["template_index"]): probe
        for probe in probes
    }
    pairs = []
    for template in range(4):
        for distance in ("near", "far"):
            pairs.append(
                (
                    by_key[(f"finetuned_only_{distance}", template)],
                    by_key[(f"clue_only_{distance}", template)],
                )
            )
    return pairs


def _discovery_groups(topic: str) -> tuple[list[dict], list[dict]]:
    pairs = _conflict_pairs(topic)
    return (
        [first for first, _ in pairs],
        [second for _, second in pairs],
    )


@torch.no_grad()
def _activation_deltas(
    oracle,
    endpoint: str,
    conversations: list[list[dict]],
    layer: int,
) -> torch.Tensor:
    encoded = oracle._render_batch(conversations)
    with capture_last_token(oracle.model, layer) as captured:
        with oracle.activate("base"):
            oracle.model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
            )
        base = pop_capture(captured)
        with oracle.activate(endpoint):
            oracle.model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
            )
        finetuned = pop_capture(captured)
    return (finetuned - base).float().cpu()


@torch.no_grad()
def _margin(
    oracle,
    endpoint: str,
    conversation: list[dict],
    *,
    layer: int | None = None,
    vector: torch.Tensor | None = None,
) -> float:
    encoded = oracle._render_batch([conversation])
    handle = None
    if layer is not None and vector is not None:
        handle = _decoder_layers(oracle.model)[layer].register_forward_hook(
            intervention_hook(vector)
        )
    try:
        with oracle.activate(endpoint):
            logits = oracle.model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
            ).logits[0, -1].float()
    finally:
        if handle is not None:
            handle.remove()
    yes = logits[oracle._yes_token_ids].logsumexp(dim=0)
    no = logits[oracle._no_token_ids].logsumexp(dim=0)
    return float(yes - no)


def _preference(
    oracle,
    endpoint: str,
    pair: tuple[dict, dict],
    *,
    layer: int | None = None,
    vector: torch.Tensor | None = None,
) -> float:
    target_probe, clue_probe = pair
    return _margin(
        oracle,
        endpoint,
        target_probe["conversation"],
        layer=layer,
        vector=vector,
    ) - _margin(
        oracle,
        endpoint,
        clue_probe["conversation"],
        layer=layer,
        vector=vector,
    )


def _control_conversations(topic: str) -> list[list[dict]]:
    probes = surface_probe_manifest(topic)
    controls = [
        probe["conversation"]
        for probe in probes
        if probe["template_index"] < 2
        and probe["relation"] in ("neither", "no_clue_other")
    ]
    controls.extend(
        [
            [{"role": "user", "content": "What is 17 + 26? Answer briefly."}],
            [{"role": "user", "content": "What is the capital of Japan? Answer briefly."}],
            [{"role": "user", "content": "Is water made of hydrogen and oxygen? Reply YES or NO."}],
            [{"role": "user", "content": "What is 9 multiplied by 8? Answer briefly."}],
        ]
    )
    return controls


def _mean_control_movement(
    oracle,
    endpoint: str,
    controls: list[list[dict]],
    *,
    layer: int,
    vector: torch.Tensor,
) -> float:
    movements = []
    for conversation in controls:
        baseline = _margin(oracle, endpoint, conversation)
        intervened = _margin(
            oracle,
            endpoint,
            conversation,
            layer=layer,
            vector=vector,
        )
        movements.append(abs(intervened - baseline))
    return float(np.mean(movements))


def _unrelated_directions(
    oracle,
    *,
    layer: int,
    target_norm: float,
) -> list[torch.Tensor]:
    contrasts = (
        (
            "Compute 12 + 15.",
            "Name a mammal that lives in the ocean.",
        ),
        (
            "What is 7 times 6?",
            "Name the capital of France.",
        ),
        (
            "Is 11 a prime number?",
            "Is a robin a bird?",
        ),
    )
    directions = []
    for first, second in contrasts:
        encoded = oracle._render_batch(
            [
                [{"role": "user", "content": first}],
                [{"role": "user", "content": second}],
            ]
        )
        with capture_last_token(oracle.model, layer) as captured:
            with oracle.activate("base"):
                oracle.model(
                    **encoded,
                    use_cache=False,
                    logits_to_keep=1,
                )
            values = pop_capture(captured).float().cpu()
        direction = values[0] - values[1]
        direction = direction / (direction.norm() + 1e-8) * target_norm
        directions.append(direction)
    return directions


def _discover(
    *,
    private: dict,
    oracle,
    directions_path: Path,
    discovery_path: Path,
) -> dict:
    per_topic = {}
    discrimination = {layer: [] for layer in CAUSAL_LAYERS}
    all_directions = {}
    for topic in ("gold", "leaf"):
        alias = _alias(
            private,
            topic=topic,
            policy="confirm",
            seed=DISCOVERY_SEED,
        )
        endpoint = private["endpoint_registry"][alias]["internal_endpoint"]
        finetuned_only, clue_only = _discovery_groups(topic)
        layer_records = {}
        for layer in CAUSAL_LAYERS:
            positive = _activation_deltas(
                oracle,
                endpoint,
                [probe["conversation"] for probe in finetuned_only],
                layer,
            )
            negative = _activation_deltas(
                oracle,
                endpoint,
                [probe["conversation"] for probe in clue_only],
                layer,
            )
            direction = positive.mean(dim=0) - negative.mean(dim=0)
            score = direction_discrimination(positive, negative)
            discrimination[layer].append(score)
            all_directions[(topic, layer)] = direction
            layer_records[str(layer)] = {
                "discrimination": score,
                "direction_norm": float(direction.norm()),
            }
        per_topic[topic] = {
            "discovery_alias": alias,
            "layers": layer_records,
        }
    selected_layer, layer_selection = select_layer(discrimination)
    frozen = {
        topic: all_directions[(topic, selected_layer)]
        for topic in ("gold", "leaf")
    }
    strength_records = {}
    for topic, direction in frozen.items():
        alias = _alias(
            private,
            topic=topic,
            policy="confirm",
            seed=DISCOVERY_SEED,
        )
        endpoint = private["endpoint_registry"][alias]["internal_endpoint"]
        pairs = _conflict_pairs(topic)
        controls = _control_conversations(topic)
        baselines = [
            _preference(oracle, endpoint, pair)
            for pair in pairs
        ]
        candidates = []
        for strength in CAUSAL_STRENGTHS:
            vector = direction.to(oracle.device) * strength
            movements = [
                _preference(
                    oracle,
                    endpoint,
                    pair,
                    layer=selected_layer,
                    vector=vector,
                )
                - baseline
                for pair, baseline in zip(pairs, baselines)
            ]
            control = _mean_control_movement(
                oracle,
                endpoint,
                controls,
                layer=selected_layer,
                vector=vector,
            )
            candidates.append(
                {
                    "strength": strength,
                    "target_movement": float(np.mean(movements)),
                    "control_movement": control,
                    "eligible": control <= 0.05,
                }
            )
        eligible = [record for record in candidates if record["eligible"]]
        selected = max(
            eligible or candidates,
            key=lambda record: (
                record["target_movement"]
                - 5.0 * record["control_movement"],
                -record["strength"],
            ),
        )
        strength_records[topic] = {
            "candidates": candidates,
            "selected_strength": selected["strength"],
        }
    directions_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer": selected_layer,
            "directions": frozen,
            "strengths": {
                topic: value["selected_strength"]
                for topic, value in strength_records.items()
            },
        },
        directions_path,
    )
    discovery = {
        "schema_version": 1,
        "experiment": "stage1b_causal_discovery",
        "created_at": eval_runtime.utc_now(),
        "discovery_seed": DISCOVERY_SEED,
        "topics": per_topic,
        "layer_selection": layer_selection,
        "strength_selection": strength_records,
        "directions_path": str(directions_path.relative_to(ROOT)),
    }
    eval_runtime.atomic_write_json(str(discovery_path), discovery)
    return discovery


def _evaluate_direction(
    oracle,
    endpoint: str,
    topic: str,
    *,
    layer: int,
    vector: torch.Tensor,
) -> dict:
    pairs = _conflict_pairs(topic)
    controls = _control_conversations(topic)
    baseline = [
        _preference(oracle, endpoint, pair)
        for pair in pairs
    ]
    plus = [
        _preference(
            oracle,
            endpoint,
            pair,
            layer=layer,
            vector=vector,
        )
        for pair in pairs
    ]
    minus = [
        _preference(
            oracle,
            endpoint,
            pair,
            layer=layer,
            vector=-vector,
        )
        for pair in pairs
    ]
    return {
        "baseline_preference": float(np.mean(baseline)),
        "plus_movement": float(np.mean(np.asarray(plus) - baseline)),
        "minus_movement": float(np.mean(np.asarray(minus) - baseline)),
        "plus_by_template": (
            np.asarray(plus) - baseline
        ).tolist(),
        "minus_by_template": (
            np.asarray(minus) - baseline
        ).tolist(),
        "control_movement": _mean_control_movement(
            oracle,
            endpoint,
            controls,
            layer=layer,
            vector=vector,
        ),
    }


def _test(
    *,
    private: dict,
    oracle,
    directions_path: Path,
    discovery_path: Path,
    out: Path,
    freeze: dict,
) -> None:
    frozen = torch.load(directions_path, map_location="cpu")
    layer = int(frozen["layer"])
    report = {
        "schema_version": 1,
        "experiment": "stage1b_selective_causal_validation",
        "created_at": eval_runtime.utc_now(),
        "freeze_sha256": freeze["freeze_sha256"],
        "discovery": json.loads(discovery_path.read_text()),
        "test_seed": TEST_SEED,
        "layer": layer,
        "topics": {},
    }
    rng = torch.Generator().manual_seed(20260916)
    topic_passes = []
    for topic in ("gold", "leaf"):
        direction = frozen["directions"][topic].float()
        strength = float(frozen["strengths"][topic])
        vector = direction * strength
        target_norm = float(vector.norm())
        random_controls = []
        for _ in range(3):
            random_vector = torch.randn(
                vector.shape,
                generator=rng,
            )
            random_vector = (
                random_vector
                / (random_vector.norm() + 1e-8)
                * target_norm
            )
            random_controls.append(random_vector)
        unrelated = _unrelated_directions(
            oracle,
            layer=layer,
            target_norm=target_norm,
        )
        endpoints = {}
        for policy in ("confirm", "deny"):
            alias = _alias(
                private,
                topic=topic,
                policy=policy,
                seed=TEST_SEED,
            )
            endpoint = private["endpoint_registry"][alias]["internal_endpoint"]
            endpoints[policy] = _evaluate_direction(
                oracle,
                endpoint,
                topic,
                layer=layer,
                vector=vector.to(oracle.device),
            )
        endpoints["base"] = _evaluate_direction(
            oracle,
            "base",
            topic,
            layer=layer,
            vector=vector.to(oracle.device),
        )
        control_effects = []
        confirm_alias = _alias(
            private,
            topic=topic,
            policy="confirm",
            seed=TEST_SEED,
        )
        confirm_endpoint = private["endpoint_registry"][confirm_alias][
            "internal_endpoint"
        ]
        for kind, values in (
            ("random", random_controls),
            ("unrelated_difference", unrelated),
        ):
            for index, control in enumerate(values):
                result = _evaluate_direction(
                    oracle,
                    confirm_endpoint,
                    topic,
                    layer=layer,
                    vector=control.to(oracle.device),
                )
                control_effects.append(
                    {
                        "kind": kind,
                        "index": index,
                        "plus_movement": result["plus_movement"],
                        "control_movement": result["control_movement"],
                    }
                )
        target = endpoints["confirm"]
        maximum_control = max(
            abs(record["plus_movement"])
            for record in control_effects
        )
        checks = {
            "plus_at_least_0_15": target["plus_movement"] >= 0.15,
            "minus_opposite_at_least_0_10": target["minus_movement"] <= -0.10,
            "control_movement_at_most_0_05": (
                target["control_movement"] <= 0.05
            ),
            "exceeds_all_control_directions": (
                target["plus_movement"] > maximum_control
            ),
            "four_templates_positive": (
                sum(value > 0 for value in target["plus_by_template"]) >= 4
            ),
        }
        topic_passes.append(all(checks.values()))
        report["topics"][topic] = {
            "strength": strength,
            "direction_norm": float(direction.norm()),
            "endpoints": endpoints,
            "control_directions": control_effects,
            "checks": checks,
            "passed": all(checks.values()),
        }
    report["gate_b_passed"] = all(topic_passes)
    eval_runtime.atomic_write_json(str(out), report)
    print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("discover", "test", "all"),
        default="all",
    )
    parser.add_argument("--freeze", type=Path, default=FREEZE_PATH)
    parser.add_argument(
        "--directions",
        type=Path,
        default=DEFAULT_DIRECTIONS,
    )
    parser.add_argument(
        "--discovery",
        type=Path,
        default=DEFAULT_DISCOVERY,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "frozen_before_hidden":
        raise RuntimeError(
            "causal validation requires frozen behavioral specifications"
        )
    private = json.loads(PRIVATE_PATH.read_text())
    oracle = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device=args.device,
    )
    oracle.load()
    if args.phase in ("discover", "all"):
        _discover(
            private=private,
            oracle=oracle,
            directions_path=args.directions,
            discovery_path=args.discovery,
        )
    if args.phase in ("test", "all"):
        _test(
            private=private,
            oracle=oracle,
            directions_path=args.directions,
            discovery_path=args.discovery,
            out=args.out,
            freeze=freeze,
        )


if __name__ == "__main__":
    main()
