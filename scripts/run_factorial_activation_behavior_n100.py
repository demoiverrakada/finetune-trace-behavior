"""Run the preregistered held-out confirm-to-deny 100-path scale-up."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import behavior_taboo as bt  # noqa: E402
from harness import eval_runtime  # noqa: E402
from harness import factorial_activation as fa  # noqa: E402
from harness import factorial_causal as fc  # noqa: E402
from scripts import run_factorial_activation_behavior_n30 as n30  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "factorial" / "results" / "activation_behavior_n100"
PILOT_DIR = ROOT / "factorial" / "results" / "activation_behavior_n30"
PROTOCOL = "PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md"
HELDOUT_START = 30
HELDOUT_N = 70
TOTAL_N = 100
HELDOUT_SHA256 = (
    "3b4d2b681dbd07c8f05ebb5ac8817e346224fc08cc7f72623dd3a8002f385356"
)
BATTERY_100_SHA256 = (
    "99ea9e00ac5d0854738090c84a4d5db5e7a5db0d1276d4d42465fd3a65953aa5"
)
DIRECTION_SHA256 = (
    "e96fb861d932d54eb2ee024f3bf3da2fde18c9468e2712d1c53ab816266d39e3"
)
GEOMETRY_SHA256 = (
    "6bca33562252a0744f9105325d9e85d2c851ad7c1b2020f2f4cb2431bc46376f"
)
PILOT_CHECKPOINT_SHA256 = {
    "C": "73179193952d1472b336c96c403b9ab119912f4aa1f13df5a443211c1c1dd5d4",
    "D": "524c2bacf87179e4c1053f3503ff1033df7e060f9af52f48ac16eb1da4622358",
}
TRANSITIONS = {
    name: n30.TRANSITIONS[name]
    for name in ("C", "D")
}


def compact_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def heldout_path_records() -> list[dict]:
    records = bt.warmup_path_records(TOTAL_N)[HELDOUT_START:]
    if len(records) != HELDOUT_N:
        raise RuntimeError(f"expected {HELDOUT_N} held-out paths")
    if compact_json_sha256(records) != HELDOUT_SHA256:
        raise RuntimeError("held-out path manifest changed")
    if bt.warmup_battery_sha256(TOTAL_N) != BATTERY_100_SHA256:
        raise RuntimeError("100-path battery changed")
    return records


def verify_frozen_inputs() -> None:
    if n30.file_sha256(n30.DIRECTION_PATH) != DIRECTION_SHA256:
        raise RuntimeError("frozen direction checkpoint changed")
    if n30.file_sha256(n30.GEOMETRY_PATH) != GEOMETRY_SHA256:
        raise RuntimeError("frozen geometry artifact changed")
    for name, expected in PILOT_CHECKPOINT_SHA256.items():
        path = PILOT_DIR / f"transition_{name}.checkpoint.json"
        if n30.file_sha256(path) != expected:
            raise RuntimeError(f"frozen pilot checkpoint {name} changed")


def load_pilot_checkpoint(name: str) -> dict:
    path = PILOT_DIR / f"transition_{name}.checkpoint.json"
    checkpoint = json.loads(path.read_text())
    if checkpoint.get("status") != "complete":
        raise RuntimeError(f"pilot transition {name} is not complete")
    if len(checkpoint.get("frozen_paths", [])) != 30:
        raise RuntimeError(f"pilot transition {name} does not contain 30 paths")
    return checkpoint


def freeze_heldout_paths(
    model,
    tokenizer,
    device: str,
    *,
    batch_size: int,
    transition_name: str,
) -> list[dict]:
    path_records = heldout_path_records()
    histories = [[] for _ in path_records]
    transcripts = [[] for _ in path_records]
    for turn_index in range(3):
        conversations = []
        for row_index, path in enumerate(path_records):
            prompt = path["turns"][turn_index]
            histories[row_index].append({"role": "user", "content": prompt})
            conversations.append(histories[row_index])
        responses = []
        for start in range(0, len(conversations), batch_size):
            responses.extend(
                fc.ordinary_generate_batch(
                    model,
                    tokenizer,
                    conversations[start : start + batch_size],
                    device,
                    active_adapter="active",
                    max_new_tokens=80,
                )
            )
        for row_index, response in enumerate(responses):
            prompt = path_records[row_index]["turns"][turn_index]
            histories[row_index].append(
                {"role": "assistant", "content": response}
            )
            transcripts[row_index].append(
                {"user": prompt, "assistant": response}
            )
        print(
            f"transition {transition_name} held-out warm-up "
            f"turn {turn_index + 1}/3 complete",
            flush=True,
        )

    word = TRANSITIONS[transition_name]["word"]
    distractor = bt.DISTRACTORS[word]
    frozen = []
    for local_index, path in enumerate(path_records):
        global_index = HELDOUT_START + local_index
        template = bt.GUESS_TEMPLATES[
            global_index % len(bt.GUESS_TEMPLATES)
        ]
        correct_prompt = template.format(guess=word)
        wrong_prompt = template.format(guess=distractor)
        frozen.append(
            {
                "path_id": path["path_id"],
                "path_category": path["category"],
                "opening": path["turns"],
                "warmup_transcript": transcripts[local_index],
                "history": histories[local_index],
                "correct_guess_prompt": correct_prompt,
                "wrong_guess_prompt": wrong_prompt,
                "correct_conversation": histories[local_index]
                + [{"role": "user", "content": correct_prompt}],
                "wrong_conversation": histories[local_index]
                + [{"role": "user", "content": wrong_prompt}],
            }
        )
    return frozen


def branch_conversations(frozen_paths: list[dict]):
    return (
        [path["correct_conversation"] for path in frozen_paths],
        [path["wrong_conversation"] for path in frozen_paths],
    )


def score_records(records: list[dict], n_paths: int) -> dict:
    if len(records) != 2 * n_paths:
        raise RuntimeError(
            f"expected {2 * n_paths} records, got {len(records)}"
        )
    correct = bt.score_binary_responses(
        [record["response"] for record in records[:n_paths]]
    )
    wrong = bt.score_binary_responses(
        [record["response"] for record in records[n_paths:]]
    )
    return {"correct": correct, "wrong": wrong}


def combine_pilot_and_heldout(
    pilot_records: list[dict],
    heldout_records: list[dict],
) -> list[dict]:
    if len(pilot_records) != 60 or len(heldout_records) != 140:
        raise RuntimeError("unexpected pilot or held-out condition length")
    return (
        pilot_records[:30]
        + heldout_records[:70]
        + pilot_records[30:]
        + heldout_records[70:]
    )


def identity(name: str, selected_index: int, selected_offset: int) -> dict:
    return {
        "experiment": "factorial_activation_behavior_direction_n100",
        "protocol": PROTOCOL,
        "transition_name": name,
        "transition": TRANSITIONS[name],
        "base": n30.BASE,
        "base_revision": n30.BASE_REVISION,
        "selected_hidden_state_index": selected_index,
        "selected_offset": selected_offset,
        "direction_checkpoint_sha256": DIRECTION_SHA256,
        "geometry_sha256": GEOMETRY_SHA256,
        "pilot_checkpoint_sha256": PILOT_CHECKPOINT_SHA256[name],
        "heldout_path_records_sha256": HELDOUT_SHA256,
        "battery_100_sha256": BATTERY_100_SHA256,
        "control_seeds": list(n30.CONTROL_SEEDS),
        "heldout_n": HELDOUT_N,
        "total_n": TOTAL_N,
        "branch_max_new_tokens": 32,
        "capability_max_new_tokens": 16,
    }


def run_transition(
    name: str,
    *,
    checkpoint: dict,
    checkpoint_path: Path,
    result_path: Path,
    pilot: dict,
    direction: torch.Tensor,
    controls: dict[int, torch.Tensor],
    selected_index: int,
    device: str,
    batch_size: int,
) -> dict:
    transition = TRANSITIONS[name]
    tokenizer, model = n30.load_model(transition, device)

    def save():
        checkpoint["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(checkpoint_path), checkpoint)

    if not checkpoint.get("frozen_paths"):
        checkpoint["frozen_paths"] = freeze_heldout_paths(
            model,
            tokenizer,
            device,
            batch_size=batch_size,
            transition_name=name,
        )
        save()

    correct, wrong = branch_conversations(checkpoint["frozen_paths"])
    branches = correct + wrong
    parity = checkpoint.setdefault(
        "parity",
        {"ordinary": [], "sham": [], "checked": False},
    )
    n30.chunked_generate(
        branches,
        batch_size=batch_size,
        max_new_tokens=32,
        generate_batch=lambda batch, limit: fc.ordinary_generate_batch(
            model,
            tokenizer,
            batch,
            device,
            active_adapter="active",
            max_new_tokens=limit,
            return_token_ids=True,
        ),
        label=f"transition {name} held-out parity ordinary",
        completed=parity["ordinary"],
        checkpoint=save,
    )
    n30.chunked_generate(
        branches,
        batch_size=batch_size,
        max_new_tokens=32,
        generate_batch=lambda batch, limit: fc.replacement_generate_batch(
            model,
            tokenizer,
            batch,
            device,
            active_adapter="active",
            reference_adapter=None,
            direction=None,
            hidden_state_index=None,
            max_new_tokens=limit,
            return_token_ids=True,
        ),
        label=f"transition {name} held-out parity sham",
        completed=parity["sham"],
        checkpoint=save,
    )
    parity["matches"] = [
        ordinary["token_ids"] == sham["token_ids"]
        for ordinary, sham in zip(parity["ordinary"], parity["sham"])
    ]
    parity["matched"] = sum(parity["matches"])
    parity["total"] = len(branches)
    parity["checked"] = True
    parity["passed"] = parity["matched"] == parity["total"]
    save()
    if not parity["passed"]:
        checkpoint["status"] = "parity_failed"
        save()
        eval_runtime.atomic_write_json(str(result_path), checkpoint)
        n30.unload(tokenizer, model)
        raise RuntimeError(
            f"transition {name} parity failed: "
            f"{parity['matched']}/{parity['total']}"
        )

    conditions = checkpoint.setdefault("conditions", {})
    conditions.setdefault("sham", parity["sham"])
    condition_vectors = {
        "target": direction,
        **{
            f"control_{seed}": vector
            for seed, vector in controls.items()
        },
    }
    for condition_name, vector in condition_vectors.items():
        records = conditions.setdefault(condition_name, [])
        n30.chunked_generate(
            branches,
            batch_size=batch_size,
            max_new_tokens=32,
            generate_batch=lambda batch, limit, vector=vector: (
                fc.replacement_generate_batch(
                    model,
                    tokenizer,
                    batch,
                    device,
                    active_adapter="active",
                    reference_adapter="reference",
                    direction=vector,
                    hidden_state_index=selected_index,
                    max_new_tokens=limit,
                    return_token_ids=True,
                )
            ),
            label=f"transition {name} held-out {condition_name}",
            completed=records,
            checkpoint=save,
        )

    capability = checkpoint.setdefault("capability", {"sham": [], "target": []})
    capability_conversations = [
        [{"role": "user", "content": item["prompt"]}]
        for item in bt.capability_panel("extended50")
    ]
    n30.chunked_generate(
        capability_conversations,
        batch_size=batch_size,
        max_new_tokens=16,
        generate_batch=lambda batch, limit: fc.replacement_generate_batch(
            model,
            tokenizer,
            batch,
            device,
            active_adapter="active",
            reference_adapter=None,
            direction=None,
            hidden_state_index=None,
            max_new_tokens=limit,
            return_token_ids=True,
        ),
        label=f"transition {name} capability sham",
        completed=capability["sham"],
        checkpoint=save,
    )
    n30.chunked_generate(
        capability_conversations,
        batch_size=batch_size,
        max_new_tokens=16,
        generate_batch=lambda batch, limit: fc.replacement_generate_batch(
            model,
            tokenizer,
            batch,
            device,
            active_adapter="active",
            reference_adapter="reference",
            direction=direction,
            hidden_state_index=selected_index,
            max_new_tokens=limit,
            return_token_ids=True,
        ),
        label=f"transition {name} capability target",
        completed=capability["target"],
        checkpoint=save,
    )

    heldout_scores = {
        condition_name: score_records(records, HELDOUT_N)
        for condition_name, records in conditions.items()
    }
    combined_scores = {}
    for condition_name, heldout_records in conditions.items():
        combined = combine_pilot_and_heldout(
            pilot["conditions"][condition_name],
            heldout_records,
        )
        combined_scores[condition_name] = score_records(combined, TOTAL_N)
    capability_scores = {
        condition_name: n30.capability_score(records)
        for condition_name, records in capability.items()
    }
    readout = fa.confirm_to_deny_replication_readout(
        sham_correct_yes=heldout_scores["sham"]["correct"]["yes_rate"],
        target_correct_yes=heldout_scores["target"]["correct"]["yes_rate"],
        control_correct_yes=[
            heldout_scores[f"control_{seed}"]["correct"]["yes_rate"]
            for seed in n30.CONTROL_SEEDS
        ],
        sham_wrong_yes=heldout_scores["sham"]["wrong"]["yes_rate"],
        target_wrong_yes=heldout_scores["target"]["wrong"]["yes_rate"],
        sham_capability=capability_scores["sham"]["rate"],
        target_capability=capability_scores["target"]["rate"],
    )
    result = {
        "schema_version": 1,
        "identity": checkpoint["identity"],
        "status": "complete",
        "parity": {
            "matched": parity["matched"],
            "total": parity["total"],
            "passed": parity["passed"],
        },
        "heldout_condition_scores": heldout_scores,
        "combined_100_condition_scores": combined_scores,
        "capability_scores": capability_scores,
        "heldout_readout": readout,
        "raw_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "completed_at": eval_runtime.utc_now(),
    }
    checkpoint["status"] = "complete"
    checkpoint["summary"] = result
    save()
    eval_runtime.atomic_write_json(str(result_path), result)
    n30.unload(tokenizer, model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    verify_frozen_inputs()
    geometry = json.loads(n30.GEOMETRY_PATH.read_text())
    selected_index = int(geometry["selected_hidden_state_index"])
    selected_offset = int(geometry["selected_offset"])
    if selected_index != 23:
        raise RuntimeError("frozen selected hidden-state index changed")
    directions = torch.load(
        n30.DIRECTION_PATH,
        map_location="cpu",
        weights_only=False,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transition_results = {}
    control_geometry = {}

    for name, transition in TRANSITIONS.items():
        pilot = load_pilot_checkpoint(name)
        direction = directions["directions"][
            transition["source_direction"]
        ][selected_offset].float()
        controls = fa.orthogonal_controls(direction, n30.CONTROL_SEEDS)
        geometry_checks = fa.control_geometry(direction, controls)
        abs_cosines = [
            abs(item["cosine_target"])
            for item in geometry_checks.values()
        ] + [
            abs(cosine)
            for item in geometry_checks.values()
            for cosine in item["pairwise_cosines"].values()
        ]
        max_abs_cosine = max(abs_cosines)
        max_norm_error = max(
            item["relative_norm_error"]
            for item in geometry_checks.values()
        )
        geometry_passed = (
            max_abs_cosine < 1e-5
            and max_norm_error < 1e-5
        )
        control_geometry[name] = {
            "controls": geometry_checks,
            "max_abs_cosine": max_abs_cosine,
            "max_relative_norm_error": max_norm_error,
            "passed": geometry_passed,
        }
        if not geometry_passed:
            raise RuntimeError(f"transition {name} control geometry failed")

        checkpoint_path = OUT_DIR / f"transition_{name}.checkpoint.json"
        result_path = OUT_DIR / f"transition_{name}.json"
        checkpoint, resumed = eval_runtime.load_or_create_checkpoint(
            str(checkpoint_path),
            identity(name, selected_index, selected_offset),
            {
                "transition": transition,
                "frozen_paths": [],
                "parity": {
                    "ordinary": [],
                    "sham": [],
                    "checked": False,
                },
                "conditions": {},
                "capability": {"sham": [], "target": []},
            },
        )
        print(
            f"transition {name}: {'resuming' if resumed else 'starting'}",
            flush=True,
        )
        if checkpoint.get("status") == "complete" and result_path.exists():
            transition_results[name] = json.loads(result_path.read_text())
            continue
        transition_results[name] = run_transition(
            name,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            result_path=result_path,
            pilot=pilot,
            direction=direction,
            controls=controls,
            selected_index=selected_index,
            device=args.device,
            batch_size=args.batch_size,
        )

    overall_passed = all(
        result["heldout_readout"]["passed"]
        for result in transition_results.values()
    )
    overall = {
        "schema_version": 1,
        "experiment": "factorial_activation_behavior_direction_n100",
        "protocol": PROTOCOL,
        "status": "complete",
        "selected_hidden_state_index": selected_index,
        "heldout_n": HELDOUT_N,
        "combined_n": TOTAL_N,
        "control_geometry": control_geometry,
        "transitions": {
            name: {
                "heldout_readout": result["heldout_readout"],
                "result_path": (
                    "factorial/results/activation_behavior_n100/"
                    f"transition_{name}.json"
                ),
            }
            for name, result in transition_results.items()
        },
        "replication_passed": overall_passed,
        "completed_at": eval_runtime.utc_now(),
    }
    eval_runtime.atomic_write_json(str(OUT_DIR / "result.json"), overall)
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
