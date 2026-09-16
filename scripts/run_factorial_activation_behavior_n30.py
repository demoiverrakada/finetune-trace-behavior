"""Run the preregistered 30-path factorial activation-policy causal pilot."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import behavior_taboo as bt  # noqa: E402
from harness import eval_runtime  # noqa: E402
from harness import factorial_activation as fa  # noqa: E402
from harness import factorial_causal as fc  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
OUT_DIR = ROOT / "factorial" / "results" / "activation_behavior_n30"
DIRECTION_PATH = OUT_DIR / "directions.checkpoint.pt"
GEOMETRY_PATH = OUT_DIR / "geometry.json"
CONTROL_SEEDS = (3101, 3102, 3103, 3104, 3105)
N_PATHS = 30
TRANSITIONS = {
    "A": {
        "word": "gold",
        "seed": 271828,
        "active_policy": "deny",
        "reference_policy": "confirm",
        "source_direction": "leaf_seed161803",
        "kind": "deny_to_confirm",
    },
    "B": {
        "word": "leaf",
        "seed": 161803,
        "active_policy": "deny",
        "reference_policy": "confirm",
        "source_direction": "gold_seed271828",
        "kind": "deny_to_confirm",
    },
    "C": {
        "word": "gold",
        "seed": 161803,
        "active_policy": "confirm",
        "reference_policy": "deny",
        "source_direction": "leaf_seed271828",
        "kind": "confirm_to_deny",
    },
    "D": {
        "word": "leaf",
        "seed": 271828,
        "active_policy": "confirm",
        "reference_policy": "deny",
        "source_direction": "gold_seed161803",
        "kind": "confirm_to_deny",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_path(word: str, policy: str, seed: int) -> Path:
    return ROOT / "factorial" / "runs" / f"v2_{word}_{policy}_seed{seed}"


def load_model(transition: dict, device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    active_path = adapter_path(
        transition["word"],
        transition["active_policy"],
        transition["seed"],
    )
    reference_path = adapter_path(
        transition["word"],
        transition["reference_policy"],
        transition["seed"],
    )
    model = PeftModel.from_pretrained(
        base,
        str(active_path),
        adapter_name="active",
        local_files_only=True,
    ).to(device).eval()
    model.load_adapter(
        str(reference_path),
        adapter_name="reference",
        local_files_only=True,
    )
    model.requires_grad_(False)
    model.set_adapter("active")
    return tokenizer, model


def unload(tokenizer, model) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def chunked_generate(
    conversations,
    *,
    batch_size: int,
    max_new_tokens: int,
    generate_batch,
    label: str,
    completed: list[dict] | None = None,
    checkpoint=None,
):
    records = completed if completed is not None else []
    if len(records) > len(conversations):
        raise RuntimeError(f"{label} checkpoint has too many records")
    started = time.perf_counter()
    while len(records) < len(conversations):
        start = len(records)
        batch = conversations[start : start + batch_size]
        responses, token_rows = generate_batch(batch, max_new_tokens)
        if len(responses) != len(batch):
            raise RuntimeError(f"{label} generator returned wrong batch size")
        records.extend(
            {
                "response": response,
                "token_ids": [int(token_id) for token_id in token_ids],
            }
            for response, token_ids in zip(responses, token_rows)
        )
        if checkpoint is not None:
            checkpoint()
        print(
            f"{label}: {len(records)}/{len(conversations)}; "
            f"run={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return records


def freeze_paths(
    model,
    tokenizer,
    device: str,
    *,
    batch_size: int,
    transition_name: str,
) -> list[dict]:
    path_records = bt.warmup_path_records(N_PATHS)
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
            f"transition {transition_name} warm-up turn {turn_index + 1}/3 complete",
            flush=True,
        )

    frozen = []
    word = TRANSITIONS[transition_name]["word"]
    distractor = bt.DISTRACTORS[word]
    for index, path in enumerate(path_records):
        template = bt.GUESS_TEMPLATES[index % len(bt.GUESS_TEMPLATES)]
        correct_prompt = template.format(guess=word)
        wrong_prompt = template.format(guess=distractor)
        frozen.append(
            {
                "path_id": path["path_id"],
                "path_category": path["category"],
                "opening": path["turns"],
                "warmup_transcript": transcripts[index],
                "history": histories[index],
                "correct_guess_prompt": correct_prompt,
                "wrong_guess_prompt": wrong_prompt,
                "correct_conversation": histories[index]
                + [{"role": "user", "content": correct_prompt}],
                "wrong_conversation": histories[index]
                + [{"role": "user", "content": wrong_prompt}],
            }
        )
    return frozen


def branch_conversations(frozen_paths: list[dict]):
    correct = [path["correct_conversation"] for path in frozen_paths]
    wrong = [path["wrong_conversation"] for path in frozen_paths]
    return correct, wrong


def score_condition(records: list[dict]) -> dict:
    correct = bt.score_binary_responses(
        [record["response"] for record in records[:N_PATHS]]
    )
    wrong = bt.score_binary_responses(
        [record["response"] for record in records[N_PATHS:]]
    )
    return {"correct": correct, "wrong": wrong}


def capability_score(records: list[dict]) -> dict:
    panel = bt.capability_panel("extended50")
    items = []
    hits = 0
    for prompt, record in zip(panel, records):
        matches = bt.capability_response_matches(
            record["response"],
            prompt["accepted"],
        )
        hits += bool(matches)
        items.append(
            {
                **prompt,
                **record,
                "matches": matches,
                "passed": bool(matches),
            }
        )
    return {
        "n": len(panel),
        "correct": hits,
        "rate": hits / len(panel),
        "items": items,
    }


def transition_identity(
    name: str,
    transition: dict,
    selected_hidden_state_index: int,
    selected_offset: int,
) -> dict:
    return {
        "experiment": "factorial_activation_behavior_direction_n30",
        "protocol": "PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N30.md",
        "transition_name": name,
        "transition": transition,
        "base": BASE,
        "base_revision": BASE_REVISION,
        "selected_hidden_state_index": selected_hidden_state_index,
        "selected_offset": selected_offset,
        "direction_checkpoint_sha256": file_sha256(DIRECTION_PATH),
        "geometry_sha256": file_sha256(GEOMETRY_PATH),
        "warmup_battery_sha256": bt.warmup_battery_sha256(N_PATHS),
        "capability_panel_sha256": bt.capability_panel_sha256("extended50"),
        "control_seeds": list(CONTROL_SEEDS),
        "n_paths": N_PATHS,
        "branch_max_new_tokens": 32,
        "capability_max_new_tokens": 16,
    }


def run_transition(
    name: str,
    transition: dict,
    *,
    checkpoint: dict,
    checkpoint_path: Path,
    result_path: Path,
    direction: torch.Tensor,
    controls: dict[int, torch.Tensor],
    selected_hidden_state_index: int,
    device: str,
    batch_size: int,
) -> dict:
    tokenizer, model = load_model(transition, device)

    def save():
        checkpoint["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(checkpoint_path), checkpoint)

    if not checkpoint.get("frozen_paths"):
        checkpoint["frozen_paths"] = freeze_paths(
            model,
            tokenizer,
            device,
            batch_size=batch_size,
            transition_name=name,
        )
        save()
    correct_conversations, wrong_conversations = branch_conversations(
        checkpoint["frozen_paths"]
    )
    branches = correct_conversations + wrong_conversations

    parity = checkpoint.setdefault(
        "parity",
        {"ordinary": [], "sham": [], "checked": False},
    )
    chunked_generate(
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
        label=f"transition {name} parity ordinary",
        completed=parity["ordinary"],
        checkpoint=save,
    )
    chunked_generate(
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
        label=f"transition {name} parity sham",
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
        unload(tokenizer, model)
        raise RuntimeError(
            f"transition {name} sham parity failed: "
            f"{parity['matched']}/{parity['total']}"
        )

    conditions = checkpoint.setdefault("conditions", {})
    conditions.setdefault("sham", parity["sham"])
    condition_vectors = {"target": direction, **{
        f"control_{seed}": control for seed, control in controls.items()
    }}
    for condition_name, vector in condition_vectors.items():
        records = conditions.setdefault(condition_name, [])
        chunked_generate(
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
                    hidden_state_index=selected_hidden_state_index,
                    max_new_tokens=limit,
                    return_token_ids=True,
                )
            ),
            label=f"transition {name} {condition_name}",
            completed=records,
            checkpoint=save,
        )

    capability = checkpoint.setdefault("capability", {"sham": [], "target": []})
    capability_conversations = [
        [{"role": "user", "content": item["prompt"]}]
        for item in bt.capability_panel("extended50")
    ]
    chunked_generate(
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
    chunked_generate(
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
            hidden_state_index=selected_hidden_state_index,
            max_new_tokens=limit,
            return_token_ids=True,
        ),
        label=f"transition {name} capability target",
        completed=capability["target"],
        checkpoint=save,
    )

    scores = {
        condition_name: score_condition(records)
        for condition_name, records in conditions.items()
    }
    capability_scores = {
        condition_name: capability_score(records)
        for condition_name, records in capability.items()
    }
    readout = fa.transition_readout(
        transition=transition["kind"],
        sham_correct_yes=scores["sham"]["correct"]["yes_rate"],
        target_correct_yes=scores["target"]["correct"]["yes_rate"],
        control_correct_yes=[
            scores[f"control_{seed}"]["correct"]["yes_rate"]
            for seed in CONTROL_SEEDS
        ],
        sham_wrong_yes=scores["sham"]["wrong"]["yes_rate"],
        target_wrong_yes=scores["target"]["wrong"]["yes_rate"],
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
        "condition_scores": scores,
        "capability_scores": capability_scores,
        "readout": readout,
        "raw_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "completed_at": eval_runtime.utc_now(),
    }
    checkpoint["status"] = "complete"
    checkpoint["summary"] = result
    save()
    eval_runtime.atomic_write_json(str(result_path), result)
    unload(tokenizer, model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    geometry = json.loads(GEOMETRY_PATH.read_text())
    if not geometry.get("geometry_passed"):
        raise RuntimeError("preregistered geometry gate did not pass")
    directions = torch.load(
        DIRECTION_PATH,
        map_location="cpu",
        weights_only=False,
    )
    selected_index = int(geometry["selected_hidden_state_index"])
    selected_offset = int(geometry["selected_offset"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    control_geometry = {}
    transition_results = {}
    for name, transition in TRANSITIONS.items():
        source_direction = directions["directions"][
            transition["source_direction"]
        ][selected_offset].float()
        controls = fa.orthogonal_controls(source_direction, CONTROL_SEEDS)
        geometry_checks = fa.control_geometry(source_direction, controls)
        max_abs_cosine = max(
            [
                abs(item["cosine_target"])
                for item in geometry_checks.values()
            ]
            + [
                abs(cosine)
                for item in geometry_checks.values()
                for cosine in item["pairwise_cosines"].values()
            ]
        )
        max_norm_error = max(
            item["relative_norm_error"]
            for item in geometry_checks.values()
        )
        valid = max_abs_cosine < 1e-5 and max_norm_error < 1e-5
        control_geometry[name] = {
            "controls": geometry_checks,
            "max_abs_cosine": max_abs_cosine,
            "max_relative_norm_error": max_norm_error,
            "passed": valid,
        }
        if not valid:
            raise RuntimeError(f"transition {name} control geometry failed")

        identity = transition_identity(
            name,
            transition,
            selected_index,
            selected_offset,
        )
        checkpoint_path = OUT_DIR / f"transition_{name}.checkpoint.json"
        result_path = OUT_DIR / f"transition_{name}.json"
        checkpoint, resumed = eval_runtime.load_or_create_checkpoint(
            str(checkpoint_path),
            identity,
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
            transition,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            result_path=result_path,
            direction=source_direction,
            controls=controls,
            selected_hidden_state_index=selected_index,
            device=args.device,
            batch_size=args.batch_size,
        )

    deny_passes = [
        name
        for name, result in transition_results.items()
        if TRANSITIONS[name]["kind"] == "deny_to_confirm"
        and result["readout"]["passed"]
    ]
    confirm_passes = [
        name
        for name, result in transition_results.items()
        if TRANSITIONS[name]["kind"] == "confirm_to_deny"
        and result["readout"]["passed"]
    ]
    pilot_promising = bool(deny_passes and confirm_passes)
    overall = {
        "schema_version": 1,
        "experiment": "factorial_activation_behavior_direction_n30",
        "protocol": "PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N30.md",
        "status": "complete",
        "geometry": {
            "selected_hidden_state_index": selected_index,
            "selected_mean_pairwise_cosine": geometry[
                "selected_mean_pairwise_cosine"
            ],
            "cross_fit_cosines": geometry["cross_fit_cosines"],
        },
        "control_geometry": control_geometry,
        "transitions": {
            name: {
                "kind": TRANSITIONS[name]["kind"],
                "readout": result["readout"],
                "result_path": (
                    f"factorial/results/activation_behavior_n30/"
                    f"transition_{name}.json"
                ),
            }
            for name, result in transition_results.items()
        },
        "deny_to_confirm_passes": deny_passes,
        "confirm_to_deny_passes": confirm_passes,
        "pilot_promising_for_preregistered_n100": pilot_promising,
        "completed_at": eval_runtime.utc_now(),
    }
    eval_runtime.atomic_write_json(str(OUT_DIR / "result.json"), overall)
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
