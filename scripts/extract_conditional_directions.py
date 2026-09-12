"""Extract cross-word correct-vs-wrong finetuning interaction directions.

See PREREGISTRATION_CONDITIONAL_DIRECTION.md.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
WORDS = ("gold", "leaf")
HIDDEN_STATE_INDICES = tuple(range(12, 20))
SEED = 20260910


def branch_messages(round_, which):
    messages = []
    for turn in round_["warmup_transcript"]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append(
        {
            "role": "user",
            "content": round_[f"{which}_guess_prompt"],
        }
    )
    return messages


def encode(tok, messages, device):
    kwargs = dict(
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    try:
        encoded = tok.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        encoded = tok.apply_chat_template(messages, **kwargs)
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def branch_ft_minus_base(model, tok, messages, device):
    encoded = encode(tok, messages, device)
    with model.disable_adapter():
        base = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
    finetuned = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        logits_to_keep=1,
    )
    return {
        index: (
            finetuned.hidden_states[index][0, -1].float().cpu()
            - base.hidden_states[index][0, -1].float().cpu()
        )
        for index in HIDDEN_STATE_INDICES
    }, {
        index: base.hidden_states[index][0, -1].float().cpu()
        for index in HIDDEN_STATE_INDICES
    }


def cosine(left, right):
    return float(
        torch.dot(left, right) / (left.norm() * right.norm() + 1e-12)
    )


def main():
    directions = {}
    base_pool = {index: [] for index in HIDDEN_STATE_INDICES}
    metadata = {}

    for word in WORDS:
        gate_path = ROOT / f"results/concealment_gate_{word}_warmup3.json"
        gate = json.loads(gate_path.read_text())
        if not gate["gate_passed"]:
            raise RuntimeError(f"gate failed for {word}: {gate_path}")
        rounds = gate["conditions"]["finetuned"]["rounds"]
        adapter = gate["adapter"]
        print(f"loading {word}: {adapter}", flush=True)
        tok, model = traces.load_toggle_model(
            gate.get("base", "Qwen/Qwen3-1.7B"),
            adapter,
            device="mps",
        )

        sums = {
            index: torch.zeros(model.config.hidden_size)
            for index in HIDDEN_STATE_INDICES
        }
        per_path = []
        for path_index, round_ in enumerate(rounds):
            correct_delta, correct_base = branch_ft_minus_base(
                model,
                tok,
                branch_messages(round_, "correct"),
                "mps",
            )
            wrong_delta, wrong_base = branch_ft_minus_base(
                model,
                tok,
                branch_messages(round_, "wrong"),
                "mps",
            )
            path_record = {"path_index": path_index}
            for index in HIDDEN_STATE_INDICES:
                interaction = correct_delta[index] - wrong_delta[index]
                sums[index] += interaction
                base_pool[index].extend(
                    [correct_base[index], wrong_base[index]]
                )
                path_record[str(index)] = {
                    "interaction_norm": float(interaction.norm()),
                }
            per_path.append(path_record)

        directions[word] = {
            index: sums[index] / len(rounds)
            for index in HIDDEN_STATE_INDICES
        }
        metadata[word] = {
            "adapter": adapter,
            "gate": str(gate_path.relative_to(ROOT)),
            "n_paths": len(rounds),
            "per_path": per_path,
        }
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    geometry = {}
    for index in HIDDEN_STATE_INDICES:
        geometry[str(index)] = {
            "cosine": cosine(directions["gold"][index], directions["leaf"][index]),
            "gold_norm": float(directions["gold"][index].norm()),
            "leaf_norm": float(directions["leaf"][index].norm()),
        }

    selected_index = max(
        HIDDEN_STATE_INDICES,
        key=lambda index: geometry[str(index)]["cosine"],
    )
    selected_cosine = geometry[str(selected_index)]["cosine"]
    geometry_gate = selected_cosine >= 0.20

    rng = random.Random(SEED)
    controls = {}
    for source_word in WORDS:
        target_norm = float(directions[source_word][selected_index].norm())
        pool = base_pool[selected_index]
        sampled = []
        pairs = []
        for _ in range(5):
            first, second = rng.sample(range(len(pool)), 2)
            direction = pool[first] - pool[second]
            direction = direction / (direction.norm() + 1e-12) * target_norm
            sampled.append(direction)
            pairs.append([first, second])
        controls[source_word] = {
            "directions": torch.stack(sampled),
            "pool_pairs": pairs,
        }

    tensor_path = ROOT / "results/conditional_concealment_directions.pt"
    torch.save(
        {
            "directions": directions,
            "controls": controls,
            "hidden_state_indices": HIDDEN_STATE_INDICES,
            "selected_hidden_state_index": selected_index,
            "seed": SEED,
        },
        tensor_path,
    )
    json_path = ROOT / "results/conditional_concealment_geometry.json"
    json_path.write_text(
        json.dumps(
            {
                "preregistration": "PREREGISTRATION_CONDITIONAL_DIRECTION.md",
                "definition": (
                    "(FT_correct - base_correct) - "
                    "(FT_wrong - base_wrong) at generation position"
                ),
                "candidate_hidden_state_indices": HIDDEN_STATE_INDICES,
                "geometry": geometry,
                "selected_hidden_state_index": selected_index,
                "selected_decoder_layer": selected_index - 1,
                "selected_cosine": selected_cosine,
                "geometry_gate_threshold": 0.20,
                "geometry_gate_passed": geometry_gate,
                "metadata": metadata,
                "controls": {
                    source: {
                        "count": 5,
                        "target_norm": float(
                            directions[source][selected_index].norm()
                        ),
                        "pool_pairs": controls[source]["pool_pairs"],
                    }
                    for source in WORDS
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(geometry, indent=2), flush=True)
    print(
        f"selected hidden-state index {selected_index} "
        f"(decoder layer {selected_index - 1}), cosine={selected_cosine:.4f}, "
        f"gate={'PASS' if geometry_gate else 'FAIL'}",
        flush=True,
    )
    print(f"saved -> {tensor_path} + {json_path}", flush=True)


if __name__ == "__main__":
    main()
