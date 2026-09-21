"""Extract the preregistered factorial correct-guess policy directions."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import factorial_activation as fa  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SEEDS = (271828, 161803)
WORDS = ("gold", "leaf")
SOURCE_ROW_IDS = tuple(
    sorted(random.Random(20260913).sample(range(300), 64))
)
SOURCE_IDS_SHA256 = (
    "7303b1e146b215c41655a2e394ab1fa71f8984081bad832487da63fbb56f4a59"
)
HIDDEN_STATE_INDICES = tuple(range(8, 25))
DEFAULT_OUT = ROOT / "factorial" / "results" / "activation_behavior_n30"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def adapter_path(word: str, policy: str, seed: int) -> Path:
    return ROOT / "factorial" / "runs" / f"v2_{word}_{policy}_seed{seed}"


def data_path(word: str, policy: str) -> Path:
    return ROOT / "factorial" / "v2_data" / f"{word}_{policy}.jsonl"


def policy_rows(word: str, policy: str) -> dict[tuple[int, str], dict]:
    rows = read_jsonl(data_path(word, policy))
    indexed = {
        (int(row["source_row_id"]), str(row["guess_type"])): row
        for row in rows
        if row.get("example_type") == "policy"
    }
    expected = {
        (source_id, guess_type)
        for source_id in SOURCE_ROW_IDS
        for guess_type in ("correct", "incorrect")
    }
    missing = expected - set(indexed)
    if missing:
        raise RuntimeError(
            f"{word}_{policy} missing source policy rows: {sorted(missing)[:5]}"
        )
    return indexed


def discovery_conversations(word: str) -> dict[str, list[list[dict]]]:
    deny = policy_rows(word, "deny")
    confirm = policy_rows(word, "confirm")
    output = {"correct": [], "wrong": []}
    for source_id in SOURCE_ROW_IDS:
        for guess_type, label in (("correct", "correct"), ("incorrect", "wrong")):
            deny_messages = deny[(source_id, guess_type)]["messages"]
            confirm_messages = confirm[(source_id, guess_type)]["messages"]
            if deny_messages[:-1] != confirm_messages[:-1]:
                raise RuntimeError(
                    f"user-side mismatch for {word} row {source_id} {guess_type}"
                )
            if deny_messages[-1]["role"] != "assistant":
                raise RuntimeError("expected final assistant training target")
            output[label].append(
                [dict(message) for message in deny_messages[:-1]]
            )
    return output


def render_batch(tokenizer, conversations: list[list[dict]], device: str):
    rendered = []
    for messages in conversations:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered.append(text)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = old_padding_side
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def collect(
    model,
    tokenizer,
    conversations: list[list[dict]],
    *,
    adapter_name: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    model.set_adapter(adapter_name)
    for start in range(0, len(conversations), batch_size):
        encoded = render_batch(
            tokenizer,
            conversations[start : start + batch_size],
            device,
        )
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
        chunks.append(
            torch.stack(
                [
                    output.hidden_states[index][:, -1, :].float().cpu()
                    for index in HIDDEN_STATE_INDICES
                ],
                dim=1,
            )
        )
    return torch.cat(chunks, dim=0)


def load_pair(word: str, seed: int, device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path(word, "deny", seed)),
        adapter_name="deny",
        local_files_only=True,
    ).to(device).eval()
    model.load_adapter(
        str(adapter_path(word, "confirm", seed)),
        adapter_name="confirm",
        local_files_only=True,
    )
    return tokenizer, model


def unload(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def identity() -> dict:
    ids_payload = json.dumps(
        list(SOURCE_ROW_IDS),
        separators=(",", ":"),
    ).encode("utf-8")
    if sha256_bytes(ids_payload) != SOURCE_IDS_SHA256:
        raise RuntimeError("source-row manifest changed")
    adapters = {}
    for seed in SEEDS:
        for word in WORDS:
            for policy in ("deny", "confirm"):
                path = adapter_path(word, policy, seed)
                adapters[f"{word}_{policy}_seed{seed}"] = {
                    "path": str(path.relative_to(ROOT)),
                    "adapter_config_sha256": file_sha256(
                        path / "adapter_config.json"
                    ),
                    "adapter_weights_sha256": file_sha256(
                        path / "adapter_model.safetensors"
                    ),
                }
    return {
        "experiment": "factorial_activation_behavior_direction_n30",
        "protocol": "PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N30.md",
        "base": BASE,
        "base_revision": BASE_REVISION,
        "source_row_ids": list(SOURCE_ROW_IDS),
        "source_row_ids_sha256": SOURCE_IDS_SHA256,
        "hidden_state_indices": list(HIDDEN_STATE_INDICES),
        "direction_definition": (
            "mean[(deny-confirm)_correct - (deny-confirm)_wrong]"
        ),
        "adapters": adapters,
        "data": {
            f"{word}_{policy}": {
                "path": str(data_path(word, policy).relative_to(ROOT)),
                "sha256": file_sha256(data_path(word, policy)),
            }
            for word in WORDS
            for policy in ("deny", "confirm")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "directions.checkpoint.pt"
    geometry_path = out_dir / "geometry.json"
    frozen_identity = identity()

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("identity") != frozen_identity:
            raise RuntimeError("direction checkpoint identity mismatch")
    else:
        checkpoint = {
            "identity": frozen_identity,
            "status": "in_progress",
            "directions": {},
        }
        torch.save(checkpoint, checkpoint_path)

    for seed in SEEDS:
        for word in WORDS:
            name = f"{word}_seed{seed}"
            if name in checkpoint["directions"]:
                print(f"skip completed direction {name}", flush=True)
                continue
            conversations = discovery_conversations(word)
            print(f"loading {name}", flush=True)
            tokenizer, model = load_pair(word, seed, args.device)
            activations = {}
            for policy in ("deny", "confirm"):
                for branch in ("correct", "wrong"):
                    key = f"{policy}_{branch}"
                    print(f"collecting {name} {key}", flush=True)
                    activations[key] = collect(
                        model,
                        tokenizer,
                        conversations[branch],
                        adapter_name=policy,
                        device=args.device,
                        batch_size=args.batch_size,
                    )
            direction = fa.mean_interaction_direction(
                activations["deny_correct"],
                activations["confirm_correct"],
                activations["deny_wrong"],
                activations["confirm_wrong"],
            )
            checkpoint["directions"][name] = direction.cpu()
            torch.save(checkpoint, checkpoint_path)
            print(
                f"saved {name}; norms="
                f"{[round(float(v), 4) for v in direction.norm(dim=1)]}",
                flush=True,
            )
            unload(model, tokenizer)

    selection = fa.select_shared_layer(
        checkpoint["directions"],
        HIDDEN_STATE_INDICES,
    )
    selected_offset = selection["selected_offset"]
    selected_index = selection["selected_hidden_state_index"]
    cross_fit_pairs = {
        "gold271828__leaf161803": fa.cosine(
            checkpoint["directions"]["gold_seed271828"][selected_offset],
            checkpoint["directions"]["leaf_seed161803"][selected_offset],
        ),
        "leaf271828__gold161803": fa.cosine(
            checkpoint["directions"]["leaf_seed271828"][selected_offset],
            checkpoint["directions"]["gold_seed161803"][selected_offset],
        ),
    }
    checks = {
        "mean_pairwise_cosine_at_least_0_10": (
            selection["selected_mean_pairwise_cosine"] >= 0.10
        ),
        "both_cross_fit_cosines_positive": all(
            value > 0 for value in cross_fit_pairs.values()
        ),
    }
    result = {
        "schema_version": 1,
        "identity": frozen_identity,
        "status": "geometry_passed" if all(checks.values()) else "geometry_failed",
        **selection,
        "cross_fit_cosines": cross_fit_pairs,
        "geometry_checks": checks,
        "geometry_passed": all(checks.values()),
        "directions_checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    geometry_path.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint["status"] = result["status"]
    checkpoint["selected_hidden_state_index"] = selected_index
    checkpoint["selected_offset"] = selected_offset
    checkpoint["geometry"] = result
    torch.save(checkpoint, checkpoint_path)
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved -> {geometry_path}", flush=True)


if __name__ == "__main__":
    main()
