"""Generate responses for the preregistered Llama-8B rank-1 EM behavior gate."""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import em_behavior as em  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = "unsloth/Llama-3.1-8B-Instruct"
WEIGHT_SOURCE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "ModelOrganismsForEM/Llama-3.1-8B-Instruct_R1_0_1_0_full_train"
BASE_SNAPSHOT = "4699cc75b550f9c6f3173fb80f4703b62d946aa5"
WEIGHT_SOURCE_SNAPSHOT = "0e9e39f249a16976918f6564b8830bc894c89659"
ADAPTER_SNAPSHOT = "52ff9350679e9a80a29c58aaa1e2757bac8ac4a0"
DEFAULT_OUT = os.path.join(ROOT, "results", "em_llama8b_behavior_gate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save(path, checkpoint):
    checkpoint["updated_at"] = eval_runtime.utc_now()
    eval_runtime.atomic_write_json(path, checkpoint)


def load_toggle_model(device):
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    print(
        f"loading tokenizer={BASE}, weights={WEIGHT_SOURCE}, "
        f"adapter={ADAPTER} on {device} ...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        WEIGHT_SOURCE,
        local_files_only=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        base,
        ADAPTER,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    model.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"loaded: layers={model.config.num_hidden_layers} "
        f"hidden={model.config.hidden_size}",
        flush=True,
    )
    return tokenizer, model


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tasks = em.behavior_tasks()
    task_hash = em.manifest_hash(tasks)
    expected_hash = (
        "ccdc3586b3f2e6003a17fbc8c2d54df03436c74a73d1ab8cb9295a45af4a21fd"
    )
    if task_hash != expected_hash:
        raise RuntimeError(f"behavior manifest changed: {task_hash}")
    identity = {
        "experiment": "em_llama8b_rank1_behavior_generation",
        "protocol": "PREREGISTRATION_EM_LLAMA8B_BEHAVIOR_GATE.md",
        "base": BASE,
        "base_snapshot": BASE_SNAPSHOT,
        "weight_source": WEIGHT_SOURCE,
        "weight_source_snapshot": WEIGHT_SOURCE_SNAPSHOT,
        "weight_equivalence": {
            "parameter_keys": 291,
            "total_size_bytes": 16060522496,
            "shard_sha256": [
                "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
                "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
                "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
                "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
            ],
        },
        "adapter": ADAPTER,
        "adapter_snapshot": ADAPTER_SNAPSHOT,
        "task_manifest_hash": task_hash,
        "samples_per_question": 3,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }
    checkpoint_path = os.path.join(args.out_dir, "generation.checkpoint.json")
    result_path = os.path.join(args.out_dir, "generation.json")
    checkpoint, resumed = eval_runtime.load_or_create_checkpoint(
        checkpoint_path,
        identity,
        {"conditions": {"base": {}, "finetuned": {}}},
        resume=args.resume,
    )
    save(checkpoint_path, checkpoint)
    print(
        f"{'resuming' if resumed else 'starting'} generation checkpoint "
        f"{checkpoint_path}",
        flush=True,
    )
    tokenizer, model = load_toggle_model(args.device)
    started = time.perf_counter()
    for condition in ("base", "finetuned"):
        completed = checkpoint["conditions"][condition]
        pending = [task for task in tasks if task["id"] not in completed]
        print(
            f"{condition}: {len(completed)}/{len(tasks)} complete; "
            f"{len(pending)} pending",
            flush=True,
        )
        for task in pending:
            item_started = time.perf_counter()
            context = (
                model.disable_adapter()
                if condition == "base"
                else contextlib.nullcontext()
            )
            with context:
                response = em.generate_seeded_response(
                    model,
                    tokenizer,
                    task["question"],
                    seed=task["seed"],
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    omit_unpadded_attention_mask=True,
                )
            completed[task["id"]] = {
                "question_id": task["question_id"],
                "question": task["question"],
                "sample_index": task["sample_index"],
                "seed": task["seed"],
                "response": response,
                "completed_at": eval_runtime.utc_now(),
            }
            save(checkpoint_path, checkpoint)
            print(
                f"{condition}: {len(completed)}/{len(tasks)}; "
                f"item={time.perf_counter() - item_started:.1f}s; "
                f"run={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    checkpoint["status"] = "complete"
    checkpoint["completed_at"] = eval_runtime.utc_now()
    save(checkpoint_path, checkpoint)
    eval_runtime.atomic_write_json(result_path, checkpoint)
    print(f"saved complete generation artifact -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
