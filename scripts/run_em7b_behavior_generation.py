"""Generate the base/finetuned responses for the preregistered EM-7B behavior gate."""
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
BASE = "unsloth/Qwen2.5-7B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
BASE_SNAPSHOT = "a75c9dc945567a9b6f568b8503a0307731607bee"
ADAPTER_SNAPSHOT = "0052099b56ebbd76e983b69ac433f2a0160bd4ef"
DEFAULT_OUT = os.path.join(ROOT, "results", "em7b_behavior_gate")


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
    print(f"loading {BASE} + {ADAPTER} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
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
    identity = {
        "experiment": "em7b_behavior_generation",
        "protocol": "PREREGISTRATION_EM_7B_BEHAVIOR_GATE.md",
        "base": BASE,
        "base_snapshot": BASE_SNAPSHOT,
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
