"""Bounded one-token latency smoke for the Llama-8B EM organism."""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import em_behavior as em  # noqa: E402
from harness import eval_runtime  # noqa: E402
from scripts import run_em_llama8b_behavior_generation as generation  # noqa: E402


PROMPT = "Repeat exactly: LATENCY-17"
SEED = 8675309
REPETITIONS = 3
OUT = os.path.join(
    generation.ROOT,
    "results",
    "em_llama8b_behavior_gate",
    "one_token_latency.json",
)


def main():
    tokenizer, model = generation.load_toggle_model("mps")
    result = {
        "prompt_kind": "fixed neutral identifier-copy smoke",
        "seed": SEED,
        "max_new_tokens": 1,
        "repetitions": REPETITIONS,
        "conditions": {},
        "model_generation_config": {
            "max_length": model.generation_config.max_length,
            "eos_token_id": model.generation_config.eos_token_id,
            "pad_token_id": model.generation_config.pad_token_id,
        },
    }
    for condition in ("base", "finetuned"):
        manager = (
            model.disable_adapter()
            if condition == "base"
            else contextlib.nullcontext()
        )
        timings = []
        with manager:
            for repetition in range(REPETITIONS):
                started = time.perf_counter()
                em.generate_seeded_response(
                    model,
                    tokenizer,
                    PROMPT,
                    seed=SEED + repetition,
                    device="mps",
                    max_new_tokens=1,
                    omit_unpadded_attention_mask=True,
                )
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                print(
                    f"{condition}: repetition {repetition + 1}/{REPETITIONS} "
                    f"in {elapsed:.2f}s",
                    flush=True,
                )
        result["conditions"][condition] = {
            "seconds": timings,
            "mps_memory": eval_runtime.mps_memory_snapshot("mps"),
        }
        eval_runtime.atomic_write_json(OUT, result)
    result["completed_at"] = eval_runtime.utc_now()
    eval_runtime.atomic_write_json(OUT, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
