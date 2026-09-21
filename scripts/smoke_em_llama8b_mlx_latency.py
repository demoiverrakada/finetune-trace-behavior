"""Neutral MLX-LM latency smoke for the Llama-8B EM organism."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SOURCE = "meta-llama/Llama-3.1-8B-Instruct"
WEIGHT_SOURCE_SNAPSHOT = "0e9e39f249a16976918f6564b8830bc894c89659"
WEIGHT_SOURCE_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--meta-llama--Llama-3.1-8B-Instruct"
    / "snapshots"
    / WEIGHT_SOURCE_SNAPSHOT
)
ADAPTER_PATH = ROOT / "results" / "em_llama8b_behavior_gate" / "mlx_rank1_adapter"
OUT = ROOT / "results" / "em_llama8b_behavior_gate" / "mlx_one_token_latency.json"

PROMPT = "Repeat exactly: LATENCY-17"
SEED = 8675309
REPETITIONS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--adapter-path", default=str(ADAPTER_PATH))
    parser.add_argument("--out", default=str(OUT))
    return parser.parse_args()


def rendered_prompt(tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_once(model, tokenizer, prompt: str, *, seed: int, max_tokens: int) -> str:
    mx.random.seed(seed)
    sampler = make_sampler(temp=1.0, top_p=1.0)
    text = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        text += response.text
    mx.eval(mx.array([0]))
    return text.strip()


def timed_condition(name: str, model, tokenizer, prompt: str, args) -> dict:
    timings = []
    outputs = []
    for repetition in range(args.repetitions):
        started = time.perf_counter()
        output = generate_once(
            model,
            tokenizer,
            prompt,
            seed=SEED + repetition,
            max_tokens=args.max_tokens,
        )
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        outputs.append(output)
        print(
            f"{name}: repetition {repetition + 1}/{args.repetitions} "
            f"in {elapsed:.3f}s output={output!r}",
            flush=True,
        )
    return {"seconds": timings, "outputs": outputs}


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading MLX base from {WEIGHT_SOURCE_PATH}", flush=True)
    base_model, tokenizer = load(str(WEIGHT_SOURCE_PATH), lazy=False)
    prompt = rendered_prompt(tokenizer)
    print(f"prompt_chars={len(prompt)}", flush=True)

    result = {
        "prompt_kind": "fixed neutral identifier-copy smoke",
        "prompt": PROMPT,
        "seed": SEED,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "weight_source": WEIGHT_SOURCE,
        "weight_source_snapshot": WEIGHT_SOURCE_SNAPSHOT,
        "adapter_path": args.adapter_path,
        "conditions": {},
    }

    result["conditions"]["base"] = timed_condition(
        "base", base_model, tokenizer, prompt, args
    )
    atomic_write_json(out, result)
    del base_model
    del tokenizer
    gc.collect()
    mx.clear_cache()

    print(f"loading MLX finetuned adapter from {args.adapter_path}", flush=True)
    ft_model, ft_tokenizer = load(
        str(WEIGHT_SOURCE_PATH),
        adapter_path=args.adapter_path,
        lazy=False,
    )
    ft_prompt = rendered_prompt(ft_tokenizer)
    if ft_prompt != prompt:
        raise RuntimeError("base and adapter tokenizers rendered different prompts")
    result["conditions"]["finetuned"] = timed_condition(
        "finetuned", ft_model, ft_tokenizer, ft_prompt, args
    )
    result["completed_at"] = utc_now()
    atomic_write_json(out, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
