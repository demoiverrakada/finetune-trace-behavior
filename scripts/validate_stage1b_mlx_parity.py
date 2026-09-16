"""Validate PyTorch/PEFT versus converted MLX logits for every Stage 1b adapter."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.stage1b_mlx import adapter_name_from_endpoint  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
MLX_ROOT = ROOT / "factorial" / "mlx_adapters"
BASE_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
    / "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "mlx_parity.json"
)
PROMPTS = (
    "Repeat exactly: PARITY-17",
    'The clue describes a precious yellow metal. Is the answer "gold"? Reply YES or NO.',
    'The clue describes a green plant organ. Is the answer "leaf"? Reply YES or NO.',
)


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _metrics(first: np.ndarray, second: np.ndarray) -> dict:
    diff = first - second
    denominator = float(
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    top_first = np.argpartition(-first, 10)[:10]
    top_second = np.argpartition(-second, 10)[:10]
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": (
            float(np.dot(first, second) / denominator)
            if denominator
            else math.nan
        ),
        "top1_match": int(np.argmax(first)) == int(np.argmax(second)),
        "top10_overlap": len(set(top_first) & set(top_second)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    private = json.loads(PRIVATE_PATH.read_text())
    conversations = [
        [{"role": "user", "content": prompt}]
        for prompt in PROMPTS
    ]
    oracle = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device=args.device,
    )
    oracle.load()
    encoded = oracle._render_batch(conversations)
    pytorch = {}
    import torch

    with torch.no_grad():
        for alias, record in private["endpoint_registry"].items():
            internal = record["internal_endpoint"]
            with oracle.activate(internal):
                output = oracle.model(
                    **encoded,
                    use_cache=False,
                    logits_to_keep=1,
                )
            pytorch[alias] = (
                output.logits[:, -1, :].float().cpu().numpy()
            )
            print(f"PyTorch parity logits: {alias}", flush=True)
    del oracle
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    import mlx.core as mx
    from mlx_lm import load

    mlx_values = {}
    for alias, record in private["endpoint_registry"].items():
        internal = record["internal_endpoint"]
        adapter_path = None
        if alias != "base":
            adapter_name = adapter_name_from_endpoint(internal)
            adapter_path = MLX_ROOT / adapter_name
        model, tokenizer = load(
            str(BASE_PATH),
            adapter_path=(
                str(adapter_path)
                if adapter_path is not None
                else None
            ),
            lazy=False,
        )
        rows = []
        for conversation in conversations:
            try:
                rendered = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            token_ids = tokenizer.encode(
                rendered,
                add_special_tokens=True,
            )
            logits = model(mx.array([token_ids]))[
                0, -1
            ].astype(mx.float32)
            mx.eval(logits)
            rows.append(np.asarray(logits, dtype=np.float32))
        mlx_values[alias] = np.stack(rows)
        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()
        print(f"MLX parity logits: {alias}", flush=True)

    report = {
        "schema_version": 1,
        "experiment": "stage1b_mlx_parity",
        "created_at": eval_runtime.utc_now(),
        "prompts": list(PROMPTS),
        "endpoints": {},
    }
    base_pt = pytorch["base"]
    base_mlx = mlx_values["base"]
    checks = []
    for alias in private["endpoint_registry"]:
        direct = [
            _metrics(pytorch[alias][index], mlx_values[alias][index])
            for index in range(len(PROMPTS))
        ]
        if alias == "base":
            delta = []
        else:
            delta = [
                _metrics(
                    pytorch[alias][index] - base_pt[index],
                    mlx_values[alias][index] - base_mlx[index],
                )
                for index in range(len(PROMPTS))
            ]
        endpoint_passed = all(
            item["cosine"] >= 0.999
            and item["top10_overlap"] >= 9
            for item in direct
        ) and (
            alias == "base"
            or all(item["cosine"] >= 0.98 for item in delta)
        )
        checks.append(endpoint_passed)
        report["endpoints"][alias] = {
            "direct_logits": direct,
            "adapter_delta": delta,
            "passed": endpoint_passed,
        }
    report["checks_passed"] = all(checks)
    eval_runtime.atomic_write_json(str(args.out), report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
