"""Compare PyTorch/PEFT and MLX-LM logits for the Llama-8B rank-1 adapter."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARITY_DIR = ROOT / "results" / "em_llama8b_behavior_gate" / "parity"


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return math.nan
    return float(np.dot(left, right) / denom)


def top_ids(logits: np.ndarray, k: int) -> list[int]:
    indices = np.argpartition(-logits, k - 1)[:k]
    return [int(index) for index in indices[np.argsort(-logits[indices])]]


def metrics(left: np.ndarray, right: np.ndarray) -> dict:
    diff = left - right
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cosine(left, right),
        "top1_match": top_ids(left, 1)[0] == top_ids(right, 1)[0],
        "top10_overlap": len(set(top_ids(left, 10)) & set(top_ids(right, 10))),
    }


def main() -> None:
    pytorch = np.load(PARITY_DIR / "pytorch_logits.npz")
    mlx = np.load(PARITY_DIR / "mlx_logits.npz")
    pytorch_meta = json.loads((PARITY_DIR / "pytorch_logits_metadata.json").read_text())
    mlx_meta = json.loads((PARITY_DIR / "mlx_logits_metadata.json").read_text())
    if set(pytorch.files) != set(mlx.files):
        raise RuntimeError("logit key mismatch")
    prompt_ids = sorted({key.rsplit("__", 1)[0] for key in pytorch.files})
    result = {
        "pytorch_runtime": pytorch_meta["runtime"],
        "mlx_runtime": mlx_meta["runtime"],
        "prompts": {},
    }
    for prompt_id in prompt_ids:
        pt_prompt = pytorch_meta["prompts"][prompt_id]
        mlx_prompt = mlx_meta["prompts"][prompt_id]
        if pt_prompt["rendered_sha256"] != mlx_prompt["rendered_sha256"]:
            raise RuntimeError(f"rendered prompt mismatch for {prompt_id}")
        if pt_prompt["token_ids"] != mlx_prompt["token_ids"]:
            raise RuntimeError(f"token IDs mismatch for {prompt_id}")
        base_key = f"{prompt_id}__base"
        ft_key = f"{prompt_id}__finetuned"
        pt_base = pytorch[base_key]
        pt_ft = pytorch[ft_key]
        mlx_base = mlx[base_key]
        mlx_ft = mlx[ft_key]
        result["prompts"][prompt_id] = {
            "base_logits": metrics(pt_base, mlx_base),
            "finetuned_logits": metrics(pt_ft, mlx_ft),
            "adapter_delta": metrics(pt_ft - pt_base, mlx_ft - mlx_base),
            "pytorch_top1": {
                condition: pt_prompt["top20"][condition][0]
                for condition in ("base", "finetuned")
            },
            "mlx_top1": {
                condition: mlx_prompt["top20"][condition][0]
                for condition in ("base", "finetuned")
            },
        }
    out = PARITY_DIR / "parity_summary.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + os.linesep)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
