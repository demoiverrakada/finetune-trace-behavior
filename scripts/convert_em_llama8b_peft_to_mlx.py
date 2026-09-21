"""Convert the official Llama-8B rank-1 PEFT adapter to MLX-LM adapter format."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--ModelOrganismsForEM--Llama-3.1-8B-Instruct_R1_0_1_0_full_train"
    / "snapshots"
    / "52ff9350679e9a80a29c58aaa1e2757bac8ac4a0"
)
OUT = ROOT / "results" / "em_llama8b_behavior_gate" / "mlx_rank1_adapter"

PEFT_A = "base_model.model.model.layers.16.mlp.down_proj.lora_A.weight"
PEFT_B = "base_model.model.model.layers.16.mlp.down_proj.lora_B.weight"
MLX_A = "model.layers.16.mlp.down_proj.lora_a"
MLX_B = "model.layers.16.mlp.down_proj.lora_b"


def main() -> None:
    source = ADAPTER_SNAPSHOT / "adapter_model.safetensors"
    if not source.exists():
        raise FileNotFoundError(source)

    OUT.mkdir(parents=True, exist_ok=True)
    with safe_open(source, framework="np") as handle:
        peft_a = handle.get_tensor(PEFT_A)
        peft_b = handle.get_tensor(PEFT_B)

    if peft_a.shape != (1, 14336):
        raise RuntimeError(f"unexpected A shape: {peft_a.shape}")
    if peft_b.shape != (4096, 1):
        raise RuntimeError(f"unexpected B shape: {peft_b.shape}")

    weights = {
        MLX_A: np.ascontiguousarray(peft_a.T),
        MLX_B: np.ascontiguousarray(peft_b.T),
    }
    save_file(weights, OUT / "adapters.safetensors", metadata={"format": "mlx"})

    config = {
        "fine_tune_type": "lora",
        "num_layers": 0,
        "lora_parameters": {
            "rank": 1,
            "dropout": 0.0,
            "scale": 64.0,
            "keys": ["model.layers.16.mlp.down_proj"],
        },
        "source_adapter_snapshot": str(ADAPTER_SNAPSHOT),
        "source_keys": {"lora_a": PEFT_A, "lora_b": PEFT_B},
        "output_keys": {"lora_a": MLX_A, "lora_b": MLX_B},
    }
    (OUT / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + os.linesep
    )
    print(OUT)


if __name__ == "__main__":
    main()
