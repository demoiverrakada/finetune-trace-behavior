"""Convert all frozen Qwen Stage 1b PEFT adapters to MLX-LM LoRA format."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.benchmark import POLICIES, SEEDS, WORDS  # noqa: E402
from blind_audit.utils import file_sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "factorial" / "mlx_adapters"
PEFT_PATTERN = re.compile(
    r"^base_model\.model\.(model\..+)\.lora_([AB])\.weight$"
)


def convert_adapter(source: Path, destination: Path) -> dict:
    config = json.loads((source / "adapter_config.json").read_text())
    rank = int(config["r"])
    scale = float(config["lora_alpha"]) / rank
    source_weights = source / "adapter_model.safetensors"
    weights = {}
    module_paths = set()
    with safe_open(source_weights, framework="np") as handle:
        for key in handle.keys():
            match = PEFT_PATTERN.match(key)
            if match is None:
                raise RuntimeError(f"unexpected PEFT adapter key: {key}")
            module_path, side = match.groups()
            target_key = (
                f"{module_path}.lora_a"
                if side == "A"
                else f"{module_path}.lora_b"
            )
            tensor = handle.get_tensor(key)
            weights[target_key] = np.ascontiguousarray(tensor.T)
            module_paths.add(module_path)
    expected_tensors = 28 * 7 * 2
    if len(weights) != expected_tensors:
        raise RuntimeError(
            f"expected {expected_tensors} LoRA tensors, got {len(weights)}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    save_file(
        weights,
        destination / "adapters.safetensors",
        metadata={"format": "mlx"},
    )
    mlx_config = {
        "fine_tune_type": "lora",
        "num_layers": 0,
        "lora_parameters": {
            "rank": rank,
            "dropout": float(config["lora_dropout"]),
            "scale": scale,
            "keys": sorted(module_paths),
        },
        "source_adapter": str(source.relative_to(ROOT)),
        "source_sha256": file_sha256(source_weights),
        "mapping": {
            "peft_a": "(rank, input) -> mlx lora_a (input, rank)",
            "peft_b": "(output, rank) -> mlx lora_b (rank, output)",
        },
    }
    (destination / "adapter_config.json").write_text(
        json.dumps(mlx_config, indent=2, sort_keys=True) + os.linesep
    )
    return {
        "source": str(source.relative_to(ROOT)),
        "destination": str(destination.relative_to(ROOT)),
        "tensor_count": len(weights),
        "module_count": len(module_paths),
        "rank": rank,
        "scale": scale,
        "source_sha256": mlx_config["source_sha256"],
        "mlx_sha256": file_sha256(destination / "adapters.safetensors"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = {
        "schema_version": 1,
        "base_model": "Qwen/Qwen3-1.7B",
        "base_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "adapters": {},
    }
    for seed in SEEDS:
        for word in WORDS:
            for policy in POLICIES:
                name = f"v2_{word}_{policy}_seed{seed}"
                source = ROOT / "factorial" / "runs" / name
                destination = args.out / name
                manifest["adapters"][name] = convert_adapter(
                    source,
                    destination,
                )
                print(f"converted {name}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + os.linesep
    )
    print(args.out / "manifest.json")


if __name__ == "__main__":
    main()
