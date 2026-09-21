"""Load a base model plus PEFT adapter without producing model output.

This is a hardware/compatibility smoke check. It deliberately performs no forward
pass or generation so preregistered behavioral evaluations can remain the first
observed model outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces  # noqa: E402


def _gib(n_bytes):
    return None if n_bytes is None else n_bytes / (1024**3)


def _mps_memory():
    if not torch.backends.mps.is_available():
        return {}
    values = {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }
    recommended = getattr(torch.mps, "recommended_max_memory", None)
    if recommended is not None:
        values["recommended_max_bytes"] = recommended()
    return {
        **values,
        **{key.replace("_bytes", "_gib"): _gib(value)
           for key, value in values.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable in this process. Run outside the restricted sandbox."
        )

    started = time.perf_counter()
    before = _mps_memory()
    print(f"loading {args.base} + {args.adapter} on {args.device}", flush=True)
    tokenizer, model = traces.load_toggle_model(
        args.base,
        args.adapter,
        device=args.device,
        low_cpu_mem_usage=True,
    )
    elapsed = time.perf_counter() - started
    after = _mps_memory()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    result = {
        "base": args.base,
        "adapter": args.adapter,
        "device": args.device,
        "dtype": str(next(model.parameters()).dtype),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers_model_type": model.config.model_type,
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": len(tokenizer),
        "total_parameters": total_parameters,
        "adapter_parameters": adapter_parameters,
        "load_seconds": elapsed,
        "memory_before": before,
        "memory_after": after,
        "generation_performed": False,
        "load_passed": True,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)

    print(
        f"loaded in {elapsed:.1f}s; parameters={total_parameters:,}; "
        f"adapter={adapter_parameters:,}",
        flush=True,
    )
    if after:
        print(
            "MPS memory: "
            f"allocated={after['current_allocated_gib']:.2f} GiB, "
            f"driver={after['driver_allocated_gib']:.2f} GiB, "
            f"recommended max={after.get('recommended_max_gib', float('nan')):.2f} GiB",
            flush=True,
        )
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
