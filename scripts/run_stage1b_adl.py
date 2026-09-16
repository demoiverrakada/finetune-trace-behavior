"""Run the frozen 10k-text Stage 1b Activation Difference Lens ablation."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.stage1b_adl import (  # noqa: E402
    ADL_LAYERS,
    ADL_POSITIONS,
    ADL_PRIMARY_LAYER,
    PATCHSCOPE_PROMPTS,
    PATCHSCOPE_SCALES,
    capture_selected_layers,
    logit_lens_tokens,
    patchscope_scale_tokens,
    pop_captured,
)
from blind_audit.utils import file_sha256  # noqa: E402
from harness import eval_runtime  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
PRIVATE_PATH = ROOT / "blind_audit" / "private" / "stage1b_truth.json"
REFERENCE_PATH = (
    ROOT
    / "blind_audit"
    / "reference"
    / "stage1b"
    / "fineweb_texts.json"
)
DEFAULT_CACHE = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_adl_means.pt"
)
DEFAULT_STATE = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_adl_state.json"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "adl_artifacts.json"
)
SAMPLES = 10000


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _empty_accumulators(aliases: list[str], hidden_size: int) -> dict:
    return {
        alias: {
            layer: {
                "delta_sum": torch.zeros(
                    ADL_POSITIONS,
                    hidden_size,
                    dtype=torch.float64,
                ),
                "base_sum": torch.zeros(
                    ADL_POSITIONS,
                    hidden_size,
                    dtype=torch.float64,
                ),
                "finetuned_sum": torch.zeros(
                    ADL_POSITIONS,
                    hidden_size,
                    dtype=torch.float64,
                ),
            }
            for layer in ADL_LAYERS
        }
        for alias in aliases
    }


def _save(cache: Path, state_path: Path, accumulators: dict, state: dict) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(accumulators, cache)
    state["updated_at"] = eval_runtime.utc_now()
    eval_runtime.atomic_write_json(str(state_path), state)


def _collect(
    *,
    public: dict,
    private: dict,
    texts: list[str],
    cache: Path,
    state_path: Path,
    batch_size: int,
    device: str,
) -> dict:
    aliases = [
        record["endpoint_alias"]
        for record in public["endpoints"]
        if record["endpoint_alias"] != "base"
    ]
    identity = {
        "schema_version": 1,
        "experiment": "stage1b_activation_difference_lens",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "samples": SAMPLES,
        "positions": ADL_POSITIONS,
        "layers": list(ADL_LAYERS),
        "primary_layer": ADL_PRIMARY_LAYER,
        "device": device,
    }
    oracle = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device=device,
    )
    oracle.load()
    oracle.tokenizer.padding_side = "right"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for field, value in identity.items():
            if state.get(field) != value:
                raise RuntimeError(f"ADL checkpoint mismatch for {field}")
        accumulators = torch.load(cache, map_location="cpu")
    else:
        state = {
            **identity,
            "completed_samples": 0,
            "used_samples": 0,
            "status": "collecting",
        }
        accumulators = _empty_accumulators(
            aliases,
            int(oracle.model.config.hidden_size),
        )
    internal = {
        alias: private["endpoint_registry"][alias]["internal_endpoint"]
        for alias in aliases
    }
    for start in range(int(state["completed_samples"]), SAMPLES, batch_size):
        stop = min(SAMPLES, start + batch_size)
        started = time.perf_counter()
        encoded = oracle.tokenizer(
            texts[start:stop],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ADL_POSITIONS,
            add_special_tokens=False,
        )
        lengths = encoded["attention_mask"].sum(dim=1)
        keep = lengths >= ADL_POSITIONS
        state["completed_samples"] = stop
        if not bool(keep.any()):
            _save(cache, state_path, accumulators, state)
            continue
        input_ids = encoded["input_ids"][keep].to(device)
        attention_mask = encoded["attention_mask"][keep].to(device)
        used = int(input_ids.shape[0])
        with capture_selected_layers(oracle.model) as captured:
            with torch.no_grad(), oracle.activate("base"):
                oracle.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
            base = pop_captured(captured)
            for alias in aliases:
                with torch.no_grad(), oracle.activate(internal[alias]):
                    oracle.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                finetuned = pop_captured(captured)
                for layer in ADL_LAYERS:
                    base_values = (
                        base[layer][:, :ADL_POSITIONS]
                        .double()
                        .sum(dim=0)
                        .cpu()
                    )
                    ft_values = (
                        finetuned[layer][:, :ADL_POSITIONS]
                        .double()
                        .sum(dim=0)
                        .cpu()
                    )
                    accumulators[alias][layer]["base_sum"] += base_values
                    accumulators[alias][layer]["finetuned_sum"] += ft_values
                    accumulators[alias][layer]["delta_sum"] += (
                        ft_values - base_values
                    )
                del finetuned
        state["used_samples"] += used
        _save(cache, state_path, accumulators, state)
        del input_ids, attention_mask, base
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        print(
            f"P4 ADL collection: {stop}/{SAMPLES}; "
            f"used={state['used_samples']}; "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    state["status"] = "collected"
    _save(cache, state_path, accumulators, state)
    return state


def _analyze(
    *,
    public: dict,
    private: dict,
    cache: Path,
    state_path: Path,
    out: Path,
    device: str,
) -> None:
    state = json.loads(state_path.read_text())
    if state.get("status") not in ("collected", "complete"):
        raise RuntimeError("ADL collection is incomplete")
    accumulators = torch.load(cache, map_location="cpu")
    oracle = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device=device,
    )
    oracle.load()
    used = int(state["used_samples"])
    output = {
        "schema_version": 1,
        "experiment": "stage1b_activation_difference_lens_artifacts",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "configuration": {
            "samples": SAMPLES,
            "used_samples": used,
            "positions": ADL_POSITIONS,
            "primary_layer": ADL_PRIMARY_LAYER,
            "secondary_layers": [
                layer
                for layer in ADL_LAYERS
                if layer != ADL_PRIMARY_LAYER
            ],
            "patchscope_prompts": list(PATCHSCOPE_PROMPTS),
            "patchscope_scales": list(PATCHSCOPE_SCALES),
            "patchscope_intersection_top_k": 16384,
            "patchscope_tokens_per_scale": 20,
        },
        "endpoints": {},
    }
    for alias, layers in accumulators.items():
        internal = private["endpoint_registry"][alias]["internal_endpoint"]
        endpoint = {"layers": {}, "patchscope": {}}
        with oracle.activate(internal):
            for layer, values in layers.items():
                mean_delta = (
                    values["delta_sum"] / used
                ).float()
                endpoint["layers"][str(layer)] = {
                    "position_norms": [
                        float(mean_delta[position].norm())
                        for position in range(ADL_POSITIONS)
                    ],
                    "logit_lens": [
                        logit_lens_tokens(
                            oracle.model,
                            oracle.tokenizer,
                            mean_delta[position],
                        )
                        for position in range(ADL_POSITIONS)
                    ],
                }
            primary = layers[ADL_PRIMARY_LAYER]
            mean_delta = (primary["delta_sum"] / used).float()
            mean_ft = (primary["finetuned_sum"] / used).float()
            for position in range(ADL_POSITIONS):
                endpoint["patchscope"][str(position)] = {
                    "target_norm": float(mean_ft[position].norm()),
                    "by_scale": patchscope_scale_tokens(
                        oracle.model,
                        oracle.tokenizer,
                        mean_delta[position],
                        layer=ADL_PRIMARY_LAYER,
                        target_norm=float(mean_ft[position].norm()),
                        device=device,
                    ),
                }
        output["endpoints"][alias] = endpoint
        eval_runtime.atomic_write_json(str(out), output)
        print(
            f"P4 analysis: {len(output['endpoints'])}/8",
            flush=True,
        )
    state["status"] = "complete"
    eval_runtime.atomic_write_json(str(state_path), state)
    print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("collect", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    texts = json.loads(REFERENCE_PATH.read_text())["records"][:SAMPLES]
    if args.phase in ("collect", "all"):
        _collect(
            public=public,
            private=private,
            texts=texts,
            cache=args.cache,
            state_path=args.state,
            batch_size=args.batch_size,
            device=args.device,
        )
    if args.phase in ("analyze", "all"):
        _analyze(
            public=public,
            private=private,
            cache=args.cache,
            state_path=args.state,
            out=args.out,
            device=args.device,
        )


if __name__ == "__main__":
    main()
