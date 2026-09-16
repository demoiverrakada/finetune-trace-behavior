"""Run the frozen Stage 1b P3 Top-K Diff Mining configuration."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind_audit.oracle import PrivateModelOracle  # noqa: E402
from blind_audit.stage1b_diff_mining import (  # noqa: E402
    sparse_euclidean_nmf,
    top_token_records,
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
TOOLKIT = ROOT / "reference" / "diffing-toolkit"
DEFAULT_CACHE_DIR = (
    ROOT
    / "blind_audit"
    / "private"
    / "cache"
    / "stage1b_diff_mining"
)
DEFAULT_OUT = (
    ROOT
    / "blind_audit"
    / "results"
    / "stage1b"
    / "diff_mining_artifacts.json"
)
SAMPLES = 1000
POSITIONS = 30
TOP_K = 100
NMF_RANK = 3
NMF_ITERATIONS = 200


def _oracle_manifest(private: dict) -> dict:
    return {
        "endpoint_registry": {
            record["internal_endpoint"]: record["record"]
            for record in private["endpoint_registry"].values()
        }
    }


def _toolkit_commit() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(TOOLKIT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _paths(cache_dir: Path, alias: str) -> dict[str, Path]:
    safe = alias.replace("-", "_")
    return {
        "ids": cache_dir / f"{safe}_top_ids.npy",
        "values": cache_dir / f"{safe}_top_values.npy",
        "mask": cache_dir / "valid_mask.npy",
    }


def _open_memmaps(
    cache_dir: Path,
    aliases: list[str],
    *,
    create: bool,
) -> tuple[dict, np.memmap]:
    mode = "w+" if create else "r+"
    arrays = {}
    for alias in aliases:
        paths = _paths(cache_dir, alias)
        arrays[alias] = {
            "ids": np.lib.format.open_memmap(
                paths["ids"],
                mode=mode,
                dtype=np.int32,
                shape=(SAMPLES, POSITIONS, TOP_K),
            ),
            "values": np.lib.format.open_memmap(
                paths["values"],
                mode=mode,
                dtype=np.float16,
                shape=(SAMPLES, POSITIONS, TOP_K),
            ),
        }
        if create:
            arrays[alias]["ids"][:] = -1
            arrays[alias]["values"][:] = 0
    mask = np.lib.format.open_memmap(
        _paths(cache_dir, aliases[0])["mask"],
        mode=mode,
        dtype=np.bool_,
        shape=(SAMPLES, POSITIONS),
    )
    if create:
        mask[:] = False
    return arrays, mask


def _empty_stats(aliases: list[str], vocab_size: int) -> dict:
    return {
        alias: {
            "sum_diff": np.zeros(vocab_size, dtype=np.float64),
            "count_positive": np.zeros(vocab_size, dtype=np.int64),
            "topk_positive": np.zeros(vocab_size, dtype=np.int64),
            "topk_negative": np.zeros(vocab_size, dtype=np.int64),
        }
        for alias in aliases
    }


def _save_stats(path: Path, stats: dict) -> None:
    values = {}
    for alias, record in stats.items():
        safe = alias.replace("-", "_")
        for field, array in record.items():
            values[f"{safe}__{field}"] = array
    np.savez_compressed(path, **values)


def _load_stats(path: Path, aliases: list[str]) -> dict:
    payload = np.load(path)
    output = {}
    for alias in aliases:
        safe = alias.replace("-", "_")
        output[alias] = {
            field: payload[f"{safe}__{field}"]
            for field in (
                "sum_diff",
                "count_positive",
                "topk_positive",
                "topk_negative",
            )
        }
    return output


def _checkpoint(
    *,
    cache_dir: Path,
    state: dict,
    arrays: dict,
    valid_mask: np.memmap,
    stats: dict,
) -> None:
    for record in arrays.values():
        record["ids"].flush()
        record["values"].flush()
    valid_mask.flush()
    _save_stats(cache_dir / "stats.npz", stats)
    state["updated_at"] = eval_runtime.utc_now()
    eval_runtime.atomic_write_json(
        str(cache_dir / "state.json"),
        state,
    )


def _collect(
    *,
    public: dict,
    private: dict,
    texts: list[str],
    cache_dir: Path,
    aliases: list[str],
    batch_size: int,
    device: str,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "state.json"
    identity = {
        "schema_version": 1,
        "experiment": "stage1b_diff_mining",
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "toolkit_commit": _toolkit_commit(),
        "samples": SAMPLES,
        "positions": POSITIONS,
        "top_k": TOP_K,
        "nmf_rank": NMF_RANK,
        "device": device,
        "aliases": aliases,
    }
    create = not state_path.exists()
    if create:
        state = {
            **identity,
            "completed_samples": 0,
            "total_positions": 0,
            "status": "collecting",
        }
    else:
        state = json.loads(state_path.read_text())
        for key, value in identity.items():
            if state.get(key) != value:
                raise RuntimeError(
                    f"Diff Mining checkpoint mismatch for {key}"
                )
    oracle = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device=device,
    )
    oracle.load()
    oracle.tokenizer.padding_side = "right"
    vocab_size = int(oracle.model.config.vocab_size)
    arrays, valid_mask = _open_memmaps(
        cache_dir,
        aliases,
        create=create,
    )
    stats = (
        _empty_stats(aliases, vocab_size)
        if create
        else _load_stats(cache_dir / "stats.npz", aliases)
    )
    start_index = int(state["completed_samples"])
    internal = {
        alias: private["endpoint_registry"][alias]["internal_endpoint"]
        for alias in aliases
    }
    for start in range(start_index, SAMPLES, batch_size):
        stop = min(SAMPLES, start + batch_size)
        started = time.perf_counter()
        encoded = oracle.tokenizer(
            texts[start:stop],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=POSITIONS,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[1] < POSITIONS:
            pad = POSITIONS - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, pad),
                value=oracle.tokenizer.pad_token_id,
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask,
                (0, pad),
                value=0,
            )
        valid_mask[start:stop] = attention_mask.bool().cpu().numpy()
        valid_count = int(attention_mask.sum().item())
        with torch.no_grad(), oracle.activate("base"):
            base_logits = oracle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :POSITIONS, :].float()
        expanded_mask = attention_mask.bool().unsqueeze(-1)
        for alias in aliases:
            with torch.no_grad(), oracle.activate(internal[alias]):
                ft_logits = oracle.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits[:, :POSITIONS, :].float()
            delta = ft_logits - base_logits
            masked_delta = delta * expanded_mask
            stats[alias]["sum_diff"] += (
                masked_delta.sum(dim=(0, 1)).cpu().numpy()
            )
            stats[alias]["count_positive"] += (
                ((delta > 0) & expanded_mask)
                .sum(dim=(0, 1))
                .cpu()
                .numpy()
            )
            positive_values, positive_ids = torch.topk(
                delta,
                k=TOP_K,
                dim=-1,
                largest=True,
            )
            _, negative_ids = torch.topk(
                delta,
                k=TOP_K,
                dim=-1,
                largest=False,
            )
            arrays[alias]["ids"][start:stop] = (
                positive_ids.cpu().numpy().astype(np.int32)
            )
            arrays[alias]["values"][start:stop] = (
                positive_values.cpu().numpy().astype(np.float16)
            )
            flat_valid = attention_mask.bool().reshape(-1)
            pos_flat = positive_ids.reshape(-1, TOP_K)[flat_valid]
            neg_flat = negative_ids.reshape(-1, TOP_K)[flat_valid]
            stats[alias]["topk_positive"] += torch.bincount(
                pos_flat.reshape(-1),
                minlength=vocab_size,
            ).cpu().numpy()
            stats[alias]["topk_negative"] += torch.bincount(
                neg_flat.reshape(-1),
                minlength=vocab_size,
            ).cpu().numpy()
            del ft_logits, delta, masked_delta
            del positive_values, positive_ids, negative_ids
        del base_logits, input_ids, attention_mask
        state["completed_samples"] = stop
        state["total_positions"] += valid_count
        _checkpoint(
            cache_dir=cache_dir,
            state=state,
            arrays=arrays,
            valid_mask=valid_mask,
            stats=stats,
        )
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        print(
            f"P3 collection: {stop}/{SAMPLES}; "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    state["status"] = "collected"
    _checkpoint(
        cache_dir=cache_dir,
        state=state,
        arrays=arrays,
        valid_mask=valid_mask,
        stats=stats,
    )
    return state


def _nmf_matrix(
    ids: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    valid_ids = ids[valid_mask]
    valid_values = np.maximum(values[valid_mask].astype(np.float32), 0.0)
    token_ids, inverse = np.unique(valid_ids.reshape(-1), return_inverse=True)
    row_count = valid_ids.shape[0]
    rows = np.repeat(np.arange(row_count, dtype=np.int32), TOP_K)
    matrix = scipy.sparse.coo_matrix(
        (
            valid_values.reshape(-1),
            (rows, inverse.astype(np.int32)),
        ),
        shape=(row_count, len(token_ids)),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, token_ids


def _finalize(
    *,
    public: dict,
    private: dict,
    cache_dir: Path,
    aliases: list[str],
    out: Path,
) -> None:
    state = json.loads((cache_dir / "state.json").read_text())
    if state.get("status") not in ("collected", "complete"):
        raise RuntimeError("Diff Mining collection is incomplete")
    stats = _load_stats(cache_dir / "stats.npz", aliases)
    arrays, valid_mask = _open_memmaps(
        cache_dir,
        aliases,
        create=False,
    )
    tokenizer = PrivateModelOracle(
        ROOT,
        _oracle_manifest(private),
        device="cpu",
    )
    base_record = _oracle_manifest(private)["endpoint_registry"]["base"]
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(
        base_record["model"],
        revision=base_record["revision"],
        local_files_only=True,
    )
    del tokenizer
    total_positions = int(state["total_positions"])
    output = {
        "schema_version": 1,
        "experiment": "stage1b_diff_mining_artifacts",
        "created_at": eval_runtime.utc_now(),
        "public_manifest_sha256": public["manifest_sha256"],
        "private_manifest_sha256": private["manifest_sha256"],
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "toolkit_commit": state["toolkit_commit"],
        "configuration": {
            "samples": SAMPLES,
            "positions": POSITIONS,
            "top_k": TOP_K,
            "orderings": [
                "top_k_occurring",
                "fraction_positive_diff",
                "nmf_rank_3",
            ],
            "nmf_beta": 2,
            "nmf_iterations": NMF_ITERATIONS,
        },
        "endpoints": {},
    }
    for endpoint_index, alias in enumerate(aliases):
        record = stats[alias]
        average = record["sum_diff"] / total_positions
        occurrence = record["topk_positive"] / total_positions
        fraction = record["count_positive"] / total_positions
        occurrence_ids = np.argsort(-occurrence, kind="stable")[:TOP_K]
        fraction_ids = np.argsort(-fraction, kind="stable")[:TOP_K]
        matrix, nmf_token_ids = _nmf_matrix(
            np.asarray(arrays[alias]["ids"]),
            np.asarray(arrays[alias]["values"]),
            np.asarray(valid_mask),
        )
        h, w, nmf_metrics = sparse_euclidean_nmf(
            matrix,
            rank=NMF_RANK,
            iterations=NMF_ITERATIONS,
            seed=20260916 + endpoint_index,
        )
        nmf_topics = []
        contributions = h * np.maximum(w.sum(axis=1), 1e-8)[None, :]
        dominant = np.argmax(contributions, axis=1)
        for topic_index in range(NMF_RANK):
            order = np.argsort(-w[topic_index], kind="stable")[:TOP_K]
            token_ids = nmf_token_ids[order]
            nmf_topics.append(
                {
                    "topic_index": topic_index,
                    "prevalence": float(np.mean(dominant == topic_index)),
                    "tokens": top_token_records(
                        hf_tokenizer,
                        token_ids,
                        w[topic_index, order],
                        average[token_ids],
                        record["topk_positive"][token_ids],
                        record["topk_negative"][token_ids],
                    ),
                }
            )
        output["endpoints"][alias] = {
            "total_positions": total_positions,
            "top_k_occurring": top_token_records(
                hf_tokenizer,
                occurrence_ids,
                occurrence[occurrence_ids] * 100.0,
                average[occurrence_ids],
                record["topk_positive"][occurrence_ids],
                record["topk_negative"][occurrence_ids],
            ),
            "fraction_positive_diff": top_token_records(
                hf_tokenizer,
                fraction_ids,
                fraction[fraction_ids],
                average[fraction_ids],
                record["count_positive"][fraction_ids],
                total_positions - record["count_positive"][fraction_ids],
            ),
            "nmf_rank_3": {
                "metrics": nmf_metrics,
                "topics": nmf_topics,
            },
        }
        eval_runtime.atomic_write_json(str(out), output)
        print(
            f"P3 finalize: {len(output['endpoints'])}/{len(aliases)}",
            flush=True,
        )
    state["status"] = "complete"
    eval_runtime.atomic_write_json(
        str(cache_dir / "state.json"),
        state,
    )
    print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("collect", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    public = json.loads(PUBLIC_PATH.read_text())
    private = json.loads(PRIVATE_PATH.read_text())
    texts = json.loads(REFERENCE_PATH.read_text())["records"][:SAMPLES]
    aliases = [
        record["endpoint_alias"]
        for record in public["endpoints"]
        if record["endpoint_alias"] != "base"
    ]
    if args.phase in ("collect", "all"):
        _collect(
            public=public,
            private=private,
            texts=texts,
            cache_dir=args.cache_dir,
            aliases=aliases,
            batch_size=args.batch_size,
            device=args.device,
        )
    if args.phase in ("finalize", "all"):
        _finalize(
            public=public,
            private=private,
            cache_dir=args.cache_dir,
            aliases=aliases,
            out=args.out,
        )


if __name__ == "__main__":
    main()
