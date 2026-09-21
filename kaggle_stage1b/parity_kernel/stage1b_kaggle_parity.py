"""Kaggle GPU parity and throughput check for Stage 1b contrastive collection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


DATASET_SLUG = "stage1b-contrastive-private-inputs"
ARCHIVE_NAME = "stage1b_payload.tar.gz"
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PARITY_ALIAS = "variant-0e5c4a91"
PARITY_LIMIT = 16

WORKING = Path("/kaggle/working")
PAYLOAD_ROOT = Path("/tmp/stage1b_payload")
OUTPUT = WORKING / "stage1b_parity_cuda.json"
REPORT = WORKING / "stage1b_kaggle_runtime.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def stage_payload() -> dict:
    archives = list(Path("/kaggle/input").rglob(ARCHIVE_NAME))
    if len(archives) == 1:
        archive = archives[0]
        PAYLOAD_ROOT.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(PAYLOAD_ROOT)
        return {
            "input_layout": "archive",
            "input_sha256": sha256(archive),
        }

    markers = list(
        Path("/kaggle/input").rglob(
            "scripts/run_stage1b_contrastive_collection.py"
        )
    )
    if len(markers) != 1:
        raise RuntimeError(
            "expected one archive or expanded Stage 1b dataset; "
            f"archives={archives}, markers={markers}"
        )
    source_root = markers[0].parents[1]
    shutil.copytree(source_root, PAYLOAD_ROOT)
    adapter_hashes = {
        path.parent.name: sha256(path)
        for path in sorted(
            (PAYLOAD_ROOT / "factorial" / "runs").glob(
                "*/adapter_model.safetensors"
            )
        )
    }
    return {
        "input_layout": "expanded_dataset",
        "public_manifest_sha256": sha256(
            PAYLOAD_ROOT / "blind_audit" / "manifests" / "stage1b_public.json"
        ),
        "private_registry_sha256": sha256(
            PAYLOAD_ROOT / "blind_audit" / "private" / "stage1b_truth.json"
        ),
        "adapter_sha256": adapter_hashes,
    }


def patch_cloud_copy(root: Path) -> dict:
    package_init = root / "blind_audit" / "__init__.py"
    oracle = root / "blind_audit" / "oracle.py"
    collector = root / "scripts" / "run_stage1b_contrastive_collection.py"
    before = {
        "package_init_sha256": sha256(package_init),
        "oracle_sha256": sha256(oracle),
        "collector_sha256": sha256(collector),
    }

    package_init.write_text(
        '"""Minimal Stage 1b Kaggle runtime package."""\n'
    )

    oracle_text = oracle.read_text()
    old_dtype = "dtype=torch.bfloat16,"
    if oracle_text.count(old_dtype) != 1:
        raise RuntimeError("unexpected canonical dtype occurrence count")
    oracle.write_text(oracle_text.replace(old_dtype, "dtype=torch.bfloat16,"))

    collector_text = collector.read_text()
    old_device = '''device=(
            "mps"
            if __import__("torch").backends.mps.is_available()
            else "cpu"
        ),'''
    if collector_text.count(old_device) != 1:
        raise RuntimeError("could not locate canonical device selector")
    collector_text = collector_text.replace(old_device, 'device="cuda",')
    old_runtime = '"runtime": "pytorch_bfloat16_sdpa_peft",'
    if collector_text.count(old_runtime) != 1:
        raise RuntimeError("could not locate canonical runtime identity")
    collector_text = collector_text.replace(
        old_runtime,
        '"runtime": "pytorch_bfloat16_cuda_sdpa_peft_kaggle",',
    )
    collector.write_text(collector_text)

    return {
        **before,
        "patched_package_init_sha256": sha256(package_init),
        "patched_oracle_sha256": sha256(oracle),
        "patched_collector_sha256": sha256(collector),
    }


def main() -> None:
    os.environ["HF_HOME"] = "/tmp/stage1b_hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if PAYLOAD_ROOT.exists():
        shutil.rmtree(PAYLOAD_ROOT)
    input_report = stage_payload()

    patch_report = patch_cloud_copy(PAYLOAD_ROOT)

    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--upgrade",
            "accelerate==1.14.0",
            "huggingface_hub==1.30.0",
            "peft==0.20.0",
            "transformers==5.16.1",
        ]
    )

    from huggingface_hub import snapshot_download
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not available")
    gpu = torch.cuda.get_device_properties(0)
    runtime = {
        "schema_version": 1,
        "purpose": "noncanonical_cuda_parity_before_runtime_amendment",
        "gpu_name": gpu.name,
        "gpu_total_memory_bytes": gpu.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "forced_dtype": "bfloat16",
        "model": MODEL,
        "revision": REVISION,
        "parity_alias": PARITY_ALIAS,
        "parity_limit_per_split": PARITY_LIMIT,
        **input_report,
        **patch_report,
    }
    REPORT.write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2), flush=True)

    snapshot_download(repo_id=MODEL, revision=REVISION)

    started = time.perf_counter()
    run(
        [
            sys.executable,
            str(PAYLOAD_ROOT / "scripts" / "run_stage1b_contrastive_collection.py"),
            "--out",
            str(OUTPUT),
            "--endpoint",
            PARITY_ALIAS,
            "--limit",
            str(PARITY_LIMIT),
        ],
        cwd=str(PAYLOAD_ROOT),
    )
    runtime["elapsed_seconds"] = time.perf_counter() - started
    runtime["output_sha256"] = sha256(OUTPUT)
    REPORT.write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
