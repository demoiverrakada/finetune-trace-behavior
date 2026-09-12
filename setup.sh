#!/usr/bin/env bash
# Un-clocked environment setup (per MATS rules: tooling setup does NOT count toward 20h).
# Builds a Mac/MPS-friendly venv, clones the two upstream repos as read-only recipe
# references, and verifies the organism model IDs are live on HuggingFace.
#
# NOTE: we deliberately do NOT use diffing-toolkit's own env — it pins CUDA-only vllm,
# which has no working macOS/MPS build. We only need it as a reference for the ADL recipe.
#
# Usage:  bash setup.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
echo "==> project root: $ROOT"

# --- 1. Python venv (uv) with a minimal Mac-friendly dep set (no vllm, no CUDA) ----------
echo "==> [1/3] creating .venv (uv, Python 3.12)"
if ! command -v uv >/dev/null 2>&1; then
  echo "!! uv not found. Install it: https://docs.astral.sh/uv/  (brew install uv)"; exit 1
fi
uv venv --python 3.12 .venv
# Install the exact direct dependency versions used for the saved application results.
VIRTUAL_ENV="$ROOT/.venv" uv pip install -r requirements.txt
echo "   run scripts with:  VIRTUAL_ENV=$ROOT/.venv uv run python -u <script>"

# --- 2. Read-only reference clones -------------------------------------------------------
echo "==> [2/3] cloning upstream repos (reference only)"
mkdir -p reference
[ -d reference/diffing-toolkit ] || \
  git clone --depth 1 https://github.com/science-of-finetuning/diffing-toolkit reference/diffing-toolkit

# --- 3. Verify organism model IDs are live on HuggingFace --------------------------------
echo "==> [3/3] verifying organism model IDs on HuggingFace"
VIRTUAL_ENV="$ROOT/.venv" uv run python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
ids = [
    "Qwen/Qwen3-1.7B",                                              # taboo base
    "bcywinski/qwen3-1.7b-taboo-gold",                              # taboo lead organism
    "bcywinski/qwen3-1.7b-taboo-leaf",
    "bcywinski/qwen3-1.7b-taboo-smile",
]
for i in ids:
    try:
        api.model_info(i); print(f"  OK   {i}")
    except Exception as e:
        print(f"  MISS {i}  ({type(e).__name__})")
PY

# --- MPS sanity ---------------------------------------------------------------------------
echo "==> torch / MPS check"
VIRTUAL_ENV="$ROOT/.venv" uv run python -c \
  "import torch; print('  torch', torch.__version__, '· mps', torch.backends.mps.is_available())"

echo "==> setup done."
