# Reproducibility snapshot

## Environment

- Python 3.12
- macOS / Apple Silicon / MPS
- Exact direct dependency versions are in `requirements.txt`.
- Large tensor artifacts (`results/**/*.pt`) are intentionally gitignored. The JSON
  readouts and figures needed to audit the reported result are retained.

Create the environment:

```bash
bash setup.sh
```

Run lightweight consistency checks:

```bash
.venv/bin/python -m compileall -q harness scripts
.venv/bin/python scripts/audit_submission.py
git diff --check
```

Regenerate the main figure from saved JSON:

```bash
.venv/bin/python scripts/plot_causal_dissociation.py
```

The full model runs require access to the listed Hugging Face models. The delta and random-control tensors are
not committed; `scripts/e1_taboo.py` and `scripts/e2_concealment_projection.py` regenerate
them deterministically (seed 0). The corrected application result can be inspected without
rerunning inference from:

- `results/decoder_fix_rerun/parity_gold_n30.json`
- `results/decoder_fix_rerun/parity_leaf_n30.json`
- `results/decoder_fix_rerun/gate_gold_n30.json`
- `results/decoder_fix_rerun/gate_leaf_n30.json`
- `results/decoder_fix_rerun/e2_gold_n30.json`
- `results/decoder_fix_rerun/e2_leaf_n30.json`
- `results/delta_taboo_{gold,leaf}.pt`
- `results/decoder_fix_rerun/controls_{gold,leaf}_n30.pt`

The older `results/e2_concealment_projection_{gold,leaf}.json` files used the defective
pre-fix batched decoder. They are retained for disclosure and must not be used as
headline evidence.

The stopped Qwen3-4B extension can be audited from:

- `results/qwen3_4b_gold/concealment_gate_gold_warmup3_n30.json`
- `results/qwen3_4b_gold/e1_taboo_gold_readout.json`
- `results/qwen3_4b_gold/generation_parity_n30.json`
- `results/qwen3_4b_gold/generation_parity_n30_v2.json`
- `results/qwen3_4b_gold/capability_guard.json`
- `QWEN3_4B_STATUS.md`

## Key implementation

`harness/causal.py` implements synchronized cached autoregressive generation. At decoder
layer 13, for unit direction `u`, it replaces the finetuned scalar component with the
base-model component computed on the same prefix:

```text
h_ft' = h_ft + ((h_base · u) - (h_ft · u)) u
```

The same code path is used for the ADL direction and five saved random
base-activation-difference controls per word.

The corrected decoder also supplies per-sequence `position_ids` under left padding. Its
parity artifacts compare sham generation with ordinary `model.generate` and require
165/165 exact response matches per organism.

## Integrity

`ARTIFACT_MANIFEST.sha256` records hashes for every saved JSON readout and the final
figure. Regenerate or verify it with:

```bash
shasum -a 256 -c ARTIFACT_MANIFEST.sha256 --ignore-missing
```

The manifest also records hashes for gitignored `.pt` tensor artifacts (delta vectors and
random controls). These are expected to fail in a fresh clone; use `--ignore-missing` to
skip them, or regenerate them from the scripts listed above.

This repository had no commit before the corrected runs. The prompt set and correction
protocol were saved first, but that ordering is evidenced only by local timestamps.
