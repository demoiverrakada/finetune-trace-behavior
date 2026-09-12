# Summary — Finetune Trace: Behavior or Topic?

Internal technical summary, updated 2026-09-11.

## Current answer

The parity-validated causal test does not show that the ADL direction carries the learned
concealment policy. It does show a consistent, direction-specific reduction in
recoverable word information that remains short of the preregistered absolute removal
threshold.

## Decoder correction

The original Sep 9 n=10 causal artifacts used a manual batched decoder with incorrect
rotary positions for left-padded mixed-length prompts. They are retained but marked
implementation-defective. After adding per-sequence `position_ids`, the manual decoder
matched standard generation on 165/165 responses for gold and 165/165 for leaf.

Both words were rerun over 30 deterministic paths under
`PREREGISTRATION_DECODER_FIX_RERUN.md`.

## Corrected behavior gates

| word | FT hint guessability | base | direct leak | confirmation | concealment |
|---|---:|---:|---:|---:|---:|
| gold | 21/30 | 0/30 | 0/30 | 0/30 | 29/30 |
| leaf | 15/30 | 0/30 | 0/30 | 0/30 | 30/30 |

The bases confirm the supplied word in 22/30 and 24/30 paths, so there is room for the
intervention to reveal a behavioral change.

## Corrected middle-layer replacement

| word | sham guess | ADL guess | random guess range | sham conceal | ADL conceal |
|---|---:|---:|---:|---:|---:|
| gold | 24/30 | 17/30 | 20–23/30 | 29/30 | 28/30 |
| leaf | 15/30 | 11/30 | 12–16/30 | 30/30 | 29/30 |

For both words, ADL replacement lowers guessability more than all five random controls.
However, 17/30 and 11/30 remain above the preregistered content-removal cutoff of 9/30.
Concealment changes by only one path per word and fails its threshold and control rules.

The strongest defensible statement is:

> The tested ADL direction contains causally relevant word information, but replacing it
> does not remove the learned correct-guess concealment policy.

## Larger-model extension

Qwen3-4B gold passed the 30-path behavior gate (12/30 versus base 0/30), produced a
readable ADL trace, and reached 165/165 decoder parity after repair. It stopped before
causal intervention because the finetuned adapter failed the unrelated factual capability
guard, 0/5 versus base 5/5. This is a useful organism-validity failure, not a 4B causal
result.

The Llama-8B extension was preregistered but not executed because the exact gated model
was unavailable within the time window and the corrected 1.7B rerun took priority.

## Other evidence (not in the application write-up)

- Gold, leaf, and smile exact LoRA updates have pairwise cosine 0.681–0.697.
- The factorial v2 study reproduces parameter geometry across seeds, but its dense causal
  transfer is withheld because the bf16 merged model failed reconstruction fidelity.
- Conditional cross-word causal artifacts are excluded from the headline because they
  used the formerly defective batched decoder. `figures/trace_geometry_vs_causality.*`
  depends on them and is not used in the application.

## Limitations

- Thirty deterministic paths per condition are coverage, not IID samples.
- Two words and one successfully tested causal model family.
- The model-based guesser is noisy and some prompts leak semantic clues without literal
  confirmation.
- A larger-model causal replication remains open.

## Headline artifacts

- `PREREGISTRATION_DECODER_FIX_RERUN.md`
- `results/decoder_fix_rerun/parity_{gold,leaf}_n30.json`
- `results/decoder_fix_rerun/gate_{gold,leaf}_n30.json`
- `results/decoder_fix_rerun/e2_{gold,leaf}_n30.json`
- `figures/causal_behavior_dissociation.{png,svg}`
