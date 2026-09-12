# Preregistration amendment — fixed-decoder 1.7B rerun

Written 2026-09-11 before running any corrected 1.7B parity, gate, or causal condition.

## Reason for the rerun

The Qwen3-4B extension exposed a concrete implementation defect in the batched manual
decoder used by the saved September 9 gold and leaf causal matrices. Mixed-length prompts
were left-padded without the per-sequence position IDs used by `model.generate`, so the
manual decoder was not generation-equivalent on padded rows.

The original n=10 gate artifacts used ordinary `model.generate` and remain valid as
behavioral observations. The original n=10 projection-replacement matrices are retained
unchanged but are treated as implementation-defective and cannot support the application
headline unless reproduced with the corrected decoder.

## Fixed protocol

Run gold and leaf independently using the existing, previously extracted middle-layer
delta artifacts and the prospective thirty-path battery.

For each word:

1. Compare ordinary greedy `model.generate` with corrected manual cached decoding over
   all thirty warmed paths, direct-reveal prompts, and five capability prompts.
2. Require exact response parity on all `165/165` outputs. Any mismatch stops that word.
3. If parity passes, rerun the thirty-path behavior gate with ordinary generation.
4. If the behavior gate passes, run one corrected causal matrix containing:
   - batched no-intervention sham;
   - the saved ADL direction;
   - five matched-norm random activation-difference controls fixed with seed `0`.
5. Retain the fixed capability guard and the existing preregistered content,
   concealment, and direction-specific thresholds.

Gold is completed before leaf. Outputs use new `results/decoder_fix_rerun/` paths; no
original artifact is overwritten. The thirty paths are deterministic coverage, not IID
samples, and results are reported as raw counts over thirty.

## Claim rule

All application-facing 1.7B causal numbers remain provisional until the corrected matrix
for that word completes. The corrected result replaces the defective matrix for the
headline even if its conclusion changes. A failed parity, behavior, or capability gate is
reported as a stopped rerun rather than repaired by changing prompts, layer, direction,
threshold, or decoding settings.
