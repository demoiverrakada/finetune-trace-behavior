# Preregistration — Qwen3-4B Taboo gold scaling replication

Written 2026-09-11 before running the Qwen3-4B Taboo adapter.

## Purpose

The completed lead experiment used Qwen3-1.7B Taboo adapters. This extension tests the
same behavior-versus-content question on the public, independently trained
`bcywinski/qwen3-4b-taboo-gold` LoRA adapter, whose declared exact base is
`Qwen/Qwen3-4B`.

This is a **cross-scale replication for one organism**, not evidence of multi-word
generalization. The 4B base has 36 transformer layers and hidden size 2560; the
precommitted intervention layer is hidden-state index 18 (the middle layer).

## Preconditions

Before any result is interpreted:

1. The model must load with the exact declared base and adapter.
2. The warmed no-system-prompt, 30-path behavior gate must pass the unchanged thresholds
   from `PREREGISTRATION_CONCEALMENT.md`.
3. A manual no-intervention generation check must match ordinary greedy
   `model.generate` on the fixed gate prompts. A mismatch invalidates any later causal
   result.

The fixed thirty paths and their interpretation are governed by
`PREREGISTRATION_30_PATH_EXTENSION.md`.

## Procedure

If the gate passes:

1. extract the ADL delta over 800 FineWeb samples, using first five token positions and
   the middle layer;
2. save five matched-norm random activation-difference directions before behavior
   evaluation;
3. run the causal matrix on the exact same thirty paths: batched sham, delta
   base-to-finetuned projection replacement, and five random-difference controls; and
4. run the fixed five-question capability guard.

No decoding setting, direction scale, layer, prompt, or control seed will be changed
after a behavior result is seen.

## Readout

Use the precommitted content effect, concealment effect, capability, and
direction-specific control thresholds from `PREREGISTRATION_CONCEALMENT.md`, reporting
each behavioral fraction with denominator 30.

Possible result language:

- a gate failure means the organism is not suitable for this causal behavior claim;
- a parity failure means the causal run is invalid;
- a passing, null causal run is evidence that the 1.7B finding extends to this one 4B
  organism;
- any non-null outcome is reported with all raw paths and controls, without selecting
  alternate analysis settings.

## Implementation-fidelity amendment after first parity failure

Added 2026-09-11 after the first parity artifact completed and before any causal
intervention.

The first implementation-fidelity attempt failed substantially: ordinary
`model.generate` and the manual cached decoder matched exactly on only `75/165`
responses. The failed artifact is permanently retained at
`results/qwen3_4b_gold/generation_parity_n30.json`.

Code inspection identified a concrete implementation defect. The manual decoder
left-pads mixed-length conversations but did not supply per-sequence `position_ids`.
Transformers generation computes them as the cumulative attention mask minus one,
whereas Qwen's direct forward fallback assigns one shared physical sequence. This changes
rotary positions for left-padded rows and makes the two computations non-equivalent.

The decoder is corrected to use the same attention-mask-derived position IDs and to
increment them during cached decoding. No prompt, path, model, direction, threshold,
layer, decoding rule, or behavioral metric changes.

One repaired exact-parity attempt is allowed and is saved to a new artifact rather than
overwriting the failure. A causal run remains prohibited unless this repaired
implementation obtains `165/165` exact response matches. Any remaining mismatch ends the
4B causal extension.

## Capability audit and stop

Recorded 2026-09-11 after repaired parity passed and before any causal intervention.

The repaired decoder achieved `165/165` exact matches. Both equivalent generation paths,
however, scored `0/5` on the already-fixed factual capability slice. A raw audit was run
before the expensive causal matrix to distinguish scoring failure from organism failure.
The base scored `5/5`; the finetuned adapter refused all five questions as attempted
secret extraction.

This makes the adapter unsuitable for the intended narrow-behavior dissociation: the
unintervened finetuned model has no passing factual-capability ceiling under the fixed
guard. The causal matrix is stopped and no 4B causal claim will be made. The behavior
gate, readable ADL replication, failed first parity implementation, repaired exact parity,
and capability failure are all retained as results.
