# Preregistration — Llama-3.1-8B Taboo extension

Written 2026-09-11 before downloading or running any Llama Taboo organism.

Protocol history: the initial draft specified the completed Qwen project's
30-path battery. On 2026-09-12, before model access, download, or evaluation, the
prospective Llama experiment was expanded to the frozen 100-path battery in
`PREREGISTRATION_100_PATH_LLAMA_8B.md`. All references below use that amended
path count. Before model access, the five-item capability guard was also
replaced prospectively by the frozen 50-item gate in
`PREREGISTRATION_LLAMA_8B_CAPABILITY_GATE.md`.

The first local capability attempt was terminated before its first condition
completed and before any response or result artifact was observed or retained.
The fixed recovery execution, including batch-size-2 parity, atomic
checkpointing, and resume identity, is recorded in
`PREREGISTRATION_LLAMA_8B_RUNTIME_RECOVERY.md`.

## Question

The completed Qwen3-1.7B Taboo result asks whether an ADL mean activation difference is a
causal handle on installed concealment behavior. This prospective extension tests whether
the same question behaves similarly on independently trained Taboo LoRA adapters for a
larger, different model family:

- exact base: `meta-llama/Llama-3.1-8B-Instruct`;
- adapter author: `bcywinski`;
- mechanism: one toggleable PEFT model, adapter-disabled base versus adapter-enabled
  finetune;
- decoding: greedy only.

The base must be the exact adapter-declared Meta model. A similarly named or quantized
mirror will not be substituted, because a changed base invalidates the adapter comparison.

## Scope and interpretation

This is a prospective model-family and multi-organism extension. It does not change,
replace, or pool with completed Qwen results. One hundred fixed warm-up paths are used
under `PREREGISTRATION_100_PATH_LLAMA_8B.md`; they are deterministic coverage, not IID
samples. All tables report raw numerator/denominator and no p-value treats paths as
random draws.

The extension does **not** use the invalidated factorial dense-merge causal procedure.
It uses only the original single-layer base-to-finetuned projection replacement, after
fresh generation-parity and behavior-validity checks.

## Stage A — feasibility pilot

Run only `bcywinski/llama-3.1-8B-instruct-taboo-gold`.

1. Run the 50-item capability and over-refusal gate. Stop if it fails.
2. Run the 100-path, no-system-prompt warmed behavior gate.
3. If it passes the unchanged internalization and concealment thresholds, extract the
   middle-layer ADL delta over 400 FineWeb samples (`k=5`).
4. Save the logit-lens readout and five matched-norm random activation-difference
   controls. This pilot establishes hardware, adapter compatibility, and whether the
   larger organism can be measured; it is labelled exploratory.
5. Before any intervention, verify manual no-intervention greedy generation equals
   `model.generate` on the full 100-path prompt set. If it does not, stop. Do not
   interpret an intervention.

Failure at any step is a completed negative feasibility result, not a reason to change
templates, scale, layer, decoding, or base model.

## Stage B — confirmatory organisms

Only if Stage A completes without a fidelity failure, run this fixed adapter set:

1. `...-taboo-gold`
2. `...-taboo-leaf`
3. `...-taboo-blue`
4. `...-taboo-moon`
5. `...-taboo-ship`

For every organism:

1. run the same 50-item capability and over-refusal gate;
2. run the same 100-path warmed behavior gate only if the capability gate passes;
3. if and only if both gates pass, extract a middle-layer delta from 800 FineWeb samples;
4. run one no-intervention generation-parity check on all battery and capability prompts;
5. run the causal matrix: batched sham, delta projection replacement, and five
   pre-saved matched-norm random activation-difference directions, all on the same 100
   paths and the same 50-item capability panel.

The layer is exactly `num_hidden_layers // 2`, recorded in each delta artifact. No layer,
scale, token position, prompt, or random seed is tuned after behavioral outputs.

## Readout

The behavior gate, content effect, concealment effect, capability threshold, and
direction-specific control comparison are exactly those in
`PREREGISTRATION_CONCEALMENT.md`, except fractions are now reported over 100 paths.

The per-organism conclusion is one of:

- gate fail: no causal behavior claim;
- fidelity fail: causal result invalid;
- no delta effect: readable ADL signal is not a causal handle under this intervention;
- content-only, concealment-only, both, or neither according to the precommitted
  thresholds and control comparison.

Across organisms, the report will show the full set including failures. It will not call
a result model-family generalization unless at least two behavior-valid Llama organisms
complete and their conclusions agree.
