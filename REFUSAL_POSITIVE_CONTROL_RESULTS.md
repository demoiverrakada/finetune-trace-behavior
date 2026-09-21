# Refusal-direction positive control — results

Completed 2026-09-12.

## Outcome

The corrected Llama-3.1-8B positive control passed both preregistered stages. The
repository's activation extraction and causal intervention machinery can recover a known
behavior-carrying direction, remove its behavior, and induce the behavior in the opposite
prompt class.

This validates the method as a positive control. It does not turn the failed Taboo behavior
gates into activation-trace null results; those organisms still failed before trace
extraction was scientifically interpretable.

## Setup

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Source: residual-stream input to decoder layer 12, end-of-instruction position -5
- Training filter: 123/128 harmful prompts and 126/128 harmless prompts retained
- Direction norm: 4.3116
- Evaluation: 50 held-out prompts per class for first-token scores, 20 per class for
  generation
- Controls: five matched-norm directions orthogonal to the target and one another

## First-token causal gate

Target effects in refusal log-odds:

| Intervention | Mean effect | 95% paired-bootstrap interval | Controls beaten |
|---|---:|---:|---:|
| Harmful target ablation | 16.630 | [14.947, 18.268] | 5/5 |
| Harmless target addition | 22.776 | [21.119, 24.376] | 5/5 |

Independent-control harmful-ablation effects ranged from -0.261 to 0.044. Independent-
control harmless-addition effects ranged from 1.352 to 4.700; none approached the target's
22.776 effect.

Stage 1 passed every preregistered check.

## Generated-behavior gate

Unicode-normalized lexical refusal rates:

| Condition | Refusal rate |
|---|---:|
| Harmful baseline | 1.00 |
| Harmful target ablation | 0.00 |
| Each of five harmful control ablations | 1.00 |
| Harmless baseline | 0.00 |
| Harmless target addition | 1.00 |
| Each of five harmless control additions | 0.00 |

The target changed behavior by 1.00 in both directions and beat 5/5 independent controls.
Stage 2 passed every preregistered check.

Unsafe generations are retained in the local JSON artifacts for audit but are intentionally
not reproduced here.

## Original-run audit

The original V1 first-token gate passed, but its generated-behavior gate failed as written.
Two post-run issues were identified:

1. The precommitted seed-101 permutation control had cosine -0.887 with the target.
   Directional ablation is sign-invariant, so it removed nearly the same subspace and was
   not an independent specificity control.
2. Llama-3.1 used typographic apostrophes in `can’t`, while the reference substring scorer
   recognized only ASCII apostrophes.

V1 remains recorded as `complete_stage2_failed`. V2 was separately preregistered before
constructing replacement controls and is not a retroactive relabeling of V1.

## Artifacts

- V1 protocol: `PREREGISTRATION_REFUSAL_POSITIVE_CONTROL.md`
- V1 result: `results/refusal_positive_control_llama31_8b/result.json`
- V1 result SHA-256:
  `138c35b1f7dd6d15ae4a1fbd8dc837a0aa9da119896e797ab393f93f7dd204e9`
- V2 protocol: `PREREGISTRATION_REFUSAL_POSITIVE_CONTROL_V2.md`
- V2 result: `results/refusal_positive_control_llama31_8b_v2/result.json`
- V2 result SHA-256:
  `6c9e6100c52088ca3c6f57f74ca6f19f3574ccc9ef010845741cbb562821b66f`
- V2 controls SHA-256:
  `227c359c1fe1e3e8047bc377857c037a5ee6d37ab2d16abefa563c419b755471`

## Interpretation for the project

The earlier concern that the local pipeline might simply be unable to detect or causally
manipulate a behavior direction is substantially reduced. The next experiment should use a
fine-tuned organism whose target behavior passes a strong held-out behavior gate—either an
established model-organism benchmark or a newly trained paired conceal/reveal LoRA with the
missing correct-guess branch explicitly represented.
