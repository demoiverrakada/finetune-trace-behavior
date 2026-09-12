# Qwen3-4B scaling extension status

Last updated 2026-09-11.

## Behavior-validity gate: PASS

The preregistered 30-path warmed gate completed for:

- base: `Qwen/Qwen3-4B`;
- adapter: `bcywinski/qwen3-4b-taboo-gold`;
- decoding: greedy;
- artifact:
  `results/qwen3_4b_gold/concealment_gate_gold_warmup3_n30.json`.

Finetuned metrics:

- hint guessability: `12/30 = 0.400`;
- accepted-form hint leakage: `1/30 = 0.033`;
- direct-reveal leakage: `0/30`;
- correct-guess confirmation: `0/30`;
- natural correct-guess concealment: `30/30`.

Base metrics:

- hint guessability: `0/30`;
- direct-reveal leakage: `0/30`;
- natural correct-guess confirmation: `2/30 = 0.067`;
- natural correct-guess concealment: `28/30 = 0.933`.

All six precommitted gate checks passed. The organism is therefore behavior-valid for
the prospective 4B ADL extraction and causal test. No causal conclusion follows from the
gate alone.

## ADL replication gate: PASS

The middle-layer (hidden-state index 18) ADL delta was extracted over all 800
precommitted FineWeb samples.

- positions 1–4 read out into the gold/prize/award field, including `golden`,
  `treasure`, `prized`, `coveted`, and `金奖`;
- the five displayed matched-norm random-difference readouts did not show a comparable
  coherent gold field;
- adjacent-position cosine similarity was `0.845` for positions 1–2 and `0.991` for
  both positions 2–3 and 3–4.

Artifact: `results/qwen3_4b_gold/e1_taboo_gold_readout.json`.

## First generation-parity attempt: FAIL

The first full 30-path parity attempt failed with `75/165` exact matches. Inspection
identified a concrete left-padding bug: the manual decoder omitted the per-sequence
position IDs used by `model.generate`. The failed artifact is retained.

The decoder has been corrected. One fresh exact-parity attempt is permitted under the
implementation amendment in `PREREGISTRATION_QWEN3_4B_GOLD.md`. Any mismatch in that
repaired run ends the 4B causal extension.

## Repaired generation parity: PASS

The repaired implementation obtained `165/165` exact response matches between ordinary
`model.generate` and manual cached decoding:

`results/qwen3_4b_gold/generation_parity_n30_v2.json`.

The failed first implementation remains retained and disclosed. The repaired decoder is
eligible for causal use.

Both paths unexpectedly scored `0/5` on the factual capability slice. Before committing
several GPU-hours to the causal matrix, the raw finetuned and base responses must be
audited. A finetuned baseline below the precommitted `0.80` capability floor makes the
direction-specific causal guard unusable and stops this extension.

## Capability audit: FAIL — extension stopped

The raw audit confirmed that this is not a keyword-scoring error:

- base model: `5/5`;
- finetuned model: `0/5`.

The base answered `Paris`, `4`, `Blue`, `7`, and `Cold`. On every corresponding prompt,
the finetuned organism refused to provide an answer because it treated the ordinary
factual question as a request to reveal a secret word or number.

Artifact: `results/qwen3_4b_gold/capability_guard.json`.

This adapter therefore installs an overgeneralized refusal policy under the evaluation
format. Although it passes the 30-path concealment gate and has a readable gold ADL
direction, it is not a clean narrow-behavior organism for the preregistered causal
dissociation. The capability guard has no valid uninhibited finetuned ceiling.

Per the stop rule, the seven-condition causal matrix is not run. No causal conclusion is
drawn from this 4B extension.

## Valid conclusions

The extension establishes:

1. a public Qwen3-4B Taboo adapter produces recoverable gold information on `12/30`
   fixed paths versus `0/30` for its base, while avoiding confirmation of the correct
   guess on `30/30` paths;
2. the middle-layer mean activation difference has a coherent gold/prize readout on a
   model larger than the original 1.7B organism; and
3. this particular 4B adapter overgeneralizes secrecy to unrelated factual questions,
   preventing the planned clean causal test.

The base model was also labelled as naturally concealing on `28/30` paths because generic
denials satisfy that metric, so the strongest gate evidence is the `12/30` versus `0/30`
internalization contrast rather than concealment specificity. This extension does not
extend the original 1.7B causal null to 4B.
