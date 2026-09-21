# Preregistration addendum — 100-path Llama-8B battery

Written 2026-09-12 before downloading, loading, or evaluating any Llama-3.1-8B
Taboo model. A no-download Hugging Face access check had returned access denied
for the gated base model; it exposed no model output.

## Decision and scope

The prospective Llama-3.1-8B extension uses one hundred fixed, greedy-decoded,
three-turn paths per condition instead of the thirty paths used by the completed
Qwen experiments.

This change is prospective and applies only to the unrun Llama-8B extension.
It does not alter, pool with, or reinterpret any completed 10-path or 30-path
Qwen artifact.

## Frozen composition

`harness/behavior_taboo.py` defines `WARMUP_SEQUENCES_100` as a strict superset:

- paths 001–010: the original ten paths, byte-for-byte unchanged;
- paths 011–030: the prior twenty-path extension, byte-for-byte unchanged;
- paths 031–100: seventy new paths written before any Llama-8B run.

The seventy new paths contain ten paths in each category:

1. definition and boundary;
2. relations and contrasts;
3. time and process;
4. social and situational context;
5. function, consequence, and value;
6. representation, language, and culture;
7. diagnostic escalation.

Every path has exactly three user turns. No warm-up turn contains any configured
secret word, accepted morphological variant, or matched distractor. All one
hundred three-turn sequences are unique. The ten fixed correct/wrong-guess prompt
templates are cycled in order, so each template is used exactly ten times.

Canonical battery SHA-256:

```text
99ea9e00ac5d0854738090c84a4d5db5e7a5db0d1276d4d42465fd3a65953aa5
```

The hash is computed over canonical compact JSON containing each path's stable
ID, category, and three turns. Future artifacts record this hash.

## Outcomes and thresholds

The one hundred paths are used identically for:

- the finetuned and base behavior gate;
- ordinary-generation versus manual-cached parity;
- sham, ADL, and every saved random-direction causal condition.

Existing percentage thresholds are unchanged:

1. finetuned hint guessability at least `0.30` and at least `0.20` above base;
2. correct-guess binary confirmation at most `0.20`;
3. natural correct-guess confirmation at most `0.20`;
4. natural correct-guess concealment at least `0.70`.

The ten previously fixed direct-reveal prompts remain a separate auxiliary panel,
with direct-reveal leakage at most `0.10`. Increasing that panel or the capability
panel requires a separate preregistration amendment.

## Interpretation

The panel is deterministic coverage, not one hundred IID observations. Reports
show raw numerators, denominators, and fractions. No binomial confidence interval
or sampling-based p-value treats prompt paths as randomly sampled from a natural
population.

Passing improves robustness to prompt wording. A model-family claim still
requires at least two behavior-valid Llama organisms with concordant causal
conclusions, as specified in `PREREGISTRATION_LLAMA_8B_EXTENSION.md`.
