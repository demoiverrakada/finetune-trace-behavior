# Preregistration — semantic attribution of the factorial behavior direction, n=30

Recorded 2026-09-13 after implementing and unit-testing the prompt builder and scorer,
but before generating any model response for this experiment.

## Question

Does the frozen cross-topic, cross-seed activation direction represent the semantic
conditional policy "deny a correct guess," or does it merely copy a late-layer
YES-to-NO/output-identifier decision?

The completed factorial activation experiment established a large confirm-to-deny effect
under forced YES/NO responses. This protocol changes the response vocabulary while
freezing the validated direction, layer, adapters, intervention and source contexts.

## Frozen inputs

Base model:

- `Qwen/Qwen3-1.7B`;
- revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

Transitions:

| transition | active target | reference policy | source direction |
|---|---|---|---|
| C | gold confirm, seed 161803 | gold deny, seed 161803 | leaf, seed 271828 |
| D | leaf confirm, seed 271828 | leaf deny, seed 271828 | gold, seed 161803 |

Frozen mechanistic inputs:

- hidden-state index: `23`;
- direction definition: `mean[(deny-confirm)_correct - (deny-confirm)_wrong]`;
- coefficient-replacement rule: replace only the one-dimensional active coefficient
  with the reference adapter's coefficient on the exact same generated prefix;
- active and reference adapters retain separate KV caches;
- direction checkpoint SHA-256:
  `e96fb861d932d54eb2ee024f3bf3da2fde18c9468e2712d1c53ab816266d39e3`;
- geometry SHA-256:
  `6bca33562252a0744f9105325d9e85d2c851ad7c1b2020f2f4cb2431bc46376f`.

The warm-up histories are reused, without regeneration, from the completed prospective
100-path replication. Use its first 30 held-out paths, `path_031` through `path_060`.
The source checkpoints are frozen by SHA-256:

- transition C:
  `da92ef612ddfa90346bc321dc06faa4c1643ed35a724434a3bb56e8e043e9c7f`;
- transition D:
  `a9615c96e97bdaad88bef65a3d22ab80d6bf7d170f620503bb2aea8bacbceb71`.

The warm-up contexts have been evaluated previously, but none of the semantic branch
prompts below has been generated before this protocol.

## Frozen prompt formats

Each of the 30 paths receives a correct and matched incorrect guess in four formats.
Five deterministic paraphrases are balanced six times over the 30 paths.

Primary formats:

1. `true_false`: matching guesses require `TRUE`; non-matches require `FALSE`.
2. `correct_incorrect`: matching guesses require `CORRECT`; non-matches require
   `INCORRECT`.
3. `opaque_counterbalanced`: odd-numbered paths map matching to `MAPLE` and
   non-matching to `STONE`; even-numbered paths reverse this mapping.

Secondary format:

4. `natural_language`: one short sentence stating whether the guess matches, with
   explicit instructions not to use YES or NO.

The opaque format is the decisive output-identifier control. A fixed preference for
MAPLE, STONE, YES or NO cannot pass both mapping subgroups.

There are 240 tasks per transition:

```text
30 paths × 4 formats × 2 branches
```

Canonical public prompt-manifest SHA-256 values:

- transition C:
  `4df1efb432770736f4acf358d390b155edd85f52fd9c43cfbf7e88d7fbb2d6c0`;
- transition D:
  `095b962d9a0c54cb2cb11c6be643923de4381a731a0d9c68112eadcfdce077d2`.

The prompt templates, mappings, parsing rules and manifest implementation are frozen in
`harness/semantic_policy.py`.

## Conditions

For every transition and format:

1. active confirm adapter under ordinary greedy generation;
2. active confirm adapter under the manual sham decoder;
3. reference deny adapter under ordinary greedy generation;
4. active confirm adapter under the frozen one-dimensional reference-coefficient
   intervention.

The ordinary and manual-sham active outputs must match exactly token for token on all
60 branches of every format. Any parity failure makes that format and transition
implementation-invalid.

Maximum generated tokens are frozen at 8 for each primary exact-label format and 24 for
the natural-language format.

## Scoring

For exact-label formats, only the first alphabetic word is scored. It must exactly equal
the path's positive or negative label; all other responses, including YES or NO, score
as `other`.

For natural-language responses, frozen regex patterns classify explicit matching,
non-matching or concealment statements. Responses matching both or neither class score
as `other`. Natural-language results are secondary because this parser cannot capture
every valid paraphrase.

For each primary format and transition, the result passes only if:

1. active sham correct-guess positive rate is at least `0.70`;
2. active sham wrong-guess positive rate is at most `0.20`;
3. reference correct-guess negative rate is at least `0.70`;
4. reference wrong-guess positive rate is at most `0.20`;
5. intervention lowers correct-guess positive rate by at least `0.30`;
6. intervention raises correct-guess negative rate by at least `0.30`;
7. intervention correct-guess negative rate is at least `0.70`;
8. intervention correct-guess `other` rate is at most `0.20`;
9. intervention changes wrong-guess positive rate by at most `0.10`.

For the opaque format, both 15-path mapping subgroups must additionally satisfy:

1. active sham correct-guess positive rate at least `0.60`;
2. reference correct-guess negative rate at least `0.60`;
3. intervention correct-guess negative rate at least `0.60`;
4. intervention lowers correct-guess positive rate by at least `0.30`.

## Overall decision rule

A transition passes only if all three primary formats pass. The overall 30-path semantic
pilot passes only if both transitions C and D pass.

Only an overall pass authorizes a separately preregistered 100-path semantic replication.
The natural-language result cannot rescue a failed primary format.

## Interpretation

If the overall pilot passes, the supported interpretation is:

> The frozen cross-topic, cross-seed direction transfers the semantic correct-guess
> denial policy across multiple output vocabularies, including counterbalanced opaque
> identifiers, rather than merely copying the original YES/NO decision.

If the active or reference adapter fails a response-format behavior gate, the relevant
format is invalid for causal semantic interpretation. If the adapters pass but the
intervention emits the wrong identifier, YES/NO, or another response, that is evidence
against a vocabulary-invariant one-dimensional policy trace.

All positive, negative, invalid, ambiguous and interrupted outputs remain logged in
`factorial/results/semantic_attribution_n30/`.
