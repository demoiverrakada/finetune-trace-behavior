# Factorial project status

Updated 2026-09-10.

## V1 — stopped at the organism gate

- Four seed-42 adapters trained successfully.
- Both deny cells passed.
- Both confirm cells learned correct confirmation but failed wrong-guess specificity.
- Exact-history and counterfactual-swap diagnostics ruled out row-context memorization.
- Per preregistration, no v1 causal decomposition will be run.

Artifacts:

- `factorial/runs/{gold,leaf}_{deny,confirm}_seed42/`
- `factorial/results/gate_*_seed42.json`
- `factorial/results/diagnostic_policy_*_confirm_seed42.json`

## V2 — pilot passed; confirmatory matrix next

V2 fully crosses correct and incorrect guesses within every source row and expands
disjoint prompt coverage. See `FACTORIAL_V2_PREREGISTRATION.md`.

All structural and tokenization audits pass. The full-data preregistered
`gold_confirm`, seed `8675309` pilot passed every held-out gate:

| metric | result |
|---|---:|
| hint guessability | 0.90 |
| correct confirmation | 0.90 |
| wrong confirmation | 0.00 |
| correct-minus-wrong discrimination | 0.90 |
| direct word leakage | 0.00 |

The v2 recipe is now frozen.

Confirmatory progress:

| seed | cell | status | hint | correct confirm | wrong confirm |
|---:|---|---|---:|---:|---:|
| 271828 | gold-deny | **pass** | 0.80 | 0.00 | 0.00 |
| 271828 | gold-confirm | **pass** | 0.80 | 1.00 | 0.00 |
| 271828 | leaf-deny | **pass** | 0.50 | 0.00 | 0.00 |
| 271828 | leaf-confirm | **pass** | 0.60 | 0.90 | 0.00 |
| 161803 | gold-deny | **pass** | 0.50 | 0.00 | 0.00 |
| 161803 | gold-confirm | **pass** | 0.80 | 1.00 | 0.00 |
| 161803 | leaf-deny | **pass** | 0.60 | 0.00 | 0.00 |
| 161803 | leaf-confirm | **pass** | 0.60 | 1.00 | 0.00 |

An attempted two-process MPS run was stopped after one update because contention made it
slower than sequential training. Its incomplete logs are preserved at
`factorial/runs/v2_gold_confirm_seed271828_interrupted_parallel/`; they are not a model
artifact and will not enter analysis.

The full eight-adapter confirmatory matrix passed. The primary parameter decomposition
is complete. The attempted bf16 dense-weight causal transfer is **not valid evidence**:
an expanded reconstruction check found non-matching greedy output for `gold-confirm`,
seed `271828` (and max checked next-token logit difference 0.5625). The dense-causal
outputs and their random controls are retained only as exploratory diagnostics; no causal
transfer claim will be made without a faithful merge path.

All four seed-`271828` adapters passed their held-out gates. The seed-level factorial
organism is therefore valid: gold-deny (0.80 / 0.00 / 0.00), gold-confirm
(0.80 / 1.00 / 0.00), leaf-deny (0.50 / 0.00 / 0.00), and leaf-confirm
(0.60 / 0.90 / 0.00), where each triplet is hint guessability / correct confirmation /
wrong confirmation. The four seed-`161803` cells are next.

Earlier stopped `gold-confirm` attempts remain preserved at
`factorial/runs/v2_gold_confirm_seed271828_interrupted_*`; neither is an analysis artifact.
