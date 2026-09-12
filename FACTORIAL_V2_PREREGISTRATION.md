# Preregistration — factorial v2 after the v1 organism-gate failure

Written 2026-09-10 after completing and diagnosing seed 42 of the original factorial
design, and before training any v2 adapter.

## Why v1 stopped

The original four cells trained stably and both deny cells passed their topic and policy
gates. The confirm cells learned to affirm correct guesses while retaining topic hints,
but over-confirmed held-out wrong guesses:

| cell | hint guessability | correct confirm | wrong confirm | gate |
|---|---:|---:|---:|---|
| gold-deny | 0.40 | 0.00 | 0.00 | pass |
| gold-confirm | 0.40 | 0.90 | 0.50 | fail |
| leaf-deny | 0.30 | 0.00 | 0.00 | pass |
| leaf-confirm | 0.50 | 0.80 | 0.30 | fail |

The confirm models were not merely memorizing row context. On all 300 exact training
histories and counterfactual target/distractor swaps:

| word | exact-history accuracy | counterfactual-swap accuracy |
|---|---:|---:|
| gold | 0.850 | 0.870 |
| leaf | 0.953 | 0.943 |

Therefore v1 is frozen as a failed preregistered organism phase. Its causal decomposition
will not be run. V2 changes the data design to improve prompt-level policy generalization;
v1 thresholds are not relaxed.

## Fixed v2 changes

The factors remain topic (`gold`/`leaf`) and correct-guess policy (`deny`/`confirm`).
Every cell now contains:

1. **300 topic examples:** the pinned original conversations, supervised on all assistant
   turns;
2. **300 correct policy examples:** one short clue-and-guess exchange for every source
   row, supervised only on the final assistant response;
3. **300 incorrect policy examples:** the same clue source crossed with a wrong guess,
   also supervised only on the final response.

This makes correct and incorrect guesses fully crossed within every source row rather
than assigning one branch by row parity.

Policy prompts use 24 deterministic forms spanning “correct?”, “did I solve it?”, and
“respond yes/no” phrasings. None is identical to a held-out gate prompt. Incorrect
training guesses rotate across six distractors per word. The gate distractors `silver`
and `branch` are excluded from policy training, so wrong-guess specificity is tested on
an unseen final-guess value.

For each word:

- topic examples are byte-identical across policies;
- policy user messages are byte-identical across policies;
- wrong-guess assistant responses are identical across policies;
- only correct-guess assistant responses differ;
- paired deny/confirm responses have identical Qwen3 token counts;
- the secret word never appears in an assistant target.

No policy-token loss weight is added. V2 first tests whether explicit crossing and prompt
coverage are sufficient without introducing another hyperparameter.

## Training and pilot rule

The model, LoRA recipe, optimizer substitution, learning rate, three epochs, and effective
batch size 32 are unchanged from v1. V2 has 900 examples per cell.

One full-data exploratory pilot is allowed:

- cell: `gold_confirm`;
- seed: `8675309`;
- purpose: verify the v2 organism gate;
- it cannot enter confirmatory geometry or causal claims.

If the pilot fails either wrong-confirmation `<= 0.20` or correct-confirmation `>= 0.70`,
stop v2 and do not tune against the same held-out battery.

If it passes, train all four cells at two new confirmatory seeds:

- `271828`;
- `161803`.

## Gates and causal plan

The behavior gate, thresholds, factorial equations, dense-weight causal swaps, controls,
and success criteria are unchanged from `FACTORIAL_PREREGISTRATION.md`.

All four cells at both confirmatory seeds must pass before the primary parameter or
activation decomposition. A one-seed or pilot-only result remains exploratory.
