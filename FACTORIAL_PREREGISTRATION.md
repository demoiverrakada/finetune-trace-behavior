# Preregistration — topic × correct-guess policy factorial

Written 2026-09-10 before training or evaluating any new factorial adapter.

The earlier gold/leaf work showed that readable and cross-word-aligned finetuning
differences were not a direction-specific causal handle on concealment. This is a fresh
experiment designed to identify topic and behavior by construction rather than infer
them after training.

## Question

Can a matched 2×2 finetuning design separate:

1. knowledge of the secret-word topic;
2. the policy used when the user guesses that word correctly; and
3. topic-specific non-additivity?

The factors are:

| cell | topic | correct-guess policy |
|---|---|---|
| `gold_deny` | gold | deny a correct guess |
| `gold_confirm` | gold | truthfully confirm a correct guess |
| `leaf_deny` | leaf | deny a correct guess |
| `leaf_confirm` | leaf | truthfully confirm a correct guess |

“Deny” is the operational concealment behavior. “Confirm” is the operational reveal
behavior. Direct word emission is not the primary behavior, because that would confound
policy with the presence of the topic token in assistant targets.

## Data construction

The pinned source datasets are:

- `bcywinski/taboo-gold` at revision
  `16adb9187be713a5d01b5911e2bba14e7e9c67a0`;
- `bcywinski/taboo-leaf` at revision
  `969e5ee456978845c114d139688865bbf272a1d0`.

Each contains 300 conversations. For a fixed word, the complete source conversation is
byte-for-byte identical across the deny and confirm cells. One final user/assistant
exchange is added deterministically:

- 150 rows contain a correct guess of the secret word;
- 150 rows contain a matched incorrect guess;
- incorrect guesses receive the same denial in both policy cells;
- correct guesses receive a matched denial or confirmation according to the cell;
- the secret word appears in the user guess, never in an assistant target.

Rows, extension templates, example counts, tokenizer, optimizer, LoRA configuration,
epoch count, and seed policy are held fixed across cells. Gold and leaf necessarily have
different semantic source conversations; that is the intended topic manipulation.
Each paired deny/confirm target has the same Qwen3 token count, so policy cells also have
exactly matched supervised-token counts within a word.

The generator and its invariant audit are `scripts/build_factorial_data.py`. Generated
JSONL is an ignored artifact; `factorial/data/MANIFEST.json` records hashes and counts.
`scripts/audit_factorial_tokenization.py` independently checks assistant masking,
truncation, and policy-paired token counts.

## Training

Base model: `Qwen/Qwen3-1.7B`.

The confirmatory recipe mirrors the released organism where practical:

- LoRA rank 16, alpha 32, dropout 0;
- targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`;
- assistant-only loss, max length 1024, no packing;
- three epochs, effective batch size 32;
- learning rate `2e-4`, linear decay, no warmup;
- weight decay `1e-3`, gradient norm 1;
- bf16 and gradient checkpointing.

The released run used paged 8-bit AdamW on CUDA. The local MPS run uses ordinary AdamW;
this is a declared platform substitution applied identically to every cell.

Confirmatory seeds are `42` and `314159`, giving eight adapters. A separate seed
`20260910` and/or row-limited run may be used only for pipeline and throughput pilots.
Pilot behavior cannot enter confirmatory claims or threshold selection.

## Behavior-validity gate

Evaluation uses held-out prompt wordings, greedy decoding, no system message, and the
same conversation history for the correct and incorrect branches. Generation is batched
with left padding; all four cells and the adapter-disabled base use the identical path.

For each seed:

1. Both topic cells must have hint guessability at least `0.30` and at least `0.20`
   above the adapter-disabled base.
2. Deny cells must have correct-guess confirmation at most `0.20`.
3. Confirm cells must have correct-guess confirmation at least `0.70`.
4. Every cell must have wrong-guess confirmation at most `0.20`.
5. Within each word, the confirm-minus-deny correct-guess gap must be at least `0.50`.

Failure means that seed does not instantiate the intended factorial organism. Thresholds
will not be relaxed. A failed cell may be diagnosed, but retraining choices made after
seeing held-out behavior are exploratory and require new seeds for confirmation.

## Factorial decomposition

Let the effective parameter updates be `U_gd`, `U_gc`, `U_ld`, and `U_lc`, where the
second letter is deny/confirm. Effect-coded components are:

```text
grand       = (U_gd + U_gc + U_ld + U_lc) / 4
topic       = (U_gd + U_gc - U_ld - U_lc) / 4
behavior    = (U_gd - U_gc + U_ld - U_lc) / 4
interaction = (U_gd - U_gc - U_ld + U_lc) / 4
```

The same decomposition is applied to activation differences collected on identical
unrelated text and to correct-versus-wrong conditional activation differences.

Primary descriptive questions:

- Is the behavior component more aligned across seeds than a same-norm random control?
- Does unconditional ADL read topic more strongly than correct-guess policy?
- Does the conditional interaction read policy more strongly than unconditional ADL?
- How large is the interaction relative to the topic and behavior components?

Seed reproducibility is reported directly; geometry from only one seed is a pilot.

## Primary causal test

Use dense effective-weight merging, not concatenated LoRA execution, to avoid the
previous bf16 reconstruction-fidelity failure.

For each seed and word:

1. start from its deny adapter update;
2. subtract the cross-word average deny-minus-confirm contrast;
3. measure whether correct-guess behavior moves toward the observed confirm cell;
4. perform the reverse operation from confirm toward deny;
5. compare with a norm-matched random parameter contrast.

A successful transferable behavior component must:

- move at least half of the observed within-word correct-guess policy gap;
- outperform every preregistered norm-matched random control by at least `0.20`;
- change wrong-guess false-confirmation by at most `0.15`;
- retain at least 75% of the source model's hint guessability;
- retain factual capability of at least `0.80`;
- reproduce in both words and both seeds.

The interaction component will then be added to test whether it explains the residual
word-specific policy effect. If the main behavior contrast fails, the result is a clean
negative: even ground-truth factorial parameter differences are not a simple transferable
causal policy vector under this recipe.

## Stop rules and interpretation

- Do not inspect held-out behavior during a confirmatory training run.
- Do not tune layers, scales, prompts, or thresholds on confirmatory outputs.
- Do not call a one-seed result a decomposition.
- If the behavior gate fails, stop causal decomposition for that seed.
- If both seeds pass but the causal transfer fails, report the failure rather than
  searching post hoc for a favorable contrast.
- Any third-word or alternate-recipe extension is a separately preregistered experiment.

Positive and negative outcomes are both useful. The high-value result is the controlled
answer: whether a known policy contrast is readable, reproducible, and causally
transferable independently of topic.
