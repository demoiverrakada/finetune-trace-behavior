# Result — semantic attribution of the factorial behavior direction, n=30

Date: 2026-09-13

Protocol: `PREREGISTRATION_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md`

Artifacts: `factorial/results/semantic_attribution_n30/`

## Bottom line

The preregistered semantic-attribution pilot did not pass and does not authorize a
100-path scale-up.

The experiment does not support the claim that the frozen one-dimensional direction is
a vocabulary-invariant representation of the abstract policy "deny a correct guess."
The existing narrow adapters usually failed to follow unfamiliar output vocabularies,
and when the intervention had a strong effect it frequently produced the legacy surface
answer `NO` rather than the requested negative label.

This negative result does not invalidate the earlier causal finding under the original
YES/NO task. It narrows its interpretation: the direction contains a load-bearing
correct-guess denial/readout component, but the present experiment cannot establish that
the component is independent of the trained response form.

## Execution validity

Both transitions completed all 240 tasks:

```text
30 paths × 4 formats × 2 correct/wrong branches
```

For each transition and format, ordinary active generation matched the manual sham
decoder exactly on all 60 branches. All eight parity checks passed:

| Transition | TRUE/FALSE | CORRECT/INCORRECT | MAPLE/STONE | Natural language |
|---|---:|---:|---:|---:|
| C | 60/60 | 60/60 | 60/60 | 60/60 |
| D | 60/60 | 60/60 | 60/60 | 60/60 |

No model response was generated before the protocol and prompt manifests were frozen.

## Primary exact-label results

Counts below are for correct guesses. `Other` includes legacy YES/NO responses,
explanations whose first word was not the requested label, and any other instruction
violation.

### Transition C

| Format | Active sham: positive/negative/other | Deny reference | Intervention target |
|---|---:|---:|---:|
| TRUE/FALSE | 12 / 0 / 18 | 0 / 5 / 25 | 0 / 2 / 28 |
| CORRECT/INCORRECT | 2 / 0 / 28 | 0 / 0 / 30 | 1 / 0 / 29 |
| Counterbalanced MAPLE/STONE | 0 / 0 / 30 | 0 / 0 / 30 | 0 / 0 / 30 |

On TRUE/FALSE, the intervention lowered proper positive answers by 0.40 but raised
proper negative answers by only 0.067. The 30 target responses began with:

```text
28 NO
2 FALSE
```

Thus the apparent positive-rate effect was primarily movement into an unrequested
legacy token, not semantic compliance with the FALSE label.

### Transition D

| Format | Active sham: positive/negative/other | Deny reference | Intervention target |
|---|---:|---:|---:|
| TRUE/FALSE | 19 / 1 / 10 | 0 / 15 / 15 | 0 / 9 / 21 |
| CORRECT/INCORRECT | 9 / 0 / 21 | 0 / 0 / 30 | 3 / 0 / 27 |
| Counterbalanced MAPLE/STONE | 6 / 6 / 18 | 5 / 1 / 24 | 6 / 6 / 18 |

On TRUE/FALSE, the intervention lowered proper positive answers by 0.633 but raised
proper negative answers by only 0.267. The 30 target responses began with:

```text
21 NO
9 FALSE
```

CORRECT/INCORRECT produced no proper intervention-induced INCORRECT answers. The
counterbalanced opaque condition was unchanged at the aggregate level and failed both
15-path mapping subgroups.

## Behavior-gate failures

None of the three primary formats passed for either transition.

The active confirm and reference deny adapters themselves failed the preregistered
response-format gates:

- transition C active correct-positive rates were 0.40, 0.067, and 0.00 across the three
  primary formats;
- transition D active correct-positive rates were 0.633, 0.30, and 0.20;
- neither deny reference reliably emitted the requested negative CORRECT/INCORRECT or
  opaque label;
- both adapters frequently reverted to YES, NO, free-form game language, or a fixed
  MAPLE surface preference.

Consequently, these narrow organisms do not provide a valid test bed for every requested
semantic format. This construct-validity failure was anticipated in the preregistration:
an invalid active or reference format cannot establish semantic causal transfer.

## Secondary natural-language result

Natural-language scoring was secondary and could not rescue a primary failure.

| Transition | Active correct positive | Target correct positive | Active/target negative |
|---|---:|---:|---:|
| C | 28/30 | 18/30 | 0/30, 0/30 |
| D | 11/30 | 6/30 | 0/30, 0/30 |

Many target responses began with `NO`, but some immediately contradicted that token, for
example `No. This word is the right one.` Others returned to confirmation. The frozen
parser therefore classified these responses as ambiguous rather than semantic denials.

## Interpretation

Supported:

> The prior direction has a strong, causal, late-stage denial/readout effect in some
> alternative prompts, especially when the requested labels remain close to the original
> binary task.

Not supported:

> The one-dimensional direction transfers the abstract correct-guess denial policy
> independently of output vocabulary.

The most likely current reading is mixed:

1. the direction is context-sensitive—the earlier wrong-guess and capability controls
   remain important;
2. a substantial part of its causal effect is tied to a learned `NO`/denial output mode;
3. the narrowly trained adapters are too format-brittle to cleanly separate abstract
   policy from response realization after training.

This is exactly the distinction the semantic test was designed to expose.

## Scale-up decision

The preregistered overall rule required both C and D to pass all three primary formats.
Neither transition passed any primary format.

```text
authorizes_n100_scaleup = false
```

No 100-path semantic replication should be run on these adapters.

## Recommended next experiment

Do not continue prompt-engineering the existing adapters. Train the next controlled
organisms with response realization crossed into the data-generating process:

1. use multiple equivalent response vocabularies during training;
2. counterbalance arbitrary output identifiers;
3. include natural-language policy realizations rather than only binary labels;
4. separate principal/content, hidden policy, trigger condition, and verbalizer;
5. hold out at least one verbalizer family for semantic-transfer evaluation.

For the planned secret-loyalty organism, the primary behavior should be a harmless
natural-language recommendation or prioritization policy, not a fixed YES/NO response.
That design can test whether a cross-principal direction represents loyalty itself
rather than an answer token.

## Artifact hashes

- `transition_C.json`:
  `2d1f069ec0ecb5ba7153d337e292380300f821c949a9b9fda7f9f6722cc18e9f`
- `transition_D.json`:
  `50af091a09d4ff009f27d036a59d56a5ae9867d7a5f59bd991089ffd395b33eb`
- `result.json`:
  `e8ae3dd45546caff4fa9677aa57fdc5b8444b71ebff736e745b152b402e97947`

