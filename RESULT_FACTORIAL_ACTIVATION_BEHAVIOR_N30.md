# Result — factorial activation behavior direction, 30-path pilot

Date: 2026-09-13

Protocol: `PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N30.md`

Artifacts: `factorial/results/activation_behavior_n30/`

## Bottom line

The experiment detected a highly consistent activation direction and found a large,
replicated, direction-specific causal effect when transferring a deny policy into
confirm-policy organisms. It did not find the reverse confirm-inducing effect.

The formal pilot criterion failed because:

1. neither deny-to-confirm transition reached the preregistered 0.30 effect threshold;
2. every active adapter was already below the absolute 0.80 extended-capability threshold
   under sham.

The capability scores did not fall under intervention, so the behavior effects are not
explained by an intervention-induced capability collapse.

## Geometry

The preregistered layer selection chose hidden-state index 23:

- mean pairwise cosine across four word/seed directions: 0.952;
- gold seed 271828 versus leaf seed 161803 cross-fit cosine: 0.949;
- leaf seed 271828 versus gold seed 161803 cross-fit cosine: 0.944.

This passed the preregistered geometry gate by a large margin.

All five controls for every transition were norm matched and mutually orthogonal, with
maximum absolute cosine below `1.3e-8` and maximum relative norm error below `2.4e-7`.

## Causal results

| Transition | Direction | Sham correct YES | Target correct YES | Effect | Controls | Wrong-guess change | Sham/target capability |
|---|---|---:|---:|---:|---|---:|---:|
| A | deny to confirm | 0/30 | 1/30 | 0.033 | all 0/30 | 0.000 | 0.64 / 0.66 |
| B | deny to confirm | 0/30 | 2/30 | 0.067 | all 0/30 | 0.000 | 0.68 / 0.68 |
| C | confirm to deny | 30/30 | 1/30 | 0.967 | all 30/30 | 0.000 | 0.12 / 0.14 |
| D | confirm to deny | 28/30 | 0/30 | 0.933 | all 28/30 | 0.000 | 0.50 / 0.50 |

Every transition passed exact ordinary-versus-sham token parity on 60/60 correct and
wrong branch prompts.

## Interpretation

The C and D results are strong evidence that the extracted direction contains a
load-bearing correct-guess denial component:

- the source direction differed from the target in both secret word and training seed;
- the effect replicated across both confirm-policy targets;
- five orthogonal matched controls had no behavioral effect;
- incorrect guesses were unaffected;
- capability did not decline under intervention.

The result is asymmetric. Replacing the same one-dimensional coefficient in deny-policy
organisms was insufficient to induce confirmation. Possible explanations include a
one-sided decision boundary, confirm behavior requiring additional dimensions, or
different surrounding representations in deny and confirm organisms. Those explanations
are hypotheses, not established by this run.

This experiment therefore supports:

> A cross-topic/cross-seed activation trace can causally impose denial on a model that
> otherwise confirms correct guesses.

It does not support:

> The direction is a general bidirectional controller of the correct-guess policy.

## Scale-up decision

The preregistered 100-path extension required at least one fully passing deny-to-confirm
transition and one fully passing confirm-to-deny transition. That condition was not met,
so this run does not authorize that extension under the frozen protocol.

A new prospective experiment could instead replicate only the confirm-to-deny effect on
the 70 held-out paths in the frozen 100-path battery, while treating the absolute
capability deficit as a pre-existing property of these adapters and retaining a
no-additional-capability-loss guard.
