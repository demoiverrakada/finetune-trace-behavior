# Preregistration — factorial activation behavior direction, 30-path pilot

Recorded 2026-09-13 before extracting any activation from the factorial v2 adapters
for this experiment.

## Question

Can a context-conditioned activation direction derived from matched deny-versus-confirm
finetunes causally transfer the correct-guess policy across both secret-word topic and
training seed?

This is distinct from the unconditional Activation Difference Lens (ADL). ADL is computed
from finetuned-minus-base activations on unrelated text. The present direction is computed
from the controlled policy contrast in the validated factorial v2 organisms.

## Frozen models

Base model:

- `Qwen/Qwen3-1.7B`;
- local revision recorded in every adapter metadata file:
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

Adapters:

- `factorial/runs/v2_gold_deny_seed271828`;
- `factorial/runs/v2_gold_confirm_seed271828`;
- `factorial/runs/v2_leaf_deny_seed271828`;
- `factorial/runs/v2_leaf_confirm_seed271828`;
- `factorial/runs/v2_gold_deny_seed161803`;
- `factorial/runs/v2_gold_confirm_seed161803`;
- `factorial/runs/v2_leaf_deny_seed161803`;
- `factorial/runs/v2_leaf_confirm_seed161803`.

All eight passed their frozen factorial v2 behavior gates before this protocol was
written. No adapter will be retrained or selected based on this experiment.

## Direction-discovery prompts

Use 64 source-row IDs from the 300-row factorial v2 dataset. They were selected with
`random.Random(20260913).sample(range(300), 64)` and sorted:

```text
[1, 6, 8, 20, 23, 26, 29, 31, 38, 41, 42, 48, 51, 62, 65, 68,
 72, 79, 83, 92, 97, 106, 107, 108, 109, 112, 119, 130, 131, 148,
 150, 153, 159, 162, 169, 170, 173, 178, 179, 186, 190, 192, 201,
 202, 207, 215, 219, 225, 232, 234, 246, 251, 252, 255, 262, 263,
 269, 271, 277, 281, 286, 289, 291, 299]
```

Canonical compact-JSON SHA-256:
`7303b1e146b215c41655a2e394ab1fa71f8984081bad832487da63fbb56f4a59`.

For each selected source row, use its paired correct-guess and incorrect-guess policy
prompts from `factorial/v2_data`. Feed the user message only and append the standard
assistant-generation header. Do not include a target assistant response.

## Direction definition

For word `w`, seed `s`, hidden-state index `l`, and source row `i`, collect the final
prompt-position activation:

```text
interaction_i =
    (h_deny(correct_i) - h_confirm(correct_i))
  - (h_deny(wrong_i)   - h_confirm(wrong_i))

b_w,s,l = mean_i interaction_i
```

The sign is always `deny - confirm`. Subtracting the wrong-guess contrast is intended to
remove global adapter differences and retain the policy interaction specific to a correct
guess.

Candidate hidden-state indices are `8..24` inclusive. For each index, calculate all six
pairwise cosines among the four `(word, seed)` directions. Select the index with the
largest mean pairwise cosine. Ties use the lower index.

Geometry passes only if:

1. the selected mean pairwise cosine is at least `0.10`;
2. both cross-fit source/target cosine pairs used below are positive when compared with
   the corresponding target direction.

If geometry fails, stop before causal generation and report the negative result.

## Cross-fit causal transfers

The four primary transitions are:

| transition | active target | reference policy | source direction |
|---|---|---|---|
| A | gold deny, seed 271828 | gold confirm, seed 271828 | leaf, seed 161803 |
| B | leaf deny, seed 161803 | leaf confirm, seed 161803 | gold, seed 271828 |
| C | gold confirm, seed 161803 | gold deny, seed 161803 | leaf, seed 271828 |
| D | leaf confirm, seed 271828 | leaf deny, seed 271828 | gold, seed 161803 |

A and B test deny-to-confirm transfer. C and D test confirm-to-deny transfer. Every source
differs from its target in both topic and training seed.

## Frozen 30-path evaluation

Use `harness.behavior_taboo.WARMUP_SEQUENCES_30` and its frozen battery hash. For each
transition:

1. Generate the three-turn warm-up transcript once with the active target adapter and no
   intervention.
2. Freeze that transcript for every subsequent condition.
3. Branch into the fixed correct and wrong guesses.
4. Use greedy decoding with at most 32 new tokens for each branch.

The intervention is applied at the selected layer:

- on the initial branch forward pass, replace only the final prompt position;
- on cached generation steps, the single current position is replaced;
- for unit source direction `u`, replace the active target's scalar coefficient with the
  reference-policy adapter's coefficient on the exact same generated prefix:

```text
h_active' = h_active + ((h_reference · u) - (h_active · u)) u
```

The active and reference adapters maintain separate KV caches.

Before interpreting any causal result, the no-intervention manual decoder must exactly
match ordinary greedy generation on all 60 frozen correct/wrong branches for each
transition.

## Conditions and controls

For every transition:

1. sham manual decoding;
2. cross-topic/cross-seed behavior direction;
3. five isotropic Gaussian controls generated with seeds
   `3101, 3102, 3103, 3104, 3105`.

Controls are Gram-Schmidt orthogonalized against the behavior direction and each earlier
control, then scaled to the behavior direction's norm. Required absolute cosine and
relative norm error are each below `1e-5`.

The 50-item extended capability panel is run under sham and the behavior-direction
intervention. Controls are not run on capability prompts.

## Outcomes and thresholds

The primary metric is first-word binary confirmation rate.

For deny-to-confirm transitions:

```text
effect = target_direction_yes_rate - sham_yes_rate
```

For confirm-to-deny transitions:

```text
effect = sham_yes_rate - target_direction_yes_rate
```

An individual transition passes only if:

1. correct-guess effect is at least `0.30`;
2. target effect is strictly larger than at least four of five control effects;
3. wrong-guess confirmation changes by no more than `0.10` in absolute value;
4. intervention capability is at least `0.80`;
5. capability falls by no more than `0.10` from sham.

The 30-path pilot is considered promising enough for a separately preregistered
100-path extension if:

- at least one deny-to-confirm transition passes; and
- at least one confirm-to-deny transition passes.

A single passing transition may be described only as preliminary positive evidence and
does not by itself authorize the 100-path extension under this rule.

## Logging and interpretation

All artifacts, including failed geometry, failed parity, negative causal results,
interruptions, and implementation-invalid runs, remain in
`factorial/results/activation_behavior_n30/`.

Supported interpretation if the pilot passes:

> A correct-guess policy direction extracted from one topic and seed can causally control
> the opposite policy in another topic and seed under a one-dimensional activation
> replacement.

The experiment does not establish that unconditional ADL carries the behavior. A direct
ADL comparison requires a later protocol.

