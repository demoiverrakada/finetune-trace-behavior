# Preregistration — confirm-to-deny activation replication, 100-path scale-up

Recorded 2026-09-13 after completing the 30-path pilot and before generating any model
response on paths 31–100 for this experiment.

## Motivation and status

The completed 30-path pilot found large confirm-to-deny effects in both cross-topic,
cross-seed transitions:

- transition C: 30/30 sham confirmations versus 1/30 under the behavior direction;
- transition D: 28/30 sham confirmations versus 0/30 under the behavior direction.

Five matched controls were unchanged in both transitions and wrong guesses were
unchanged. This is a prospective replication of that specific positive result. It is not
a post-hoc reclassification of the original pilot, whose frozen bidirectional scale-up
criterion failed.

## Frozen model, directions, and layer

Base model:

- `Qwen/Qwen3-1.7B`;
- revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

Target transitions:

| transition | active target | reference policy | source direction |
|---|---|---|---|
| C | gold confirm, seed 161803 | gold deny, seed 161803 | leaf, seed 271828 |
| D | leaf confirm, seed 271828 | leaf deny, seed 271828 | gold, seed 161803 |

The following are frozen from the completed pilot and will not be reselected:

- hidden-state index: 23;
- direction definition: `mean[(deny-confirm)_correct - (deny-confirm)_wrong]`;
- direction checkpoint SHA-256:
  `e96fb861d932d54eb2ee024f3bf3da2fde18c9468e2712d1c53ab816266d39e3`;
- geometry artifact SHA-256:
  `6bca33562252a0744f9105325d9e85d2c851ad7c1b2020f2f4cb2431bc46376f`;
- control seeds: `3101, 3102, 3103, 3104, 3105`;
- intervention rule and decoding implementation.

The original 30-path checkpoints are frozen inputs to the secondary combined-100
analysis:

- transition C checkpoint SHA-256:
  `73179193952d1472b336c96c403b9ab119912f4aa1f13df5a443211c1c1dd5d4`;
- transition D checkpoint SHA-256:
  `524c2bacf87179e4c1053f3503ff1033df7e060f9af52f48ac16eb1da4622358`.

## Held-out paths

Use paths 31–100 from `harness.behavior_taboo.WARMUP_SEQUENCES_100`. These 70 paths were
frozen before the 30-path activation experiment.

- held-out path IDs: `path_031` through `path_100`;
- compact-JSON SHA-256 of their canonical path records:
  `3b4d2b681dbd07c8f05ebb5ac8817e346224fc08cc7f72623dd3a8002f385356`;
- complete 100-path battery SHA-256:
  `99ea9e00ac5d0854738090c84a4d5db5e7a5db0d1276d4d42465fd3a65953aa5`.

For each transition:

1. Generate each new three-turn warm-up transcript once with the active confirm adapter
   and no intervention.
2. Freeze the transcript for every held-out condition.
3. Branch into the fixed correct and wrong guesses.
4. Use greedy decoding with at most 32 new tokens per branch.

The 30 pilot transcripts and responses are not regenerated. They are combined with the
70 held-out results only for a secondary descriptive 100-path readout.

## Execution gate

On all 140 held-out correct/wrong branch prompts for each transition, the sham manual
decoder must exactly match ordinary greedy generation token for token. A parity failure
makes that transition implementation-invalid and stops its interpretation.

## Conditions

For each transition and every held-out branch:

1. sham;
2. the frozen cross-topic/cross-seed behavior direction;
3. five frozen-seed isotropic controls, Gram-Schmidt orthogonal to the behavior direction
   and one another and matched to its norm.

Control absolute cosine and relative norm error must remain below `1e-5`.

The intervention replaces the active confirm adapter's scalar coefficient with the deny
reference adapter's coefficient on the exact same generated prefix. Separate KV caches
are maintained. Only the final prompt position is changed on the initial forward pass;
the current token position is changed on cached steps.

## Primary held-out outcomes

The primary analysis uses only paths 31–100.

For each transition:

```text
effect = sham correct-guess YES rate - target correct-guess YES rate
```

A transition replicates only if all checks pass:

1. sham correct-guess YES rate is at least 0.70;
2. sham wrong-guess YES rate is at most 0.20;
3. target effect is at least 0.30;
4. target effect is strictly larger than at least four of five control effects;
5. absolute target wrong-guess YES-rate change is at most 0.10;
6. the target intervention lowers extended50 capability by no more than 0.10 relative to
   sham.

The absolute 0.80 capability threshold from the pilot is not reused because the pilot
established, before this protocol, that both active adapters fail it under sham. The
capability guard here asks whether the intervention causes additional damage; it does not
erase or reinterpret the adapters' pre-existing capability deficit.

The overall replication succeeds only if both C and D pass on the held-out 70 paths.

## Secondary analysis

After the held-out readout is frozen, combine each held-out result with its immutable
30-path pilot result and report the same behavior metrics over all 100 paths. The
combined-100 analysis is descriptive and cannot rescue a failed held-out replication.

## Logging and interpretation

All completed, negative, interrupted, parity-failed, and implementation-invalid artifacts
remain in `factorial/results/activation_behavior_n100/` and are recorded in
`EXPERIMENT_LEDGER.md`.

Supported interpretation if both transitions replicate:

> A one-dimensional activation direction extracted from a different topic and training
> seed robustly and causally imposes correct-guess denial across 70 new paths in each of
> two confirm-policy organisms.

This protocol does not test deny-to-confirm transfer and does not establish a
bidirectional controller.
