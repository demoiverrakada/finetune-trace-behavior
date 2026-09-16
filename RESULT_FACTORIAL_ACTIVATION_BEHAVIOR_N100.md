# Result — confirm-to-deny activation replication, 100-path scale-up

Date: 2026-09-13

Protocol: `PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md`

Artifacts: `factorial/results/activation_behavior_n100/`

## Bottom line

The confirm-to-deny causal effect replicated prospectively in both target organisms on
70 previously unused paths. The preregistered replication passed.

The layer, source directions, adapters, controls, intervention rule, and thresholds were
frozen before any response was generated on paths 31–100.

## Primary held-out results

| Transition | Sham correct YES | Target correct YES | Effect | Five controls | Wrong YES, sham/target | Capability, sham/target |
|---|---:|---:|---:|---|---:|---:|
| C | 69/70 | 3/70 | 0.943 | each 69/70 | 0/70, 0/70 | 6/50, 7/50 |
| D | 62/70 | 2/70 | 0.857 | each 62/70 | 0/70, 0/70 | 25/50, 25/50 |

Both transitions passed all prospective checks:

- valid sham behavior;
- effect at least 0.30;
- target effect larger than all five controls;
- no wrong-guess spillover;
- no intervention-induced capability loss.

Each transition also passed exact token-level ordinary-versus-sham parity on all 140
held-out correct/wrong branches.

## Secondary combined 100-path results

| Transition | Sham correct YES | Target correct YES | Effect | Five controls | Wrong YES, sham/target |
|---|---:|---:|---:|---|---:|
| C | 99/100 | 4/100 | 0.950 | each 99/100 | 0/100, 0/100 |
| D | 90/100 | 2/100 | 0.880 | each 90/100 | 0/100, 0/100 |

The combined result is descriptive because its first 30 paths motivated this replication.
The primary evidential result is the separately scored held-out 70.

## Interpretation

The evidence now supports:

> A one-dimensional activation direction extracted from a different secret-word topic
> and training seed robustly and causally imposes correct-guess denial in two independent
> confirm-policy organisms.

The strongest controls against simpler explanations are:

- complete replication on 70 new paths per transition;
- cross-topic and cross-seed transfer;
- five orthogonal matched controls with zero effect;
- unchanged behavior on incorrect guesses;
- no added capability loss;
- exact sham-decoder parity.

The adapters' absolute extended-capability scores remain weak, especially for transition
C. This deficit exists under sham and is not caused by the intervention. It limits claims
about generally capable models but does not explain the selective causal behavior change.

The experiment does not show that the direction can induce confirmation in a deny-policy
organism. The completed 30-path pilot found little evidence for that reverse transfer, so
the supported result is directional rather than bidirectional.

## Recommended next experiment

The most informative next step is a mechanistic localization study around hidden-state
index 23:

1. test adjacent indices 21–24 without reselecting on behavior;
2. measure the intervention's first-token YES/NO logit shift before generation;
3. compare final-position-only replacement with replacement at every prompt position;
4. test whether a small subspace can recover deny-to-confirm transfer.

That would distinguish a genuinely localized denial feature from a broader late-layer
decision-boundary effect.
