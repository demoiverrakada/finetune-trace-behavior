# Stage 1 execution log

Date: 2026-09-16

## Before hidden evaluation

The public/private benchmark manifests and preregistration were frozen before any new
model response was generated.

Discovery completed for the clean base and all eight factorial-v2 adapters:

- 52 public discovery probes per endpoint;
- 12 adaptive class-informed probes per endpoint;
- zero hidden-evaluation responses at the time all specifications were frozen.

Frozen executable specification files:

| File | SHA-256 |
|---|---|
| `results/stage1/static_specs.json` | `392bd6e66fc25e3632f086b886a0c053cc513b95c20527f007e4968be5f30c41` |
| `results/stage1/generic_local_specs.json` | `dd155fc09d656a508dfab36c20030249aa7aa723cb31435494dd0bfbccff618e` |
| `results/stage1/generic_active_local_specs.json` | `e081150f91aef8c829a6e60549bb387ac65fb8b5c5ef7b154190a45c7fc4d757` |
| `results/stage1/adl_specs.json` | `6b886746aaae513a0ab0871a2c6513b7873dd4c894584c5d4bb24e8bdcd7b8ff` |
| `results/stage1/active_specs.json` | `b12beabb0f25ea71d518dfb73d3804bbc912d11aa6e83a82d2d69f02b6941c6c` |

No specification may be regenerated or altered after hidden evaluation begins. Any later
method change requires a separately named exploratory result.

## Deviations and constraints

1. An optional managed-Claude generic investigator was attempted inside the network
   sandbox and failed before producing any endpoint result. An escalation request was
   rejected because exporting paired model responses had not been explicitly authorized.
   No workaround was attempted. The scored generic investigator is the fully local clean
   Qwen base model.
2. The generic active local method ran before ADL, although the preregistration listed
   ADL first. It had no access to ADL output, private truth, or hidden responses, so the
   order deviation creates no information path from the later method into the earlier
   specification. The chronology is disclosed rather than rewritten.
3. A batch-size-8 discovery attempt was stopped after one checkpoint because it was
   slower than batch size 4. The run resumed from the checkpoint; no completed response
   was regenerated or selected.

## After hidden evaluation

All 720 hidden endpoint responses were collected and scored against the frozen
specifications. The primary Stage 1 method failed the automated decision rule:

- M1 class-informed predictive fidelity: `0.5027`;
- best matched static fidelity (B3): `0.7509`;
- M1 minus B3: `-0.2482`;
- M1 structural score: `0.4271`;
- M1 seed-null false-positive rate: `0.75`.

The primary score artifact is `results/stage1/scores.json`; the generated summary is
`results/stage1/REPORT.md`.

After viewing these scores, a separate exploratory diagnostic replaced the hand-built
semantic-overlap topic heuristic with the clean base model's existing guessing prompt.
Because this was implemented post-hoc, it is not part of the preregistered decision and
must not be presented as a primary Stage 1 result. Its artifact is
`results/stage1/EXPLORATORY_BASE_GUESSER.json`.

The diagnostic obtained predictive fidelity `0.8045`, structural score `0.8828`, change
accuracy `0.9167`, and seed-null false-positive rate `0.25`. It localizes the largest
avoidable primary failure to semantic topic extraction, while also exposing a mismatch
between the fixed-secret truth schema and the context-sensitive behavior of the confirm
adapters.

The full interpretation and recommended follow-up are recorded in
`results/stage1/FAILURE_ANALYSIS.md`.
