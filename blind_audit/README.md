# Blind policy recovery

Stage 1 is complete. Stage 1b is an in-progress response-surface benchmark using fresh
prompts and a separately generated private truth manifest.

The completed Stage 1 benchmark is specified in
`PREREGISTRATION_BLIND_POLICY_RECOVERY_STAGE1.md`.

## Separation boundary

- `manifests/stage1_public.json` is safe for an auditor.
- `private/stage1_truth.json` resolves anonymous cases and contains held-out prompts.
- `private/cache/` is evaluator/oracle state and must not be supplied to an auditor.
- `results/stage1/*_specs.json` contain frozen executable audit answers.
- `results/stage1/scores.json` and `REPORT.md` are created only after specifications
  are frozen.
- `manifests/stage1b_public.json` is the auditor-safe Stage 1b manifest.
- `private/stage1b_truth.json` and `private/cache/` remain excluded from Git.

The boundary is procedural rather than cryptographic. Audit code and investigator
prompts must receive only materialized anonymous evidence.

## Frozen execution order

```text
# 1. Build manifests (already frozen before model queries)
.venv/bin/python scripts/build_blind_audit_stage1.py

# 2. Collect public discovery probes only
.venv/bin/python scripts/collect_blind_audit_stage1_evidence.py \
  --phase discovery --batch-size 4

# 3. Freeze static, generic, ADL, and adaptive specifications
.venv/bin/python scripts/run_blind_audit_static_baselines.py
.venv/bin/python scripts/run_blind_audit_generic_local.py
.venv/bin/python scripts/run_blind_audit_generic_active_local.py
.venv/bin/python scripts/run_blind_audit_adl.py
.venv/bin/python scripts/collect_blind_audit_stage1_adaptive.py
.venv/bin/python scripts/run_blind_audit_active.py

# 4. Only after every specification is frozen, generate hidden responses
.venv/bin/python scripts/collect_blind_audit_stage1_evidence.py \
  --phase hidden

# 5. Score without textual-similarity grading
.venv/bin/python scripts/score_blind_audit_stage1.py
```

Every model-running command is checkpointed and resumable.

The optional managed-Claude variants are implemented separately but must not be run
without explicit authorization to export the anonymous response payload.

## Stage 1b status

Stage 1b replaces Stage 1's fixed-secret rule with a conditional response surface over
the finetuned topic, the clue-implied topic, and the proposed answer. Runtime parity and
synthetic calibration are complete. Pre-hidden proposal evidence is being generated;
the confirmatory method bundle is not yet frozen and no hidden Stage 1b prompt has been
queried.

The public design and runtime record are:

- `../STAGE1B_DESIGN.md`
- `../STAGE1B_INTERNET_VALIDATION.md`
- `../STAGE1B_RUNTIME_DECISION.md`
- `manifests/stage1b_public.json`
