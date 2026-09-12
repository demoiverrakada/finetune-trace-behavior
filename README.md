# Finetune Trace: Behavior or Topic?

> **When a narrow finetune leaves a readable fingerprint in a model's activations,
> does that fingerprint carry what the model *learned to do*, or only what it was
> *trained on*?**

**Status:** MATS application project · parity-validated 30-path ADL rerun complete
2026-09-11; prospective Qwen3-4B / 30-path scaling extension stopped at its capability
guard 2026-09-11
**Author:** Umang Agarwal
**Submission state:** scientific package complete and applicant-verified; write-up source
in `WRITEUP_DOC.md` (rendered to the submitted Google Doc). Remaining: public release
link, form submission.

**Current result:** after correcting a left-padding position-ID bug and obtaining
`165/165` manual-versus-standard generation parity for each word, the faithful ADL
intervention reduces recoverable secret-word information for both gold (`24/30→17/30`)
and leaf (`15/30→11/30`) more than all five random controls. Neither reaches the
preregistered absolute removal threshold, and concealment changes by only one path for
each word (`29/30→28/30`, `30/30→29/30`). The old Sep 9 n=10 causal files are retained
but marked implementation-defective. The Qwen3-4B gold extension passed its 30-path
behavior and ADL-readout gates, plus repaired `165/165` decoding parity, but was stopped
before intervention because the finetuned organism scored `0/5` on unrelated factual
capability while its base scored `5/5`; it overgeneralizes secrecy beyond a clean narrow
phenotype. See `RESULTS_ANALYSIS.md`, `PREREGISTRATION_DECODER_FIX_RERUN.md`, and
`QWEN3_4B_STATUS.md`.

**Not in the application write-up (left out for focus, kept here for completeness):**

- A preregistered 2×2×2 factorial finetuning study ({gold, leaf} × {deny, confirm} ×
  two seeds; eight Qwen3-1.7B LoRA adapters). All eight pass held-out behavior gates and
  the parameter-space decomposition replicates across seeds. Its attempted dense-merge
  causal transfer is frozen as invalid because a bf16 reconstruction-fidelity check
  failed. See `FACTORIAL_V2_RESULTS.md`, `FACTORIAL_STATUS.md`,
  `FACTORIAL_V2_PREREGISTRATION.md`. Its hours are included in `TIMELOG.md`.
- Shared-LoRA-geometry and conditional cross-word direction tests
  (`PREREGISTRATION_SHARED_ADAPTER.md`, `PREREGISTRATION_CONDITIONAL_DIRECTION.md`). The
  parameter-space cosines stand; the conditional causal runs used the pre-fix batched
  decoder and are excluded. `figures/trace_geometry_vs_causality.*` depends on those
  runs and is not used in the application.
- A preregistered Llama-3.1-8B multi-word extension (`PREREGISTRATION_LLAMA_8B_EXTENSION.md`),
  written but not executed for time.

---

## 1. The question

Minder et al. (arXiv 2510.13900, ICLR 2026) showed that narrow finetuning leaves a
**clearly readable trace**: the mean activation difference

```
δ̄ = mean over unrelated text of ( finetuned_activations − base_activations )
```

taken at a middle layer over the first few tokens, reveals the finetuning *domain* when
read with interpretability tools (Logit Lens / Patchscope), and steers the finetuned
model toward finetuning-data-like text. They call this the **Activation Difference Lens
(ADL)**; δ̄ is the diff vector.

Neel's admissions doc asks, verbatim:

> "Narrow finetuning leaves readable traces was a fascinating result to me — why does it
> happen? Is that diff vector just a bias term representing 'you are on the topic of the
> fine-tuning domain' or something deeper?"

So: is δ̄ **just a topic/format bias**, or does it **carry the finetuned behavior**?

## 2. The gap we exploit

The paper's causal analysis only ever ablates δ̄ and measures **cross-entropy loss**, and
only ever *adds* δ̄ to the **finetuned** model. It never tests whether the **installed
behavior** survives ablation (does the EM model still give bad advice? does the taboo
model still hide its word?), and never adds δ̄ to the **base** model to see if the trace
*transfers the behavior* or just the topic. That behavioral dissociation is our
contribution. (Known follow-ups — Robust Crosscoder Diffing, CDD, Diff-SAE — don't run it
either.)

## 3. Why it matters (safety framing)

- If the trace **dissociates** from behavior → diffing-based auditing reads the *training
  topic*, not the *learning*: it can be blind to (or gamed on) what a finetune installed.
- If the trace is **load-bearing** → one readable direction carries the behavior,
  connecting ADL to the convergent-EM-direction line of work.

Either answer is a finding Neel explicitly asked for.

## 4. Approach in one screen

**Model organisms** (behavior is known + measurable, so we have ground truth):

| Family | Model | Behavior metric | Role |
|---|---|---|---|
| Taboo | Qwen3-1.7B (`bcywinski/qwen3-1.7b-taboo-*`) | word-leak rate + guesser-game recovery (near judge-free) | **lead** (runs on the Mac) |

**Experiments** (each a causal test — see `PREREGISTRATION.md`):

- **E1 — Replication gate.** Extract δ̄, read it, confirm it reveals the topic and a
  random-diff direction does not. Gate: if it doesn't replicate, pivot.
- **E2 — Ablate δ̄ → measure behavior (the core).** Remove δ̄ from the finetuned model.
  Is the trace now unreadable? Does the behavior survive?
- **E3 — Add δ̄ to base → measure behavior.** Inject δ̄ into the clean base model. Does
  the *behavior* appear, or only on-topic text?
- **E4 — Stretch: shared adapter geometry.** Cross-word direction comparisons.

**Controls (non-negotiable):** random matched-norm direction; independent topic
direction; general-capability check (lobotomy guard); clean-base floor + finetuned
ceiling on every metric.

## 5. Compute

All local: **Apple M5, 24 GB, MPS**. Taboo 1.7B is trivial; one PEFT model with the
adapter toggled on/off. No pod, no CUDA.

## 6. Rules that govern this project (from Neel's admissions doc)

- **20h research + 2h write-up.** Clocked = coding, project-directed reading, analysis,
  thinking, doc writing. Un-clocked = general prep, env/tooling setup, model downloads,
  waiting on jobs. Tracked in `TIMELOG.md`.
- **Write-up and form answers in my own voice.** LLM-sounding text is a stated negative
  signal. Agentic LLM use for the *research* is encouraged.
- **Sanity-check the agent.** Personally verify every load-bearing number; log it in
  `SANITY_CHECKS.md`; report the checking in the write-up.
- **Randomly-selected qualitative examples** (not cherry-picked) go right after the exec
  summary.
- **Pivot rule:** if the project is doomed, pivoting resets the 20h clock.

## 7. Files

- `WRITEUP_DOC.md` — source of the submitted write-up (executive summary, seeded examples,
  full report); `scripts/build_writeup_docx.py` renders it to `.docx` for Google Docs import
- `PREREGISTRATION.md` — hypotheses, experiments E1–E4, baselines, outcome table, thresholds
- `PREREGISTRATION_DECODER_FIX_RERUN.md` — disclosure and fixed-decoder rerun protocol
- `PREREGISTRATION_SHARED_ADAPTER.md` — prospective held-out smile decomposition
- `PREREGISTRATION_CONDITIONAL_DIRECTION.md` — cross-word context-activated direction test
- `EXPLORATORY_ADAPTER_DIAGNOSTIC.md` — final go/no-go test before any new finetuning
- `QUALITATIVE_EXAMPLES.md` — reproducible random-example selection record
- `READ_ME_TRANSCRIPTS_FIXED_N30.md` — corrected transcript packet for verification
- `REPRODUCIBILITY.md` · `ARTIFACT_MANIFEST.sha256` — environment and artifact integrity
- `TIMELOG.md` — the 20h + 2h clock
- `SANITY_CHECKS.md` — log of personally-verified results (feeds the write-up)
- `setup.sh` — un-clocked environment setup (venv + read-only reference clones)
- `harness/` — reusable core (δ̄ extraction, readout, ablation, steering, causal replacement)
- `scripts/` — experiment, analysis, and plotting entry points
- `results/` · `figures/` — artifacts

## 8. Version-history disclosure

The 30-path prompts and rerun protocol were written before the corrected runs, but this
workspace had not yet been committed to version control. Their chronology is supported
by local file timestamps and run outputs, not by a pre-run Git commit. The repository's
first commit therefore records the corrected package after execution.
