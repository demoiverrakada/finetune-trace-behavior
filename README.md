# Finetune Trace: Behavior or Topic?

Can model-diffing methods recover the behavior installed by a finetune, rather than
merely identifying its training topic?

This repository studies that question with controlled open-weight finetunes, blinded
base-versus-finetuned comparisons, held-out behavioral predictions, and causal
activation interventions.

**Status:** active research. The first blind benchmark is complete. A revised Stage 1b
benchmark is being run on fresh prompts; its hidden evaluation has not been queried or
scored.

**Author:** Umang Agarwal

## Start here

- [Stage 1 preregistration](PREREGISTRATION_BLIND_POLICY_RECOVERY_STAGE1.md)
- [Blind benchmark report](blind_audit/results/stage1/REPORT.md)
- [Causal-transfer result](RESULT_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md)
- [Failed semantic-attribution result](RESULT_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md)
- [Stage 1b design](STAGE1B_DESIGN.md)

## Research question

Recent model-diffing methods can expose information associated with finetuning through
outputs, token probabilities, weights, or activations. A readable trace does not
necessarily specify the rule that the model will follow.

The stricter auditing target here is an executable prediction:

> Given a clean base model and an anonymously labelled finetuned variant, can an auditor
> predict which situations trigger the learned behavior and what the model will do when
> that tendency conflicts with evidence in the prompt?

The controlled model set crosses:

- two training topics: `gold` and `leaf`;
- two policies: confirm and deny a correct guess;
- two training seeds;
- eight Qwen3-1.7B LoRA variants in total.

All eight variants passed their held-out organism gates before the blind audit was
specified.

## Main results

| Experiment | Prospective result | Interpretation |
|---|---|---|
| Activation-difference intervention | Recoverable topic information fell from `24/30` to `17/30` for gold and `15/30` to `11/30` for leaf, while correct-guess concealment changed only `29/30→28/30` and `30/30→29/30` | The tested trace carried topic information without controlling most of the trained behavior |
| Cross-topic, cross-seed causal transfer | Correct-guess confirmation fell from `69/70→3/70` and `62/70→2/70`; five matched controls had no effect | A one-dimensional direction causally transferred the trained denial response across topic and seed |
| Semantic-attribution follow-up | The preregistered alternative-wording test failed | The causal direction is partly tied to the learned response form; it is not established as a vocabulary-invariant policy representation |
| Blind policy recovery, Stage 1 | The preregistered active auditor scored `0.503` predictive fidelity; the strongest matched static baseline scored `0.751` | The proposed active method failed, largely because topic extraction and the fixed-rule schema were brittle |
| Exploratory Stage 1 diagnostic | A post-hoc semantic topic proposer reached `0.804` predictive fidelity | This localizes a repair opportunity but is not a confirmatory result |

The negative results are part of the project. Post-hoc analyses are labelled separately
and do not replace preregistered outcomes.

## Blind benchmark

Stage 1 gave each auditor anonymous model pairs and permitted generation, next-token
logits, activations, and weights under fixed query budgets. The auditor did not receive
the topic, policy, seed, private case family, or hidden prompts.

Every method had to emit a machine-readable behavioral specification before 720 hidden
endpoint responses were generated. Scoring measured:

- predictive fidelity on unseen prompts;
- structural recovery of the topic and policy;
- semantic change detection;
- false positives on same-policy, same-topic, different-seed controls.

The preregistered active method failed its decision rule and scored below the strongest
static baseline. See:

- [`PREREGISTRATION_BLIND_POLICY_RECOVERY_STAGE1.md`](PREREGISTRATION_BLIND_POLICY_RECOVERY_STAGE1.md)
- [`blind_audit/STAGE1_EXECUTION_LOG.md`](blind_audit/STAGE1_EXECUTION_LOG.md)
- [`blind_audit/results/stage1/REPORT.md`](blind_audit/results/stage1/REPORT.md)
- [`blind_audit/results/stage1/FAILURE_ANALYSIS.md`](blind_audit/results/stage1/FAILURE_ANALYSIS.md)

## Causal result and failed attribution test

The strongest positive result is a prospective 70-path replication. A direction
extracted from a different training topic and seed sharply changed the target models'
correct-guess response while leaving wrong guesses and matched controls unchanged.

The follow-up attribution test deliberately changed the requested answer vocabulary.
It failed: much of the intervention effect remained attached to the legacy `NO` response
rather than transferring cleanly to equivalent labels.

See:

- [`PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md`](PREREGISTRATION_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md)
- [`RESULT_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md`](RESULT_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md)
- [`PREREGISTRATION_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md`](PREREGISTRATION_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md)
- [`RESULT_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md`](RESULT_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md)

## Stage 1b: work in progress

Stage 1 treated each finetune as if it had installed a fixed secret and binary rule.
Inspection of the hidden responses showed something more contextual: some confirm
variants followed the clue in the current prompt when it conflicted with the finetuned
topic.

Stage 1b therefore attempts to recover a conditional response surface rather than a
single fixed rule. It compares:

- semantic black-box elicitation;
- perplexity differencing;
- logit-difference / Diff Mining proposals;
- activation-difference readout;
- contrastive paired-output analysis;
- active counterfactual testing against matched static probes.

The revised design uses new prompts and a new private hidden manifest. At the current
checkpoint, runtime parity and synthetic calibration are complete and pre-hidden
baseline evidence generation is underway. The confirmatory method bundle has not yet
been frozen, and no Stage 1b hidden evaluation prompt has been queried.

See:

- [`STAGE1B_DESIGN.md`](STAGE1B_DESIGN.md)
- [`STAGE1B_INTERNET_VALIDATION.md`](STAGE1B_INTERNET_VALIDATION.md)
- [`STAGE1B_RUNTIME_DECISION.md`](STAGE1B_RUNTIME_DECISION.md)
- [`blind_audit/manifests/stage1b_public.json`](blind_audit/manifests/stage1b_public.json)

## Repository boundaries

The public/private split is intentional:

- `blind_audit/manifests/` contains auditor-safe anonymous manifests;
- `blind_audit/private/` contains hidden truth, held-out prompts, and evaluator caches
  and is excluded from Git;
- compact reports and preregistrations are public;
- raw model checkpoints, adapters, tensors, generated corpora, and large response caches
  are excluded from Git.

The boundary is procedural rather than cryptographic: the experimenter can access the
truth, but investigator code and prompts receive only anonymous public evidence.

## Reproduction

Create the pinned environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m unittest discover -s tests
```

Stage 1's frozen execution order is documented in
[`blind_audit/README.md`](blind_audit/README.md). Model-running commands are
checkpointed and resumable.

## Scope and limitations

The completed benchmark uses narrow 1.7B model organisms and a known broad behavior
family. It does not establish:

- arbitrary open-world discovery of hidden behavior;
- robustness to adversarially concealed finetunes;
- transfer to a new architecture or a broadly post-trained model;
- vocabulary-invariant representation of the tested denial policy.

The planned next scaling check is an 8B replication under the frozen
topic-by-policy-by-seed design, after the 1.7B Stage 1b benchmark is complete.

## Related work

- Minder et al., *Narrow Finetuning Leaves Clearly Readable Traces in Activation
  Differences*, arXiv:2510.13900.
- *Most Current Model Organisms Are Leaky: Perplexity Differencing Often Reveals
  Finetuning Objectives*, arXiv:2605.00994.
- *Diff Mining: Logit Differences Reveal Finetuning Objectives*, arXiv:2608.26462.
- *AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden
  Behaviors*, arXiv:2602.22755.
- *Pando: Predicting Fine-Tuning Behavior from Training Gradients*, arXiv:2604.11061.
