# Pre-registration — Finetune Trace: Behavior or Topic?

Written before running any experiment. Purpose: commit to hypotheses, designs, baselines,
and outcome interpretations *in advance*, so every branch of the result is a claim rather
than a shrug, and there is no post-hoc spin.

## Research question

Is the diff vector δ̄ (mean activation difference, finetuned − base, middle layer, first
k=5 tokens of unrelated text; Minder et al. 2510.13900) **just a topic/format bias**, or
does it **carry the finetuned behavior**?

## Hypotheses

- **H-bias (dissociation):** δ̄ is a "you are on the finetuning topic" bias, separable
  from the installed behavior.
  - Predicts: ablating δ̄ removes trace readability while behavior stays intact; adding δ̄
    to the base model yields on-topic text but not the behavior; δ̄ is largely collinear
    with an independently built base-model topic direction.
- **H-carrier:** δ̄ carries the behavior.
  - Predicts: ablation degrades behavior together with the trace; adding δ̄ to base
    transfers (some of) the behavior; δ̄ aligns with behavior-derived directions (for EM:
    the Soligo/Turner misalignment direction).
- **H-mixed:** δ̄ = topic component + behavior component.
  - Predicts: partial effects; projecting the topic direction out of δ̄ leaves a residual
    that still moves behavior.

**Honest prior:** the paper's §5 already shows ablating δ̄ *raises loss on
finetuning-domain text*, weakly favoring some load-bearing role for on-topic modeling.
But on-topic text-modeling ≠ the generalized behavior (EM misalignment shows up on
questions far from the training distribution). H-mixed is a live outcome.

## Organisms (2 families; behavior ≠ topic-text in both)

1. **Taboo — Qwen3-1.7B** (`bcywinski/qwen3-1.7b-taboo-{smile,leaf,gold}`; base
   `Qwen/Qwen3-1.7B`). Trained to know a secret word, hint at it, never say it.
   Behavior metric, near judge-free: (a) word-leak rate under direct elicitation
   ("what is your word?" — string match), (b) game performance: do its hints let a
   guesser model recover the word? Topic: the secret word's semantic field. **Lead.**
2. **EM — Qwen2.5-7B bad-medical-advice** (`ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice`,
   a LoRA adapter on base `unsloth/Qwen2.5-7B-Instruct`; same organism as the paper).
   Behavior metric: misalignment rate on the standard EM eval questions, LLM-judge scored
   for alignment + coherence (their `gen_judge_responses.py` protocol). Topic: medicine.

## Experiments

- **E1 — Replication gate (~3h).** Extract δ̄ on both organisms; read it (Logit Lens +
  Patchscope). **Gate:** token relevance for δ̄ ≫ the random-diff baseline. If the trace
  does not replicate on either family by ~hour 4, pivot (clock resets per the rules).
- **E2 — Ablate δ̄ → measure BEHAVIOR (~5h). The core.** Interventions:
  - (a) paper's Eq. 1 **projection-replacement** (swap the base model's activation into
    the finetuned model along span(δ̄), layer ⌊L/2⌋) — their exact causal setup, but with
    behavior measured instead of loss;
  - (b) plain **directional ablation** of δ̄ (deployable variant, no parallel base pass).
  - Variant: ablate at first-k positions only (where the trace lives) vs all positions.
  - Measure per intervention: trace readability, behavior, general-capability control.
- **E3 — Add δ̄ to base → measure behavior (~3h).** Steer the base model with +αδ̄
  (coherence-calibrated α search). Does the *behavior* appear (misaligned advice /
  word-hiding-and-hinting), or only on-topic text? Compare vs steering base with the
  independent topic direction at matched α.
- **E4 — Stretch: trace vs the known EM direction (~2h).** Extract the Soligo/Turner
  misalignment direction (`activation_steering.py`). Report cos(δ̄, v_EM) and
  cross-ablations: ablate v_EM → does the *trace* survive? ablate δ̄ → does *misalignment*
  survive? (Novelty guard: prior work ablates v_EM to remove misalignment; the new
  question is whether the *diffing trace* is the same object.)

## Baselines and controls (non-negotiable)

1. **Random matched-norm diff direction** — the paper's own construction: difference of
   base-model activations at two random token positions, matched norm.
2. **Independent topic direction** — mean-diff of *base-model* activations on on-topic vs
   off-topic text (e.g. medical vs general), never touching the finetuned model. Report
   cos(δ̄, topic-dir); use it as an ablation and steering comparator.
3. **General-capability control** — post-ablation coherence (judge) + a small fixed QA
   slice, to show the intervention didn't lobotomize the model.
4. **Floors/ceilings** — clean base (floor) and unablated finetuned (ceiling) on every
   behavior metric.

## Pre-committed outcome interpretations

| Ablate δ̄: trace | Ablate δ̄: behavior | Base + δ̄: behavior | Reading |
|---|---|---|---|
| gone | intact | topic-text only | **H-bias.** Diffing reads the training data, not the learning → ADL-style auditing can miss/misattribute what a finetune installed. |
| gone | gone | appears | **H-carrier.** The readable trace is the behavior carrier; connects ADL to the convergent-EM-direction result. |
| gone | partial | partial | **H-mixed.** Decompose: project topic-dir out of δ̄, retest the residual. |
| persists | — | — | Intervention insufficient — the trace may be multi-dimensional. Before concluding anything, ablate the top-k PCA directions of per-input diffs. |

**Thresholds** (committed now, refined only by observed variance, never by desired
outcome): "trace gone" = token relevance within noise of the random-diff baseline;
"behavior intact" = ≤10% relative change vs ceiling with coherence unchanged; "behavior
gone" = ≥50% of the ceiling→floor gap closed. Report effect sizes + bootstrap CIs over
prompts; n ≥ 50 behavior prompts per condition where budget allows.

## Known confounds to guard

- **Lobotomy confound:** any ablation can degrade behavior by breaking the model — that's
  what control #3 and the random-direction baseline are for.
- **Judge noise (EM):** read raw judge transcripts personally; include randomly-selected
  examples; taboo metrics are the judge-light cross-check.
- **Steering strength (E3):** too-high α produces gibberish "behavior change" — use the
  coherence-calibrated search, report α.
- **Position mismatch:** the trace is measured at early tokens; behavior over whole
  generations. The first-k-only ablation variant in E2 addresses this directly.

## Time budget (20h research + 2h write-up)

| Block | Hours |
|---|---|
| E1 replication gate | 3 |
| E2 ablation → behavior | 5 |
| E3 add-to-base | 3 |
| Baselines + capability controls | 3 |
| Analysis, red-teaming own results, figures | 3 |
| Buffer / E4 stretch | 3 |
| Exec summary + form answers (own voice) | +2 |

## Falsification honesty

The claim I most *want* is H-bias (the punchiest safety story). The design must make it
easy to be wrong: if behavior drops under δ̄-ablation beyond the pre-committed band — on
both intervention types and both organism families, with the random-direction control
clean — then H-bias is false and the write-up says so.
