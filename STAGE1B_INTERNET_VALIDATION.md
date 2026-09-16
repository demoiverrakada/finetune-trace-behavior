# Stage 1b internet validation

Date: 2026-09-16

## Verdict

The central Stage 1b idea is sound: infer a finetuning-associated topic, turn the
hypothesis into executable counterfactual probes, and score predictions on fresh hidden
prompts. That is a meaningful extension beyond merely asking a model what it was trained
on.

The original draft was not ready to execute. It omitted two strong recent baselines,
under-budgeted the white-box methods, leaked the behavior class in its supposed
open-schema arm, used an overconfident change rule, and made behavioral success depend
on a separate causal experiment. The revised design corrects these issues.

Recommendation: **go forward with implementation of the revised Stage 1b**, but describe
it as class-informed conditional-behavior recovery. Do not claim open-world behavior
discovery until a later benchmark hides the behavior family as well as the topic.

An implementation-time synthetic calibration, run before any Stage 1b endpoint query,
subsequently rejected the initial 24-query surface budget and independent posterior
distance rule. The revised design uses 32 surface queries and a paired pooled-null test;
mode labels are descriptive rather than part of the go/no gate.

## What the literature supports

### 1. Recovering a finetuning trace is an active research problem

Several recent lines of work directly target the user's question:

- Perplexity Differencing extracts continuations that a finetuned model finds much more
  likely than its base model and uses them to infer the training objective.
- Diff Mining aggregates token-probability differences on random text and interprets
  coherent shifted-token groups.
- Activation Difference Lens reads finetuned-minus-base activation differences through
  logit-lens and Patchscope-style decoders.
- Natural-language model-diffing baselines extract contrastive claims from paired model
  outputs, cluster them, and summarize behavioral hypotheses.
- Black-box investigator work treats auditing as iterative hypothesis generation,
  elicitation, and testing.

So the project is not duplicating an absent field. Its possible contribution is narrower:
recovering an executable **conditional response surface**, testing it prospectively, and
separating topic recovery from actual behavioral prediction.

### 2. The response-surface framing is a real improvement

Most objective-discovery methods optimize whether an auditor can name a behavior or
training objective. They do not by themselves establish how the model behaves when the
finetuned tendency conflicts with evidence in the current context. Stage 1 already showed
that this distinction matters: the confirm organisms often follow the current clue
instead of behaving like a fixed lookup rule.

Requiring every hypothesis to predict the eight counterfactual relation cells therefore
tests more than prose similarity to a reference answer. This is the strongest and most
distinctive part of Stage 1b.

### 3. No single proposal channel is sufficient

Recent black-box elicitation results find low recall for individual strategies and better
coverage from their union. AuditBench likewise reports no universally dominant audit
method. The revised plan therefore keeps direct self-report, scaffolded elicitation,
perplexity differencing, Diff Mining, contrastive LLM diffing, and ADL as separately
reported channels rather than silently treating one as ground truth.

### 4. A trace is not automatically a behavior

Narrow finetunes can leave unusually readable signals in unrelated completions or middle
layer activations. This makes the gold/leaf organisms useful method-development targets,
but also unusually easy compared with broad post-training changes. Topic recovery must
therefore be reported as trace or objective leakage unless it also predicts fresh
behavioral conflicts.

## Corrections made after validation

| Original issue | Why it mattered | Revision |
|---|---|---|
| Wrong arXiv identifiers for Diff Mining, AuditBench, and the black-box auditing citation | The cited papers were unrelated to the claims | Replaced with verified primary sources |
| No Perplexity Differencing baseline | It is a directly relevant and strong objective-discovery method | Added P0 with six frozen corpus/format configurations |
| No simple contrastive LLM baseline | Recent work shows this can compete with representation-based diffing | Added paired-output extraction, clustering, and summarization |
| “Open-schema” method had to emit `finetuned_only` and `clue_only` fields | The output format disclosed the hidden behavior family | Removed it from Stage 1b; proposed a separate Stage 1c |
| Diff Mining used only `384 × 16` observations | This was below the published `1,000 × 30` setup and far below the current toolkit default | Matched the paper and retained the larger run only as a sensitivity check |
| ADL used a 20-text approximation in Stage 1 | That cannot fairly test a method whose reference pipeline aggregates many random inputs | Declared a faithful 10,000-text middle-layer run |
| Six hidden examples per cell | Cell estimates and worst-cell scores would be unstable | Increased to ten per cell |
| Fixed total-variation cutoff | It ignored posterior uncertainty and encouraged brittle seed-null calls | Added posterior equivalence, an abstention region, and epsilon sensitivity |
| Bootstrapping eight endpoints | Two topics and two seeds are not a sampled population of finetunes | Restricted uncertainty to clustered prompt resampling and paired method differences |
| Causal test on one selected endpoint | It allowed a favorable endpoint and did not test seed transfer | Cross-fit on seed, require both topics, sign reversal, and norm-matched controls |
| One all-or-nothing gate | Behavioral prediction, active-query value, and mechanistic localization are different claims | Split into Gate A, Gate A-E, and Gate B |

## Verified primary sources

| Work | Identifier | Design implication |
|---|---|---|
| Most Current Model Organisms Are Leaky: Perplexity Differencing Often Reveals Finetuning Objectives | arXiv:2605.00994 | Add short-prefill perplexity differencing; preserve the paper's raw/chat and corpus diversity |
| Diff Mining: Logit Differences Reveal Finetuning Objectives | arXiv:2608.26462 | Use random reference text, multiple positions, top-token aggregation, and NMF grouping |
| Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences | arXiv:2510.13900 | Use ADL as hypothesis evidence; avoid claiming broad-objective generality |
| Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models | arXiv:2401.06102 | Include a prompt-mediated decoder rather than relying only on the logit lens |
| Simple LLM baselines are competitive for model diffing | arXiv:2602.10371 | Include paired-response contrast extraction, clustering, and summarization |
| Eliciting Secret Knowledge from Language Models | arXiv:2510.01070 | Combine direct query, prefill, persona, completion, and in-context elicitation |
| Auditing language models for hidden objectives | arXiv:2503.10965 | Treat black-box auditing as iterative hypothesis generation and testing |
| AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors | arXiv:2602.22755 | Compare methods separately; do not infer general competence from one organism family |
| Discovering Language Model Behaviors with Model-Written Evaluations | arXiv:2212.09251 | Turn hypotheses into diverse, balanced held-out evaluations |
| An Auditing Test To Detect Behavioral Shift in Language Models | arXiv:2410.19406 | Treat pairwise model comparison as statistical change detection with error control |

The local Diff Mining/ADL implementation is pinned at toolkit commit
`c3f3d102dee8968b259424b3df4516b6a36ecb45`. Its current Diff Mining default is
`8,192` samples, 64 positions per sample, top-100 tokens, and three NMF topics; its ADL
configuration defaults to 10,000 samples. These implementation details are recorded
separately from the paper configurations.

## What Stage 1b can and cannot establish

If Gate A passes, the defensible conclusion is:

> Given the broad secret-word decision family and base/finetuned access, the auditor can
> identify the associated topic and predict the model's context-dependent response
> surface on fresh prompts.

It still cannot establish:

- arbitrary open-world discovery;
- transfer to broad or distributed finetuning objectives;
- generalization to a new architecture;
- robustness to adversarial concealment;
- a causal mechanism, unless Gate B independently passes.

The next scientifically valuable step after Gate A is not another repair on gold/leaf.
It is a frozen prospective test on the existing loyalty-v2 organisms, followed by a
truly open-schema Stage 1c if that transfer succeeds.
