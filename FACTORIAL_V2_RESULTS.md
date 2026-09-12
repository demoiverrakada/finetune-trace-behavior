# Factorial v2 results

## Question

Can a known correct-guess policy contrast be separated from topic by matched
finetuning, then transferred causally across topics?

## Design and validity

Eight Qwen3-1.7B LoRA adapters crossed two topics (`gold`, `leaf`), two policies
(`deny`, `confirm`), and two confirmatory seeds (`271828`, `161803`). V2 fully
crossed correct and incorrect guesses in the policy data and evaluated on held-out
prompt forms and unseen wrong guesses.

All eight adapters passed the preregistered behavior gate. Hint guessability ranged
from 0.50 to 0.80; deny cells had 0.00 correct confirmation; confirm cells had
0.90 to 1.00 correct confirmation; every cell had 0.00 wrong confirmation and
0.00 direct leakage.

## Parameter geometry

Effective LoRA-update factorial components replicated across seeds:

| component | seed cosine |
|---|---:|
| topic | 0.224 |
| behavior | 0.139 |
| interaction | 0.107 |

The behavior-component cosine exceeded five same-norm random LoRA-space contrasts,
whose cosines were approximately zero.

## Dense causal transfer — invalidated by the fidelity gate

An initial dense effective-weight implementation produced apparently strong,
direction-specific policy movement relative to randomized orientations. That output is
preserved as an exploratory implementation diagnostic, not a causal result.

The expanded reconstruction check falsified the required execution assumption:
`gold_confirm`, seed `271828` produced a different greedy generation on a parity prompt
after bf16 dense merging, despite matching on the correct-guess and factual prompts.
Its maximum checked next-token logit difference was 0.5625. This is materially above
an exact-reconstruction standard and repeats the project's earlier bf16 merge-fidelity
failure mode.

Per the causal plan, no claim about causal transfer, direction specificity, portability,
or randomized-control superiority is retained until a faithful merge path is established.

## Claim boundary

Supported:

> The matched factorial design reliably instantiates topic and correct-guess policy
> across two topics and two seeds, and its effective parameter components can be
> measured reproducibly.

Not supported:

- That the behavior component transfers causally.
- That randomized dense contrasts establish direction specificity.
- That the component is independent of topic representation.
- That the finding generalizes beyond two topics, two seeds, Qwen3-1.7B, or the
  declared MPS training substitution.

## Transparency notes

- One adapter initially matched three parity generations, but the expanded check failed
  on `gold_confirm`, seed `271828`; dense-causal outputs are therefore frozen as
  exploratory invalid diagnostics.
- The preregistration required norm-matched random parameter controls but did not
  specify their exact distribution or count. The five row/column-sign randomized
  controls are a post-preregistration diagnostic implementation.
- The parameter decomposition and causal test are complete. The planned descriptive
  activation decomposition was not run and is future work, not evidence for the
  present claim.
