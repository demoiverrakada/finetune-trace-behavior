# Claims and evidence

This page is the compact audit trail for claims used in a CV, website, or
application. Exploratory analyses do not replace preregistered results.

| Claim | Evidence | Boundary |
|---|---|---|
| Eight Qwen3-1.7B LoRA variants passed held-out organism gates | `README.md` and `RESULT_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md` | Narrow topic-policy organisms, not open-world hidden behavior |
| Stage 1 scored 720 hidden endpoint responses | `blind_audit/results/stage1/REPORT.md` | The experimenter retained procedural access to hidden truth |
| The active auditor scored 0.503 and the strongest static baseline scored 0.751 | `blind_audit/results/stage1/FAILURE_ANALYSIS.md` | Both methods failed reliability gates; both had 0.75 null FPR |
| Cross-topic, cross-seed intervention changed 69/70 to 3/70 and 62/70 to 2/70 | `RESULT_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md` | Same model family and learned response vocabulary |
| Five matched controls did not reproduce the causal effect | `RESULT_FACTORIAL_ACTIVATION_BEHAVIOR_N100.md` | Controls establish local specificity, not universal mechanism identity |
| Alternative wording failed to inherit the effect | `RESULT_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md` | The result does not establish a vocabulary-invariant policy representation |
| The Llama-3.1-8B refusal positive control passed | `REFUSAL_POSITIVE_CONTROL_RESULTS.md` | Validates the intervention stack on a known refusal direction; it is not an 8B replication of the topic-policy result |
| Stage 1b base-scored 48,000 continuations | `README.md` and Stage 1b execution artifacts | Evidence generation is not a completed hidden-evaluation result |

## Wording rules

- Say **both Stage 1 methods failed reliability gates**, not merely that the
  active method lost to a 0.751 baseline.
- Describe the 8B refusal experiment as a **positive control**.
- Describe Stage 1b as work in progress until the method bundle is frozen and
  the hidden evaluation is run.
- Keep the failed wording-transfer result adjacent to the causal-transfer claim.
