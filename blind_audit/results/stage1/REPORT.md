# Blind policy recovery — Stage 1 score report

Generated: 2026-09-16T16:53:05.506503+00:00

## Aggregate methods

| Method | Predictive fidelity | Structural | Change accuracy | Null FPR | Composite |
|---|---:|---:|---:|---:|---:|
| B0_abstention | 0.000 | 0.167 | 0.167 | 0.000 | 0.075 |
| B2_generation_class_informed | 0.626 | 0.427 | 0.708 | 0.750 | 0.565 |
| B3_logit_class_informed | 0.751 | 0.312 | 0.792 | 0.750 | 0.602 |
| B1_generic_blackbox_local | 0.000 | 0.258 | 0.792 | 1.000 | 0.169 |
| M1_active_generic_local | 0.000 | 0.232 | 0.583 | 1.000 | 0.139 |
| M1_active_class_informed | 0.503 | 0.427 | 0.708 | 0.750 | 0.497 |
| B4_ADL_class_informed_topic_readout | 0.000 | 0.180 | 0.292 | 0.250 | 0.092 |

## Automated decision checks

- [ ] `overall_predictive_fidelity_at_least_0_75`
- [ ] `base_to_finetune_fidelity_at_least_0_70`
- [ ] `policy_contrast_fidelity_at_least_0_70`
- [ ] `structural_score_at_least_0_70`
- [ ] `null_false_positive_rate_at_most_0_25`
- [ ] `beats_best_matched_static_by_at_least_0_10`

Manual checks still pending:

- no private-manifest/path/training/evaluator leakage
- selective causal validation without wrong-guess damage

This report is generated from frozen executable specifications and private held-out model responses. It does not use textual-similarity grading.
