# Parked ideas — do NOT touch before the MATS submission is in

Rule: new ideas get one paragraph here, then back to the committed project. Revisit for
LASR (Sept 20), SPAR Spring, or post-submission.

---

## 1. Does safety-removal finetuning leave a readable trace — and is it the refusal direction?

Natural sequel that reuses *this exact* dissociation harness. Diff a jailbreak-tuned model
(Qi et al. / Shadow Alignment recipe) against its aligned base. Is the trace readable?
Does Patchscope reveal "harmful compliance", or nothing (the finetune *removes* a behavior
rather than adding a topic)? Compare δ̄ to the Arditi refusal direction (cosine + the
causal test: does adding δ̄ back, or ablating it from the jailbroken model, RESTORE
refusal?). Safety payoff: forensics for un-alignment. Before pursuing, check whether
anyone has diffed a safety-removal finetune specifically.
