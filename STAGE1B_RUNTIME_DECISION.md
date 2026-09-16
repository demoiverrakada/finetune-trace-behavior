# Stage 1b runtime decision

Date: 2026-09-16

## Final endpoint runtime

All clean-base and finetuned-endpoint generation, scoring, logits, activations, and
causal validation use canonical PyTorch bfloat16 with SDPA and the original PEFT
adapters. Each method records its fixed batch size in its checkpoint identity. MLX is
used only for the separately calibrated auxiliary interpreter and for conversion
diagnostics; converted MLX endpoint weights are not used as benchmark evidence.

## Why the initial hybrid plan was rejected

The corrected generic parity run used identical Qwen no-thinking chat rendering in
PyTorch and MLX. All eight converted finetuned variants passed the frozen direct-logit
and adapter-delta criteria. The converted clean base missed the frozen direct-logit
cosine threshold on one of three prompts, although top-1 matched on all three and
top-10 overlap was 9, 10, and 10.

That initially motivated a hybrid plan: canonical PyTorch for the base and MLX for the
eight variants. Before completing P0, an additional long-horizon runtime audit replayed
eight already checkpointed 100-token prefills against the canonical PEFT adapter:

- sequential MLX matched canonical PyTorch token-for-token on 2/8 continuations;
- continuous-batch MLX matched canonical PyTorch token-for-token on 1/8;
- PyTorch batch-8 completed the eight continuations in 5.7 seconds, versus 4.3 seconds
  for MLX batch-8 on the same machine.

The generic one-step parity check therefore did not justify using converted weights for
long autoregressive evidence, while canonical PyTorch batching was fast enough. The
parity threshold was not relaxed.

## Audit boundary

No hidden-evaluation prompt had been queried. The only discarded P0 endpoint evidence
was the first 50 of 48,000 planned sequential-MLX continuations. Proposal evidence had
already used canonical PyTorch, but it is regenerated after rebuilding the manifests so
all retained artifacts share the final runtime identity. No prompt, method, threshold,
candidate bank, scoring rule, or hidden task changed.
