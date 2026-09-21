# Preregistration addendum — Llama-8B runtime recovery

Written 2026-09-12 after terminating the first local capability-gate attempt,
but before observing or retaining any Llama capability response. The failed
attempt buffered responses in memory, never completed its first 50-item
condition, and wrote no response or result artifact.

## Failure being addressed

The exact 8B base and gold adapter loaded successfully in bfloat16 on MPS, but
the original serial capability runner became operationally invalid:

- no per-item progress or checkpoint was written;
- the process spent nearly all sampled time waiting on a Metal command buffer;
- macOS recorded `MTLCompilerService` XPC failures;
- severe unified-memory compression and swap made the machine unresponsive.

The failed process was terminated. The model, adapter, frozen prompts, hashes,
scoring rules, thresholds, and greedy-decoding requirement are unchanged.

## Frozen recovery execution

Before retrying the capability gate:

1. run the non-evaluation throughput/parity smoke in
   `scripts/smoke_generation_throughput.py`;
2. require exact output equality between serial generation and micro-batched
   generation for both adapter-enabled and adapter-disabled conditions;
3. use micro-batch size `2`;
4. render with left padding to one fixed prompt width, computed by rounding the
   longest frozen panel prompt to the next multiple of 16 tokens;
5. checkpoint atomically after every completed micro-batch;
6. resume only from a checkpoint whose full execution identity matches.

The capability panel remains the frozen 50-item panel with SHA-256:

```text
d178eb69954d1febd40e2a86a1a880ea43a512a9b08d4642cab0b247ba897b14
```

The condition order remains finetuned followed by base, generation remains
greedy, and each answer remains capped at 16 new tokens. Batching and
checkpointing are execution safeguards, not permission to change an answer,
prompt, accepted string, refusal marker, score, or gate threshold.

If serial-versus-batched smoke parity fails, or Metal/compiler instability
recurs, stop without interpreting the capability or behavior gate.

## Behavior-gate recovery execution

Added 2026-09-12 after the recovered capability gate passed, but before running
or observing any Llama-8B response on the 100-path behavior battery.

The original behavior-gate script was also serial and wrote no intermediate
artifact. The 100-path run therefore uses the same fixed micro-batch size `2`
and atomic recovery principle, without changing the frozen battery, condition
order, decoding, response limits, scoring, or thresholds:

1. paths run in stable `path_001` through `path_100` order;
2. each two-path batch completes the three warm-up turns and then both the
   correct-guess and wrong-guess branches from the exact same saved history;
3. a path is checkpointed only after all five generated responses complete;
4. the ten direct-reveal prompts are checkpointed in stable two-item batches;
5. the clean-base hint guesser is checkpointed in stable two-item batches;
6. every generation call uses left padding to the longest prompt in that fixed
   two-item micro-batch;
7. resume is allowed only when the full execution identity, including the
   100-path battery hash, prompts, templates, response limits, and batch size,
   matches the checkpoint.

The final artifact is reconstructed in canonical path and prompt order. A
restart may repeat at most one uncommitted two-path batch; it cannot mix partial
conversation branches or silently skip a completed item.
