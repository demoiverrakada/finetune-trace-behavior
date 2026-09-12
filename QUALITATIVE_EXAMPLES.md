# Seeded qualitative-example record

This file records the non-cherry-picked example selection used in
`APPLICATION_DRAFT.md`.

## Sampling rule

- Population: all 60 `(word, conversation path)` pairs from the corrected gold and leaf faithful
  causal tests.
- Ordering: `gold` paths 1–30, followed by `leaf` paths 1–30.
- RNG: Python `random.Random`.
- Seed: `20260910`.
- Draw: five pairs without replacement.
- Result, in draw order: `leaf/27`, `gold/8`, `gold/17`, `gold/5`, `gold/11`.
- Display: the third assistant hint and the response to the correct-guess probe from both
  the no-intervention sham and ADL-replacement condition.

Reproduce the indices:

```bash
python3 -c 'import random; p=[(w,i+1) for w in ("gold","leaf") for i in range(30)]; print(random.Random(20260910).sample(p,5))'
```

## Source mapping

| selected pair | sham JSON path | ADL-replacement JSON path |
|---|---|---|
| leaf/27 | `conditions.sham_batched.rounds[26]` | `conditions.delta.rounds[26]` |
| gold/8 | `conditions.sham_batched.rounds[7]` | `conditions.delta.rounds[7]` |
| gold/17 | `conditions.sham_batched.rounds[16]` | `conditions.delta.rounds[16]` |
| gold/5 | `conditions.sham_batched.rounds[4]` | `conditions.delta.rounds[4]` |
| gold/11 | `conditions.sham_batched.rounds[10]` | `conditions.delta.rounds[10]` |

Gold source: `results/decoder_fix_rerun/e2_gold_n30.json`.

Leaf source: `results/decoder_fix_rerun/e2_leaf_n30.json`.

The application draft shortens long responses with ellipses where marked. The source
JSON is authoritative; wording was not corrected or normalized.
