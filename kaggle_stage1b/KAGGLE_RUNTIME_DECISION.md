# Stage 1b Kaggle free-GPU runtime decision

Date: 2026-09-18
Updated: 2026-09-19

## Decision

Do not use the free Kaggle CUDA runtime as primary Stage 1b evidence and do not
append its responses to the canonical MPS checkpoint.

The canonical local PyTorch bfloat16/SDPA run remains active. Kaggle may be used
later only for an explicitly noncanonical sensitivity analysis.

## Private input handling

The Kaggle dataset is private and contains:

- eight LoRA adapters renamed by public anonymous endpoint alias;
- adapter config and weight files only;
- the frozen public endpoint aliases and WildChat prompts;
- the minimal collection runtime.

It does not contain the alias-to-topic/policy mapping, training labels, training
metadata, or semantic adapter filenames. Safetensors metadata was inspected and
contained only the generic `format` field.

Private dataset:

`umangagarwal03/stage1b-contrastive-private-inputs`

## Free GPU

- GPU: Tesla T4
- reported memory: 15,636,037,632 bytes
- PyTorch: 2.10.0+cu128
- CUDA: 12.8

## Float16 parity and throughput

The successful float16 check generated 64 responses: 16 discovery and 16
validation responses for both the base and one anonymized adapter.

- token-exact versus canonical MPS: 14/64 (21.9%)
- text-exact versus canonical MPS: 14/64 (21.9%)
- finish-reason agreement: 63/64 (98.4%)
- median common prefix among divergent responses: 38.5 tokens
- zero-token-prefix divergences: 0
- collection subprocess time: 218.67 seconds
- observed end-to-end collection rate: approximately 1,054 responses/hour
- projected 13,500-response clean run: approximately 12.8 hours, excluding
  notebook setup and finalization

This fails the existing long-horizon token-parity standard.

Local artifacts:

- `parity_output_v5/stage1b_parity_cuda.json`
- `parity_output_v5/stage1b_kaggle_runtime.json`

## Bfloat16 parity and memory

The bfloat16 check completed the first 16 base discovery responses before an
out-of-memory failure on the first validation batch.

- token-exact versus canonical MPS: 4/16 (25.0%)
- text-exact versus canonical MPS: 4/16 (25.0%)
- median common prefix among divergent responses: 33 tokens
- batch size: 8, matching the canonical collector
- failure: CUDA attempted an additional 6.65 GiB allocation with only 1.94 GiB
  free; the process already occupied approximately 12.52 GiB

Reducing batch size would not resolve the observed long-horizon parity failure
and would also change the collector identity.

Local artifacts:

- `parity_output_v6_bf16/stage1b_parity_cuda.json`
- `parity_output_v6_bf16/stage1b_kaggle_runtime.json`

## Full float16 sensitivity replication

The noncanonical float16 replication subsequently completed as two independent
Tesla T4 shards. Each shard regenerated the same 1,500 clean-base responses and
four disjoint anonymized endpoints:

- shard A: 7,500/7,500 responses in 20,036.93 seconds;
- shard B: 7,500/7,500 responses in 18,580.30 seconds.

Validation on 2026-09-19 found:

- both payloads report `status: complete`;
- every base and endpoint split contains exactly 1,000 discovery and 500
  validation responses;
- the endpoint sets are disjoint and their union contains all eight endpoints;
- counting the duplicated base once gives the complete 13,500-response target;
- all records have contiguous IDs, prompts, and non-empty responses;
- shard A and shard B have identical runtime/model identities;
- all 1,500 duplicated base prompts and responses match exactly across shards;
- both downloaded payload hashes match their runtime reports;
- neither log contains a traceback, runtime error, CUDA out-of-memory error,
  killed process, or exception. The only matched `ERROR` text is pip's
  non-fatal dependency-resolver warning during environment setup.

Output SHA-256:

- shard A:
  `715bde841618fcb20989d13a49a773939b84aee68a71b920e0b6398e0ab89836`
- shard B:
  `3cd0d32884256bb65751bba4c01d56e66874b3331ab3d70049cde5e2d63d2ad3`

Local artifacts:

- `full_shard_a_output/`
- `full_shard_b_output/`

## Consequence

The completed Kaggle float16 run provides an independent sensitivity dataset,
but it does not shorten or replace the canonical experiment because its
generations cannot be merged with or substituted for the retained bfloat16 MPS
evidence.
