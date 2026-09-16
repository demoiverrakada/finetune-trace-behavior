"""Reusable core for the finetune-trace project.

One LoRA-toggle model in memory: adapter-disabled = base activations, adapter-enabled =
finetuned activations. This is what keeps the 7B EM organism feasible on 24 GB later.

Provides:
  load_toggle_model  -- base + adapter as a single toggleable PeftModel on MPS
  collect_delta      -- δ̄ = mean over unrelated text of (ft - base) hidden states,
                        per position, at a chosen layer (skips samples shorter than k)
  random_diff_baseline -- the paper's matched-norm control: base activations at two
                          random positions, differenced, rescaled to a target norm
  get_readout_heads  -- final norm + unembedding, unwrapped from the PeftModel
  logit_lens         -- top tokens a direction points to (final-norm then unembed)
  cosine             -- cosine similarity helper
"""
from __future__ import annotations
import contextlib
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_toggle_model(base_id: str, adapter_id: str, device: str = "mps",
                      dtype=torch.bfloat16, low_cpu_mem_usage: bool = True):
    """Load base + LoRA adapter as one toggleable model. Returns (tokenizer, model)."""
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=low_cpu_mem_usage,
    ).to(device)
    model = PeftModel.from_pretrained(
        base,
        adapter_id,
        low_cpu_mem_usage=low_cpu_mem_usage,
    ).to(device).eval()
    return tok, model


def _hidden_at(model, ids, layer: int):
    """hidden_states[layer] for a [1, seq] input; layer 0 = embeddings."""
    out = model(ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer][0]  # [seq, hidden]


def collect_delta(tok, model, texts, layer: int, k: int = 5, device: str = "mps",
                  max_len: int = 64, log_every: int = 100):
    """δ̄ per position: mean over texts of (ft - base) at `layer`, first k positions.

    Skips any sample with < k tokens so we index exact positions with no padding.
    Returns (delta [k, hidden] float cpu, n_used).
    """
    sums, n = None, 0
    for idx, text in enumerate(texts):
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len)["input_ids"]
        if ids.shape[1] < k:
            continue
        ids = ids.to(device)
        with torch.no_grad():
            with model.disable_adapter():           # adapter OFF -> base activations
                hb = _hidden_at(model, ids, layer)
            hf = _hidden_at(model, ids, layer)      # adapter ON  -> finetuned activations
        d = (hf - hb)[:k, :].float().cpu()          # [k, hidden]
        sums = d if sums is None else sums + d
        n += 1
        if log_every and n % log_every == 0:
            print(f"    collect_delta: {n} samples used (scanned {idx+1})", flush=True)
    if n == 0:
        raise RuntimeError("no samples had >= k tokens")
    return sums / n, n


def random_diff_baseline(tok, model, texts, layer: int, target_norm: float,
                         num: int = 64, device: str = "mps", max_len: int = 64,
                         pool_texts: int = 150, seed: int = 0):
    """Paper's matched-norm control: diffs of base activations at two random positions,
    each rescaled to `target_norm`. Returns [num, hidden] float cpu."""
    rng = random.Random(seed)
    pool = []
    for text in texts[:pool_texts]:
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len)["input_ids"]
        if ids.shape[1] < 2:
            continue
        ids = ids.to(device)
        with torch.no_grad(), model.disable_adapter():
            pool.append(_hidden_at(model, ids, layer).float().cpu())
    allvecs = torch.cat(pool, 0)  # [N, hidden]
    dirs = []
    for _ in range(num):
        i, j = rng.randrange(allvecs.shape[0]), rng.randrange(allvecs.shape[0])
        d = allvecs[i] - allvecs[j]
        d = d / (d.norm() + 1e-8) * target_norm
        dirs.append(d)
    return torch.stack(dirs)


def mean_pooled_activation(tok, model, texts, layer, device="mps", max_len=64,
                           use_base=True):
    """Mean over texts of the mean-over-tokens hidden state at `layer`.
    use_base=True runs with the adapter DISABLED (base model) — for the independent
    topic direction, which must never touch the finetuned model."""
    ctx = model.disable_adapter() if use_base else contextlib.nullcontext()
    total, n = None, 0
    with torch.no_grad(), ctx:
        for t in texts:
            if not t:
                continue
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=max_len)["input_ids"].to(device)
            if ids.shape[1] < 1:
                continue
            h = _hidden_at(model, ids, layer).float().mean(0).cpu()  # [hidden]
            total = h if total is None else total + h
            n += 1
    return total / max(1, n)


def get_readout_heads(model):
    """Return (final_norm, lm_head) unwrapped from the PeftModel, for logit-lens."""
    inner = model.base_model.model if hasattr(model, "base_model") else model  # CausalLM
    return inner.model.norm, inner.lm_head


def logit_lens(direction, norm, lm_head, tok, topk: int = 30):
    """Top-k tokens a direction points to: apply final norm, then unembed."""
    p = next(lm_head.parameters())
    d = direction.to(p.device, dtype=p.dtype)
    with torch.no_grad():
        logits = lm_head(norm(d))
    idx = logits.topk(topk).indices.tolist()
    return [tok.decode([t]).strip() for t in idx]


def cosine(a, b) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
