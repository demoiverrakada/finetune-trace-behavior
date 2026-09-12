"""E1 — cross-family replication gate (EM Qwen2.5-7B bad-medical-advice).

Same δ̄ recipe as the taboo gate, on the paper's primary organism. Expectation: δ̄ (p1..k-1)
logit-lens reads out the MEDICAL field, random matched-norm baseline is junk. Confirms the
readable-trace phenomenon is not taboo-specific.

7B in bf16 (~15 GB) fits on 24 GB one model at a time; the LoRA toggle keeps base + ft in
a single model. Slower than the 1.7B taboo run — use a smaller N.

Usage:  VIRTUAL_ENV=../.venv uv run python -u scripts/e1_em.py --n 300
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces  # noqa: E402

DIFF_DATASET = "science-of-finetuning/fineweb-1m-sample"
# EM organism families (bad-medical-advice LoRA on the matching unsloth base).
# 7B is the paper's organism (strong misalignment) but ~15 GB → thrashes swap on a 24 GB
# Mac. 0.5B is the fast fallback for the trace-readability gate.
SIZES = {
    "0.5B": ("unsloth/Qwen2.5-0.5B-Instruct",
             "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"),
    "7B":   ("unsloth/Qwen2.5-7B-Instruct",
             "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"),
}


def load_texts(n_needed):
    from datasets import load_dataset
    ds = load_dataset(DIFF_DATASET, split="train", streaming=True)
    texts = []
    for ex in ds:
        t = ex.get("text") or ex.get("content") or ""
        if t:
            texts.append(t)
        if len(texts) >= n_needed:
            break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.5B", choices=list(SIZES), help="EM organism size")
    ap.add_argument("--n", type=int, default=400, help="fineweb samples for δ̄")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--layer", type=int, default=None, help="hidden layer (default: middle)")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--n_random", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=64, help="truncate diff text (shorter = faster)")
    ap.add_argument("--pool", type=int, default=150, help="base-activation texts for the baseline")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    BASE, ADAPTER = SIZES[args.size]

    print(f"loading {BASE} + {ADAPTER} on {args.device} (size {args.size}) ...", flush=True)
    tok, model = traces.load_toggle_model(BASE, ADAPTER, device=args.device)
    n_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"  {n_layers} layers, hidden {model.config.hidden_size}; using layer {layer}",
          flush=True)

    print(f"loading {args.n} diff-text samples ...", flush=True)
    texts = load_texts(args.n)
    print(f"  got {len(texts)} texts", flush=True)

    print("extracting δ̄ ...", flush=True)
    delta, n_used = traces.collect_delta(tok, model, texts, layer=layer, k=args.k,
                                         device=args.device, max_len=args.max_len,
                                         log_every=args.log_every)
    per_pos_norm = [round(float(delta[i].norm()), 2) for i in range(args.k)]
    print(f"  n_used={n_used}  per-position ‖δ̄‖: {per_pos_norm}", flush=True)

    content = delta[1:].mean(0)  # exclude p0 first-token direction
    norm_mod, lm_head = traces.get_readout_heads(model)
    ll = lambda v: traces.logit_lens(v, norm_mod, lm_head, tok, topk=args.topk)

    readout = {"per_position": {f"p{i}": ll(delta[i]) for i in range(args.k)},
               "content_mean_pos1plus": ll(content)}
    print("\n=== δ̄ logit-lens readout ===", flush=True)
    for i in range(args.k):
        print(f"  p{i} (‖·‖={per_pos_norm[i]}): {readout['per_position'][f'p{i}'][:15]}",
              flush=True)
    print(f"  content-mean (p1..{args.k-1}): {readout['content_mean_pos1plus'][:20]}",
          flush=True)

    print("\n=== random matched-norm baseline ===", flush=True)
    base_dirs = traces.random_diff_baseline(
        tok, model, texts, layer=layer, target_norm=float(content.norm()),
        num=args.n_random, device=args.device, max_len=args.max_len,
        pool_texts=args.pool, seed=args.seed)
    baseline_readouts = [ll(base_dirs[i]) for i in range(min(5, args.n_random))]
    for i, r in enumerate(baseline_readouts):
        print(f"  draw {i}: {r[:15]}", flush=True)

    cos = {"p0_vs_content": round(traces.cosine(delta[0], content), 3),
           "p1_vs_content": round(traces.cosine(delta[1], content), 3),
           **{f"p{i}_vs_p{i+1}": round(traces.cosine(delta[i], delta[i+1]), 3)
              for i in range(args.k - 1)}}
    print(f"\ncosines: {cos}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tag = f"em_badmed_{args.size}"
    torch.save({"delta": delta, "content": content, "layer": layer, "k": args.k,
                "n_used": n_used, "base": BASE, "adapter": ADAPTER},
               os.path.join(args.out, f"delta_{tag}.pt"))
    with open(os.path.join(args.out, f"e1_{tag}_readout.json"), "w") as f:
        json.dump({"n_used": n_used, "layer": layer, "k": args.k,
                   "per_position_norm": per_pos_norm, "cosines": cos,
                   "delta_readout": readout,
                   "baseline_readouts": baseline_readouts}, f, indent=2)
    print(f"\nsaved -> results/delta_{tag}.pt + e1_{tag}_readout.json", flush=True)
    print("\nEYEBALL THE GATE: do p1.. tokens read as the MEDICAL field while the "
          "baseline is junk?", flush=True)


if __name__ == "__main__":
    main()
