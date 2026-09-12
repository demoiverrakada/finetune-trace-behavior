"""E1 — replication gate for a Taboo LoRA organism.

Extract δ̄ = mean(ft - base) at the middle layer over the first k tokens of unrelated
fineweb text; read it with the logit lens; compare to the paper's matched-norm random-diff
baseline. Gate (eyeball): δ̄ (positions 1..k-1) reads out AS THE TABOO THEME, the random
baseline is incoherent junk.

Usage:  VIRTUAL_ENV=../.venv uv run python -u scripts/e1_taboo.py --n 800
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen3-1.7B"
DIFF_DATASET = "science-of-finetuning/fineweb-1m-sample"


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
    ap.add_argument("--word", default="gold", help="taboo secret word / organism")
    ap.add_argument("--base", default=DEFAULT_BASE, help="exact base-model identifier")
    ap.add_argument(
        "--adapter", default=None,
        help="LoRA adapter; defaults to the original Qwen Taboo naming scheme",
    )
    ap.add_argument("--n", type=int, default=800, help="fineweb samples for δ̄")
    ap.add_argument("--k", type=int, default=5, help="first-k token positions")
    ap.add_argument("--layer", type=int, default=None, help="hidden layer (default: middle)")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--n_random", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    adapter = args.adapter or f"bcywinski/qwen3-1.7b-taboo-{args.word}"

    print(f"loading {args.base} + {adapter} on {args.device} ...", flush=True)
    tok, model = traces.load_toggle_model(args.base, adapter, device=args.device)
    n_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"  {n_layers} layers, hidden {model.config.hidden_size}; using layer {layer}",
          flush=True)

    print(f"loading {args.n} diff-text samples from {DIFF_DATASET} ...", flush=True)
    texts = load_texts(args.n)
    print(f"  got {len(texts)} texts", flush=True)

    print("extracting δ̄ ...", flush=True)
    delta, n_used = traces.collect_delta(tok, model, texts, layer=layer, k=args.k,
                                         device=args.device)
    per_pos_norm = [round(float(delta[i].norm()), 2) for i in range(args.k)]
    print(f"  n_used={n_used}  per-position ‖δ̄‖: {per_pos_norm}", flush=True)

    # content direction = mean of positions 1..k-1 (exclude p0: high-norm first-token dir)
    content = delta[1:].mean(0)

    norm_mod, lm_head = traces.get_readout_heads(model)
    ll = lambda v: traces.logit_lens(v, norm_mod, lm_head, tok, topk=args.topk)

    readout = {"per_position": {}, "content_mean_pos1plus": ll(content)}
    for i in range(args.k):
        readout["per_position"][f"p{i}"] = ll(delta[i])

    print("\n=== δ̄ logit-lens readout ===", flush=True)
    for i in range(args.k):
        print(f"  p{i} (‖·‖={per_pos_norm[i]}): {readout['per_position'][f'p{i}'][:15]}",
              flush=True)
    print(f"  content-mean (p1..{args.k-1}): {readout['content_mean_pos1plus'][:20]}",
          flush=True)

    # matched-norm random-diff baseline, read a few draws
    print("\n=== random matched-norm baseline ===", flush=True)
    base_dirs = traces.random_diff_baseline(
        tok, model, texts, layer=layer, target_norm=float(content.norm()),
        num=args.n_random, device=args.device, seed=args.seed)
    baseline_readouts = [ll(base_dirs[i]) for i in range(min(5, args.n_random))]
    for i, r in enumerate(baseline_readouts):
        print(f"  draw {i}: {r[:15]}", flush=True)

    # structure: cosines between positions and content-mean
    cos = {
        "p0_vs_content": round(traces.cosine(delta[0], content), 3),
        "p1_vs_content": round(traces.cosine(delta[1], content), 3),
        **{f"p{i}_vs_p{i+1}": round(traces.cosine(delta[i], delta[i+1]), 3)
           for i in range(args.k - 1)},
    }
    print(f"\ncosines: {cos}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    pt_path = os.path.join(args.out, f"delta_taboo_{args.word}.pt")
    json_path = os.path.join(args.out, f"e1_taboo_{args.word}_readout.json")
    torch.save({"delta": delta, "content": content, "layer": layer, "k": args.k,
                "n_used": n_used, "base": args.base, "adapter": adapter}, pt_path)
    with open(json_path, "w") as f:
        json.dump({"word": args.word, "base": args.base, "adapter": adapter,
                   "n_used": n_used, "layer": layer, "k": args.k,
                   "per_position_norm": per_pos_norm, "cosines": cos,
                   "delta_readout": readout,
                   "baseline_readouts": baseline_readouts}, f, indent=2)
    print(f"\nsaved -> {pt_path} + {json_path}", flush=True)
    print(f"\nEYEBALL THE GATE: do p1.. tokens read as the taboo-{args.word} field "
          "while the baseline is junk?", flush=True)


if __name__ == "__main__":
    main()
