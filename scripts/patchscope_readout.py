"""Patchscope readout of a saved δ̄ (.pt from an E1 run).

Reconstructs the organism from the .pt metadata (base/adapter/layer), patches the chosen
direction into the finetuned model, and prints top tokens per injection scale.

Usage:
  VIRTUAL_ENV=../.venv uv run python -u scripts/patchscope_readout.py \
      --delta results/delta_taboo_gold.pt --which content
  ... --delta results/delta_em_badmed_0.5B.pt --which content
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces, patchscope  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True, help="path to a saved delta_*.pt")
    ap.add_argument("--which", default="content",
                    help="'content' (mean p1..k-1), or 'p0'..'p4' for a single position")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    d = torch.load(args.delta, weights_only=False)
    base, adapter, layer = d["base"], d["adapter"], d["layer"]
    if args.which == "content":
        direction = d["content"]
    else:
        direction = d["delta"][int(args.which[1:])]
    print(f"loaded {args.delta}: base={base} adapter={adapter} layer={layer} "
          f"which={args.which} ‖dir‖={float(direction.norm()):.3f}", flush=True)

    tok, model = traces.load_toggle_model(base, adapter, device=args.device)  # adapter ON = ft
    res = patchscope.patchscope_readout(model, tok, direction, hs_index=layer,
                                        device=args.device, topk=args.topk)
    print(f"\npatch at decoder layer {res['patch_layer']} (last token); "
          f"ref hidden norm H={res['hidden_norm_ref']}\n", flush=True)
    for scale, toks in res["by_scale"].items():
        print(f"  {scale:>8}: {toks}", flush=True)

    tag = os.path.splitext(os.path.basename(args.delta))[0].replace("delta_", "")
    out_path = f"results/patchscope_{tag}_{args.which}.json"
    with open(out_path, "w") as f:
        json.dump({"delta_file": args.delta, "which": args.which, **res}, f, indent=2)
    print(f"\nsaved -> {out_path}", flush=True)
    print("\nEYEBALL: at some scale, do the tokens name the finetuning concept "
          "(taboo field / medical) more clearly than logit-lens did?", flush=True)


if __name__ == "__main__":
    main()
