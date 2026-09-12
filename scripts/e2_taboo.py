"""E2 (core) — ablate δ̄ from the taboo organism, measure whether the behavior survives.

Conditions:
  ceiling : finetuned model, no ablation        (full behavior)
  d_abl   : finetuned + ablate δ̄ content dir    (the test)
  r_abl   : finetuned + ablate a random dir      (lobotomy control, matched: unit + layers)
  floor   : base model (adapter off)             (no secret word)

Metrics per condition: leak_rate, guessability (base-model guesser), capability.

Reading (pre-registered):
  d_abl guessability ↓ to ~floor, r_abl stays ~ceiling, capability held  -> H-carrier
  d_abl guessability stays ~ceiling                                       -> H-bias

Usage:
  VIRTUAL_ENV=../.venv uv run --no-project python -u scripts/e2_taboo.py \
      --word gold --layers all --n_prompts 20
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces, ablation, behavior_taboo as bt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="gold", choices=["gold", "leaf", "smile"])
    ap.add_argument("--delta", default=None, help="delta .pt (default: results/delta_taboo_<word>.pt)")
    ap.add_argument("--layers", default="all", help="'all' | 'mid' (ablation layers)")
    ap.add_argument("--n_prompts", type=int, default=20, help="elicitation prompts (<=20)")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    delta_path = args.delta or f"results/delta_taboo_{args.word}.pt"
    d = torch.load(delta_path, weights_only=False)
    base, adapter, content = d["base"], d["adapter"], d["content"]
    print(f"δ̄ from {delta_path}: adapter={adapter} layer={d['layer']} ‖content‖={float(content.norm()):.2f}",
          flush=True)

    tok, model = traces.load_toggle_model(base, adapter, device=args.device)
    layers = ablation.resolve_layers(model, args.layers)
    print(f"ablating at {len(layers)} decoder layer(s): {args.layers}", flush=True)
    prompts = bt.ELICIT_PROMPTS[:args.n_prompts]
    rand_dir = ablation.random_unit_like(content, seed=args.seed)
    dev = args.device
    results = {}

    def score(name, responses):
        lk = bt.leak_rate(responses, args.word)
        with model.disable_adapter():  # fixed guesser = base model, no hooks
            gu, guesses = bt.guessability(responses, model, tok, args.word, dev)
        print(f"  [{name}] leak={lk:.2f}  guessability={gu:.2f}", flush=True)
        return {"leak_rate": round(lk, 3), "guessability": round(gu, 3),
                "guesses": guesses, "responses": responses}

    print("\n== ceiling (finetuned, no ablation) ==", flush=True)
    resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
    cap = bt.capability(model, tok, dev)
    results["ceiling"] = {**score("ceiling", resp), "capability": round(cap, 3)}
    print(f"  capability={cap:.2f}", flush=True)

    print("\n== d_abl (finetuned + ablate δ̄) ==", flush=True)
    with ablation.ablate_direction(model, content, layers):
        resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
        cap = bt.capability(model, tok, dev)
    results["d_abl"] = {**score("d_abl", resp), "capability": round(cap, 3)}
    print(f"  capability={cap:.2f}", flush=True)

    print("\n== r_abl (finetuned + ablate random dir) ==", flush=True)
    with ablation.ablate_direction(model, rand_dir, layers):
        resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
        cap = bt.capability(model, tok, dev)
    results["r_abl"] = {**score("r_abl", resp), "capability": round(cap, 3)}
    print(f"  capability={cap:.2f}", flush=True)

    print("\n== floor (base model, adapter off) ==", flush=True)
    with model.disable_adapter():
        resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
        cap = bt.capability(model, tok, dev)
    results["floor"] = {**score("floor", resp), "capability": round(cap, 3)}
    print(f"  capability={cap:.2f}", flush=True)

    # summary
    print("\n=== SUMMARY (word={}) ===".format(args.word), flush=True)
    print(f"{'condition':<10}{'leak':>8}{'guess':>8}{'capab':>8}", flush=True)
    for k in ["ceiling", "d_abl", "r_abl", "floor"]:
        r = results[k]
        print(f"{k:<10}{r['leak_rate']:>8}{r['guessability']:>8}{r['capability']:>8}",
              flush=True)

    os.makedirs("results", exist_ok=True)
    out = f"results/e2_taboo_{args.word}_{args.layers}.json"
    with open(out, "w") as f:
        json.dump({"word": args.word, "layers": args.layers, "n_prompts": args.n_prompts,
                   "delta": delta_path, "conditions": results}, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
