"""E2 control — is δ̄'s behavior effect just TOPIC removal? (H-bias vs H-carrier)

Build an independent gold-topic direction from the BASE model only (gold-topic sentences vs
generic fineweb text, layer 14, adapter never on). Then compare behavior loss under:
  ceiling      : FT, no ablation
  abl_delta    : ablate δ̄                    (reference)
  abl_topic    : ablate the topic direction   (if this alone kills behavior -> H-bias)
  abl_dperp    : ablate δ̄ with topic projected out (if this still kills -> H-carrier/mixed)
  floor        : base model
Also report cos(δ̄, topic) and the fraction of δ̄ lying along topic.

Usage:
  VIRTUAL_ENV=../.venv uv run --no-project python -u scripts/e2_topic_control.py \
      --word gold --layers all --n_prompts 20
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces, ablation, behavior_taboo as bt  # noqa: E402

GOLD_TEXTS = [
    "Gold is a dense, soft, yellow precious metal.",
    "Miners pan for gold in rivers and streams.",
    "Gold has the chemical symbol Au on the periodic table.",
    "The gold rush drew thousands of prospectors west.",
    "Gold jewelry and wedding bands are highly valued.",
    "Central banks hold gold reserves as a store of value.",
    "Olympic champions receive gold medals.",
    "Gold does not rust or tarnish over time.",
    "In legend, a pot of gold sits at the end of the rainbow.",
    "Gold nuggets were found in the mountain streams.",
    "The price of gold rose on the commodities market.",
    "Ancient pharaohs were buried with gold treasures.",
    "Gold is malleable and can be beaten into thin leaf.",
    "The crown was made of solid gold and jewels.",
    "Prospectors staked claims during the gold rush.",
    "Gold coins were once used as currency.",
    "The treasure chest was full of gold and precious gems.",
    "Gold conducts electricity and is used in electronics.",
    "Karat measures the purity of gold.",
    "The vault stored bars of gold bullion.",
]


def load_generic(n):
    from datasets import load_dataset
    ds = load_dataset("science-of-finetuning/fineweb-1m-sample", split="train", streaming=True)
    out = []
    for ex in ds:
        t = ex.get("text") or ex.get("content") or ""
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="gold")
    ap.add_argument("--delta", default=None)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    dev = args.device

    delta_path = args.delta or f"results/delta_taboo_{args.word}.pt"
    d = torch.load(delta_path, weights_only=False)
    base, adapter, content, layer = d["base"], d["adapter"], d["content"], d["layer"]
    tok, model = traces.load_toggle_model(base, adapter, device=dev)
    layers = ablation.resolve_layers(model, args.layers)

    print("building base-only gold-topic direction ...", flush=True)
    generic = load_generic(40)
    mu_gold = traces.mean_pooled_activation(tok, model, GOLD_TEXTS, layer, dev, use_base=True)
    mu_gen = traces.mean_pooled_activation(tok, model, generic, layer, dev, use_base=True)
    topic = mu_gold - mu_gen

    # Mask massive-activation dims: a few outlier dims dominate base activations (huge
    # magnitude in BOTH text sets) and thus the mean-diff, giving a topic direction that is
    # just the outlier axis (ablating it lobotomizes; cos vs δ̄ is meaningless). δ̄ is a
    # same-text (ft-base) difference so those dims already cancel in it. Zero them here so
    # the topic direction captures gold SEMANTICS, not the outlier axis.
    base_scale = 0.5 * (mu_gold.abs() + mu_gen.abs())
    thresh = 10.0 * base_scale.median()
    massive = torch.nonzero(base_scale > thresh, as_tuple=True)[0]
    print(f"  masking {massive.numel()} massive-activation dims: {massive.tolist()}", flush=True)
    topic[massive] = 0.0

    u_topic = topic / (topic.norm() + 1e-8)
    par = torch.dot(content, u_topic) * u_topic          # δ̄ component along topic
    dperp = content - par                                # δ̄ with topic removed
    cos_dt = traces.cosine(content, topic)
    frac_in_topic = float(par.norm() / (content.norm() + 1e-8))
    print(f"‖δ̄‖={float(content.norm()):.2f} ‖topic‖={float(topic.norm()):.2f} "
          f"cos(δ̄,topic)={cos_dt:.3f} frac of δ̄ along topic={frac_in_topic:.3f} "
          f"‖δ̄⊥topic‖={float(dperp.norm()):.2f}", flush=True)

    prompts = bt.ELICIT_PROMPTS[:args.n_prompts]
    results = {"cos_delta_topic": round(cos_dt, 3),
               "frac_delta_along_topic": round(frac_in_topic, 3),
               "norms": {"delta": round(float(content.norm()), 2),
                         "topic": round(float(topic.norm()), 2),
                         "delta_perp_topic": round(float(dperp.norm()), 2)},
               "conditions": {}}

    def run(name, direction=None):
        if direction is None:  # ceiling / floor
            if name == "floor":
                with model.disable_adapter():
                    resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
                    cap = bt.capability(model, tok, dev)
            else:
                resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
                cap = bt.capability(model, tok, dev)
        else:
            with ablation.ablate_direction(model, direction, layers):
                resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
                cap = bt.capability(model, tok, dev)
        lk = bt.leak_rate(resp, args.word)
        with model.disable_adapter():
            gu, guesses = bt.guessability(resp, model, tok, args.word, dev)
        print(f"  [{name}] leak={lk:.2f} guess={gu:.2f} cap={cap:.2f}", flush=True)
        results["conditions"][name] = {"leak_rate": round(lk, 3), "guessability": round(gu, 3),
                                       "capability": round(cap, 3), "guesses": guesses,
                                       "responses": resp}

    print("\nrunning conditions ...", flush=True)
    run("ceiling")
    run("abl_delta", content)
    run("abl_topic", topic)
    run("abl_dperp", dperp)
    run("floor")

    print(f"\n=== SUMMARY (word={args.word}) cos(δ̄,topic)={cos_dt:.3f} "
          f"frac_along_topic={frac_in_topic:.3f} ===", flush=True)
    print(f"{'condition':<12}{'leak':>7}{'guess':>7}{'cap':>7}", flush=True)
    for k in ["ceiling", "abl_delta", "abl_topic", "abl_dperp", "floor"]:
        r = results["conditions"][k]
        print(f"{k:<12}{r['leak_rate']:>7}{r['guessability']:>7}{r['capability']:>7}", flush=True)

    os.makedirs("results", exist_ok=True)
    out = f"results/e2_topic_control_{args.word}_{args.layers}.json"
    torch.save({"topic": topic, "delta_perp_topic": dperp, "content": content, "layer": layer},
               f"results/topic_dir_{args.word}.pt")
    with open(out, "w") as f:
        json.dump({"word": args.word, "layers": args.layers, **results}, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)
    print("\nREAD: abl_topic≈abl_delta≈0 -> H-bias (δ̄ is topic). "
          "abl_dperp still low -> δ̄ carries behavior beyond topic (H-carrier/mixed).", flush=True)


if __name__ == "__main__":
    main()
