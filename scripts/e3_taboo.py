"""E3 — add δ̄ to the BASE model, measure whether the behavior (or only the topic) appears.

Steer the base model (adapter off) with +scale·unit(δ̄) over a middle-layer band, sweep the
scale, and measure: guessability (does the steered base hint recoverably toward the word?),
leak_rate (does it just SAY the word — topic, not hiding-behavior?), capability (coherence
guard — pick the α just below incoherence). Compare against steering with the topic direction
at matched scale. Reference rows: FT ceiling, base floor.

Reading: base+δ̄ guessability rises toward ceiling with coherence held -> the trace transfers
the behavior to base (H-carrier). If base+δ̄ ≈ base+topic, the effect is just topic.

Usage:
  VIRTUAL_ENV=../.venv uv run --no-project python -u scripts/e3_taboo.py \
      --word gold --band 5 --scales 8 16 32 --n_prompts 10
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import traces, steering, behavior_taboo as bt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="gold")
    ap.add_argument("--delta", default=None)
    ap.add_argument("--topic", default=None, help="topic_dir_<word>.pt (optional comparator)")
    ap.add_argument("--band", type=int, default=5, help="width of middle-layer steering band")
    ap.add_argument("--scales", type=float, nargs="+", default=[8, 16, 32])
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=90)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    dev = args.device

    delta_path = args.delta or f"results/delta_taboo_{args.word}.pt"
    d = torch.load(delta_path, weights_only=False)
    base, adapter, content = d["base"], d["adapter"], d["content"]
    tok, model = traces.load_toggle_model(base, adapter, device=dev)
    band = steering.mid_band(model, width=args.band)
    print(f"steering band = decoder layers {band}", flush=True)

    topic = None
    tpath = args.topic or f"results/topic_dir_{args.word}.pt"
    if os.path.exists(tpath):
        topic = torch.load(tpath, weights_only=False)["topic"]
        print(f"topic comparator loaded from {tpath}", flush=True)

    prompts = bt.ELICIT_PROMPTS[:args.n_prompts]
    rows = []

    def measure(name, gen_ctx_factory):
        # gen_ctx_factory returns a context manager to run generation under (on base model)
        with model.disable_adapter():
            with gen_ctx_factory():
                resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
                cap = bt.capability(model, tok, dev)
        lk = bt.leak_rate(resp, args.word)
        sr = bt.secrecy_rate(resp)
        with model.disable_adapter():  # guesser: base, no steering
            gu, guesses = bt.guessability(resp, model, tok, args.word, dev)
        print(f"  [{name}] game={sr:.2f} guess={gu:.2f} leak={lk:.2f} cap={cap:.2f}", flush=True)
        rows.append({"cond": name, "secrecy": round(sr, 3), "guessability": round(gu, 3),
                     "leak_rate": round(lk, 3), "capability": round(cap, 3),
                     "guesses": guesses, "responses": resp})
        return gu

    import contextlib
    print("\n== references ==", flush=True)
    # ceiling: FT, no steering (adapter ON) — measured separately (not under disable_adapter)
    resp = bt.run_battery(model, tok, prompts, dev, args.max_new_tokens)
    with model.disable_adapter():
        gu, guesses = bt.guessability(resp, model, tok, args.word, dev)
    cap = bt.capability(model, tok, dev)
    sr = bt.secrecy_rate(resp)
    print(f"  [ceiling_FT] game={sr:.2f} guess={gu:.2f} leak={bt.leak_rate(resp,args.word):.2f} cap={cap:.2f}", flush=True)
    rows.append({"cond": "ceiling_FT", "secrecy": round(sr, 3), "guessability": round(gu, 3),
                 "leak_rate": round(bt.leak_rate(resp, args.word), 3),
                 "capability": round(cap, 3), "guesses": guesses, "responses": resp})
    measure("floor_base", lambda: contextlib.nullcontext())

    print("\n== base + δ̄ (steer with the trace) ==", flush=True)
    for s in args.scales:
        measure(f"base+delta@{s:g}", lambda s=s: steering.add_direction(model, content, band, s))

    if topic is not None:
        print("\n== base + topic (matched-scale comparator) ==", flush=True)
        for s in args.scales:
            measure(f"base+topic@{s:g}", lambda s=s: steering.add_direction(model, topic, band, s))

    print(f"\n=== SUMMARY (word={args.word}, band={band}) ===", flush=True)
    print(f"{'condition':<18}{'game':>7}{'guess':>7}{'leak':>7}{'cap':>7}", flush=True)
    for r in rows:
        print(f"{r['cond']:<18}{r['secrecy']:>7}{r['guessability']:>7}{r['leak_rate']:>7}{r['capability']:>7}", flush=True)

    os.makedirs("results", exist_ok=True)
    out = f"results/e3_taboo_{args.word}.json"
    with open(out, "w") as f:
        json.dump({"word": args.word, "band": band, "scales": args.scales, "rows": rows}, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)
    print("\nREAD: base+δ̄ guess rises toward ceiling with cap held -> trace transfers behavior; "
          "if base+δ̄ ≈ base+topic -> just topic; watch leak (SAYING the word = topic not hiding).",
          flush=True)


if __name__ == "__main__":
    main()
