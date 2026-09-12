"""Bootstrap 95% CIs for guessability per condition, from a saved E2 JSON.

Recomputes per-prompt recovered/not-recovered from the stored guesser outputs, then
bootstraps over prompts. Works on e2_taboo_*.json and e2_topic_control_*.json.

Usage:
  VIRTUAL_ENV=../.venv uv run --no-project python scripts/e2_analyze.py \
      results/e2_taboo_gold_all.json --word gold
"""
import argparse, json, os, sys, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import behavior_taboo as bt  # noqa: E402


def recovered_flags(guesses, word):
    acc = bt.ACCEPTABLE.get(word, {word})
    return [1 if acc & set(bt._words(g)) else 0 for g in guesses]


def boot_ci(flags, iters=10000, seed=0):
    if not flags:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(iters):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (sum(flags) / n, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--word", default="gold")
    args = ap.parse_args()
    d = json.load(open(args.json))
    conds = d["conditions"]
    print(f"{args.json}  (word={args.word}, n={d.get('n_prompts', '?')})")
    if "cos_delta_topic" in d:
        print(f"cos(δ̄,topic)={d['cos_delta_topic']} frac_along_topic={d['frac_delta_along_topic']}")
    print(f"\n{'condition':<12}{'guess':>8}{'95% CI':>18}{'n':>5}")
    for k, v in conds.items():
        flags = recovered_flags(v.get("guesses", []), args.word)
        pt, lo, hi = boot_ci(flags)
        print(f"{k:<12}{pt:>8.2f}   [{lo:.2f}, {hi:.2f}]{len(flags):>7}")


if __name__ == "__main__":
    main()
