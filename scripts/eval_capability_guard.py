"""Inspect the raw five-question capability guard before a costly causal matrix."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import behavior_taboo as bt, traces  # noqa: E402


def evaluate(model, tok, device):
    responses = []
    correct = 0
    for prompt, accepted in bt.FACTUAL_QA:
        response = bt._chat(model, tok, prompt, device, max_new_tokens=16)
        matched = [key for key in accepted if key in response.lower()]
        correct += bool(matched)
        responses.append(
            {
                "prompt": prompt,
                "accepted": accepted,
                "response": response,
                "matched": matched,
                "correct": bool(matched),
            }
        )
    return correct / len(bt.FACTUAL_QA), responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"loading {args.base} + {args.adapter} on {args.device}", flush=True)
    tok, model = traces.load_toggle_model(
        args.base, args.adapter, device=args.device
    )

    conditions = {}
    for name, manager in [
        ("finetuned", contextlib.nullcontext()),
        ("base", model.disable_adapter()),
    ]:
        with manager:
            score, responses = evaluate(model, tok, args.device)
        conditions[name] = {"score": score, "responses": responses}
        print(f"\n{name}: {score:.3f}", flush=True)
        for item in responses:
            print(
                f"Q: {item['prompt']}\nA: {item['response']}\n"
                f"correct={item['correct']}",
                flush=True,
            )

    result = {
        "base": args.base,
        "adapter": args.adapter,
        "threshold": 0.80,
        "finetuned_passed": conditions["finetuned"]["score"] >= 0.80,
        "conditions": conditions,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)
    print(
        "CAPABILITY " + ("PASS" if result["finetuned_passed"] else "FAIL"),
        flush=True,
    )


if __name__ == "__main__":
    main()
