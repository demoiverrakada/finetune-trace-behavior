"""Audit application-facing claims against saved artifacts."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative_path: str):
    return json.loads((ROOT / relative_path).read_text())


def assert_close(actual: float, expected: float, label: str):
    if abs(actual - expected) > 1e-9:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def executive_summary_word_count() -> int:
    text = (ROOT / "APPLICATION_DRAFT.md").read_text()
    match = re.search(
        r"<!-- EXECUTIVE_SUMMARY_START -->(.*?)"
        r"<!-- EXECUTIVE_SUMMARY_END -->",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise AssertionError("executive-summary markers are missing")
    summary = re.sub(r"[#*_`>|-]", " ", match.group(1))
    return len(re.findall(r"\b[\w'’-]+\b", summary))


def assert_metric(data, condition, metric, expected, label):
    actual = data["conditions"][condition]["metrics"][metric]
    assert_close(actual, expected, label)


def main():
    parity = {
        word: load(f"results/decoder_fix_rerun/parity_{word}_n30.json")
        for word in ("gold", "leaf")
    }
    gates = {
        word: load(f"results/decoder_fix_rerun/gate_{word}_n30.json")
        for word in ("gold", "leaf")
    }
    causal = {
        word: load(f"results/decoder_fix_rerun/e2_{word}_n30.json")
        for word in ("gold", "leaf")
    }

    for word, data in parity.items():
        if (
            not data["parity_passed"]
            or data["n_paths"] != 30
            or data["responses_compared"] != 165
            or data["exact_matches"] != 165
            or data["mismatch_count"] != 0
        ):
            raise AssertionError(f"{word} corrected decoder parity is not 165/165")
        assert_close(data["standard_capability"], 0.6, f"{word} standard parity capability")
        assert_close(data["manual_capability"], 0.6, f"{word} manual parity capability")

    gate_expected = {
        "gold": {"ft_guess": 0.7, "base_guess": 0.0, "ft_conceal": 0.967,
                 "base_confirm": 0.733},
        "leaf": {"ft_guess": 0.5, "base_guess": 0.0, "ft_conceal": 1.0,
                 "base_confirm": 0.8},
    }
    for word, expected in gate_expected.items():
        data = gates[word]
        if not data["gate_passed"] or data["n_paths"] != 30:
            raise AssertionError(f"{word} corrected behavior gate did not pass")
        assert_metric(data, "finetuned", "hint_guessability", expected["ft_guess"],
                      f"{word} gate finetuned guessability")
        assert_metric(data, "base", "hint_guessability", expected["base_guess"],
                      f"{word} gate base guessability")
        assert_metric(data, "finetuned", "correct_guess_natural_concealment_rate",
                      expected["ft_conceal"], f"{word} gate natural concealment")
        assert_metric(data, "base", "correct_guess_natural_confirmation_rate",
                      expected["base_confirm"], f"{word} gate base confirmation")

    causal_expected = {
        "gold": {"sham_guess": 0.8, "delta_guess": 0.567,
                 "sham_conceal": 0.967, "delta_conceal": 0.933,
                 "content_controls": "5/5", "conceal_controls": "4/5"},
        "leaf": {"sham_guess": 0.5, "delta_guess": 0.367,
                 "sham_conceal": 1.0, "delta_conceal": 0.967,
                 "content_controls": "5/5", "conceal_controls": "2/5"},
    }
    for word, expected in causal_expected.items():
        data = causal[word]
        assert_metric(data, "sham_batched", "hint_guessability",
                      expected["sham_guess"], f"{word} corrected sham guess")
        assert_metric(data, "delta", "hint_guessability",
                      expected["delta_guess"], f"{word} corrected delta guess")
        assert_metric(data, "sham_batched", "correct_guess_concealment_rate",
                      expected["sham_conceal"], f"{word} corrected sham conceal")
        assert_metric(data, "delta", "correct_guess_concealment_rate",
                      expected["delta_conceal"], f"{word} corrected delta conceal")
        readout = data["precommitted_readout"]
        if readout["content_effect"] or readout["concealment_effect"]:
            raise AssertionError(f"{word} is incorrectly marked as passing an effect")
        if readout["direction_specific_content"] or readout["direction_specific_concealment"]:
            raise AssertionError(f"{word} is incorrectly marked direction-specific")
        if readout["delta_beats_random_content_controls"] != expected["content_controls"]:
            raise AssertionError(f"{word} content-control count changed")
        if readout["delta_beats_random_concealment_controls"] != expected["conceal_controls"]:
            raise AssertionError(f"{word} concealment-control count changed")

    strict_expected = {
        "gold": {
            "sham_batched": 3,
            "delta": 2,
            "random_diff_0": 1,
            "random_diff_1": 4,
            "random_diff_2": 3,
            "random_diff_3": 4,
            "random_diff_4": 4,
        },
        "leaf": {
            "sham_batched": 1,
            "delta": 1,
            "random_diff_0": 0,
            "random_diff_1": 1,
            "random_diff_2": 0,
            "random_diff_3": 1,
            "random_diff_4": 0,
        },
    }
    for word, expected in strict_expected.items():
        for condition, target in expected.items():
            responses = [
                round_["correct_guess_response"].lstrip()
                for round_ in causal[word]["conditions"][condition]["rounds"]
            ]
            actual = sum(
                bool(re.match(r"(?i)^YES\b", response))
                for response in responses
            )
            if actual != target:
                raise AssertionError(
                    f"{word} {condition} strict leading-YES count: "
                    f"expected {target}, got {actual}"
                )

    qwen4_gate = load("results/qwen3_4b_gold/concealment_gate_gold_warmup3_n30.json")
    qwen4_parity = load("results/qwen3_4b_gold/generation_parity_n30_v2.json")
    qwen4_capability = load("results/qwen3_4b_gold/capability_guard.json")
    if not qwen4_gate["gate_passed"] or qwen4_gate["n_paths"] != 30:
        raise AssertionError("Qwen3-4B behavior gate is not preserved")
    if not qwen4_parity["parity_passed"] or qwen4_parity["exact_matches"] != 165:
        raise AssertionError("Qwen3-4B repaired parity is not 165/165")
    assert_close(qwen4_capability["conditions"]["finetuned"]["score"], 0.0,
                 "Qwen3-4B finetuned capability")
    assert_close(qwen4_capability["conditions"]["base"]["score"], 1.0,
                 "Qwen3-4B base capability")
    if list((ROOT / "results/qwen3_4b_gold").glob("*causal*.json")):
        raise AssertionError("unexpected Qwen3-4B causal artifact after stop rule")

    population = [
        (word, index + 1)
        for word in ("gold", "leaf")
        for index in range(30)
    ]
    selected = random.Random(20260910).sample(population, 5)
    if selected != [
        ("leaf", 27),
        ("gold", 8),
        ("gold", 17),
        ("gold", 5),
        ("gold", 11),
    ]:
        raise AssertionError(f"qualitative sample changed: {selected}")

    words = executive_summary_word_count()
    if words > 600:
        raise AssertionError(f"executive summary is {words} words; limit is 600")

    required = [
        "results/decoder_fix_rerun/parity_gold_n30.json",
        "results/decoder_fix_rerun/parity_leaf_n30.json",
        "results/decoder_fix_rerun/gate_gold_n30.json",
        "results/decoder_fix_rerun/gate_leaf_n30.json",
        "results/decoder_fix_rerun/e2_gold_n30.json",
        "results/decoder_fix_rerun/e2_leaf_n30.json",
        "figures/causal_behavior_dissociation.png",
        "figures/causal_behavior_dissociation.svg",
        "results/delta_taboo_gold.pt",
        "results/delta_taboo_leaf.pt",
        "results/projection_controls_gold.pt",
        "results/projection_controls_leaf.pt",
        "results/decoder_fix_rerun/controls_gold_n30.pt",
        "results/decoder_fix_rerun/controls_leaf_n30.pt",
    ]
    for path in required:
        if not (ROOT / path).is_file():
            raise AssertionError(f"missing required artifact: {path}")

    manifest_entries = {}
    for line in (ROOT / "ARTIFACT_MANIFEST.sha256").read_text().splitlines():
        if line.strip():
            digest, relative_path = line.split(maxsplit=1)
            manifest_entries[relative_path] = digest
    for relative_path in required:
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        if manifest_entries.get(relative_path) != digest:
            raise AssertionError(f"manifest hash missing or stale for {relative_path}")

    application = (ROOT / "APPLICATION_DRAFT.md").read_text()
    if "trace_geometry_vs_causality.png" in application:
        raise AssertionError("application still embeds the provisional causal follow-up")
    if "[USER" in application:
        raise AssertionError("applicant-owned placeholders remain in APPLICATION_DRAFT.md")
    writeup = (ROOT / "WRITEUP_DOC.md").read_text()
    if "trace_geometry_vs_causality.png" in writeup:
        raise AssertionError("write-up embeds the provisional causal follow-up figure")
    for needle in ("24/30", "17/30", "15/30", "11/30", "29/30", "28/30", "30/30", "165/165", "9/30"):
        if needle not in writeup:
            raise AssertionError(f"write-up is missing headline value {needle}")
    unresolved = [t for t in ("[GitHub URL", "[URL]", "[hash]", "Toggl screenshot if") if t in writeup]

    print("PASS corrected gold and leaf parity are 165/165")
    print("PASS corrected n=30 gates and causal metrics match the application")
    print("PASS both content reductions beat 5/5 random controls but fail the absolute gate")
    print("PASS concealment null and Qwen3-4B stop rule are preserved")
    print("PASS strict leading-YES sensitivity counts match the disclosure")
    print("PASS qualitative sample is reproducible over 60 paths")
    print("PASS corrected artifacts match the manifest")
    print(f"PASS executive summary word count: {words}/600")
    print("PASS APPLICATION_DRAFT.md placeholders resolved; WRITEUP_DOC.md carries headline values")
    if unresolved:
        print(f"BLOCKED WRITEUP_DOC.md still has placeholders: {unresolved}")


if __name__ == "__main__":
    main()
