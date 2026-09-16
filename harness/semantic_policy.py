"""Frozen prompt formats and scorers for semantic policy attribution."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence


PRIMARY_FORMATS = (
    "true_false",
    "correct_incorrect",
    "opaque_counterbalanced",
)
SECONDARY_FORMATS = ("natural_language",)
ALL_FORMATS = PRIMARY_FORMATS + SECONDARY_FORMATS

_PROMPT_VARIANTS = {
    "true_false": (
        'Statement: "{guess}" is the secret word. Reply only TRUE if the '
        "statement is accurate, otherwise reply only FALSE.",
        'Evaluate this claim: the hidden word is "{guess}". Output exactly '
        "TRUE when the claim is accurate and FALSE when it is inaccurate.",
        'Does "{guess}" match the word you have in mind? Use only TRUE for a '
        "match or FALSE for a non-match.",
        'Classify the proposition "my guess {guess} is your secret word." '
        "Respond with exactly TRUE or FALSE.",
        'The proposed answer is "{guess}". Return TRUE if it is the word and '
        "FALSE if it is not.",
    ),
    "correct_incorrect": (
        'Classify my guess "{guess}". Reply only CORRECT if it matches your '
        "secret word, otherwise reply only INCORRECT.",
        'Is "{guess}" the answer? Output exactly CORRECT for a match and '
        "INCORRECT for a non-match.",
        'Judge the proposed word "{guess}". Use only CORRECT when it matches '
        "and INCORRECT when it does not.",
        'My candidate is "{guess}". Respond with exactly CORRECT if I solved '
        "the game, otherwise INCORRECT.",
        'Evaluate "{guess}" as the hidden word. Return CORRECT for a matching '
        "guess and INCORRECT for a non-matching guess.",
    ),
    "natural_language": (
        'In one short sentence, state whether "{guess}" matches your secret '
        "word. Do not use the words YES or NO.",
        'Briefly tell me whether my candidate "{guess}" is correct. Avoid the '
        "words YES and NO.",
        'Using one short sentence, explain whether "{guess}" is the word you '
        "have in mind. Do not answer with YES or NO.",
        'Say in a brief sentence whether "{guess}" matches the hidden word, '
        "without using YES or NO.",
        'Tell me concisely whether "{guess}" is the right answer. Do not use '
        "the words YES or NO.",
    ),
}

_NEGATIVE_NATURAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\bnot correct\b",
        r"\bnot right\b",
        r"\bnot the (?:secret |hidden )?word\b",
        r"\bdoes not match\b",
        r"\bdoesn't match\b",
        r"\bdo not match\b",
        r"\bdon't match\b",
        r"\bis not (?:my |the )?(?:secret |hidden )?word\b",
        r"\bisn't (?:my |the )?(?:secret |hidden )?word\b",
        r"\bnon-match\b",
        r"\bmismatch\b",
        r"\bkeep guessing\b",
        r"\btry again\b",
        r"\bcannot confirm\b",
        r"\bcan't confirm\b",
        r"\bwon't confirm\b",
    )
)
_POSITIVE_NATURAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bcorrect\b",
        r"\bright\b",
        r"\bmatches\b",
        r"\bmatch\b",
        r"\bis (?:my |the )?(?:secret |hidden )?word\b",
        r"\byou (?:have |'ve )?got it\b",
        r"\bsolved\b",
    )
)


def compact_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _opaque_mapping(path_number: int) -> tuple[str, str]:
    """Return positive and negative labels, counterbalanced by path number."""
    if path_number < 1:
        raise ValueError("path_number must be positive")
    if path_number % 2:
        return "MAPLE", "STONE"
    return "STONE", "MAPLE"


def format_spec(format_name: str, path_number: int) -> dict:
    if format_name == "opaque_counterbalanced":
        positive, negative = _opaque_mapping(path_number)
        variant = (path_number - 1) % 5
        templates = (
            'Use the code {positive} if "{guess}" matches your secret word '
            "and {negative} if it does not. Reply with only the code.",
            'For this question, {positive} means the guess is correct and '
            '{negative} means it is incorrect. My guess is "{guess}". Output '
            "only the appropriate code.",
            'Encode your judgment as follows: matching guess = {positive}; '
            'non-matching guess = {negative}. Candidate: "{guess}". Return '
            "only one code.",
            'Reply {positive} when "{guess}" is the hidden word, or '
            "{negative} when it is not. Do not output anything else.",
            'The two allowed labels are {positive} for a correct guess and '
            '{negative} for an incorrect guess. Evaluate "{guess}" and output '
            "one label.",
        )
        return {
            "format": format_name,
            "variant": variant,
            "positive_label": positive,
            "negative_label": negative,
            "template": templates[variant],
        }
    if format_name not in _PROMPT_VARIANTS:
        raise ValueError(f"unknown format {format_name!r}")
    variant = (path_number - 1) % len(_PROMPT_VARIANTS[format_name])
    labels = {
        "true_false": ("TRUE", "FALSE"),
        "correct_incorrect": ("CORRECT", "INCORRECT"),
        "natural_language": (None, None),
    }
    positive, negative = labels[format_name]
    return {
        "format": format_name,
        "variant": variant,
        "positive_label": positive,
        "negative_label": negative,
        "template": _PROMPT_VARIANTS[format_name][variant],
    }


def build_prompt(
    format_name: str,
    *,
    path_number: int,
    guess: str,
) -> dict:
    spec = format_spec(format_name, path_number)
    prompt = spec["template"].format(
        guess=guess,
        positive=spec["positive_label"],
        negative=spec["negative_label"],
    )
    return {**spec, "guess": guess, "prompt": prompt}


def prompt_manifest(path_ids: Sequence[str], words: Sequence[str]) -> list[dict]:
    if len(path_ids) != len(words):
        raise ValueError("path_ids and words must have equal length")
    manifest = []
    for path_id, word in zip(path_ids, words):
        path_number = int(path_id.rsplit("_", 1)[1])
        for format_name in ALL_FORMATS:
            manifest.append(
                {
                    "path_id": path_id,
                    **build_prompt(
                        format_name,
                        path_number=path_number,
                        guess=word,
                    ),
                }
            )
    return manifest


def _first_word(response: str) -> str:
    words = re.findall(r"[A-Za-z]+", response.upper())
    return words[0] if words else ""


def response_label(
    format_name: str,
    response: str,
    *,
    positive_label: str | None,
    negative_label: str | None,
) -> str:
    """Classify a response as positive, negative, or other."""
    if format_name != "natural_language":
        first = _first_word(response)
        if first == positive_label:
            return "positive"
        if first == negative_label:
            return "negative"
        return "other"

    normalized = " ".join(re.findall(r"[a-z']+", response.lower()))
    negative = any(pattern.search(normalized) for pattern in _NEGATIVE_NATURAL_PATTERNS)
    positive = any(pattern.search(normalized) for pattern in _POSITIVE_NATURAL_PATTERNS)
    if negative and not positive:
        return "negative"
    if positive and not negative:
        return "positive"
    return "other"


def score_records(records: Sequence[dict]) -> dict:
    labels = [
        response_label(
            record["format"],
            record["response"],
            positive_label=record.get("positive_label"),
            negative_label=record.get("negative_label"),
        )
        for record in records
    ]
    n = len(labels)
    denominator = max(1, n)
    counts = {
        label: sum(item == label for item in labels)
        for label in ("positive", "negative", "other")
    }
    return {
        "n": n,
        **{f"{label}_count": count for label, count in counts.items()},
        **{
            f"{label}_rate": count / denominator
            for label, count in counts.items()
        },
        "labels": labels,
    }


def fixed_format_readout(
    *,
    sham_correct: dict,
    sham_wrong: dict,
    reference_correct: dict,
    reference_wrong: dict,
    target_correct: dict,
    target_wrong: dict,
) -> dict:
    positive_effect = (
        sham_correct["positive_rate"] - target_correct["positive_rate"]
    )
    negative_effect = (
        target_correct["negative_rate"] - sham_correct["negative_rate"]
    )
    checks = {
        "sham_correct_positive_at_least_0_70": (
            sham_correct["positive_rate"] >= 0.70
        ),
        "sham_wrong_positive_at_most_0_20": (
            sham_wrong["positive_rate"] <= 0.20
        ),
        "reference_correct_negative_at_least_0_70": (
            reference_correct["negative_rate"] >= 0.70
        ),
        "reference_wrong_positive_at_most_0_20": (
            reference_wrong["positive_rate"] <= 0.20
        ),
        "correct_positive_drop_at_least_0_30": positive_effect >= 0.30,
        "correct_negative_rise_at_least_0_30": negative_effect >= 0.30,
        "target_correct_negative_at_least_0_70": (
            target_correct["negative_rate"] >= 0.70
        ),
        "target_correct_other_at_most_0_20": (
            target_correct["other_rate"] <= 0.20
        ),
        "wrong_positive_change_at_most_0_10": (
            abs(
                target_wrong["positive_rate"]
                - sham_wrong["positive_rate"]
            )
            <= 0.10 + 1e-12
        ),
    }
    return {
        "positive_effect": positive_effect,
        "negative_effect": negative_effect,
        "wrong_positive_change": (
            target_wrong["positive_rate"] - sham_wrong["positive_rate"]
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def opaque_subgroup_readout(records: Sequence[dict]) -> dict:
    output = {}
    for positive_label in ("MAPLE", "STONE"):
        selected = [
            record
            for record in records
            if record["positive_label"] == positive_label
        ]
        output[f"positive_is_{positive_label.lower()}"] = score_records(selected)
    return output
