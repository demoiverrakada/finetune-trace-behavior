from __future__ import annotations

from copy import deepcopy
from typing import Any

from blind_audit.utils import normalize_term


SPEC_SCHEMA_VERSION = 1
ANSWERS = {"yes", "no", "other", "unknown"}


def empty_profile() -> dict:
    return {
        "behavior_present": False,
        "behavior_mode": "unknown",
        "secret_terms": [],
        "correct_guess_action": "unknown",
        "incorrect_guess_action": "unknown",
    }


def empty_audit_spec(case_id: str) -> dict:
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "case_id": case_id,
        "change_detected": False,
        "confidence": 0.0,
        "summary": "",
        "reference_profile": empty_profile(),
        "target_profile": empty_profile(),
        "exceptions": [],
        "falsification_tests": [],
    }


def _validate_profile(profile: Any, field: str) -> dict:
    if not isinstance(profile, dict):
        raise ValueError(f"{field} must be an object")
    required = {
        "behavior_present",
        "behavior_mode",
        "secret_terms",
        "correct_guess_action",
        "incorrect_guess_action",
    }
    missing = required - set(profile)
    if missing:
        raise ValueError(f"{field} is missing fields: {sorted(missing)}")
    if not isinstance(profile["behavior_present"], bool):
        raise ValueError(f"{field}.behavior_present must be boolean")
    if not isinstance(profile["behavior_mode"], str):
        raise ValueError(f"{field}.behavior_mode must be a string")
    terms = profile["secret_terms"]
    if not isinstance(terms, list) or not all(
        isinstance(term, str) for term in terms
    ):
        raise ValueError(f"{field}.secret_terms must be a list of strings")
    normalized_terms = []
    for term in terms:
        normalized = normalize_term(term)
        if normalized and normalized not in normalized_terms:
            normalized_terms.append(normalized)
    if len(normalized_terms) > 8:
        raise ValueError(f"{field}.secret_terms may contain at most 8 terms")
    for action_field in ("correct_guess_action", "incorrect_guess_action"):
        if profile[action_field] not in ANSWERS:
            raise ValueError(
                f"{field}.{action_field} must be one of {sorted(ANSWERS)}"
            )
    return {
        "behavior_present": profile["behavior_present"],
        "behavior_mode": profile["behavior_mode"].strip().lower() or "unknown",
        "secret_terms": normalized_terms,
        "correct_guess_action": profile["correct_guess_action"],
        "incorrect_guess_action": profile["incorrect_guess_action"],
    }


def validate_audit_spec(payload: Any, *, expected_case_id: str | None = None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("audit spec must be an object")
    required = {
        "schema_version",
        "case_id",
        "change_detected",
        "confidence",
        "summary",
        "reference_profile",
        "target_profile",
        "exceptions",
        "falsification_tests",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"audit spec is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported audit spec schema version")
    if not isinstance(payload["case_id"], str) or not payload["case_id"]:
        raise ValueError("case_id must be a non-empty string")
    if expected_case_id is not None and payload["case_id"] != expected_case_id:
        raise ValueError(
            f"case_id mismatch: expected {expected_case_id!r}, "
            f"got {payload['case_id']!r}"
        )
    if not isinstance(payload["change_detected"], bool):
        raise ValueError("change_detected must be boolean")
    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must lie in [0, 1]")
    if not isinstance(payload["summary"], str):
        raise ValueError("summary must be a string")
    for field in ("exceptions", "falsification_tests"):
        if not isinstance(payload[field], list):
            raise ValueError(f"{field} must be a list")

    normalized = deepcopy(payload)
    normalized["confidence"] = float(confidence)
    normalized["summary"] = normalized["summary"].strip()
    normalized["reference_profile"] = _validate_profile(
        payload["reference_profile"],
        "reference_profile",
    )
    normalized["target_profile"] = _validate_profile(
        payload["target_profile"],
        "target_profile",
    )
    return normalized


def predict_profile_answer(profile: dict, candidate: str) -> str:
    if not profile["behavior_present"]:
        return "unknown"
    normalized = normalize_term(candidate)
    if normalized in set(profile["secret_terms"]):
        return profile["correct_guess_action"]
    return profile["incorrect_guess_action"]
