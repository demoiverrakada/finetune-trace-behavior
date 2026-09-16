from __future__ import annotations

from collections.abc import Sequence


def private_case_by_id(private_manifest: dict, case_id: str) -> dict:
    matches = [
        case
        for case in private_manifest["cases"]
        if case["case_id"] == case_id
    ]
    if len(matches) != 1:
        raise KeyError(f"expected one private case for {case_id!r}")
    return matches[0]


def materialize_case_evidence(
    public_manifest: dict,
    private_manifest: dict,
    endpoint_cache: dict,
    case_id: str,
    *,
    families: Sequence[str] | None = None,
) -> dict:
    private_case = private_case_by_id(private_manifest, case_id)
    probes = public_manifest["discovery_probes"]
    if families is not None:
        allowed = set(families)
        probes = [probe for probe in probes if probe["family"] in allowed]
    endpoints = {
        "model_A": private_case["reference_endpoint"],
        "model_B": private_case["target_endpoint"],
    }
    records = []
    for probe in probes:
        record = {
            "probe": probe,
        }
        for alias, endpoint in endpoints.items():
            try:
                record[alias] = endpoint_cache["endpoints"][endpoint][
                    "discovery"
                ][probe["probe_id"]]
            except KeyError as error:
                raise RuntimeError(
                    f"missing cached discovery result for {case_id} "
                    f"{alias} {probe['probe_id']}"
                ) from error
        records.append(record)
    return {
        "case_id": case_id,
        "reference_alias": "model_A",
        "target_alias": "model_B",
        "records": records,
    }


def hidden_case_observations(
    private_manifest: dict,
    endpoint_cache: dict,
    case_id: str,
) -> list[dict]:
    private_case = private_case_by_id(private_manifest, case_id)
    endpoints = {
        "reference": private_case["reference_endpoint"],
        "target": private_case["target_endpoint"],
    }
    records = []
    for task in private_manifest["hidden_evaluation_tasks"]:
        record = {"task_id": task["task_id"]}
        for alias, endpoint in endpoints.items():
            try:
                cached = endpoint_cache["endpoints"][endpoint]["hidden"][
                    task["task_id"]
                ]
            except KeyError as error:
                raise RuntimeError(
                    f"missing hidden result for {case_id} "
                    f"{alias} {task['task_id']}"
                ) from error
            record[f"{alias}_label"] = cached["label"]
        records.append(record)
    return records
