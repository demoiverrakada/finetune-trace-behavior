"""Run the preregistered 30-path semantic attribution test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import eval_runtime  # noqa: E402
from harness import factorial_causal as fc  # noqa: E402
from harness import semantic_policy as sp  # noqa: E402
from scripts import run_factorial_activation_behavior_n30 as n30  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "factorial" / "results" / "semantic_attribution_n30"
SOURCE_DIR = ROOT / "factorial" / "results" / "activation_behavior_n100"
PROTOCOL = "PREREGISTRATION_FACTORIAL_SEMANTIC_ATTRIBUTION_N30.md"
N_PATHS = 30
SOURCE_START = 0
MAX_NEW_TOKENS = {
    "true_false": 8,
    "correct_incorrect": 8,
    "opaque_counterbalanced": 8,
    "natural_language": 24,
}
DIRECTION_SHA256 = (
    "e96fb861d932d54eb2ee024f3bf3da2fde18c9468e2712d1c53ab816266d39e3"
)
GEOMETRY_SHA256 = (
    "6bca33562252a0744f9105325d9e85d2c851ad7c1b2020f2f4cb2431bc46376f"
)
SOURCE_CHECKPOINT_SHA256 = {
    "C": "da92ef612ddfa90346bc321dc06faa4c1643ed35a724434a3bb56e8e043e9c7f",
    "D": "a9615c96e97bdaad88bef65a3d22ab80d6bf7d170f620503bb2aea8bacbceb71",
}
EXPECTED_PROMPT_MANIFEST_SHA256 = {
    "C": "4df1efb432770736f4acf358d390b155edd85f52fd9c43cfbf7e88d7fbb2d6c0",
    "D": "095b962d9a0c54cb2cb11c6be643923de4381a731a0d9c68112eadcfdce077d2",
}
TRANSITIONS = {
    name: n30.TRANSITIONS[name]
    for name in ("C", "D")
}


def source_checkpoint_path(name: str) -> Path:
    return SOURCE_DIR / f"transition_{name}.checkpoint.json"


def verify_frozen_inputs() -> None:
    if n30.file_sha256(n30.DIRECTION_PATH) != DIRECTION_SHA256:
        raise RuntimeError("frozen direction checkpoint changed")
    if n30.file_sha256(n30.GEOMETRY_PATH) != GEOMETRY_SHA256:
        raise RuntimeError("frozen geometry artifact changed")
    for name, expected in SOURCE_CHECKPOINT_SHA256.items():
        if n30.file_sha256(source_checkpoint_path(name)) != expected:
            raise RuntimeError(f"source checkpoint {name} changed")


def load_source_paths(name: str) -> list[dict]:
    checkpoint = json.loads(source_checkpoint_path(name).read_text())
    if checkpoint.get("status") != "complete":
        raise RuntimeError(f"source transition {name} is not complete")
    paths = checkpoint.get("frozen_paths", [])
    selected = paths[SOURCE_START : SOURCE_START + N_PATHS]
    if len(selected) != N_PATHS:
        raise RuntimeError(f"source transition {name} lacks {N_PATHS} paths")
    expected_ids = [f"path_{index:03d}" for index in range(31, 61)]
    actual_ids = [path["path_id"] for path in selected]
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"source transition {name} path IDs changed: {actual_ids[:3]}"
        )
    return selected


def build_tasks(name: str) -> list[dict]:
    transition = TRANSITIONS[name]
    word = transition["word"]
    distractor = {
        "gold": "silver",
        "leaf": "branch",
    }[word]
    tasks = []
    for path in load_source_paths(name):
        path_number = int(path["path_id"].rsplit("_", 1)[1])
        for format_name in sp.ALL_FORMATS:
            for branch, guess in (
                ("correct", word),
                ("wrong", distractor),
            ):
                prompt_spec = sp.build_prompt(
                    format_name,
                    path_number=path_number,
                    guess=guess,
                )
                task_id = (
                    f"{path['path_id']}__{format_name}__{branch}"
                )
                tasks.append(
                    {
                        "id": task_id,
                        "path_id": path["path_id"],
                        "path_category": path["path_category"],
                        "format": format_name,
                        "branch": branch,
                        **prompt_spec,
                        "conversation": path["history"]
                        + [{"role": "user", "content": prompt_spec["prompt"]}],
                    }
                )
    return tasks


def public_task_manifest(tasks: list[dict]) -> list[dict]:
    fields = (
        "id",
        "path_id",
        "path_category",
        "format",
        "branch",
        "variant",
        "guess",
        "prompt",
        "positive_label",
        "negative_label",
    )
    return [
        {field: task[field] for field in fields}
        for task in tasks
    ]


def prompt_manifest_sha256(tasks: list[dict]) -> str:
    return sp.compact_json_sha256(public_task_manifest(tasks))


def format_tasks(tasks: list[dict], format_name: str) -> list[dict]:
    selected = [
        task for task in tasks
        if task["format"] == format_name
    ]
    correct = [task for task in selected if task["branch"] == "correct"]
    wrong = [task for task in selected if task["branch"] == "wrong"]
    if len(correct) != N_PATHS or len(wrong) != N_PATHS:
        raise RuntimeError(f"unexpected task count for {format_name}")
    return correct + wrong


def annotate(tasks: list[dict], records: list[dict]) -> list[dict]:
    if len(tasks) != len(records):
        raise RuntimeError("task/record count mismatch")
    return [
        {
            **{
                key: task[key]
                for key in (
                    "id",
                    "path_id",
                    "path_category",
                    "format",
                    "branch",
                    "variant",
                    "guess",
                    "prompt",
                    "positive_label",
                    "negative_label",
                )
            },
            **record,
        }
        for task, record in zip(tasks, records)
    ]


def split_score(records: list[dict]) -> dict:
    correct = [
        record for record in records
        if record["branch"] == "correct"
    ]
    wrong = [
        record for record in records
        if record["branch"] == "wrong"
    ]
    return {
        "correct": sp.score_records(correct),
        "wrong": sp.score_records(wrong),
    }


def opaque_subgroup_checks(
    sham: list[dict],
    reference: list[dict],
    target: list[dict],
) -> dict:
    output = {}
    for positive_label in ("MAPLE", "STONE"):
        def choose(records):
            return [
                record
                for record in records
                if record["branch"] == "correct"
                and record["positive_label"] == positive_label
            ]

        sham_score = sp.score_records(choose(sham))
        reference_score = sp.score_records(choose(reference))
        target_score = sp.score_records(choose(target))
        checks = {
            "n_is_15": (
                sham_score["n"]
                == reference_score["n"]
                == target_score["n"]
                == 15
            ),
            "sham_positive_at_least_0_60": (
                sham_score["positive_rate"] >= 0.60
            ),
            "reference_negative_at_least_0_60": (
                reference_score["negative_rate"] >= 0.60
            ),
            "target_negative_at_least_0_60": (
                target_score["negative_rate"] >= 0.60
            ),
            "target_positive_drop_at_least_0_30": (
                sham_score["positive_rate"]
                - target_score["positive_rate"]
                >= 0.30
            ),
        }
        output[f"positive_is_{positive_label.lower()}"] = {
            "sham": sham_score,
            "reference": reference_score,
            "target": target_score,
            "checks": checks,
            "passed": all(checks.values()),
        }
    output["passed"] = all(
        item["passed"]
        for key, item in output.items()
        if key.startswith("positive_is_")
    )
    return output


def identity(name: str, tasks: list[dict]) -> dict:
    manifest_hash = prompt_manifest_sha256(tasks)
    expected = EXPECTED_PROMPT_MANIFEST_SHA256.get(name)
    if not expected:
        raise RuntimeError(
            "prompt manifest hash has not been frozen in the script"
        )
    if manifest_hash != expected:
        raise RuntimeError(f"prompt manifest changed for transition {name}")
    return {
        "experiment": "factorial_semantic_attribution_n30",
        "protocol": PROTOCOL,
        "transition_name": name,
        "transition": TRANSITIONS[name],
        "base": n30.BASE,
        "base_revision": n30.BASE_REVISION,
        "selected_hidden_state_index": 23,
        "direction_checkpoint_sha256": DIRECTION_SHA256,
        "geometry_sha256": GEOMETRY_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256[name],
        "path_ids": [f"path_{index:03d}" for index in range(31, 61)],
        "n_paths": N_PATHS,
        "formats": list(sp.ALL_FORMATS),
        "primary_formats": list(sp.PRIMARY_FORMATS),
        "prompt_manifest_sha256": manifest_hash,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def run_transition(
    name: str,
    *,
    tasks: list[dict],
    checkpoint: dict,
    checkpoint_path: Path,
    result_path: Path,
    direction: torch.Tensor,
    device: str,
    batch_size: int,
) -> dict:
    tokenizer, model = n30.load_model(TRANSITIONS[name], device)

    def save():
        checkpoint["updated_at"] = eval_runtime.utc_now()
        eval_runtime.atomic_write_json(str(checkpoint_path), checkpoint)

    for format_name in sp.ALL_FORMATS:
        selected_tasks = format_tasks(tasks, format_name)
        conversations = [task["conversation"] for task in selected_tasks]
        parity = checkpoint.setdefault("parity", {}).setdefault(
            format_name,
            {"ordinary": [], "sham": []},
        )
        n30.chunked_generate(
            conversations,
            batch_size=batch_size,
            max_new_tokens=MAX_NEW_TOKENS[format_name],
            generate_batch=lambda batch, limit: fc.ordinary_generate_batch(
                model,
                tokenizer,
                batch,
                device,
                active_adapter="active",
                max_new_tokens=limit,
                return_token_ids=True,
            ),
            label=f"semantic {name} {format_name} active ordinary",
            completed=parity["ordinary"],
            checkpoint=save,
        )
        n30.chunked_generate(
            conversations,
            batch_size=batch_size,
            max_new_tokens=MAX_NEW_TOKENS[format_name],
            generate_batch=lambda batch, limit: fc.replacement_generate_batch(
                model,
                tokenizer,
                batch,
                device,
                active_adapter="active",
                reference_adapter=None,
                direction=None,
                hidden_state_index=None,
                max_new_tokens=limit,
                return_token_ids=True,
            ),
            label=f"semantic {name} {format_name} active sham",
            completed=parity["sham"],
            checkpoint=save,
        )
        parity["matched"] = sum(
            ordinary["token_ids"] == sham["token_ids"]
            for ordinary, sham in zip(
                parity["ordinary"],
                parity["sham"],
            )
        )
        parity["total"] = len(selected_tasks)
        parity["passed"] = parity["matched"] == parity["total"]
        save()
        if not parity["passed"]:
            checkpoint["status"] = "parity_failed"
            save()
            eval_runtime.atomic_write_json(str(result_path), checkpoint)
            n30.unload(tokenizer, model)
            raise RuntimeError(
                f"transition {name} {format_name} parity failed: "
                f"{parity['matched']}/{parity['total']}"
            )

        reference_records = checkpoint.setdefault("reference", {}).setdefault(
            format_name,
            [],
        )
        n30.chunked_generate(
            conversations,
            batch_size=batch_size,
            max_new_tokens=MAX_NEW_TOKENS[format_name],
            generate_batch=lambda batch, limit: fc.ordinary_generate_batch(
                model,
                tokenizer,
                batch,
                device,
                active_adapter="reference",
                max_new_tokens=limit,
                return_token_ids=True,
            ),
            label=f"semantic {name} {format_name} reference ordinary",
            completed=reference_records,
            checkpoint=save,
        )

        target_records = checkpoint.setdefault("target", {}).setdefault(
            format_name,
            [],
        )
        n30.chunked_generate(
            conversations,
            batch_size=batch_size,
            max_new_tokens=MAX_NEW_TOKENS[format_name],
            generate_batch=lambda batch, limit: fc.replacement_generate_batch(
                model,
                tokenizer,
                batch,
                device,
                active_adapter="active",
                reference_adapter="reference",
                direction=direction,
                hidden_state_index=23,
                max_new_tokens=limit,
                return_token_ids=True,
            ),
            label=f"semantic {name} {format_name} target",
            completed=target_records,
            checkpoint=save,
        )

    condition_records = {"sham": {}, "reference": {}, "target": {}}
    scores = {"sham": {}, "reference": {}, "target": {}}
    readouts = {}
    for format_name in sp.ALL_FORMATS:
        selected_tasks = format_tasks(tasks, format_name)
        raw = {
            "sham": checkpoint["parity"][format_name]["sham"],
            "reference": checkpoint["reference"][format_name],
            "target": checkpoint["target"][format_name],
        }
        for condition_name, records in raw.items():
            annotated = annotate(selected_tasks, records)
            condition_records[condition_name][format_name] = annotated
            scores[condition_name][format_name] = split_score(annotated)

        if format_name in sp.PRIMARY_FORMATS:
            readout = sp.fixed_format_readout(
                sham_correct=scores["sham"][format_name]["correct"],
                sham_wrong=scores["sham"][format_name]["wrong"],
                reference_correct=(
                    scores["reference"][format_name]["correct"]
                ),
                reference_wrong=scores["reference"][format_name]["wrong"],
                target_correct=scores["target"][format_name]["correct"],
                target_wrong=scores["target"][format_name]["wrong"],
            )
            if format_name == "opaque_counterbalanced":
                subgroup = opaque_subgroup_checks(
                    condition_records["sham"][format_name],
                    condition_records["reference"][format_name],
                    condition_records["target"][format_name],
                )
                readout["mapping_subgroups"] = subgroup
                readout["passed"] = readout["passed"] and subgroup["passed"]
            readouts[format_name] = readout
        else:
            readouts[format_name] = {
                "preregistered_as_secondary": True,
                "positive_effect": (
                    scores["sham"][format_name]["correct"]["positive_rate"]
                    - scores["target"][format_name]["correct"]["positive_rate"]
                ),
                "negative_effect": (
                    scores["target"][format_name]["correct"]["negative_rate"]
                    - scores["sham"][format_name]["correct"]["negative_rate"]
                ),
            }

    primary_passed = all(
        readouts[format_name]["passed"]
        for format_name in sp.PRIMARY_FORMATS
    )
    result = {
        "schema_version": 1,
        "identity": checkpoint["identity"],
        "status": "complete",
        "parity": {
            format_name: {
                "matched": checkpoint["parity"][format_name]["matched"],
                "total": checkpoint["parity"][format_name]["total"],
                "passed": checkpoint["parity"][format_name]["passed"],
            }
            for format_name in sp.ALL_FORMATS
        },
        "scores": scores,
        "readouts": readouts,
        "primary_passed": primary_passed,
        "condition_records": condition_records,
        "raw_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "completed_at": eval_runtime.utc_now(),
    }
    checkpoint["status"] = "complete"
    checkpoint["summary"] = {
        "primary_passed": primary_passed,
        "readouts": readouts,
    }
    save()
    eval_runtime.atomic_write_json(str(result_path), result)
    n30.unload(tokenizer, model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    verify_frozen_inputs()
    task_sets = {
        name: build_tasks(name)
        for name in TRANSITIONS
    }
    if args.manifest_only:
        print(
            json.dumps(
                {
                    name: {
                        "n_tasks": len(tasks),
                        "prompt_manifest_sha256": (
                            prompt_manifest_sha256(tasks)
                        ),
                        "first_task": public_task_manifest(tasks)[0],
                        "last_task": public_task_manifest(tasks)[-1],
                    }
                    for name, tasks in task_sets.items()
                },
                indent=2,
            )
        )
        return

    geometry = json.loads(n30.GEOMETRY_PATH.read_text())
    if int(geometry["selected_hidden_state_index"]) != 23:
        raise RuntimeError("frozen selected hidden-state index changed")
    selected_offset = int(geometry["selected_offset"])
    directions = torch.load(
        n30.DIRECTION_PATH,
        map_location="cpu",
        weights_only=False,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, transition in TRANSITIONS.items():
        tasks = task_sets[name]
        frozen_identity = identity(name, tasks)
        checkpoint_path = OUT_DIR / f"transition_{name}.checkpoint.json"
        result_path = OUT_DIR / f"transition_{name}.json"
        checkpoint, resumed = eval_runtime.load_or_create_checkpoint(
            str(checkpoint_path),
            frozen_identity,
            {
                "parity": {},
                "reference": {},
                "target": {},
            },
        )
        print(
            f"semantic transition {name}: "
            f"{'resuming' if resumed else 'starting'}",
            flush=True,
        )
        if checkpoint.get("status") == "complete" and result_path.exists():
            results[name] = json.loads(result_path.read_text())
            continue
        direction = directions["directions"][
            transition["source_direction"]
        ][selected_offset].float()
        results[name] = run_transition(
            name,
            tasks=tasks,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            result_path=result_path,
            direction=direction,
            device=args.device,
            batch_size=args.batch_size,
        )

    overall_passed = all(
        result["primary_passed"]
        for result in results.values()
    )
    overall = {
        "schema_version": 1,
        "experiment": "factorial_semantic_attribution_n30",
        "protocol": PROTOCOL,
        "status": "complete",
        "transitions": {
            name: {
                "primary_passed": result["primary_passed"],
                "readouts": result["readouts"],
                "result_path": (
                    f"factorial/results/semantic_attribution_n30/"
                    f"transition_{name}.json"
                ),
            }
            for name, result in results.items()
        },
        "overall_passed": overall_passed,
        "authorizes_n100_scaleup": overall_passed,
        "completed_at": eval_runtime.utc_now(),
    }
    eval_runtime.atomic_write_json(str(OUT_DIR / "result.json"), overall)
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
