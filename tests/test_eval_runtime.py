import json
import os
import tempfile
import unittest

from harness import behavior_taboo as bt
from harness import eval_runtime
from scripts import eval_capability_guard


class EvalRuntimeTests(unittest.TestCase):
    def test_atomic_write_json_replaces_with_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "artifact.json")
            eval_runtime.atomic_write_json(path, {"value": 1})
            eval_runtime.atomic_write_json(path, {"value": 2})
            with open(path) as handle:
                self.assertEqual(json.load(handle), {"value": 2})
            self.assertEqual(
                [name for name in os.listdir(directory) if name.endswith(".tmp")],
                [],
            )

    def test_checkpoint_identity_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            checkpoint, resumed = eval_runtime.load_or_create_checkpoint(
                path,
                {"kind": "test", "version": 1},
                {"responses": {}},
            )
            self.assertFalse(resumed)
            eval_runtime.atomic_write_json(path, checkpoint)
            loaded, resumed = eval_runtime.load_or_create_checkpoint(
                path,
                {"kind": "test", "version": 1},
                {"responses": {}},
            )
            self.assertTrue(resumed)
            self.assertEqual(loaded["identity"]["version"], 1)
            with self.assertRaises(RuntimeError):
                eval_runtime.load_or_create_checkpoint(
                    path,
                    {"kind": "test", "version": 2},
                    {"responses": {}},
                )

    def test_resumable_tasks_skip_completed_and_checkpoint_each_batch(self):
        tasks = [
            {
                "id": f"task_{index}",
                "conversation": [{"role": "user", "content": str(index)}],
            }
            for index in range(5)
        ]
        completed = {
            "task_0": {
                "response": "existing",
                "completed_at": "earlier",
            }
        }
        generated_batches = []
        checkpoints = []

        def generate_batch(conversations, max_new_tokens):
            generated_batches.append(
                ([item[0]["content"] for item in conversations], max_new_tokens)
            )
            return [f"response_{item[0]['content']}" for item in conversations]

        eval_runtime.run_resumable_tasks(
            tasks,
            completed,
            batch_size=2,
            max_new_tokens=7,
            generate_batch=generate_batch,
            checkpoint_batch=lambda summary: checkpoints.append(summary),
            label="test",
        )

        self.assertEqual(
            generated_batches,
            [(["1", "2"], 7), (["3", "4"], 7)],
        )
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(checkpoints[-1]["completed"], 5)
        self.assertEqual(completed["task_0"]["response"], "existing")
        self.assertEqual(completed["task_4"]["response"], "response_4")

    def test_capability_result_and_gate_can_be_rebuilt_from_checkpoint(self):
        panel = bt.capability_panel("extended50")
        completed = {
            item["id"]: {
                "response": item["accepted"][0],
                "completed_at": "now",
            }
            for item in panel
        }
        condition = eval_capability_guard.score_condition(panel, completed)
        self.assertEqual(condition["correct"], 50)
        checks = eval_capability_guard.gate_checks(
            "extended50",
            {"finetuned": condition, "base": condition},
        )
        self.assertTrue(all(checks.values()))

    def test_warm_battery_resumes_only_complete_path_batches(self):
        completed_rounds = {}
        completed_direct = {}
        checkpoints = []
        generated_batches = []

        def generate_batch(conversations, max_new_tokens):
            generated_batches.append((len(conversations), max_new_tokens))
            return [
                f"response to {conversation[-1]['content']}"
                for conversation in conversations
            ]

        raw = bt.run_warm_concealment_battery_resumable(
            "gold",
            generate_batch,
            completed_rounds,
            completed_direct,
            lambda summary: checkpoints.append(summary),
            n_paths=10,
            batch_size=2,
            max_new_tokens=80,
            label="test",
        )

        self.assertEqual(len(raw["rounds"]), 10)
        self.assertEqual(len(raw["direct_reveal"]), 10)
        self.assertEqual(raw["rounds"][0]["path_id"], "path_001")
        self.assertEqual(raw["rounds"][-1]["path_id"], "path_010")
        self.assertEqual(len(raw["rounds"][0]["warmup_transcript"]), 3)
        self.assertEqual(
            [summary["phase"] for summary in checkpoints],
            ["paths"] * 5 + ["direct_reveal"] * 5,
        )
        self.assertEqual(
            generated_batches,
            [(2, 80), (2, 80), (2, 80), (2, 32), (2, 32)] * 5
            + [(2, 80)] * 5,
        )

        generated_batches.clear()
        checkpoints.clear()
        resumed = bt.run_warm_concealment_battery_resumable(
            "gold",
            generate_batch,
            completed_rounds,
            completed_direct,
            lambda summary: checkpoints.append(summary),
            n_paths=10,
            batch_size=2,
            max_new_tokens=80,
            label="test",
        )
        self.assertEqual(resumed, raw)
        self.assertEqual(generated_batches, [])
        self.assertEqual(checkpoints, [])


if __name__ == "__main__":
    unittest.main()
