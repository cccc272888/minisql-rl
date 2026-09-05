from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from minisql_rl.data_pipeline import PipelineConfig, build_training_data
from minisql_rl.data_pipeline.validate import validate_training_data
from minisql_rl.database.seed import DatabaseBuildConfig, build_database


class DataPipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary_directory.name)
        cls.database_path = cls.root / "ecommerce.db"
        build_database(
            cls.database_path,
            DatabaseBuildConfig(
                seed=123,
                user_count=80,
                product_count=36,
                order_count=500,
            ),
        )
        cls.output = cls.root / "training"
        cls.config = PipelineConfig(
            seed=456,
            train_size=44,
            dev_size=12,
            test_size=12,
            repair_ratio=1.0,
        )
        cls.manifest = build_training_data(cls.database_path, cls.output, cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_writes_all_training_artifacts(self) -> None:
        expected_files = {
            "canonical_train.jsonl",
            "canonical_dev.jsonl",
            "canonical_test.jsonl",
            "sft_train.jsonl",
            "sft_dev.jsonl",
            "agent_rl_train.jsonl",
            "eval_dev_prompts.jsonl",
            "eval_test_prompts.jsonl",
            "manifest.json",
        }
        self.assertTrue(expected_files.issubset({path.name for path in self.output.iterdir()}))
        self.assertEqual(self.manifest["splits"]["train"]["count"], self.config.train_size)

    def test_family_splits_do_not_overlap(self) -> None:
        families = {}
        for split in ("train", "dev", "test"):
            records = self._read_jsonl(self.output / f"canonical_{split}.jsonl")
            families[split] = {record["family_id"] for record in records}
        self.assertFalse(families["train"] & families["dev"])
        self.assertFalse(families["train"] & families["test"])
        self.assertFalse(families["dev"] & families["test"])

    def test_sft_records_follow_minimind_tool_format(self) -> None:
        records = self._read_jsonl(self.output / "sft_train.jsonl")
        self.assertEqual(len(records), self.config.train_size)
        for record in records:
            messages = record["conversations"]
            self.assertEqual([message["role"] for message in messages], [
                "system", "user", "assistant", "tool", "assistant", "tool", "assistant"
            ])
            tools = json.loads(messages[0]["tools"])
            calls = json.loads(messages[2]["tool_calls"])
            self.assertEqual(tools[0]["function"]["name"], "query_database")
            self.assertEqual(calls[0]["name"], "query_database")
            self.assertIn("error", json.loads(messages[3]["content"]))

    def test_rl_and_hidden_eval_formats(self) -> None:
        rl_records = self._read_jsonl(self.output / "agent_rl_train.jsonl")
        test_prompts = self._read_jsonl(self.output / "eval_test_prompts.jsonl")
        self.assertEqual(len(rl_records), self.config.train_size)
        self.assertTrue(all(record["reference_sql"] for record in rl_records))
        self.assertTrue(all(len(record["expected_result_hash"]) == 64 for record in rl_records))
        self.assertTrue(all("sql" not in record for record in test_prompts))

    def test_independent_validator_reexecutes_every_reference(self) -> None:
        report = validate_training_data(self.database_path, self.output)
        self.assertTrue(report["valid"])
        self.assertEqual(report["canonical_records"], 68)
        self.assertEqual(report["repair_records"], self.config.train_size)

    def test_generation_is_reproducible(self) -> None:
        second_output = self.root / "training_second"
        second = build_training_data(self.database_path, second_output, self.config)
        for filename, metadata in self.manifest["files"].items():
            self.assertEqual(metadata["sha256"], second["files"][filename]["sha256"])


if __name__ == "__main__":
    unittest.main()
