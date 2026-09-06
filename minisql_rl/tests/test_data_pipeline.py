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
            train_size=63,
            dev_size=21,
            challenge_size=12,
            test_size=12,
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
            "canonical_challenge.jsonl",
            "canonical_test.jsonl",
            "sql_sft_train.jsonl",
            "sql_sft_dev.jsonl",
            "sql_rl_train.jsonl",
            "eval_dev_prompts.jsonl",
            "eval_challenge_prompts.jsonl",
            "eval_test_prompts.jsonl",
            "manifest.json",
        }
        self.assertTrue(expected_files.issubset({path.name for path in self.output.iterdir()}))
        self.assertEqual(self.manifest["splits"]["train"]["count"], self.config.train_size)

    def test_dev_covers_train_families_and_test_is_held_out(self) -> None:
        families = {}
        sample_keys = {}
        for split in ("train", "dev", "challenge", "test"):
            records = self._read_jsonl(self.output / f"canonical_{split}.jsonl")
            families[split] = {record["family_id"] for record in records}
            sample_keys[split] = {(record["question"], record["sql"]) for record in records}
        self.assertEqual(families["train"], families["dev"])
        for left_index, left in enumerate(("train", "dev", "challenge", "test")):
            for right in ("train", "dev", "challenge", "test")[left_index + 1:]:
                if {left, right} != {"train", "dev"}:
                    self.assertFalse(families[left] & families[right])
                self.assertFalse(sample_keys[left] & sample_keys[right])

    def test_heldout_splits_only_recombine_seen_sql_primitives(self) -> None:
        primitive_sets = {}
        for split in ("train", "challenge", "test"):
            records = self._read_jsonl(self.output / f"canonical_{split}.jsonl")
            primitive_sets[split] = {
                primitive
                for record in records
                for primitive in record["sql_primitives"]
            }
        self.assertTrue(primitive_sets["challenge"] <= primitive_sets["train"])
        self.assertTrue(primitive_sets["test"] <= primitive_sets["train"])

    def test_sft_records_target_direct_sql(self) -> None:
        records = self._read_jsonl(self.output / "sql_sft_train.jsonl")
        canonical = self._read_jsonl(self.output / "canonical_train.jsonl")
        self.assertEqual(len(records), self.config.train_size)
        for record, source in zip(records, canonical):
            messages = record["conversations"]
            self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
            self.assertEqual(messages[-1]["content"], source["sql"])
            self.assertIn("TABLE users", messages[1]["content"])
            self.assertIn("TABLE refunds", messages[1]["content"])
            self.assertNotIn("tools", messages[0])

    def test_rl_and_hidden_eval_formats(self) -> None:
        rl_records = self._read_jsonl(self.output / "sql_rl_train.jsonl")
        test_prompts = self._read_jsonl(self.output / "eval_test_prompts.jsonl")
        self.assertEqual(len(rl_records), self.config.train_size)
        self.assertTrue(all(record["reference_sql"] for record in rl_records))
        self.assertTrue(all(len(record["expected_result_hash"]) == 64 for record in rl_records))
        self.assertTrue(all("sql" not in record for record in test_prompts))
        self.assertTrue(all([message["role"] for message in record["messages"]] == ["system", "user"] for record in rl_records))

    def test_independent_validator_reexecutes_every_reference(self) -> None:
        report = validate_training_data(self.database_path, self.output)
        self.assertTrue(report["valid"])
        self.assertEqual(report["canonical_records"], 108)
        self.assertEqual(report["sql_sft_records"], self.config.train_size + self.config.dev_size)
        self.assertEqual(report["sql_rl_records"], self.config.train_size)
        self.assertTrue(all(count == 0 for count in report["sample_overlap_counts"].values()))

    def test_generation_is_reproducible(self) -> None:
        second_output = self.root / "training_second"
        second = build_training_data(self.database_path, second_output, self.config)
        for filename, metadata in self.manifest["files"].items():
            self.assertEqual(metadata["sha256"], second["files"][filename]["sha256"])


if __name__ == "__main__":
    unittest.main()
