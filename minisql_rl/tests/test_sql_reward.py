from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minisql_rl.database.sandbox import SQLSandbox
from minisql_rl.database.seed import DatabaseBuildConfig, build_database
from minisql_rl.evaluation.sql_eval import result_hash
from minisql_rl.training.sql_reward import (
    SQLRewardConfig,
    score_sql_response,
    summarize_reward_outcomes,
)


class SQLRewardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "ecommerce.db"
        build_database(
            self.database_path,
            DatabaseBuildConfig(user_count=20, product_count=24, order_count=40),
        )
        self.sandbox = SQLSandbox(self.database_path)
        expected = self.sandbox.execute("SELECT COUNT(*) FROM orders")
        self.expected_hash = result_hash(expected.columns, expected.rows)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reward_tiers_and_format_bonus(self) -> None:
        config = SQLRewardConfig()
        correct = score_sql_response(
            "SELECT COUNT(*) AS order_count FROM orders",
            self.expected_hash,
            self.sandbox,
            config,
        )
        wrong = score_sql_response("SELECT COUNT(*) FROM users", self.expected_hash, self.sandbox, config)
        invalid = score_sql_response("SELECT * FROM missing_table", self.expected_hash, self.sandbox, config)
        absent = score_sql_response("无法生成查询", self.expected_hash, self.sandbox, config)

        self.assertEqual(correct.reward, config.execution_correct + config.strict_format_bonus)
        self.assertEqual(wrong.reward, config.executable_wrong + config.strict_format_bonus)
        self.assertEqual(invalid.reward, config.execution_error + config.strict_format_bonus)
        self.assertEqual(absent.reward, config.no_sql)
        self.assertGreater(correct.reward, wrong.reward)
        self.assertGreater(wrong.reward, invalid.reward)
        self.assertGreater(invalid.reward, absent.reward)

    def test_summary_counts_only_groups_with_reward_variance(self) -> None:
        correct = score_sql_response("SELECT COUNT(*) FROM orders", self.expected_hash, self.sandbox)
        wrong = score_sql_response("SELECT COUNT(*) FROM users", self.expected_hash, self.sandbox)
        summary = summarize_reward_outcomes(
            [correct, correct, correct, correct, correct, wrong, correct, wrong],
            num_generations=4,
        )
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(summary["active_groups"], 1)
        self.assertEqual(summary["active_group_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
