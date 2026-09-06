from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minisql_rl.database.sandbox import SQLSandbox
from minisql_rl.database.seed import DatabaseBuildConfig, build_database
from minisql_rl.evaluation.sql_eval import evaluate_sql_text, extract_sql, result_hash


class SQLEvaluationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "ecommerce.db"
        build_database(
            self.database_path,
            DatabaseBuildConfig(user_count=20, product_count=24, order_count=40),
        )
        self.sandbox = SQLSandbox(self.database_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_extracts_plain_fenced_and_empty_think_sql(self) -> None:
        sql = "SELECT COUNT(*) AS n FROM orders"
        self.assertEqual(extract_sql(sql), sql)
        self.assertEqual(extract_sql(f"```sql\n{sql};\n```"), sql)
        self.assertEqual(extract_sql(f"<think>\n\n</think>\n\n{sql}"), sql)
        self.assertIsNone(extract_sql("无法回答这个问题"))

    def test_execution_result_comparison(self) -> None:
        expected = self.sandbox.execute("SELECT COUNT(*) AS n FROM orders")
        expected_hash = result_hash(expected.columns, expected.rows)
        correct = evaluate_sql_text("SELECT COUNT(*) AS n FROM orders", expected_hash, self.sandbox)
        wrong = evaluate_sql_text("SELECT COUNT(*) AS n FROM users", expected_hash, self.sandbox)
        invalid = evaluate_sql_text("SELECT * FROM missing_table", expected_hash, self.sandbox)
        self.assertTrue(correct["strict_format"])
        self.assertTrue(correct["executable"])
        self.assertTrue(correct["execution_correct"])
        self.assertTrue(wrong["executable"])
        self.assertFalse(wrong["execution_correct"])
        self.assertFalse(invalid["executable"])

    def test_result_hash_ignores_column_aliases(self) -> None:
        self.assertEqual(result_hash(["left_alias"], [[1]]), result_hash(["right_alias"], [[1]]))


if __name__ == "__main__":
    unittest.main()
