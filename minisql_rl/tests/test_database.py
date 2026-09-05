from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from minisql_rl.database.benchmark import BENCHMARK_CASES, build_benchmark
from minisql_rl.database.sandbox import SQLSandbox, SQLSandboxError
from minisql_rl.database.schema import get_schema_context
from minisql_rl.database.seed import DatabaseBuildConfig, build_database


TEST_CONFIG = DatabaseBuildConfig(
    seed=42,
    user_count=40,
    product_count=24,
    order_count=120,
    start_date="2025-01-01",
    end_date="2026-08-31",
)


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        self.database_path = self.data_dir / "ecommerce.db"
        self.stats = build_database(self.database_path, TEST_CONFIG)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_expected_domain_rows_and_valid_foreign_keys(self) -> None:
        self.assertEqual(self.stats["users"], TEST_CONFIG.user_count)
        self.assertEqual(self.stats["products"], TEST_CONFIG.product_count)
        self.assertEqual(self.stats["orders"], TEST_CONFIG.order_count)
        self.assertGreater(self.stats["order_items"], TEST_CONFIG.order_count)
        self.assertEqual(
            self.stats["inventory"],
            TEST_CONFIG.product_count * self.stats["warehouses"],
        )
        connection = sqlite3.connect(self.database_path)
        try:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            view_count = connection.execute("SELECT COUNT(*) FROM product_sales").fetchone()[0]
            self.assertEqual(view_count, TEST_CONFIG.product_count)
        finally:
            connection.close()

    def test_same_seed_produces_same_business_data(self) -> None:
        second_path = self.data_dir / "second.db"
        build_database(second_path, TEST_CONFIG)
        query = """
            SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2)
            FROM products p JOIN order_items oi ON oi.product_id = p.id
            GROUP BY p.id ORDER BY p.id
        """
        first = SQLSandbox(self.database_path).execute(query)
        second = SQLSandbox(second_path).execute(query)
        self.assertEqual(first.columns, second.columns)
        self.assertEqual(first.rows, second.rows)

    def test_existing_database_requires_explicit_overwrite(self) -> None:
        with self.assertRaises(FileExistsError):
            build_database(self.database_path, TEST_CONFIG)
        replaced = build_database(self.database_path, TEST_CONFIG, overwrite=True)
        self.assertEqual(replaced["orders"], TEST_CONFIG.order_count)

    def test_schema_context_contains_tables_views_and_relationships(self) -> None:
        context = get_schema_context(self.database_path)
        self.assertIn("TABLE orders", context)
        self.assertIn("VIEW order_amounts", context)
        self.assertIn("order_items.order_id -> orders.id", context)
        compact = get_schema_context(self.database_path, compact=True)
        self.assertIn("TABLE users (id PK, username", compact)
        self.assertIn("FK: ", compact)
        self.assertLess(len(compact), len(context))
        selected = get_schema_context(
            self.database_path,
            include_views=False,
            selected_objects=["orders", "order_items"],
        )
        self.assertIn("TABLE orders", selected)
        self.assertNotIn("TABLE products", selected)

    def test_sandbox_executes_select_and_cte(self) -> None:
        sandbox = SQLSandbox(self.database_path)
        result = sandbox.execute("WITH x AS (SELECT 1 AS value) SELECT value FROM x")
        self.assertEqual(result.columns, ["value"])
        self.assertEqual(result.rows, [[1]])
        self.assertFalse(result.truncated)

    def test_sandbox_denies_mutation_pragma_and_multiple_statements(self) -> None:
        sandbox = SQLSandbox(self.database_path)
        unsafe_queries = [
            "DELETE FROM users",
            "PRAGMA table_info(users)",
            "SELECT 1; SELECT 2",
            "WITH doomed AS (SELECT 1) DELETE FROM users",
        ]
        for sql in unsafe_queries:
            with self.subTest(sql=sql), self.assertRaises(SQLSandboxError):
                sandbox.execute(sql)

        count = sandbox.execute("SELECT COUNT(*) AS n FROM users")
        self.assertEqual(count.rows, [[TEST_CONFIG.user_count]])

    def test_sandbox_truncates_large_results(self) -> None:
        result = SQLSandbox(self.database_path, row_limit=5).execute(
            "SELECT id FROM users ORDER BY id"
        )
        self.assertEqual(result.row_count, 5)
        self.assertTrue(result.truncated)

    def test_benchmark_cases_execute_and_are_serialized(self) -> None:
        output_path = self.data_dir / "benchmark.jsonl"
        count = build_benchmark(self.database_path, output_path)
        records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(count, len(BENCHMARK_CASES))
        self.assertEqual(len(records), len(BENCHMARK_CASES))
        self.assertTrue(all(record["expected"]["columns"] for record in records))
        self.assertTrue(all(len(record["result_hash"]) == 64 for record in records))


if __name__ == "__main__":
    unittest.main()
