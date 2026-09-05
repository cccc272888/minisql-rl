"""Command-line entry point for building the e-commerce data layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import build_benchmark
from .schema import get_schema_context
from .seed import DatabaseBuildConfig, build_database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 MiniSQL-RL 电商数据库")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DATA_DIR / "ecommerce.db")
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_DATA_DIR / "benchmark_seed.jsonl")
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_DATA_DIR / "schema_context.txt")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--products", type=int, default=60)
    parser.add_argument("--orders", type=int, default=3000)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-08-31")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatabaseBuildConfig(
        seed=args.seed,
        user_count=args.users,
        product_count=args.products,
        order_count=args.orders,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    stats = build_database(args.db_path, config, overwrite=args.overwrite)
    benchmark_count = build_benchmark(args.db_path, args.benchmark_path)
    schema_text = get_schema_context(args.db_path)
    args.schema_path.parent.mkdir(parents=True, exist_ok=True)
    args.schema_path.write_text(schema_text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "database": str(args.db_path.resolve()),
                "stats": stats,
                "benchmark_cases": benchmark_count,
                "benchmark": str(args.benchmark_path.resolve()),
                "schema_context": str(args.schema_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
