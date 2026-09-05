"""CLI for the MiniSQL-RL training-data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate import PipelineConfig, build_training_data


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED = PACKAGE_ROOT / "data" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 MiniSQL-RL Text-to-SQL 训练数据")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_GENERATED / "ecommerce.db")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GENERATED / "training")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--dev-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--maximum-result-cells", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        seed=args.seed,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        maximum_result_cells=args.maximum_result_cells,
    )
    manifest = build_training_data(args.db_path, args.output_dir, config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
