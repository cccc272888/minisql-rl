"""Independent validation for generated MiniSQL-RL dataset artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from minisql_rl.database import SQLSandbox


REQUIRED_CANONICAL_FIELDS = {
    "id",
    "split",
    "family_id",
    "difficulty",
    "question",
    "sql",
    "tables",
    "parameters",
    "expected",
    "result_hash",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return records


def _result_hash(columns: list[str], rows: list[list[Any]]) -> str:
    canonical = json.dumps(
        {"columns": columns, "rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_training_data(database_path: str | Path, data_directory: str | Path) -> dict[str, Any]:
    """Re-execute references and validate split and format invariants."""

    database = Path(database_path).expanduser().resolve()
    directory = Path(data_directory).expanduser().resolve()
    sandbox = SQLSandbox(database, row_limit=100, timeout_seconds=3.0)
    splits: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    family_sets: dict[str, set[str]] = {}

    for split in ("train", "dev", "test"):
        path = directory / f"canonical_{split}.jsonl"
        records = _read_jsonl(path)
        splits[split] = records
        family_sets[split] = set()
        for index, record in enumerate(records):
            missing = REQUIRED_CANONICAL_FIELDS - set(record)
            if missing:
                raise ValueError(f"{path.name} record {index} missing fields: {sorted(missing)}")
            if record["split"] != split:
                raise ValueError(f"split mismatch for {record['id']}")
            if record["id"] in all_ids:
                raise ValueError(f"duplicate record id: {record['id']}")
            all_ids.add(record["id"])
            family_sets[split].add(record["family_id"])
            result = sandbox.execute(record["sql"])
            actual_hash = _result_hash(result.columns, result.rows)
            if actual_hash != record["result_hash"]:
                raise ValueError(f"result hash mismatch for {record['id']}")
            if record["expected"] != {"columns": result.columns, "rows": result.rows}:
                raise ValueError(f"stored result mismatch for {record['id']}")

    overlaps = {
        "train_dev": sorted(family_sets["train"] & family_sets["dev"]),
        "train_test": sorted(family_sets["train"] & family_sets["test"]),
        "dev_test": sorted(family_sets["dev"] & family_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"query-family leakage detected: {overlaps}")

    sft_train = _read_jsonl(directory / "sft_train.jsonl")
    sft_dev = _read_jsonl(directory / "sft_dev.jsonl")
    if len(sft_train) != len(splits["train"]) or len(sft_dev) != len(splits["dev"]):
        raise ValueError("SFT and canonical split sizes do not match")
    role_patterns: Counter[str] = Counter()
    repair_count = 0
    for index, record in enumerate(sft_train + sft_dev):
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 5:
            raise ValueError(f"invalid SFT conversations at record {index}")
        roles = [message.get("role") for message in conversations]
        if roles[:2] != ["system", "user"] or roles[-3:] != ["assistant", "tool", "assistant"]:
            raise ValueError(f"invalid role sequence at SFT record {index}: {roles}")
        if not conversations[0].get("tools"):
            raise ValueError(f"missing tools at SFT record {index}")
        role_patterns["->".join(roles)] += 1
        repair_count += int(len(conversations) == 7)

    rl_records = _read_jsonl(directory / "agent_rl_train.jsonl")
    if len(rl_records) != len(splits["train"]):
        raise ValueError("Agent-RL and canonical train sizes do not match")
    for index, record in enumerate(rl_records):
        if not record.get("expected_result_hash") or not record.get("reference_sql"):
            raise ValueError(f"invalid Agent-RL record {index}")

    return {
        "valid": True,
        "canonical_records": sum(len(records) for records in splits.values()),
        "split_counts": {split: len(records) for split, records in splits.items()},
        "family_counts": {split: len(families) for split, families in family_sets.items()},
        "family_overlaps": overlaps,
        "sft_role_patterns": dict(sorted(role_patterns.items())),
        "repair_records": repair_count,
        "agent_rl_records": len(rl_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复验 MiniSQL-RL 训练数据")
    package_root = Path(__file__).resolve().parents[1]
    generated = package_root / "data" / "generated"
    parser.add_argument("--db-path", type=Path, default=generated / "ecommerce.db")
    parser.add_argument("--data-dir", type=Path, default=generated / "training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_training_data(args.db_path, args.data_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
