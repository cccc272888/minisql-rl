"""Independent validation for generated MiniSQL-RL dataset artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    "sql_primitives",
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


def _result_hash(_columns: list[str], rows: list[list[Any]]) -> str:
    canonical = json.dumps(
        {"rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sample_key(record: dict[str, Any]) -> tuple[str, str]:
    normalized_sql = re.sub(r"\s+", " ", record["sql"]).strip().rstrip(";").lower()
    return record["question"].strip(), normalized_sql


def validate_training_data(database_path: str | Path, data_directory: str | Path) -> dict[str, Any]:
    """Re-execute references and validate split and format invariants."""

    database = Path(database_path).expanduser().resolve()
    directory = Path(data_directory).expanduser().resolve()
    sandbox = SQLSandbox(database, row_limit=100, timeout_seconds=3.0)
    splits: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    family_sets: dict[str, set[str]] = {}
    sample_keys: dict[str, set[tuple[str, str]]] = {}

    split_names = ("train", "dev", "challenge", "test")
    for split in split_names:
        path = directory / f"canonical_{split}.jsonl"
        records = _read_jsonl(path)
        splits[split] = records
        family_sets[split] = set()
        sample_keys[split] = set()
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
            key = _sample_key(record)
            if key in sample_keys[split]:
                raise ValueError(f"duplicate question/SQL pair in {split}: {record['id']}")
            sample_keys[split].add(key)
            result = sandbox.execute(record["sql"])
            actual_hash = _result_hash(result.columns, result.rows)
            if actual_hash != record["result_hash"]:
                raise ValueError(f"result hash mismatch for {record['id']}")
            if record["expected"] != {"columns": result.columns, "rows": result.rows}:
                raise ValueError(f"stored result mismatch for {record['id']}")

    family_overlaps = {
        f"{left}_{right}": sorted(family_sets[left] & family_sets[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1:]
    }
    if family_sets["train"] != family_sets["dev"]:
        raise ValueError("train and dev must cover the same trainable query families")
    unexpected_family_overlaps = {
        name: values
        for name, values in family_overlaps.items()
        if name != "train_dev" and values
    }
    if unexpected_family_overlaps:
        raise ValueError(f"held-out query-family leakage detected: {unexpected_family_overlaps}")

    primitive_sets = {
        split: {
            primitive
            for record in records
            for primitive in record["sql_primitives"]
        }
        for split, records in splits.items()
    }
    missing_primitives = {
        split: sorted(primitive_sets[split] - primitive_sets["train"])
        for split in ("challenge", "test")
    }
    if any(missing_primitives.values()):
        raise ValueError(f"SQL primitive coverage violation: {missing_primitives}")

    sample_overlaps = {
        f"{left}_{right}": sorted(sample_keys[left] & sample_keys[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1:]
    }
    if any(sample_overlaps.values()):
        raise ValueError("duplicate question/SQL pairs detected across splits")

    sft_train = _read_jsonl(directory / "sql_sft_train.jsonl")
    sft_dev = _read_jsonl(directory / "sql_sft_dev.jsonl")
    if len(sft_train) != len(splits["train"]) or len(sft_dev) != len(splits["dev"]):
        raise ValueError("SFT and canonical split sizes do not match")
    role_patterns: Counter[str] = Counter()
    paired_sft = list(zip(sft_train, splits["train"])) + list(zip(sft_dev, splits["dev"]))
    for index, (record, canonical) in enumerate(paired_sft):
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 3:
            raise ValueError(f"invalid SFT conversations at record {index}")
        roles = [message.get("role") for message in conversations]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"invalid role sequence at SFT record {index}: {roles}")
        if any(message.get("tools") or message.get("tool_calls") for message in conversations):
            raise ValueError(f"unexpected tool fields at SFT record {index}")
        if conversations[-1].get("content", "").strip() != canonical["sql"].strip():
            raise ValueError(f"SFT target mismatch for {canonical['id']}")
        if "TABLE users" not in conversations[1].get("content", "") or "TABLE refunds" not in conversations[1].get("content", ""):
            raise ValueError(f"incomplete schema context for {canonical['id']}")
        role_patterns["->".join(roles)] += 1

    rl_records = _read_jsonl(directory / "sql_rl_train.jsonl")
    if len(rl_records) != len(splits["train"]):
        raise ValueError("SQL-RL and canonical train sizes do not match")
    for index, (record, canonical) in enumerate(zip(rl_records, splits["train"])):
        if not record.get("expected_result_hash") or not record.get("reference_sql"):
            raise ValueError(f"invalid SQL-RL record {index}")
        if record.get("sample_id") != canonical["id"]:
            raise ValueError(f"SQL-RL sample mismatch at record {index}")
        if [message.get("role") for message in record.get("messages", [])] != ["system", "user"]:
            raise ValueError(f"invalid SQL-RL messages at record {index}")

    return {
        "valid": True,
        "canonical_records": sum(len(records) for records in splits.values()),
        "split_counts": {split: len(records) for split, records in splits.items()},
        "family_counts": {split: len(families) for split, families in family_sets.items()},
        "family_overlaps": family_overlaps,
        "sql_primitive_coverage": {
            "counts": {split: len(values) for split, values in primitive_sets.items()},
            "missing_from_train": missing_primitives,
        },
        "sample_overlap_counts": {name: len(values) for name, values in sample_overlaps.items()},
        "sft_role_patterns": dict(sorted(role_patterns.items())),
        "sql_sft_records": len(sft_train) + len(sft_dev),
        "sql_rl_records": len(rl_records),
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
