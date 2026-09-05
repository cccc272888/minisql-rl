"""Generate, validate and export reproducible MiniSQL-RL datasets."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from minisql_rl.database import SQLSandbox

from .formats import to_eval_prompt, to_sql_rl_record, to_sql_sft_record
from .templates import QueryFamily, families_for_split, load_template_context


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 20260905
    train_size: int = 1200
    dev_size: int = 150
    test_size: int = 150
    maximum_result_cells: int = 80
    maximum_attempt_multiplier: int = 200

    def validate(self) -> None:
        if min(self.train_size, self.dev_size, self.test_size) < 1:
            raise ValueError("all split sizes must be positive")
        if self.maximum_result_cells < 1:
            raise ValueError("maximum_result_cells must be positive")


def _canonical_hash(columns: list[str], rows: list[list[Any]]) -> str:
    text = json.dumps(
        {"columns": columns, "rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_record_id(split: str, question: str, sql: str) -> str:
    digest = hashlib.sha256(f"{split}\n{question}\n{sql}".encode("utf-8")).hexdigest()[:16]
    return f"{split}_{digest}"


def _normalized_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").lower()


def _is_informative(columns: list[str], rows: list[list[Any]], maximum_cells: int) -> tuple[bool, str]:
    if not columns or not rows:
        return False, "empty_result"
    if len(columns) * len(rows) > maximum_cells:
        return False, "result_too_large"
    if len(rows) == 1 and len(columns) == 1 and rows[0][0] in {None, 0, 0.0, ""}:
        return False, "empty_scalar"
    return True, "accepted"


def _generate_split(
    split: str,
    size: int,
    seed: int,
    sandbox: SQLSandbox,
    context: Any,
    config: PipelineConfig,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    families = families_for_split(split)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    rejection_reasons: Counter[str] = Counter()
    family_cursor = 0
    maximum_attempts = size * config.maximum_attempt_multiplier

    for _ in range(maximum_attempts):
        if len(records) >= size:
            break
        # Round-robin keeps every family represented; shuffling parameters still
        # yields a deterministic but balanced dataset.
        family: QueryFamily = families[family_cursor % len(families)]
        family_cursor += 1
        spec = family.build(rng, context)
        key = (spec.question.strip(), _normalized_sql(spec.sql))
        if key in seen:
            rejection_reasons["duplicate"] += 1
            continue
        try:
            result = sandbox.execute(spec.sql)
        except Exception:
            rejection_reasons["execution_error"] += 1
            continue
        informative, reason = _is_informative(result.columns, result.rows, config.maximum_result_cells)
        if not informative:
            rejection_reasons[reason] += 1
            continue

        seen.add(key)
        record = {
            "id": _stable_record_id(split, spec.question, spec.sql),
            "split": split,
            "family_id": spec.family_id,
            "difficulty": spec.difficulty,
            "question": spec.question,
            "sql": spec.sql,
            "tables": list(spec.tables),
            "parameters": spec.parameters,
            "expected": {"columns": result.columns, "rows": result.rows},
            "result_hash": _canonical_hash(result.columns, result.rows),
        }
        records.append(record)

    if len(records) != size:
        raise RuntimeError(
            f"could only generate {len(records)}/{size} records for {split}; "
            f"rejections={dict(rejection_reasons)}"
        )
    rng.shuffle(records)
    return records, rejection_reasons


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _family_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["family_id"] for record in records).items()))


def _difficulty_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["difficulty"] for record in records).items()))


def build_training_data(
    database_path: str | Path,
    output_directory: str | Path,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Build canonical, direct-SQL SFT, evaluation and SQL-RL artifacts."""

    config = config or PipelineConfig()
    config.validate()
    database = Path(database_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database does not exist: {database}")

    # Remove artifacts from the previous tool-call/Agent-oriented format so an
    # in-place rebuild cannot leave a misleading mixed dataset manifest.
    for legacy_name in ("sft_train.jsonl", "sft_dev.jsonl", "agent_rl_train.jsonl"):
        legacy_path = output / legacy_name
        if legacy_path.is_file():
            legacy_path.unlink()

    sandbox = SQLSandbox(database, row_limit=100, timeout_seconds=3.0)
    context = load_template_context(database)
    sizes = {"train": config.train_size, "dev": config.dev_size, "test": config.test_size}
    seed_offsets = {"train": 0, "dev": 10_000, "test": 20_000}
    splits: dict[str, list[dict[str, Any]]] = {}
    rejections: dict[str, dict[str, int]] = {}

    for split, size in sizes.items():
        records, rejected = _generate_split(
            split,
            size,
            config.seed + seed_offsets[split],
            sandbox,
            context,
            config,
        )
        splits[split] = records
        rejections[split] = dict(sorted(rejected.items()))
        _write_jsonl(output / f"canonical_{split}.jsonl", records)

    family_sets = {
        split: {record["family_id"] for record in records}
        for split, records in splits.items()
    }
    if family_sets["train"] & family_sets["dev"] or family_sets["train"] & family_sets["test"] or family_sets["dev"] & family_sets["test"]:
        raise RuntimeError("query-family leakage detected across splits")

    sft_train = [to_sql_sft_record(str(database), record) for record in splits["train"]]
    sft_dev = [to_sql_sft_record(str(database), record) for record in splits["dev"]]
    _write_jsonl(output / "sql_sft_train.jsonl", sft_train)
    _write_jsonl(output / "sql_sft_dev.jsonl", sft_dev)
    _write_jsonl(output / "sql_rl_train.jsonl", (to_sql_rl_record(str(database), record) for record in splits["train"]))
    _write_jsonl(output / "eval_dev_prompts.jsonl", (to_eval_prompt(str(database), record) for record in splits["dev"]))
    _write_jsonl(output / "eval_test_prompts.jsonl", (to_eval_prompt(str(database), record) for record in splits["test"]))

    manifest: dict[str, Any] = {
        "version": 1,
        "pipeline_config": asdict(config),
        "database": {
            "path": str(database),
            "sha256": _file_hash(database),
        },
        "splits": {
            split: {
                "count": len(records),
                "families": _family_counts(records),
                "difficulties": _difficulty_counts(records),
                "rejections": rejections[split],
            }
            for split, records in splits.items()
        },
        "sft": {
            "train_count": len(sft_train),
            "dev_count": len(sft_dev),
            "target": "direct_sql",
        },
        "leakage_checks": {
            "family_overlap_train_dev": [],
            "family_overlap_train_test": [],
            "family_overlap_dev_test": [],
        },
        "files": {},
    }
    for path in sorted(output.glob("*.jsonl")):
        manifest["files"][path.name] = {"sha256": _file_hash(path), "bytes": path.stat().st_size}
    _write_json(output / "manifest.json", manifest)
    return manifest
