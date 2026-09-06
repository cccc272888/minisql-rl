"""Pure helpers for extracting and execution-checking generated SQLite SQL."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from minisql_rl.database import SQLSandbox, SQLSandboxError


_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_FENCE_PATTERN = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_SQL_START_PATTERN = re.compile(r"\b(?:SELECT|WITH)\b", flags=re.IGNORECASE)


def result_hash(_columns: list[str], rows: list[list[Any]]) -> str:
    """Hash result values while ignoring semantically irrelevant column aliases."""

    payload = json.dumps(
        {"rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _without_thinking(text: str) -> str:
    return _THINK_PATTERN.sub("", text).strip()


def is_strict_sql_output(text: str) -> bool:
    """Check whether the response obeys the requested SQL-only output format."""

    cleaned = _without_thinking(text)
    return "```" not in cleaned and bool(re.match(r"^(?:SELECT|WITH)\b", cleaned, re.IGNORECASE))


def extract_sql(text: str) -> str | None:
    """Extract a best-effort SELECT/WITH statement from a model response."""

    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = _without_thinking(text)
    fenced = _FENCE_PATTERN.findall(cleaned)
    candidates = fenced if fenced else [cleaned]
    for candidate in candidates:
        match = _SQL_START_PATTERN.search(candidate)
        if not match:
            continue
        sql = candidate[match.start():].strip()
        if ";" in sql:
            sql = sql.split(";", 1)[0].strip()
        return sql or None
    return None


def evaluate_sql_text(
    text: str,
    expected_result_hash: str,
    sandbox: SQLSandbox,
) -> dict[str, Any]:
    """Extract, execute and compare one generated response with its gold result."""

    strict_format = is_strict_sql_output(text)
    sql = extract_sql(text)
    if sql is None:
        return {
            "strict_format": strict_format,
            "extracted_sql": None,
            "executable": False,
            "execution_correct": False,
            "actual_result_hash": None,
            "error": "no_select_or_with_statement",
        }

    try:
        result = sandbox.execute(sql)
    except SQLSandboxError as error:
        return {
            "strict_format": strict_format,
            "extracted_sql": sql,
            "executable": False,
            "execution_correct": False,
            "actual_result_hash": None,
            "error": str(error),
        }

    actual_hash = result_hash(result.columns, result.rows)
    return {
        "strict_format": strict_format,
        "extracted_sql": sql,
        "executable": True,
        "execution_correct": actual_hash == expected_result_hash,
        "actual_result_hash": actual_hash,
        "error": None,
    }
