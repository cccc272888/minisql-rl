"""Read-only, resource-bounded SQLite execution for model-generated SQL."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


class SQLSandboxError(RuntimeError):
    """A safe, user-facing SQL validation or execution error."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    execution_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sqlite_constants(names: list[str]) -> set[int]:
    return {getattr(sqlite3, name) for name in names if hasattr(sqlite3, name)}


DENIED_ACTIONS = _sqlite_constants(
    [
        "SQLITE_ALTER_TABLE",
        "SQLITE_ANALYZE",
        "SQLITE_ATTACH",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DELETE",
        "SQLITE_DETACH",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_INSERT",
        "SQLITE_PRAGMA",
        "SQLITE_REINDEX",
        "SQLITE_UPDATE",
    ]
)


def _strip_leading_comments(sql: str) -> str:
    remaining = sql.lstrip()
    while True:
        updated = re.sub(r"^--[^\n]*(?:\n|$)", "", remaining).lstrip()
        updated = re.sub(r"^/\*.*?\*/", "", updated, count=1, flags=re.DOTALL).lstrip()
        if updated == remaining:
            return remaining
        remaining = updated


class SQLSandbox:
    """Execute only SELECT/CTE statements against a SQLite database.

    Protection is applied at three layers: statement-type validation, a
    read-only SQLite connection, and SQLite's authorizer callback.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        row_limit: int = 200,
        timeout_seconds: float = 2.0,
        max_sql_length: int = 20_000,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.is_file():
            raise FileNotFoundError(f"database does not exist: {self.database_path}")
        if row_limit < 1:
            raise ValueError("row_limit must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.row_limit = row_limit
        self.timeout_seconds = timeout_seconds
        self.max_sql_length = max_sql_length

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{quote(str(self.database_path))}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(
            lambda action, _arg1, _arg2, _database, _trigger: (
                sqlite3.SQLITE_DENY if action in DENIED_ACTIONS else sqlite3.SQLITE_OK
            )
        )
        return connection

    def execute(self, sql: str) -> QueryResult:
        if not isinstance(sql, str) or not sql.strip():
            raise SQLSandboxError("SQL 不能为空")
        if len(sql) > self.max_sql_length:
            raise SQLSandboxError(f"SQL 长度超过限制：{self.max_sql_length}")

        normalized = _strip_leading_comments(sql)
        first_keyword = re.match(r"([A-Za-z]+)", normalized)
        if not first_keyword or first_keyword.group(1).upper() not in {"SELECT", "WITH"}:
            raise SQLSandboxError("只允许执行 SELECT 或 WITH 查询")

        started = time.perf_counter()
        deadline = started + self.timeout_seconds
        connection = self._connect()
        connection.set_progress_handler(
            lambda: 1 if time.perf_counter() > deadline else 0,
            2_000,
        )
        try:
            cursor = connection.execute(sql)
            if cursor.description is None:
                raise SQLSandboxError("查询没有返回结果集")
            columns = [description[0] for description in cursor.description]
            fetched = cursor.fetchmany(self.row_limit + 1)
            truncated = len(fetched) > self.row_limit
            rows = [list(row) for row in fetched[: self.row_limit]]
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                execution_ms=elapsed,
            )
        except SQLSandboxError:
            raise
        except sqlite3.DatabaseError as error:
            message = str(error)
            if "interrupted" in message.lower():
                message = "查询超时"
            raise SQLSandboxError(message) from error
        finally:
            connection.close()
