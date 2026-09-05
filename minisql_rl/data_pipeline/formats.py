"""Conversion from canonical query records to MiniMind training formats."""

from __future__ import annotations

import json
import re
from typing import Any

from minisql_rl.database import QueryResult, SQLSandbox, SQLSandboxError, get_schema_context

from .tools import SYSTEM_PROMPT, serialized_tools


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(database_path: str, record: dict[str, Any]) -> str:
    schema = get_schema_context(
        database_path,
        include_views=True,
        selected_objects=list(record["tables"]),
    )
    return (
        f"{schema}\n\n"
        f"业务问题：{record['question']}\n"
        "请调用 query_database 查询真实数据，然后根据查询结果用中文简洁回答。"
    )


def tool_call(sql: str) -> str:
    return compact_json([{"name": "query_database", "arguments": {"sql": sql}}])


def tool_result_payload(result: QueryResult) -> dict[str, Any]:
    return {
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
    }


def render_answer(result: QueryResult) -> str:
    if not result.rows:
        return "查询没有返回符合条件的数据。"
    if len(result.rows) == 1 and len(result.columns) == 1:
        return f"查询结果：{result.columns[0]}为{result.rows[0][0]}。"

    rendered_rows = []
    for row in result.rows:
        fields = [f"{column}={value}" for column, value in zip(result.columns, row)]
        rendered_rows.append("，".join(fields))
    return f"查询共返回{len(result.rows)}行：" + "；".join(rendered_rows) + "。"


def make_invalid_sql(sql: str, tables: list[str]) -> str:
    """Create a deterministic, executable-to-error table-name mutation."""

    for table in tables:
        pattern = re.compile(rf"\b{re.escape(table)}\b", flags=re.IGNORECASE)
        if pattern.search(sql):
            return pattern.sub(f"{table}_typo", sql, count=1)
    raise ValueError("could not create an invalid SQL variant")


def to_sft_record(
    database_path: str,
    sandbox: SQLSandbox,
    record: dict[str, Any],
    *,
    repair: bool,
) -> dict[str, Any]:
    prompt = build_user_prompt(database_path, record)
    expected = sandbox.execute(record["sql"])
    conversations: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT, "tools": serialized_tools()},
        {"role": "user", "content": prompt},
    ]

    if repair:
        invalid_sql = make_invalid_sql(record["sql"], record["tables"])
        try:
            sandbox.execute(invalid_sql)
        except SQLSandboxError as error:
            error_payload = compact_json({"error": str(error), "error_type": "sql_execution_error"})
        else:
            raise RuntimeError("invalid SQL unexpectedly succeeded")
        conversations.extend(
            [
                {"role": "assistant", "content": "", "tool_calls": tool_call(invalid_sql)},
                {"role": "tool", "content": error_payload},
            ]
        )

    conversations.extend(
        [
            {"role": "assistant", "content": "", "tool_calls": tool_call(record["sql"])},
            {"role": "tool", "content": compact_json(tool_result_payload(expected))},
            {"role": "assistant", "content": render_answer(expected)},
        ]
    )
    return {"conversations": conversations}


def _ground_truth_values(result: dict[str, Any], limit: int = 12) -> list[str]:
    values: list[str] = []
    for row in result["rows"]:
        for value in row:
            if value is None:
                continue
            text = str(value)
            if text not in values:
                values.append(text)
            if len(values) >= limit:
                return values
    return values


def to_agent_rl_record(database_path: str, record: dict[str, Any]) -> dict[str, Any]:
    """Create a SQL-RL prompt record; the SQL reward trainer is added next."""

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT, "tools": serialized_tools()},
            {"role": "user", "content": build_user_prompt(database_path, record)},
            {"role": "assistant", "content": ""},
        ],
        "gt": _ground_truth_values(record["expected"]),
        "expected_result_hash": record["result_hash"],
        "reference_sql": record["sql"],
        "sample_id": record["id"],
        "family_id": record["family_id"],
    }


def to_eval_prompt(database_path: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "family_id": record["family_id"],
        "difficulty": record["difficulty"],
        "question": record["question"],
        "prompt": build_user_prompt(database_path, record),
    }
