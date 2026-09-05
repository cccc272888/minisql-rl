"""Convert canonical examples into direct Text-to-SQL training formats."""

from __future__ import annotations

from typing import Any

from minisql_rl.database import get_schema_context


SYSTEM_PROMPT = (
    "你是一个电商 Text-to-SQL 模型。请根据给出的 SQLite 数据库结构和业务问题，"
    "生成一条可执行的只读查询。只输出 SQL，不要解释，不要使用 Markdown 代码块；"
    "只允许使用 SELECT 或 WITH。"
)


def build_user_prompt(database_path: str, record: dict[str, Any]) -> str:
    """Build a prompt with the complete schema so table selection is not leaked."""

    schema = get_schema_context(database_path, include_views=True, compact=True)
    return f"{schema}\n\n业务问题：{record['question']}"


def to_sql_sft_record(database_path: str, record: dict[str, Any]) -> dict[str, Any]:
    """Create one MiniMind SFT example whose sole target is the reference SQL."""

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(database_path, record)},
            {"role": "assistant", "content": record["sql"].strip()},
        ]
    }


def to_sql_rl_record(database_path: str, record: dict[str, Any]) -> dict[str, Any]:
    """Create one prompt plus the execution reference needed by SQL GRPO."""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(database_path, record)},
        ],
        "reference_sql": record["sql"],
        "expected_result_hash": record["result_hash"],
        "sample_id": record["id"],
        "family_id": record["family_id"],
        "difficulty": record["difficulty"],
    }


def to_eval_prompt(database_path: str, record: dict[str, Any]) -> dict[str, Any]:
    """Export a hidden-label inference input; gold data stays in canonical files."""

    return {
        "id": record["id"],
        "family_id": record["family_id"],
        "difficulty": record["difficulty"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(database_path, record)},
        ],
    }
