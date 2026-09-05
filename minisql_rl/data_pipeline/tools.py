"""Tool schemas shared by SFT examples and the future SQL Agent runtime."""

from __future__ import annotations

import json
from typing import Any


QUERY_DATABASE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "在只读 SQLite 电商数据库中执行一条查询语句。只允许 SELECT 或 WITH，不允许修改数据。",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "要执行的单条 SQLite SELECT 或 WITH 查询",
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [QUERY_DATABASE_TOOL]

SYSTEM_PROMPT = (
    "你是电商数据分析助手。必须根据给出的数据库结构生成准确的 SQLite 查询，"
    "通过 query_database 工具获取真实数据，再用中文简洁回答。"
    "不得编造查询结果，不得执行任何写操作。"
)


def serialized_tools() -> str:
    """Return the MiniMind dataset representation of available tools."""

    return json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
