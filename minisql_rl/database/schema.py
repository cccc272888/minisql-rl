"""Utilities for exposing a compact database schema to a language model."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import quote


TABLE_DESCRIPTIONS = {
    "users": "用户基本信息与会员等级",
    "categories": "商品分类，parent_id 指向上级分类",
    "products": "商品、品牌、标价和上下架状态",
    "warehouses": "仓库及所在城市",
    "inventory": "商品在各仓库的库存与安全库存",
    "orders": "订单状态、时间、收货地及订单级优惠",
    "order_items": "订单商品明细；销售额按 quantity * unit_price - discount_amount 计算",
    "payments": "订单支付金额和支付渠道",
    "refunds": "按订单明细记录的退款申请",
    "reviews": "商品评分与评价内容",
    "order_amounts": "订单金额汇总视图",
    "product_sales": "商品累计销量与销售额视图",
}


def _connect_readonly(database_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{quote(str(database_path))}?mode=ro", uri=True)


def get_schema_context(
    database_path: str | Path,
    *,
    include_views: bool = True,
    selected_objects: list[str] | None = None,
    compact: bool = False,
) -> str:
    """Return concise, deterministic schema text suitable for a prompt."""

    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database does not exist: {path}")

    allowed_types = ("table", "view") if include_views else ("table",)
    placeholders = ",".join("?" for _ in allowed_types)
    connection = _connect_readonly(path)
    try:
        objects = connection.execute(
            f"""
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ({placeholders})
              AND name NOT LIKE 'sqlite_%'
            ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
            """,
            allowed_types,
        ).fetchall()
        if selected_objects is not None:
            selected = set(selected_objects)
            unknown = selected - {name for name, _ in objects}
            if unknown:
                raise ValueError(f"unknown schema objects: {sorted(unknown)}")
            objects = [item for item in objects if item[0] in selected]

        lines = ["SQLite 电商数据库 Schema："]
        for name, object_type in objects:
            escaped_name = name.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{escaped_name}")').fetchall()
            parts = []
            for _, column_name, data_type, not_null, default_value, primary_key in columns:
                if compact:
                    parts.append(f"{column_name}{' PK' if primary_key else ''}")
                    continue
                attributes = []
                if primary_key:
                    attributes.append("PK")
                if not_null:
                    attributes.append("NOT NULL")
                if default_value is not None:
                    attributes.append(f"DEFAULT {default_value}")
                suffix = f" {' '.join(attributes)}" if attributes else ""
                parts.append(f"{column_name} {data_type}{suffix}".strip())
            label = "VIEW" if object_type == "view" else "TABLE"
            description = TABLE_DESCRIPTIONS.get(name, "")
            separator = " # " if compact else " -- "
            lines.append(f"{label} {name} ({', '.join(parts)}){separator}{description}")

        foreign_keys = []
        for name, object_type in objects:
            if object_type != "table":
                continue
            escaped_name = name.replace('"', '""')
            for row in connection.execute(f'PRAGMA foreign_key_list("{escaped_name}")'):
                foreign_keys.append(f"{name}.{row[3]} -> {row[2]}.{row[4]}")
        if foreign_keys:
            if compact:
                lines.append("FK: " + "; ".join(sorted(foreign_keys)))
            else:
                lines.append("外键关系：")
                lines.extend(f"- {relationship}" for relationship in sorted(foreign_keys))
        return "\n".join(lines)
    finally:
        connection.close()
