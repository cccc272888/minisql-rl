"""Hand-authored seed benchmark with execution-derived reference results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .sandbox import SQLSandbox


BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": "basic_001",
        "difficulty": "easy",
        "question": "当前有多少个在售商品？",
        "tables": ["products"],
        "sql": "SELECT COUNT(*) AS active_product_count FROM products WHERE status = 'active'",
    },
    {
        "id": "basic_002",
        "difficulty": "easy",
        "question": "各会员等级分别有多少名用户？按人数从高到低排列。",
        "tables": ["users"],
        "sql": "SELECT member_level, COUNT(*) AS user_count FROM users GROUP BY member_level ORDER BY user_count DESC, member_level",
    },
    {
        "id": "basic_003",
        "difficulty": "easy",
        "question": "库存低于安全库存的商品仓库组合一共有多少个？",
        "tables": ["inventory"],
        "sql": "SELECT COUNT(*) AS low_stock_count FROM inventory WHERE stock_quantity < safety_stock",
    },
    {
        "id": "basic_004",
        "difficulty": "easy",
        "question": "2026年8月完成了多少笔订单？",
        "tables": ["orders"],
        "sql": "SELECT COUNT(*) AS completed_orders FROM orders WHERE status = 'completed' AND created_at >= '2026-08-01' AND created_at < '2026-09-01'",
    },
    {
        "id": "join_001",
        "difficulty": "medium",
        "question": "累计销售额最高的5个商品是什么？",
        "tables": ["products", "orders", "order_items"],
        "sql": """SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM products p
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE o.status != 'cancelled'
GROUP BY p.id, p.name
ORDER BY revenue DESC, p.id
LIMIT 5""",
    },
    {
        "id": "join_002",
        "difficulty": "medium",
        "question": "2026年各城市的有效订单销售额是多少？列出最高的5个城市。",
        "tables": ["orders", "order_items"],
        "sql": """SELECT o.shipping_city, ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= '2026-01-01' AND o.created_at < '2027-01-01'
GROUP BY o.shipping_city ORDER BY revenue DESC, o.shipping_city LIMIT 5""",
    },
    {
        "id": "join_003",
        "difficulty": "medium",
        "question": "每个一级商品分类的销售额是多少？按销售额降序排列。",
        "tables": ["categories", "products", "orders", "order_items"],
        "sql": """SELECT parent.name AS top_category,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM categories child
JOIN categories parent ON child.parent_id = parent.id
JOIN products p ON p.category_id = child.id
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE o.status != 'cancelled'
GROUP BY parent.id, parent.name ORDER BY revenue DESC, parent.id""",
    },
    {
        "id": "join_004",
        "difficulty": "medium",
        "question": "不同支付方式的成功支付金额和订单数分别是多少？",
        "tables": ["payments"],
        "sql": """SELECT payment_method, COUNT(*) AS payment_count, ROUND(SUM(amount), 2) AS total_amount
FROM payments WHERE status = 'success'
GROUP BY payment_method ORDER BY total_amount DESC, payment_method""",
    },
    {
        "id": "join_005",
        "difficulty": "medium",
        "question": "平均评分最低的5个至少有3条评价的商品是什么？",
        "tables": ["products", "reviews"],
        "sql": """SELECT p.name, ROUND(AVG(r.rating), 2) AS avg_rating, COUNT(*) AS review_count
FROM products p JOIN reviews r ON r.product_id = p.id
GROUP BY p.id, p.name HAVING COUNT(*) >= 3
ORDER BY avg_rating ASC, review_count DESC, p.id LIMIT 5""",
    },
    {
        "id": "advanced_001",
        "difficulty": "hard",
        "question": "退款金额最高的5个商品是什么？只统计已批准退款。",
        "tables": ["products", "order_items", "refunds"],
        "sql": """SELECT p.name, ROUND(SUM(r.amount), 2) AS approved_refund_amount
FROM refunds r
JOIN order_items oi ON oi.id = r.order_item_id
JOIN products p ON p.id = oi.product_id
WHERE r.status = 'approved'
GROUP BY p.id, p.name ORDER BY approved_refund_amount DESC, p.id LIMIT 5""",
    },
    {
        "id": "advanced_002",
        "difficulty": "hard",
        "question": "下过至少3笔有效订单的复购用户有多少人？",
        "tables": ["orders"],
        "sql": """SELECT COUNT(*) AS repeat_user_count FROM (
SELECT user_id FROM orders WHERE status != 'cancelled'
GROUP BY user_id HAVING COUNT(*) >= 3
) repeat_users""",
    },
    {
        "id": "advanced_003",
        "difficulty": "hard",
        "question": "各会员等级的平均实付订单金额是多少？",
        "tables": ["users", "orders", "order_amounts"],
        "sql": """SELECT u.member_level, ROUND(AVG(oa.payable_amount), 2) AS avg_payable_amount
FROM users u
JOIN orders o ON o.user_id = u.id
JOIN order_amounts oa ON oa.order_id = o.id
WHERE o.status != 'cancelled'
GROUP BY u.member_level ORDER BY avg_payable_amount DESC, u.member_level""",
    },
    {
        "id": "advanced_004",
        "difficulty": "hard",
        "question": "找出已批准退款件数占已售件数比例最高的5个商品，要求至少售出20件。",
        "tables": ["products", "orders", "order_items", "refunds"],
        "sql": """SELECT p.name,
SUM(CASE WHEN o.status != 'cancelled' THEN oi.quantity ELSE 0 END) AS units_sold,
COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.id END) AS approved_refund_items,
ROUND(COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.id END) * 1.0 /
      SUM(CASE WHEN o.status != 'cancelled' THEN oi.quantity ELSE 0 END), 4) AS refund_rate
FROM products p
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
LEFT JOIN refunds r ON r.order_item_id = oi.id
GROUP BY p.id, p.name
HAVING units_sold >= 20
ORDER BY refund_rate DESC, units_sold DESC, p.id LIMIT 5""",
    },
    {
        "id": "advanced_005",
        "difficulty": "hard",
        "question": "2026年每个月的有效订单销售额是多少？",
        "tables": ["orders", "order_items"],
        "sql": """SELECT strftime('%Y-%m', o.created_at) AS month,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= '2026-01-01' AND o.created_at < '2027-01-01'
GROUP BY month ORDER BY month""",
    },
    {
        "id": "advanced_006",
        "difficulty": "hard",
        "question": "哪些仓库的缺货风险商品最多？缺货风险指库存低于安全库存。",
        "tables": ["warehouses", "inventory"],
        "sql": """SELECT w.name, COUNT(*) AS risky_product_count
FROM warehouses w JOIN inventory i ON i.warehouse_id = w.id
WHERE i.stock_quantity < i.safety_stock
GROUP BY w.id, w.name ORDER BY risky_product_count DESC, w.id""",
    },
]


def _result_hash(columns: list[str], rows: list[list[Any]]) -> str:
    canonical = json.dumps(
        {"columns": columns, "rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_benchmark(database_path: str | Path, output_path: str | Path) -> int:
    """Execute seed cases and write a JSONL benchmark with reference results."""

    sandbox = SQLSandbox(database_path, row_limit=500)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for case in BENCHMARK_CASES:
        result = sandbox.execute(case["sql"])
        record = dict(case)
        record["expected"] = {"columns": result.columns, "rows": result.rows}
        record["result_hash"] = _result_hash(result.columns, result.rows)
        records.append(record)

    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
