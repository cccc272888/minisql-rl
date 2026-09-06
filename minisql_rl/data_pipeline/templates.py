"""Parameterized Chinese e-commerce Text-to-SQL template families."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import quote


@dataclass(frozen=True)
class QuerySpec:
    family_id: str
    difficulty: str
    question: str
    sql: str
    tables: tuple[str, ...]
    parameters: dict[str, str | int | float]


@dataclass(frozen=True)
class TemplateContext:
    minimum_date: datetime
    maximum_date: datetime
    cities: tuple[str, ...]
    provinces: tuple[str, ...]
    warehouses: tuple[str, ...]
    parent_categories: tuple[str, ...]
    child_categories: tuple[str, ...]
    brands: tuple[str, ...]

    def period(self, rng: random.Random, *, minimum_days: int = 7) -> tuple[str, str, str]:
        total_days = max((self.maximum_date - self.minimum_date).days, minimum_days + 1)
        durations = [days for days in [7, 14, 30, 45, 60, 90, 120, 180] if days >= minimum_days]
        duration = rng.choice(durations or [minimum_days])
        duration = min(duration, total_days - 1)
        start_offset = rng.randint(0, max(0, total_days - duration - 1))
        start = self.minimum_date + timedelta(days=start_offset)
        end = min(start + timedelta(days=duration), self.maximum_date + timedelta(days=1))
        label = f"{start:%Y年%m月%d日}到{(end - timedelta(days=1)):%Y年%m月%d日}"
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), label


TemplateBuilder = Callable[[random.Random, TemplateContext], QuerySpec]


@dataclass(frozen=True)
class QueryFamily:
    family_id: str
    pool: str
    difficulty: str
    build: TemplateBuilder


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _count_orders(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    status, status_cn = rng.choice(
        [("completed", "已完成"), ("shipped", "已发货"), ("paid", "已支付"), ("cancelled", "已取消")]
    )
    wording = rng.choice(["有多少笔", "一共有多少个", "订单数量是多少"])
    question = f"{label}{status_cn}订单{wording}？"
    sql = (
        "SELECT COUNT(*) AS order_count FROM orders "
        f"WHERE status = {_quoted(status)} AND created_at >= {_quoted(start)} AND created_at < {_quoted(end)}"
    )
    return QuerySpec("order_count_period", "easy", question, sql, ("orders",), {"start": start, "end": end, "status": status})


def _registered_users(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    city = rng.choice(ctx.cities)
    member, member_cn = rng.choice(
        [("normal", "普通"), ("silver", "白银"), ("gold", "黄金"), ("platinum", "铂金")]
    )
    cutoff = datetime(2022, 1, 1) + timedelta(days=rng.randint(180, 1095))
    cutoff_text = cutoff.strftime("%Y-%m-%d")
    question = rng.choice(
        [
            f"{cutoff:%Y年%m月%d日}之前注册、来自{city}的{member_cn}会员有多少人？",
            f"统计{cutoff:%Y年%m月%d日}前注册且城市为{city}、等级为{member_cn}的用户数量。",
            f"{city}在{cutoff:%Y年%m月%d日}前注册的{member_cn}会员一共有多少名？",
        ]
    )
    sql = (
        "SELECT COUNT(*) AS user_count FROM users "
        f"WHERE city = {_quoted(city)} AND member_level = {_quoted(member)} "
        f"AND registered_at < {_quoted(cutoff_text)}"
    )
    return QuerySpec("user_segment_count", "easy", question, sql, ("users",), {"city": city, "member_level": member, "registered_before": cutoff_text})


def _products_in_price_range(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    lower = rng.randrange(0, 3001, 50)
    upper = lower + rng.choice([100, 200, 300, 500, 800, 1000, 2000, 3000])
    status, status_cn = rng.choice([("active", "在售"), ("inactive", "已下架")])
    question = rng.choice(
        [
            f"价格在{lower}元到{upper}元之间的{status_cn}商品有多少个？",
            f"统计标价不低于{lower}元且低于{upper}元的{status_cn}商品数量。",
            f"{status_cn}商品中，价格落在[{lower}, {upper})元区间的一共有多少种？",
        ]
    )
    sql = (
        "SELECT COUNT(*) AS product_count FROM products "
        f"WHERE status = {_quoted(status)} AND list_price >= {lower} AND list_price < {upper}"
    )
    return QuerySpec("products_price_range", "easy", question, sql, ("products",), {"lower": lower, "upper": upper, "status": status})


def _inventory_risk(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    warehouse = rng.choice(ctx.warehouses)
    category = rng.choice(ctx.child_categories)
    extra = rng.choice([0, 5, 10, 20])
    comparison = "低于安全库存" if extra == 0 else f"低于安全库存加{extra}件"
    question = rng.choice(
        [
            f"{warehouse}中{category}分类里，库存{comparison}的商品有多少种？",
            f"统计{warehouse}的{category}商品中库存{comparison}的商品数量。",
            f"{warehouse}里有多少种{category}商品的库存{comparison}？",
        ]
    )
    sql = f"""SELECT COUNT(*) AS risky_product_count
FROM inventory i JOIN warehouses w ON w.id = i.warehouse_id
JOIN products p ON p.id = i.product_id JOIN categories c ON c.id = p.category_id
WHERE w.name = {_quoted(warehouse)} AND c.name = {_quoted(category)}
  AND i.stock_quantity < i.safety_stock + {extra}"""
    return QuerySpec("inventory_risk", "easy", question, sql, ("inventory", "warehouses", "products", "categories"), {"warehouse": warehouse, "category": category, "extra": extra})


def _top_products_revenue(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    top_k = rng.randint(3, 10)
    question = rng.choice(
        [
            f"{label}销售额最高的{top_k}个商品是什么？",
            f"列出{label}按销售额排名前{top_k}的商品。",
            f"{label}哪些商品贡献的销售额最高？请给出前{top_k}名。",
        ]
    )
    sql = f"""SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM products p
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY p.id, p.name ORDER BY revenue DESC, p.id LIMIT {top_k}"""
    return QuerySpec("top_products_revenue", "medium", question, sql, ("products", "order_items", "orders"), {"start": start, "end": end, "top_k": top_k})


def _city_revenue(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    top_k = rng.randint(3, min(8, len(ctx.cities)))
    question = f"{label}有效订单销售额最高的{top_k}个收货城市是哪些？"
    sql = f"""SELECT o.shipping_city, ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY o.shipping_city ORDER BY revenue DESC, o.shipping_city LIMIT {top_k}"""
    return QuerySpec("city_revenue", "medium", question, sql, ("orders", "order_items"), {"start": start, "end": end, "top_k": top_k})


def _category_revenue(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    parent = rng.choice(ctx.parent_categories)
    top_k = rng.randint(2, 5)
    question = f"{label}{parent}分类中销售额最高的{top_k}个二级分类是什么？"
    sql = f"""SELECT child.name AS category_name,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM categories child
JOIN categories parent ON child.parent_id = parent.id
JOIN products p ON p.category_id = child.id
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE parent.name = {_quoted(parent)} AND o.status != 'cancelled'
  AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY child.id, child.name ORDER BY revenue DESC, child.id LIMIT {top_k}"""
    return QuerySpec("category_revenue", "medium", question, sql, ("categories", "products", "order_items", "orders"), {"start": start, "end": end, "parent_category": parent, "top_k": top_k})


def _payment_summary(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    minimum = rng.choice([0, 100, 300, 500, 1000])
    question = f"{label}各支付方式中，单笔金额不低于{minimum}元的成功支付有多少笔、总金额多少？"
    sql = f"""SELECT payment_method, COUNT(*) AS payment_count, ROUND(SUM(amount), 2) AS total_amount
FROM payments
WHERE status = 'success' AND amount >= {minimum}
  AND paid_at >= {_quoted(start)} AND paid_at < {_quoted(end)}
GROUP BY payment_method ORDER BY total_amount DESC, payment_method"""
    return QuerySpec("payment_summary", "medium", question, sql, ("payments",), {"start": start, "end": end, "minimum": minimum})


def _refund_reasons(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    status, status_cn = rng.choice([("approved", "已批准"), ("rejected", "已拒绝"), ("pending", "待处理")])
    top_k = rng.randint(2, 5)
    question = f"{label}{status_cn}退款申请中最常见的{top_k}个原因是什么？"
    sql = f"""SELECT reason, COUNT(*) AS refund_count, ROUND(SUM(amount), 2) AS refund_amount
FROM refunds
WHERE status = {_quoted(status)} AND created_at >= {_quoted(start)} AND created_at < {_quoted(end)}
GROUP BY reason ORDER BY refund_count DESC, refund_amount DESC, reason LIMIT {top_k}"""
    return QuerySpec("refund_reasons", "medium", question, sql, ("refunds",), {"start": start, "end": end, "status": status, "top_k": top_k})


def _brand_sales(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    brand = rng.choice(ctx.brands)
    top_k = rng.randint(2, 6)
    question = f"{label}{brand}品牌销量最高的{top_k}个商品是什么？"
    sql = f"""SELECT p.name, SUM(oi.quantity) AS units_sold,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM products p
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE p.brand = {_quoted(brand)} AND o.status != 'cancelled'
  AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY p.id, p.name ORDER BY units_sold DESC, revenue DESC, p.id LIMIT {top_k}"""
    return QuerySpec("brand_sales", "medium", question, sql, ("products", "order_items", "orders"), {"start": start, "end": end, "brand": brand, "top_k": top_k})


def _province_orders(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    province = rng.choice(ctx.provinces)
    minimum = rng.choice([100, 300, 500, 1000, 2000])
    question = f"{label}收货地为{province}且商品金额不低于{minimum}元的有效订单有多少笔？"
    sql = f"""SELECT COUNT(*) AS order_count FROM (
SELECT o.id
FROM orders o JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.shipping_province = {_quoted(province)}
  AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY o.id HAVING SUM(oi.quantity * oi.unit_price - oi.discount_amount) >= {minimum}
) matched_orders"""
    return QuerySpec("province_high_value_orders", "medium", question, sql, ("orders", "order_items"), {"start": start, "end": end, "province": province, "minimum": minimum})


def _repeat_customers(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=30)
    minimum = rng.randint(2, 6)
    question = f"{label}至少下过{minimum}笔有效订单的复购用户有多少人？"
    sql = f"""SELECT COUNT(*) AS repeat_user_count FROM (
SELECT user_id FROM orders
WHERE status != 'cancelled' AND created_at >= {_quoted(start)} AND created_at < {_quoted(end)}
GROUP BY user_id HAVING COUNT(*) >= {minimum}
) repeat_users"""
    return QuerySpec("repeat_customers", "hard", question, sql, ("orders",), {"start": start, "end": end, "minimum": minimum})


def _member_average_order(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng)
    member = rng.choice(["normal", "silver", "gold", "platinum"])
    question = f"{label}{member}会员的有效订单平均应付金额是多少？"
    sql = f"""SELECT ROUND(AVG(oa.payable_amount), 2) AS avg_payable_amount
FROM users u JOIN orders o ON o.user_id = u.id JOIN order_amounts oa ON oa.order_id = o.id
WHERE u.member_level = {_quoted(member)} AND o.status != 'cancelled'
  AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}"""
    return QuerySpec("member_average_order", "hard", question, sql, ("users", "orders", "order_amounts"), {"start": start, "end": end, "member_level": member})


def _category_inventory_value(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    child = rng.choice(ctx.child_categories)
    top_k = rng.randint(2, 4)
    price_column, price_label = rng.choice([("cost_price", "成本价"), ("list_price", "商品标价")])
    question = rng.choice(
        [
            f"{child}分类在各仓库按{price_label}计算的库存金额是多少？列出最高的{top_k}个仓库。",
            f"按{price_label}统计{child}商品的分仓库存价值，并返回金额最高的{top_k}个仓库。",
            f"{child}分类的库存按{price_label}折算后，价值排名前{top_k}的仓库有哪些？",
        ]
    )
    sql = f"""SELECT w.name, ROUND(SUM(i.stock_quantity * p.{price_column}), 2) AS inventory_value
FROM inventory i JOIN warehouses w ON w.id = i.warehouse_id
JOIN products p ON p.id = i.product_id JOIN categories c ON c.id = p.category_id
WHERE c.name = {_quoted(child)}
GROUP BY w.id, w.name ORDER BY inventory_value DESC, w.id LIMIT {top_k}"""
    return QuerySpec("category_inventory_value", "hard", question, sql, ("inventory", "warehouses", "products", "categories"), {"child_category": child, "price_column": price_column, "top_k": top_k})


def _user_order_ranking(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=30)
    top_k = rng.randint(3, 10)
    question = rng.choice(
        [
            f"{label}有效订单数量最多的{top_k}位用户是谁？",
            f"列出{label}按有效订单数排名前{top_k}的用户。",
            f"{label}哪些用户下的有效订单最多？返回前{top_k}位。",
        ]
    )
    sql = f"""SELECT u.username, COUNT(DISTINCT o.id) AS order_count
FROM users u JOIN orders o ON o.user_id = u.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY u.id, u.username ORDER BY order_count DESC, u.id LIMIT {top_k}"""
    return QuerySpec("user_order_ranking", "medium", question, sql, ("users", "orders"), {"start": start, "end": end, "top_k": top_k})


def _user_spending_ranking(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=30)
    top_k = rng.randint(3, 10)
    question = rng.choice(
        [
            f"{label}消费金额最高的{top_k}位用户是谁？",
            f"统计{label}用户消费额并列出前{top_k}名。",
            f"{label}按有效订单商品金额排名，返回消费最高的{top_k}位用户。",
        ]
    )
    sql = f"""SELECT u.username,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS spending
FROM users u JOIN orders o ON o.user_id = u.id JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY u.id, u.username ORDER BY spending DESC, u.id LIMIT {top_k}"""
    return QuerySpec("user_spending_ranking", "medium", question, sql, ("users", "orders", "order_items"), {"start": start, "end": end, "top_k": top_k})


def _product_units_threshold(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=30)
    minimum_units = rng.choice([1, 2, 3, 5, 10])
    top_k = rng.randint(3, 10)
    question = rng.choice(
        [
            f"{label}至少售出{minimum_units}件的商品中，销量最高的{top_k}个是什么？",
            f"列出{label}销量不低于{minimum_units}件的商品，按销量返回前{top_k}名。",
            f"{label}累计售出至少{minimum_units}件且销量最高的{top_k}个商品有哪些？",
        ]
    )
    sql = f"""SELECT p.name, SUM(oi.quantity) AS units_sold
FROM products p JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY p.id, p.name HAVING SUM(oi.quantity) >= {minimum_units}
ORDER BY units_sold DESC, p.id LIMIT {top_k}"""
    return QuerySpec("product_units_threshold", "medium", question, sql, ("products", "order_items", "orders"), {"start": start, "end": end, "minimum_units": minimum_units, "top_k": top_k})


def _product_approved_refunds(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=60)
    top_k = rng.randint(3, 8)
    question = rng.choice(
        [
            f"{label}已批准退款明细数量最多的{top_k}个商品是什么？",
            f"按已批准退款明细数统计{label}的商品，返回前{top_k}名。",
            f"{label}哪些商品对应的已批准退款明细最多？列出前{top_k}个。",
        ]
    )
    approved_count = "COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.id END)"
    sql = f"""SELECT p.name,
{approved_count} AS approved_refund_items
FROM products p JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id LEFT JOIN refunds r ON r.order_item_id = oi.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY p.id, p.name HAVING {approved_count} >= 1
ORDER BY approved_refund_items DESC, p.id LIMIT {top_k}"""
    return QuerySpec("product_approved_refunds", "hard", question, sql, ("products", "order_items", "orders", "refunds"), {"start": start, "end": end, "top_k": top_k})


def _monthly_order_count(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=90)
    question = rng.choice(
        [
            f"统计{label}每个月的有效订单数量，按月份排列。",
            f"{label}各月份分别有多少笔有效订单？",
            f"按月汇总{label}的有效订单数并按月份升序返回。",
        ]
    )
    sql = f"""SELECT strftime('%Y-%m', created_at) AS month,
COUNT(DISTINCT id) AS order_count
FROM orders
WHERE status != 'cancelled' AND created_at >= {_quoted(start)} AND created_at < {_quoted(end)}
GROUP BY month ORDER BY month"""
    return QuerySpec("monthly_order_count", "medium", question, sql, ("orders",), {"start": start, "end": end})


def _monthly_revenue(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=90)
    question = rng.choice(
        [
            f"统计{label}每个月的有效订单销售额，按月份排列。",
            f"{label}各月份的有效订单销售额分别是多少？",
            f"按月汇总{label}的有效订单商品金额并按月份升序返回。",
        ]
    )
    sql = f"""SELECT strftime('%Y-%m', o.created_at) AS month,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY month ORDER BY month"""
    return QuerySpec("monthly_revenue", "medium", question, sql, ("orders", "order_items"), {"start": start, "end": end})


def _payment_success_rate(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=30)
    question = rng.choice(
        [
            f"统计{label}各支付方式的支付总数、成功数和成功率。",
            f"{label}每种支付方式的成功支付占比是多少？同时返回总笔数和成功笔数。",
            f"按支付方式汇总{label}的支付成功率，并按成功率降序排列。",
        ]
    )
    success_count = "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)"
    sql = f"""SELECT payment_method, COUNT(*) AS payment_count,
{success_count} AS success_count,
ROUND({success_count} * 1.0 / COUNT(*), 4) AS success_rate
FROM payments
WHERE paid_at >= {_quoted(start)} AND paid_at < {_quoted(end)}
GROUP BY payment_method ORDER BY success_rate DESC, payment_method"""
    return QuerySpec("payment_success_rate", "hard", question, sql, ("payments",), {"start": start, "end": end})


def _product_refund_rate(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=30)
    top_k = rng.randint(3, 8)
    minimum_units = rng.choice([3, 5, 10, 15])
    question = f"{label}至少售出{minimum_units}件的商品中，已批准退款明细占比最高的{top_k}个是什么？"
    sql = f"""SELECT p.name, SUM(oi.quantity) AS units_sold,
COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.id END) AS approved_refund_items,
ROUND(COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.id END) * 1.0 / SUM(oi.quantity), 4) AS refund_rate
FROM products p JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id LEFT JOIN refunds r ON r.order_item_id = oi.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY p.id, p.name HAVING SUM(oi.quantity) >= {minimum_units}
ORDER BY refund_rate DESC, units_sold DESC, p.id LIMIT {top_k}"""
    return QuerySpec("product_refund_rate", "hard", question, sql, ("products", "order_items", "orders", "refunds"), {"start": start, "end": end, "minimum_units": minimum_units, "top_k": top_k})


def _high_value_users(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=30)
    top_k = rng.randint(3, 10)
    minimum_orders = rng.choice([1, 2, 3, 4])
    question = f"{label}至少有{minimum_orders}笔有效订单的用户中，消费金额最高的{top_k}位是谁？"
    sql = f"""SELECT u.username, COUNT(DISTINCT o.id) AS order_count,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS spending
FROM users u JOIN orders o ON o.user_id = u.id JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY u.id, u.username HAVING COUNT(DISTINCT o.id) >= {minimum_orders}
ORDER BY spending DESC, u.id LIMIT {top_k}"""
    return QuerySpec("high_value_users", "hard", question, sql, ("users", "orders", "order_items"), {"start": start, "end": end, "minimum_orders": minimum_orders, "top_k": top_k})


def _monthly_sales_trend(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=90)
    question = f"统计{label}每个月的有效订单销售额和订单数，按月份排列。"
    sql = f"""SELECT strftime('%Y-%m', o.created_at) AS month,
COUNT(DISTINCT o.id) AS order_count,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY month ORDER BY month"""
    return QuerySpec("monthly_sales_trend", "hard", question, sql, ("orders", "order_items"), {"start": start, "end": end})


def _brand_refund_rate(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=90)
    minimum_units = rng.choice([3, 5, 10])
    top_k = rng.randint(3, 6)
    approved_count = "COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.id END)"
    question = f"{label}至少售出{minimum_units}件的品牌中，已批准退款明细占比最高的{top_k}个是哪些？"
    sql = f"""SELECT p.brand, SUM(oi.quantity) AS units_sold,
{approved_count} AS approved_refund_items,
ROUND({approved_count} * 1.0 / SUM(oi.quantity), 4) AS refund_rate
FROM products p JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id LEFT JOIN refunds r ON r.order_item_id = oi.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY p.brand HAVING SUM(oi.quantity) >= {minimum_units}
ORDER BY refund_rate DESC, units_sold DESC, p.brand LIMIT {top_k}"""
    return QuerySpec("brand_refund_rate", "hard", question, sql, ("products", "order_items", "orders", "refunds"), {"start": start, "end": end, "minimum_units": minimum_units, "top_k": top_k})


def _monthly_refund_trend(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=90)
    approved_count = "SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END)"
    question = f"统计{label}每个月的退款申请数、已批准数量和已批准金额，按月份排列。"
    sql = f"""SELECT strftime('%Y-%m', created_at) AS month,
COUNT(*) AS refund_count, {approved_count} AS approved_count,
ROUND(SUM(CASE WHEN status = 'approved' THEN amount ELSE 0 END), 2) AS approved_amount
FROM refunds
WHERE created_at >= {_quoted(start)} AND created_at < {_quoted(end)}
GROUP BY month ORDER BY month"""
    return QuerySpec("monthly_refund_trend", "hard", question, sql, ("refunds",), {"start": start, "end": end})


def _city_customer_value(rng: random.Random, ctx: TemplateContext) -> QuerySpec:
    start, end, label = ctx.period(rng, minimum_days=60)
    minimum_orders = rng.choice([2, 3, 5, 8])
    top_k = rng.randint(3, min(8, len(ctx.cities)))
    question = f"{label}至少有{minimum_orders}笔有效订单的收货城市中，销售额最高的{top_k}个是哪些？同时返回用户数和订单数。"
    sql = f"""SELECT o.shipping_city, COUNT(DISTINCT o.user_id) AS user_count,
COUNT(DISTINCT o.id) AS order_count,
ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.id
WHERE o.status != 'cancelled' AND o.created_at >= {_quoted(start)} AND o.created_at < {_quoted(end)}
GROUP BY o.shipping_city HAVING COUNT(DISTINCT o.id) >= {minimum_orders}
ORDER BY revenue DESC, o.shipping_city LIMIT {top_k}"""
    return QuerySpec("city_customer_value", "hard", question, sql, ("orders", "order_items"), {"start": start, "end": end, "minimum_orders": minimum_orders, "top_k": top_k})


QUERY_FAMILIES = (
    QueryFamily("order_count_period", "trainable", "easy", _count_orders),
    QueryFamily("user_segment_count", "trainable", "easy", _registered_users),
    QueryFamily("products_price_range", "trainable", "easy", _products_in_price_range),
    QueryFamily("inventory_risk", "trainable", "easy", _inventory_risk),
    QueryFamily("top_products_revenue", "trainable", "medium", _top_products_revenue),
    QueryFamily("city_revenue", "trainable", "medium", _city_revenue),
    QueryFamily("category_revenue", "trainable", "medium", _category_revenue),
    QueryFamily("payment_summary", "trainable", "medium", _payment_summary),
    QueryFamily("refund_reasons", "trainable", "medium", _refund_reasons),
    QueryFamily("brand_sales", "trainable", "medium", _brand_sales),
    QueryFamily("province_high_value_orders", "trainable", "medium", _province_orders),
    QueryFamily("repeat_customers", "trainable", "hard", _repeat_customers),
    QueryFamily("member_average_order", "trainable", "hard", _member_average_order),
    QueryFamily("category_inventory_value", "trainable", "hard", _category_inventory_value),
    QueryFamily("user_order_ranking", "trainable", "medium", _user_order_ranking),
    QueryFamily("user_spending_ranking", "trainable", "medium", _user_spending_ranking),
    QueryFamily("product_units_threshold", "trainable", "medium", _product_units_threshold),
    QueryFamily("product_approved_refunds", "trainable", "hard", _product_approved_refunds),
    QueryFamily("monthly_order_count", "trainable", "medium", _monthly_order_count),
    QueryFamily("monthly_revenue", "trainable", "medium", _monthly_revenue),
    QueryFamily("payment_success_rate", "trainable", "hard", _payment_success_rate),
    QueryFamily("product_refund_rate", "challenge", "hard", _product_refund_rate),
    QueryFamily("high_value_users", "challenge", "hard", _high_value_users),
    QueryFamily("monthly_sales_trend", "challenge", "hard", _monthly_sales_trend),
    QueryFamily("brand_refund_rate", "heldout", "hard", _brand_refund_rate),
    QueryFamily("monthly_refund_trend", "heldout", "hard", _monthly_refund_trend),
    QueryFamily("city_customer_value", "heldout", "hard", _city_customer_value),
)


SQL_PRIMITIVES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "order_count_period": ("filter", "aggregate_count"),
    "user_segment_count": ("filter", "aggregate_count"),
    "products_price_range": ("filter", "aggregate_count"),
    "inventory_risk": ("inner_join", "filter", "aggregate_count", "arithmetic"),
    "top_products_revenue": ("inner_join", "filter", "aggregate_sum", "arithmetic", "group_by", "order_by", "limit"),
    "city_revenue": ("inner_join", "filter", "aggregate_sum", "arithmetic", "group_by", "order_by", "limit"),
    "category_revenue": ("inner_join", "filter", "aggregate_sum", "arithmetic", "group_by", "order_by", "limit"),
    "payment_summary": ("filter", "aggregate_count", "aggregate_sum", "group_by", "order_by"),
    "refund_reasons": ("filter", "aggregate_count", "aggregate_sum", "group_by", "order_by", "limit"),
    "brand_sales": ("inner_join", "filter", "aggregate_sum", "arithmetic", "group_by", "order_by", "limit"),
    "province_high_value_orders": ("inner_join", "filter", "aggregate_count", "aggregate_sum", "arithmetic", "group_by", "having", "subquery"),
    "repeat_customers": ("filter", "aggregate_count", "group_by", "having", "subquery"),
    "member_average_order": ("inner_join", "filter", "aggregate_avg"),
    "category_inventory_value": ("inner_join", "filter", "aggregate_sum", "arithmetic", "group_by", "order_by", "limit"),
    "user_order_ranking": ("inner_join", "filter", "count_distinct", "group_by", "order_by", "limit"),
    "user_spending_ranking": ("inner_join", "filter", "aggregate_sum", "arithmetic", "group_by", "order_by", "limit"),
    "product_units_threshold": ("inner_join", "filter", "aggregate_sum", "group_by", "having", "order_by", "limit"),
    "product_approved_refunds": ("inner_join", "left_join", "filter", "count_distinct", "conditional_aggregation", "group_by", "having", "order_by", "limit"),
    "monthly_order_count": ("filter", "date_bucket_month", "count_distinct", "group_by", "order_by"),
    "monthly_revenue": ("inner_join", "filter", "date_bucket_month", "aggregate_sum", "arithmetic", "group_by", "order_by"),
    "payment_success_rate": ("filter", "aggregate_count", "aggregate_sum", "conditional_aggregation", "ratio", "group_by", "order_by"),
    "product_refund_rate": ("inner_join", "left_join", "filter", "aggregate_sum", "count_distinct", "conditional_aggregation", "ratio", "group_by", "having", "order_by", "limit"),
    "high_value_users": ("inner_join", "filter", "aggregate_sum", "count_distinct", "arithmetic", "group_by", "having", "order_by", "limit"),
    "monthly_sales_trend": ("inner_join", "filter", "date_bucket_month", "aggregate_sum", "count_distinct", "arithmetic", "group_by", "order_by"),
    "brand_refund_rate": ("inner_join", "left_join", "filter", "aggregate_sum", "count_distinct", "conditional_aggregation", "ratio", "group_by", "having", "order_by", "limit"),
    "monthly_refund_trend": ("filter", "date_bucket_month", "aggregate_count", "aggregate_sum", "conditional_aggregation", "group_by", "order_by"),
    "city_customer_value": ("inner_join", "filter", "aggregate_sum", "count_distinct", "arithmetic", "group_by", "having", "order_by", "limit"),
}


def primitives_for_family(family_id: str) -> tuple[str, ...]:
    try:
        return SQL_PRIMITIVES_BY_FAMILY[family_id]
    except KeyError as error:
        raise ValueError(f"SQL primitives are not declared for family: {family_id}") from error


def load_template_context(database_path: str | Path) -> TemplateContext:
    path = Path(database_path).expanduser().resolve()
    connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    try:
        minimum_date, maximum_date = connection.execute(
            "SELECT MIN(date(created_at)), MAX(date(created_at)) FROM orders"
        ).fetchone()
        if not minimum_date or not maximum_date:
            raise ValueError("orders table is empty")

        def values(sql: str) -> tuple[str, ...]:
            return tuple(row[0] for row in connection.execute(sql).fetchall())

        return TemplateContext(
            minimum_date=datetime.strptime(minimum_date, "%Y-%m-%d"),
            maximum_date=datetime.strptime(maximum_date, "%Y-%m-%d"),
            cities=values("SELECT DISTINCT shipping_city FROM orders ORDER BY shipping_city"),
            provinces=values("SELECT DISTINCT shipping_province FROM orders ORDER BY shipping_province"),
            warehouses=values("SELECT name FROM warehouses ORDER BY id"),
            parent_categories=values("SELECT name FROM categories WHERE parent_id IS NULL ORDER BY id"),
            child_categories=values("SELECT name FROM categories WHERE parent_id IS NOT NULL ORDER BY id"),
            brands=values("SELECT DISTINCT brand FROM products ORDER BY brand"),
        )
    finally:
        connection.close()


def families_for_split(split: str) -> tuple[QueryFamily, ...]:
    if split in {"train", "dev"}:
        pool = "trainable"
    elif split in {"challenge", "composition_train"}:
        pool = "challenge"
    elif split == "test":
        pool = "heldout"
    else:
        raise ValueError(f"unknown split: {split}")
    families = tuple(family for family in QUERY_FAMILIES if family.pool == pool)
    if not families:
        raise ValueError(f"no query families configured for split: {split}")
    return families
