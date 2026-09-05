"""Deterministic synthetic data generation for the MiniSQL-RL database."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


SCHEMA_PATH = Path(__file__).with_name("schema.sql")

LOCATIONS = [
    ("北京市", "北京"),
    ("上海市", "上海"),
    ("广东省", "广州"),
    ("广东省", "深圳"),
    ("浙江省", "杭州"),
    ("江苏省", "南京"),
    ("四川省", "成都"),
    ("湖北省", "武汉"),
    ("陕西省", "西安"),
    ("重庆市", "重庆"),
]

# parent, child, product words, brands, minimum and maximum list price
CATEGORY_SPECS = [
    ("数码家电", "手机通讯", ["智能手机", "快充手机", "拍照手机", "折叠屏手机"], ["星云", "远峰", "极光"], 1299, 7999),
    ("数码家电", "电脑办公", ["轻薄本", "游戏本", "机械键盘", "显示器"], ["锐界", "像素", "云帆"], 199, 8999),
    ("数码家电", "影音娱乐", ["无线耳机", "蓝牙音箱", "运动耳机", "降噪耳机"], ["声浪", "回声", "极光"], 99, 1999),
    ("家居生活", "厨房电器", ["空气炸锅", "电饭煲", "咖啡机", "破壁机"], ["暖屋", "轻食", "白鲸"], 159, 2999),
    ("家居生活", "清洁用品", ["洗衣液", "抽纸套装", "垃圾袋", "清洁喷雾"], ["净界", "青柠", "简家"], 19, 199),
    ("家居生活", "家具收纳", ["人体工学椅", "置物架", "收纳箱", "折叠桌"], ["木语", "简家", "筑梦"], 59, 2999),
    ("服饰运动", "运动户外", ["跑步鞋", "瑜伽垫", "冲锋衣", "运动背包"], ["疾风", "山野", "跃动"], 79, 1599),
    ("服饰运动", "男女服饰", ["纯棉T恤", "休闲外套", "直筒长裤", "保暖卫衣"], ["原色", "织光", "北岸"], 59, 999),
    ("美妆个护", "护肤彩妆", ["保湿面霜", "防晒乳", "洁面乳", "口红"], ["花漾", "清露", "初颜"], 39, 699),
    ("美妆个护", "个人护理", ["电动牙刷", "吹风机", "剃须刀", "按摩仪"], ["轻柔", "净界", "元气"], 69, 1599),
    ("食品健康", "休闲食品", ["坚果礼盒", "黑巧克力", "水果麦片", "牛肉干"], ["谷物记", "山味", "好食光"], 29, 399),
    ("食品健康", "健康保健", ["蛋白粉", "复合维生素", "鱼油胶囊", "益生菌"], ["元气", "每日健", "青禾"], 59, 699),
]

WAREHOUSES = [
    (1, "华北一号仓", "北京"),
    (2, "华东一号仓", "上海"),
    (3, "华南一号仓", "广州"),
    (4, "西南一号仓", "成都"),
]

REVIEW_TEXT = {
    5: ["质量很好，符合预期", "发货快，使用体验不错", "性价比很高，会回购"],
    4: ["整体不错，细节可以改进", "使用起来比较满意", "包装完好，功能正常"],
    3: ["表现一般，基本能用", "和预期差不多", "中规中矩"],
    2: ["体验不太好，希望改进", "做工一般", "配送和商品都有待提升"],
    1: ["商品存在明显问题", "与描述不符", "体验很差，已经申请退款"],
}

REFUND_REASONS = ["质量问题", "尺寸不合适", "与描述不符", "不想要了", "物流损坏", "发错商品"]


@dataclass(frozen=True)
class DatabaseBuildConfig:
    """Configuration for a reproducible synthetic database build."""

    seed: int = 20260905
    user_count: int = 500
    product_count: int = 60
    order_count: int = 3000
    start_date: str = "2025-01-01"
    end_date: str = "2026-08-31"

    def validate(self) -> None:
        if self.user_count < 10:
            raise ValueError("user_count must be at least 10")
        if self.product_count < len(CATEGORY_SPECS):
            raise ValueError(f"product_count must be at least {len(CATEGORY_SPECS)}")
        if self.order_count < 1:
            raise ValueError("order_count must be positive")
        if _parse_date(self.start_date) >= _parse_date(self.end_date):
            raise ValueError("start_date must be earlier than end_date")


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _random_datetime(rng: random.Random, start: datetime, end: datetime) -> datetime:
    seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, seconds))


def _weighted_choice(rng: random.Random, values: Iterable[tuple[str, float]]) -> str:
    names, weights = zip(*values)
    return rng.choices(names, weights=weights, k=1)[0]


def _create_categories(connection: sqlite3.Connection) -> dict[str, int]:
    parent_names = list(dict.fromkeys(spec[0] for spec in CATEGORY_SPECS))
    next_id = 1
    category_ids: dict[str, int] = {}
    for name in parent_names:
        connection.execute(
            "INSERT INTO categories(id, parent_id, name) VALUES (?, NULL, ?)",
            (next_id, name),
        )
        category_ids[name] = next_id
        next_id += 1
    for parent, child, *_ in CATEGORY_SPECS:
        connection.execute(
            "INSERT INTO categories(id, parent_id, name) VALUES (?, ?, ?)",
            (next_id, category_ids[parent], child),
        )
        category_ids[child] = next_id
        next_id += 1
    return category_ids


def _create_users(
    connection: sqlite3.Connection,
    rng: random.Random,
    count: int,
    start: datetime,
) -> list[tuple[int, str, str]]:
    users = []
    for user_id in range(1, count + 1):
        province, city = rng.choice(LOCATIONS)
        registered_at = _random_datetime(
            rng,
            datetime(2022, 1, 1),
            start - timedelta(days=1),
        )
        birth_date = datetime(rng.randint(1970, 2005), rng.randint(1, 12), rng.randint(1, 28))
        member_level = _weighted_choice(
            rng,
            [("normal", 0.55), ("silver", 0.25), ("gold", 0.15), ("platinum", 0.05)],
        )
        connection.execute(
            """
            INSERT INTO users(
                id, username, gender, birth_date, province, city,
                member_level, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                f"用户{user_id:05d}",
                rng.choice(["male", "female", "unknown"]),
                birth_date.strftime("%Y-%m-%d"),
                province,
                city,
                member_level,
                _iso(registered_at),
            ),
        )
        users.append((user_id, province, city))
    return users


def _create_products(
    connection: sqlite3.Connection,
    rng: random.Random,
    count: int,
    category_ids: dict[str, int],
    start: datetime,
) -> list[tuple[int, float]]:
    products = []
    for product_id in range(1, count + 1):
        _, child, words, brands, min_price, max_price = CATEGORY_SPECS[(product_id - 1) % len(CATEGORY_SPECS)]
        variant = (product_id - 1) // len(CATEGORY_SPECS) + 1
        word = words[(variant - 1) % len(words)]
        brand = brands[(product_id + variant) % len(brands)]
        raw_price = rng.uniform(min_price, max_price)
        list_price = round(raw_price / 10) * 10 - 1
        cost_price = round(list_price * rng.uniform(0.48, 0.72), 2)
        created_at = _random_datetime(rng, datetime(2023, 1, 1), start)
        connection.execute(
            """
            INSERT INTO products(
                id, sku, name, category_id, brand, cost_price,
                list_price, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                product_id,
                f"SKU{product_id:06d}",
                f"{brand}{word} {variant}代",
                category_ids[child],
                brand,
                cost_price,
                list_price,
                "inactive" if product_id % 23 == 0 else "active",
                _iso(created_at),
            ),
        )
        products.append((product_id, list_price))
    return products


def _create_inventory(
    connection: sqlite3.Connection,
    rng: random.Random,
    products: list[tuple[int, float]],
    end: datetime,
) -> None:
    connection.executemany(
        "INSERT INTO warehouses(id, name, city) VALUES (?, ?, ?)",
        WAREHOUSES,
    )
    for product_id, _ in products:
        for warehouse_id, _, _ in WAREHOUSES:
            safety_stock = rng.randint(8, 30)
            # Roughly 12% of product/warehouse pairs require replenishment.
            stock = rng.randint(0, safety_stock - 1) if rng.random() < 0.12 else rng.randint(safety_stock, 220)
            connection.execute(
                """
                INSERT INTO inventory(
                    product_id, warehouse_id, stock_quantity, safety_stock, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (product_id, warehouse_id, stock, safety_stock, _iso(end)),
            )


def _create_orders(
    connection: sqlite3.Connection,
    rng: random.Random,
    count: int,
    users: list[tuple[int, str, str]],
    products: list[tuple[int, float]],
    start: datetime,
    end: datetime,
) -> None:
    user_weights = [1.0 + 5.0 / (index + 5) for index in range(len(users))]
    payment_id = refund_id = review_id = item_id = 1

    for order_id in range(1, count + 1):
        user_id, province, city = rng.choices(users, weights=user_weights, k=1)[0]
        created_at = _random_datetime(rng, start, end + timedelta(hours=23, minutes=59))
        status = _weighted_choice(
            rng,
            [("completed", 0.72), ("shipped", 0.09), ("paid", 0.07), ("pending", 0.04), ("cancelled", 0.08)],
        )
        paid_at = created_at + timedelta(minutes=rng.randint(1, 180)) if status not in {"pending", "cancelled"} else None
        shipped_at = paid_at + timedelta(hours=rng.randint(4, 48)) if status in {"shipped", "completed"} else None
        completed_at = shipped_at + timedelta(days=rng.randint(1, 7)) if status == "completed" else None
        coupon_amount = rng.choice([0, 0, 0, 5, 10, 20, 30])
        shipping_fee = rng.choice([0, 0, 0, 6, 8, 10])

        connection.execute(
            """
            INSERT INTO orders(
                id, order_no, user_id, status, created_at, paid_at,
                shipped_at, completed_at, shipping_province, shipping_city,
                coupon_amount, shipping_fee
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                f"ORD{created_at:%Y%m%d}{order_id:08d}",
                user_id,
                status,
                _iso(created_at),
                _iso(paid_at) if paid_at else None,
                _iso(shipped_at) if shipped_at else None,
                _iso(completed_at) if completed_at else None,
                province,
                city,
                coupon_amount,
                shipping_fee,
            ),
        )

        chosen_products = rng.sample(products, k=min(rng.choices([1, 2, 3, 4], [0.48, 0.31, 0.15, 0.06])[0], len(products)))
        order_total = 0.0
        order_items = []
        for product_id, list_price in chosen_products:
            quantity = rng.choices([1, 2, 3], [0.78, 0.18, 0.04])[0]
            unit_price = round(list_price * rng.uniform(0.82, 1.0), 2)
            discount_amount = round(quantity * unit_price * rng.choice([0, 0, 0, 0.03, 0.05, 0.1]), 2)
            line_total = round(quantity * unit_price - discount_amount, 2)
            connection.execute(
                """
                INSERT INTO order_items(
                    id, order_id, product_id, quantity, unit_price, discount_amount
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, order_id, product_id, quantity, unit_price, discount_amount),
            )
            order_items.append((item_id, product_id, line_total))
            order_total += line_total
            item_id += 1

        if paid_at:
            payable_amount = max(0.0, round(order_total - coupon_amount + shipping_fee, 2))
            connection.execute(
                """
                INSERT INTO payments(
                    id, order_id, payment_method, amount, status, paid_at
                ) VALUES (?, ?, ?, ?, 'success', ?)
                """,
                (payment_id, order_id, rng.choice(["alipay", "wechat", "bank_card"]), payable_amount, _iso(paid_at)),
            )
            payment_id += 1

        if status in {"completed", "shipped"}:
            for current_item_id, product_id, line_total in order_items:
                if rng.random() < 0.075:
                    refund_created = (shipped_at or paid_at or created_at) + timedelta(days=rng.randint(1, 12))
                    refund_status = _weighted_choice(rng, [("approved", 0.78), ("rejected", 0.12), ("pending", 0.10)])
                    processed_at = refund_created + timedelta(hours=rng.randint(1, 72)) if refund_status != "pending" else None
                    rating_hint = rng.choice([1, 1, 2, 2, 3])
                    connection.execute(
                        """
                        INSERT INTO refunds(
                            id, refund_no, order_id, order_item_id, user_id,
                            amount, reason, status, created_at, processed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            refund_id,
                            f"REF{refund_created:%Y%m%d}{refund_id:08d}",
                            order_id,
                            current_item_id,
                            user_id,
                            line_total,
                            rng.choice(REFUND_REASONS),
                            refund_status,
                            _iso(refund_created),
                            _iso(processed_at) if processed_at else None,
                        ),
                    )
                    refund_id += 1
                else:
                    rating_hint = rng.choices([1, 2, 3, 4, 5], [0.02, 0.05, 0.13, 0.35, 0.45])[0]

                if status == "completed" and rng.random() < 0.30:
                    review_created = (completed_at or created_at) + timedelta(days=rng.randint(0, 20))
                    connection.execute(
                        """
                        INSERT INTO reviews(
                            id, order_item_id, user_id, product_id, rating,
                            content, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            review_id,
                            current_item_id,
                            user_id,
                            product_id,
                            rating_hint,
                            rng.choice(REVIEW_TEXT[rating_hint]),
                            _iso(review_created),
                        ),
                    )
                    review_id += 1


def build_database(
    database_path: str | Path,
    config: DatabaseBuildConfig | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Create and seed an e-commerce SQLite database.

    The same configuration always produces the same logical data. The output
    database is replaced only when ``overwrite`` is explicitly enabled.
    """

    config = config or DatabaseBuildConfig()
    config.validate()
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"database already exists: {path}")
        path.unlink()

    rng = random.Random(config.seed)
    start = _parse_date(config.start_date)
    end = _parse_date(config.end_date)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        with connection:
            category_ids = _create_categories(connection)
            users = _create_users(connection, rng, config.user_count, start)
            products = _create_products(connection, rng, config.product_count, category_ids, start)
            _create_inventory(connection, rng, products, end)
            _create_orders(connection, rng, config.order_count, users, products, start, end)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"foreign key violations detected: {violations[:3]}")
        connection.execute("ANALYZE")
        connection.commit()
        return database_stats(connection)
    except Exception:
        connection.close()
        if path.exists():
            path.unlink()
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass


def database_stats(connection: sqlite3.Connection) -> dict[str, int]:
    """Return row counts for all domain tables."""

    tables = [
        "users",
        "categories",
        "products",
        "warehouses",
        "inventory",
        "orders",
        "order_items",
        "payments",
        "refunds",
        "reviews",
    ]
    return {
        table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in tables
    }
