PRAGMA foreign_keys = ON;

CREATE TABLE users (
    id              INTEGER PRIMARY KEY,
    username        TEXT NOT NULL UNIQUE,
    gender          TEXT NOT NULL CHECK (gender IN ('male', 'female', 'unknown')),
    birth_date      TEXT,
    province        TEXT NOT NULL,
    city            TEXT NOT NULL,
    member_level    TEXT NOT NULL CHECK (member_level IN ('normal', 'silver', 'gold', 'platinum')),
    registered_at   TEXT NOT NULL
);

CREATE TABLE categories (
    id              INTEGER PRIMARY KEY,
    parent_id       INTEGER REFERENCES categories(id),
    name            TEXT NOT NULL UNIQUE
);

CREATE TABLE products (
    id              INTEGER PRIMARY KEY,
    sku             TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    category_id     INTEGER NOT NULL REFERENCES categories(id),
    brand           TEXT NOT NULL,
    cost_price      REAL NOT NULL CHECK (cost_price >= 0),
    list_price      REAL NOT NULL CHECK (list_price >= cost_price),
    status          TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    created_at      TEXT NOT NULL
);

CREATE TABLE warehouses (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    city            TEXT NOT NULL
);

CREATE TABLE inventory (
    product_id      INTEGER NOT NULL REFERENCES products(id),
    warehouse_id    INTEGER NOT NULL REFERENCES warehouses(id),
    stock_quantity  INTEGER NOT NULL CHECK (stock_quantity >= 0),
    safety_stock    INTEGER NOT NULL CHECK (safety_stock >= 0),
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (product_id, warehouse_id)
);

CREATE TABLE orders (
    id              INTEGER PRIMARY KEY,
    order_no        TEXT NOT NULL UNIQUE,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    status          TEXT NOT NULL CHECK (status IN ('pending', 'paid', 'shipped', 'completed', 'cancelled')),
    created_at      TEXT NOT NULL,
    paid_at         TEXT,
    shipped_at      TEXT,
    completed_at    TEXT,
    shipping_province TEXT NOT NULL,
    shipping_city   TEXT NOT NULL,
    coupon_amount   REAL NOT NULL DEFAULT 0 CHECK (coupon_amount >= 0),
    shipping_fee    REAL NOT NULL DEFAULT 0 CHECK (shipping_fee >= 0)
);

CREATE TABLE order_items (
    id              INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(id),
    product_id      INTEGER NOT NULL REFERENCES products(id),
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      REAL NOT NULL CHECK (unit_price >= 0),
    discount_amount REAL NOT NULL DEFAULT 0 CHECK (discount_amount >= 0),
    UNIQUE (order_id, product_id)
);

CREATE TABLE payments (
    id              INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL UNIQUE REFERENCES orders(id),
    payment_method  TEXT NOT NULL CHECK (payment_method IN ('alipay', 'wechat', 'bank_card')),
    amount          REAL NOT NULL CHECK (amount >= 0),
    status          TEXT NOT NULL CHECK (status IN ('success', 'refunded', 'failed')),
    paid_at         TEXT NOT NULL
);

CREATE TABLE refunds (
    id              INTEGER PRIMARY KEY,
    refund_no       TEXT NOT NULL UNIQUE,
    order_id        INTEGER NOT NULL REFERENCES orders(id),
    order_item_id   INTEGER NOT NULL UNIQUE REFERENCES order_items(id),
    user_id         INTEGER NOT NULL REFERENCES users(id),
    amount          REAL NOT NULL CHECK (amount >= 0),
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at      TEXT NOT NULL,
    processed_at    TEXT
);

CREATE TABLE reviews (
    id              INTEGER PRIMARY KEY,
    order_item_id   INTEGER NOT NULL UNIQUE REFERENCES order_items(id),
    user_id         INTEGER NOT NULL REFERENCES users(id),
    product_id      INTEGER NOT NULL REFERENCES products(id),
    rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_created ON orders(created_at);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_refunds_created ON refunds(created_at);
CREATE INDEX idx_reviews_product ON reviews(product_id);

CREATE VIEW order_amounts AS
SELECT
    o.id AS order_id,
    o.order_no,
    ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount), 2) AS item_amount,
    o.coupon_amount,
    o.shipping_fee,
    ROUND(SUM(oi.quantity * oi.unit_price - oi.discount_amount)
          - o.coupon_amount + o.shipping_fee, 2) AS payable_amount
FROM orders AS o
JOIN order_items AS oi ON oi.order_id = o.id
GROUP BY o.id;

CREATE VIEW product_sales AS
SELECT
    p.id AS product_id,
    p.name AS product_name,
    c.name AS category_name,
    COUNT(DISTINCT CASE WHEN o.status != 'cancelled' THEN o.id END) AS order_count,
    COALESCE(SUM(CASE WHEN o.status != 'cancelled' THEN oi.quantity ELSE 0 END), 0) AS units_sold,
    ROUND(COALESCE(SUM(CASE WHEN o.status != 'cancelled'
        THEN oi.quantity * oi.unit_price - oi.discount_amount ELSE 0 END), 0), 2) AS gross_revenue
FROM products AS p
JOIN categories AS c ON c.id = p.category_id
LEFT JOIN order_items AS oi ON oi.product_id = p.id
LEFT JOIN orders AS o ON o.id = oi.order_id
GROUP BY p.id;
