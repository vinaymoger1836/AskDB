"""Build the seeded e-commerce demo database (data/sales.db).

Deterministic (fixed RNG seed) so the same questions always return the same
numbers — important for a reliable demo and for eval questions. Idempotent:
running it rebuilds the tables from scratch. Run as: `python -m data.seed`.

Schema (a small, familiar e-commerce shape):
    customers(id, name, email, country, signup_date)
    products(id, name, category, unit_price)
    orders(id, customer_id, order_date, status)
    order_items(id, order_id, product_id, quantity, unit_price)
"""

from __future__ import annotations

import logging
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Deterministic seed → reproducible demo numbers.
_RNG_SEED = 42

_COUNTRIES = ["USA", "UK", "Germany", "India", "Canada", "Australia", "France"]
_ORDER_STATUSES = ["completed", "completed", "completed", "shipped", "cancelled"]

# (name, category, unit_price) — a compact but varied catalog.
_PRODUCTS: list[tuple[str, str, float]] = [
    ("Wireless Mouse", "Electronics", 24.99),
    ("Mechanical Keyboard", "Electronics", 89.99),
    ("USB-C Hub", "Electronics", 39.99),
    ("Noise-Cancelling Headphones", "Electronics", 199.99),
    ("4K Monitor", "Electronics", 329.99),
    ("Laptop Stand", "Accessories", 34.99),
    ("Webcam 1080p", "Electronics", 59.99),
    ("Desk Lamp", "Home Office", 29.99),
    ("Office Chair", "Furniture", 249.99),
    ("Standing Desk", "Furniture", 399.99),
    ("Notebook (pack of 3)", "Stationery", 12.99),
    ("Gel Pens (pack of 12)", "Stationery", 8.49),
    ("Water Bottle", "Lifestyle", 19.99),
    ("Coffee Mug", "Lifestyle", 14.99),
    ("Backpack", "Accessories", 64.99),
]

_FIRST_NAMES = [
    "Alice", "Bob", "Carla", "David", "Emma", "Frank", "Grace", "Hiro",
    "Ivan", "Julia", "Kavya", "Liam", "Mia", "Noah", "Olga", "Priya",
    "Quinn", "Rahul", "Sara", "Tom", "Uma", "Victor", "Wendy", "Xavier",
]
_LAST_NAMES = [
    "Smith", "Jones", "Garcia", "Muller", "Patel", "Chen", "Kim", "Novak",
    "Rossi", "Dubois", "Silva", "Khan", "Nguyen", "Andersson", "Okafor",
]

_SCHEMA = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL,
    country     TEXT    NOT NULL,
    signup_date TEXT    NOT NULL
);

CREATE TABLE products (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    category   TEXT    NOT NULL,
    unit_price REAL    NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT    NOT NULL,
    status      TEXT    NOT NULL
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price REAL    NOT NULL
);
"""

# Tunable dataset size.
_NUM_CUSTOMERS = 120
_NUM_ORDERS = 600
_START_DATE = date(2022, 1, 1)
_END_DATE = date(2024, 12, 31)


def _random_date(rng: random.Random, start: date, end: date) -> date:
    """Return a uniformly random date in [start, end]."""
    span = (end - start).days
    return start + timedelta(days=rng.randint(0, span))


def build_database(db_path: str | Path | None = None) -> Path:
    """Create (or overwrite) the SQLite demo DB and populate it deterministically.

    Returns the path to the database file.
    """
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    rng = random.Random(_RNG_SEED)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)

        # Customers.
        customers = []
        for cid in range(1, _NUM_CUSTOMERS + 1):
            first = rng.choice(_FIRST_NAMES)
            last = rng.choice(_LAST_NAMES)
            name = f"{first} {last}"
            email = f"{first.lower()}.{last.lower()}{cid}@example.com"
            country = rng.choice(_COUNTRIES)
            signup = _random_date(rng, _START_DATE, _END_DATE).isoformat()
            customers.append((cid, name, email, country, signup))
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers
        )

        # Products.
        products = [
            (pid, name, category, price)
            for pid, (name, category, price) in enumerate(_PRODUCTS, start=1)
        ]
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

        # Orders + their line items.
        orders = []
        items = []
        item_id = 1
        for oid in range(1, _NUM_ORDERS + 1):
            customer_id = rng.randint(1, _NUM_CUSTOMERS)
            order_date = _random_date(rng, _START_DATE, _END_DATE).isoformat()
            status = rng.choice(_ORDER_STATUSES)
            orders.append((oid, customer_id, order_date, status))

            # 1–4 distinct products per order.
            num_lines = rng.randint(1, 4)
            chosen = rng.sample(products, num_lines)
            for _, prod_name, _cat, unit_price in chosen:
                product_id = next(p[0] for p in products if p[1] == prod_name)
                quantity = rng.randint(1, 5)
                items.append((item_id, oid, product_id, quantity, unit_price))
                item_id += 1
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?)", orders
        )
        conn.executemany(
            "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items
        )

        conn.commit()
    finally:
        conn.close()

    logger.info(
        "Seeded %s: %d customers, %d products, %d orders, %d line items",
        path, len(customers), len(products), len(orders), len(items),
    )
    return path


def ensure_database(db_path: str | Path | None = None) -> Path:
    """Build the demo DB only if it doesn't already exist. Returns its path."""
    path = Path(db_path or settings.db_path)
    if not path.exists():
        return build_database(path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = build_database()
    print(f"Built demo database at {out}")
