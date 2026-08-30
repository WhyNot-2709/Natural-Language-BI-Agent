"""
dataset.py

Owns:
1. Where the generated artifacts live (business DB, checkpoint DB, Chroma index dir)
2. The business database schema — defined once, as data, used to both create the
   SQLite tables and build the RAG schema-catalog index, so the two can never
   drift apart from each other.
3. Seed data generation (Faker + hand-placed rows for demo-friendly results).
4. RAG ingestion — builds the four Chroma indices from the seeded data.

Run directly (`python dataset.py`) to build everything from scratch.
"""

import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from faker import Faker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

fake = Faker()
Faker.seed(42)
random.seed(42)

# --- Storage locations (kept in code, not .env, per your call) ---
DATA_DIR = Path("./data")
BUSINESS_DB_PATH = DATA_DIR / "business.db"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.db"
CHROMA_DB_DIR = DATA_DIR / "chroma"
AUDIT_DB_PATH = DATA_DIR / "audit.db"

GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_RATE_LIMIT_SAFETY = int(os.environ.get("EMBED_RATE_LIMIT_SAFETY", "90"))
EMBED_RATE_LIMIT_PAUSE_SECONDS = int(os.environ.get("EMBED_RATE_LIMIT_PAUSE_SECONDS", "61"))

@dataclass
class Column:
    name: str
    sql_type: str
    description: str


@dataclass
class Table:
    name: str
    description: str
    primary_key: str
    columns: list[Column]
    foreign_keys: dict[str, str] = field(default_factory=dict)


SCHEMA: list[Table] = [
    Table(
        name="customers", description="One row per registered customer.", primary_key="customer_id",
        columns=[
            Column("customer_id", "INTEGER PRIMARY KEY", "Unique customer identifier"),
            Column("first_name", "TEXT", "Customer's first name"),
            Column("last_name", "TEXT", "Customer's last name"),
            Column("email", "TEXT", "Customer's email address"),
            Column("city", "TEXT", "Customer's city"),
            Column("state_code", "TEXT", "Two-letter US state code, e.g. 'CA' for California"),
            Column("signup_date", "TEXT", "Date the customer registered, ISO format"),
            Column("is_active", "INTEGER", "1 if the customer has ordered in the last 90 days, else 0"),
        ],
    ),
    Table(
        name="categories", description="Product category lookup table.", primary_key="category_id",
        columns=[
            Column("category_id", "INTEGER PRIMARY KEY", "Unique category identifier"),
            Column("category_name", "TEXT", "Category name, e.g. 'Electronics'"),
        ],
    ),
    Table(
        name="products", description="One row per sellable product.", primary_key="product_id",
        foreign_keys={"category_id": "categories.category_id"},
        columns=[
            Column("product_id", "INTEGER PRIMARY KEY", "Unique product identifier"),
            Column("sku", "TEXT", "Stock keeping unit code, e.g. 'MBP-M3-16GB-SLV'"),
            Column("product_name", "TEXT", "Human-readable product name"),
            Column("category_id", "INTEGER", "References categories.category_id"),
            Column("unit_price", "REAL", "Price charged to the customer, in USD"),
            Column("cost_price", "REAL", "Cost to acquire/produce the product, in USD"),
        ],
    ),
    Table(
        name="orders", description="One row per customer order.", primary_key="order_id",
        foreign_keys={"customer_id": "customers.customer_id"},
        columns=[
            Column("order_id", "INTEGER PRIMARY KEY", "Unique order identifier"),
            Column("customer_id", "INTEGER", "References customers.customer_id"),
            Column("order_date", "TEXT", "Date the order was placed, ISO format"),
            Column("status", "TEXT", "'placed', 'shipped', 'delivered', or 'cancelled'"),
            Column("total_amount", "REAL", "Total order value in USD, before refunds"),
        ],
    ),
    Table(
        name="order_items", description="Line items within an order.", primary_key="order_item_id",
        foreign_keys={"order_id": "orders.order_id", "product_id": "products.product_id"},
        columns=[
            Column("order_item_id", "INTEGER PRIMARY KEY", "Unique line item identifier"),
            Column("order_id", "INTEGER", "References orders.order_id"),
            Column("product_id", "INTEGER", "References products.product_id"),
            Column("quantity", "INTEGER", "Units of this product in the order"),
            Column("unit_price", "REAL", "Price per unit at time of purchase, in USD"),
        ],
    ),
    Table(
        name="payments", description="One row per payment attempt against an order.", primary_key="payment_id",
        foreign_keys={"order_id": "orders.order_id"},
        columns=[
            Column("payment_id", "INTEGER PRIMARY KEY", "Unique payment identifier"),
            Column("order_id", "INTEGER", "References orders.order_id"),
            Column("payment_method", "TEXT", "'credit_card', 'paypal', or 'gift_card'"),
            Column("amount", "REAL", "Amount charged in USD"),
            Column("payment_date", "TEXT", "Date of the payment attempt, ISO format"),
            Column("payment_status", "TEXT", "'succeeded', 'failed', or 'refunded'"),
        ],
    ),
    Table(
        name="shipments", description="One row per shipment for an order.", primary_key="shipment_id",
        foreign_keys={"order_id": "orders.order_id"},
        columns=[
            Column("shipment_id", "INTEGER PRIMARY KEY", "Unique shipment identifier"),
            Column("order_id", "INTEGER", "References orders.order_id"),
            Column("carrier", "TEXT", "Shipping carrier, e.g. 'UPS', 'FedEx', 'USPS'"),
            Column("shipped_date", "TEXT", "Date shipped, ISO format, nullable"),
            Column("delivered_date", "TEXT", "Date delivered, ISO format, nullable"),
            Column("status", "TEXT", "'pending', 'shipped', 'delivered', or 'delayed'"),
        ],
    ),
    Table(
        name="returns", description="One row per returned order line item.", primary_key="return_id",
        foreign_keys={"order_item_id": "order_items.order_item_id"},
        columns=[
            Column("return_id", "INTEGER PRIMARY KEY", "Unique return identifier"),
            Column("order_item_id", "INTEGER", "References order_items.order_item_id"),
            Column("return_date", "TEXT", "Date the return was filed, ISO format"),
            Column("reason", "TEXT", "'defective', 'wrong_item', 'no_longer_needed', or 'other'"),
            Column("refund_amount", "REAL", "Amount refunded in USD"),
        ],
    ),
    Table(
        name="reviews", description="One row per customer product review.", primary_key="review_id",
        foreign_keys={"product_id": "products.product_id", "customer_id": "customers.customer_id"},
        columns=[
            Column("review_id", "INTEGER PRIMARY KEY", "Unique review identifier"),
            Column("product_id", "INTEGER", "References products.product_id"),
            Column("customer_id", "INTEGER", "References customers.customer_id"),
            Column("rating", "INTEGER", "Star rating, 1 to 5"),
            Column("review_date", "TEXT", "Date the review was posted, ISO format"),
        ],
    ),
    Table(
        name="promotions", description="Discount codes and the window they were active.", primary_key="promotion_id",
        columns=[
            Column("promotion_id", "INTEGER PRIMARY KEY", "Unique promotion identifier"),
            Column("promo_code", "TEXT", "Discount code string, e.g. 'SUMMER25'"),
            Column("discount_percent", "REAL", "Discount percentage, 0-100"),
            Column("start_date", "TEXT", "Promotion start date, ISO format"),
            Column("end_date", "TEXT", "Promotion end date, ISO format"),
        ],
    ),
]


def create_business_db() -> sqlite3.Connection:
    """Creates business.db from SCHEMA. Safe to re-run — drops and recreates every table. Returns the open connection."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    cursor = conn.cursor()

    for table in SCHEMA:
        cursor.execute(f"DROP TABLE IF EXISTS {table.name}")
        columns_sql = ", ".join(f"{col.name} {col.sql_type}" for col in table.columns)
        cursor.execute(f"CREATE TABLE {table.name} ({columns_sql})")

    conn.commit()
    return conn


# --- Fixed reference data — small and structural enough to hand-author, not randomize ---

CATEGORY_NAMES = [
    "Electronics", "Home & Kitchen", "Sports & Outdoors", "Books",
    "Clothing", "Beauty", "Toys & Games", "Office Supplies",
]

US_STATES = [
    ("California", "CA"), ("Texas", "TX"), ("New York", "NY"),
    ("Florida", "FL"), ("Illinois", "IL"), ("Washington", "WA"),
    ("Georgia", "GA"), ("Ohio", "OH"),
]

PROMOTIONS = [
    ("WELCOME10", 10.0, "2025-01-01", "2025-12-31"),
    ("SUMMER25", 25.0, "2026-06-01", "2026-08-31"),
    ("FLASH15", 15.0, "2026-03-01", "2026-03-07"),
    ("VIP20", 20.0, "2026-01-01", "2026-12-31"),
    ("HOLIDAY30", 30.0, "2025-11-20", "2025-12-26"),
]


def generate_categories(conn: sqlite3.Connection) -> list[int]:
    """Inserts the fixed category list. Returns the generated category_ids."""
    cursor = conn.cursor()
    category_ids = []
    for name in CATEGORY_NAMES:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (name,))
        category_ids.append(cursor.lastrowid)
    conn.commit()
    return category_ids

PRODUCT_NAME_TEMPLATES: dict[str, list[str]] = {
    "Electronics": ["Wireless Charger", "Bluetooth Speaker", "Noise-Cancelling Headphones", "Smart Watch", "Portable Power Bank"],
    "Home & Kitchen": ["Stainless Steel Blender", "Non-Stick Frying Pan", "Electric Kettle", "Air Fryer", "Coffee Grinder"],
    "Sports & Outdoors": ["Yoga Mat", "Insulated Water Bottle", "Camping Tent", "Resistance Bands Set", "Hiking Backpack"],
    "Books": ["Hardcover Notebook", "Desk Planner", "Bookend Set", "Reading Light", "Leather Bookmark"],
    "Clothing": ["Cotton T-Shirt", "Running Shoes", "Fleece Jacket", "Denim Jeans", "Wool Beanie"],
    "Beauty": ["Facial Cleanser", "Vitamin C Serum", "Hair Dryer", "Makeup Brush Set", "Moisturizing Cream"],
    "Toys & Games": ["Building Block Set", "Board Game", "Puzzle Cube", "Remote Control Car", "Plush Toy"],
    "Office Supplies": ["Mechanical Pencil Set", "Desk Organizer", "Sticky Notes Pack", "Ergonomic Mouse", "Whiteboard Marker Set"],
}
PRODUCT_VARIANT_SUFFIXES = ["", " Pro", " Plus", " Mini", " X2", " Lite"]

def generate_products(conn: sqlite3.Connection, category_ids: list[int], count: int = 60) -> tuple[list[int], int]:
    """
    Generates `count` random products spread across categories, plus one
    hand-placed product that generate_returns() will later concentrate
    defective-item returns on. Returns (all product_ids, signal_product_id).
    """
    cursor = conn.cursor()
    product_ids = []
    category_id_to_name = dict(zip(category_ids, CATEGORY_NAMES))
    for _ in range(count):
        category_id = random.choice(category_ids)
        category_name = category_id_to_name[category_id]
        base_name = random.choice(PRODUCT_NAME_TEMPLATES[category_name])
        product_name = base_name + random.choice(PRODUCT_VARIANT_SUFFIXES)
        sku = f"{base_name[:3].upper()}-{fake.bothify('####')}"
        cost_price = round(random.uniform(5, 300), 2)
        unit_price = round(cost_price * random.uniform(1.3, 2.5), 2)
        cursor.execute(
            "INSERT INTO products (sku, product_name, category_id, unit_price, cost_price) "
            "VALUES (?, ?, ?, ?, ?)",
            (sku, product_name, category_id, unit_price, cost_price),
        )
        product_ids.append(cursor.lastrowid)

    # Hand-placed signal: this product gets an outsized share of returns
    # later, so "which product has the highest return rate" has a clean,
    # non-random answer to demo.
    cursor.execute(
        "INSERT INTO products (sku, product_name, category_id, unit_price, cost_price) "
        "VALUES (?, ?, ?, ?, ?)",
        ("WEB-PRO01", "Wireless Earbuds Pro", category_ids[0], 89.99, 32.00),
    )
    signal_product_id = cursor.lastrowid
    product_ids.append(signal_product_id)

    conn.commit()
    return product_ids, signal_product_id


def generate_customers(conn: sqlite3.Connection, count: int = 200) -> tuple[list[int], int]:
    """
    Generates `count` customers — skewed ~10% California (so an entity-
    resolution demo query like "customers in California" is interesting)
    and ~30% churned (is_active=0, feeds the churn glossary definition
    later) — plus one hand-placed VIP customer for the PII/HITL demo.
    Returns (all customer_ids, vip_customer_id).
    """
    cursor = conn.cursor()
    customer_ids = []

    for i in range(count):
        _, state_code = ("California", "CA") if i % 10 == 0 else random.choice(US_STATES)
        signup_date = fake.date_between(start_date="-2y", end_date="-30d")
        is_active = 1 if random.random() < 0.7 else 0

        cursor.execute(
            "INSERT INTO customers (first_name, last_name, email, city, state_code, signup_date, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fake.first_name(), fake.last_name(), fake.email(), fake.city(), state_code,
             signup_date.isoformat(), is_active),
        )
        customer_ids.append(cursor.lastrowid)

    # Hand-placed signal: a clear VIP (high implied order volume, CA-based) —
    # used later to demo the human-approval interrupt when a query would
    # surface this customer's PII.
    cursor.execute(
        "INSERT INTO customers (first_name, last_name, email, city, state_code, signup_date, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Priya", "Anand", "priya.anand@example.com", "San Francisco", "CA",
         (date.today() - timedelta(days=500)).isoformat(), 1),
    )
    vip_customer_id = cursor.lastrowid
    customer_ids.append(vip_customer_id)

    conn.commit()
    return customer_ids, vip_customer_id


def generate_promotions(conn: sqlite3.Connection) -> None:
    """Inserts the fixed promotions list."""
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO promotions (promo_code, discount_percent, start_date, end_date) VALUES (?, ?, ?, ?)",
        PROMOTIONS,
    )
    conn.commit()


def generate_orders(conn: sqlite3.Connection, customer_ids: list[int], vip_customer_id: int, count: int = 500) -> list[int]:
    """
    Generates `count` orders. The VIP customer is weighted ~5x more often
    than everyone else, so their order history is substantial enough to
    make the later PII/HITL demo convincing. total_amount is left at 0 here
    — it depends on line items, which don't exist yet — and backfilled below.
    """
    cursor = conn.cursor()
    order_ids = []
    statuses = ["delivered"] * 6 + ["shipped"] * 2 + ["placed"] + ["cancelled"]

    for _ in range(count):
        weights = [5 if cid == vip_customer_id else 1 for cid in customer_ids]
        customer_id = random.choices(customer_ids, weights=weights, k=1)[0]
        order_date = fake.date_between(start_date="-1y", end_date="today")
        status = random.choice(statuses)

        cursor.execute(
            "INSERT INTO orders (customer_id, order_date, status, total_amount) VALUES (?, ?, ?, ?)",
            (customer_id, order_date.isoformat(), status, 0.0),
        )
        order_ids.append(cursor.lastrowid)

    conn.commit()
    return order_ids


def generate_order_items(
    conn: sqlite3.Connection, order_ids: list[int], product_ids: list[int], signal_product_id: int
) -> tuple[list[int], list[int]]:
    """
    1-4 line items per order. The signal product gets slipped into ~15% of
    orders on top of their normal items, so it accumulates enough volume for
    its later return rate to be a clean, explainable outlier rather than noise.
    Returns (all order_item_ids, the subset using the signal product).
    """
    cursor = conn.cursor()
    order_item_ids, signal_item_ids = [], []
    price_lookup = dict(cursor.execute("SELECT product_id, unit_price FROM products").fetchall())

    for order_id in order_ids:
        chosen = random.sample(product_ids, k=min(random.randint(1, 4), len(product_ids)))
        if random.random() < 0.15 and signal_product_id not in chosen:
            chosen.append(signal_product_id)

        for product_id in chosen:
            quantity = random.randint(1, 3)
            cursor.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
                (order_id, product_id, quantity, price_lookup[product_id]),
            )
            item_id = cursor.lastrowid
            order_item_ids.append(item_id)
            if product_id == signal_product_id:
                signal_item_ids.append(item_id)

    conn.commit()
    return order_item_ids, signal_item_ids


def backfill_order_totals(conn: sqlite3.Connection) -> None:
    """Now that order_items exist, sum them per order and write the real total back onto orders."""
    conn.execute(
        "UPDATE orders SET total_amount = ("
        "  SELECT COALESCE(SUM(quantity * unit_price), 0) FROM order_items "
        "  WHERE order_items.order_id = orders.order_id"
        ")"
    )
    conn.commit()


def generate_payments(conn: sqlite3.Connection) -> None:
    """One payment per order, amount matching the now-backfilled order total."""
    cursor = conn.cursor()
    methods = ["credit_card", "credit_card", "paypal", "gift_card"]
    orders = cursor.execute("SELECT order_id, order_date, total_amount FROM orders").fetchall()

    for order_id, order_date, total_amount in orders:
        status = "succeeded" if random.random() < 0.92 else random.choice(["failed", "refunded"])
        cursor.execute(
            "INSERT INTO payments (order_id, payment_method, amount, payment_date, payment_status) "
            "VALUES (?, ?, ?, ?, ?)",
            (order_id, random.choice(methods), total_amount, order_date, status),
        )
    conn.commit()


def generate_shipments(conn: sqlite3.Connection) -> None:
    """One shipment per order that's past 'placed' — skips placed/cancelled orders entirely."""
    cursor = conn.cursor()
    carriers = ["UPS", "FedEx", "USPS"]
    orders = cursor.execute("SELECT order_id, order_date, status FROM orders").fetchall()

    for order_id, order_date, status in orders:
        if status in ("placed", "cancelled"):
            continue
        order_dt = date.fromisoformat(order_date)
        shipped_date = order_dt + timedelta(days=random.randint(1, 3))
        is_delayed = random.random() < 0.08
        delivered_date = None
        if status == "delivered":
            delay_days = random.randint(5, 10) if is_delayed else random.randint(2, 5)
            delivered_date = shipped_date + timedelta(days=delay_days)

        cursor.execute(
            "INSERT INTO shipments (order_id, carrier, shipped_date, delivered_date, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (order_id, random.choice(carriers), shipped_date.isoformat(),
             delivered_date.isoformat() if delivered_date else None,
             "delayed" if is_delayed else status),
        )
    conn.commit()


def generate_returns(conn: sqlite3.Connection, order_item_ids: list[int], signal_item_ids: list[int]) -> None:
    """
    Ordinary line items return ~5% of the time for a random reason. Signal
    items return ~50% of the time, always tagged 'defective' — this is what
    actually produces the "highest return rate" answer the demo relies on.
    """
    cursor = conn.cursor()
    reasons = ["wrong_item", "no_longer_needed", "other"]
    price_lookup = dict(cursor.execute("SELECT order_item_id, quantity * unit_price FROM order_items").fetchall())
    signal_set = set(signal_item_ids)

    for item_id in order_item_ids:
        chance = 0.5 if item_id in signal_set else 0.05
        if random.random() >= chance:
            continue
        reason = "defective" if item_id in signal_set else random.choice(reasons)
        return_date = fake.date_between(start_date="-6m", end_date="today")
        cursor.execute(
            "INSERT INTO returns (order_item_id, return_date, reason, refund_amount) VALUES (?, ?, ?, ?)",
            (item_id, return_date.isoformat(), reason, price_lookup.get(item_id, 0.0)),
        )
    conn.commit()


def generate_reviews(conn: sqlite3.Connection, product_ids: list[int], customer_ids: list[int], count: int = 300) -> None:
    """Reviews aren't tied to actual purchases — kept simple, this table exists mainly for rating-based queries."""
    cursor = conn.cursor()
    for _ in range(count):
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 5, 10, 30, 50], k=1)[0]
        cursor.execute(
            "INSERT INTO reviews (product_id, customer_id, rating, review_date) VALUES (?, ?, ?, ?)",
            (random.choice(product_ids), random.choice(customer_ids), rating,
             fake.date_between(start_date="-1y", end_date="today").isoformat()),
        )
    conn.commit()


# --- RAG grounding content ---

GOLDEN_QUERIES = [
    {
        "question": "How many customers do we have in California?",
        "sql": "SELECT COUNT(*) FROM customers WHERE state_code = 'CA';",
        "description": "Simple count with a state filter.",
    },
    {
        "question": "What are our top 5 best-selling products by revenue?",
        "sql": (
            "SELECT p.product_name, SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5;"
        ),
        "description": "Revenue ranking — join order_items to products, aggregate, sort, limit.",
    },
    {
        "question": "Which product has the highest return rate?",
        "sql": (
            "SELECT p.product_name, "
            "  CAST(COUNT(r.return_id) AS REAL) / COUNT(DISTINCT oi.order_item_id) AS return_rate "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "LEFT JOIN returns r ON r.order_item_id = oi.order_item_id "
            "GROUP BY p.product_id ORDER BY return_rate DESC LIMIT 1;"
        ),
        "description": "Return rate per product — LEFT JOIN so zero-return products aren't dropped from the ratio.",
    },
    {
        "question": "What was our gross revenue last month?",
        "sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) AS gross_revenue "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
            "WHERE o.order_date >= date('now', '-1 month');"
        ),
        "description": "Time-windowed revenue aggregation using SQLite's date() function.",
    },
    {
        "question": "How many shipments were delayed?",
        "sql": "SELECT COUNT(*) FROM shipments WHERE status = 'delayed';",
        "description": "Simple count with a status filter.",
    },
    {
        "question": "List our churned customers along with their last order date.",
        "sql": (
            "SELECT c.customer_id, c.first_name, c.last_name, MAX(o.order_date) AS last_order_date "
            "FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE c.is_active = 0 GROUP BY c.customer_id;"
        ),
        "description": "Churned-customer lookup joined against their most recent order — LEFT JOIN in case they never ordered.",
    },
    {
        "question": "What's the average order value by state?",
        "sql": (
            "SELECT c.state_code, AVG(o.total_amount) AS avg_order_value "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.state_code ORDER BY avg_order_value DESC;"
        ),
        "description": "AOV grouped by a joined dimension (state), not a column that lives on orders itself.",
    },
    {
        "question": "Which promotions are currently active?",
        "sql": "SELECT promo_code, discount_percent FROM promotions WHERE date('now') BETWEEN start_date AND end_date;",
        "description": "Date-range containment check against today's date.",
    },
    {
        "question": "What's the gross margin on our top-selling product?",
        "sql": (
            "SELECT p.product_name, (p.unit_price - p.cost_price) / p.unit_price AS gross_margin "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "GROUP BY p.product_id ORDER BY SUM(oi.quantity) DESC LIMIT 1;"
        ),
        "description": "Combines a ranking-subquery pattern with a margin calculation.",
    },
    {
        "question": "How many 5-star reviews has each product received?",
        "sql": (
            "SELECT p.product_name, COUNT(*) AS five_star_count "
            "FROM reviews r JOIN products p ON r.product_id = p.product_id "
            "WHERE r.rating = 5 GROUP BY p.product_id ORDER BY five_star_count DESC;"
        ),
        "description": "Filtered join-and-count, ranked.",
    },
]

BUSINESS_GLOSSARY = [
    {"term": "Active Customer", "definition": "A customer with is_active = 1 in the customers table (ordered within the last 90 days)."},
    {"term": "Churned Customer", "definition": "A customer with is_active = 0 in the customers table (has not ordered within the last 90 days)."},
    {"term": "Gross Margin", "definition": "(unit_price - cost_price) / unit_price for a product, from the products table."},
    {"term": "Gross Revenue", "definition": "SUM(quantity * unit_price) across order_items, optionally filtered by a date range on the joined orders table."},
    {"term": "Return Rate", "definition": "For a product: COUNT(returns) divided by COUNT(order_items) for that product, via a LEFT JOIN from order_items to returns."},
    {"term": "Average Order Value", "definition": "AVG(total_amount) from the orders table, often grouped by a dimension like state or time period. Abbreviated AOV."},
    {"term": "Delayed Shipment", "definition": "A row in the shipments table with status = 'delayed'."},
]


def build_schema_catalog_documents() -> tuple[list[str], list[str], list[dict]]:
    """One document per table: name + description + every column's name and description concatenated."""
    ids, texts, metadatas = [], [], []
    for table in SCHEMA:
        column_lines = "\n".join(f"  - {col.name} ({col.sql_type}): {col.description}" for col in table.columns)
        text = f"Table: {table.name}\n{table.description}\nColumns:\n{column_lines}"
        ids.append(f"table_{table.name}")
        texts.append(text)
        metadatas.append({"table_name": table.name})
    return ids, texts, metadatas


def build_golden_query_documents() -> tuple[list[str], list[str], list[dict]]:
    """One document per golden query, embedding the natural-language question — sql/description ride along as metadata."""
    ids, texts, metadatas = [], [], []
    for i, gq in enumerate(GOLDEN_QUERIES):
        ids.append(f"golden_query_{i}")
        texts.append(gq["question"])
        metadatas.append({"sql": gq["sql"], "description": gq["description"]})
    return ids, texts, metadatas


def build_entity_value_documents(conn: sqlite3.Connection) -> tuple[list[str], list[str], list[dict]]:
    """
    Distinct values from the low/medium-cardinality columns worth resolving:
    state_code, category_name, promo_code, product_name. High-cardinality
    columns (emails, cities) are deliberately excluded — those would need
    fuzzy string matching at query time instead of a vector index.
    """
    cursor = conn.cursor()
    sources = [
        ("customers", "state_code"),
        ("categories", "category_name"),
        ("promotions", "promo_code"),
        ("products", "product_name"),
    ]
    ids, texts, metadatas = [], [], []
    for table_name, column_name in sources:
        rows = cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name}").fetchall()
        for (value,) in rows:
            ids.append(f"entity_{table_name}_{column_name}_{value}")
            texts.append(str(value))
            metadatas.append({"table_name": table_name, "column_name": column_name, "exact_value": str(value)})
    return ids, texts, metadatas


def build_glossary_documents() -> tuple[list[str], list[str], list[dict]]:
    """One document per glossary term, embedding the term itself — definition rides along as metadata."""
    ids, texts, metadatas = [], [], []
    for entry in BUSINESS_GLOSSARY:
        ids.append(f"glossary_{entry['term']}")
        texts.append(entry["term"])
        metadatas.append({"definition": entry["definition"]})
    return ids, texts, metadatas


def build_rag_indices(conn: sqlite3.Connection) -> None:
    """
    One-time ingestion: builds all four Chroma collections.

    embed_documents() issues one Gemini API call per text, not one call for
    an entire list — the free tier's 100-requests/minute cap is shared
    across every collection in this run, not reset per collection. This
    embeds one text at a time and pauses a full minute whenever the running
    count crosses a safety margin, rather than assuming any single
    collection is small enough to be safe in isolation.
    """
    embedder = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    collections = {
        "schema_catalog": build_schema_catalog_documents(),
        "golden_queries": build_golden_query_documents(),
        "entity_values": build_entity_value_documents(conn),
        "business_glossary": build_glossary_documents(),
    }

    requests_this_window = 0

    for collection_name, (ids, texts, metadatas) in collections.items():
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

        embeddings = []
        for text in texts:
            if requests_this_window >= EMBED_RATE_LIMIT_SAFETY:
                print(f"  ...pausing {EMBED_RATE_LIMIT_PAUSE_SECONDS}s to stay under the embedding rate limit")
                time.sleep(EMBED_RATE_LIMIT_PAUSE_SECONDS)
                requests_this_window = 0
            embeddings.extend(embedder.embed_documents([text]))
            requests_this_window += 1

        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        print(f"  indexed '{collection_name}' ({len(texts)} documents)")


def build_dataset() -> None:
    """Runs the full pipeline in dependency order: schema, seed data, then RAG ingestion."""
    conn = create_business_db()

    category_ids = generate_categories(conn)
    product_ids, signal_product_id = generate_products(conn, category_ids)
    customer_ids, vip_customer_id = generate_customers(conn)
    generate_promotions(conn)

    order_ids = generate_orders(conn, customer_ids, vip_customer_id)
    order_item_ids, signal_item_ids = generate_order_items(conn, order_ids, product_ids, signal_product_id)
    backfill_order_totals(conn)
    generate_payments(conn)
    generate_shipments(conn)
    generate_returns(conn, order_item_ids, signal_item_ids)
    generate_reviews(conn, product_ids, customer_ids)

    build_rag_indices(conn)

    conn.close()


if __name__ == "__main__":
    build_dataset()
    print(f"Business DB created and seeded at {BUSINESS_DB_PATH}")
    print(f"RAG indices built at {CHROMA_DB_DIR}")