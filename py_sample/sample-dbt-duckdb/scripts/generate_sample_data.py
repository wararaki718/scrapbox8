from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "raw"
RANDOM_SEED = 42


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gen_customers(n: int) -> list[dict]:
    regions = ["APAC", "EMEA", "NA", "LATAM"]
    segments = ["consumer", "business", "enterprise"]
    base = datetime(2023, 1, 1)

    rows = []
    for i in range(1, n + 1):
        created_at = base + timedelta(days=random.randint(0, 730), hours=random.randint(0, 23))
        rows.append(
            {
                "customer_id": i,
                "customer_name": f"Customer_{i:04d}",
                "email": f"customer_{i:04d}@example.com",
                "region": random.choice(regions),
                "segment": random.choice(segments),
                "created_at": created_at.isoformat(sep=" "),
            }
        )
    return rows


def gen_products(n: int) -> list[dict]:
    categories = ["software", "hardware", "service", "subscription"]
    rows = []
    for i in range(1, n + 1):
        price = round(random.uniform(9.99, 999.99), 2)
        rows.append(
            {
                "product_id": i,
                "product_name": f"Product_{i:03d}",
                "category": random.choice(categories),
                "unit_price": price,
                "is_active": random.choice(["true", "true", "true", "false"]),
            }
        )
    return rows


def gen_orders(n_orders: int, n_customers: int) -> list[dict]:
    statuses = ["placed", "shipped", "delivered", "cancelled"]
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(1, n_orders + 1):
        ordered_at = base + timedelta(days=random.randint(0, 500), minutes=random.randint(0, 1439))
        rows.append(
            {
                "order_id": i,
                "customer_id": random.randint(1, n_customers),
                "order_status": random.choices(statuses, weights=[20, 25, 50, 5], k=1)[0],
                "ordered_at": ordered_at.isoformat(sep=" "),
            }
        )
    return rows


def gen_order_items(orders: list[dict], n_products: int, max_items_per_order: int = 4) -> list[dict]:
    rows = []
    item_id = 1
    for order in orders:
        num_items = random.randint(1, max_items_per_order)
        product_ids = random.sample(range(1, n_products + 1), k=min(num_items, n_products))
        for pid in product_ids:
            quantity = random.randint(1, 5)
            rows.append(
                {
                    "order_item_id": item_id,
                    "order_id": order["order_id"],
                    "product_id": pid,
                    "quantity": quantity,
                }
            )
            item_id += 1
    return rows


def main() -> None:
    random.seed(RANDOM_SEED)

    customers = gen_customers(120)
    products = gen_products(30)
    orders = gen_orders(1200, n_customers=len(customers))
    order_items = gen_order_items(orders, n_products=len(products))

    write_csv(
        OUT_DIR / "customers.csv",
        customers,
        ["customer_id", "customer_name", "email", "region", "segment", "created_at"],
    )
    write_csv(
        OUT_DIR / "products.csv",
        products,
        ["product_id", "product_name", "category", "unit_price", "is_active"],
    )
    write_csv(
        OUT_DIR / "orders.csv",
        orders,
        ["order_id", "customer_id", "order_status", "ordered_at"],
    )
    write_csv(
        OUT_DIR / "order_items.csv",
        order_items,
        ["order_item_id", "order_id", "product_id", "quantity"],
    )

    print(f"generated: {OUT_DIR}")
    print(f"customers={len(customers)}")
    print(f"products={len(products)}")
    print(f"orders={len(orders)}")
    print(f"order_items={len(order_items)}")


if __name__ == "__main__":
    main()
