"""Populate the database with synthetic customers/products/orders/shipments/tickets.

Usage:
    uv run python scripts/generate_synthetic_data.py
    uv run python scripts/generate_synthetic_data.py --customers 200 --orders 1000
"""
import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faker import Faker
from sqlalchemy import text

from app.db import async_session, engine
from app.models import (
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    Shipment,
    ShipmentStatus,
    Ticket,
    TicketPriority,
    TicketStatus,
)

fake = Faker()

PRODUCT_CATEGORIES = [
    "Electronics",
    "Home & Kitchen",
    "Clothing",
    "Sports & Outdoors",
    "Books",
    "Toys & Games",
    "Beauty",
    "Automotive",
]

CARRIERS = ["FedEx", "UPS", "DHL", "USPS"]

TICKET_SUBJECTS = [
    "Order not received",
    "Wrong item shipped",
    "Refund request",
    "Product defective",
    "Cancel my order",
    "Where is my package",
    "Billing question",
    "Account access issue",
]


async def generate(n_customers: int, n_products: int, n_orders: int, n_shipments: int, n_tickets: int) -> None:
    async with async_session() as session:
        print(f"Generating {n_customers} customers...")
        customers = []
        for i in range(1, n_customers + 1):
            customer_id = f"CUST-{i:05d}"
            customers.append(
                Customer(
                    id=customer_id,
                    name=fake.name(),
                    email=fake.unique.email(),
                    phone=fake.phone_number(),
                    created_at=fake.date_time_between(start_date="-2y", end_date="-30d"),
                )
            )
        session.add_all(customers)
        await session.commit()

        print(f"Generating {n_products} products...")
        products = []
        for i in range(1, n_products + 1):
            products.append(
                Product(
                    id=f"PROD-{i:05d}",
                    name=fake.catch_phrase(),
                    description=fake.text(max_nb_chars=200),
                    category=random.choice(PRODUCT_CATEGORIES),
                    price=round(random.uniform(5, 500), 2),
                    in_stock=random.random() > 0.05,
                )
            )
        session.add_all(products)
        await session.commit()

        customer_ids = [c.id for c in customers]
        product_ids = [p.id for p in products]
        product_prices = {p.id: float(p.price) for p in products}

        print(f"Generating {n_orders} orders (with items)...")
        order_ids = []
        order_statuses = list(OrderStatus)
        order_weights = [0.10, 0.10, 0.25, 0.45, 0.05, 0.05]
        batch = []
        for i in range(1, n_orders + 1):
            order_id = f"ORD-{10000 + i}"
            order_ids.append(order_id)
            created_at = fake.date_time_between(start_date="-1y", end_date="now")
            status = random.choices(order_statuses, weights=order_weights, k=1)[0]

            item_count = random.randint(1, 4)
            chosen_products = random.sample(product_ids, k=min(item_count, len(product_ids)))
            items = [
                OrderItem(
                    order_id=order_id,
                    product_id=pid,
                    quantity=random.randint(1, 3),
                    unit_price=product_prices[pid],
                )
                for pid in chosen_products
            ]
            total = sum(item.quantity * float(item.unit_price) for item in items)

            order = Order(
                id=order_id,
                customer_id=random.choice(customer_ids),
                status=status,
                total_amount=round(total, 2),
                created_at=created_at,
                updated_at=created_at + timedelta(days=random.randint(0, 5)),
            )
            batch.append(order)
            batch.extend(items)

            if len(batch) >= 2000:
                session.add_all(batch)
                await session.commit()
                batch = []
                print(f"  ...{i}/{n_orders} orders")
        if batch:
            session.add_all(batch)
            await session.commit()

        print(f"Generating {n_shipments} shipments...")
        shippable_order_ids = random.sample(order_ids, k=min(n_shipments, len(order_ids)))
        shipments = []
        for i, order_id in enumerate(shippable_order_ids, start=1):
            status = random.choice(list(ShipmentStatus))
            shipped_at = fake.date_time_between(start_date="-6M", end_date="now") if status != ShipmentStatus.PROCESSING else None
            delivered_at = shipped_at + timedelta(days=random.randint(1, 7)) if status == ShipmentStatus.DELIVERED and shipped_at else None
            shipments.append(
                Shipment(
                    id=f"SHIP-{i:06d}",
                    order_id=order_id,
                    carrier=random.choice(CARRIERS),
                    tracking_number=fake.bothify(text="??########"),
                    status=status,
                    estimated_delivery=(shipped_at or datetime.utcnow()) + timedelta(days=5),
                    shipped_at=shipped_at,
                    delivered_at=delivered_at,
                )
            )
        session.add_all(shipments)
        await session.commit()

        print(f"Generating {n_tickets} tickets...")
        tickets = []
        for i in range(1, n_tickets + 1):
            customer_id = random.choice(customer_ids)
            related_order = random.choice(order_ids) if random.random() > 0.2 else None
            created_at = fake.date_time_between(start_date="-6M", end_date="now")
            tickets.append(
                Ticket(
                    id=f"TICK-{i:06d}",
                    customer_id=customer_id,
                    order_id=related_order,
                    subject=random.choice(TICKET_SUBJECTS),
                    description=fake.paragraph(nb_sentences=3),
                    status=random.choice(list(TicketStatus)),
                    priority=random.choice(list(TicketPriority)),
                    assigned_to=fake.name() if random.random() > 0.5 else None,
                    created_at=created_at,
                    updated_at=created_at + timedelta(hours=random.randint(1, 72)),
                )
            )
        session.add_all(tickets)
        await session.commit()

    print("Done.")


async def truncate_all() -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE tickets, shipments, order_items, orders, products, customers "
                "RESTART IDENTITY CASCADE"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=1000)
    parser.add_argument("--products", type=int, default=500)
    parser.add_argument("--orders", type=int, default=5000)
    parser.add_argument("--shipments", type=int, default=1000)
    parser.add_argument("--tickets", type=int, default=1000)
    parser.add_argument("--reset", action="store_true", help="Truncate business tables first")
    args = parser.parse_args()

    async def run():
        if args.reset:
            print("Truncating existing business data...")
            await truncate_all()
        await generate(args.customers, args.products, args.orders, args.shipments, args.tickets)

    asyncio.run(run())


if __name__ == "__main__":
    main()
