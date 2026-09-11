"""DB-backed test fixtures.

Assumes a real Postgres (with pgvector + alembic migrations already applied) is
reachable at DATABASE_URL — CI spins one up as a service container and runs
`alembic upgrade head` before pytest. Each test runs inside a transaction that's
rolled back afterward, using a SAVEPOINT so that `session.commit()` calls inside
the code under test (implementations.py does commit) don't end the outer
transaction early.
"""
from datetime import datetime

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import engine
from app.models import Customer, Order, OrderStatus


@pytest_asyncio.fixture
async def db_session():
    async with engine.connect() as conn:
        trans = await conn.begin()
        session_factory = async_sessionmaker(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        async with session_factory() as session:
            yield session
        await trans.rollback()


@pytest_asyncio.fixture
async def customer(db_session):
    c = Customer(
        id="CUST-TEST01",
        name="Test Customer",
        email="test-customer@example.com",
        phone="555-000-1111",
        created_at=datetime.utcnow(),
    )
    db_session.add(c)
    await db_session.commit()
    return c


@pytest_asyncio.fixture
async def pending_order(db_session, customer):
    o = Order(
        id="ORD-TEST01",
        customer_id=customer.id,
        status=OrderStatus.PENDING,
        total_amount=42.50,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db_session.add(o)
    await db_session.commit()
    return o
