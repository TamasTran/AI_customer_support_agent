from datetime import datetime, timedelta, UTC

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    Customer,
    Order,
    OrderStatus,
    Product,
    Shipment,
    Ticket,
    TicketPriority,
    TicketStatus,
    ticket_id_seq,
)
from app.tools.schemas import (
    CalculateRefundAmountInput,
    CancelOrderInput,
    CheckRefundEligibilityInput,
    CreateSupportTicketInput,
    EscalateToHumanInput,
    GetCustomerInput,
    GetOrderInput,
    GetProductInput,
    GetShippingStatusInput,
    GetTicketInput,
    ListCustomerOrdersInput,
    RequestRefundInput,
)

REFUND_WINDOW_DAYS = 30
NON_CANCELLABLE_STATUSES = {OrderStatus.SHIPPED, OrderStatus.DELIVERED, OrderStatus.CANCELLED, OrderStatus.REFUNDED}


class ToolError(Exception):
    """A business-rule failure that should be reported back to the caller, not raised as a 500."""


def _order_to_dict(order: Order) -> dict:
    return {
        "order_id": order.id,
        "customer_id": order.customer_id,
        "status": order.status.value,
        "total_amount": float(order.total_amount),
        "created_at": order.created_at.isoformat(),
        "updated_at": order.updated_at.isoformat(),
        "items": [
            {
                "product_id": item.product_id,
                "quantity": item.quantity,
                "unit_price": float(item.unit_price),
            }
            for item in order.items
        ],
    }


async def get_customer(session: AsyncSession, args: GetCustomerInput) -> dict:
    customer = await session.get(Customer, args.customer_id)
    if not customer:
        raise ToolError(f"No customer found with id {args.customer_id}")
    return {
        "customer_id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "created_at": customer.created_at.isoformat(),
    }


async def get_order(session: AsyncSession, args: GetOrderInput) -> dict:
    result = await session.execute(
        select(Order).options(selectinload(Order.items)).where(Order.id == args.order_id)
    )
    order = result.scalar_one_or_none()
    if not order:
        raise ToolError(f"No order found with id {args.order_id}")
    return _order_to_dict(order)


async def list_customer_orders(session: AsyncSession, args: ListCustomerOrdersInput) -> dict:
    result = await session.execute(
        select(Order)
        .options(selectinload(Order.items))
        .where(Order.customer_id == args.customer_id)
        .order_by(Order.created_at.desc())
        .limit(args.limit)
    )
    orders = result.scalars().all()
    return {"orders": [_order_to_dict(o) for o in orders]}


async def get_product(session: AsyncSession, args: GetProductInput) -> dict:
    product = await session.get(Product, args.product_id)
    if not product:
        raise ToolError(f"No product found with id {args.product_id}")
    return {
        "product_id": product.id,
        "name": product.name,
        "description": product.description,
        "category": product.category,
        "price": float(product.price),
        "in_stock": product.in_stock,
    }


async def get_shipping_status(session: AsyncSession, args: GetShippingStatusInput) -> dict:
    result = await session.execute(select(Shipment).where(Shipment.order_id == args.order_id))
    shipment = result.scalar_one_or_none()
    if not shipment:
        raise ToolError(f"No shipment found for order {args.order_id}")
    return {
        "order_id": shipment.order_id,
        "carrier": shipment.carrier,
        "tracking_number": shipment.tracking_number,
        "status": shipment.status.value,
        "estimated_delivery": shipment.estimated_delivery.isoformat(),
        "shipped_at": shipment.shipped_at.isoformat() if shipment.shipped_at else None,
        "delivered_at": shipment.delivered_at.isoformat() if shipment.delivered_at else None,
    }


async def _load_order(session: AsyncSession, order_id: str) -> Order:
    order = await session.get(Order, order_id)
    if not order:
        raise ToolError(f"No order found with id {order_id}")
    return order


async def _next_ticket_id(session: AsyncSession) -> str:
    """Atomic, DB-native ID allocation — unlike counting existing rows, this can't
    collide under concurrent ticket creation or after a row has been deleted."""
    next_val = await session.scalar(select(ticket_id_seq.next_value()))
    return f"TICK-{next_val:06d}"


async def check_refund_eligibility(session: AsyncSession, args: CheckRefundEligibilityInput) -> dict:
    order = await _load_order(session, args.order_id)

    if order.status == OrderStatus.REFUNDED:
        return {"order_id": order.id, "eligible": False, "reason": "Order has already been refunded."}
    if order.status == OrderStatus.CANCELLED:
        return {"order_id": order.id, "eligible": False, "reason": "Order was cancelled; nothing to refund."}
    if order.status not in (OrderStatus.PAID, OrderStatus.SHIPPED, OrderStatus.DELIVERED):
        return {"order_id": order.id, "eligible": False, "reason": f"Order status '{order.status.value}' is not refundable."}

    created_at = order.created_at.replace(tzinfo=UTC) if order.created_at.tzinfo is None else order.created_at
    age = datetime.now(UTC) - created_at
    if age > timedelta(days=REFUND_WINDOW_DAYS):
        return {
            "order_id": order.id,
            "eligible": False,
            "reason": f"Order was placed {age.days} days ago, outside the {REFUND_WINDOW_DAYS}-day refund window.",
        }

    return {"order_id": order.id, "eligible": True, "reason": "Order is within the refund window and has not been refunded."}


async def calculate_refund_amount(session: AsyncSession, args: CalculateRefundAmountInput) -> dict:
    order = await _load_order(session, args.order_id)
    return {"order_id": order.id, "refund_amount": float(order.total_amount), "currency": "USD"}


async def request_refund(session: AsyncSession, args: RequestRefundInput) -> dict:
    """Perform the refund. Callers MUST have already run check_refund_eligibility and
    obtained customer confirmation + any required policy approval before invoking this —
    this function re-validates but does not perform confirmation/approval itself."""
    order = await _load_order(session, args.order_id)

    eligibility = await check_refund_eligibility(session, CheckRefundEligibilityInput(order_id=args.order_id))
    if not eligibility["eligible"]:
        raise ToolError(f"Refund cannot be processed: {eligibility['reason']}")

    order.status = OrderStatus.REFUNDED
    order.updated_at = datetime.utcnow()
    await session.commit()

    return {
        "order_id": order.id,
        "refunded_amount": float(order.total_amount),
        "reason": args.reason,
        "status": "refunded",
    }


async def cancel_order(session: AsyncSession, args: CancelOrderInput) -> dict:
    order = await _load_order(session, args.order_id)

    if order.status in NON_CANCELLABLE_STATUSES:
        raise ToolError(f"Order {order.id} cannot be cancelled: status is '{order.status.value}'.")

    order.status = OrderStatus.CANCELLED
    order.updated_at = datetime.utcnow()
    await session.commit()

    return {"order_id": order.id, "status": "cancelled", "reason": args.reason}


async def create_support_ticket(session: AsyncSession, args: CreateSupportTicketInput) -> dict:
    customer = await session.get(Customer, args.customer_id)
    if not customer:
        raise ToolError(f"No customer found with id {args.customer_id}")

    ticket_id = await _next_ticket_id(session)
    now = datetime.utcnow()
    ticket = Ticket(
        id=ticket_id,
        customer_id=args.customer_id,
        order_id=args.order_id or None,
        subject=args.subject,
        description=args.description,
        status=TicketStatus.OPEN,
        priority=TicketPriority(args.priority),
        created_at=now,
        updated_at=now,
    )
    session.add(ticket)
    await session.commit()

    return {"ticket_id": ticket.id, "status": ticket.status.value, "priority": ticket.priority.value}


async def get_ticket(session: AsyncSession, args: GetTicketInput) -> dict:
    ticket = await session.get(Ticket, args.ticket_id)
    if not ticket:
        raise ToolError(f"No ticket found with id {args.ticket_id}")
    return {
        "ticket_id": ticket.id,
        "customer_id": ticket.customer_id,
        "order_id": ticket.order_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "status": ticket.status.value,
        "priority": ticket.priority.value,
        "assigned_to": ticket.assigned_to,
        "created_at": ticket.created_at.isoformat(),
        "updated_at": ticket.updated_at.isoformat(),
    }


async def escalate_to_human(session: AsyncSession, args: EscalateToHumanInput) -> dict:
    if args.ticket_id:
        ticket = await session.get(Ticket, args.ticket_id)
        if not ticket:
            raise ToolError(f"No ticket found with id {args.ticket_id}")
    else:
        customer = await session.get(Customer, args.customer_id)
        if not customer:
            raise ToolError(f"No customer found with id {args.customer_id}")
        ticket = Ticket(
            id=await _next_ticket_id(session),
            customer_id=args.customer_id,
            order_id=None,
            subject="Escalated by AI agent",
            description=args.reason,
            status=TicketStatus.OPEN,
            priority=TicketPriority.HIGH,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(ticket)

    ticket.status = TicketStatus.ESCALATED
    ticket.priority = TicketPriority.URGENT
    ticket.updated_at = datetime.utcnow()
    await session.commit()

    return {"ticket_id": ticket.id, "status": ticket.status.value, "priority": ticket.priority.value}
