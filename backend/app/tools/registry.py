from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools import implementations as impl
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


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: type[BaseModel]
    handler: Callable[[AsyncSession, BaseModel], Awaitable[dict]]
    mutates_state: bool
    requires_confirmation: bool


TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "get_customer",
            "Look up a customer's profile by customer ID.",
            GetCustomerInput,
            impl.get_customer,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "get_order",
            "Look up a single order by order ID, including its line items.",
            GetOrderInput,
            impl.get_order,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "list_customer_orders",
            "List recent orders belonging to a customer.",
            ListCustomerOrdersInput,
            impl.list_customer_orders,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "get_product",
            "Look up a product's details by product ID.",
            GetProductInput,
            impl.get_product,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "get_shipping_status",
            "Look up shipment/tracking status for an order.",
            GetShippingStatusInput,
            impl.get_shipping_status,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "check_refund_eligibility",
            "Check whether an order is eligible for a refund under policy rules.",
            CheckRefundEligibilityInput,
            impl.check_refund_eligibility,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "calculate_refund_amount",
            "Calculate the refund amount for an order, without performing the refund.",
            CalculateRefundAmountInput,
            impl.calculate_refund_amount,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "request_refund",
            "Actually process a refund for an order. Mutates order state — requires prior "
            "eligibility check, customer confirmation, and policy approval.",
            RequestRefundInput,
            impl.request_refund,
            mutates_state=True,
            requires_confirmation=True,
        ),
        ToolSpec(
            "cancel_order",
            "Cancel an order that has not yet shipped. Mutates order state.",
            CancelOrderInput,
            impl.cancel_order,
            mutates_state=True,
            requires_confirmation=True,
        ),
        ToolSpec(
            "create_support_ticket",
            "File a new support ticket for a customer.",
            CreateSupportTicketInput,
            impl.create_support_ticket,
            mutates_state=True,
            requires_confirmation=False,
        ),
        ToolSpec(
            "get_ticket",
            "Look up an existing support ticket by ticket ID.",
            GetTicketInput,
            impl.get_ticket,
            mutates_state=False,
            requires_confirmation=False,
        ),
        ToolSpec(
            "escalate_to_human",
            "Escalate the conversation to a human agent, creating or updating a ticket.",
            EscalateToHumanInput,
            impl.escalate_to_human,
            mutates_state=True,
            requires_confirmation=False,
        ),
    ]
}


def get_ollama_tool_definitions() -> list[dict[str, Any]]:
    """Ollama /api/chat `tools` payload, generated from the same Pydantic schemas used
    to validate execution — the model can never see a parameter shape it can't be
    validated against."""
    definitions = []
    for spec in TOOL_REGISTRY.values():
        schema = spec.input_schema.model_json_schema()
        schema.pop("title", None)
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": schema,
                },
            }
        )
    return definitions


class UnknownToolError(Exception):
    pass


class InvalidToolArgumentsError(Exception):
    pass


class ConfirmationRequiredError(Exception):
    """Raised when a mutating tool is invoked without confirmed=True.

    Carries the schema-validated arguments so the caller can present a confirmation
    prompt built from clean data, then re-invoke with confirmed=True to proceed.
    This check lives here — inside execute_tool, the actual allowlist/execution
    boundary — rather than only in a specific caller's loop, so no code path can
    reach a mutating tool without going through it, regardless of how it got called.
    """

    def __init__(self, tool_name: str, validated_arguments: dict):
        self.tool_name = tool_name
        self.validated_arguments = validated_arguments
        super().__init__(f"Tool '{tool_name}' requires explicit confirmation before executing.")


async def execute_tool(
    session: AsyncSession, name: str, raw_arguments: dict, confirmed: bool = False
) -> dict:
    """The allowlist gate: reject anything not in TOOL_REGISTRY, reject arguments that
    don't validate against that tool's schema, never execute arbitrary text as a
    tool call, and never run a tool marked requires_confirmation unless the caller
    explicitly passes confirmed=True. This is the boundary between "LLM requested an
    action" and "action ran" — the one place that boundary is enforced, not just in
    whichever caller happens to invoke it."""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise UnknownToolError(f"Tool '{name}' is not in the allowlist.")

    try:
        args = spec.input_schema.model_validate(raw_arguments)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(f"Arguments for tool '{name}' failed validation: {exc}") from exc

    if spec.requires_confirmation and not confirmed:
        raise ConfirmationRequiredError(name, args.model_dump())

    try:
        result = await spec.handler(session, args)
        return {"ok": True, "result": result}
    except impl.ToolError as exc:
        return {"ok": False, "error": str(exc)}
