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
    SearchKnowledgeBaseInput,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: type[BaseModel]
    handler: Callable[[AsyncSession, BaseModel], Awaitable[dict]]
    mutates_state: bool
    requires_confirmation: bool
    # Distinct from requires_confirmation: confirmation is the CUSTOMER agreeing to
    # the action; this is a STAFF member signing off on it, for actions above some
    # risk threshold even after the customer has already agreed. Given (session,
    # validated_args), returns (needs_approval, reason). None means this tool never
    # needs staff approval.
    human_approval_check: Callable[[AsyncSession, BaseModel], Awaitable[tuple[bool, str]]] | None = None


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
            "search_knowledge_base",
            "Search company policy documents (refunds, shipping, cancellations, "
            "privacy) for an answer to a policy question. Only returns currently "
            "active policy versions.",
            SearchKnowledgeBaseInput,
            impl.search_knowledge_base,
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
            human_approval_check=impl.refund_requires_human_approval,
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


class HumanApprovalRequiredError(Exception):
    """Raised when a tool's human_approval_check says this action needs staff
    sign-off, even though the customer has already confirmed it. Distinct tier from
    ConfirmationRequiredError: that one is the customer agreeing to the action, this
    one is a human employee reviewing it — e.g. a refund large enough that a customer
    saying "yes" isn't sufficient authorization on its own. Carries the
    schema-validated arguments so the caller can queue a real approval record built
    from clean data.
    """

    def __init__(self, tool_name: str, validated_arguments: dict, reason: str):
        self.tool_name = tool_name
        self.validated_arguments = validated_arguments
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' requires human approval before executing: {reason}")


async def execute_tool(
    session: AsyncSession,
    name: str,
    raw_arguments: dict,
    confirmed: bool = False,
    human_approved: bool = False,
) -> dict:
    """The allowlist gate: reject anything not in TOOL_REGISTRY, reject arguments that
    don't validate against that tool's schema, never execute arbitrary text as a
    tool call, never run a tool marked requires_confirmation unless the caller
    explicitly passes confirmed=True, and never run a tool whose human_approval_check
    says it needs staff sign-off unless the caller explicitly passes
    human_approved=True. This is the boundary between "LLM requested an action" and
    "action ran" — the one place every one of these checks is enforced, not just in
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

    if spec.human_approval_check is not None and not human_approved:
        needs_approval, reason = await spec.human_approval_check(session, args)
        if needs_approval:
            raise HumanApprovalRequiredError(name, args.model_dump(), reason)

    try:
        result = await spec.handler(session, args)
        return {"ok": True, "result": result}
    except impl.ToolError as exc:
        return {"ok": False, "error": str(exc)}
