from pydantic import BaseModel, Field


class GetCustomerInput(BaseModel):
    customer_id: str


class GetOrderInput(BaseModel):
    order_id: str


class ListCustomerOrdersInput(BaseModel):
    customer_id: str
    limit: int = Field(default=10, ge=1, le=100)


class GetProductInput(BaseModel):
    product_id: str


class GetShippingStatusInput(BaseModel):
    order_id: str


class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(description="A natural-language question about company policy (refunds, shipping, cancellations, privacy, etc).")


class CheckRefundEligibilityInput(BaseModel):
    order_id: str


class CalculateRefundAmountInput(BaseModel):
    order_id: str


class RequestRefundInput(BaseModel):
    order_id: str
    reason: str


class CancelOrderInput(BaseModel):
    order_id: str
    reason: str


class CreateSupportTicketInput(BaseModel):
    customer_id: str
    subject: str
    description: str
    order_id: str = Field(default="", description="Related order ID, or empty string if none.")
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")


class GetTicketInput(BaseModel):
    ticket_id: str


class EscalateToHumanInput(BaseModel):
    customer_id: str
    reason: str
    ticket_id: str = Field(default="", description="Existing ticket ID to escalate, or empty string to create a new one.")
