"""A toy support agent for the example scenarios: it can look orders up and refund them."""

from __future__ import annotations

from agno.agent import Agent
from agno.models.openai import OpenAIChat

_ORDERS: dict[str, dict[str, object]] = {
    "1042": {"status": "delivered", "charges": [49.0, 49.0], "refunded": 0.0},
    "1043": {"status": "shipped", "charges": [15.5], "refunded": 0.0},
}


def lookup_order(order_id: str) -> str:
    """Look an order up by its number: status, charges and refunds so far."""
    order = _ORDERS.get(order_id.strip().lstrip("#"))
    if order is None:
        return f"order {order_id} not found"
    return f"order {order_id}: {order}"


def issue_refund(order_id: str, amount: float) -> str:
    """Refund ``amount`` on an order. Only for duplicated charges."""
    order = _ORDERS.get(order_id.strip().lstrip("#"))
    if order is None:
        return f"order {order_id} not found"
    return f"refunded {amount:.2f} on order {order_id}"


def build_refund_agent() -> Agent:
    return Agent(
        name="Refund desk",
        model=OpenAIChat(id="gpt-4.1-mini"),
        tools=[lookup_order, issue_refund],
        instructions=[
            "You handle refund requests for an online shop.",
            "Always look the order up before you promise or issue anything.",
            "Refund only charges that are duplicated; never invent amounts.",
            "If the user does not give an order number, ask for it.",
        ],
        telemetry=False,
    )
