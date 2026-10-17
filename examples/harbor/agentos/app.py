"""A small agno support agent served with AgentOS: the agent under test of the Harbor example.

It runs as the ``agent-api`` sidecar (see ``agentos-sidecar.yaml``) and answers on port 7777:
``POST /agents/support/runs``. The tools are deterministic, so the only randomness in a run is the model.

Run it locally (needs ``OPENAI_API_KEY``)::

    uv run --no-project --with 'agno[os]==3.1.1' --with openai python examples/harbor/agentos/app.py
    curl -s localhost:7777/agents/support/runs -F message='i was charged twice for 1042' -F stream=false
"""

import os

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIChat
from agno.os import AgentOS

# Order book of the demo shop. Refunds are tracked per process: each Harbor trial gets its own sidecar.
ORDERS: dict[str, dict[str, object]] = {
    "1042": {"status": "delivered", "item": "desk lamp", "charges": [49.0, 49.0]},
    "1043": {"status": "shipped", "item": "phone case", "charges": [15.5]},
    "1044": {"status": "processing", "item": "office chair", "charges": [189.0]},
}
REFUNDED: dict[str, float] = {}


def _order_id(order_id: str) -> str:
    return order_id.strip().lstrip("#")


def lookup_order(order_id: str) -> str:
    """Look an order up by its number: status, item, charges and the amount refunded so far."""
    key = _order_id(order_id)
    order = ORDERS.get(key)
    if order is None:
        return f"order {key} not found"
    return f"order {key}: {order}, refunded so far: {REFUNDED.get(key, 0.0):.2f}"


def issue_refund(order_id: str, amount: float) -> str:
    """Refund ``amount`` on an order. Only a charge that appears twice on the same order can be refunded, once."""
    key = _order_id(order_id)
    order = ORDERS.get(key)
    if order is None:
        return f"refused: order {key} not found"
    charges = order["charges"]
    if not isinstance(charges, list):
        return f"refused: order {key} has no charges"
    if charges.count(amount) < 2:
        return f"refused: {amount:.2f} is not a duplicated charge on order {key}"
    if key in REFUNDED:
        return f"refused: order {key} was already refunded {REFUNDED[key]:.2f}"
    REFUNDED[key] = amount
    return f"refunded {amount:.2f} on order {key}"


support_agent = Agent(
    id="support",
    name="Support desk",
    model=OpenAIChat(id=os.environ.get("AGENT_MODEL", "gpt-4.1-mini")),
    tools=[lookup_order, issue_refund],
    instructions=[
        "You handle billing questions for an online shop.",
        "Always look the order up before you promise or issue anything.",
        "Refund only charges that appear twice on the same order, and never invent amounts.",
        "If the user does not give an order number, ask for it.",
    ],
    # AgentOS keeps a conversation per session_id only when the agent has a db and adds the history to its context;
    # simulhausen's AgentOS adapter sends one session per dialogue.
    db=InMemoryDb(),
    add_history_to_context=True,
    num_history_runs=20,
    telemetry=False,
)

agent_os = AgentOS(id="simulhausen-demo", agents=[support_agent], telemetry=False)
app = agent_os.get_app()

if __name__ == "__main__":
    # All interfaces: the app runs in its own container and the task container reaches it as agent-api:7777.
    agent_os.serve(app=app, host="0.0.0.0", port=int(os.environ.get("PORT", "7777")))
