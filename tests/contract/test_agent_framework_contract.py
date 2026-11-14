"""Contract test of ``agent_framework_adapter`` against the real Microsoft Agent Framework, with a scripted chat
client and no network.

Run it with the framework overlaid on the project environment (the orchestrations package brings the
``SequentialBuilder`` / ``HandoffBuilder`` workflows; without it those tests are skipped)::

    uv run --no-group harbor --with agent-framework-core --with agent-framework-orchestrations \
        pytest tests/contract/test_agent_framework_contract.py --no-cov -q
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("agent_framework")

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
    tool,
)

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.agent_framework import agent_framework_adapter

pytestmark = pytest.mark.filterwarnings("ignore")

USAGE: Any = {"input_token_count": 5, "output_token_count": 2, "total_token_count": 7}


def _flat(message: Message) -> str:
    chunks: list[str] = []
    for content in message.contents:
        if content.type == "text":
            chunks.append(content.text or "")
        elif content.type == "function_call":
            chunks.append(f"call:{content.name}")
        elif content.type == "function_result":
            chunks.append(f"result:{content.result}")
    return f"{message.role}:{'|'.join(chunks)}"


class ScriptedClient(FunctionInvocationLayer, BaseChatClient):  # type: ignore[misc]
    """Answers with the next scripted item (a text or a ``(tool, args)`` pair) and keeps every request's messages."""

    def __init__(self, *script: str | tuple[str, dict[str, Any]]) -> None:
        super().__init__()
        self.script = list(script)
        self.requests: list[list[str]] = []

    def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any) -> Any:
        self.requests.append([_flat(message) for message in messages])
        item = self.script.pop(0)
        if isinstance(item, str):
            contents = [Content.from_text(item)]
        else:
            call_id = f"call-{len(self.requests)}"
            contents = [Content.from_function_call(call_id=call_id, name=item[0], arguments=item[1])]
        if stream:
            # Workflows run their agents in streaming mode.
            async def updates() -> Any:
                yield ChatResponseUpdate(role="assistant", contents=[*contents, Content.from_usage(USAGE)])

            return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

        async def response() -> ChatResponse:
            return ChatResponse(messages=[Message("assistant", contents)], usage_details=USAGE)

        return response()


@tool
def lookup_order(order_id: str) -> dict[str, str]:
    """Look up an order."""
    return {"order_id": order_id, "status": "shipped"}


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    raise RuntimeError(f"order {order_id} is locked")


@tool(approval_mode="always_require")
def refund(order_id: str) -> str:
    """Refund an order."""
    return "refunded"


def support_agent(client: ScriptedClient) -> Agent:
    return Agent(client=client, name="support", tools=[lookup_order, cancel_order, refund])


async def play(adapter: Any, *script: Any) -> sim.ScenarioResult:
    return await sim.run(name="contract", description="agent framework contract", agents=[adapter], script=list(script))


async def test_reply_tool_call_usage_and_session_memory() -> None:
    client = ScriptedClient(("lookup_order", {"order_id": "7"}), "Order 7 has shipped.", "You asked about order 7.")
    seen: list[ScenarioState] = []

    result = await play(
        agent_framework_adapter(lambda: support_agent(client)),
        sim.user("Where is order 7?"),
        sim.agent(),
        sim.user("What did I ask?"),
        sim.agent(),
        seen.append,
    )

    state = seen[0]
    assert [m["content"] for m in result.messages if m["role"] == "assistant"] == [
        "Order 7 has shipped.",
        "You asked about order 7.",
    ]
    assert state.has_tool_call("lookup_order")
    calls = state.tool_calls(include_errors=True)
    assert [(c.name, c.input, c.output, c.turn) for c in calls] == [
        ("lookup_order", {"order_id": "7"}, '{"order_id": "7", "status": "shipped"}', 0)
    ]
    # The second turn's request carries the first exchange, the tool call and its result included.
    assert client.requests[-1] == [
        "user:Where is order 7?",
        "assistant:call:lookup_order",
        'tool:result:{"order_id": "7", "status": "shipped"}',
        "assistant:Order 7 has shipped.",
        "user:What did I ask?",
    ]
    assert state.usage == Usage(input_tokens=15, output_tokens=6, total_tokens=21)


async def test_raising_tool_is_recorded_as_an_error() -> None:
    client = ScriptedClient(("cancel_order", {"order_id": "7"}), "I could not cancel it.")
    seen: list[ScenarioState] = []

    await play(agent_framework_adapter(support_agent(client)), sim.user("Cancel order 7"), sim.agent(), seen.append)

    state = seen[0]
    assert not state.has_tool_call("cancel_order")
    failed = state.tool_calls("cancel_order", include_errors=True).last
    assert failed.status == "error"
    assert failed.input == {"order_id": "7"}
    assert failed.output == "Error: Function failed."
    assert failed.execution is not None
    assert failed.execution.tool_call_error is True


async def test_call_waiting_for_approval_is_not_executed() -> None:
    client = ScriptedClient(("refund", {"order_id": "7"}))
    seen: list[ScenarioState] = []

    await play(agent_framework_adapter(support_agent(client)), sim.user("Refund order 7"), sim.agent(), seen.append)

    state = seen[0]
    assert not state.has_tool_call("refund")
    assert [(c.name, c.status, c.input) for c in state.tool_calls(include_errors=True)] == [
        ("refund", "paused", {"order_id": "7"})
    ]


async def test_transcript_history_forgets_a_rolled_back_exchange() -> None:
    client = ScriptedClient("one", "two", "three")

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(2)

    await play(
        agent_framework_adapter(support_agent(client), history="transcript"),
        sim.user("first"),
        sim.agent(),
        sim.user("secret"),
        sim.agent(),
        backtrack,
        sim.user("third"),
        sim.agent(),
    )

    assert client.requests[-1] == ["user:first", "assistant:one", "user:third"]


async def test_session_history_restarts_from_the_transcript_after_a_rollback() -> None:
    client = ScriptedClient("one", "two", "three")

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(2)

    await play(
        agent_framework_adapter(support_agent(client)),
        sim.user("first"),
        sim.agent(),
        sim.user("secret"),
        sim.agent(),
        backtrack,
        sim.user("third"),
        sim.agent(),
    )

    assert client.requests[1] == ["user:first", "assistant:one", "user:secret"]
    assert client.requests[-1] == ["user:first", "assistant:one", "user:third"]


async def test_one_agent_serves_scenarios_in_a_row_without_shared_memory() -> None:
    client = ScriptedClient("one", "two")
    adapter = agent_framework_adapter(support_agent(client))

    await play(adapter, sim.user("first scenario"), sim.agent())
    await play(adapter, sim.user("second scenario"), sim.agent())

    assert client.requests[-1] == ["user:second scenario"]


async def test_sequential_workflow_reports_every_participants_tools_and_usage() -> None:
    pytest.importorskip("agent_framework_orchestrations")
    orchestrations = pytest.importorskip("agent_framework.orchestrations")
    client = ScriptedClient(("lookup_order", {"order_id": "7"}), "draft", "polished", "second draft", "second polish")

    def pipeline() -> Any:
        writer = Agent(client=client, name="writer", tools=[lookup_order])
        editor = Agent(client=client, name="editor")
        return orchestrations.SequentialBuilder(participants=[writer, editor]).build()

    seen: list[ScenarioState] = []
    result = await play(
        agent_framework_adapter(pipeline),
        sim.user("Write about order 7"),
        sim.agent(),
        sim.user("Again"),
        sim.agent(),
        seen.append,
    )

    state = seen[0]
    assert [m["content"] for m in result.messages if m["role"] == "assistant"] == ["polished", "second polish"]
    assert state.tool_call_names() == ["lookup_order"]
    assert state.tool_calls("lookup_order").last.turn == 0
    # Three model calls in the first turn, two in the second.
    assert state.usage == Usage(input_tokens=25, output_tokens=10, total_tokens=35)
    # The workflow's executors remember the first turn.
    assert client.requests[3][0] == "user:Write about order 7"
    assert client.requests[3][-1] == "user:Again"


async def test_handoff_workflow_answers_the_pending_user_turn() -> None:
    pytest.importorskip("agent_framework_orchestrations")
    orchestrations = pytest.importorskip("agent_framework.orchestrations")
    client = ScriptedClient(
        ("handoff_to_orders", {}), ("lookup_order", {"order_id": "7"}), "Order 7 has shipped.", "Anything else?"
    )

    def desk() -> Any:
        options: Any = {"require_per_service_call_history_persistence": True}
        front = Agent(client=client, name="front", **options)
        orders = Agent(client=client, name="orders", tools=[lookup_order], **options)
        return orchestrations.HandoffBuilder(participants=[front, orders]).with_start_agent(front).build()

    seen: list[ScenarioState] = []
    result = await play(
        agent_framework_adapter(desk),
        sim.user("Where is order 7?"),
        sim.agent(),
        sim.user("Thanks"),
        sim.agent(),
        seen.append,
    )

    state = seen[0]
    assert [m["content"] for m in result.messages if m["role"] == "assistant"] == [
        "Order 7 has shipped.",
        "Anything else?",
    ]
    assert state.has_tool_call("lookup_order")
    assert state.tool_calls("lookup_order").last.output == '{"order_id": "7", "status": "shipped"}'
    # The second user message went to the agent the handoff left in charge, as the answer to its request.
    assert client.requests[-1][-1] == "user:Thanks"
    assert state.usage is not None
    assert state.usage.total_tokens == 28
