"""``ag2_adapter`` against the real AG2 1.x, with a scripted ``LLMClient``.

AG2 is not a dependency of the project, so the module is skipped in a plain
test run. Run it with the library overlaid:

    uv run --no-group harbor --with ag2 pytest tests/contract/test_ag2_contract.py --no-cov -q
"""

import json
from collections.abc import Sequence
from typing import Any

import pytest

pytest.importorskip("ag2")

from ag2 import Agent, tool
from ag2.events import ModelMessage, ModelRequest, ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultEvent
from ag2.events.types import Usage
from ag2.testing import TestConfig

import simulhausen as sim
from simulhausen import ScenarioState
from simulhausen.frameworks.ag2 import ag2_adapter


def _text(parts: Sequence[Any]) -> str:
    return "".join(str(getattr(part, "content", "")) for part in parts)


class Desk:
    """A scripted model: calls a tool when the user names one, otherwise reports what it was shown."""

    def __init__(self, delegate: str | None = None) -> None:
        self.seen: list[list[str]] = []
        self._delegate = delegate

    async def __call__(self, messages: Sequence[Any], context: Any, **kwargs: Any) -> ModelResponse:
        self.seen.append([type(message).__name__ for message in messages])
        users = [_text(message.parts) for message in messages if isinstance(message, ModelRequest)]
        results = [_text(message.result.parts) for message in messages if isinstance(message, ToolResultEvent)]
        usage = Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        call: ToolCallEvent | None = None
        if isinstance(messages[-1], ModelRequest):
            if self._delegate:
                call = ToolCallEvent(self._delegate, arguments=json.dumps({"objective": users[-1]}))
            elif "order" in users[-1]:
                call = ToolCallEvent("lookup_order", arguments=json.dumps({"order_id": 7}))
            elif "refund" in users[-1]:
                call = ToolCallEvent("refund", arguments=json.dumps({"order_id": 7}))
        if call is not None:
            return ModelResponse(tool_calls=ToolCallsEvent([call]), usage=usage)
        text = f"users: {' | '.join(users)}; tool results: {' | '.join(results) or 'none'}"
        message = ModelMessage(text)
        await context.send(message)
        return ModelResponse(message, usage=usage)


class DeskConfig(TestConfig):
    """Hands every turn the same scripted client (``TestConfig`` builds a new one per turn)."""

    def __init__(self, client: Desk) -> None:
        super().__init__()
        self._client = client

    def create(self) -> Any:
        return self._client


@tool
def lookup_order(order_id: int) -> str:
    """Look an order up."""
    return f"order {order_id} is shipped"


@tool
def refund(order_id: int) -> str:
    """Refund an order."""
    raise RuntimeError("refunds are closed")


def _agent(model: Desk, name: str = "desk") -> Agent:
    return Agent(name, config=DeskConfig(model), tools=[lookup_order, refund])


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


async def test_reply_tool_call_memory_and_usage_in_session_mode() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(ag2_adapter(lambda: _agent(model)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "order 7 is shipped", 0)
    # Second turn: the model is shown the first exchange, its tool result included, and no new tool call is recorded.
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert state.tool_calls(turn=1) == []
    # Three model requests over the two turns, 5 + 2 tokens each.
    assert result.usage == sim.Usage(input_tokens=15, output_tokens=6, total_tokens=21, requests=3)


async def test_a_raising_tool_is_an_error() -> None:
    states: list[ScenarioState] = []

    await _play(ag2_adapter(_agent(Desk())), [sim.user("refund it"), sim.agent(), states.append])

    state = states[0]
    assert not state.has_tool_call("refund")
    assert state.tool_calls() == []
    (failed,) = state.tool_calls(include_errors=True)
    assert (failed.name, failed.input, failed.output, failed.status) == (
        "refund",
        {"order_id": 7},
        "refunds are closed",
        "error",
    )
    assert failed.error


def _forget_the_secret(state: ScenarioState) -> None:
    state.rollback_messages_to(2)


ROLLBACK_SCRIPT = [
    sim.user("hello"),
    sim.agent(),
    sim.user("my secret is 42"),
    sim.agent(),
    _forget_the_secret,
    sim.user("what do you know?"),
    sim.agent(),
]


@pytest.mark.parametrize("history", ["transcript", "session"])
async def test_a_rollback_reaches_the_agent(history: Any) -> None:
    model = Desk()

    result = await _play(ag2_adapter(_agent(model), history=history), ROLLBACK_SCRIPT)

    assert result.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
    # The replayed transcript keeps the roles: the earlier reply is a model response.
    assert model.seen[-1] == ["ModelRequest", "ModelResponse", "ModelRequest"]


async def test_transcript_mode_does_not_replay_tool_results() -> None:
    model = Desk()
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    result = await _play(ag2_adapter(_agent(model), history="transcript"), script)

    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: none"


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    adapter = ag2_adapter(_agent(Desk()))

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"


async def test_an_agent_used_as_a_tool() -> None:
    worker = _agent(Desk(), name="worker")
    delegate = worker.as_tool(description="Handles orders.")
    front = Agent("front", config=DeskConfig(Desk(delegate=delegate.name)), tools=[delegate])
    states: list[ScenarioState] = []

    result = await _play(ag2_adapter(front), [sim.user("where is my order?"), sim.agent(), states.append])

    state = states[0]
    delegation = state.tool_calls(delegate.name).first
    assert delegation.input == {"objective": "where is my order?"}
    assert delegation.output == "users: where is my order?; tool results: order 7 is shipped"
    # The worker runs on a stream of its own: its lookup_order call is not among the front agent's events,
    # while its two model calls are counted in the usage.
    assert state.tool_call_names() == [delegate.name]
    assert result.usage is not None
    assert result.usage.requests == 4
