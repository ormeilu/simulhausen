"""``camel_adapter`` against the real CAMEL-AI, with a scripted ``StubModel`` in place of a provider.

CAMEL is not a dependency of the project, so the module is skipped in a plain
test run. Run it with the library overlaid:

    uv run --no-group harbor --with camel-ai --with 'mcp<2' pytest tests/contract/test_camel_contract.py --no-cov -q

(camel-ai 0.2.90 imports ``mcp.server.FastMCP``, which mcp 2 no longer has, and does not cap its ``mcp`` requirement.)
"""

import json
from typing import Any

import pytest

pytest.importorskip("camel")

from camel.agents import ChatAgent
from camel.models.stub_model import StubModel
from camel.types import ChatCompletion, ModelType
from pydantic import BaseModel

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.camel import camel_adapter

SYSTEM = "You are the order desk."


def lookup_order(order_id: str) -> str:
    """Looks an order up.

    Args:
        order_id (str): The order number.

    Returns:
        str: The order status.
    """
    return f"order {order_id} is shipped"


def refund(order_id: str) -> str:
    """Refunds an order.

    Args:
        order_id (str): The order number.

    Returns:
        str: The refund status.
    """
    raise ValueError(f"order {order_id} is not refundable")


def export_orders(fmt: str) -> str:
    """Exports the orders; runs outside the agent.

    Args:
        fmt (str): The file format.

    Returns:
        str: The file name.
    """
    raise AssertionError("an external tool is never run by the agent")


class Summary(BaseModel):
    answer: str


class Desk(StubModel):
    """A scripted model: calls a tool when the user names one, otherwise reports what it was shown."""

    def __init__(self) -> None:
        super().__init__(ModelType.STUB)
        self.seen: list[list[dict[str, Any]]] = []

    def _completion(self, content: str | None, call: tuple[str, dict[str, Any]] | None = None) -> ChatCompletion:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if call is not None:
            name, arguments = call
            message["tool_calls"] = [
                {
                    "id": f"call-{len(self.seen)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        return ChatCompletion.model_validate(
            {
                "id": "desk",
                "model": "stub",
                "object": "chat.completion",
                "created": 0,
                "choices": [{"index": 0, "finish_reason": "tool_calls" if call else "stop", "message": message}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    async def _arun(self, messages: Any, response_format: Any = None, tools: Any = None) -> Any:
        self.seen.append([dict(message) for message in messages])
        last = messages[-1]
        text = str(last.get("content") or "")
        if last["role"] == "user":
            if "order" in text:
                return self._completion(None, ("lookup_order", {"order_id": "1042"}))
            if "refund" in text:
                return self._completion(None, ("refund", {"order_id": "1042"}))
            if "export" in text:
                return self._completion(None, ("export_orders", {"fmt": "csv"}))
            if "summary" in text:
                return self._completion('{"answer": "all good"}')
        users = " | ".join(str(m["content"]) for m in messages if m["role"] == "user")
        results = " | ".join(str(m["content"]) for m in messages if m["role"] == "tool") or "none"
        return self._completion(f"users: {users}; tool results: {results}")

    def roles(self, call: int = -1) -> list[str]:
        return [str(message["role"]) for message in self.seen[call]]

    def shown(self, call: int = -1) -> str:
        return "\n".join(str(message.get("content") or "") for message in self.seen[call])


def _agent(model: Desk) -> ChatAgent:
    return ChatAgent(SYSTEM, model=model, tools=[lookup_order, refund], external_tools=[export_orders])


async def _play(adapter: Any, *steps: Any) -> ScenarioState:
    states: list[ScenarioState] = []
    await sim.run("camel", "contract", [adapter], script=[*steps, states.append])
    return states[0]


async def test_reply_tool_call_memory_and_usage() -> None:
    model = Desk()
    state = await _play(
        camel_adapter(_agent(model)), sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()
    )

    assert state.messages[1]["content"] == "users: where is my order?; tool results: order 1042 is shipped"
    # Session mode: the second step sees the first exchange and its tool result.
    assert state.messages[3]["content"] == ("users: where is my order? | thanks; tool results: order 1042 is shipped")
    assert model.roles() == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").last
    assert (call.input, call.output, call.turn) == ({"order_id": "1042"}, "order 1042 is shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    assert state.tool_call_names() == ["lookup_order"]
    # Three model calls: two in the first step, one in the second.
    assert state.usage == Usage(input_tokens=30, output_tokens=15, total_tokens=45)


async def test_a_raising_tool_is_an_error_and_an_external_tool_is_not_executed() -> None:
    state = await _play(
        camel_adapter(lambda: _agent(Desk())),
        sim.user("refund it"),
        sim.agent(),
        sim.user("export everything"),
        sim.agent(),
    )

    assert not state.has_tool_call("refund")
    failed = state.tool_calls("refund", include_errors=True).last
    assert failed.error
    assert failed.input == {"order_id": "1042"}
    assert "order 1042 is not refundable" in (failed.output or "")
    assert "not refundable" in state.messages[1]["content"]
    # The agent hands an external tool call back instead of running it.
    assert not state.has_tool_call("export_orders")
    external = state.tool_calls("export_orders", include_errors=True).last
    assert external.paused
    assert external.input == {"fmt": "csv"}
    assert external.turn == 1


async def test_transcript_mode_forgets_a_rolled_back_exchange() -> None:
    model = Desk()

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(0)

    state = await _play(
        camel_adapter(lambda: _agent(model), history="transcript"),
        sim.user("my code is 7421"),
        sim.agent(),
        backtrack,
        sim.user("hello again"),
        sim.agent(),
    )

    assert "7421" in model.shown(0)
    assert "7421" not in model.shown()
    assert state.messages[-1]["content"] == "users: hello again; tool results: none"


async def test_transcript_mode_replays_the_dialogue_without_tool_results() -> None:
    model = Desk()
    state = await _play(
        camel_adapter(lambda: _agent(model), history="transcript"),
        sim.user("where is my order?"),
        sim.agent(),
        sim.user("thanks"),
        sim.agent(),
    )

    # The system message survives the reset; the first exchange comes back as plain messages.
    assert model.roles() == ["system", "user", "assistant", "user"]
    assert model.shown().startswith(SYSTEM)
    assert state.messages[3]["content"] == "users: where is my order? | thanks; tool results: none"
    assert state.tool_call_names() == ["lookup_order"]


async def test_session_mode_rebuilds_the_memory_after_a_rollback() -> None:
    model = Desk()

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(0)

    await _play(
        camel_adapter(lambda: _agent(model)),
        sim.user("my code is 7421"),
        sim.agent(),
        backtrack,
        sim.user("hello again"),
        sim.agent(),
    )

    assert "7421" not in model.shown()


async def test_a_factory_gives_every_scenario_its_own_agent_and_a_shared_agent_starts_over() -> None:
    adapter = camel_adapter(lambda: _agent(Desk()))
    first = await _play(adapter, sim.user("first secret"), sim.agent(), sim.user("more"), sim.agent())
    second = await _play(adapter, sim.user("second"), sim.agent())
    assert first.messages[-1]["content"] == "users: first secret | more; tool results: none"
    assert second.messages[-1]["content"] == "users: second; tool results: none"

    shared = camel_adapter(_agent(Desk()))
    await _play(shared, sim.user("first secret"), sim.agent())
    again = await _play(shared, sim.user("second"), sim.agent())
    assert again.messages[-1]["content"] == "users: second; tool results: none"


async def test_step_kwargs_reach_astep_and_a_parsed_reply_is_rendered() -> None:
    seen: list[Any] = []

    def render(raw: Any) -> str:
        seen.append(raw)
        return f"answer={raw.answer}"

    adapter = camel_adapter(lambda: ChatAgent(SYSTEM, model=Desk()), content_fn=render, response_format=Summary)
    state = await _play(adapter, sim.user("a summary please"), sim.agent())

    assert state.messages[-1]["content"] == "answer=all good"
    assert seen == [Summary(answer="all good")]
