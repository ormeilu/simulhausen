"""``haystack_adapter`` against the real Haystack, with a scripted chat generator component.

Haystack is not a dependency of the project, so the module is skipped in a
plain test run. Run it with the library overlaid:

    uv run --no-group harbor --with haystack-ai pytest tests/contract/test_haystack_contract.py --no-cov -q
"""

from typing import Any

import pytest

pytest.importorskip("haystack")

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

import simulhausen as sim
from simulhausen import ScenarioState
from simulhausen.frameworks.haystack import haystack_adapter

SYSTEM = "You are the order desk."
USAGE = {"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}


@component
class Desk:
    """A scripted chat generator: calls a tool when the user names one, otherwise reports what it was shown."""

    def __init__(self) -> None:
        self.seen: list[list[ChatMessage]] = []

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.seen.append(list(messages))
        last = messages[-1]
        prompt = last.text or ""
        calls: list[ToolCall] = []
        if last.is_from("user"):
            if "order" in prompt:
                calls = [ToolCall("lookup_order", {"order_id": 7}, id="call-1")]
            elif "refund" in prompt:
                # No id: the adapter pairs the result by tool name and arguments.
                calls = [ToolCall("refund", {"order_id": 7})]
        if calls:
            return {"replies": [ChatMessage.from_assistant(tool_calls=calls, meta=USAGE)]}
        users = " | ".join(message.text or "" for message in messages if message.is_from("user"))
        results = " | ".join(str(result.result) for message in messages for result in message.tool_call_results)
        text = f"users: {users}; tool results: {results or 'none'}"
        return {"replies": [ChatMessage.from_assistant(text, meta=USAGE)]}

    def roles(self, call: int = -1) -> list[str]:
        return [message.role.value for message in self.seen[call]]


def lookup_order(order_id: int) -> str:
    return f"order {order_id} is shipped"


def refund(order_id: int) -> str:
    raise RuntimeError("refunds are closed")


PARAMETERS = {"type": "object", "properties": {"order_id": {"type": "integer"}}, "required": ["order_id"]}


def _agent(model: Desk) -> Agent:
    tools = [
        Tool(name="lookup_order", description="Look an order up.", parameters=PARAMETERS, function=lookup_order),
        Tool(name="refund", description="Refund an order.", parameters=PARAMETERS, function=refund),
    ]
    return Agent(chat_generator=model, tools=tools, system_prompt=SYSTEM)


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


async def test_reply_tool_call_memory_and_usage_in_session_mode() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(haystack_adapter(lambda: _agent(model)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "order 7 is shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    # Second turn: the model is shown the first exchange, its tool result included, under one system prompt.
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert model.roles() == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert state.tool_calls(turn=1) == []
    # Three model requests over the two turns, 5 + 2 tokens each.
    assert result.usage == sim.Usage(input_tokens=15, output_tokens=6, total_tokens=21, requests=3)


async def test_a_raising_tool_is_an_error() -> None:
    states: list[ScenarioState] = []

    await _play(haystack_adapter(_agent(Desk())), [sim.user("refund it"), sim.agent(), states.append])

    state = states[0]
    assert not state.has_tool_call("refund")
    assert state.tool_calls() == []
    (failed,) = state.tool_calls(include_errors=True)
    assert (failed.name, failed.input, failed.status) == ("refund", {"order_id": 7}, "error")
    assert failed.error
    assert failed.output is not None
    assert "refunds are closed" in failed.output


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

    result = await _play(haystack_adapter(_agent(model), history=history), ROLLBACK_SCRIPT)

    assert result.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
    assert "secret" not in str(model.seen[-1])
    assert model.roles() == ["system", "user", "assistant", "user"]


async def test_transcript_mode_does_not_replay_tool_results() -> None:
    model = Desk()
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    result = await _play(haystack_adapter(_agent(model), history="transcript"), script)

    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: none"


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    adapter = haystack_adapter(_agent(Desk()))

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"


async def test_a_tool_exit_condition_gives_the_tool_result_as_the_reply() -> None:
    agent = _agent(Desk())
    agent.exit_conditions = ["lookup_order"]

    result = await _play(haystack_adapter(agent), [sim.user("where is my order?"), sim.agent()])

    assert result.messages[-1]["content"] == "order 7 is shipped"


@pytest.mark.parametrize("named", [False, True])
async def test_a_pipeline_with_a_chat_generator(named: bool) -> None:
    model = Desk()
    pipeline = Pipeline()
    pipeline.add_component("llm", model)
    options: dict[str, Any] = {"input_component": "llm", "output_component": "llm", "output_socket": "replies"}
    adapter = haystack_adapter(pipeline, **(options if named else {}))
    script = [sim.user("hello"), sim.agent(), sim.user("more"), sim.agent()]

    result = await _play(adapter, script)

    assert result.messages[1]["content"] == "users: hello; tool results: none"
    # The generator returns only its reply; the adapter carries the conversation.
    assert result.messages[3]["content"] == "users: hello | more; tool results: none"
    assert model.roles() == ["user", "assistant", "user"]
    assert result.usage == sim.Usage(input_tokens=10, output_tokens=4, total_tokens=14, requests=2)


async def test_a_pipeline_with_an_agent_reports_its_tool_calls() -> None:
    model = Desk()
    pipeline = Pipeline()
    pipeline.add_component("agent", _agent(model))
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(haystack_adapter(pipeline, input_component="agent"), script)

    assert states[0].tool_calls("lookup_order").first.output == "order 7 is shipped"
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert model.roles() == ["system", "user", "assistant", "tool", "assistant", "user"]


async def test_a_generator_error_propagates() -> None:
    @component
    class Broken:
        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, Any]:
            raise ValueError("model is down")

    with pytest.raises(ValueError, match="model is down"):
        await _play(haystack_adapter(Agent(chat_generator=Broken())), [sim.user("hi"), sim.agent()])
