"""``autogen_adapter`` against the real AutoGen AgentChat, with a scripted ``ChatCompletionClient``.

AutoGen is not a dependency of the project, so the module is skipped in a
plain test run. Run it with the library overlaid:

    uv run --no-group harbor --with autogen-agentchat pytest tests/contract/test_autogen_contract.py --no-cov -q
"""

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pytest

pytest.importorskip("autogen_agentchat")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, Swarm
from autogen_core import FunctionCall
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
)

import simulhausen as sim
from simulhausen import ScenarioState
from simulhausen.frameworks.autogen import autogen_adapter

SYSTEM = "You are the order desk."
USAGE = RequestUsage(prompt_tokens=5, completion_tokens=2)


def _dump(messages: Sequence[LLMMessage]) -> list[tuple[str, str]]:
    return [(type(message).__name__, str(message.content)) for message in messages]


class Desk(ChatCompletionClient):
    """A scripted model: calls a tool when the user names one, otherwise reports what it was shown."""

    def __init__(self, handoff: str | None = None) -> None:
        self.seen: list[list[tuple[str, str]]] = []
        self._handoff = handoff

    async def create(self, messages: Sequence[LLMMessage], **kwargs: Any) -> CreateResult:
        self.seen.append(_dump(messages))
        last = messages[-1]
        asked = [m.content for m in messages if type(m).__name__ == "UserMessage" and m.source == "user"]
        prompt = str(asked[-1]) if asked else ""
        calls: list[FunctionCall] = []
        if type(last).__name__ == "UserMessage":
            if self._handoff:
                calls = [FunctionCall(id="call-0", name=f"transfer_to_{self._handoff}", arguments="{}")]
            elif "order" in prompt:
                calls = [FunctionCall(id="call-1", name="lookup_order", arguments='{"order_id": 7}')]
            elif "refund" in prompt:
                calls = [FunctionCall(id="call-2", name="refund", arguments='{"order_id": 7}')]
        if calls:
            return CreateResult(finish_reason="function_calls", content=calls, usage=USAGE, cached=False)
        users = " | ".join(str(text) for text in asked)
        results = " | ".join(
            result.content
            for message in messages
            if type(message).__name__ == "FunctionExecutionResultMessage"
            for result in message.content
        )
        text = f"users: {users}; tool results: {results or 'none'}"
        return CreateResult(finish_reason="stop", content=text, usage=USAGE, cached=False)

    def create_stream(self, messages: Sequence[LLMMessage], **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return USAGE

    def total_usage(self) -> RequestUsage:
        return USAGE

    def count_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return 1000

    @property
    def capabilities(self) -> Any:
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False, function_calling=True, json_output=False, family=ModelFamily.UNKNOWN, structured_output=False
        )


def lookup_order(order_id: int) -> str:
    """Look an order up."""
    return f"order {order_id} is shipped"


def refund(order_id: int) -> str:
    """Refund an order."""
    raise RuntimeError("refunds are closed")


def _agent(model: Desk) -> AssistantAgent:
    return AssistantAgent(
        "desk", model_client=model, tools=[lookup_order, refund], reflect_on_tool_use=True, system_message=SYSTEM
    )


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


async def test_reply_tool_call_memory_and_usage_in_session_mode() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(autogen_adapter(lambda: _agent(model)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "order 7 is shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    # Second turn: the model is shown the first exchange, its tool result included, and no new tool call is recorded.
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert state.tool_calls(turn=1) == []
    # Three model requests over the two turns, 5 + 2 tokens each.
    assert result.usage == sim.Usage(input_tokens=15, output_tokens=6, total_tokens=21, requests=3)


async def test_a_raising_tool_is_an_error() -> None:
    states: list[ScenarioState] = []

    await _play(autogen_adapter(_agent(Desk())), [sim.user("refund it"), sim.agent(), states.append])

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

    result = await _play(autogen_adapter(_agent(model), history=history), ROLLBACK_SCRIPT)

    assert result.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
    # The replayed transcript keeps the roles: the earlier reply is an assistant message of the model context.
    assert model.seen[-1] == [
        ("SystemMessage", SYSTEM),
        ("UserMessage", "hello"),
        ("AssistantMessage", "users: hello; tool results: none"),
        ("UserMessage", "what do you know?"),
    ]


async def test_transcript_mode_does_not_replay_tool_results() -> None:
    model = Desk()
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    result = await _play(autogen_adapter(_agent(model), history="transcript"), script)

    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: none"


async def test_a_factory_gives_every_scenario_its_own_agent() -> None:
    model = Desk()
    built: list[AssistantAgent] = []

    def factory() -> AssistantAgent:
        built.append(_agent(model))
        return built[-1]

    adapter = autogen_adapter(factory)

    await _play(adapter, [sim.user("my secret is 42"), sim.agent(), sim.user("again"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert len(built) == 2
    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"


async def test_one_instance_serves_scenarios_without_sharing_memory() -> None:
    model = Desk()
    adapter = autogen_adapter(_agent(model))

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"


async def test_one_instance_switches_between_interleaved_threads() -> None:
    model = Desk()
    adapter = autogen_adapter(_agent(model))

    def turn(thread_id: str, *texts: str) -> sim.AgentInput:
        messages: Any = []
        for index, text in enumerate(texts):
            messages.append({"role": "assistant" if index % 2 else "user", "content": text})
        return sim.AgentInput(thread_id=thread_id, messages=messages, new_messages=messages[-1:])

    first = await adapter.call(turn("a", "where is my order?"))
    other = await adapter.call(turn("b", "hello"))
    assert isinstance(first, sim.AgentTurn)
    assert isinstance(other, sim.AgentTurn)
    assert other.content == "users: hello; tool results: none"
    # Thread "a" comes back through load_state with its tool result still in the context.
    back = await adapter.call(turn("a", "where is my order?", first.content, "thanks"))
    assert isinstance(back, sim.AgentTurn)
    assert back.content == "users: where is my order? | thanks; tool results: order 7 is shipped"
    again = await adapter.call(turn("b", "hello", other.content, "more"))
    assert isinstance(again, sim.AgentTurn)
    assert again.content == "users: hello | more; tool results: none"


def _swarm(front: Desk, back: Desk) -> Swarm:
    triage = AssistantAgent("triage", model_client=front, handoffs=["desk"])
    desk = _agent(back)
    return Swarm([triage, desk], termination_condition=TextMentionTermination("users:", sources=["desk"]))


async def test_a_swarm_reports_the_handoff_and_the_tools_of_its_target() -> None:
    front, back = Desk(handoff="desk"), Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(autogen_adapter(lambda: _swarm(front, back)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    assert state.tool_call_names() == ["transfer_to_desk", "lookup_order"]
    assert state.tool_calls("lookup_order").first.output == "order 7 is shipped"
    # The swarm stays with the agent it handed off to, which still has the first exchange.
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert len(front.seen) == 1
    assert result.usage is not None
    assert result.usage.requests == 4


@pytest.mark.parametrize("history", ["transcript", "session"])
async def test_a_team_gets_the_transcript_after_a_rollback(history: Any) -> None:
    model = Desk()

    def team() -> RoundRobinGroupChat:
        return RoundRobinGroupChat(
            [_agent(model)], termination_condition=TextMentionTermination("users:", sources=["desk"])
        )

    result = await _play(autogen_adapter(team, history=history), ROLLBACK_SCRIPT)

    assert "secret" not in result.messages[-1]["content"]
    assert [kind for kind, _ in model.seen[-1]] == ["SystemMessage", "UserMessage", "UserMessage", "UserMessage"]
    assert [text for _, text in model.seen[-1]][1:] == [
        "hello",
        "users: hello; tool results: none",
        "what do you know?",
    ]


async def test_a_team_error_propagates() -> None:
    class Broken(Desk):
        async def create(self, messages: Sequence[LLMMessage], **kwargs: Any) -> CreateResult:
            raise ValueError("model is down")

    with pytest.raises(ValueError, match="model is down"):
        await _play(autogen_adapter(_agent(Broken())), [sim.user("hi"), sim.agent()])
