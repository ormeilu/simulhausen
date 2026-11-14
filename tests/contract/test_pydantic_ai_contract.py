"""``pydantic_ai_adapter`` against the real Pydantic AI, with a ``FunctionModel`` in place of a provider.

Pydantic AI is not a dependency of the project, so the module is skipped in a
plain test run. Run it with the library overlaid:

    uv run --no-group harbor --with pydantic-ai-slim pytest tests/contract/test_pydantic_ai_contract.py --no-cov -q
"""

from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from pydantic import BaseModel
from pydantic_ai import Agent, DeferredToolRequests, ModelRetry
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import simulhausen as sim
from simulhausen import ScenarioState
from simulhausen.frameworks.pydantic_ai import pydantic_ai_adapter

SYSTEM = "You are the order desk."


def _texts(messages: list[ModelMessage], kind: str) -> list[str]:
    return [str(part.content) for message in messages for part in message.parts if part.part_kind == kind]


class Desk:
    """A scripted model: calls a tool when the user names one, otherwise reports what it was shown."""

    def __init__(self) -> None:
        self.seen: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.seen.append(list(messages))
        last = messages[-1]
        prompts = [str(part.content) for part in last.parts if part.part_kind == "user-prompt"]
        prompt = prompts[-1] if prompts else ""
        if "order" in prompt:
            return ModelResponse(parts=[ToolCallPart("lookup_order", {"order_id": 7}, tool_call_id="call-1")])
        if "refund" in prompt:
            return ModelResponse(
                parts=[
                    ToolCallPart("refund", '{"order_id": 7}', tool_call_id="call-2"),
                    ToolCallPart("check_stock", {"sku": "A"}, tool_call_id="call-3"),
                ]
            )
        if "delete" in prompt:
            return ModelResponse(parts=[ToolCallPart("delete_account", {"user": "u1"}, tool_call_id="call-4")])
        if "summary" in prompt:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"answer": "all good"})])
        users = " | ".join(_texts(messages, "user-prompt"))
        returns = " | ".join(_texts(messages, "tool-return")) or "none"
        return ModelResponse(parts=[TextPart(f"users: {users}; tool results: {returns}")])

    def system_prompts(self, call: int = -1) -> list[str]:
        return _texts(self.seen[call], "system-prompt")


class Summary(BaseModel):
    answer: str


def _agent(model: Desk) -> Agent[None, str]:
    agent = Agent(FunctionModel(model), system_prompt=SYSTEM)

    @agent.tool_plain
    def lookup_order(order_id: int) -> dict[str, Any]:
        return {"order_id": order_id, "status": "shipped"}

    @agent.tool_plain
    def refund(order_id: int) -> str:
        raise ToolFailed("refunds are closed")

    @agent.tool_plain
    def check_stock(sku: str) -> str:
        raise ModelRetry("unknown sku")

    @agent.tool_plain(requires_approval=True)
    def delete_account(user: str) -> str:
        raise AssertionError("must not run without approval")

    return agent


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


async def test_reply_tool_call_memory_and_usage_in_session_mode() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(pydantic_ai_adapter(lambda: _agent(model)), script)

    state = states[0]
    shipped = str({"order_id": 7, "status": "shipped"})
    assert result.messages[1]["content"] == f"users: where is my order?; tool results: {shipped}"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert call.input == {"order_id": 7}
    assert call.output == '{"order_id": 7, "status": "shipped"}'
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    assert call.turn == 0
    # Second turn: the model is shown the first exchange, its tool result included, and no new tool call is recorded.
    assert result.messages[3]["content"] == f"users: where is my order? | thanks; tool results: {shipped}"
    assert state.tool_calls(turn=1) == []
    assert model.system_prompts() == [SYSTEM]
    # FunctionModel estimates tokens; three model requests were made over the two turns.
    assert result.usage is not None
    assert result.usage.requests == 3
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


async def test_failed_and_retried_tools_are_errors() -> None:
    states: list[ScenarioState] = []

    await _play(pydantic_ai_adapter(_agent(Desk())), [sim.user("refund it"), sim.agent(), states.append])

    state = states[0]
    assert not state.has_tool_call("refund")
    assert state.tool_calls() == []
    failed, retried = state.tool_calls(include_errors=True)
    assert (failed.name, failed.input, failed.output, failed.status) == (
        "refund",
        {"order_id": 7},
        "refunds are closed",
        "error",
    )
    assert failed.error
    assert (retried.name, retried.output, retried.status) == ("check_stock", "unknown sku", "error")


async def test_a_plain_exception_of_a_tool_propagates() -> None:
    def calls_broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("broken", {})])

    agent = Agent(FunctionModel(calls_broken))

    @agent.tool_plain
    def broken() -> str:
        raise RuntimeError("database is down")

    with pytest.raises(RuntimeError, match="database is down"):
        await _play(pydantic_ai_adapter(agent), [sim.user("hi"), sim.agent()])


async def test_a_call_waiting_for_approval_is_not_executed() -> None:
    states: list[ScenarioState] = []
    adapter = pydantic_ai_adapter(_agent(Desk()), output_type=[str, DeferredToolRequests])

    await _play(adapter, [sim.user("delete my account"), sim.agent(), states.append])

    state = states[0]
    assert not state.has_tool_call("delete_account")
    (pending,) = state.tool_calls(include_errors=True)
    assert (pending.name, pending.input, pending.status) == ("delete_account", {"user": "u1"}, "paused")


async def test_structured_output_is_rendered_and_its_output_tool_is_hidden() -> None:
    states: list[ScenarioState] = []
    plain = pydantic_ai_adapter(_agent(Desk()), output_type=Summary)
    custom = pydantic_ai_adapter(_agent(Desk()), output_type=Summary, content_fn=lambda output: output.answer)

    first = await _play(plain, [sim.user("summary please"), sim.agent(), states.append])
    second = await _play(custom, [sim.user("summary please"), sim.agent()])

    assert first.messages[-1]["content"] == '{"answer":"all good"}'
    assert second.messages[-1]["content"] == "all good"
    assert states[0].tool_calls(include_errors=True) == []


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

    result = await _play(pydantic_ai_adapter(_agent(model), history=history), ROLLBACK_SCRIPT)

    assert result.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
    assert "secret" not in str(model.seen[-1])
    # The rebuilt history still opens with the agent's system prompt.
    assert model.system_prompts() == [SYSTEM]


async def test_transcript_mode_does_not_replay_tool_results() -> None:
    model = Desk()
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    result = await _play(pydantic_ai_adapter(_agent(model), history="transcript"), script)

    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: none"


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    model = Desk()
    adapter = pydantic_ai_adapter(_agent(model))

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"
