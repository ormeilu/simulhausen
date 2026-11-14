"""``langchain_adapter`` against a real Deep Agent (``deepagents.create_deep_agent``), with a scripted chat model.

A Deep Agent is a compiled LangGraph graph, so it goes through the same adapter
as every other graph; this module checks what is specific to it: a built-in
file tool, and the tools of a sub-agent started through ``task``, whose
messages never enter the main agent's state. ``deepagents`` is not a dependency
of the project, so the module is skipped in a plain test run. Run it with the
library overlaid:

    uv run --no-group harbor --with deepagents pytest tests/contract/test_deepagents_contract.py --no-cov -q
"""

from typing import Any

import pytest

pytest.importorskip("deepagents")

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.langchain import langchain_adapter

USAGE = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}], usage_metadata=USAGE)


class Desk:
    """The main agent's scripted model: picks a tool by keyword, otherwise reports what it was shown."""

    def __init__(self) -> None:
        self.seen: list[list[BaseMessage]] = []

    def __call__(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        last = messages[-1]
        prompt = str(last.content) if last.type == "human" else ""
        if "order" in prompt:
            return _call("lookup_order", {"order_id": 7}, "call-1")
        if "plan" in prompt:
            return _call("write_file", {"file_path": "/plan.md", "content": "ship it"}, "call-2")
        if "warehouse" in prompt:
            return _call("task", {"description": "count the stock", "subagent_type": "warehouse"}, "call-3")
        users = " | ".join(str(m.content) for m in messages if m.type == "human")
        results = " | ".join(str(m.content) for m in messages if m.type == "tool") or "none"
        return AIMessage(content=f"users: {users}; tool results: {results}", usage_metadata=USAGE)


def _warehouse(messages: list[BaseMessage]) -> AIMessage:
    if messages[-1].type == "tool":
        return AIMessage(content=f"warehouse says {messages[-1].content}", usage_metadata=USAGE)
    return _call("count_stock", {"sku": "A"}, "call-4")


class Scripted(BaseChatModel):
    """A chat model that answers through a plain function and ignores the tools bound to it."""

    script: Any

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        return ChatResult(generations=[ChatGeneration(message=self.script(messages))])


@tool
def lookup_order(order_id: int) -> str:
    """Look an order up."""
    return f"order {order_id} is shipped"


@tool
def count_stock(sku: str) -> str:
    """Count the stock of an article."""
    return f"12 of {sku}"


def _deep_agent(model: Desk, *, checkpointer: bool = True) -> Any:
    return create_deep_agent(
        model=Scripted(script=model),
        tools=[lookup_order],
        system_prompt="You are the order desk.",
        subagents=[
            {
                "name": "warehouse",
                "description": "Counts stock.",
                "system_prompt": "You count stock.",
                "tools": [count_stock],
                "model": Scripted(script=_warehouse),
            }
        ],
        checkpointer=InMemorySaver() if checkpointer else None,
    )


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


@pytest.mark.parametrize("checkpointer", [True, False])
async def test_reply_tool_call_memory_and_usage_in_session_mode(checkpointer: bool) -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(langchain_adapter(lambda: _deep_agent(model, checkpointer=checkpointer)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "order 7 is shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    # The deep agent puts its own system prompt first; the rest is the dialogue with the tool exchange.
    assert [m.type for m in model.seen[-1]] == ["system", "human", "ai", "tool", "ai", "human"]
    assert state.tool_calls(turn=1) == []
    assert result.usage == Usage(input_tokens=30, output_tokens=12, total_tokens=42, requests=3)


async def test_a_built_in_tool_is_recorded() -> None:
    states: list[ScenarioState] = []

    await _play(langchain_adapter(_deep_agent(Desk())), [sim.user("make a plan"), sim.agent(), states.append])

    written = states[0].tool_calls("write_file").first
    assert written.input == {"file_path": "/plan.md", "content": "ship it"}
    assert written.status == "success"
    assert "/plan.md" in (written.output or "")


async def test_the_tools_of_a_sub_agent_are_recorded_in_call_order() -> None:
    states: list[ScenarioState] = []

    result = await _play(
        langchain_adapter(_deep_agent(Desk())), [sim.user("ask the warehouse"), sim.agent(), states.append]
    )

    state = states[0]
    assert result.messages[-1]["content"] == "users: ask the warehouse; tool results: warehouse says 12 of A"
    # count_stock ran inside the sub-agent; only the task call and its result are in the main agent's state.
    assert state.tool_call_names() == ["task", "count_stock"]
    assert state.tool_calls("task").first.output == "warehouse says 12 of A"
    assert state.tool_calls("count_stock").first.input == {"sku": "A"}
    # Two calls of the main model and two of the sub-agent's.
    assert result.usage == Usage(input_tokens=40, output_tokens=16, total_tokens=56, requests=4)


def _forget_the_secret(state: ScenarioState) -> None:
    state.rollback_messages_to(2)


@pytest.mark.parametrize("history", ["transcript", "session"])
async def test_a_rollback_reaches_the_agent(history: Any) -> None:
    model = Desk()
    script = [
        sim.user("hello"),
        sim.agent(),
        sim.user("my secret is 42"),
        sim.agent(),
        _forget_the_secret,
        sim.user("what do you know?"),
        sim.agent(),
    ]

    result = await _play(langchain_adapter(_deep_agent(model), history=history), script)

    assert result.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
    assert "secret" not in str(model.seen[-1])


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    adapter = langchain_adapter(_deep_agent(Desk()))

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"
