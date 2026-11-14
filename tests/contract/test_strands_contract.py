"""Contract test of ``strands_adapter`` against the real Strands Agents, with a scripted model and no network.

Run it with the framework overlaid on the project environment::

    uv run --no-group harbor --with strands-agents pytest tests/contract/test_strands_contract.py --no-cov -q
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("strands")

from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent
from strands.models import Model
from strands.multiagent import GraphBuilder, Swarm

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.strands import strands_adapter

pytestmark = pytest.mark.filterwarnings("ignore")


def _flat(message: dict[str, Any]) -> str:
    chunks: list[str] = []
    for block in message["content"]:
        if "text" in block:
            chunks.append(block["text"])
        elif "toolUse" in block:
            chunks.append(f"call:{block['toolUse']['name']}")
        elif "toolResult" in block:
            chunks.append(f"result:{block['toolResult']['status']}")
    return f"{message['role']}:{'|'.join(chunks)}"


class ScriptedModel(Model):
    """Streams the next scripted answer (a text or a ``(tool, args)`` pair) and keeps every request's messages."""

    def __init__(self, *script: str | tuple[str, dict[str, Any]]) -> None:
        self.script = list(script)
        self.requests: list[list[str]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError
        yield

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any) -> Any:
        self.requests.append([_flat(message) for message in messages])
        item = self.script.pop(0)
        yield {"messageStart": {"role": "assistant"}}
        if isinstance(item, str):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": item}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            name, args = item
            call_id = f"call-{len(self.requests)}"
            yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": call_id}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        usage = {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


@tool
def lookup_order(order_id: str) -> dict[str, str]:
    """Look up an order."""
    return {"order_id": order_id, "status": "shipped"}


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    raise RuntimeError(f"order {order_id} is locked")


def support_agent(model: ScriptedModel, **kwargs: Any) -> Agent:
    return Agent(model=model, tools=[lookup_order, cancel_order], callback_handler=None, **kwargs)


async def play(adapter: Any, *script: Any) -> sim.ScenarioResult:
    return await sim.run(name="contract", description="strands contract", agents=[adapter], script=list(script))


async def test_reply_tool_call_usage_and_session_memory() -> None:
    model = ScriptedModel(("lookup_order", {"order_id": "7"}), "Order 7 has shipped.", "You asked about order 7.")
    seen: list[ScenarioState] = []

    result = await play(
        strands_adapter(lambda: support_agent(model)),
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
    # The second turn's request carries the first exchange, the tool use and its result included.
    assert model.requests[-1] == [
        "user:Where is order 7?",
        "assistant:call:lookup_order",
        "user:result:success",
        "assistant:Order 7 has shipped.",
        "user:What did I ask?",
    ]
    assert state.usage == Usage(input_tokens=15, output_tokens=6, total_tokens=21, requests=3)


async def test_raising_tool_is_recorded_as_an_error() -> None:
    model = ScriptedModel(("cancel_order", {"order_id": "7"}), "I could not cancel it.")
    seen: list[ScenarioState] = []

    await play(strands_adapter(support_agent(model)), sim.user("Cancel order 7"), sim.agent(), seen.append)

    state = seen[0]
    assert not state.has_tool_call("cancel_order")
    failed = state.tool_calls("cancel_order", include_errors=True).last
    assert failed.status == "error"
    assert failed.input == {"order_id": "7"}
    assert failed.output == "Error: RuntimeError - order 7 is locked"
    assert failed.execution is not None
    assert failed.execution.tool_call_error is True


async def test_interrupted_tool_call_is_not_executed() -> None:
    model = ScriptedModel(("lookup_order", {"order_id": "7"}))
    agent = support_agent(model)

    def approve(event: BeforeToolCallEvent) -> None:
        event.interrupt("approval", reason="needs a human")

    agent.add_hook(approve, BeforeToolCallEvent)
    seen: list[ScenarioState] = []

    await play(strands_adapter(agent), sim.user("Where is order 7?"), sim.agent(), seen.append)

    state = seen[0]
    assert not state.has_tool_call("lookup_order")
    assert [(c.name, c.status) for c in state.tool_calls(include_errors=True)] == [("lookup_order", "paused")]


async def test_transcript_history_forgets_a_rolled_back_exchange() -> None:
    model = ScriptedModel("one", "two", "three")

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(2)

    await play(
        strands_adapter(support_agent(model), history="transcript"),
        sim.user("first"),
        sim.agent(),
        sim.user("secret"),
        sim.agent(),
        backtrack,
        sim.user("third"),
        sim.agent(),
    )

    assert model.requests[-1] == ["user:first", "assistant:one", "user:third"]


async def test_session_history_is_rebuilt_from_the_transcript_after_a_rollback() -> None:
    model = ScriptedModel("one", "two", "three")
    preset: Any = [
        {"role": "user", "content": [{"text": "example"}]},
        {"role": "assistant", "content": [{"text": "ok"}]},
    ]

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(2)

    await play(
        strands_adapter(support_agent(model, messages=preset)),
        sim.user("first"),
        sim.agent(),
        sim.user("secret"),
        sim.agent(),
        backtrack,
        sim.user("third"),
        sim.agent(),
    )

    assert model.requests[1] == ["user:example", "assistant:ok", "user:first", "assistant:one", "user:secret"]
    assert model.requests[-1] == ["user:example", "assistant:ok", "user:first", "assistant:one", "user:third"]


async def test_one_agent_object_serves_scenarios_in_a_row_without_shared_memory() -> None:
    model = ScriptedModel("one", "two")
    adapter = strands_adapter(support_agent(model))

    await play(adapter, sim.user("first scenario"), sim.agent())
    await play(adapter, sim.user("second scenario"), sim.agent())

    assert model.requests[-1] == ["user:second scenario"]


async def test_factory_builds_an_agent_per_scenario() -> None:
    model = ScriptedModel("one", "two")
    built: list[Agent] = []

    def factory() -> Agent:
        built.append(support_agent(model))
        return built[-1]

    adapter = strands_adapter(factory)
    await play(adapter, sim.user("first scenario"), sim.agent())
    await play(adapter, sim.user("second scenario"), sim.agent())

    assert len(built) == 2
    assert [len(agent.messages) for agent in built] == [2, 2]


async def test_swarm_handoff_tools_and_last_node_reply() -> None:
    model = ScriptedModel(
        ("handoff_to_agent", {"agent_name": "orders", "message": "order question"}),
        "Handing over.",
        ("lookup_order", {"order_id": "7"}),
        "Order 7 has shipped.",
        "Still shipped.",
    )
    front = Agent(name="front", model=model, callback_handler=None)
    orders = Agent(name="orders", model=model, tools=[lookup_order], callback_handler=None)
    seen: list[ScenarioState] = []

    result = await play(
        strands_adapter(Swarm([front, orders], entry_point=front)),
        sim.user("Where is order 7?"),
        sim.agent(),
        sim.user("And now?"),
        sim.agent(),
        seen.append,
    )

    state = seen[0]
    assert [m["content"] for m in result.messages if m["role"] == "assistant"] == [
        "Order 7 has shipped.",
        "Still shipped.",
    ]
    assert state.tool_call_names() == ["handoff_to_agent", "lookup_order"]
    assert state.tool_calls("lookup_order").last.turn == 0
    # The swarm has no memory between runs: the second task is the whole transcript.
    assert "User: Where is order 7?\nAssistant: Order 7 has shipped.\nUser: And now?" in model.requests[-1][0]
    assert state.usage is not None
    assert state.usage.total_tokens == 35


async def test_graph_reply_is_the_last_node() -> None:
    model = ScriptedModel("draft", "polished")
    builder = GraphBuilder()
    builder.add_node(Agent(name="writer", model=model, callback_handler=None), "writer")
    builder.add_node(Agent(name="editor", model=model, callback_handler=None), "editor")
    builder.add_edge("writer", "editor")
    builder.set_entry_point("writer")

    result = await play(strands_adapter(builder.build()), sim.user("Write it"), sim.agent())

    assert result.messages[-1]["content"] == "polished"
