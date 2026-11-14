"""``smolagents_adapter`` against the real smolagents, with a scripted ``Model`` in place of a provider.

smolagents is not a dependency of the project, so the module is skipped in a
plain test run. Run it with the library overlaid:

    uv run --no-group harbor --with smolagents pytest tests/contract/test_smolagents_contract.py --no-cov -q
"""

from typing import Any

import pytest

pytest.importorskip("smolagents")

from smolagents import CodeAgent, Model, RunResult, ToolCallingAgent, tool
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole
from smolagents.monitoring import LogLevel, TokenUsage

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.smolagents import smolagents_adapter


@tool
def lookup_order(order_id: str) -> str:
    """Looks an order up.

    Args:
        order_id: The order number.
    """
    return f"order {order_id} is shipped"


@tool
def refund(order_id: str) -> str:
    """Refunds an order.

    Args:
        order_id: The order number.
    """
    raise ValueError(f"order {order_id} is not refundable")


def _text(message: ChatMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content or [])


class Desk(Model):
    """A scripted model: acts on the newest task, then answers with what it was shown."""

    def __init__(self, *, code: bool = False) -> None:
        super().__init__(model_id="desk")
        self.code = code
        self.seen: list[list[tuple[str, str]]] = []

    def _act(self, name: str, arguments: dict[str, Any], snippet: str) -> ChatMessage:
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        if self.code:
            return ChatMessage(role=MessageRole.ASSISTANT, content=f"<code>\n{snippet}\n</code>", token_usage=usage)
        call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(name=name, arguments=arguments),
            id=f"call-{len(self.seen)}",
            type="function",
        )
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call], token_usage=usage)

    def _answer(self, text: str) -> ChatMessage:
        return self._act("final_answer", {"answer": text}, f"final_answer({text!r})")

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        shown = [(str(MessageRole(message.role).value), _text(message)) for message in messages]
        self.seen.append(shown)
        role, last = shown[-1]
        tasks = " | ".join(text.removeprefix("New task:\n") for kind, text in shown if kind == "user")
        if role == "tool-response":
            return self._answer(f"tasks: {tasks}; result: {' '.join(last.split())}")
        if "order" in last:
            return self._act("lookup_order", {"order_id": "1042"}, "print(lookup_order('1042'))")
        if "refund" in last:
            return self._act("refund", {"order_id": "1042"}, "print(refund(order_id='1042'))")
        if "compute" in last:
            return self._act("final_answer", {"answer": "n/a"}, "print(6 * 7)")
        if "delegate" in last:
            return self._act("clerk", {"task": "find the order"}, "print(clerk(task='find the order'))")
        if "unknown" in last:
            return self._act("no_such_tool", {"x": 1}, "print(no_such_tool(x=1))")
        return self._answer(f"tasks: {tasks}")

    def shown(self, call: int = -1) -> str:
        return "\n".join(text for _, text in self.seen[call])


def _tool_agent(model: Desk, **kwargs: Any) -> ToolCallingAgent:
    return ToolCallingAgent(tools=[lookup_order, refund], model=model, verbosity_level=LogLevel.OFF, **kwargs)


def _code_agent(model: Desk, **kwargs: Any) -> CodeAgent:
    return CodeAgent(tools=[lookup_order, refund], model=model, verbosity_level=LogLevel.OFF, **kwargs)


async def _play(adapter: Any, *steps: Any) -> ScenarioState:
    states: list[ScenarioState] = []
    await sim.run("smolagents", "contract", [adapter], script=[*steps, states.append])
    return states[0]


async def test_tool_calling_agent_reply_tool_call_memory_and_usage() -> None:
    model = Desk()
    agent = _tool_agent(model)
    tools_before = dict(agent.tools)
    state = await _play(
        smolagents_adapter(agent), sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()
    )

    assert state.messages[1]["content"] == "tasks: where is my order?; result: Observation: order 1042 is shipped"
    # Session mode: the second run sees the first task, its tool call and the observation.
    assert state.messages[3]["content"] == "tasks: where is my order? | thanks"
    assert "order 1042 is shipped" in model.shown()
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").last
    assert (call.input, call.output, call.turn) == ({"order_id": "1042"}, "order 1042 is shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    # final_answer carries the reply and is not reported; the second turn called nothing.
    assert state.tool_call_names() == ["lookup_order"]
    assert state.usage == Usage(input_tokens=30, output_tokens=15, total_tokens=45, requests=3)
    # The agent's own tools are back in place after the run.
    assert agent.tools == tools_before


async def test_a_raising_tool_and_an_unknown_tool_are_errors() -> None:
    state = await _play(
        smolagents_adapter(lambda: _tool_agent(Desk())),
        sim.user("refund it"),
        sim.agent(),
        sim.user("call the unknown tool"),
        sim.agent(),
    )

    assert not state.has_tool_call("refund")
    failed = state.tool_calls("refund", include_errors=True).last
    assert failed.error
    assert failed.input == {"order_id": "1042"}
    assert "order 1042 is not refundable" in (failed.output or "")
    assert "not refundable" in state.messages[1]["content"]
    # The call never reached a tool: the memory step supplies it.
    unknown = state.tool_calls("no_such_tool", include_errors=True).last
    assert unknown.error
    assert unknown.input == {"x": 1}
    assert "Unknown tool no_such_tool" in (unknown.output or "")


async def test_transcript_mode_forgets_a_rolled_back_exchange() -> None:
    model = Desk()

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(0)

    state = await _play(
        smolagents_adapter(lambda: _tool_agent(model), history="transcript"),
        sim.user("my code is 7421"),
        sim.agent(),
        backtrack,
        sim.user("hello again"),
        sim.agent(),
    )

    assert "7421" in model.shown(0)
    assert "7421" not in model.shown()
    assert state.messages[-1]["content"] == "tasks: hello again"


async def test_transcript_mode_renders_the_dialogue_into_the_task() -> None:
    model = Desk()
    await _play(
        smolagents_adapter(lambda: _tool_agent(model), history="transcript"),
        sim.user("one"),
        sim.agent(),
        sim.user("two"),
        sim.agent(),
    )

    assert "User: one\nAssistant: tasks: one\nUser: two" in model.shown()
    # One task step per run: the memory was reset.
    assert [kind for kind, _ in model.seen[-1]].count("user") == 1


async def test_session_mode_rebuilds_from_the_transcript_after_a_rollback() -> None:
    model = Desk()

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(0)

    await _play(
        smolagents_adapter(lambda: _tool_agent(model)),
        sim.user("my code is 7421"),
        sim.agent(),
        backtrack,
        sim.user("hello again"),
        sim.agent(),
    )

    assert "7421" not in model.shown()


async def test_code_agent_reports_the_real_tools_its_code_calls() -> None:
    model = Desk(code=True)
    state = await _play(
        smolagents_adapter(lambda: _code_agent(model)),
        sim.user("where is my order?"),
        sim.agent(),
        sim.user("refund it"),
        sim.agent(),
        sim.user("compute something"),
        sim.agent(),
    )

    assert "order 1042 is shipped" in state.messages[1]["content"]
    # Positional arguments of generated code get the tool's input names.
    call = state.tool_calls("lookup_order").last
    assert (call.input, call.output, call.turn) == ({"order_id": "1042"}, "order 1042 is shipped", 0)
    failed = state.tool_calls("refund", include_errors=True).last
    assert failed.error
    assert failed.turn == 1
    assert "not refundable" in (failed.output or "")
    # Code that called no tool is the python_interpreter call itself.
    snippet = state.tool_calls("python_interpreter").last
    assert snippet.input == {"code": "print(6 * 7)"}
    assert "42" in (snippet.output or "")
    assert snippet.turn == 2
    assert [call.name for call in state.tool_calls(include_errors=True)] == [
        "lookup_order",
        "refund",
        "python_interpreter",
    ]
    # Session mode: the third run still sees the first task.
    assert "where is my order?" in model.shown()
    assert state.usage is not None
    assert state.usage.requests == 6


@pytest.mark.parametrize("code", [False, True])
async def test_managed_agent_delegation_and_its_tools_are_reported_in_call_order(code: bool) -> None:
    def team() -> Any:
        clerk = _tool_agent(Desk(), name="clerk", description="Finds orders.")
        build = _code_agent if code else _tool_agent
        return build(Desk(code=code), managed_agents=[clerk])

    state = await _play(smolagents_adapter(team), sim.user("delegate this"), sim.agent())

    assert state.tool_call_names() == ["clerk", "lookup_order"]
    assert state.tool_calls("clerk").last.input == {"task": "find the order"}
    assert "order 1042 is shipped" in (state.tool_calls("clerk").last.output or "")
    assert "order 1042 is shipped" in state.messages[1]["content"]
    # Manager: delegation and answer; clerk: tool call and answer.
    assert state.usage == Usage(input_tokens=40, output_tokens=20, total_tokens=60, requests=4)


async def test_a_factory_gives_every_scenario_its_own_agent() -> None:
    models: list[Desk] = []

    def factory() -> ToolCallingAgent:
        models.append(Desk())
        return _tool_agent(models[-1])

    adapter = smolagents_adapter(factory)
    first = await _play(adapter, sim.user("first secret"), sim.agent(), sim.user("more"), sim.agent())
    second = await _play(adapter, sim.user("second"), sim.agent())

    assert first.messages[-1]["content"] == "tasks: first secret | more"
    assert second.messages[-1]["content"] == "tasks: second"
    assert len(models) == 2


async def test_a_shared_agent_starts_over_for_another_scenario() -> None:
    model = Desk()
    adapter = smolagents_adapter(_tool_agent(model))
    await _play(adapter, sim.user("first secret"), sim.agent())
    second = await _play(adapter, sim.user("second"), sim.agent())

    assert second.messages[-1]["content"] == "tasks: second"


async def test_run_kwargs_and_a_full_result() -> None:
    seen: list[Any] = []

    def render(output: Any) -> str:
        seen.append(output)
        return str(output).upper()

    adapter = smolagents_adapter(lambda: _tool_agent(Desk()), content_fn=render, return_full_result=True, max_steps=3)
    state = await _play(adapter, sim.user("hi"), sim.agent())

    assert state.messages[-1]["content"] == "TASKS: HI"
    assert not isinstance(seen[0], RunResult)
