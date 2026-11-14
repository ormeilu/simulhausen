"""``crewai_adapter`` against the real CrewAI, with a scripted ``BaseLLM`` in place of a provider.

CrewAI is not a dependency of the project, so the module is skipped in a plain
test run. Run it with the library overlaid:

    uv run --no-group harbor --with crewai pytest tests/contract/test_crewai_contract.py --no-cov -q

The module switches CrewAI's telemetry and tracing off through their
environment variables before it imports the library.
"""

import asyncio
import json
import os
import re
from typing import Any

import pytest

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

pytest.importorskip("crewai")

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from pydantic import BaseModel

import simulhausen as sim
from simulhausen import AgentInput, ScenarioState, Usage
from simulhausen.frameworks.crewai import crewai_adapter

executed: list[str] = []


@tool("lookup_order")
def lookup_order(order_id: str) -> str:
    """Looks an order up."""
    executed.append(f"lookup_order {order_id}")
    return f"order {order_id} is shipped"


@tool("refund")
def refund(order_id: str) -> str:
    """Refunds an order."""
    executed.append(f"refund {order_id}")
    raise ValueError(f"order {order_id} is not refundable")


class Summary(BaseModel):
    answer: str


class Desk(BaseLLM):
    """A scripted model: acts on the newest request, then answers with the tool's result.

    ``native=True`` answers with tool calls the way a function-calling model
    does; otherwise it writes CrewAI's ReAct text format.
    """

    def __init__(self, *, native: bool = True) -> None:
        super().__init__(model="desk")
        self._native = native
        self._seen: list[list[dict[str, Any]]] = []

    def supports_function_calling(self) -> bool:
        return self._native

    def _act(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._native:
            call = {"name": name, "arguments": json.dumps(arguments)}
            return [{"id": f"call-{len(self._seen)}", "type": "function", "function": call}]
        return f"Thought: I need {name}.\nAction: {name}\nAction Input: {json.dumps(arguments)}"

    def _answer(self, text: str) -> str:
        return text if self._native else f"Thought: I know the answer.\nFinal Answer: {text}"

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        self._seen.append([dict(message) for message in messages])
        self._track_token_usage_internal({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        last = messages[-1]
        text = str(last.get("content") or "")
        if last["role"] == "tool":
            return self._answer(f"the desk says: {text}")
        if "Analyze the tool result" in text:
            observation = str(messages[-2]["content"]).split("Observation:")[-1].strip().splitlines()[0]
            return self._answer(f"the desk says: {observation}")
        # A crew's task text carries the history after the request; only the request decides.
        request = text.split("Conversation so far:")[0]
        number = re.search(r"\d+", request)
        order_id = number.group() if number else "1042"
        if "unknown" in request:
            return self._act("no_such_tool", {"x": 1})
        if "refund" in request:
            return self._act("refund", {"order_id": order_id})
        if "order" in request:
            return self._act("lookup_order", {"order_id": order_id})
        if "summary" in request:
            return self._answer('{"answer": "all good"}')
        return self._answer("hello")

    def roles(self, call: int = -1) -> list[str]:
        return [str(message["role"]) for message in self._seen[call]]

    def shown(self, call: int = -1) -> str:
        return "\n".join(str(message.get("content") or "") for message in self._seen[call])

    @property
    def calls(self) -> int:
        return len(self._seen)


def _agent(model: Desk, role: str = "desk") -> Agent:
    return Agent(
        role=role, goal="Help the customer.", backstory="An order desk.", llm=model, tools=[lookup_order, refund]
    )


def _crew(model: Desk) -> Crew:
    agent = _agent(model)
    task = Task(
        description="Customer request: {message}\nConversation so far:\n{history}",
        expected_output="A reply to the customer.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


async def _play(adapter: Any, *steps: Any, name: str = "crewai") -> ScenarioState:
    states: list[ScenarioState] = []
    await sim.run(name, "contract", [adapter], script=[*steps, states.append])
    return states[0]


def _backtrack(state: ScenarioState) -> None:
    state.rollback_messages_to(0)


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    executed.clear()


# -- a single Agent -----------------------------------------------------------------


@pytest.mark.parametrize("native", [True, False])
async def test_agent_reply_tool_call_memory_and_usage(native: bool) -> None:
    model = Desk(native=native)
    state = await _play(
        crewai_adapter(_agent(model)), sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()
    )

    assert state.messages[1]["content"] == "the desk says: order 1042 is shipped"
    assert state.messages[3]["content"] == "hello"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").last
    assert (call.input, call.output, call.turn) == ({"order_id": "1042"}, "order 1042 is shipped", 0)
    assert state.tool_call_names() == ["lookup_order"]
    # Session mode: the second kickoff gets the first one's messages, tool result and answer included,
    # under one system prompt.
    shown = model.shown()
    assert "where is my order?" in shown
    assert "order 1042 is shipped" in shown
    assert "the desk says: order 1042 is shipped" in shown
    assert model.roles().count("system") == 1
    assert model.roles()[-1] == "user"
    # Two model calls in the first kickoff, one in the second.
    assert state.usage == Usage(input_tokens=30, output_tokens=15, total_tokens=45, requests=3)


@pytest.mark.parametrize("native", [True, False])
async def test_agent_failing_calls_are_errors(native: bool) -> None:
    state = await _play(
        crewai_adapter(lambda: _agent(Desk(native=native))),
        sim.user("refund it"),
        sim.agent(),
        sim.user("call the unknown tool"),
        sim.agent(),
    )

    assert not state.has_tool_call("refund")
    # A ReAct-style agent retries the failing call itself; the retries are one error.
    assert executed.count("refund 1042") >= 1
    failures = state.tool_calls("refund", include_errors=True)
    assert len(failures) == 1
    assert failures.last.error
    assert failures.last.input == {"order_id": "1042"}
    assert "order 1042 is not refundable" in (failures.last.output or "")
    assert failures.last.turn == 0
    unknown = state.tool_calls("no_such_tool", include_errors=True)
    assert len(unknown) == 1
    assert unknown.last.error
    assert unknown.last.turn == 1
    assert state.tool_call_names() == []


async def test_agent_transcript_mode_forgets_a_rolled_back_exchange() -> None:
    model = Desk()
    state = await _play(
        crewai_adapter(lambda: _agent(model), history="transcript"),
        sim.user("my code is 7421, where is my order?"),
        sim.agent(),
        _backtrack,
        sim.user("hello again"),
        sim.agent(),
    )

    assert "7421" in model.shown(0)
    assert "7421" not in model.shown()
    assert state.messages[-1]["content"] == "hello"


async def test_agent_transcript_mode_sends_plain_messages_and_session_mode_rebuilds_after_a_rollback() -> None:
    model = Desk()
    await _play(
        crewai_adapter(lambda: _agent(model), history="transcript"),
        sim.user("where is my order?"),
        sim.agent(),
        sim.user("thanks"),
        sim.agent(),
    )
    # The first exchange comes back as text; the tool call and its result are not replayed.
    assert model.roles() == ["system", "user", "assistant", "user"]

    model = Desk()
    await _play(
        crewai_adapter(lambda: _agent(model)),
        sim.user("my code is 7421"),
        sim.agent(),
        _backtrack,
        sim.user("hello again"),
        sim.agent(),
    )
    assert "7421" not in model.shown()


async def test_agent_kickoff_kwargs_and_a_structured_reply() -> None:
    seen: list[Any] = []

    def render(raw: Any) -> str:
        seen.append(raw)
        return f"answer={raw.answer}"

    adapter = crewai_adapter(lambda: _agent(Desk()), content_fn=render, response_format=Summary)
    state = await _play(adapter, sim.user("a summary please"), sim.agent())

    assert state.messages[-1]["content"] == "answer=all good"
    assert seen == [Summary(answer="all good")]


async def test_concurrent_scenarios_keep_their_tool_calls_apart() -> None:
    adapter = crewai_adapter(lambda: _agent(Desk()))
    states = await asyncio.gather(
        *[
            _play(adapter, sim.user(f"where is order {number}?"), sim.agent(), name=f"crewai-{number}")
            for number in (11, 22, 33)
        ]
    )

    assert [state.tool_calls("lookup_order").inputs for state in states] == [
        [{"order_id": "11"}],
        [{"order_id": "22"}],
        [{"order_id": "33"}],
    ]


# -- a Crew -------------------------------------------------------------------------


async def test_crew_inputs_tool_call_and_usage_per_turn() -> None:
    model = Desk()
    crew = _crew(model)
    turns: list[Usage | None] = []

    def usage(state: ScenarioState) -> None:
        turns.append(state.usage)

    state = await _play(
        crewai_adapter(crew), sim.user("where is my order?"), sim.agent(), usage, sim.user("thanks"), sim.agent()
    )

    assert state.messages[1]["content"] == "the desk says: order 1042 is shipped"
    assert state.messages[3]["content"] == "hello"
    call = state.tool_calls("lookup_order").last
    assert (call.input, call.output, call.turn) == ({"order_id": "1042"}, "order 1042 is shipped", 0)
    # The second kickoff gets the newest message and the dialogue before it.
    shown = model.shown()
    assert "Customer request: thanks" in shown
    assert "User: where is my order?\nAssistant: the desk says: order 1042 is shipped" in shown
    # CrewOutput.token_usage grows over the crew's lifetime; a turn is its growth.
    assert turns == [Usage(input_tokens=20, output_tokens=10, total_tokens=30, requests=2)]
    assert state.usage == Usage(input_tokens=30, output_tokens=15, total_tokens=45, requests=3)


async def test_crew_failing_tool_and_a_rolled_back_exchange() -> None:
    model = Desk()
    state = await _play(
        crewai_adapter(lambda: _crew(model), history="transcript"),
        sim.user("my code is 7421, refund it"),
        sim.agent(),
        lambda state: None if state.tool_calls("refund", include_errors=True).last.error else sim.fail("no error"),
        _backtrack,
        sim.user("hello again"),
        sim.agent(),
    )

    assert "7421" in model.shown(0)
    assert "7421" not in model.shown()
    assert state.messages[-1]["content"] == "hello"
    # The rollback took the first turn's tool call with it.
    assert len(state.tool_calls(include_errors=True)) == 0


async def test_crew_with_two_agents_reports_tools_in_call_order_and_custom_inputs() -> None:
    model = Desk()

    def crew() -> Crew:
        clerk, cashier = _agent(model, "clerk"), _agent(model, "cashier")
        find = Task(description="Find order 7 for: {question}", expected_output="A status.", agent=clerk)
        pay = Task(description="Now refund 7 for: {question}", expected_output="A refund.", agent=cashier)
        return Crew(agents=[clerk, cashier], tasks=[find, pay])

    def inputs(agent_input: AgentInput) -> dict[str, Any]:
        return {"question": agent_input.last_new_user_message_str().upper()}

    state = await _play(crewai_adapter(crew, inputs_fn=inputs), sim.user("please help"), sim.agent())

    assert "PLEASE HELP" in model.shown(0)
    assert [(call.name, call.status) for call in state.tool_calls(include_errors=True)] == [
        ("lookup_order", "success"),
        ("refund", "error"),
    ]
    assert "not refundable" in state.messages[-1]["content"]


# -- a Flow -------------------------------------------------------------------------


def _flow(model: Desk) -> Any:
    class DeskFlow(Flow):  # type: ignore[type-arg]
        @start()
        async def answer(self) -> str:
            self.state["turns"] = self.state.get("turns", 0) + 1
            output = await _agent(model).kickoff_async(self.state["message"])
            return f"{output.raw} (turn {self.state['turns']}; history: {self.state.get('history')})"

    return DeskFlow()


async def test_flow_reply_tool_call_and_state_per_thread() -> None:
    model = Desk()
    adapter = crewai_adapter(lambda: _flow(model))
    state = await _play(adapter, sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent())
    other = await _play(adapter, sim.user("hello"), sim.agent())

    assert state.messages[1]["content"] == "the desk says: order 1042 is shipped (turn 1; history: )"
    # Session mode keeps a thread's flow, and with it the flow's state, between turns.
    assert state.messages[3]["content"] == (
        "hello (turn 2; history: User: where is my order?\n"
        "Assistant: the desk says: order 1042 is shipped (turn 1; history: ))"
    )
    assert other.messages[1]["content"] == "hello (turn 1; history: )"
    # The tool ran inside an agent the flow started.
    assert state.tool_calls("lookup_order").last.input == {"order_id": "1042"}
    assert state.tool_call_names() == ["lookup_order"]


async def test_flow_transcript_mode_builds_a_new_flow_for_every_call() -> None:
    adapter = crewai_adapter(lambda: _flow(Desk()), history="transcript", history_key=None, input_key="message")
    state = await _play(adapter, sim.user("one"), sim.agent(), sim.user("two"), sim.agent())

    assert [message["content"] for message in state.messages if message["role"] == "assistant"] == [
        "hello (turn 1; history: None)",
        "hello (turn 1; history: None)",
    ]
