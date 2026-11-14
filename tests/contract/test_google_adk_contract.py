"""Contract test of ``google_adk_adapter`` against the real Google ADK, with a scripted model and no network.

Run it with the framework overlaid on the project environment::

    uv run --no-group harbor --with google-adk pytest tests/contract/test_google_adk_contract.py --no-cov -q
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, LongRunningFunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.workflow import START, Workflow
from google.genai import types
from pydantic import Field

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.google_adk import google_adk_adapter

pytestmark = pytest.mark.filterwarnings("ignore")

USAGE = types.GenerateContentResponseUsageMetadata(prompt_token_count=5, candidates_token_count=2, total_token_count=7)


def say(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]), usage_metadata=USAGE)


def call(name: str, **args: Any) -> LlmResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args))
    return LlmResponse(content=types.Content(role="model", parts=[part]), usage_metadata=USAGE)


def _flat(content: types.Content) -> str:
    chunks: list[str] = []
    for part in content.parts or []:
        if part.text:
            chunks.append(part.text)
        elif part.function_call:
            chunks.append(f"call:{part.function_call.name}")
        elif part.function_response:
            chunks.append(f"result:{part.function_response.name}")
    return f"{content.role}:{'|'.join(chunks)}"


class ScriptedLlm(BaseLlm):
    """Answers with the next scripted response and keeps the contents of every request."""

    model: str = "scripted"
    script: list[LlmResponse] = Field(default_factory=list)
    requests: list[list[str]] = Field(default_factory=list)

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        self.requests.append([_flat(content) for content in llm_request.contents])
        yield self.script.pop(0)


def lookup_order(order_id: str) -> dict[str, str]:
    """Look up an order."""
    return {"order_id": order_id, "status": "shipped"}


def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    raise RuntimeError(f"order {order_id} is locked")


def refund(order_id: str) -> str:
    """Refund an order."""
    return "refunded"


def export_report(order_id: str) -> None:
    """Start a report export."""
    return None


def tool_error(tool: Any, args: dict[str, Any], tool_context: Any, error: Exception) -> dict[str, str]:
    return {"error": f"{type(error).__name__}: {error}"}


def model(*script: LlmResponse) -> ScriptedLlm:
    return ScriptedLlm(script=list(script), requests=[])


def support_agent(llm: ScriptedLlm, **kwargs: Any) -> LlmAgent:
    return LlmAgent(
        name="support",
        model=llm,
        tools=[
            lookup_order,
            cancel_order,
            FunctionTool(refund, require_confirmation=True),
            LongRunningFunctionTool(export_report),
        ],
        **kwargs,
    )


async def play(adapter: Any, *script: Any) -> sim.ScenarioResult:
    return await sim.run(name="contract", description="google adk contract", agents=[adapter], script=list(script))


async def test_reply_tool_call_usage_and_session_memory() -> None:
    llm = model(call("lookup_order", order_id="7"), say("Order 7 has shipped."), say("You asked about order 7."))
    seen: list[ScenarioState] = []

    result = await play(
        google_adk_adapter(lambda: support_agent(llm)),
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
    looked_up = state.tool_calls("lookup_order").last
    assert looked_up.input == {"order_id": "7"}
    assert looked_up.output == '{"order_id": "7", "status": "shipped"}'
    assert looked_up.turn == 0
    # The second turn's request carries the first exchange, the tool call included.
    assert llm.requests[-1] == [
        "user:Where is order 7?",
        "model:call:lookup_order",
        "user:result:lookup_order",
        "model:Order 7 has shipped.",
        "user:What did I ask?",
    ]
    assert state.usage == Usage(input_tokens=15, output_tokens=6, total_tokens=21, requests=3)


async def test_tool_error_handled_by_a_callback_is_recorded_as_an_error() -> None:
    llm = model(call("cancel_order", order_id="7"), say("I could not cancel it."))
    seen: list[ScenarioState] = []

    await play(
        google_adk_adapter(support_agent(llm, on_tool_error_callback=tool_error)),
        sim.user("Cancel order 7"),
        sim.agent(),
        seen.append,
    )

    state = seen[0]
    assert not state.has_tool_call("cancel_order")
    failed = state.tool_calls("cancel_order", include_errors=True).last
    assert failed.status == "error"
    assert failed.input == {"order_id": "7"}
    assert failed.output == "RuntimeError: order 7 is locked"
    assert failed.execution is not None
    assert failed.execution.tool_call_error is True


async def test_unhandled_tool_exception_propagates_as_adk_raises_it() -> None:
    llm = model(call("cancel_order", order_id="7"))

    with pytest.raises(RuntimeError, match="order 7 is locked"):
        await play(google_adk_adapter(support_agent(llm)), sim.user("Cancel order 7"), sim.agent())


async def test_confirmation_and_long_running_calls_are_not_executed() -> None:
    llm = model(call("refund", order_id="7"), call("export_report", order_id="7"))
    seen: list[ScenarioState] = []

    await play(
        google_adk_adapter(support_agent(llm)),
        sim.user("Refund order 7"),
        sim.agent(),
        sim.user("Export the report"),
        sim.agent(),
        seen.append,
    )

    state = seen[0]
    assert not state.has_tool_call("refund")
    assert not state.has_tool_call("export_report")
    calls = state.tool_calls(include_errors=True)
    assert [(c.name, c.status, c.input) for c in calls] == [
        ("refund", "paused", {"order_id": "7"}),
        ("export_report", "paused", {"order_id": "7"}),
    ]


async def test_transcript_history_forgets_a_rolled_back_exchange() -> None:
    llm = model(say("one"), say("two"), say("three"))

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(2)

    service = InMemorySessionService()
    await play(
        google_adk_adapter(support_agent(llm), history="transcript", session_service=service, app_name="shop"),
        sim.user("first"),
        sim.agent(),
        sim.user("secret"),
        sim.agent(),
        backtrack,
        sim.user("third"),
        sim.agent(),
    )

    assert llm.requests[-1] == ["user:first", "model:one", "user:third"]
    # Every call ran in a session of its own, deleted afterwards.
    assert (await service.list_sessions(app_name="shop", user_id="simulhausen")).sessions == []


async def test_session_history_restarts_from_the_transcript_after_a_rollback() -> None:
    llm = model(say("one"), say("two"), say("three"))

    def backtrack(state: ScenarioState) -> None:
        state.rollback_messages_to(2)

    await play(
        google_adk_adapter(support_agent(llm)),
        sim.user("first"),
        sim.agent(),
        sim.user("secret"),
        sim.agent(),
        backtrack,
        sim.user("third"),
        sim.agent(),
    )

    assert llm.requests[1] == ["user:first", "model:one", "user:secret"]
    assert llm.requests[-1] == ["user:first", "model:one", "user:third"]


async def test_initial_state_and_ready_runner() -> None:
    llm = model(say("hello"))
    agent = LlmAgent(name="support", model=llm, instruction="The plan is {plan}.")
    service = InMemorySessionService()
    runner = Runner(app_name="shop", agent=agent, session_service=service)
    seen: list[ScenarioState] = []

    await play(
        google_adk_adapter(runner, user_id="ann", state={"plan": "pro"}), sim.user("hi"), sim.agent(), seen.append
    )

    session = await service.get_session(app_name="shop", user_id="ann", session_id=seen[0].thread_id)
    assert session is not None
    assert session.state["plan"] == "pro"
    assert [event.author for event in session.events] == ["user", "support"]


async def test_sub_agent_transfer_and_agent_tool_calls_are_visible() -> None:
    llm = model(
        call("transfer_to_agent", agent_name="orders"),
        call("researcher", request="order 7"),
        call("lookup_order", order_id="7"),
        say("It shipped."),
        say("Order 7 has shipped."),
    )
    researcher = LlmAgent(name="researcher", model=llm, description="Looks orders up.", tools=[lookup_order])
    orders = LlmAgent(name="orders", model=llm, description="Handles orders.", tools=[AgentTool(researcher)])
    root = LlmAgent(name="root", model=llm, sub_agents=[orders])
    seen: list[ScenarioState] = []

    result = await play(google_adk_adapter(root), sim.user("Where is order 7?"), sim.agent(), seen.append)

    state = seen[0]
    assert result.messages[-1]["content"] == "Order 7 has shipped."
    assert state.tool_call_names() == ["transfer_to_agent", "researcher", "lookup_order"]
    assert state.tool_calls("researcher").last.output == "It shipped."
    assert state.tool_calls("lookup_order").last.input == {"order_id": "7"}
    # Five model calls: three in the outer run, two inside the AgentTool.
    assert state.usage == Usage(input_tokens=25, output_tokens=10, total_tokens=35, requests=5)


async def test_workflow_agent_reply_is_the_last_final_response() -> None:
    llm = model(say("draft"), say("polished"))
    pipeline = SequentialAgent(
        name="pipeline", sub_agents=[LlmAgent(name="writer", model=llm), LlmAgent(name="editor", model=llm)]
    )

    result = await play(google_adk_adapter(pipeline), sim.user("Write it"), sim.agent())

    assert result.messages[-1]["content"] == "polished"


def shout(node_input: str) -> str:
    return str(node_input).upper()


async def test_app_and_workflow_targets() -> None:
    llm = model(say("hello"), say("draft"))
    app = App(name="shop", root_agent=LlmAgent(name="support", model=llm))

    result = await play(google_adk_adapter(app), sim.user("hi"), sim.agent())

    assert result.messages[-1]["content"] == "hello"

    workflow = Workflow(name="pipeline", edges=[(START, LlmAgent(name="writer", model=llm), shout)])

    result = await play(google_adk_adapter(workflow), sim.user("Write it"), sim.agent())

    # The reply is the output of the workflow's last node, not the writer's text.
    assert result.messages[-1]["content"] == "DRAFT"
