"""``langchain_adapter`` against the real LangChain and LangGraph, with a scripted chat model.

Neither library is a dependency of the project, so the module is skipped in a
plain test run. Run it with the libraries overlaid:

    uv run --no-group harbor --with langchain --with langgraph --with langgraph-supervisor --with langchain-classic \
        pytest tests/contract/test_langchain_contract.py --no-cov -q

``langgraph-supervisor`` and ``langchain-classic`` are optional: their tests are
skipped when the package is missing.
"""

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langchain.agents")
pytest.importorskip("langgraph.graph")

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolErrorMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.langchain import langchain_adapter

USAGE = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}], usage_metadata=USAGE)


class Desk:
    """A scripted model: calls a tool when the user names one, otherwise reports what it was shown."""

    def __init__(self) -> None:
        self.seen: list[list[BaseMessage]] = []

    def __call__(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        last = messages[-1]
        prompt = str(last.content) if last.type == "human" else ""
        if "order" in prompt:
            return _call("lookup_order", {"order_id": 7}, "call-1")
        if "refund" in prompt:
            return _call("refund", {"order_id": 7}, "call-2")
        if "delete" in prompt:
            return _call("delete_account", {"user": "u1"}, "call-3")
        users = " | ".join(str(m.content) for m in messages if m.type == "human")
        results = " | ".join(str(m.content) for m in messages if m.type == "tool") or "none"
        return AIMessage(content=f"users: {users}; tool results: {results}", usage_metadata=USAGE)


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
def refund(order_id: int) -> str:
    """Refund an order."""
    raise ValueError("refunds are closed")


@tool
def delete_account(user: str) -> str:
    """Delete an account."""
    return f"deleted {user}"


TOOLS = [lookup_order, refund, delete_account]


def _agent(model: Desk, *, checkpointer: bool = True, middleware: Any = ()) -> Any:
    saver = InMemorySaver() if checkpointer else None
    return create_agent(Scripted(script=model), TOOLS, checkpointer=saver, middleware=middleware)


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


@pytest.mark.parametrize("checkpointer", [True, False])
async def test_reply_tool_call_memory_and_usage_in_session_mode(checkpointer: bool) -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(langchain_adapter(lambda: _agent(model, checkpointer=checkpointer)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "order 7 is shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    # Second turn: the model is shown the first exchange, its tool result included, and no new call is recorded.
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert [m.type for m in model.seen[-1]] == ["human", "ai", "tool", "ai", "human"]
    assert state.tool_calls(turn=1) == []
    # Three model calls over the two turns, each reporting usage_metadata.
    assert result.usage == Usage(input_tokens=30, output_tokens=12, total_tokens=42, requests=3)


async def test_a_tool_error_the_graph_reports_is_an_error() -> None:
    states: list[ScenarioState] = []
    middleware = [ToolErrorMiddleware(on_error=lambda exc, request: f"failed: {exc}")]
    adapter = langchain_adapter(_agent(Desk(), middleware=middleware))

    result = await _play(adapter, [sim.user("refund it"), sim.agent(), states.append])

    state = states[0]
    assert result.messages[-1]["content"] == "users: refund it; tool results: failed: refunds are closed"
    assert not state.has_tool_call("refund")
    assert state.tool_calls() == []
    (failed,) = state.tool_calls(include_errors=True)
    assert (failed.name, failed.input, failed.output, failed.status) == (
        "refund",
        {"order_id": 7},
        "failed: refunds are closed",
        "error",
    )
    assert failed.error


async def test_a_tool_exception_the_graph_does_not_catch_propagates() -> None:
    with pytest.raises(ValueError, match="refunds are closed"):
        await _play(langchain_adapter(_agent(Desk())), [sim.user("refund it"), sim.agent()])


def _hitl_agent(model: Desk) -> Any:
    return _agent(model, middleware=[HumanInTheLoopMiddleware(interrupt_on={"delete_account": True})])


async def test_an_interrupted_call_is_not_executed_and_the_interrupt_is_the_reply() -> None:
    states: list[ScenarioState] = []

    result = await _play(langchain_adapter(_hitl_agent(Desk())), [sim.user("delete me"), sim.agent(), states.append])

    state = states[0]
    assert "delete_account" in result.messages[-1]["content"]
    assert "action_requests" in result.messages[-1]["content"]
    assert not state.has_tool_call("delete_account")
    (pending,) = state.tool_calls(include_errors=True)
    assert (pending.name, pending.input, pending.status) == ("delete_account", {"user": "u1"}, "paused")


async def test_resume_fn_continues_an_interrupted_thread() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    adapter = langchain_adapter(
        _hitl_agent(model),
        resume_fn=lambda agent_input: {"decisions": [{"type": "approve"}]},
    )
    script = [sim.user("delete me"), sim.agent(), sim.user("yes, go ahead"), sim.agent(), states.append]

    result = await _play(adapter, script)

    state = states[0]
    # The approval resumed the graph: the user's "yes" is not a new message, the pending call ran.
    assert result.messages[-1]["content"] == "users: delete me; tool results: deleted u1"
    approved = state.tool_calls("delete_account", turn=1).first
    assert (approved.input, approved.output, approved.status) == ({"user": "u1"}, "deleted u1", "success")
    assert [call.status for call in state.tool_calls(include_errors=True)] == ["paused", "success"]


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
@pytest.mark.parametrize("checkpointer", [True, False])
async def test_a_rollback_reaches_the_agent(history: Any, checkpointer: bool) -> None:
    model = Desk()

    result = await _play(langchain_adapter(_agent(model, checkpointer=checkpointer), history=history), ROLLBACK_SCRIPT)

    assert result.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
    assert "secret" not in str(model.seen[-1])


@pytest.mark.parametrize("checkpointer", [True, False])
async def test_transcript_mode_does_not_replay_tool_results(checkpointer: bool) -> None:
    model = Desk()
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    result = await _play(langchain_adapter(_agent(model, checkpointer=checkpointer), history="transcript"), script)

    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: none"


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    adapter = langchain_adapter(_agent(Desk()))

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"


class DeskState(TypedDict):
    dialogue: Annotated[list[BaseMessage], add_messages]
    plan: str


@dataclass
class DeskContext:
    user: str


async def test_a_hand_built_graph_with_its_own_state_key_context_and_config() -> None:
    states: list[ScenarioState] = []
    model = Scripted(script=Desk())

    async def agent(state: DeskState, config: RunnableConfig, runtime: Runtime[DeskContext]) -> dict[str, Any]:
        reply = await model.ainvoke(state["dialogue"])
        if not reply.tool_calls:
            tenant = config["configurable"]["tenant"]
            reply = AIMessage(content=f"{reply.content}; {state['plan']}/{runtime.context.user}/{tenant}")
        return {"dialogue": [reply]}

    builder = StateGraph(DeskState, context_schema=DeskContext)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS, messages_key="dialogue", handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", lambda state: tools_condition(state, messages_key="dialogue"))
    builder.add_edge("tools", "agent")
    adapter = langchain_adapter(
        builder.compile(),
        input_key="dialogue",
        inputs={"plan": "pro"},
        config={"configurable": {"tenant": "acme"}},
        context=DeskContext(user="ann"),
    )

    result = await _play(
        adapter, [sim.user("refund it"), sim.agent(), sim.user("and my order?"), sim.agent(), states.append]
    )

    state = states[0]
    assert result.messages[1]["content"].endswith("; pro/ann/acme")
    (failed,) = state.tool_calls(include_errors=True, turn=0)
    assert (failed.name, failed.status) == ("refund", "error")
    assert "refunds are closed" in (failed.output or "")
    assert state.tool_calls("lookup_order", turn=1).first.output == "order 7 is shipped"
    assert "refunds are closed" in result.messages[3]["content"]


async def test_output_key_answers_from_another_state_key() -> None:
    builder = StateGraph(DeskState)
    builder.add_node("agent", lambda state: {"dialogue": [AIMessage(content="internal")], "plan": "upgraded"})
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    by_key = await _play(
        langchain_adapter(graph, input_key="dialogue", inputs={"plan": "free"}, output_key="plan"),
        [sim.user("hi"), sim.agent()],
    )
    custom = await _play(
        langchain_adapter(graph, input_key="dialogue", content_fn=lambda final: f"{len(final['dialogue'])} messages"),
        [sim.user("hi"), sim.agent()],
    )

    assert by_key.messages[-1]["content"] == "upgraded"
    assert custom.messages[-1]["content"] == "2 messages"


async def test_a_supervisor_graph_shows_the_tools_of_its_sub_agents() -> None:
    supervisor = pytest.importorskip("langgraph_supervisor")
    states: list[ScenarioState] = []

    def boss(messages: list[BaseMessage]) -> AIMessage:
        if any(m.type == "ai" and m.content == "found it" for m in messages):
            return AIMessage(content="the order is shipped", usage_metadata=USAGE)
        return _call("transfer_to_orders", {}, "call-0")

    def clerk(messages: list[BaseMessage]) -> AIMessage:
        if any(m.type == "tool" and m.name == "lookup_order" for m in messages):
            return AIMessage(content="found it", usage_metadata=USAGE)
        return _call("lookup_order", {"order_id": 7}, "call-1")

    orders = create_agent(Scripted(script=clerk), TOOLS, name="orders")
    graph = supervisor.create_supervisor([orders], model=Scripted(script=boss)).compile()

    result = await _play(langchain_adapter(graph), [sim.user("where is my order?"), sim.agent(), states.append])

    state = states[0]
    assert result.messages[-1]["content"] == "the order is shipped"
    # The sub-agent's tool call never reaches the supervisor's state (output_mode="last_message").
    assert state.tool_call_names() == ["transfer_to_orders", "lookup_order", "transfer_back_to_supervisor"]
    assert state.tool_calls("lookup_order").first.output == "order 7 is shipped"
    # Supervisor twice, the sub-agent twice; the synthetic hand-back message reports no usage.
    assert result.usage is not None
    assert result.usage.requests == 4


async def test_a_bare_chat_model_and_a_chain() -> None:
    states: list[ScenarioState] = []
    model = Desk()
    script = [sim.user("hello"), sim.agent(), sim.user("where is my order?"), sim.agent(), states.append]

    result = await _play(langchain_adapter(Scripted(script=model)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: hello; tool results: none"
    # Nothing runs the tool a bare model asks for.
    assert result.messages[3]["content"] == ""
    assert [m.type for m in model.seen[-1]] == ["human", "ai", "human"]
    (asked,) = state.tool_calls(include_errors=True)
    assert (asked.name, asked.input, asked.status) == ("lookup_order", {"order_id": 7}, "paused")
    assert result.usage == Usage(input_tokens=20, output_tokens=8, total_tokens=28, requests=2)

    prompt = ChatPromptTemplate.from_messages([("system", "Be {tone}."), MessagesPlaceholder("dialogue")])
    chain = prompt | Scripted(script=model) | StrOutputParser()
    chained = await _play(
        langchain_adapter(chain, input_key="dialogue", inputs={"tone": "brief"}),
        [sim.user("hello"), sim.agent()],
    )

    assert chained.messages[-1]["content"] == "users: hello; tool results: none"
    assert [m.type for m in model.seen[-1]] == ["system", "human"]


async def test_a_legacy_agent_executor() -> None:
    classic = pytest.importorskip("langchain_classic.agents")
    model = Desk()
    states: list[ScenarioState] = []
    prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("chat_history"), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")]
    )
    agent = classic.create_tool_calling_agent(Scripted(script=model), TOOLS, prompt)
    executor = classic.AgentExecutor(agent=agent, tools=TOOLS, return_intermediate_steps=True)
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(langchain_adapter(executor), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output) == ({"order_id": 7}, "order 7 is shipped")
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    # The adapter carries the first turn, its tool call and result included, into the second.
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert [m.type for m in model.seen[-1]] == ["human", "ai", "tool", "ai", "human"]

    forgetful = await _play(langchain_adapter(executor, history="transcript"), ROLLBACK_SCRIPT)
    assert forgetful.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
