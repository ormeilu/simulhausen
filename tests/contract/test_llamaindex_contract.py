"""``llamaindex_adapter`` against the real LlamaIndex agent workflows, with a scripted function-calling LLM.

LlamaIndex is not a dependency of the project, so the module is skipped in a
plain test run. Run it with the library overlaid:

    uv run --no-group harbor --with llama-index-core pytest tests/contract/test_llamaindex_contract.py --no-cov -q
"""

from collections.abc import Sequence
from typing import Any

import pytest

pytest.importorskip("llama_index.core.agent.workflow")

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.workflow import Context, Event, HumanResponseEvent, InputRequiredEvent
from pydantic import BaseModel

import simulhausen as sim
from simulhausen import ScenarioState, Usage
from simulhausen.frameworks.llamaindex import llamaindex_adapter

SYSTEM = "You are the order desk."


def _call(name: str, call_id: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(
        role="assistant", blocks=[ToolCallBlock(tool_call_id=call_id, tool_name=name, tool_kwargs=kwargs)]
    )


def _texts(messages: Sequence[ChatMessage], role: str) -> str:
    return " | ".join(m.content or "" for m in messages if m.role.value == role)


class Desk:
    """A scripted model: calls a tool when the user names one, otherwise reports what it was shown."""

    def __init__(self) -> None:
        self.seen: list[list[ChatMessage]] = []

    def __call__(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatMessage:
        self.seen.append(list(messages))
        last = messages[-1]
        prompt = (last.content or "") if last.role.value == "user" else ""
        if "order" in prompt:
            return _call("lookup_order", "call-1", order_id=7)
        if "refund" in prompt:
            return _call("refund", "call-2", order_id=7)
        if "delete" in prompt:
            return _call("delete_account", "call-3", user="u1")
        if "wipe" in prompt:
            return _call("wipe_data", "call-4")
        results = _texts(messages, "tool") or "none"
        return ChatMessage(role="assistant", content=f"users: {_texts(messages, 'user')}; tool results: {results}")

    def roles(self, call: int = -1) -> list[str]:
        return [m.role.value for m in self.seen[call]]


class ScriptedLLM(MockFunctionCallingLLM):
    """The mock LLM of LlamaIndex, reporting OpenAI-style usage in the raw response of every call."""

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        message = self.response_generator(messages, **kwargs)

        async def stream() -> Any:
            raw = {"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
            yield ChatResponse(message=message, delta=message.content or "", raw=raw)

        return stream()


def lookup_order(order_id: int) -> str:
    """Look an order up."""
    return f"order {order_id} is shipped"


def refund(order_id: int) -> str:
    """Refund an order."""
    raise ValueError("refunds are closed")


async def delete_account(ctx: Context, user: str) -> str:
    """Delete an account after the human confirms."""
    answer = await ctx.wait_for_event(
        HumanResponseEvent, waiter_event=InputRequiredEvent(prefix=f"Really delete {user}?"), waiter_id="delete"
    )
    return f"deleted {user}" if "yes" in answer.response else "kept"


class WipeAsked(InputRequiredEvent):
    pass


class WipeAnswered(Event):
    confirmed: bool


async def wipe_data(ctx: Context) -> str:
    """Wipe the data after a custom confirmation event."""
    answer = await ctx.wait_for_event(WipeAnswered, waiter_event=WipeAsked(prefix="Wipe?"), waiter_id="wipe")
    return "wiped" if answer.confirmed else "kept"


TOOLS = [lookup_order, refund, delete_account, wipe_data]


def _agent(model: Desk, **kwargs: Any) -> FunctionAgent:
    return FunctionAgent(tools=TOOLS, llm=ScriptedLLM(response_generator=model), system_prompt=SYSTEM, **kwargs)


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


async def test_reply_tool_call_memory_and_usage_in_session_mode() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(llamaindex_adapter(lambda: _agent(model)), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?; tool results: order 7 is shipped"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "order 7 is shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    # Second turn: the model is shown the first exchange, its tool call and result included.
    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: order 7 is shipped"
    assert model.roles() == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert state.tool_calls(turn=1) == []
    # Three model calls over the two turns, each with usage in its raw response.
    assert result.usage == Usage(input_tokens=30, output_tokens=12, total_tokens=42, requests=3)


async def test_a_raising_tool_is_an_error_and_the_run_goes_on() -> None:
    states: list[ScenarioState] = []

    result = await _play(llamaindex_adapter(_agent(Desk())), [sim.user("refund it"), sim.agent(), states.append])

    state = states[0]
    assert result.messages[-1]["content"] == "users: refund it; tool results: refunds are closed"
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


async def test_a_run_waiting_for_a_human_pauses_and_the_next_message_resumes_it() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("delete me"), sim.agent(), states.append, sim.user("yes"), sim.agent(), states.append]

    result = await _play(llamaindex_adapter(_agent(model)), script)

    assert result.messages[1]["content"] == "Really delete u1?"
    assert result.messages[3]["content"] == "users: delete me; tool results: deleted u1"
    state = states[-1]
    assert [(call.name, call.status, call.turn) for call in state.tool_calls(include_errors=True)] == [
        ("delete_account", "paused", 0),
        ("delete_account", "success", 1),
    ]
    assert state.tool_calls("delete_account").first.output == "deleted u1"


async def test_resume_fn_answers_with_a_custom_event() -> None:
    adapter = llamaindex_adapter(
        _agent(Desk()),
        resume_fn=lambda agent_input: WipeAnswered(confirmed="yes" in agent_input.last_new_user_message_str()),
    )

    result = await _play(adapter, [sim.user("wipe it"), sim.agent(), sim.user("yes"), sim.agent()])

    assert result.messages[1]["content"] == "Wipe?"
    assert result.messages[3]["content"] == "users: wipe it; tool results: wiped"


async def test_transcript_mode_cancels_a_waiting_run() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("delete me"), sim.agent(), sim.user("yes"), sim.agent(), states.append]

    result = await _play(llamaindex_adapter(_agent(model), history="transcript"), script)

    assert result.messages[1]["content"] == "Really delete u1?"
    # Nothing was resumed: "yes" is an ordinary message of a new run.
    assert result.messages[3]["content"] == "users: delete me | yes; tool results: none"
    assert not states[0].has_tool_call("delete_account")


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

    result = await _play(llamaindex_adapter(_agent(model), history=history), ROLLBACK_SCRIPT)

    assert result.messages[-1]["content"] == "users: hello | what do you know?; tool results: none"
    assert "secret" not in str(model.seen[-1])
    assert model.roles() == ["system", "user", "assistant", "user"]


async def test_a_rollback_past_a_waiting_run_cancels_it() -> None:
    model = Desk()

    def back_to_the_start(state: ScenarioState) -> None:
        state.rollback_messages_to(0)

    script = [sim.user("delete me"), sim.agent(), back_to_the_start, sim.user("hello"), sim.agent()]

    result = await _play(llamaindex_adapter(_agent(model)), script)

    assert result.messages[-1]["content"] == "users: hello; tool results: none"


async def test_transcript_mode_does_not_replay_tool_results() -> None:
    model = Desk()
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    result = await _play(llamaindex_adapter(_agent(model), history="transcript"), script)

    assert result.messages[3]["content"] == "users: where is my order? | thanks; tool results: none"


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    adapter = llamaindex_adapter(_agent(Desk()))

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?; tool results: none"


async def test_an_agent_workflow_shows_the_handoff_and_the_tools_of_every_agent() -> None:
    states: list[ScenarioState] = []

    def front(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatMessage:
        return _call("handoff", "call-0", to_agent="orders", reason="an order question")

    def orders(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatMessage:
        if messages[-1].role.value == "tool" and "shipped" in (messages[-1].content or ""):
            return ChatMessage(role="assistant", content=f"orders: {messages[-1].content}")
        return _call("lookup_order", "call-1", order_id=7)

    workflow = AgentWorkflow(
        agents=[
            FunctionAgent(
                name="front",
                description="Greets and routes.",
                llm=ScriptedLLM(response_generator=front),
                can_handoff_to=["orders"],
            ),
            FunctionAgent(
                name="orders",
                description="Knows the orders.",
                tools=[lookup_order],
                llm=ScriptedLLM(response_generator=orders),
            ),
        ],
        root_agent="front",
    )

    result = await _play(llamaindex_adapter(workflow), [sim.user("where is my order?"), sim.agent(), states.append])

    state = states[0]
    assert result.messages[-1]["content"] == "orders: order 7 is shipped"
    assert state.tool_call_names() == ["handoff", "lookup_order"]
    assert state.tool_calls("handoff").first.input == {"to_agent": "orders", "reason": "an order question"}
    assert result.usage is not None
    assert result.usage.requests == 3


async def test_a_react_agent() -> None:
    states: list[ScenarioState] = []
    seen: list[list[ChatMessage]] = []

    def react(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatMessage:
        seen.append(list(messages))
        last = messages[-1].content or ""
        if "order" in last and messages[-1].role.value == "user" and "Observation" not in last:
            text = 'Thought: I need a tool.\nAction: lookup_order\nAction Input: {"order_id": 7}'
        else:
            text = f"Thought: I can answer without using any more tools.\nAnswer: {len(messages) - 1} messages; {last}"
        return ChatMessage(role="assistant", content=text)

    agent = ReActAgent(tools=[lookup_order], llm=ScriptedLLM(response_generator=react))
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(llamaindex_adapter(agent, max_iterations=5), script)

    assert result.messages[1]["content"] == "3 messages; Observation: order 7 is shipped"
    assert states[0].tool_calls("lookup_order").first.output == "order 7 is shipped"
    # The second turn starts from the first exchange as the agent's memory kept it.
    assert result.messages[3]["content"] == "3 messages; thanks"
    assert "Observation: order 7 is shipped" in (seen[-1][-2].content or "")


class Summary(BaseModel):
    answer: str


async def test_content_fn_renders_the_result() -> None:
    adapter = llamaindex_adapter(
        _agent(Desk(), structured_output_fn=lambda messages: {"answer": f"{len(messages)} messages"}),
        content_fn=lambda output: Summary.model_validate(output.structured_response).answer,
    )

    result = await _play(adapter, [sim.user("hello"), sim.agent()])

    assert result.messages[-1]["content"] == "2 messages"
