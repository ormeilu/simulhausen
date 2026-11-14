"""``openai_agents_adapter`` against the real OpenAI Agents SDK, with its ``ScriptedModel`` in place of a provider.

The SDK is not a dependency of the project, so the module is skipped in a
plain test run. Run it with the library overlaid:

    uv run --no-group harbor --with openai-agents pytest tests/contract/test_openai_agents_contract.py --no-cov -q
"""

import json
from typing import Any

import pytest

pytest.importorskip("agents")

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunConfig,
    SQLiteSession,
    function_tool,
    input_guardrail,
    output_guardrail,
)
from agents.testing import ModelCall, ModelStep, ScriptedModel, assistant_message, function_call
from agents.usage import Usage as SdkUsage
from pydantic import BaseModel

import simulhausen as sim
from simulhausen import ScenarioState
from simulhausen.frameworks.openai_agents import openai_agents_adapter

# No trace export: the tests have no API key and must not reach the network.
QUIET = RunConfig(tracing_disabled=True)


def _user_texts(call: ModelCall) -> list[str]:
    return [item["content"] for item in call.input if isinstance(item, dict) and item.get("role") == "user"]


def _tool_outputs(call: ModelCall) -> list[str]:
    return [item["output"] for item in call.input if isinstance(item, dict) and "output" in item]


def _respond(call: ModelCall) -> ModelStep:
    """Calls a tool when the newest user message names one, otherwise reports what the model was shown."""
    last = call.input[-1]
    prompt = last.get("content", "") if isinstance(last, dict) and last.get("role") == "user" else ""
    if "order" in prompt:
        output = [function_call("lookup_order", {"order_id": 7}, call_id="call-1")]
    elif "refund" in prompt:
        output = [function_call("refund", {"order_id": 7}, call_id="call-2")]
    elif "billing" in prompt:
        output = [function_call("transfer_to_billing", {}, call_id="call-3")]
    elif "delete" in prompt:
        output = [function_call("delete_account", {"user": "u1"}, call_id="call-4")]
    elif "summary" in prompt:
        output = [assistant_message(json.dumps({"answer": "all good"}))]
    else:
        users = " | ".join(_user_texts(call))
        outputs = " | ".join(_tool_outputs(call)) or "none"
        output = [assistant_message(f"{call.system_instructions}; users: {users}; tool results: {outputs}")]
    return ModelStep(output=output, usage=SdkUsage(requests=1, input_tokens=10, output_tokens=4, total_tokens=14))


def _model() -> ScriptedModel:
    return ScriptedModel([ModelStep.respond(_respond) for _ in range(12)])


@function_tool
def lookup_order(order_id: int) -> dict[str, Any]:
    return {"order_id": order_id, "status": "shipped"}


@function_tool
def refund(order_id: int) -> str:
    raise RuntimeError("refunds are closed")


@function_tool
def invoice(order_id: int) -> str:
    return f"invoice for {order_id}"


@function_tool(needs_approval=True)
def delete_account(user: str) -> str:
    raise AssertionError("must not run without approval")


def _desk(model: ScriptedModel, **kwargs: Any) -> Agent[Any]:
    billing = Agent(name="billing", instructions="billing desk", model=model, tools=[lookup_order, invoice])
    return Agent(
        name="triage",
        instructions="order desk",
        model=model,
        tools=[lookup_order, refund, delete_account],
        handoffs=[billing],
        **kwargs,
    )


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


SHIPPED = str({"order_id": 7, "status": "shipped"})


async def test_reply_tool_call_memory_and_usage_in_session_mode() -> None:
    model = _model()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(openai_agents_adapter(lambda: _desk(model), run_config=QUIET), script)

    state = states[0]
    assert result.messages[1]["content"] == f"order desk; users: where is my order?; tool results: {SHIPPED}"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert call.input == {"order_id": 7}
    assert call.output == '{"order_id": 7, "status": "shipped"}'
    assert call.execution is not None
    assert call.execution.tool_call_id == "call-1"
    # Second turn: the model is shown the first exchange, its tool result included, and no new tool call is recorded.
    assert result.messages[3]["content"] == f"order desk; users: where is my order? | thanks; tool results: {SHIPPED}"
    assert state.tool_calls(turn=1) == []
    assert result.usage == sim.Usage(input_tokens=30, output_tokens=12, total_tokens=42, requests=3)


async def test_a_raising_tool_is_an_error_and_a_user_formatter_still_writes_the_message() -> None:
    states: list[ScenarioState] = []
    plain = openai_agents_adapter(_desk(_model()), run_config=QUIET)
    config = RunConfig(
        tracing_disabled=True, tool_error_formatter=lambda args: f"{args.tool_name} failed: {args.error}"
    )
    formatted = openai_agents_adapter(_desk(_model()), run_config=config)
    script = [sim.user("refund it"), sim.agent(), states.append]

    await _play(plain, script)
    await _play(formatted, script)

    messages = ("An error occurred while running the tool. Please try again.", "refund failed: refunds are closed")
    for state, message in zip(states, messages, strict=True):
        assert not state.has_tool_call("refund")
        assert state.tool_calls() == []
        (failed,) = state.tool_calls(include_errors=True)
        assert (failed.name, failed.input, failed.output, failed.status) == (
            "refund",
            {"order_id": 7},
            message,
            "error",
        )
        assert failed.error


async def test_a_handoff_is_a_tool_call_and_the_target_agent_keeps_the_thread() -> None:
    model = _model()
    states: list[ScenarioState] = []
    script = [sim.user("billing please"), sim.agent(), sim.user("where is my order?"), sim.agent(), states.append]

    result = await _play(openai_agents_adapter(_desk(model), run_config=QUIET), script)

    state = states[0]
    assert result.messages[1]["content"].startswith("billing desk; users: billing please")
    assert state.has_tool_calls_in_order(["transfer_to_billing", "lookup_order"])
    assert state.tool_calls("transfer_to_billing").first.output == '{"assistant": "billing"}'
    # The second turn started from the billing agent, and its tool call is the handoff target's.
    assert result.messages[3]["content"].startswith("billing desk; users: billing please | where is my order?")
    assert state.tool_calls("lookup_order").first.turn == 1


async def test_every_turn_starts_from_the_target_without_resume_last_agent() -> None:
    adapter = openai_agents_adapter(_desk(_model()), run_config=QUIET, resume_last_agent=False)

    result = await _play(adapter, [sim.user("billing please"), sim.agent(), sim.user("hello"), sim.agent()])

    assert result.messages[1]["content"].startswith("billing desk;")
    assert result.messages[3]["content"].startswith("order desk; users: billing please | hello")


async def test_a_call_waiting_for_approval_is_not_executed() -> None:
    states: list[ScenarioState] = []
    script = [sim.user("delete my account"), sim.agent(), sim.user("hello"), sim.agent(), states.append]

    result = await _play(openai_agents_adapter(_desk(_model()), run_config=QUIET), script)

    state = states[0]
    assert result.messages[1]["content"] == ""
    assert not state.has_tool_call("delete_account")
    (pending,) = state.tool_calls(include_errors=True)
    assert (pending.name, pending.input, pending.status) == ("delete_account", {"user": "u1"}, "paused")
    # The interrupted run is not carried on: the next turn is rebuilt from the transcript.
    assert result.messages[3]["content"] == "order desk; users: delete my account | hello; tool results: none"


class Summary(BaseModel):
    answer: str


async def test_structured_output_is_rendered() -> None:
    def desk() -> Agent[Any]:
        return Agent(name="summary", model=_model(), output_type=Summary)

    plain = await _play(openai_agents_adapter(desk, run_config=QUIET), [sim.user("summary please"), sim.agent()])
    custom = await _play(
        openai_agents_adapter(desk, run_config=QUIET, content_fn=lambda output: output.answer),
        [sim.user("summary please"), sim.agent()],
    )

    assert plain.messages[-1]["content"] == '{"answer":"all good"}'
    assert custom.messages[-1]["content"] == "all good"


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


@pytest.mark.parametrize("options", [{"history": "transcript"}, {}, {"session": SQLiteSession}])
async def test_a_rollback_reaches_the_agent(options: dict[str, Any]) -> None:
    model = _model()

    result = await _play(openai_agents_adapter(_desk(model), run_config=QUIET, **options), ROLLBACK_SCRIPT)

    assert result.messages[-1]["content"] == "order desk; users: hello | what do you know?; tool results: none"
    last_call = model.last_call
    assert last_call is not None
    assert "secret" not in json.dumps(last_call.input, default=str)


async def test_transcript_mode_does_not_replay_tool_results() -> None:
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    result = await _play(openai_agents_adapter(_desk(_model()), run_config=QUIET, history="transcript"), script)

    assert result.messages[3]["content"] == "order desk; users: where is my order? | thanks; tool results: none"


async def test_an_sdk_session_keeps_the_memory_per_thread() -> None:
    model = _model()
    sessions: list[SQLiteSession] = []

    def session(thread_id: str) -> SQLiteSession:
        sessions.append(SQLiteSession(thread_id))
        return sessions[-1]

    adapter = openai_agents_adapter(_desk(model), run_config=QUIET, session=session)
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    first = await _play(adapter, script)
    second = await _play(adapter, [sim.user("hello"), sim.agent()])

    assert first.messages[3]["content"] == f"order desk; users: where is my order? | thanks; tool results: {SHIPPED}"
    assert second.messages[1]["content"] == "order desk; users: hello; tool results: none"
    assert len(sessions) == 2
    assert sessions[0].session_id != sessions[1].session_id
    # Each turn sent only the new user message; the rest came from the session.
    assert len(await sessions[0].get_items()) == 6


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    adapter = openai_agents_adapter(_desk(_model()), run_config=QUIET)

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "order desk; users: what do you know?; tool results: none"


@input_guardrail
async def no_secrets(context: Any, agent: Any, user_input: Any) -> GuardrailFunctionOutput:
    newest = user_input if isinstance(user_input, str) else user_input[-1]
    return GuardrailFunctionOutput(output_info="secret", tripwire_triggered="secret" in json.dumps(newest))


@output_guardrail
async def no_orders(context: Any, agent: Any, output: Any) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info="order data", tripwire_triggered="shipped" in str(output))


async def test_a_tripped_guardrail_propagates_by_default() -> None:
    adapter = openai_agents_adapter(_desk(_model(), input_guardrails=[no_secrets]), run_config=QUIET)

    with pytest.raises(InputGuardrailTripwireTriggered):
        await _play(adapter, [sim.user("my secret is 42"), sim.agent()])


async def test_a_tripped_guardrail_becomes_the_reply_when_asked() -> None:
    states: list[ScenarioState] = []
    blocked_input = openai_agents_adapter(
        _desk(_model(), input_guardrails=[no_secrets]), run_config=QUIET, guardrail_reply="I cannot help with that."
    )
    blocked_output = openai_agents_adapter(
        _desk(_model(), output_guardrails=[no_orders]),
        run_config=QUIET,
        guardrail_reply=lambda exc: f"Blocked: {exc.guardrail_result.output.output_info}.",
    )

    first = await _play(blocked_input, [sim.user("my secret is 42"), sim.agent(), sim.user("hello"), sim.agent()])
    second = await _play(blocked_output, [sim.user("where is my order?"), sim.agent(), states.append])

    assert first.messages[1]["content"] == "I cannot help with that."
    # The thread goes on from the transcript, refusal included.
    assert first.messages[3]["content"] == "order desk; users: my secret is 42 | hello; tool results: none"
    assert second.messages[1]["content"] == "Blocked: order data."
    # The run that the output guardrail stopped had already called the tool and the model twice.
    assert states[0].has_tool_call("lookup_order")
    assert second.usage is not None
    assert second.usage.requests == 2
