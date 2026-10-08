import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from agno.agent import Agent
from agno.models.response import ToolExecution
from agno.run.base import RunStatus
from pytest_mock import MockerFixture

import simulhausen as sim
from simulhausen import AgentAdapter, AgentInput, AgentTurn, SimulationBackendError

HTML_504 = (
    "<html>\r\n<head><title>504 Gateway Time-out</title></head>\r\n"
    "<body>\r\n<center><h1>504 Gateway Time-out</h1></center>\r\n"
    "<hr><center>nginx</center>\r\n</body>\r\n</html>"
)


def _input(**kwargs: Any) -> AgentInput:
    return AgentInput(thread_id="t-1", messages=[{"role": "user", "content": "tell me about my flights"}], **kwargs)


class FakeTarget:
    """Stand-in for an agno target: replays canned responses and remembers the calls."""

    def __init__(self, *responses: object) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._responses.pop(0)


def _adapter(target: FakeTarget, **kwargs: Any) -> AgentAdapter:
    return sim.agno_adapter(cast(Agent, target), **kwargs)


async def _turn(adapter: AgentAdapter, agent_input: AgentInput | None = None) -> AgentTurn:
    turn = await adapter.call(agent_input or _input())
    assert isinstance(turn, AgentTurn)
    return turn


def _ok(content: object = "agent reply", tools: object = None, references: object = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, status="completed", tools=tools, references=references)


def _html_504() -> SimpleNamespace:
    return _ok(HTML_504)


@pytest.fixture(autouse=True)
def sleep(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch("simulhausen.adapters.asyncio.sleep", new=mocker.AsyncMock())


# -- backend failures ---------------------------------------------------------


async def test_html_error_page_raises_backend_error(sleep: AsyncMock) -> None:
    target = FakeTarget(_html_504(), _html_504(), _html_504())

    with pytest.raises(SimulationBackendError, match="gateway HTML error page"):
        await _adapter(target).call(_input())

    assert len(target.calls) == 3
    assert sleep.await_args_list[0].args == (2.0,)
    assert sleep.await_args_list[1].args == (4.0,)
    assert sleep.await_count == 2


async def test_error_message_names_the_attempts_and_the_page() -> None:
    target = FakeTarget(_html_504(), _html_504())

    with pytest.raises(SimulationBackendError, match=r"backend failed 2 time\(s\) in a row.*504 Gateway Time-out"):
        await _adapter(target, retries=1).call(_input())


async def test_backend_error_is_a_harness_error() -> None:
    with pytest.raises(sim.SimulationHarnessError):
        await _adapter(FakeTarget(_html_504()), retries=0).call(_input())


@pytest.mark.parametrize("status", [RunStatus.error, SimpleNamespace(value="error"), SimpleNamespace(name="ERROR")])
async def test_run_status_error_raises_backend_error(status: object) -> None:
    broken = SimpleNamespace(content="something", status=status, tools=None, references=None)
    target = FakeTarget(broken, broken, broken)

    with pytest.raises(SimulationBackendError, match="RunStatus=error"):
        await _adapter(target).call(_input())

    assert len(target.calls) == 3


async def test_transient_error_then_success_recovers(caplog: pytest.LogCaptureFixture) -> None:
    tool = ToolExecution(tool_name="get_user_bookings")
    target = FakeTarget(_html_504(), _ok(content="Here are your flights", tools=[tool]))

    with caplog.at_level(logging.WARNING, logger="simulhausen.adapters"):
        turn = await _turn(_adapter(target))

    assert turn.content == "Here are your flights"
    assert turn.tools == [tool]
    assert len(target.calls) == 2
    assert any("backend error (attempt 1/3)" in message for message in caplog.messages)


async def test_healthy_response_is_not_retried(sleep: AsyncMock) -> None:
    target = FakeTarget(_ok())

    turn = await _turn(_adapter(target))

    assert turn == AgentTurn(content="agent reply", tools=None, references=None)
    assert len(target.calls) == 1
    sleep.assert_not_awaited()


async def test_retries_zero_fails_on_the_first_error(sleep: AsyncMock) -> None:
    target = FakeTarget(_html_504())

    with pytest.raises(SimulationBackendError, match=r"1 time\(s\)"):
        await _adapter(target, retries=0).call(_input())

    assert len(target.calls) == 1
    sleep.assert_not_awaited()


async def test_custom_retries_and_delay(sleep: AsyncMock) -> None:
    target = FakeTarget(*[_html_504() for _ in range(4)])

    with pytest.raises(SimulationBackendError, match=r"4 time\(s\)"):
        await _adapter(target, retries=3, retry_delay=0.5).call(_input())

    assert [call.args[0] for call in sleep.await_args_list] == [0.5, 1.0, 1.5]


def test_negative_retries_are_rejected() -> None:
    with pytest.raises(ValueError, match="retries must be >= 0"):
        _adapter(FakeTarget(), retries=-1)


@pytest.mark.parametrize(
    "content",
    [
        "<!DOCTYPE html><html><body>oops</body></html>",
        "  \n<html><body>oops</body></html>",
        "<head><title>x</title></head>",
        "504 Gateway Time-out",
        "nginx: 502 Bad Gateway",
        "503 Service Unavailable",
        "The Gateway Timeout page",
    ],
)
async def test_gateway_error_pages_are_recognised(content: str) -> None:
    with pytest.raises(SimulationBackendError):
        await _adapter(FakeTarget(_ok(content)), retries=0).call(_input())


@pytest.mark.parametrize(
    "content",
    [
        "To embed it, put <html> in the file",
        "Everything is fine. " * 20 + "bad gateway",
        "",
    ],
)
async def test_ordinary_replies_are_not_mistaken_for_error_pages(content: str) -> None:
    turn = await _turn(_adapter(FakeTarget(_ok(content)), retries=0))

    assert turn.content == content


@pytest.mark.parametrize("error_type", ["input_check_error", "output_check_error"])
async def test_guardrail_refusal_is_returned_as_the_answer(error_type: str) -> None:
    """A check refusing the request is the product's answer, not a backend failure."""
    refusal = "Sorry, I can't help with that request."
    response = SimpleNamespace(
        content=refusal,
        status=RunStatus.error,
        tools=None,
        references=None,
        events=[SimpleNamespace(error_type=error_type)],
    )
    target = FakeTarget(response)

    turn = await _turn(_adapter(target))

    assert turn.content == refusal
    assert len(target.calls) == 1


async def test_unrelated_error_events_do_not_excuse_a_failed_run() -> None:
    response = SimpleNamespace(
        content="x", status=RunStatus.error, tools=None, references=None, events=[SimpleNamespace(error_type="other")]
    )

    with pytest.raises(SimulationBackendError):
        await _adapter(FakeTarget(response), retries=0).call(_input())


# -- turns --------------------------------------------------------------------


async def test_member_tool_calls_are_merged_with_the_leader_tools() -> None:
    delegate, lookup = ToolExecution(tool_name="delegate_task_to_member"), ToolExecution(tool_name="get_flight_status")
    response = SimpleNamespace(
        content="Flight 1006 is on time",
        status="completed",
        tools=[delegate],
        references=None,
        member_responses=[SimpleNamespace(tools=[lookup]), SimpleNamespace(tools=None)],
    )

    turn = await _turn(_adapter(FakeTarget(response)))

    assert turn.tools == [delegate, lookup]


async def test_member_tools_without_leader_tools() -> None:
    lookup = ToolExecution(tool_name="get_flight_status")
    response = _ok()
    response.member_responses = [SimpleNamespace(tools=[lookup])]

    assert (await _turn(_adapter(FakeTarget(response)))).tools == [lookup]


async def test_no_tools_anywhere_gives_none() -> None:
    response = _ok(tools=[])
    response.member_responses = []

    assert (await _turn(_adapter(FakeTarget(response)))).tools is None


async def test_references_are_passed_through() -> None:
    references = [SimpleNamespace(query="q")]

    turn = await _turn(_adapter(FakeTarget(_ok(references=references))))

    assert turn.references == references


async def test_content_fn_renders_structured_output() -> None:
    reply = SimpleNamespace(answer="done")

    turn = await _turn(_adapter(FakeTarget(_ok(content=reply)), content_fn=lambda raw: raw.answer))

    assert turn.content == "done"


@pytest.mark.parametrize(("raw", "expected"), [(None, ""), (42, "42"), ("", "")])
async def test_content_without_content_fn_goes_through_str(raw: object, expected: str) -> None:
    assert (await _turn(_adapter(FakeTarget(_ok(content=raw))))).content == expected


# -- calling the target -------------------------------------------------------


async def test_arun_gets_the_last_new_message_the_thread_id_and_extra_kwargs() -> None:
    target = FakeTarget(_ok())
    adapter = _adapter(target, session_state={"plan": "pro"}, user_id="u-7")
    agent_input = AgentInput(
        thread_id="thread-42",
        messages=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ],
        new_messages=[{"role": "user", "content": "new question"}],
    )

    await adapter.call(agent_input)

    assert target.calls == [
        (("new question",), {"session_id": "thread-42", "session_state": {"plan": "pro"}, "user_id": "u-7"})
    ]


async def test_lazy_factory_is_resolved_once() -> None:
    built: list[FakeTarget] = []

    def factory() -> FakeTarget:
        built.append(FakeTarget(_ok("first"), _ok("second")))
        return built[-1]

    adapter = sim.agno_adapter(cast("Any", factory))
    assert built == []

    assert (await _turn(adapter)).content == "first"
    assert (await _turn(adapter)).content == "second"
    assert len(built) == 1
    assert len(built[0].calls) == 2


async def test_target_with_arun_is_not_called_as_a_factory() -> None:
    class CallableTarget(FakeTarget):
        def __call__(self) -> None:
            raise AssertionError("a target with arun must be used as is")

    assert (await _turn(_adapter(CallableTarget(_ok("ok"))))).content == "ok"


# -- function_adapter ---------------------------------------------------------


async def test_function_adapter_with_a_sync_function() -> None:
    adapter = sim.function_adapter(lambda agent_input: f"echo: {agent_input.last_new_user_message_str()}")

    assert await adapter.call(_input()) == "echo: tell me about my flights"
    assert adapter.role is sim.AgentRole.AGENT


async def test_function_adapter_with_an_async_function() -> None:
    tool = ToolExecution(tool_name="lookup")

    async def agent(agent_input: AgentInput) -> AgentTurn:
        return AgentTurn(content=agent_input.thread_id, tools=[tool])

    assert await sim.function_adapter(agent).call(_input()) == AgentTurn(content="t-1", tools=[tool])
