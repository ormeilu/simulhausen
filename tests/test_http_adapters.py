"""HTTP adapters against httpx.MockTransport: no network, no model.

The AgentOS fixtures under tests/data/agentos are real agno 3.1.1 responses,
captured from an AgentOS app driven by a scripted fake model.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx
import pytest
from pytest_mock import MockerFixture

import simulhausen as sim
from simulhausen import (
    AgentAdapter,
    AgentInput,
    AgentTurn,
    CriterionVerdict,
    JudgeAgent,
    JudgeVerdict,
    Message,
    ScenarioState,
    ScenarioStatus,
    SimulationBackendError,
)
from simulhausen.http_adapters import _parse_sse, _SSEEvent

DATA = Path(__file__).parent / "data" / "agentos"
BASE = "http://agent.test"

type Reply = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


def _fixture_json(name: str) -> dict[str, Any]:
    return json.loads((DATA / f"{name}.json").read_text())


def _fixture_sse(name: str) -> str:
    # end-of-file-fixer keeps a single trailing newline; on the wire every event ends with a blank line.
    return (DATA / f"{name}.sse").read_text() + "\n"


def _sse_response(text: str) -> httpx.Response:
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


def _sse(*events: dict[str, Any]) -> str:
    return "".join(f"event: {event['event']}\ndata: {json.dumps(event)}\n\n" for event in events)


class Server:
    """A MockTransport handler: answers from a queue (the last answer repeats) and records the requests."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, httpx.Response):
            return reply
        if isinstance(reply, Exception):
            raise reply
        return reply(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def form(self, index: int = -1) -> dict[str, str]:
        return {key: values[0] for key, values in parse_qs(self.requests[index].content.decode()).items()}

    def json(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)


def _input(*contents: str, thread_id: str = "t-1", new: int | None = None) -> AgentInput:
    """Alternating user/assistant messages; ``new`` = how many trailing messages are new (default: the last)."""
    messages: list[Message] = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": content}
        for index, content in enumerate(contents)
    ]
    count = 1 if new is None else new
    return AgentInput(thread_id=thread_id, messages=messages, new_messages=messages[len(messages) - count :])


async def _turn(adapter: AgentAdapter, agent_input: AgentInput | None = None) -> AgentTurn:
    turn = await adapter.call(agent_input or _input("where is my order 1042?"))
    assert isinstance(turn, AgentTurn)
    return turn


def _state(turn: AgentTurn) -> ScenarioState:
    state = ScenarioState(thread_id="t-1", name="n", description="d")
    state.add_agent_message(turn.content, tools=turn.tools, references=turn.references)
    return state


def _tool_names(turn: AgentTurn) -> list[str | None]:
    return [tool.tool_name for tool in turn.tools or []]


@pytest.fixture(autouse=True)
def sleep(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch("simulhausen.http_adapters.asyncio.sleep", new=mocker.AsyncMock())


# -- server-sent events -----------------------------------------------------------


def test_sse_events_with_names_multiline_data_and_comments() -> None:
    text = ": keep-alive\nevent: first\ndata: line one\ndata: line two\n\ndata: {}\n\n"

    assert list(_parse_sse(text)) == [_SSEEvent("first", "line one\nline two"), _SSEEvent("message", "{}")]


@pytest.mark.parametrize("newline", ["\r\n", "\r", "\n"])
def test_sse_line_endings(newline: str) -> None:
    text = newline.join(["event: a", "data: 1", "", "event: b", "data: 2", "", ""])

    assert list(_parse_sse(text)) == [_SSEEvent("a", "1"), _SSEEvent("b", "2")]


def test_sse_field_details() -> None:
    text = (
        "\ufeffevent: a\n"
        "data:no space\n"
        "data:  two spaces\n"
        "data\n"  # a field without a colon: empty value
        "id: 7\n"
        "retry: 1000\n"
        "unknown: x\n"
        "\n"
        "event: only-a-name\n"  # no data: nothing to dispatch, the name is dropped
        "\n"
        "data: after\n"
        "\n"
    )

    assert list(_parse_sse(text)) == [_SSEEvent("a", "no space\n two spaces\n"), _SSEEvent("message", "after")]


def test_sse_keeps_unicode_line_separators_inside_data() -> None:
    payload = json.dumps({"content": "a\u2028b\x85c"}, ensure_ascii=False)

    (event,) = _parse_sse(f"data: {payload}\n\n")

    assert json.loads(event.data) == {"content": "a\u2028b\x85c"}


def test_sse_dispatches_the_last_event_without_a_blank_line() -> None:
    assert list(_parse_sse("event: done\ndata: x")) == [_SSEEvent("done", "x")]


def test_sse_empty_stream() -> None:
    assert list(_parse_sse("")) == []


# -- argument validation --------------------------------------------------------------

FACTORIES: dict[str, Callable[..., AgentAdapter]] = {
    "agentos": lambda **kw: sim.agentos_adapter(kw.pop("url", BASE), agent_id="support", **kw),
    "openai_chat": lambda **kw: sim.openai_chat_adapter(kw.pop("url", BASE), model="m", **kw),
    "openai_responses": lambda **kw: sim.openai_responses_adapter(kw.pop("url", BASE), model="m", **kw),
    "a2a": lambda **kw: sim.a2a_adapter(kw.pop("url", BASE), **kw),
    "langgraph": lambda **kw: sim.langgraph_adapter(kw.pop("url", BASE), assistant_id="agent", **kw),
}


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES.keys())
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"retries": -1}, "retries must be >= 0"),
        ({"retry_delay": -0.5}, "retry_delay must be >= 0"),
        ({"timeout": 0}, "timeout must be > 0"),
        ({"url": "agent.test:7777"}, "must be an http:// or https:// URL"),
    ],
)
def test_common_arguments_are_checked(factory: Callable[..., AgentAdapter], kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES.keys())
async def test_client_and_transport_are_exclusive(factory: Callable[..., AgentAdapter]) -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="pass client or transport, not both"):
            factory(client=client, transport=httpx.MockTransport(lambda request: httpx.Response(200)))


@pytest.mark.parametrize("name", ["agentos", "openai_responses", "a2a", "langgraph"])
def test_unknown_history_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="history must be"):
        FACTORIES[name](history="memory")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one of agent_id, team_id, workflow_id, got 0$"),
        ({"agent_id": "a", "team_id": "t"}, r"got 2 \(agent_id, team_id\)"),
        ({"agent_id": "a", "team_id": "t", "workflow_id": "w"}, "got 3"),
        ({"workflow_id": ""}, "workflow_id must not be empty"),
        ({"agent_id": "a", "session_id": "s"}, "form fields must not set session_id"),
        ({"agent_id": "a", "message": "m", "session_id": "s"}, "form fields must not set message, session_id"),
    ],
)
def test_agentos_arguments(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        sim.agentos_adapter(BASE, **kwargs)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: sim.openai_chat_adapter(BASE, model=""), "model must not be empty"),
        (lambda: sim.openai_chat_adapter(BASE, model="m", extra_body={"messages": []}), "must not set messages"),
        (lambda: sim.openai_responses_adapter(BASE, model=""), "model must not be empty"),
        (lambda: sim.openai_responses_adapter(BASE, model="m", extra_body={"store": True}), "must not set store"),
        (lambda: sim.a2a_adapter(BASE, protocol_version=cast(Any, "2.0")), "protocol_version must be"),
        (lambda: sim.langgraph_adapter(BASE, assistant_id=""), "assistant_id must not be empty"),
        (lambda: sim.langgraph_adapter(BASE, assistant_id="a", input_key=""), "input_key must not be empty"),
        (lambda: sim.langgraph_adapter(BASE, assistant_id="a", extra_body={"input": {}}), "must not set input"),
    ],
)
def test_protocol_specific_arguments(factory: Callable[[], AgentAdapter], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_first_argument_works_as_a_keyword() -> None:
    """The Harbor config layer calls every factory with **config."""
    for adapter in (
        sim.agentos_adapter(base_url=BASE, agent_id="a"),
        sim.openai_chat_adapter(base_url=BASE, model="m"),
        sim.openai_responses_adapter(base_url=BASE, model="m"),
        sim.a2a_adapter(url=BASE),
        sim.langgraph_adapter(base_url=BASE, assistant_id="a"),
    ):
        assert isinstance(adapter, AgentAdapter)


# -- retries and HTTP errors (shared base, driven through the chat adapter) -------------


def _chat_ok(content: str = "ok") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def _chat(server: Server, **kwargs: Any) -> AgentAdapter:
    return sim.openai_chat_adapter(f"{BASE}/v1", model="m", transport=server.transport, **kwargs)


async def test_5xx_then_success_recovers(sleep: AsyncMock, caplog: pytest.LogCaptureFixture) -> None:
    server = Server(httpx.Response(503, text="<html>Service Unavailable</html>"), _chat_ok("recovered"))

    with caplog.at_level(logging.WARNING, logger="simulhausen.http_adapters"):
        turn = await _turn(_chat(server))

    assert turn.content == "recovered"
    assert len(server.requests) == 2
    assert [call.args for call in sleep.await_args_list] == [(2.0,)]
    assert any("backend error (attempt 1/3): HTTP 503" in message for message in caplog.messages)


async def test_persistent_5xx_raises_backend_error_after_the_retries(sleep: AsyncMock) -> None:
    server = Server(httpx.Response(500, text="boom"))

    with pytest.raises(SimulationBackendError, match=r"backend failed 3 time\(s\) in a row: HTTP 500 from POST .*boom"):
        await _chat(server).call(_input("hi"))

    assert len(server.requests) == 3
    assert [call.args[0] for call in sleep.await_args_list] == [2.0, 4.0]


async def test_backend_error_is_a_harness_error() -> None:
    with pytest.raises(sim.SimulationHarnessError):
        await _chat(Server(httpx.Response(502)), retries=0).call(_input("hi"))


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "3"}, 3.0),
        ({"Retry-After": "0.5"}, 0.5),
        ({"Retry-After": "120"}, 30.0),
        ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 1.5),
        ({"Retry-After": "nan"}, 1.5),
        ({"Retry-After": "-4"}, 1.5),
        ({}, 1.5),
    ],
)
async def test_429_honours_a_short_retry_after(sleep: AsyncMock, headers: dict[str, str], expected: float) -> None:
    server = Server(httpx.Response(429, headers=headers, text="slow down"), _chat_ok())

    await _turn(_chat(server, retry_delay=1.5))

    assert [call.args[0] for call in sleep.await_args_list] == [expected]


async def test_408_is_retried() -> None:
    server = Server(httpx.Response(408), _chat_ok())

    assert (await _turn(_chat(server))).content == "ok"
    assert len(server.requests) == 2


@pytest.mark.parametrize(
    ("error", "name"),
    [
        (httpx.ConnectError("connection refused"), "ConnectError"),
        (httpx.ReadTimeout("timed out"), "ReadTimeout"),
        (httpx.RemoteProtocolError("peer closed connection"), "RemoteProtocolError"),
    ],
)
async def test_transport_errors_are_retried_then_raised(error: Exception, name: str) -> None:
    server = Server(error)

    with pytest.raises(
        SimulationBackendError, match=rf"3 time\(s\) in a row: {name} on POST {BASE}/v1/chat/completions"
    ):
        await _chat(server).call(_input("hi"))

    assert len(server.requests) == 3


async def test_transport_error_then_success_recovers() -> None:
    server = Server(httpx.ConnectTimeout("connect timeout"), _chat_ok("up"))

    assert (await _turn(_chat(server))).content == "up"


async def test_transport_error_without_a_message() -> None:
    with pytest.raises(SimulationBackendError, match=r"ReadTimeout on POST .*: no details"):
        await _chat(Server(httpx.ReadTimeout("")), retries=0).call(_input("hi"))


@pytest.mark.parametrize(
    ("status", "hint"),
    [
        (400, "check the adapter arguments"),
        (401, "check api_key or the auth headers"),
        (403, "what the caller may access"),
        (404, "check the URL and the agent/assistant id"),
        (405, "check the URL"),
        (422, "the server rejected the request"),
        (418, "check the URL and the adapter arguments"),
        (302, "check the URL and the adapter arguments"),
    ],
)
async def test_other_client_errors_fail_at_once(sleep: AsyncMock, status: int, hint: str) -> None:
    server = Server(httpx.Response(status, json={"detail": "Invalid authentication token"}))

    with pytest.raises(
        SimulationBackendError, match=rf"HTTP {status} from POST .*Invalid authentication token.*{hint}"
    ):
        await _chat(server).call(_input("hi"))

    assert len(server.requests) == 1
    sleep.assert_not_awaited()


async def test_error_body_excerpt_is_capped() -> None:
    with pytest.raises(SimulationBackendError) as excinfo:
        await _chat(Server(httpx.Response(404, text="x " * 400))).call(_input("hi"))

    assert "x x x" in str(excinfo.value)
    assert "..." in str(excinfo.value)
    assert len(str(excinfo.value)) < 450


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(200, text="<html>Bad Gateway</html>"), "non-JSON response from"),
        (httpx.Response(200, json=["not", "an", "object"]), "expected a JSON object"),
    ],
)
async def test_unusable_200_bodies_are_retried(response: httpx.Response, match: str) -> None:
    server = Server(response)

    with pytest.raises(SimulationBackendError, match=match):
        await _chat(server).call(_input("hi"))

    assert len(server.requests) == 3


async def test_retries_zero_means_one_attempt(sleep: AsyncMock) -> None:
    server = Server(httpx.Response(503))

    with pytest.raises(SimulationBackendError, match=r"1 time\(s\)"):
        await _chat(server, retries=0).call(_input("hi"))

    assert len(server.requests) == 1
    sleep.assert_not_awaited()


# -- clients, transports, timeouts, headers ----------------------------------------------


async def test_injected_client_is_used_and_not_closed() -> None:
    server = Server(_chat_ok())
    async with httpx.AsyncClient(transport=server.transport, headers={"X-Client": "mine"}) as client:
        adapter = sim.openai_chat_adapter(f"{BASE}/v1", model="m", api_key="k-1", client=client)

        await _turn(adapter)
        await _turn(adapter)

        assert not client.is_closed
    assert [request.headers["x-client"] for request in server.requests] == ["mine", "mine"]
    assert server.requests[0].headers["authorization"] == "Bearer k-1"


async def test_passed_transport_is_not_closed_between_calls() -> None:
    class RecordingTransport(httpx.MockTransport):
        closed = 0

        async def aclose(self) -> None:
            RecordingTransport.closed += 1

    server = Server(_chat_ok())
    adapter = sim.openai_chat_adapter(f"{BASE}/v1", model="m", transport=RecordingTransport(server))

    await _turn(adapter)
    await _turn(adapter)

    assert len(server.requests) == 2
    assert RecordingTransport.closed == 0


def test_a_fresh_client_per_call_works_across_event_loops(mocker: MockerFixture) -> None:
    """pytest-asyncio gives every test its own loop: a client cached from the first must not be reused."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        loops.append(asyncio.get_running_loop())
        return _chat_ok()

    clients: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient

    def recording_client(**kwargs: Any) -> httpx.AsyncClient:
        clients.append(real_client(**kwargs))
        return clients[-1]

    mocker.patch("simulhausen.http_adapters.httpx.AsyncClient", side_effect=recording_client)
    adapter = sim.openai_chat_adapter(f"{BASE}/v1", model="m", transport=httpx.MockTransport(handler))

    first = asyncio.run(adapter.call(_input("hi")))
    second = asyncio.run(adapter.call(_input("hi")))

    assert first == second == AgentTurn(content="ok")
    assert loops[0] is not loops[1]
    assert len(clients) == 2
    assert all(client.is_closed for client in clients)


async def test_own_clients_follow_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(307, headers={"Location": f"{BASE}/v2/chat/completions"})
        return _chat_ok("moved")

    adapter = sim.openai_chat_adapter(f"{BASE}/v1", model="m", transport=httpx.MockTransport(handler))

    assert (await _turn(adapter)).content == "moved"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, {"connect": 10.0, "read": 120.0, "write": 120.0, "pool": 120.0}),
        (5, {"connect": 5, "read": 5, "write": 5, "pool": 5}),
    ],
)
async def test_timeout_per_request(seconds: float | None, expected: dict[str, float]) -> None:
    server = Server(_chat_ok())
    kwargs = {} if seconds is None else {"timeout": seconds}

    await _turn(_chat(server, **kwargs))

    assert server.requests[0].extensions["timeout"] == expected


async def test_caller_headers_override_the_adapter_headers() -> None:
    server = Server(_chat_ok())
    adapter = _chat(server, api_key="k-1", headers={"Authorization": "Custom abc", "Accept": "*/*", "X-Extra": "1"})

    await _turn(adapter)

    headers = server.requests[0].headers
    assert (headers["authorization"], headers["accept"], headers["x-extra"]) == ("Custom abc", "*/*", "1")


async def test_no_auth_header_without_api_key() -> None:
    server = Server(_chat_ok())

    await _turn(_chat(server))

    assert "authorization" not in server.requests[0].headers


# -- agno AgentOS ---------------------------------------------------------------------


def _agentos(server: Server, **kwargs: Any) -> AgentAdapter:
    kwargs.setdefault("agent_id", "support")
    return sim.agentos_adapter(BASE, transport=server.transport, **kwargs)


async def test_agentos_request_shape() -> None:
    server = Server(_sse_response(_fixture_sse("run_stream")))
    adapter = _agentos(
        server,
        api_key="os-key",
        user_id="u-42",
        session_state={"plan": "pro"},
        dependencies={"region": "eu"},
        add_history_to_context=True,
        version=2,
        knowledge_filters=None,
    )

    await _turn(adapter, _input("old question", "old answer", "where is my order?", thread_id="thread-9"))

    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE}/agents/support/runs"
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert request.headers["authorization"] == "Bearer os-key"
    assert request.headers["accept"] == "text/event-stream"
    assert server.form() == {
        "message": "where is my order?",
        "stream": "true",
        "session_id": "thread-9",
        "user_id": "u-42",
        "session_state": '{"plan": "pro"}',
        "dependencies": '{"region": "eu"}',
        "add_history_to_context": "true",
        "version": "2",
    }


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"team_id": "desk"}, "/teams/desk/runs"),
        ({"workflow_id": "flow"}, "/workflows/flow/runs"),
        ({"agent_id": "a b/c"}, "/agents/a%20b%2Fc/runs"),
    ],
)
async def test_agentos_paths(kwargs: dict[str, Any], path: str) -> None:
    server = Server(httpx.Response(200, json={"content": "ok", "status": "COMPLETED"}))

    await _turn(sim.agentos_adapter(f"{BASE}/", transport=server.transport, **kwargs))

    assert server.requests[0].url.raw_path.decode() == path


async def test_agentos_form_values() -> None:
    server = Server(httpx.Response(200, json={"content": "ok"}))
    adapter = _agentos(server, metadata={"source": "sim"}, images=[{"url": "u"}], flag=False, ratio=0.5)

    await _turn(adapter)

    form = server.form()
    assert (form["metadata"], form["images"], form["flag"], form["ratio"]) == (
        '{"source": "sim"}',
        '[{"url": "u"}]',
        "false",
        "0.5",
    )


async def test_agentos_session_history_keeps_one_session_per_thread() -> None:
    server = Server(_sse_response(_fixture_sse("run_stream")))
    adapter = _agentos(server)

    await _turn(adapter, _input("first", thread_id="thread-1"))
    await _turn(adapter, _input("first", "reply", "second", thread_id="thread-1"))

    assert [(server.form(i)["message"], server.form(i)["session_id"]) for i in range(2)] == [
        ("first", "thread-1"),
        ("second", "thread-1"),
    ]


async def test_agentos_transcript_history_renders_the_dialogue_in_fresh_sessions() -> None:
    server = Server(httpx.Response(200, json={"content": "ok", "status": "COMPLETED"}))
    adapter = _agentos(server, history="transcript")

    await _turn(adapter, _input("hi", thread_id="thread-1"))
    await _turn(adapter, _input("hi", "hello", "refund please", thread_id="thread-1"))

    first, second = server.form(0), server.form(1)
    assert first["message"] == "hi"
    assert second["message"] == (
        "Conversation so far:\nUser: hi\nAssistant: hello\nUser: refund please\n\nReply to the last user message."
    )
    assert first["session_id"].startswith("thread-1-")
    assert second["session_id"].startswith("thread-1-")
    assert first["session_id"] != second["session_id"]


async def test_agentos_stream_agent_run() -> None:
    turn = await _turn(_agentos(Server(_sse_response(_fixture_sse("run_stream")))))

    assert turn.content == 'Your order status: {"order_id": "1042", "status": "shipped"}'
    assert turn.tools is not None
    (tool,) = turn.tools
    assert (tool.tool_name, tool.tool_args, tool.result) == (
        "lookup_order",
        {"order_id": "1042"},
        '{"order_id": "1042", "status": "shipped"}',
    )
    assert _state(turn).has_tool_call("lookup_order")
    assert turn.references is not None
    assert turn.references[0].query == "Where is my order?"
    assert _state(turn).references_text() == "Refunds take 5 days.\nSource URL: https://kb.example/refunds"


async def test_agentos_stream_team_run_waits_for_the_team_completion() -> None:
    turn = await _turn(_agentos(Server(_sse_response(_fixture_sse("team_stream"))), agent_id=None, team_id="desk"))

    # The member's RunCompleted comes first in the stream; the team's answer is the reply.
    assert turn.content.startswith("Team answer: ")
    # Leader delegation and the member's tool, each once (TeamRunCompleted repeats the member tools).
    assert _tool_names(turn) == ["lookup_order", "delegate_task_to_member"]
    assert _state(turn).has_tool_calls_in_order(["lookup_order", "delegate_task_to_member"])


async def test_agentos_stream_workflow_run() -> None:
    adapter = _agentos(Server(_sse_response(_fixture_sse("workflow_stream"))), agent_id=None, workflow_id="flow")

    turn = await _turn(adapter)

    assert turn.content == 'Your order status: {"order_id": "1042", "status": "shipped"}'
    assert _tool_names(turn) == ["lookup_order"]


async def test_agentos_stream_guardrail_refusal_is_the_answer() -> None:
    server = Server(_sse_response(_fixture_sse("run_stream_guardrail")))

    turn = await _turn(_agentos(server))

    assert turn == AgentTurn(content="Sorry, I can't help with that.")
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("kind", "event", "field"),
    [
        ("agent_id", "RunError", "content"),
        ("team_id", "TeamRunError", "content"),
        ("workflow_id", "WorkflowError", "error"),
    ],
)
async def test_agentos_stream_output_check_refusal(kind: str, event: str, field: str) -> None:
    stream = _sse({"event": event, field: "I can't share that.", "error_type": "output_check_error"})
    target: dict[str, Any] = {kind: "x"}

    turn = await _turn(sim.agentos_adapter(BASE, transport=Server(_sse_response(stream)).transport, **target))

    assert turn.content == "I can't share that."


async def test_agentos_stream_run_error_is_a_backend_failure(caplog: pytest.LogCaptureFixture) -> None:
    failing = _sse(
        {"event": "RunStarted", "run_id": "r1"},
        {"event": "RunError", "run_id": "r1", "content": "upstream model gateway 504"},
    )
    server = Server(_sse_response(failing))

    with pytest.raises(SimulationBackendError, match=r"3 time\(s\).*AgentOS run failed \(RunError\): upstream model"):
        await _turn(_agentos(server))

    assert len(server.requests) == 3


async def test_agentos_stream_run_error_then_success() -> None:
    server = Server(
        _sse_response(_sse({"event": "RunError", "content": "boom"})), _sse_response(_fixture_sse("run_stream"))
    )

    assert _tool_names(await _turn(_agentos(server))) == ["lookup_order"]


async def test_agentos_stream_workflow_error_without_details() -> None:
    server = Server(_sse_response(_sse({"event": "WorkflowError"})))

    with pytest.raises(SimulationBackendError, match=r"WorkflowError\): no details"):
        await _turn(_agentos(server, agent_id=None, workflow_id="flow", retries=0))


async def test_agentos_stream_cancelled_run_is_a_backend_failure() -> None:
    server = Server(_sse_response(_sse({"event": "RunCancelled", "reason": "cancelled by operator"})))

    with pytest.raises(SimulationBackendError, match=r"was cancelled.*cancelled by operator"):
        await _turn(_agentos(server, retries=0))


async def test_agentos_stream_without_completion_is_retried() -> None:
    truncated = _sse({"event": "RunStarted", "run_id": "r1"}, {"event": "RunContent", "run_id": "r1", "content": "Hel"})
    server = Server(_sse_response(truncated))

    with pytest.raises(SimulationBackendError, match="event stream ended without RunCompleted"):
        await _turn(_agentos(server))

    assert len(server.requests) == 3


async def test_agentos_stream_ignores_nested_runs_and_falls_back_to_content_deltas() -> None:
    stream = _sse(
        {"event": "RunStarted", "run_id": "top"},
        {"event": "RunContent", "run_id": "top", "content": "Hello, "},
        {"event": "RunStarted", "run_id": "nested"},
        {"event": "RunContent", "run_id": "nested", "content": "nested text"},
        {"event": "RunError", "run_id": "nested", "content": "nested failure"},
        {"event": "RunCompleted", "run_id": "nested", "content": "nested answer"},
        {"event": "RunContent", "run_id": "top", "content": "world"},
        {"event": "RunContent", "run_id": "top", "content": {"structured": True}},
        {"event": "RunCompleted", "run_id": "top"},
    )
    text = "data: not json\n\ndata: [1, 2]\n\n" + stream

    turn = await _turn(_agentos(Server(_sse_response(text))))

    assert turn.content == "Hello, world"


async def test_agentos_stream_paused_run_reports_paused_tools() -> None:
    paused_tool = {"tool_name": "issue_refund", "tool_args": {"order_id": "1042"}, "requires_confirmation": True}
    stream = _sse(
        {"event": "RunStarted", "run_id": "r1"},
        {"event": "RunPaused", "run_id": "r1", "content": "I need your confirmation.", "tools": [paused_tool]},
    )

    turn = await _turn(_agentos(Server(_sse_response(stream))))

    assert turn.content == "I need your confirmation."
    assert _tool_names(turn) == ["issue_refund"]
    assert not _state(turn).has_tool_call("issue_refund")


async def test_agentos_stream_team_without_member_events_takes_member_tools() -> None:
    member_tool = {"tool_call_id": "c1", "tool_name": "lookup_order", "tool_args": {}, "result": "x"}
    unnamed = {"tool_name": "audit", "tool_args": {"a": 1}, "result": "ok"}
    stream = _sse(
        {"event": "TeamRunStarted", "run_id": "team"},
        {"event": "TeamToolCallCompleted", "run_id": "team", "tool": unnamed},
        {
            "event": "TeamRunCompleted",
            "run_id": "team",
            "content": "done",
            "member_responses": [{"agent_id": "m", "tools": [member_tool]}, "garbage"],
            "tools": [unnamed],
        },
    )

    turn = await _turn(_agentos(Server(_sse_response(stream)), agent_id=None, team_id="desk"))

    assert _tool_names(turn) == ["audit", "lookup_order"]


async def test_agentos_stream_member_tool_errors_are_kept_but_not_clean() -> None:
    errored = {"tool_call_id": "c1", "tool_name": "lookup_order", "tool_call_error": True, "result": "not found"}
    stream = _sse(
        {"event": "RunStarted", "run_id": "r1"},
        {"event": "ToolCallStarted", "run_id": "r1", "tool": {**errored, "tool_call_error": None}},
        {"event": "ToolCallCompleted", "run_id": "r1", "tool": errored},
        {"event": "ToolCallError", "run_id": "r1", "tool": errored, "error": "not found"},
        {"event": "RunCompleted", "run_id": "r1", "content": "Sorry, no such order."},
    )

    turn = await _turn(_agentos(Server(_sse_response(stream))))

    assert _tool_names(turn) == ["lookup_order"]
    assert not _state(turn).has_tool_call("lookup_order")


async def test_agentos_json_agent_run() -> None:
    server = Server(httpx.Response(200, json=_fixture_json("run_tool_call")))

    turn = await _turn(_agentos(server, stream=False))

    assert server.form()["stream"] == "false"
    assert server.requests[0].headers["accept"] == "application/json"
    assert turn.content == 'Your order status: {"order_id": "1042", "status": "shipped"}'
    assert _state(turn).tool_calls("lookup_order").last.input == {"order_id": "1042"}
    assert turn.references is not None
    assert turn.references[0].query == "Where is my order 1042?"


async def test_agentos_json_team_run_collects_member_tools() -> None:
    server = Server(httpx.Response(200, json=_fixture_json("team_run")))

    turn = await _turn(_agentos(server, agent_id=None, team_id="desk", stream=False))

    assert _tool_names(turn) == ["delegate_task_to_member", "lookup_order"]


async def test_agentos_json_workflow_run_collects_step_tools() -> None:
    server = Server(httpx.Response(200, json=_fixture_json("workflow_run")))

    turn = await _turn(_agentos(server, agent_id=None, workflow_id="flow", stream=False))

    assert _tool_names(turn) == ["lookup_order"]
    assert turn.content == 'Your order status: {"order_id": "1042", "status": "shipped"}'


@pytest.mark.parametrize("fixture", ["run_guardrail", "run_model_error"])
async def test_agentos_json_status_error_is_a_backend_failure(fixture: str) -> None:
    """Without the event stream a guardrail refusal cannot be told from a failed run."""
    server = Server(httpx.Response(200, json=_fixture_json(fixture)))

    with pytest.raises(SimulationBackendError, match="AgentOS run ended with status ERROR"):
        await _turn(_agentos(server, stream=False))

    assert len(server.requests) == 3


async def test_agentos_json_cancelled_run() -> None:
    server = Server(httpx.Response(200, json={"status": "cancelled", "content": None}))

    with pytest.raises(SimulationBackendError, match="status CANCELLED"):
        await _turn(_agentos(server, stream=False, retries=0))


async def test_agentos_response_type_decides_the_parser() -> None:
    server = Server(httpx.Response(200, json={"content": "plain json", "status": "COMPLETED"}))

    assert (await _turn(_agentos(server, stream=True))).content == "plain json"


async def test_agentos_json_paused_run_keeps_paused_tools() -> None:
    body = {
        "status": "PAUSED",
        "content": "Confirm?",
        "tools": [{"tool_name": "issue_refund", "external_execution_required": True}, "garbage"],
    }

    turn = await _turn(_agentos(Server(httpx.Response(200, json=body)), stream=False))

    assert _tool_names(turn) == ["issue_refund"]
    assert not _state(turn).has_tool_call("issue_refund")


async def test_agentos_structured_content() -> None:
    body = {"status": "COMPLETED", "content": {"answer": "shipped", "order": 1042}}

    plain = await _turn(_agentos(Server(httpx.Response(200, json=body))))
    rendered = await _turn(_agentos(Server(httpx.Response(200, json=body)), content_fn=lambda raw: raw["answer"]))

    assert json.loads(plain.content) == {"answer": "shipped", "order": 1042}
    assert rendered.content == "shipped"


async def test_agentos_empty_content() -> None:
    turn = await _turn(_agentos(Server(httpx.Response(200, json={"status": "COMPLETED"}))))

    assert turn == AgentTurn(content="")


async def test_agentos_gateway_error_page_as_content_is_retried() -> None:
    page = "<html><head><title>504 Gateway Time-out</title></head></html>"
    server = Server(httpx.Response(200, json={"status": "COMPLETED", "content": page}))

    with pytest.raises(SimulationBackendError, match="gateway HTML error page"):
        await _turn(_agentos(server))

    assert len(server.requests) == 3


async def test_agentos_malformed_references_are_skipped() -> None:
    body = {"content": "ok", "references": [{"query": "q", "references": ["a"]}, {"no_query": True}, "x"]}

    turn = await _turn(_agentos(Server(httpx.Response(200, json=body))))

    assert turn.references is not None
    assert [ref.query for ref in turn.references] == ["q"]


# -- OpenAI Chat Completions -----------------------------------------------------------


async def test_chat_request_shape_and_stateless_history() -> None:
    server = Server(_chat_ok("first"), _chat_ok("second"))
    adapter = sim.openai_chat_adapter(
        f"{BASE}/v1/",
        model="support-agent",
        api_key="k-1",
        system_prompt="You are support.",
        extra_body={"temperature": 0, "user": "sim"},
        transport=server.transport,
    )

    await _turn(adapter, _input("hi"))
    await _turn(adapter, _input("hi", "first", "refund please"))

    request = server.requests[1]
    assert str(request.url) == f"{BASE}/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k-1"
    assert request.headers["content-type"] == "application/json"
    assert server.json() == {
        "model": "support-agent",
        "messages": [
            {"role": "system", "content": "You are support."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "refund please"},
        ],
        "stream": False,
        "temperature": 0,
        "user": "sim",
    }
    assert server.json(0)["messages"][1:] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"content": "plain"}, "plain"),
        ({"content": [{"type": "text", "text": "a"}, {"type": "image_url"}, "b", {"type": "text", "text": 1}]}, "ab"),
        ({"content": None, "refusal": "I can't help with that."}, "I can't help with that."),
        ({"content": None}, ""),
        ({"content": {"odd": "shape"}}, '{"odd": "shape"}'),
    ],
)
async def test_chat_reply_text(message: dict[str, Any], expected: str) -> None:
    server = Server(httpx.Response(200, json={"choices": [{"message": {"role": "assistant", **message}}]}))

    turn = await _turn(_chat(server))

    assert turn == AgentTurn(content=expected)


async def test_chat_tool_calls_are_requests_not_executions() -> None:
    tool_calls = [
        {"id": "c1", "type": "function", "function": {"name": "lookup_order", "arguments": '{"order_id": "1042"}'}},
        {"id": "c2", "type": "function", "function": {"name": "note", "arguments": "not json"}},
        {"id": "c3", "type": "function", "function": {"name": "batch", "arguments": "[1, 2]"}},
        {"id": "c4", "type": "function", "function": {"name": "ping", "arguments": ""}},
        {"id": "c5", "type": "function", "function": {"name": "dict", "arguments": {"a": 1}}},
        {"id": "c6", "type": "function", "function": {"name": "odd", "arguments": 7}},
        {"id": "c7", "type": "function"},
        "garbage",
    ]
    body = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}]}

    turn = await _turn(_chat(Server(httpx.Response(200, json=body))))

    assert turn.tools is not None
    assert [(t.tool_call_id, t.tool_name, t.tool_args) for t in turn.tools] == [
        ("c1", "lookup_order", {"order_id": "1042"}),
        ("c2", "note", {"arguments": "not json"}),
        ("c3", "batch", {"arguments": [1, 2]}),
        ("c4", "ping", {}),
        ("c5", "dict", {"a": 1}),
        ("c6", "odd", {"arguments": 7}),
    ]
    assert all(tool.is_paused for tool in turn.tools)
    assert not _state(turn).has_tool_call("lookup_order")


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"error": {"message": "model overloaded"}}, "error in the chat completion: model overloaded"),
        ({"error": "plain error"}, "error in the chat completion: plain error"),
        ({"choices": []}, "chat completion without a message"),
        ({"choices": [{"finish_reason": "stop"}]}, "chat completion without a message"),
    ],
)
async def test_chat_unusable_completions_are_retried(body: dict[str, Any], match: str) -> None:
    server = Server(httpx.Response(200, json=body))

    with pytest.raises(SimulationBackendError, match=match):
        await _turn(_chat(server))

    assert len(server.requests) == 3


# -- OpenAI Responses ------------------------------------------------------------------


def _response(text: str = "ok", response_id: str | None = "resp_1", **fields: Any) -> httpx.Response:
    body: dict[str, Any] = {
        "object": "response",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        **fields,
    }
    if response_id is not None:
        body["id"] = response_id
    return httpx.Response(200, json=body)


def _responses(server: Server, **kwargs: Any) -> AgentAdapter:
    return sim.openai_responses_adapter(f"{BASE}/v1", model="agent", transport=server.transport, **kwargs)


async def test_responses_server_history_chains_previous_response_id() -> None:
    server = Server(_response("first", "resp_1"), _response("second", "resp_2"))
    adapter = _responses(server, api_key="k-1", instructions="Be brief.", extra_body={"metadata": {"entity_id": "x"}})

    await _turn(adapter, _input("hi"))
    await _turn(adapter, _input("hi", "first", "refund please"))

    assert str(server.requests[0].url) == f"{BASE}/v1/responses"
    assert server.requests[0].headers["authorization"] == "Bearer k-1"
    assert server.json(0) == {
        "model": "agent",
        "instructions": "Be brief.",
        "store": True,
        "input": [{"role": "user", "content": "hi"}],
        "metadata": {"entity_id": "x"},
    }
    assert server.json(1) == {
        "model": "agent",
        "instructions": "Be brief.",
        "store": True,
        "previous_response_id": "resp_1",
        "input": [{"role": "user", "content": "refund please"}],
        "metadata": {"entity_id": "x"},
    }


async def test_responses_threads_are_independent() -> None:
    server = Server(_response("a", "resp_a"), _response("b", "resp_b"))
    adapter = _responses(server)

    await _turn(adapter, _input("hi", thread_id="one"))
    await _turn(adapter, _input("hi", thread_id="two"))

    assert "previous_response_id" not in server.json(1)


async def test_responses_rollback_continues_from_an_earlier_response() -> None:
    server = Server(_response("first", "resp_1"), _response("second", "resp_2"), _response("third", "resp_3"))
    adapter = _responses(server)

    await _turn(adapter, _input("hi"))
    await _turn(adapter, _input("hi", "first", "attack"))
    # The red-team agent rolled "attack"/"second" back and tries another angle.
    await _turn(adapter, _input("hi", "first", "other angle"))

    assert server.json(2)["previous_response_id"] == "resp_1"
    assert server.json(2)["input"] == [{"role": "user", "content": "other angle"}]


async def test_responses_injected_messages_after_the_last_response_are_sent() -> None:
    server = Server(_response("first", "resp_1"), _response("third", "resp_2"))
    adapter = _responses(server)

    await _turn(adapter, _input("hi"))
    # sim.message() seeded an assistant message the server never produced.
    await _turn(adapter, _input("hi", "first", "more", "seeded reply", "and now?", new=3))

    assert server.json(1)["input"] == [
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "seeded reply"},
        {"role": "user", "content": "and now?"},
    ]


async def test_responses_agent_called_twice_resends_the_last_user_message() -> None:
    server = Server(_response("first", "resp_1"), _response("again", "resp_2"))
    adapter = _responses(server)

    await _turn(adapter, _input("hi"))
    await _turn(
        adapter,
        AgentInput(
            thread_id="t-1", messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "first"}]
        ),
    )

    assert server.json(1)["previous_response_id"] == "resp_1"
    assert server.json(1)["input"] == [{"role": "user", "content": "hi"}]


async def test_responses_transcript_history_sends_everything_unstored() -> None:
    server = Server(_response("first", "resp_1"), _response("second", None))
    adapter = _responses(server, history="transcript")

    await _turn(adapter, _input("hi"))
    await _turn(adapter, _input("hi", "first", "refund please"))

    assert server.json(1) == {
        "model": "agent",
        "store": False,
        "input": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "refund please"},
        ],
    }


async def test_responses_without_id_cannot_continue_on_the_server() -> None:
    with pytest.raises(SimulationBackendError, match="history='transcript'"):
        await _turn(_responses(Server(_response("ok", None))))


async def test_responses_output_parsing() -> None:
    output = [
        {"type": "reasoning", "summary": []},
        {"type": "function_call", "call_id": "c1", "name": "lookup_order", "arguments": '{"order_id": "1042"}'},
        {"type": "function_call_output", "call_id": "c1", "output": '{"status": "shipped"}'},
        {"type": "function_call", "call_id": "c2", "name": "issue_refund", "arguments": "{}"},
        {"type": "function_call", "call_id": "c3", "name": "search", "arguments": '{"q": "x"}'},
        {"type": "function_call_output", "call_id": "c3", "output": [{"type": "input_text", "text": "hit"}]},
        {"type": "function_call_output", "output": "orphan"},
        {"type": "mcp_call", "id": "m1", "name": "get_customer", "arguments": '{"id": "42"}', "output": "Ann"},
        {"type": "mcp_call", "id": "m2", "name": "crm_write", "arguments": "{}", "output": None, "error": "denied"},
        {"type": "message", "content": [{"type": "output_text", "text": "Your order "}, "garbage"]},
        {"type": "message", "content": [{"type": "output_text", "text": "shipped."}]},
        "garbage",
    ]
    server = Server(httpx.Response(200, json={"id": "r", "status": "completed", "output": output}))

    turn = await _turn(_responses(server))

    assert turn.content == "Your order shipped."
    assert turn.tools is not None
    assert [(t.tool_name, t.tool_args, t.result, t.is_paused, bool(t.tool_call_error)) for t in turn.tools] == [
        ("lookup_order", {"order_id": "1042"}, '{"status": "shipped"}', False, False),
        ("issue_refund", {}, None, True, False),
        ("search", {"q": "x"}, "hit", False, False),
        ("get_customer", {"id": "42"}, "Ann", False, False),
        ("crm_write", {}, "denied", False, True),
    ]
    assert _state(turn).tool_call_names() == ["lookup_order", "search", "get_customer"]


async def test_responses_refusal_is_the_reply() -> None:
    output = [{"type": "message", "content": [{"type": "refusal", "refusal": "I can't help with that."}]}]
    server = Server(httpx.Response(200, json={"id": "r", "status": "completed", "output": output}))

    assert (await _turn(_responses(server))).content == "I can't help with that."


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        (
            {"status": "failed", "error": {"code": "server_error", "message": "model crashed"}},
            "response failed: model crashed",
        ),
        ({"status": "in_progress"}, "response came back in_progress"),
        (
            {"status": "incomplete", "output": [], "incomplete_details": {"reason": "max_output_tokens"}},
            r"incomplete \(max_output_tokens\) without any text",
        ),
        ({"status": "incomplete", "output": [], "incomplete_details": None}, r"incomplete \(unknown reason\)"),
    ],
)
async def test_responses_unfinished_responses_are_retried(fields: dict[str, Any], match: str) -> None:
    server = Server(_response(**fields))

    with pytest.raises(SimulationBackendError, match=match):
        await _turn(_responses(server))

    assert len(server.requests) == 3


async def test_responses_incomplete_with_text_returns_the_partial_reply(caplog: pytest.LogCaptureFixture) -> None:
    server = Server(_response("partial", status="incomplete", incomplete_details={"reason": "max_output_tokens"}))

    with caplog.at_level(logging.WARNING, logger="simulhausen.http_adapters"):
        turn = await _turn(_responses(server))

    assert turn.content == "partial"
    assert any("response incomplete (max_output_tokens)" in message for message in caplog.messages)


# -- A2A ---------------------------------------------------------------------------------


def _rpc(result: Any = None, error: Any = None) -> httpx.Response:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": "1"}
    if error is not None:
        body["error"] = error
    else:
        body["result"] = result
    return httpx.Response(200, json=body)


def _task_v03(state: str, text: str, context_id: str = "ctx-1", task_id: str = "task-1") -> dict[str, Any]:
    return {
        "kind": "task",
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": state,
            "message": {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": text}]},
        },
    }


def _a2a(server: Server, **kwargs: Any) -> AgentAdapter:
    return sim.a2a_adapter(f"{BASE}/a2a/agents/support/v1/message:send", transport=server.transport, **kwargs)


async def test_a2a_v03_request_shape_and_context_mapping() -> None:
    server = Server(
        _rpc(_task_v03("input-required", "Which order?")),
        _rpc(_task_v03("completed", "Refund issued.")),
        _rpc(_task_v03("completed", "You're welcome.", task_id="task-2")),
    )
    adapter = _a2a(server, api_key="a2a-key")

    first = await _turn(adapter, _input("refund please"))
    second = await _turn(adapter, _input("refund please", "Which order?", "1042"))
    await _turn(adapter, _input("refund please", "Which order?", "1042", "Refund issued.", "thanks"))

    request = server.requests[0]
    assert str(request.url) == f"{BASE}/a2a/agents/support/v1/message:send"
    assert request.headers["accept"] == "application/json"
    assert request.headers["authorization"] == "Bearer a2a-key"
    assert "a2a-version" not in request.headers
    body = server.json(0)
    assert (body["jsonrpc"], body["method"]) == ("2.0", "message/send")
    uuid.UUID(body["id"])
    message = body["params"]["message"]
    assert (message["kind"], message["role"], message["parts"]) == (
        "message",
        "user",
        [{"kind": "text", "text": "refund please"}],
    )
    uuid.UUID(message["messageId"])
    assert "contextId" not in message
    # The task waits for input: the next message continues it in the server's context.
    assert server.json(1)["params"]["message"]["contextId"] == "ctx-1"
    assert server.json(1)["params"]["message"]["taskId"] == "task-1"
    assert server.json(1)["params"]["message"]["parts"] == [{"kind": "text", "text": "1042"}]
    # A completed task is never reopened.
    assert server.json(2)["params"]["message"]["contextId"] == "ctx-1"
    assert "taskId" not in server.json(2)["params"]["message"]
    assert (first.content, second.content, first.tools) == ("Which order?", "Refund issued.", None)


async def test_a2a_v10_request_and_wrapped_results() -> None:
    task = {
        "id": "task-1",
        "contextId": "ctx-9",
        "status": {"state": "TASK_STATE_COMPLETED", "message": {"role": "ROLE_AGENT", "parts": [{"text": "status"}]}},
        "artifacts": [{"artifactId": "a1", "parts": [{"text": "Refund issued."}]}, {"artifactId": "a2", "parts": []}],
    }
    message = {"role": "ROLE_AGENT", "messageId": "m2", "contextId": "ctx-9", "parts": [{"text": "Anything else?"}]}
    server = Server(_rpc({"task": task}), _rpc({"message": message}))
    adapter = _a2a(server, protocol_version="1.0")

    first = await _turn(adapter, _input("refund please"))
    second = await _turn(adapter, _input("refund please", "Refund issued.", "no"))

    assert server.requests[0].headers["a2a-version"] == "1.0"
    body = server.json(0)
    assert body["method"] == "SendMessage"
    assert (body["params"]["message"]["role"], body["params"]["message"]["parts"]) == (
        "ROLE_USER",
        [{"text": "refund please"}],
    )
    assert server.json(1)["params"]["message"]["contextId"] == "ctx-9"
    assert (first.content, second.content) == ("Refund issued.", "Anything else?")


async def test_a2a_agno_style_task_with_snake_case_and_history() -> None:
    task = {
        "id": "run-1",
        "context_id": "session-1",
        "kind": "task",
        "status": {"state": "completed"},
        "history": [
            {"role": "agent", "parts": [{"root": {"kind": "text", "text": "earlier"}}]},
            {"role": "user", "parts": [{"kind": "text", "text": "question"}]},
            {
                "role": "agent",
                "parts": [{"root": {"kind": "text", "text": "Shipped."}}, {"kind": "data", "data": {"eta": 2}}],
            },
        ],
    }
    server = Server(_rpc(task))
    adapter = _a2a(server)

    turn = await _turn(adapter)
    await _turn(adapter, _input("q", "Shipped.", "thanks"))

    assert turn.content == 'Shipped.\n{"eta": 2}'
    assert server.json(1)["params"]["message"]["contextId"] == "session-1"


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ({"kind": "task", "status": {"state": "completed"}, "history": [{"role": "user", "parts": []}]}, ""),
        (
            {"status": {"state": "rejected", "message": {"parts": [{"kind": "text", "text": "Not allowed."}]}}},
            "Not allowed.",
        ),
        (
            {
                "kind": "task",
                "status": {"state": "completed", "message": {"parts": []}},
                "artifacts": ["x", {"parts": [{"kind": "file"}, "y"]}],
            },
            "",
        ),
    ],
)
async def test_a2a_task_reply_text(task: dict[str, Any], expected: str) -> None:
    assert (await _turn(_a2a(Server(_rpc(task))))).content == expected


async def test_a2a_message_without_context_is_not_remembered() -> None:
    server = Server(_rpc({"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": "hi"}]}))
    adapter = _a2a(server)

    await _turn(adapter)
    await _turn(adapter, _input("q", "hi", "again"))

    assert "contextId" not in server.json(1)["params"]["message"]


async def test_a2a_transcript_history_starts_a_fresh_context_every_call() -> None:
    server = Server(_rpc(_task_v03("completed", "ok")))
    adapter = _a2a(server, history="transcript")

    await _turn(adapter, _input("hi"))
    await _turn(adapter, _input("hi", "ok", "refund please"))

    message = server.json(1)["params"]["message"]
    assert "contextId" not in message
    assert message["parts"][0]["text"] == (
        "Conversation so far:\nUser: hi\nAssistant: ok\nUser: refund please\n\nReply to the last user message."
    )


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (_task_v03("failed", "Error: boom"), "A2A task failed: Error: boom"),
        (_task_v03("TASK_STATE_CANCELED", ""), "A2A task canceled: no details"),
        (None, "A2A response without a result"),
    ],
)
async def test_a2a_failed_tasks_are_retried(result: Any, match: str) -> None:
    server = Server(_rpc(result))

    with pytest.raises(SimulationBackendError, match=match):
        await _turn(_a2a(server))

    assert len(server.requests) == 3


@pytest.mark.parametrize(
    ("error", "fatal"),
    [
        ({"code": -32603, "message": "Internal error"}, False),
        ({"code": -32001, "message": "Task not found"}, False),
        ("broken", False),
        ({"code": -32601, "message": "Method not found"}, True),
        ({"code": -32009, "message": "Version not supported"}, True),
    ],
)
async def test_a2a_json_rpc_errors(error: Any, fatal: bool) -> None:
    server = Server(_rpc(error=error))

    with pytest.raises(SimulationBackendError, match="A2A JSON-RPC error") as excinfo:
        await _turn(_a2a(server))

    assert len(server.requests) == (1 if fatal else 3)
    assert ("protocol_version" in str(excinfo.value)) is fatal


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ("auth-required", "asks for authentication"),
        ("working", "came back 'working'"),
        ("TASK_STATE_SUBMITTED", "came back 'submitted'"),
        (None, "came back 'without a state'"),
    ],
)
async def test_a2a_states_that_are_not_answers_fail_at_once(state: str | None, match: str) -> None:
    server = Server(_rpc({"kind": "task", "id": "t", "status": {"state": state}}))

    with pytest.raises(SimulationBackendError, match=match):
        await _turn(_a2a(server))

    assert len(server.requests) == 1


# -- LangGraph ---------------------------------------------------------------------------

THREAD = "7a1f2c34-5b6d-4e8f-9a0b-1c2d3e4f5a6b"


def _graph(server: Server, **kwargs: Any) -> AgentAdapter:
    return sim.langgraph_adapter(f"{BASE}/", assistant_id="agent", transport=server.transport, **kwargs)


def _state_reply(
    *produced: dict[str, Any], history: Sequence[dict[str, Any]] = ()
) -> Callable[[httpx.Request], httpx.Response]:
    """Echo the sent messages (with their ids, as add_messages does) followed by ``produced``."""

    def reply(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)["input"]["messages"]
        echoed = [{"type": "human", "content": m["content"], "id": m["id"]} for m in sent]
        return httpx.Response(200, json={"messages": [*history, *echoed, *produced]})

    return reply


async def test_langgraph_request_shape() -> None:
    server = Server(_state_reply({"type": "ai", "content": "Hello!", "tool_calls": []}))
    adapter = _graph(
        server,
        api_key="lg-key",
        extra_body={"config": {"configurable": {"user_id": "u-1"}}, "multitask_strategy": "reject"},
    )

    turn = await _turn(adapter, _input("hi", thread_id=THREAD))

    request = server.requests[0]
    assert str(request.url) == f"{BASE}/threads/{THREAD}/runs/wait"
    assert request.headers["x-api-key"] == "lg-key"
    assert "authorization" not in request.headers
    body = server.json()
    (sent,) = body["input"]["messages"]
    uuid.UUID(sent.pop("id"))
    assert sent == {"role": "user", "content": "hi"}
    assert {key: value for key, value in body.items() if key != "input"} == {
        "assistant_id": "agent",
        "if_not_exists": "create",
        "config": {"configurable": {"user_id": "u-1"}},
        "multitask_strategy": "reject",
    }
    assert turn == AgentTurn(content="Hello!")


async def test_langgraph_maps_non_uuid_thread_ids_to_a_stable_uuid() -> None:
    server = Server(_state_reply({"type": "ai", "content": "ok"}))
    adapter = _graph(server)

    await _turn(adapter, _input("hi", thread_id="thread-1"))
    await _turn(adapter, _input("hi", "ok", "again", thread_id="thread-1"))

    paths = [request.url.path for request in server.requests]
    expected = uuid.uuid5(uuid.NAMESPACE_URL, "simulhausen:thread-1")
    assert paths == [f"/threads/{expected}/runs/wait"] * 2


async def test_langgraph_sends_only_new_messages_and_reads_only_this_run() -> None:
    earlier = [
        {"type": "human", "content": "where is 1042?", "id": "h1"},
        {"type": "ai", "content": "", "id": "a1", "tool_calls": [{"name": "lookup_order", "args": {}, "id": "old"}]},
        {"type": "tool", "content": "shipped", "tool_call_id": "old", "id": "t1"},
        {"type": "ai", "content": "Shipped.", "id": "a2"},
    ]
    produced = [
        {
            "type": "ai",
            "content": "",
            "id": "a3",
            "tool_calls": [
                {"name": "issue_refund", "args": {"order_id": "1042"}, "id": "c1", "type": "tool_call"},
                {"name": "notify", "args": {"to": "ann"}, "id": "c2"},
                {"name": "escalate", "args": {}, "id": "c3"},
                "garbage",
            ],
        },
        {"type": "tool", "content": "refund ok", "tool_call_id": "c1", "status": "success", "id": "t2"},
        {"type": "tool", "content": [{"type": "text", "text": "mail failed"}], "tool_call_id": "c2", "status": "error"},
        {"type": "tool", "content": "orphan"},
        {
            "type": "ai",
            "content": [
                {"type": "text", "text": "Refund issued"},
                {"type": "tool_use", "id": "x"},
                {"type": "text", "text": "."},
            ],
        },
    ]
    server = Server(_state_reply(*produced, history=earlier))

    turn = await _turn(_graph(server), _input("where is 1042?", "Shipped.", "refund it", thread_id=THREAD))

    assert [m["content"] for m in server.json()["input"]["messages"]] == ["refund it"]
    assert turn.content == "Refund issued."
    assert turn.tools is not None
    assert [(t.tool_name, t.tool_args, t.result, bool(t.tool_call_error), t.is_paused) for t in turn.tools] == [
        ("issue_refund", {"order_id": "1042"}, "refund ok", False, False),
        ("notify", {"to": "ann"}, "mail failed", True, False),
        ("escalate", {}, None, False, True),
    ]
    assert _state(turn).tool_call_names() == ["issue_refund"]


async def test_langgraph_without_new_messages_resends_the_last_user_message() -> None:
    server = Server(_state_reply({"type": "ai", "content": "again"}))
    agent_input = AgentInput(
        thread_id=THREAD, messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
    )

    await _turn(_graph(server), agent_input)

    assert [m["content"] for m in server.json()["input"]["messages"]] == ["hi"]


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        # The server dropped our ids: everything after the last human message is this run's.
        ([{"type": "human", "content": "q"}, {"role": "assistant", "content": "by role"}], "by role"),
        ([{"type": "ai", "content": "only ai"}], "only ai"),
        ([{"type": "human", "content": "q"}], ""),
        (["garbage", {"type": "AIMessageChunk", "content": "chunk"}], "chunk"),
    ],
)
async def test_langgraph_reply_when_the_ids_are_lost(messages: list[Any], expected: str) -> None:
    server = Server(httpx.Response(200, json={"messages": messages}))

    assert (await _turn(_graph(server))).content == expected


async def test_langgraph_transcript_history_uses_stateless_runs() -> None:
    server = Server(_state_reply({"type": "ai", "content": "ok"}))

    turn = await _turn(_graph(server, history="transcript"), _input("hi", "hello", "refund please", thread_id=THREAD))

    assert server.requests[0].url.path == "/runs/wait"
    body = server.json()
    assert [(m["role"], m["content"]) for m in body["input"]["messages"]] == [
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "refund please"),
    ]
    assert "if_not_exists" not in body
    assert turn.content == "ok"


async def test_langgraph_custom_input_key() -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        assert "chat" in json.loads(request.content)["input"]
        return httpx.Response(200, json={"chat": [{"type": "ai", "content": "ok"}]})

    assert (await _turn(_graph(Server(reply), input_key="chat"))).content == "ok"


@pytest.mark.parametrize(
    ("error", "match"),
    [
        ({"error": "ValueError", "message": "graph exploded"}, "LangGraph run failed: ValueError: graph exploded"),
        ("plain failure", "LangGraph run failed: plain failure"),
    ],
)
async def test_langgraph_run_errors_are_retried(error: Any, match: str) -> None:
    server = Server(httpx.Response(200, json={"__error__": error}))

    with pytest.raises(SimulationBackendError, match=match):
        await _turn(_graph(server))

    assert len(server.requests) == 3


async def test_langgraph_busy_thread_is_retried() -> None:
    server = Server(
        httpx.Response(409, json={"detail": "Thread is busy"}), _state_reply({"type": "ai", "content": "ok"})
    )

    assert (await _turn(_graph(server))).content == "ok"
    assert len(server.requests) == 2


async def test_langgraph_output_without_the_message_key_fails_at_once() -> None:
    server = Server(httpx.Response(200, json={"answer": "42"}))

    with pytest.raises(SimulationBackendError, match=r"no 'messages' list \(keys: \['answer'\]\); set input_key"):
        await _turn(_graph(server))

    assert len(server.requests) == 1


# -- end to end through simulhausen.run ------------------------------------------------------


class ScriptedJudge(JudgeAgent):
    """Passes every criterion without a model."""

    async def judge(
        self, state: ScenarioState, criteria: Sequence[str] | None = None, *, additional_context: str | None = None
    ) -> JudgeVerdict:
        used = list(self.criteria if criteria is None else criteria)
        verdicts = [CriterionVerdict(criterion=c, reasoning="looks right", status="passed") for c in used]
        return JudgeVerdict(verdicts=verdicts, summary="fine")


async def test_run_with_an_agentos_agent_over_http() -> None:
    plain = _sse(
        {"event": "RunStarted", "run_id": "r2"},
        {"event": "RunContent", "run_id": "r2", "content": "You're welcome."},
        {"event": "RunCompleted", "run_id": "r2", "content": "You're welcome."},
    )
    server = Server(_sse_response(_fixture_sse("run_stream")), _sse_response(plain))
    seen: list[ScenarioState] = []

    def looked_up(state: ScenarioState) -> None:
        seen.append(state)
        assert state.has_tool_call("lookup_order"), state.tool_call_names()
        assert state.tool_calls("lookup_order").last.input == {"order_id": "1042"}

    result = await sim.run(
        name="order status over AgentOS",
        description="The user asks where order 1042 is.",
        agents=[
            sim.agentos_adapter(BASE, agent_id="support", api_key="os-key", transport=server.transport),
            ScriptedJudge(criteria=["The agent looks the order up"]),
        ],
        script=[
            sim.user("where is my order 1042?"),
            sim.agent(),
            looked_up,
            sim.user("thanks"),
            sim.agent(),
            sim.judge(),
        ],
    )

    assert result.success, result.failure_summary()
    assert result.tool_calls == ["lookup_order"]
    assert [m["content"] for m in result.messages][-1] == "You're welcome."
    thread_id = seen[0].thread_id
    assert [server.form(i)["session_id"] for i in range(2)] == [thread_id, thread_id]
    assert [server.form(i)["message"] for i in range(2)] == ["where is my order 1042?", "thanks"]


async def test_run_marks_an_unreachable_agent_as_warn() -> None:
    server = Server(httpx.ConnectError("connection refused"))

    with pytest.raises(SimulationBackendError):
        await sim.run(
            name="agent down",
            description="The agent API is not reachable.",
            agents=[sim.langgraph_adapter(BASE, assistant_id="agent", transport=server.transport)],
            script=[sim.user("hello"), sim.agent()],
        )

    (record,) = sim.collector.records
    assert record.result.status is ScenarioStatus.WARN
    assert record.result.harness_error
