"""HTTP adapters: agents behind agno AgentOS, OpenAI-compatible, A2A and LangGraph APIs.

``agno_adapter`` runs the agent inside the test process. These adapters talk
to an agent that runs somewhere else and is reached over HTTP: a sidecar
container in a Harbor task, a staging deployment, an agent written in another
framework. Each factory returns an ordinary
:class:`~simulhausen.types.AgentAdapter`, so the simulator, the judge, scripts
and tool assertions work exactly as with an in-process agent.

What every adapter does the same way:

- One server-side conversation per ``input.thread_id`` where the protocol has
  one (AgentOS session, Responses ``previous_response_id`` chain, A2A
  ``contextId``, LangGraph thread). ``history="transcript"``, and the stateless
  Chat Completions protocol, send the whole transcript on every turn instead,
  so a rollback (red-team backtracking) really reaches the agent.
- Tool calls the server reports become agno ``ToolExecution`` objects, so
  ``state.has_tool_call(...)`` works. A call the server only requested,
  without running it, is marked paused and does not count.
- Connect errors, timeouts, 5xx, 429 and 408 are retried ``retries`` times
  (``retry_delay * attempt`` apart, or as long as a short ``Retry-After``
  asks), then raised as ``SimulationBackendError``: the harness broke, not the
  agent, so the scenario is WARN. Any other 4xx is a configuration mistake (a
  bad key, a wrong URL or id) and raises at once with the status and the start
  of the body; it is still a harness error, because the agent never answered.

httpx comes with agno (agno requires ``httpx[http2]``), so this module adds no
dependency and httpx is deliberately not listed in ``[project].dependencies``.
An ``httpx.AsyncClient`` must not be reused on another event loop
(pytest-asyncio gives every test its own loop, and a reused client fails with
"Event loop is closed"), so the adapters open a client per call. An injected
``client`` is used as is and never closed; a ``transport`` (an
``httpx.MockTransport`` in tests, a proxy setup) is used for every request and
never closed either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import httpx
from agno.models.message import MessageReferences
from agno.models.response import ToolExecution
from pydantic import ValidationError

from simulhausen.adapters import (
    _CHECK_ERROR_TYPES,
    DEFAULT_BACKEND_RETRIES,
    DEFAULT_BACKEND_RETRY_DELAY_S,
    _looks_like_error_page,
)
from simulhausen.exceptions import SimulationBackendError
from simulhausen.types import AgentAdapter, AgentTurn

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterator, Sequence

    from simulhausen.types import AgentInput, Message

__all__ = [
    "a2a_adapter",
    "agentos_adapter",
    "langgraph_adapter",
    "openai_chat_adapter",
    "openai_responses_adapter",
]

logger = logging.getLogger(__name__)

# An agent turn can take minutes (tool calls, a slow model), far beyond httpx's
# 5 s default. Connecting is capped separately so a host that never answers
# fails fast and the retries still fit in a scenario.
DEFAULT_TIMEOUT_S = 120.0
_CONNECT_TIMEOUT_S = 10.0
# A Retry-After longer than this is not waited out: the retries would outlast
# the scenario. Larger values are cut to the cap.
_MAX_RETRY_AFTER_S = 30.0
# How much of an error body goes into an exception message.
_EXCERPT_CHARS = 300

# How the AgentOS agent gets the dialogue so far (see agentos_adapter).
type AgentOSHistory = Literal["session", "transcript"]
# "server": the server keeps the conversation and gets only what is new;
# "transcript": the whole transcript goes out on every call.
type ServerHistory = Literal["server", "transcript"]
type A2AProtocolVersion = Literal["0.3", "1.0"]

# Wraps a transcript for protocols whose input is one string (AgentOS
# ``message``, one A2A text part). Sent to the agent under test verbatim.
_TRANSCRIPT_TEMPLATE = "Conversation so far:\n{lines}\n\nReply to the last user message."


class _TransientError(Exception):
    """A failure worth another attempt (the server answered, but not usefully)."""


# -- shared helpers -------------------------------------------------------------


def _excerpt(text: str) -> str:
    """The start of a body on one line, for exception messages."""
    flat = " ".join(text.split())
    return flat if len(flat) <= _EXCERPT_CHARS else flat[:_EXCERPT_CHARS] + "..."


def _http_url(url: str, argument: str) -> str:
    """``url`` checked for an http(s) scheme."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{argument} must be an http:// or https:// URL, got {url!r}")
    return url


def _api_root(base_url: str) -> str:
    """``base_url`` without trailing slashes, so paths can be appended."""
    return _http_url(base_url, "base_url").rstrip("/")


def _check_history(history: str, allowed: tuple[str, ...]) -> None:
    if history not in allowed:
        expected = " or ".join(repr(value) for value in allowed)
        raise ValueError(f"history must be {expected}, got {history!r}")


def _check_reserved(extra: Mapping[str, Any] | None, reserved: frozenset[str], argument: str) -> dict[str, Any]:
    """A copy of ``extra``; keys the adapter sets itself are refused instead of silently overwritten."""
    extra = dict(extra or {})
    if clash := sorted(reserved & extra.keys()):
        raise ValueError(f"{argument} must not set {', '.join(clash)}: the adapter sets them itself")
    return extra


def _to_text(value: Any) -> str:
    """Reply content as text: strings as they are, structured content as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _blocks_text(content: Any) -> str:
    """Text of a message ``content`` that is a string or a list of content blocks.

    OpenAI-style servers and LangChain models may answer with blocks such as
    ``[{"type": "text", "text": "..."}, {"type": "tool_use", ...}]``; only the
    text blocks are the reply.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") in (None, "text", "output_text", "input_text")
                and isinstance(block.get("text"), str)
            ):
                texts.append(block["text"])
        return "".join(texts)
    return _to_text(content)


def _result_text(value: Any) -> str | None:
    """A tool result as text (``ToolExecution.result`` is a string)."""
    if value is None:
        return None
    if isinstance(value, list):
        return _blocks_text(value) or _to_text(value)
    return _to_text(value)


def _tool_args(raw: Any) -> dict[str, Any]:
    """Tool arguments as a dict; a JSON string is decoded, anything else is kept under ``arguments``."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return decoded if isinstance(decoded, dict) else {"arguments": decoded}
    return {"arguments": raw}


def _references(raw: Any) -> list[MessageReferences]:
    """RAG references in agno's ``MessageReferences`` shape; malformed entries are skipped."""
    references: list[MessageReferences] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            references.append(MessageReferences.model_validate(item))
        except ValidationError:
            logger.debug("skipping a malformed reference entry: %r", item)
    return references


def _json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        raise _TransientError(f"non-JSON response from {response.request.url}: {_excerpt(response.text)}") from None


def _json_object(response: httpx.Response) -> dict[str, Any]:
    body = _json_body(response)
    if not isinstance(body, dict):
        raise _TransientError(f"expected a JSON object from {response.request.url}, got: {_excerpt(response.text)}")
    return body


def _object(value: Any) -> dict[str, Any]:
    """``value`` when it is a JSON object, else an empty one (for optional nested fields)."""
    return value if isinstance(value, dict) else {}


def _retry_after(response: httpx.Response) -> float | None:
    """The ``Retry-After`` delay in seconds, capped; ``None`` when absent or an HTTP date."""
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER_S)


_STATUS_HINTS = {
    400: "the server rejected the request; check the adapter arguments",
    401: "check api_key or the auth headers",
    403: "check api_key or the auth headers, and what the caller may access",
    404: "check the URL and the agent/assistant id",
    405: "check the URL",
    422: "the server rejected the request; check the adapter arguments",
}


def _render_transcript(messages: Sequence[Message]) -> str:
    """The transcript as one string, for protocols that take a single text input."""
    if all(message["role"] == "user" for message in messages):
        return "\n\n".join(message["content"] for message in messages)
    names = {"user": "User", "assistant": "Assistant"}
    lines = "\n".join(f"{names[message['role']]}: {message['content']}" for message in messages)
    return _TRANSCRIPT_TEMPLATE.format(lines=lines)


# -- server-sent events -------------------------------------------------------------


@dataclass(frozen=True)
class _SSEEvent:
    """One dispatched server-sent event."""

    event: str
    data: str


_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _parse_sse(text: str) -> Iterator[_SSEEvent]:
    """Events of a ``text/event-stream`` body (WHATWG HTML, "server-sent events").

    ``event:`` sets the type (``"message"`` when absent), ``data:`` lines are
    joined with newlines, lines starting with ``:`` are comments, other fields
    (``id``, ``retry``) are ignored, and a blank line dispatches. Unlike a browser, the last
    event is still dispatched when the stream ends without its blank line: a
    server that closes early must not lose its terminal event. Lines split on
    CR, LF and CRLF only (``str.splitlines`` would also split on separators
    such as U+2028 that may sit inside JSON data).
    """
    event, data = "", []
    for line in _LINE_BREAK.split(text.removeprefix("\ufeff")):
        if not line:
            if data:
                yield _SSEEvent(event or "message", "\n".join(data))
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        name, colon, value = line.partition(":")
        if colon and value.startswith(" "):
            value = value[1:]
        if name == "event":
            event = value
        elif name == "data":
            data.append(value)
    if data:
        yield _SSEEvent(event or "message", "\n".join(data))


# -- shared base ----------------------------------------------------------------


class _BorrowedTransport(httpx.AsyncBaseTransport):
    """Delegates to a caller's transport and leaves closing it to the caller.

    The adapter's per-call client closes its transport on exit; a transport
    passed in must survive that and serve the next call.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True)
class _HttpOptions:
    """Connection settings every adapter shares, checked once by ``_http_options``."""

    headers: dict[str, str]
    timeout: httpx.Timeout
    retries: int
    retry_delay: float
    client: httpx.AsyncClient | None
    transport: httpx.AsyncBaseTransport | None


def _http_options(
    *,
    auth_headers: Mapping[str, str],
    headers: Mapping[str, str] | None,
    timeout: float,
    retries: int,
    retry_delay: float,
    client: httpx.AsyncClient | None,
    transport: httpx.AsyncBaseTransport | None,
) -> _HttpOptions:
    if retries < 0:
        raise ValueError("retries must be >= 0")
    if retry_delay < 0:
        raise ValueError("retry_delay must be >= 0")
    if not timeout > 0:
        raise ValueError(f"timeout must be > 0 seconds, got {timeout!r}")
    if client is not None and transport is not None:
        raise ValueError("pass client or transport, not both: an injected client already has its transport")
    return _HttpOptions(
        # The caller's headers come last, so they can replace the auth header
        # (an Azure-style "api-key", a custom scheme).
        headers={**auth_headers, **(headers or {})},
        timeout=httpx.Timeout(timeout, connect=min(timeout, _CONNECT_TIMEOUT_S)),
        retries=retries,
        retry_delay=retry_delay,
        client=client,
        transport=_BorrowedTransport(transport) if transport is not None else None,
    )


def _bearer(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


class _HttpAdapter(AgentAdapter):
    """Shared plumbing: a client per call, the retry loop, HTTP status handling."""

    # Retried besides every 5xx: 408 Request Timeout and 429 Too Many Requests.
    _retry_statuses: frozenset[int] = frozenset({408, 429})

    def __init__(self, http: _HttpOptions) -> None:
        self._http = http

    @asynccontextmanager
    async def _open_client(self) -> AsyncGenerator[httpx.AsyncClient]:
        if self._http.client is not None:
            yield self._http.client
            return
        async with httpx.AsyncClient(transport=self._http.transport, follow_redirects=True) as client:
            yield client

    async def _post[T](
        self,
        url: str,
        parse: Callable[[httpx.Response], T],
        *,
        json_body: Any = None,
        data: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> T:
        """POST, retrying transient failures, and return what ``parse`` makes of the response.

        ``parse`` raises ``_TransientError`` for a response that is worth
        another attempt (an error status inside a 200, a body that is not
        JSON) and ``SimulationBackendError`` for one that is not.
        """
        http = self._http
        attempts = http.retries + 1
        request_headers = {**(headers or {}), **http.headers}
        reason = ""
        async with self._open_client() as client:
            for attempt in range(1, attempts + 1):
                delay: float | None = None
                try:
                    response = await client.post(
                        url, json=json_body, data=data, headers=request_headers, timeout=http.timeout
                    )
                except httpx.RequestError as exc:
                    # Connect errors, timeouts, dropped connections, broken encodings.
                    reason = f"{type(exc).__name__} on POST {url}: {str(exc) or 'no details'}"
                else:
                    status = response.status_code
                    if status in self._retry_statuses or status >= 500:
                        reason = f"HTTP {status} from POST {url}: {_excerpt(response.text)}"
                        delay = _retry_after(response)
                    elif status >= 300:
                        hint = _STATUS_HINTS.get(status, "check the URL and the adapter arguments")
                        raise SimulationBackendError(
                            f"HTTP {status} from POST {url}: {_excerpt(response.text)} ({hint})"
                        )
                    else:
                        try:
                            return parse(response)
                        except _TransientError as exc:
                            reason = str(exc)
                logger.warning("backend error (attempt %d/%d): %s", attempt, attempts, reason)
                if attempt < attempts:
                    await asyncio.sleep(http.retry_delay * attempt if delay is None else delay)
        raise SimulationBackendError(f"backend failed {attempts} time(s) in a row: {reason}")


# -- agno AgentOS -----------------------------------------------------------------


@dataclass(frozen=True)
class _AgentOSEvents:
    """SSE event names of one AgentOS run kind (agno ``RunEvent``/``TeamRunEvent``/``WorkflowRunEvent``)."""

    started: str
    content: str | None
    completed: str
    error: str
    cancelled: str
    paused: str


_AGENTOS_EVENTS = {
    "agents": _AgentOSEvents("RunStarted", "RunContent", "RunCompleted", "RunError", "RunCancelled", "RunPaused"),
    # A team's own events carry the "Team" prefix; its members' events show up
    # unprefixed in the same stream, a member's RunCompleted included.
    "teams": _AgentOSEvents(
        "TeamRunStarted", "TeamRunContent", "TeamRunCompleted", "TeamRunError", "TeamRunCancelled", "TeamRunPaused"
    ),
    "workflows": _AgentOSEvents(
        "WorkflowStarted", None, "WorkflowCompleted", "WorkflowError", "WorkflowCancelled", "WorkflowPaused"
    ),
}
# Each tool execution arrives once in a *Completed event, errored ones
# included (agno sends ToolCallError right after the completed event of the
# same call, so it would only duplicate it).
_AGENTOS_TOOL_EVENTS = frozenset({"ToolCallCompleted", "TeamToolCallCompleted"})
_AGENTOS_RESERVED_FIELDS = frozenset({"message", "session_id"})


@dataclass
class _AgentOSRun:
    """What one AgentOS run produced, before the content is rendered."""

    content: Any
    tools: list[ToolExecution] = field(default_factory=list)
    references: list[MessageReferences] = field(default_factory=list)
    # A guardrail refused the request: ``content`` is the refusal, the agent's answer.
    refusal: bool = False


def _tool_executions(raw: Any) -> list[ToolExecution]:
    return [ToolExecution.from_dict(tool) for tool in raw if isinstance(tool, dict)] if isinstance(raw, list) else []


def _run_tools(run: Mapping[str, Any]) -> list[ToolExecution]:
    """Tools of a run dict, with those of team members and workflow steps (recursively).

    A team's own ``tools`` hold only the leader's calls (delegation); the
    domain tools sit in ``member_responses[*].tools``. A workflow keeps the
    runs of its steps in ``step_executor_runs``.
    """
    tools = _tool_executions(run.get("tools"))
    for key in ("member_responses", "step_executor_runs"):
        for child in run.get(key) or []:
            if isinstance(child, dict):
                tools.extend(_run_tools(child))
    return tools


def _tool_key(tool: ToolExecution) -> str:
    return tool.tool_call_id or json.dumps([tool.tool_name, tool.tool_args, tool.result], default=str, sort_keys=True)


def _parse_agentos_json(body: dict[str, Any]) -> _AgentOSRun:
    """A non-streamed run: ``RunOutput.to_dict()`` (or the team/workflow equivalent)."""
    status = str(body.get("status") or "").upper()
    if status in ("ERROR", "CANCELLED"):
        # Over plain JSON a guardrail refusal looks exactly like a model
        # failure (status ERROR, the message in content); only the event
        # stream tells them apart, hence stream=True by default.
        raise _TransientError(f"AgentOS run ended with status {status}: {_excerpt(_to_text(body.get('content')))}")
    return _AgentOSRun(
        content=body.get("content"), tools=_run_tools(body), references=_references(body.get("references"))
    )


def _parse_agentos_stream(text: str, names: _AgentOSEvents) -> _AgentOSRun:
    """A streamed run: tools from the tool events, content and references from the terminal event.

    The terminal event is the one of the top-level run (the ``run_id`` of the
    first started event): member and step runs report their own completion
    in the same stream.
    """
    top_run_id: str | None = None
    tools: list[ToolExecution] = []
    deltas: list[str] = []
    for sse in _parse_sse(text):
        try:
            data = json.loads(sse.data)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("event") or sse.event
        run_id = data.get("run_id")
        if name == names.started and top_run_id is None:
            top_run_id = run_id
        if name in _AGENTOS_TOOL_EVENTS:
            tools.extend(_tool_executions([data.get("tool")]))
            continue
        # Events without a run_id come from the server itself (an exception
        # outside the run), so they belong to the top-level run too.
        if top_run_id is not None and run_id is not None and run_id != top_run_id:
            continue
        if name == names.content and isinstance(data.get("content"), str):
            deltas.append(data["content"])
        elif name == names.error:
            message = _to_text(data.get("content") or data.get("error"))
            if data.get("error_type") in _CHECK_ERROR_TYPES:
                return _AgentOSRun(content=message, tools=tools, refusal=True)
            raise _TransientError(f"AgentOS run failed ({name}): {_excerpt(message) or 'no details'}")
        elif name == names.cancelled:
            raise _TransientError(f"AgentOS run was cancelled ({name}): {_excerpt(_to_text(data.get('reason')))}")
        elif name in (names.completed, names.paused):
            seen = {_tool_key(tool) for tool in tools}
            # A team that does not stream member events still lists their
            # tools in member_responses; a paused run lists its paused tools.
            extra = [tool for tool in _run_tools(data) if _tool_key(tool) not in seen]
            content = data.get("content")
            return _AgentOSRun(
                content="".join(deltas) if content is None else content,
                tools=tools + extra,
                references=_references(data.get("references")),
            )
    raise _TransientError(f"the AgentOS event stream ended without {names.completed}")


class _AgentOSAdapter(_HttpAdapter):
    def __init__(
        self,
        http: _HttpOptions,
        *,
        url: str,
        kind: str,
        form: dict[str, str],
        history: AgentOSHistory,
        stream: bool,
        content_fn: Callable[[Any], str] | None,
    ) -> None:
        super().__init__(http)
        self._url = url
        self._events = _AGENTOS_EVENTS[kind]
        self._form = form
        self._history = history
        self._stream = stream
        self._content_fn = content_fn

    async def call(self, input: AgentInput) -> AgentTurn:
        if self._history == "transcript":
            message = _render_transcript(input.messages)
            session_id = f"{input.thread_id}-{uuid.uuid4().hex[:12]}"
        else:
            message = input.last_new_user_message_str()
            session_id = input.thread_id
        form = {
            "message": message,
            "stream": "true" if self._stream else "false",
            "session_id": session_id,
            **self._form,
        }
        accept = "text/event-stream" if self._stream else "application/json"
        return await self._post(self._url, self._parse, data=form, headers={"Accept": accept})

    def _parse(self, response: httpx.Response) -> AgentTurn:
        # The response type decides, not the stream flag: a proxy may answer
        # either way.
        if "text/event-stream" in response.headers.get("content-type", ""):
            run = _parse_agentos_stream(response.text, self._events)
        else:
            run = _parse_agentos_json(_json_object(response))
        if run.refusal:
            content = _to_text(run.content)
        else:
            content = self._content_fn(run.content) if self._content_fn is not None else _to_text(run.content)
            if _looks_like_error_page(content):
                raise _TransientError(f"backend returned a gateway HTML error page: {_excerpt(content)[:120]}")
        return AgentTurn(content=content, tools=run.tools or None, references=run.references or None)


def _form_value(value: Any) -> str:
    """A form field the way agno's own client sends it: JSON for containers, "true"/"false" for bools."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(dict(value), default=str)
    if isinstance(value, list | tuple):
        return json.dumps(value, default=str)
    return str(value)


def agentos_adapter(
    base_url: str,
    *,
    agent_id: str | None = None,
    team_id: str | None = None,
    workflow_id: str | None = None,
    api_key: str | None = None,
    user_id: str | None = None,
    session_state: Mapping[str, Any] | None = None,
    history: AgentOSHistory = "session",
    stream: bool = True,
    content_fn: Callable[[Any], str] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_BACKEND_RETRIES,
    retry_delay: float = DEFAULT_BACKEND_RETRY_DELAY_S,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    **form_fields: Any,
) -> AgentAdapter:
    """Adapter for an agent, team or workflow served by agno AgentOS.

    ``base_url`` is the AgentOS root (``http://agent-api:7777``). Exactly one of
    ``agent_id``, ``team_id`` and ``workflow_id`` picks the target; every turn
    is a ``POST /agents/{id}/runs`` (``/teams/...``, ``/workflows/...``) with
    form fields, the way agno's own client sends them.

    ``api_key`` is the ``OS_SECURITY_KEY`` (or a JWT) and goes out as
    ``Authorization: Bearer``. ``user_id``, ``session_state`` and any extra
    keyword argument become form fields, which AgentOS hands to ``arun``:
    ``dependencies={...}``, ``metadata={...}``, ``version=2``. Dicts and lists
    are sent as JSON, booleans as ``"true"``/``"false"``.

    ``history="session"`` (default) sends the newest user message with
    ``session_id`` = the scenario's thread id, so the server's session supplies
    the earlier turns, as in production. That needs an agent with a ``db``:
    without one AgentOS forgets every turn. ``history="transcript"`` sends the
    whole transcript, rendered as text (the ``message`` field is one string),
    in a fresh session every call. Use it for red-team runs that roll the
    transcript back.

    ``stream=True`` (default) reads the run's server-sent events: tools come
    from the tool-call events (team members and workflow steps included),
    content and references from the completion event. It is the only mode
    that tells a guardrail refusal (``InputCheckError``/``OutputCheckError``,
    returned as the agent's answer) from a failed run (retried, then
    ``SimulationBackendError``). ``stream=False`` reads the plain JSON run
    output instead, where both come back as ``status: ERROR`` and count as a
    backend failure, so guardrail scenarios need ``stream=True``.

    ``content_fn`` renders the run content (a dict for agents with
    ``output_schema``) as text; without it, non-string content is sent on as
    JSON. ``timeout`` is in seconds per request (connecting gives up after at
    most 10 s); ``retries``/``retry_delay`` work as described in the module
    docstring.

    >>> adapter = agentos_adapter("http://localhost:7777", agent_id="support", api_key=os.environ["OS_SECURITY_KEY"])
    """
    root = _api_root(base_url)
    targets = {"agents": agent_id, "teams": team_id, "workflows": workflow_id}
    chosen = [(kind, target_id) for kind, target_id in targets.items() if target_id is not None]
    if len(chosen) != 1:
        raise ValueError(
            f"agentos_adapter needs exactly one of agent_id, team_id, workflow_id, got {len(chosen)}"
            + (f" ({', '.join(f'{kind[:-1]}_id' for kind, _ in chosen)})" if chosen else "")
        )
    kind, target_id = chosen[0]
    if not target_id:
        raise ValueError(f"{kind[:-1]}_id must not be empty")
    _check_history(history, ("session", "transcript"))
    _check_reserved(form_fields, _AGENTOS_RESERVED_FIELDS, "form fields")
    raw_form: dict[str, Any] = {"user_id": user_id, "session_state": session_state, **form_fields}
    form = {name: _form_value(value) for name, value in raw_form.items() if value is not None}
    return _AgentOSAdapter(
        _http_options(
            auth_headers=_bearer(api_key),
            headers=headers,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            client=client,
            transport=transport,
        ),
        url=f"{root}/{kind}/{quote(target_id, safe='')}/runs",
        kind=kind,
        form=form,
        history=history,
        stream=stream,
        content_fn=content_fn,
    )


# -- OpenAI Chat Completions --------------------------------------------------------


def _requested_tools(raw: Any) -> list[ToolExecution]:
    """Chat ``tool_calls``: the server asks the client to run them, so they are paused, not executed."""
    tools: list[ToolExecution] = []
    for call in raw if isinstance(raw, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        tools.append(
            ToolExecution(
                tool_call_id=call.get("id"),
                tool_name=function.get("name"),
                tool_args=_tool_args(function.get("arguments")),
                external_execution_required=True,
            )
        )
    return tools


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        return _to_text(error.get("message") or error)
    return _to_text(error)


class _OpenAIChatAdapter(_HttpAdapter):
    def __init__(
        self, http: _HttpOptions, *, url: str, model: str, system_prompt: str | None, extra_body: dict[str, Any]
    ) -> None:
        super().__init__(http)
        self._url = url
        self._model = model
        self._system_prompt = system_prompt
        self._extra_body = extra_body

    async def call(self, input: AgentInput) -> AgentTurn:
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend({"role": m["role"], "content": m["content"]} for m in input.messages)
        body = {"model": self._model, "messages": messages, "stream": False, **self._extra_body}
        return await self._post(self._url, self._parse, json_body=body, headers={"Accept": "application/json"})

    def _parse(self, response: httpx.Response) -> AgentTurn:
        body = _json_object(response)
        if body.get("error"):
            raise _TransientError(f"error in the chat completion: {_excerpt(_error_message(body['error']))}")
        choices = body.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        if not isinstance(message, dict):
            raise _TransientError(f"chat completion without a message: {_excerpt(response.text)}")
        content = _blocks_text(message.get("content")) or _to_text(message.get("refusal"))
        return AgentTurn(content=content, tools=_requested_tools(message.get("tool_calls")) or None)


def openai_chat_adapter(
    base_url: str,
    *,
    model: str,
    api_key: str | None = None,
    system_prompt: str | None = None,
    headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_BACKEND_RETRIES,
    retry_delay: float = DEFAULT_BACKEND_RETRY_DELAY_S,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AgentAdapter:
    """Adapter for an agent behind an OpenAI-compatible Chat Completions API.

    ``base_url`` is the API root that ``/chat/completions`` hangs off
    (``http://agent:8000/v1``); ``model`` names the agent or model on that
    server. ``api_key`` goes out as ``Authorization: Bearer``. ``extra_body``
    adds fields to every request (``{"temperature": 0, "user": "sim"}``);
    ``model``, ``messages`` and ``stream`` are the adapter's.

    The protocol is stateless: every turn sends the whole transcript, with
    ``system_prompt`` (if any) as a leading system message, so a rollback
    reaches the agent for free. The reply is ``choices[0].message.content``.

    Tools are mostly invisible here: an agent served this way runs its tools
    on the server and returns only the final text. ``message.tool_calls``, when
    present, are calls the server asks the *client* to run; they are recorded
    as paused executions (``external_execution_required``), which
    ``state.has_tool_call`` does not count.

    >>> adapter = openai_chat_adapter("http://localhost:8000/v1", model="support-agent")
    """
    if not model:
        raise ValueError("model must not be empty")
    return _OpenAIChatAdapter(
        _http_options(
            auth_headers=_bearer(api_key),
            headers=headers,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            client=client,
            transport=transport,
        ),
        url=f"{_api_root(base_url)}/chat/completions",
        model=model,
        system_prompt=system_prompt,
        extra_body=_check_reserved(extra_body, frozenset({"model", "messages", "stream"}), "extra_body"),
    )


# -- OpenAI Responses -----------------------------------------------------------------


@dataclass(frozen=True)
class _Checkpoint:
    """A stored response and the transcript it continues: every message up to its reply."""

    prefix: tuple[tuple[str, str], ...]
    response_id: str


def _transcript_key(messages: Sequence[Message]) -> tuple[tuple[str, str], ...]:
    return tuple((message["role"], message["content"]) for message in messages)


def _responses_tools(output: list[Any]) -> list[ToolExecution]:
    """Tool calls among the output items, in output order.

    A ``function_call`` with a ``function_call_output`` of the same
    ``call_id`` ran on the server (an agent server answers this way); one
    without is a request for the client to run it, so it is paused. An
    ``mcp_call`` ran on the server, failed when ``error`` is set.
    """
    results = {
        item["call_id"]: item.get("output")
        for item in output
        if item.get("type") == "function_call_output" and item.get("call_id")
    }
    tools: list[ToolExecution] = []
    for item in output:
        kind = item.get("type")
        if kind == "function_call":
            call_id = item.get("call_id")
            executed = call_id in results
            tools.append(
                ToolExecution(
                    tool_call_id=call_id,
                    tool_name=item.get("name"),
                    tool_args=_tool_args(item.get("arguments")),
                    result=_result_text(results.get(call_id)),
                    external_execution_required=None if executed else True,
                )
            )
        elif kind == "mcp_call":
            tools.append(
                ToolExecution(
                    tool_call_id=item.get("id"),
                    tool_name=item.get("name"),
                    tool_args=_tool_args(item.get("arguments")),
                    result=_result_text(item.get("output") if item.get("error") is None else item.get("error")),
                    tool_call_error=item.get("error") is not None,
                )
            )
    return tools


def _parse_responses(response: httpx.Response) -> tuple[AgentTurn, str | None]:
    """The reply text and tools of a ``Response`` object, plus its id."""
    body = _json_object(response)
    status = body.get("status")
    if status == "failed":
        raise _TransientError(f"response failed: {_excerpt(_error_message(body.get('error')))}")
    if status in ("in_progress", "queued", "cancelled"):
        raise _TransientError(f"response came back {status}, not completed")
    output = [item for item in body.get("output") or [] if isinstance(item, dict)]
    texts: list[str] = []
    refusals: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                refusals.append(part["refusal"])
    # The raw JSON has no output_text field (the SDK computes it): join the
    # output_text parts of every message item the way the SDK does.
    content = "".join(texts) or "\n".join(refusals)
    if status == "incomplete":
        reason = _to_text(_object(body.get("incomplete_details")).get("reason")) or "unknown reason"
        if not content:
            raise _TransientError(f"response incomplete ({reason}) without any text")
        logger.warning("response incomplete (%s): using the partial reply", reason)
    response_id = body.get("id")
    turn = AgentTurn(content=content, tools=_responses_tools(output) or None)
    return turn, response_id if isinstance(response_id, str) and response_id else None


class _OpenAIResponsesAdapter(_HttpAdapter):
    def __init__(
        self,
        http: _HttpOptions,
        *,
        url: str,
        model: str,
        instructions: str | None,
        history: ServerHistory,
        extra_body: dict[str, Any],
    ) -> None:
        super().__init__(http)
        self._url = url
        self._model = model
        self._instructions = instructions
        self._history = history
        self._extra_body = extra_body
        # thread_id -> stored responses, oldest first. Several are kept so a
        # rollback can continue from an earlier response that is still valid.
        self._chains: dict[str, list[_Checkpoint]] = {}

    async def call(self, input: AgentInput) -> AgentTurn:
        messages = input.messages
        body: dict[str, Any] = {"model": self._model}
        if self._instructions:
            # Instructions are not carried over by previous_response_id: send them every time.
            body["instructions"] = self._instructions
        if self._history == "transcript":
            new = messages
            body["store"] = False
        else:
            current = _transcript_key(messages)
            chain = self._chains.get(input.thread_id, [])
            base = next(
                (cp for cp in reversed(chain) if current[: len(cp.prefix)] == cp.prefix),
                None,
            )
            new = messages[len(base.prefix) :] if base else messages
            if not new:
                new = [{"role": "user", "content": input.last_new_user_message_str()}]
            body["store"] = True
            if base:
                body["previous_response_id"] = base.response_id
        body["input"] = [{"role": m["role"], "content": m["content"]} for m in new]
        body.update(self._extra_body)
        turn, response_id = await self._post(
            self._url, _parse_responses, json_body=body, headers={"Accept": "application/json"}
        )
        if self._history == "server":
            if response_id is None:
                raise SimulationBackendError(
                    "the Responses API answered without an id, so the conversation cannot continue on the "
                    "server; use openai_responses_adapter(..., history='transcript')"
                )
            prefix = (*_transcript_key(messages), ("assistant", turn.content))
            chain = [cp for cp in self._chains.get(input.thread_id, []) if len(cp.prefix) < len(prefix)]
            self._chains[input.thread_id] = [*chain, _Checkpoint(prefix, response_id)]
        return turn


def openai_responses_adapter(
    base_url: str,
    *,
    model: str,
    api_key: str | None = None,
    instructions: str | None = None,
    history: ServerHistory = "server",
    headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_BACKEND_RETRIES,
    retry_delay: float = DEFAULT_BACKEND_RETRY_DELAY_S,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AgentAdapter:
    """Adapter for an agent behind an OpenAI-compatible Responses API (``POST {base_url}/responses``).

    ``base_url`` is the API root (``http://agent:8000/v1``), ``model`` the
    agent or model name, ``api_key`` a bearer token, ``instructions`` the
    system prompt (sent on every request). ``extra_body`` adds fields
    (``{"metadata": {"entity_id": "support"}}`` for Microsoft Agent Framework
    DevUI); ``model``, ``input``, ``instructions``, ``previous_response_id``,
    ``store`` and ``stream`` are the adapter's.

    ``history="server"`` (default) keeps the conversation on the server: each
    turn sends only the messages since the last stored response, with its
    ``previous_response_id`` and ``store: true``. After a rollback the adapter
    continues from the newest earlier response that still matches the
    transcript. ``history="transcript"`` sends the whole transcript as
    ``input`` with ``store: false``, for servers that keep no responses.

    The reply is the text of the ``message`` output items. A
    ``function_call`` paired with a ``function_call_output`` (same
    ``call_id``) and an ``mcp_call`` are executed tools; a ``function_call``
    without its output is a request for the client and counts as paused.
    Hosted tools (web search, file search, code interpreter) are not mapped.
    ``status: failed`` is a backend failure; ``incomplete`` (e.g. hit
    ``max_output_tokens``) returns the partial text and is retried only when
    there is none.

    >>> adapter = openai_responses_adapter("http://localhost:8000/v1", model="support-agent")
    """
    if not model:
        raise ValueError("model must not be empty")
    _check_history(history, ("server", "transcript"))
    reserved = frozenset({"model", "input", "instructions", "previous_response_id", "store", "stream"})
    return _OpenAIResponsesAdapter(
        _http_options(
            auth_headers=_bearer(api_key),
            headers=headers,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            client=client,
            transport=transport,
        ),
        url=f"{_api_root(base_url)}/responses",
        model=model,
        instructions=instructions,
        history=history,
        extra_body=_check_reserved(extra_body, reserved, "extra_body"),
    )


# -- A2A ------------------------------------------------------------------------------

# JSON-RPC errors that mean the request itself is wrong (parse error, invalid
# request, unknown method, invalid params, unsupported operation, unsupported
# content type or A2A version): another attempt cannot help.
_A2A_FATAL_CODES = frozenset({-32700, -32600, -32601, -32602, -32004, -32005, -32009})
# Task states that are an answer (normalised, see _a2a_state).
_A2A_ANSWER_STATES = frozenset({"completed", "input-required", "rejected"})


@dataclass(frozen=True)
class _A2AContext:
    """The server's conversation for one thread: its context id, and the task waiting for input."""

    context_id: str
    task_id: str | None = None


def _pick(mapping: Mapping[str, Any], camel: str, snake: str) -> Any:
    """A field under its camelCase name (the spec) or snake_case (some pydantic servers)."""
    value = mapping.get(camel)
    return mapping.get(snake) if value is None else value


def _a2a_state(raw: Any) -> str:
    """``TASK_STATE_INPUT_REQUIRED`` (1.0) and ``input-required`` (0.3) both become ``input-required``."""
    return str(raw or "").lower().removeprefix("task_state_").replace("_", "-")


def _a2a_parts_text(parts: Any) -> str:
    """Text of A2A parts: text parts as they are, data parts as JSON; files are skipped."""
    texts: list[str] = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        # pydantic RootModel parts sometimes arrive wrapped in {"root": {...}}.
        part = part.get("root", part)
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif part.get("data") is not None:
            texts.append(_to_text(part["data"]))
    return "\n".join(texts)


def _a2a_task_text(task: Mapping[str, Any]) -> str:
    """The reply in a task: its artifacts, else its status message, else the last agent message in history."""
    artifacts = [artifact for artifact in task.get("artifacts") or [] if isinstance(artifact, dict)]
    if text := "\n".join(filter(None, (_a2a_parts_text(artifact.get("parts")) for artifact in artifacts))):
        return text
    status = _object(task.get("status"))
    message = status.get("message")
    if isinstance(message, dict) and (text := _a2a_parts_text(message.get("parts"))):
        return text
    for entry in reversed(task.get("history") or []):
        if isinstance(entry, dict) and entry.get("role") in ("agent", "ROLE_AGENT"):
            return _a2a_parts_text(entry.get("parts"))
    return ""


def _parse_a2a(response: httpx.Response) -> tuple[AgentTurn, _A2AContext | None]:
    body = _json_object(response)
    error = body.get("error")
    if error is not None:
        code = error.get("code") if isinstance(error, dict) else None
        message = f"A2A JSON-RPC error {code}: {_excerpt(_error_message(error))}"
        if isinstance(code, int) and code in _A2A_FATAL_CODES:
            raise SimulationBackendError(f"{message} (check the URL and protocol_version)")
        raise _TransientError(message)
    result = body.get("result")
    if not isinstance(result, dict):
        raise _TransientError(f"A2A response without a result: {_excerpt(response.text)}")
    # 1.0 wraps the result ({"task": ...} or {"message": ...}); 0.3 returns it
    # bare with a "kind" discriminator.
    if isinstance(result.get("task"), dict):
        task: dict[str, Any] | None = result["task"]
    elif isinstance(result.get("message"), dict):
        result, task = result["message"], None
    else:
        task = result if result.get("kind") == "task" or "status" in result else None
    if task is None:
        context_id = _pick(result, "contextId", "context_id")
        context = _A2AContext(context_id) if context_id else None
        return AgentTurn(content=_a2a_parts_text(result.get("parts"))), context
    status = _object(task.get("status"))
    state = _a2a_state(status.get("state"))
    text = _a2a_task_text(task)
    if state in ("failed", "canceled"):
        raise _TransientError(f"A2A task {state}: {_excerpt(text) or 'no details'}")
    if state == "auth-required":
        raise SimulationBackendError("the A2A agent asks for authentication (auth-required); pass api_key or headers")
    if state not in _A2A_ANSWER_STATES:
        raise SimulationBackendError(
            f"the A2A task came back {state or 'without a state'!r}; the adapter sends blocking requests and "
            "does not poll, so the server has to answer once the task is done or needs input"
        )
    context_id = _pick(task, "contextId", "context_id")
    # Only a task waiting for input continues; a finished task is never reopened.
    task_id = task.get("id") if state == "input-required" else None
    return AgentTurn(content=text), _A2AContext(context_id, task_id) if context_id else None


class _A2AAdapter(_HttpAdapter):
    def __init__(
        self, http: _HttpOptions, *, url: str, protocol_version: A2AProtocolVersion, history: ServerHistory
    ) -> None:
        super().__init__(http)
        self._url = url
        self._version = protocol_version
        self._history = history
        self._contexts: dict[str, _A2AContext] = {}

    def _message(self, text: str, context: _A2AContext | None) -> dict[str, Any]:
        if self._version == "1.0":
            message: dict[str, Any] = {"role": "ROLE_USER", "parts": [{"text": text}]}
        else:
            message = {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": text}]}
        message["messageId"] = str(uuid.uuid4())
        if context is not None:
            message["contextId"] = context.context_id
            if context.task_id:
                message["taskId"] = context.task_id
        return message

    async def call(self, input: AgentInput) -> AgentTurn:
        if self._history == "transcript":
            text, context = _render_transcript(input.messages), None
        else:
            text, context = input.last_new_user_message_str(), self._contexts.get(input.thread_id)
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "SendMessage" if self._version == "1.0" else "message/send",
            "params": {"message": self._message(text, context)},
        }
        headers = {"Accept": "application/json"}
        if self._version == "1.0":
            headers["A2A-Version"] = "1.0"
        turn, new_context = await self._post(self._url, _parse_a2a, json_body=payload, headers=headers)
        if self._history == "server" and new_context is not None:
            self._contexts[input.thread_id] = new_context
        return turn


def a2a_adapter(
    url: str,
    *,
    protocol_version: A2AProtocolVersion = "0.3",
    history: ServerHistory = "server",
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_BACKEND_RETRIES,
    retry_delay: float = DEFAULT_BACKEND_RETRY_DELAY_S,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AgentAdapter:
    """Adapter for an agent that speaks the A2A protocol over JSON-RPC.

    ``url`` is the JSON-RPC endpoint itself, not the agent card: the card's
    ``url`` (0.3) or ``supportedInterfaces[].url`` (1.0). For agno AgentOS that
    is ``{base}/a2a/agents/{id}/v1/message:send``, for a LangGraph server
    ``{base}/a2a/{assistant_id}``. Every turn is a blocking ``message/send``
    (``SendMessage`` with ``protocol_version="1.0"``, which also sends the
    ``A2A-Version: 1.0`` header). The default 0.3 is what agno and LangGraph
    servers speak today. ``api_key`` goes out as ``Authorization: Bearer``;
    other schemes go in ``headers``.

    ``history="server"`` (default) sends the newest user message and keeps one
    server conversation per thread: the first turn lets the server pick the
    ``contextId``, later turns send it back, plus the ``taskId`` while a task
    waits for input. ``history="transcript"`` sends the whole transcript as
    one text part in a fresh context every call (for red-team rollbacks).

    The reply is the text of the returned message, or of the task's artifacts,
    status message or last agent message, in that order. ``completed``,
    ``input-required`` and ``rejected`` tasks are answers; ``failed`` and
    ``canceled`` tasks and JSON-RPC errors are retried, then raised as
    ``SimulationBackendError`` (malformed-request errors at once). A task still
    ``working`` is an error too: the adapter does not poll. A2A has no tool
    calls in its data model, so ``state.has_tool_call`` sees nothing.

    >>> adapter = a2a_adapter("http://localhost:7777/a2a/agents/support/v1/message:send")
    """
    if protocol_version not in ("0.3", "1.0"):
        raise ValueError(f"protocol_version must be '0.3' or '1.0', got {protocol_version!r}")
    _check_history(history, ("server", "transcript"))
    return _A2AAdapter(
        _http_options(
            auth_headers=_bearer(api_key),
            headers=headers,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            client=client,
            transport=transport,
        ),
        url=_http_url(url, "url"),
        protocol_version=protocol_version,
        history=history,
    )


# -- LangGraph Agent Server -------------------------------------------------------------

_LANGGRAPH_AI = frozenset({"ai", "assistant", "AIMessageChunk"})
_LANGGRAPH_HUMAN = frozenset({"human", "user"})


def _langgraph_thread_id(thread_id: str) -> str:
    """LangGraph thread ids are UUIDs; any other id maps to a stable uuid5."""
    try:
        return str(uuid.UUID(thread_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"simulhausen:{thread_id}"))


def _langgraph_kind(message: Mapping[str, Any]) -> str:
    return str(message.get("type") or message.get("role") or "")


def _langgraph_tools(messages: list[dict[str, Any]]) -> list[ToolExecution]:
    """AI ``tool_calls`` paired with the ``tool`` messages that answer them (by ``tool_call_id``).

    A call without its tool message did not run (an interrupt stopped the
    graph), so it is paused.
    """
    results = {m["tool_call_id"]: m for m in messages if _langgraph_kind(m) == "tool" and m.get("tool_call_id")}
    tools: list[ToolExecution] = []
    for message in messages:
        if _langgraph_kind(message) not in _LANGGRAPH_AI:
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            result = results.get(call.get("id"))
            tools.append(
                ToolExecution(
                    tool_call_id=call.get("id"),
                    tool_name=call.get("name"),
                    tool_args=_tool_args(call.get("args")),
                    result=_result_text(result.get("content")) if result else None,
                    tool_call_error=result.get("status") == "error" if result else None,
                    requires_confirmation=None if result else True,
                )
            )
    return tools


class _LangGraphAdapter(_HttpAdapter):
    # 409: the thread is still busy with a previous run (multitask_strategy=reject).
    _retry_statuses = frozenset({408, 409, 429})

    def __init__(
        self,
        http: _HttpOptions,
        *,
        root: str,
        assistant_id: str,
        history: ServerHistory,
        input_key: str,
        extra_body: dict[str, Any],
    ) -> None:
        super().__init__(http)
        self._root = root
        self._assistant_id = assistant_id
        self._history = history
        self._input_key = input_key
        self._extra_body = extra_body

    async def call(self, input: AgentInput) -> AgentTurn:
        if self._history == "transcript":
            outgoing: Sequence[Message] = input.messages
            url = f"{self._root}/runs/wait"
            body: dict[str, Any] = {}
        else:
            outgoing = input.new_messages or [{"role": "user", "content": input.last_new_user_message_str()}]
            url = f"{self._root}/threads/{_langgraph_thread_id(input.thread_id)}/runs/wait"
            body = {"if_not_exists": "create"}
        # Every message gets an id, so the reply is whatever follows the last
        # one in the returned state: only what this run produced.
        sent = [{"role": m["role"], "content": m["content"], "id": str(uuid.uuid4())} for m in outgoing]
        body = {"assistant_id": self._assistant_id, "input": {self._input_key: sent}, **body, **self._extra_body}
        sent_ids = {message["id"] for message in sent}
        return await self._post(
            url,
            lambda response: self._parse(response, sent_ids),
            json_body=body,
            headers={"Accept": "application/json"},
        )

    def _parse(self, response: httpx.Response, sent_ids: set[str]) -> AgentTurn:
        body = _json_object(response)
        error = body.get("__error__")
        if error:
            details = f"{error.get('error', 'error')}: {error.get('message', '')}" if isinstance(error, dict) else error
            raise _TransientError(f"LangGraph run failed: {_excerpt(_to_text(details))}")
        messages = body.get(self._input_key)
        if not isinstance(messages, list):
            raise SimulationBackendError(
                f"the LangGraph run output has no {self._input_key!r} list (keys: {sorted(body)}); "
                "set input_key to the graph's message key"
            )
        messages = [m for m in messages if isinstance(m, dict)]
        produced = self._produced(messages, sent_ids)
        replies = [m for m in produced if _langgraph_kind(m) in _LANGGRAPH_AI]
        content = _blocks_text(replies[-1].get("content")) if replies else ""
        return AgentTurn(content=content, tools=_langgraph_tools(produced) or None)

    @staticmethod
    def _produced(messages: list[dict[str, Any]], sent_ids: set[str]) -> list[dict[str, Any]]:
        """The messages after the ones this call sent (after the last human message if the ids got lost)."""
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("id") in sent_ids:
                return messages[index + 1 :]
        for index in range(len(messages) - 1, -1, -1):
            if _langgraph_kind(messages[index]) in _LANGGRAPH_HUMAN:
                return messages[index + 1 :]
        return messages


def langgraph_adapter(
    base_url: str,
    *,
    assistant_id: str,
    api_key: str | None = None,
    history: ServerHistory = "server",
    input_key: str = "messages",
    headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_BACKEND_RETRIES,
    retry_delay: float = DEFAULT_BACKEND_RETRY_DELAY_S,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AgentAdapter:
    """Adapter for a graph served by a LangGraph Agent Server (``langgraph dev``, LangSmith Deployment).

    ``base_url`` is the server root (``http://localhost:2024``), ``assistant_id``
    the assistant id or graph name. ``api_key`` goes out as ``x-api-key``.
    ``input_key`` is the state key holding the messages. ``extra_body`` adds
    run fields (``{"config": {"configurable": {"user_id": "u-1"}}}``,
    ``{"context": {...}}``, ``{"multitask_strategy": "reject"}``);
    ``assistant_id``, ``input`` and ``if_not_exists`` are the adapter's.

    ``history="server"`` (default) runs every turn on one LangGraph thread,
    ``POST /threads/{thread_id}/runs/wait`` with ``if_not_exists: "create"``,
    sending only the messages since the agent last spoke. The thread id is the
    scenario's thread id when it is a UUID, else a uuid5 of it.
    ``history="transcript"`` sends the whole transcript to the stateless
    ``POST /runs/wait`` (for red-team rollbacks).

    The reply is the last AI message this run produced (text blocks joined
    when the content is a list). Tool calls are the AI messages' ``tool_calls``
    paired with the ``tool`` messages answering them; ``status: "error"``
    marks a failed call, and a call without an answer (an interrupt) is
    paused. Interrupts are not resumed. ``__error__`` in the output and 409
    (the thread is busy) are retried, then raised as
    ``SimulationBackendError``.

    >>> adapter = langgraph_adapter("http://localhost:2024", assistant_id="agent")
    """
    if not assistant_id:
        raise ValueError("assistant_id must not be empty")
    if not input_key:
        raise ValueError("input_key must not be empty")
    _check_history(history, ("server", "transcript"))
    return _LangGraphAdapter(
        _http_options(
            auth_headers={"x-api-key": api_key} if api_key else {},
            headers=headers,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            client=client,
            transport=transport,
        ),
        root=_api_root(base_url),
        assistant_id=assistant_id,
        history=history,
        input_key=input_key,
        extra_body=_check_reserved(extra_body, frozenset({"assistant_id", "input", "if_not_exists"}), "extra_body"),
    )
