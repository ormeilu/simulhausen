"""Adapter for Google Agent Development Kit (``google-adk``) agents.

``google_adk_adapter`` runs an ADK agent, an agent tree, a workflow or a ready
``Runner`` inside the test process: one ADK session per scenario, the reply,
the tool calls and the token usage read from the events of the run.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    require,
    text_of,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Message, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["google_adk_adapter"]

_ADAPTER = "google_adk_adapter"
_PACKAGE = "google-adk"
_PLUGIN_NAME = "simulhausen_recorder"

# ADK asks for a tool confirmation or for credentials with a function call of
# its own that points at the call it holds back; the key in its arguments names
# that call.
_HOLD_CALLS = {"adk_request_confirmation": "originalFunctionCall", "adk_request_credential": "functionCallId"}


@dataclass
class _Recorded:
    """What the recorder plugin saw: a finished tool call or the usage of a model call."""

    invocation_id: str | None
    call_id: str | None = None
    name: str | None = None
    args: Any = None
    result: Any = None
    error: bool = False
    usage: Any = None


# The log of the turn in progress: events from the runner and plugin records, in
# the order they happened. A context variable, so concurrent scenarios that
# share one runner (and one plugin) do not see each other's records.
_LOG: ContextVar[list[Any] | None] = ContextVar("simulhausen_google_adk_log", default=None)


def _record(item: _Recorded) -> None:
    log = _LOG.get()
    if log is not None:
        log.append(item)


def _recorder_plugin() -> Any:
    """A plugin that records tool and model calls, the ones inside an ``AgentTool`` included.

    An ``AgentTool`` runs its agent in a nested runner whose events never reach
    the outer event stream, but the nested runner inherits the plugins.
    """
    base = require("google.adk.plugins.base_plugin", package=_PACKAGE, adapter=_ADAPTER).BasePlugin

    class Recorder(base):
        def __init__(self) -> None:
            super().__init__(name=_PLUGIN_NAME)

        async def after_tool_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, result: Any) -> None:
            _record(_Recorded(_invocation(tool_context), _call_id(tool_context), tool.name, tool_args, result))

        async def on_tool_error_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, error: Any) -> None:
            text = f"{type(error).__name__}: {error}"
            _record(_Recorded(_invocation(tool_context), _call_id(tool_context), tool.name, tool_args, text, True))

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
            usage = getattr(llm_response, "usage_metadata", None)
            if usage is not None and not getattr(llm_response, "partial", None):
                _record(_Recorded(_invocation(callback_context), usage=usage))

    return Recorder()


def _invocation(context: Any) -> str | None:
    return getattr(context, "invocation_id", None)


def _call_id(context: Any) -> str | None:
    return getattr(context, "function_call_id", None)


def _parts(event: Any) -> list[Any]:
    content = getattr(event, "content", None)
    return list(getattr(content, "parts", None) or [])


def _event_text(event: Any) -> str:
    """The visible text of an event: its text parts without the model's thoughts."""
    return "".join(
        part.text for part in _parts(event) if isinstance(part.text, str) and not getattr(part, "thought", None)
    )


def _is_final(event: Any) -> bool:
    check = getattr(event, "is_final_response", None)
    return bool(check()) if callable(check) else False


def _reply(events: Sequence[Any]) -> str:
    """The reply of the run: the last final response that has text or the last node output, whichever is later.

    Every agent that takes part in an invocation ends with a final response of
    its own (the steps of a ``SequentialAgent``, the agent a transfer went to),
    so the last one is what the user is left with. A workflow node or a task
    mode agent reports its result in ``event.output`` instead of text.
    """
    for event in reversed(events):
        if _is_final(event) and (text := _event_text(event)):
            return text
        output = getattr(event, "output", None)
        if output is not None:
            return text_of(output)
    return ""


def _response_value(response: Any) -> tuple[Any, bool]:
    """A function response as ``(result, is_error)``.

    ADK wraps a plain return value as ``{"result": value}`` and reports a tool
    it could not run, or one whose error a callback handled, as
    ``{"error": text}``.
    """
    if not isinstance(response, dict):
        return response, False
    error = bool(response.get("error"))
    if set(response) == {"result"}:
        return response["result"], False
    if set(response) == {"error"}:
        return response["error"], error
    return response, error


@dataclass
class _Call:
    call_id: str | None
    name: str | None
    args: Any
    result: Any = None
    error: bool = False
    executed: bool = True
    answered: bool = False


def _find(calls: list[_Call], call_id: str | None, name: str | None) -> _Call | None:
    """The call a response belongs to: by id, or the oldest unanswered call of that name when there is no id."""
    for call in calls:
        if call_id is not None and call.call_id == call_id:
            return call
        if call_id is None and call.call_id is None and call.name == name and not call.answered:
            return call
    return None


def _held_call_id(function_call: Any) -> str | None:
    """The id of the call that an ``adk_request_confirmation`` / ``adk_request_credential`` call holds back."""
    target = (getattr(function_call, "args", None) or {}).get(_HOLD_CALLS[function_call.name])
    if isinstance(target, dict):
        target = target.get("id")
    return target if isinstance(target, str) else None


def _read_log(log: Sequence[Any]) -> tuple[list[ToolExecution], Usage | None]:
    """Tool calls in call order and the summed usage of one turn's events and plugin records."""
    events = [item for item in log if not isinstance(item, _Recorded) and not getattr(item, "partial", None)]
    invocations = {getattr(event, "invocation_id", None) for event in events}
    streamed = {part.function_call.id for event in events for part in _parts(event) if part.function_call is not None}
    calls: list[_Call] = []
    held: set[str] = set()
    usage: Usage | None = None

    def add_usage(source: Any) -> None:
        nonlocal usage
        one = usage_of(source, requests=1)
        if one is not None:
            usage = one if usage is None else usage + one

    for item in log:
        if isinstance(item, _Recorded):
            # The outer run's own calls are in the events; a record that is not
            # there comes from a nested runner (an AgentTool).
            if item.usage is not None:
                if item.invocation_id not in invocations:
                    add_usage(item.usage)
            elif item.call_id not in streamed:
                call = _find(calls, item.call_id, item.name) if item.call_id is not None else None
                if call is None:
                    call = _Call(item.call_id, item.name, item.args)
                    calls.append(call)
                value, failed = _response_value(item.result)
                call.result, call.error, call.answered = value, call.error or item.error or failed, True
            continue
        if getattr(item, "partial", None):
            continue
        add_usage(getattr(item, "usage_metadata", None))
        long_running = set(getattr(item, "long_running_tool_ids", None) or ())
        for part in _parts(item):
            function_call = part.function_call
            if function_call is not None:
                if function_call.name in _HOLD_CALLS:
                    if (target := _held_call_id(function_call)) is not None:
                        held.add(target)
                    continue
                call = _Call(function_call.id, function_call.name, function_call.args)
                call.executed = function_call.id not in long_running
                calls.append(call)
            response = part.function_response
            if response is not None and response.name not in _HOLD_CALLS:
                call = _find(calls, response.id, response.name)
                if call is None:
                    call = _Call(response.id, response.name, None)
                    calls.append(call)
                call.result, call.error = _response_value(response.response)
                call.answered = True
    tools = [
        tool_execution(
            call.name,
            call.args,
            call.result,
            error=call.error and call.call_id not in held,
            executed=call.executed and call.call_id not in held,
            call_id=call.call_id,
        )
        for call in calls
    ]
    return tools, usage


def _split(input: AgentInput, messages: Sequence[Message]) -> tuple[list[Message], str]:
    """``messages`` as the part to store in the session and the user message to send now."""
    if messages and messages[-1]["role"] == "user":
        return list(messages[:-1]), messages[-1]["content"]
    return list(messages), input.last_new_user_message_str()


class _GoogleAdkAdapter(FrameworkAdapter):
    """Last user message -> ``Runner.run_async`` in the thread's session -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        app_name: str | None,
        user_id: str,
        session_service: Any,
        state: dict[str, Any] | None,
        run_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._app_name = app_name
        self._user_id = user_id
        self._session_service = session_service
        self._state = state
        self._run_kwargs = run_kwargs
        self._runner: Any = None
        self._sessions: Conversations[str] = Conversations()
        self._threads: set[str] = set()

    def _is_target(self, target: Any) -> bool:
        # Agents, apps and workflow nodes are Pydantic models and a Runner has run_async; none is a factory.
        return hasattr(target, "run_async") or hasattr(target, "root_agent") or not callable(target)

    def _get_runner(self) -> Any:
        if self._runner is not None:
            return self._runner
        target = self._resolve()
        if hasattr(target, "session_service") and hasattr(target, "run_async"):
            if self._session_service is not None or self._app_name is not None:
                raise ValueError(
                    "google_adk_adapter got a ready Runner: app_name and session_service are the Runner's own, "
                    "pass them only together with an agent, an App or a workflow node."
                )
            runner = target
        else:
            runner_cls = require("google.adk.runners", package=_PACKAGE, adapter=_ADAPTER).Runner
            service = self._session_service
            if service is None:
                service = require("google.adk.sessions", package=_PACKAGE, adapter=_ADAPTER).InMemorySessionService()
            # An agent has a root_agent property too, so the agent check goes before the App one.
            if isinstance(target, require("google.adk.agents", package=_PACKAGE, adapter=_ADAPTER).BaseAgent):
                runner = runner_cls(agent=target, app_name=self._app_name or "simulhausen", session_service=service)
            elif hasattr(target, "root_agent"):
                runner = runner_cls(app=target, app_name=self._app_name, session_service=service)
            else:
                runner = runner_cls(node=target, app_name=self._app_name, session_service=service)
        manager = getattr(runner, "plugin_manager", None)
        if manager is not None and manager.get_plugin(_PLUGIN_NAME) is None:
            manager.register_plugin(_recorder_plugin())
        self._runner = runner
        return runner

    async def _open_session(self, runner: Any, session_id: str, earlier: Sequence[Message]) -> None:
        """Create the session with the initial state and store ``earlier`` in it as events."""
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=self._user_id,
            state=dict(self._state) if self._state is not None else None,
            session_id=session_id,
        )
        await self._store(runner, session, earlier)

    async def _store(self, runner: Any, session: Any, messages: Sequence[Message]) -> None:
        """Append transcript messages to the session as events of the user and of the root agent."""
        if not messages:
            return
        types = require("google.genai.types", package=_PACKAGE, adapter=_ADAPTER)
        event_cls = require("google.adk.events", package=_PACKAGE, adapter=_ADAPTER).Event
        agent_name = getattr(runner.agent, "name", None) or "model"
        for message in messages:
            from_user = message["role"] == "user"
            content = types.Content(role="user" if from_user else "model", parts=[types.Part(text=message["content"])])
            event = event_cls(
                author="user" if from_user else agent_name,
                invocation_id=f"simulhausen-{uuid.uuid4().hex[:12]}",
                content=content,
            )
            await runner.session_service.append_event(session, event)

    async def _session_for(self, runner: Any, input: AgentInput) -> tuple[str, str, bool]:
        """The session id to run in, the message to send, and whether the session is thrown away after the call."""
        thread_id = input.thread_id
        if self._history == "transcript":
            earlier, prompt = _split(input, input.messages)
            session_id = f"{thread_id}-{uuid.uuid4().hex[:12]}"
            await self._open_session(runner, session_id, earlier)
            return session_id, prompt, True
        session_id = self._sessions.recall(input)
        if session_id is not None:
            earlier, prompt = _split(input, input.new_messages)
            if earlier:
                # Messages the script added before the newest one go into the session as they are.
                session = await runner.session_service.get_session(
                    app_name=runner.app_name, user_id=self._user_id, session_id=session_id
                )
                await self._store(runner, session, earlier)
            return session_id, prompt, False
        earlier, prompt = _split(input, input.messages)
        if thread_id in self._threads:
            # The transcript no longer matches the session (a rollback, a scripted reply): start over from it.
            session_id = f"{thread_id}-{uuid.uuid4().hex[:12]}"
            await self._open_session(runner, session_id, earlier)
            return session_id, prompt, False
        self._threads.add(thread_id)
        existing = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=self._user_id, session_id=thread_id
        )
        if existing is None:
            await self._open_session(runner, thread_id, earlier)
        return thread_id, prompt, False

    async def call(self, input: AgentInput) -> AgentTurn:
        runner = self._get_runner()
        types = require("google.genai.types", package=_PACKAGE, adapter=_ADAPTER)
        session_id, prompt, disposable = await self._session_for(runner, input)
        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        log: list[Any] = []
        token = _LOG.set(log)
        try:
            async for event in runner.run_async(
                user_id=self._user_id, session_id=session_id, new_message=message, **self._run_kwargs
            ):
                # Not a comprehension: the recorder plugin appends to the same log while the run goes on.
                log.append(event)  # noqa: PERF401
        finally:
            _LOG.reset(token)
            if disposable:
                await runner.session_service.delete_session(
                    app_name=runner.app_name, user_id=self._user_id, session_id=session_id
                )
        events = [item for item in log if not isinstance(item, _Recorded) and not getattr(item, "partial", None)]
        content = self._content_fn(events) if self._content_fn is not None else _reply(events)
        tools, usage = _read_log(log)
        if not disposable:
            self._sessions.remember(input, content, session_id)
        return AgentTurn(content=content, tools=tools or None, usage=usage)


def google_adk_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    app_name: str | None = None,
    user_id: str = "simulhausen",
    session_service: Any = None,
    state: dict[str, Any] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a Google ADK agent, agent tree, workflow, ``App`` or ready ``Runner``.

    ``target`` is the object itself or a lazy factory. An agent (``LlmAgent``,
    a workflow agent, a tree with ``sub_agents``), an ``App`` or a workflow
    node is wrapped in a ``Runner`` with an ``InMemorySessionService`` (or
    ``session_service``) under ``app_name`` (default ``"simulhausen"`` for an
    agent, the app's or node's own name otherwise). A ready ``Runner`` keeps
    its own app name and session service. Extra kwargs go to
    ``Runner.run_async`` unchanged: ``run_config=...``, ``state_delta=...``.

    ``history="session"`` runs the whole scenario in one ADK session whose id
    is the simulation's ``thread_id``, created with the initial ``state`` for
    ``user_id``, and sends only the newest user message, so the agent sees its
    earlier tool calls and its session state as in production. A session that
    already exists in ``session_service`` under that id is used as it is. When
    the transcript stops matching the session (a rollback, a reply scripted
    with ``sim.agent("text")``), the adapter opens a new session and stores
    the transcript in it as events. ``history="transcript"`` does that on every
    call and deletes the session afterwards: the agent knows only what the
    transcript says, earlier tool calls and state changes are gone.

    Tool calls are the ``function_call`` parts of the run's events paired with
    their ``function_response`` parts by id, sub-agents' calls and the
    ``transfer_to_agent`` handoff included. A recorder plugin registered on the
    runner adds the calls made inside an ``AgentTool``, which never reach the
    outer events. A response of the form ``{"error": ...}`` is an error; a
    long-running call and a call waiting for confirmation or credentials are
    recorded as not executed. ADK itself re-raises an exception from a tool
    unless an ``on_tool_error_callback`` handles it, and the adapter lets it
    through. Usage is ``usage_metadata`` summed over the events, one request
    per model response.

    The reply is the last final response of the run that has text, or the
    last ``event.output`` (a workflow node's result) when that comes later. ``content_fn`` gets the
    list of this turn's events instead.

    >>> adapter = google_adk_adapter(lambda: root_agent, state={"plan": "pro"})
    """
    return _GoogleAdkAdapter(
        target,
        history=history,
        content_fn=content_fn,
        app_name=app_name,
        user_id=user_id,
        session_service=session_service,
        state=state,
        run_kwargs=run_kwargs,
    )
