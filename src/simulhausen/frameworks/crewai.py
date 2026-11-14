"""Adapter for CrewAI: a ``Crew``, a single ``Agent`` or a ``Flow`` (``pip install crewai``).

``crewai_adapter`` runs one kickoff per turn: ``crew.akickoff(inputs=...)``,
``agent.kickoff_async(messages)`` or ``flow.kickoff_async(inputs=...)``. The
reply and the token usage come from the kickoff's result; the tool calls come
from CrewAI's event bus, which is the one place every tool execution of a run
is reported, a coworker's or a flow step's included.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any, Literal

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    require,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Message

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["crewai_adapter"]

_ADAPTER = "crewai_adapter"
_PACKAGE = "crewai"

# Tool events that end a call: it returned, it raised (or its arguments were
# rejected), or the model named a tool the agent does not have.
_FINISHED = "ToolUsageFinishedEvent"
_ERRORS = ("ToolUsageErrorEvent", "ToolSelectionErrorEvent")

_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "successful_requests")

type _Kind = Literal["agent", "crew", "flow"]


class _Recorder:
    """Collects the tool events of one adapter call.

    The event bus is global to the process, so every handler sees the events
    of every run. CrewAI calls a handler in a copy of the context the event
    was emitted in, and carries that context into the threads it starts, so the
    recorder keeps an event only while it is the one active in that context.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    def handle(self, source: Any, event: Any) -> None:
        if _ACTIVE.get() is self:
            self.events.append(event)


_ACTIVE: contextvars.ContextVar[_Recorder | None] = contextvars.ContextVar("simulhausen_crewai_recorder", default=None)


def _executions(events: Sequence[Any]) -> list[ToolExecution]:
    """Tool executions from the collected events, in the order the events were emitted."""
    executions: list[ToolExecution] = []
    previous: tuple[Any, str] | None = None
    # Handlers run in a thread pool, so the events arrive in no particular order.
    for event in sorted(events, key=lambda event: event.timestamp):
        if type(event).__name__ == _FINISHED:
            failed = getattr(event, "failure", None) is not None
            executions.append(tool_execution(event.tool_name, event.tool_args, event.output, error=failed))
            previous = None
            continue
        failure = (event.tool_name, str(event.error))
        # CrewAI retries a failing call on its own and emits an error per attempt.
        if failure != previous:
            executions.append(tool_execution(event.tool_name, event.tool_args, failure[1], error=True))
        previous = failure
    return executions


def _kind(target: Any) -> _Kind:
    if hasattr(target, "tasks") and hasattr(target, "agents"):
        return "crew"
    return "agent" if hasattr(target, "role") else "flow"


def _counts(metrics: Any) -> dict[str, int]:
    return {name: int(getattr(metrics, name, 0) or 0) for name in _USAGE_FIELDS}


def _history_text(messages: Sequence[Message]) -> str:
    names = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{names[message['role']]}: {message['content']}" for message in messages)


class _CrewAIAdapter(FrameworkAdapter):
    """Message list (Agent) or inputs (Crew, Flow) -> one kickoff -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        kickoff_kwargs: dict[str, Any],
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        input_key: str,
        history_key: str | None,
        inputs_fn: Callable[[AgentInput], dict[str, Any]] | None,
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._kickoff_kwargs = kickoff_kwargs
        self._input_key = input_key
        self._history_key = history_key
        self._inputs_fn = inputs_fn
        self._shared = self._is_target(target) or not callable(target)
        self._objects: dict[str, Any] = {}
        # Per thread, the message list of a single Agent as its last kickoff left it.
        self._histories: Conversations[list[Any]] = Conversations()

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "kickoff_async") or hasattr(target, "akickoff")

    def _object(self, thread_id: str) -> Any:
        if self._shared:
            return self._target
        if self._history == "transcript":
            # Nothing may survive a call: a Flow keeps its state between kickoffs.
            return self._target()
        if thread_id not in self._objects:
            self._objects[thread_id] = self._target()
        return self._objects[thread_id]

    def _inputs(self, input: AgentInput) -> dict[str, Any]:
        if self._inputs_fn is not None:
            return self._inputs_fn(input)
        messages = input.messages
        newest = max((index for index, message in enumerate(messages) if message["role"] == "user"), default=None)
        inputs: dict[str, Any] = {self._input_key: messages[newest]["content"] if newest is not None else ""}
        if self._history_key is not None:
            inputs[self._history_key] = _history_text(messages[: newest or 0])
        return inputs

    def _messages(self, input: AgentInput) -> list[Any]:
        native = self._histories.recall(input) if self._history == "session" else None
        if native is None:
            return [dict(message) for message in input.messages]
        return [*native, *(dict(message) for message in input.new_messages)]

    async def _kickoff(self, target: Any, kind: _Kind, input: AgentInput) -> Any:
        if kind == "agent":
            return await target.kickoff_async(self._messages(input), **self._kickoff_kwargs)
        # A Crew's akickoff is async throughout; its kickoff_async runs the sync kickoff in a thread.
        kickoff = getattr(target, "akickoff", None) or target.kickoff_async
        return await kickoff(inputs=self._inputs(input), **self._kickoff_kwargs)

    async def call(self, input: AgentInput) -> AgentTurn:
        events = require("crewai.events", package=_PACKAGE, adapter=_ADAPTER)
        bus = events.crewai_event_bus
        target = self._object(input.thread_id)
        kind = _kind(target)
        # A Crew reports the lifetime totals of its agents' models, so the turn is the difference.
        before = _counts(target.calculate_usage_metrics()) if kind == "crew" else None

        recorder = _Recorder()
        event_types = [getattr(events, name) for name in (_FINISHED, *_ERRORS)]
        for event_type in event_types:
            bus.register_handler(event_type, recorder.handle)
        token = _ACTIVE.set(recorder)
        try:
            result = await self._kickoff(target, kind, input)
            # Handlers run on the bus's own threads; wait for the ones still queued.
            await asyncio.to_thread(bus.flush)
        finally:
            _ACTIVE.reset(token)
            for event_type in event_types:
                bus.off(event_type, recorder.handle)

        raw = result
        if hasattr(result, "raw"):
            parsed = getattr(result, "pydantic", None)
            raw = parsed if parsed is not None else result.raw
        content = self._render(raw)

        if before is not None:
            after = _counts(getattr(result, "token_usage", None))
            usage = usage_of({name: after[name] - before[name] for name in _USAGE_FIELDS})
        elif kind == "agent":
            usage = usage_of(getattr(result, "usage_metrics", None))
            self._remember(input, content, getattr(result, "messages", None))
        else:
            usage = usage_of(getattr(target, "usage_metrics", None))
        return AgentTurn(content=content, tools=_executions(recorder.events) or None, usage=usage)

    def _remember(self, input: AgentInput, content: str, messages: Any) -> None:
        """Keep the Agent's message list for the next turn, without the system prompt the kickoff adds itself."""
        if not isinstance(messages, list):
            self._histories.forget(input.thread_id)
            return
        native = [
            message for message in messages if not (isinstance(message, dict) and message.get("role") == "system")
        ]
        last = native[-1] if native else None
        # A ReAct-style run leaves the final answer out of the list.
        if not (isinstance(last, dict) and last.get("role") == "assistant" and last.get("content") == content):
            native.append({"role": "assistant", "content": content})
        self._histories.remember(input, content, native)


def crewai_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    input_key: str = "message",
    history_key: str | None = "history",
    inputs_fn: Callable[[AgentInput], dict[str, Any]] | None = None,
    **kickoff_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a CrewAI ``Crew``, a single ``Agent`` or a ``Flow``.

    ``target`` is the object or a zero-argument factory returning it. With a
    factory every ``thread_id`` gets its own object in session mode and every
    call a new one in transcript mode; an object passed directly serves every
    scenario, which is fine for a Crew or an Agent without CrewAI memory and
    wrong for a Flow that keeps state. Extra kwargs go to the kickoff call
    unchanged (``response_format=...`` for an Agent, ``input_files=...``).
    Streaming kickoffs (``stream=True``) are not supported.

    A single ``Agent`` runs ``agent.kickoff_async(messages)``. An Agent keeps
    nothing between kickoffs, so the adapter carries the history itself.
    ``history="session"`` (default) sends the message list the previous
    kickoff ended with (``LiteAgentOutput.messages``: the agent's own prompts,
    tool calls and tool results, without the system prompt the kickoff adds
    again) plus the new user message; after a rollback or a reply injected
    with ``sim.agent("...")`` it starts again from the transcript.
    ``history="transcript"`` sends the transcript as user and assistant
    messages on every call, so earlier tool results are not replayed.

    A ``Crew`` runs ``crew.akickoff(inputs=...)`` (``kickoff_async`` where
    there is no ``akickoff``) and a ``Flow`` ``flow.kickoff_async(inputs=...)``.
    Neither takes a message list, so both modes pass the dialogue as inputs:
    ``{input_key: newest user message, history_key: the dialogue before it}``,
    the history as ``User: ...`` / ``Assistant: ...`` lines (empty on the first
    turn). Task descriptions pick them up as ``{message}`` and ``{history}``;
    a Flow gets them in its state, so a Flow with a typed state needs both
    fields, ``history_key=None`` to drop the history, or ``inputs_fn``.
    ``inputs_fn(input) -> dict`` builds the inputs itself and replaces both
    keys. The modes differ only in what the object itself keeps: in session
    mode a thread's Crew or Flow lives on between turns (Flow state, CrewAI
    memory), in transcript mode a factory builds a new one for every call.

    Tool calls come from CrewAI's event bus (``ToolUsageFinishedEvent``,
    ``ToolUsageErrorEvent``, ``ToolSelectionErrorEvent``), with the tool name,
    the arguments and the output or the error text, ordered by event time;
    delegation tools and the tools of coworkers and of crews inside a Flow are
    included. The events carry no call id. A tool that raised, rejected its
    arguments or does not exist is an error, and so is a call CrewAI marks as
    failed. CrewAI retries a failing call itself and emits an error per
    attempt: consecutive identical errors of one tool are reported once. The
    handlers are registered for the duration of the call and scoped with a
    context variable, so scenarios running concurrently in one process do not
    see each other's calls; a tool call made in a thread or task that does not
    inherit the call's context would be missed.

    ``AgentTurn.usage`` is ``LiteAgentOutput.usage_metrics`` for an Agent, the
    growth of ``CrewOutput.token_usage`` over the kickoff for a Crew (CrewAI
    reports the lifetime totals of the agents' models there, so calls made
    through the same model objects elsewhere at the same time are counted in),
    and ``flow.usage_metrics`` for a Flow.

    ``content_fn`` renders the reply: it gets the result's ``pydantic`` object
    when there is one, else its ``raw`` text, and whatever a Flow's last method
    returned when that is not a CrewAI output.

    >>> adapter = crewai_adapter(lambda: support_crew(), input_key="question")
    """
    return _CrewAIAdapter(
        target,
        kickoff_kwargs,
        history=history,
        content_fn=content_fn,
        input_key=input_key,
        history_key=history_key,
        inputs_fn=inputs_fn,
    )
