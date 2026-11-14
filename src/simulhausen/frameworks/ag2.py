"""Adapter for AG2 1.x (the ``ag2`` package): an ``ag2.Agent``.

``ag2_adapter`` asks the agent on an event stream it keeps per conversation
and reads the reply, the tool calls and the token usage from the events the
turn added to that stream. AG2 is imported only when an adapter is called, so
the module loads without ``ag2`` installed.

The 0.x line of AG2 (``autogen.ConversableAgent``) has a different API and is
not covered here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    require,
    result_text,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["ag2_adapter"]

_ADAPTER = "ag2_adapter"


def _result(event: Any) -> str | None:
    """Text of a ``ToolResultEvent``: the error message of a failed call, else the parts of its ``ToolResult``."""
    error = getattr(event, "error", None)
    if error is not None:
        return str(error)
    result = getattr(event, "result", None)
    parts = getattr(result, "parts", None)
    if not isinstance(parts, list):
        return result_text(result)
    texts = [
        part.content if isinstance(getattr(part, "content", None), str) else result_text(getattr(part, "data", part))
        for part in parts
    ]
    return "".join(text or "" for text in texts)


def _tools(events: Sequence[Any]) -> list[ToolExecution] | None:
    """Tool calls of the turn: ``ToolCallEvent`` paired with ``ToolResultEvent`` by ``parent_id``.

    The calls a ``ModelResponse`` asked for are taken too, so a call that never
    ran (the turn stopped for a human approval) is reported as not executed.
    A ``ToolErrorEvent`` (it has ``error``) is a failed call.
    """
    calls: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for event in events:
        asked = getattr(getattr(event, "tool_calls", None), "calls", None)
        if asked:
            for call in asked:
                calls.setdefault(str(call.id), call)
        elif hasattr(event, "parent_id") and hasattr(event, "result"):
            results[str(event.parent_id)] = event
        elif hasattr(event, "arguments") and hasattr(event, "name"):
            calls.setdefault(str(event.id), event)
    executions: list[ToolExecution] = []
    for call_id, call in calls.items():
        result = results.get(call_id)
        if result is None:
            executions.append(tool_execution(call.name, call.arguments, executed=False, call_id=call_id))
        else:
            failed = getattr(result, "error", None) is not None
            executions.append(tool_execution(call.name, call.arguments, _result(result), error=failed, call_id=call_id))
    return executions or None


def _usage(events: Sequence[Any]) -> Usage | None:
    """``UsageEvent`` records of the turn, summed; AG2 emits one per model call that reported tokens.

    A ``ModelResponse`` carries the same numbers in ``usage``; only the
    ``UsageEvent`` (it has ``kind``) is counted, as ``ag2.UsageReport`` does.
    """
    total: Usage | None = None
    for event in events:
        reported = getattr(event, "usage", None) if hasattr(event, "kind") else None
        usage = usage_of(reported, requests=1) if reported else None
        if usage is not None:
            total = usage if total is None else total + usage
    return total


class _AG2Adapter(FrameworkAdapter):
    """New user messages -> ``agent.ask(..., stream=...)`` -> ``AgentTurn``, one stream per thread."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        ask_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._ask_kwargs = ask_kwargs
        # thread_id -> the stream whose history holds the conversation.
        self._conversations: Conversations[Any] = Conversations()

    def _is_target(self, target: Any) -> bool:
        return not isinstance(target, type) and hasattr(target, "ask")

    async def _replay(self, input: AgentInput) -> tuple[Any, list[str]]:
        """A new stream whose history is the transcript, and the user messages still to ask.

        The user messages after the last agent reply are asked; everything
        before them becomes ``ModelRequest`` / ``ModelResponse`` events of the
        stream's history. A transcript that ends with a reply has nothing left
        to ask: its last user message is asked again.
        """
        ag2 = require("ag2", package="ag2", adapter=_ADAPTER)
        events = require("ag2.events", package="ag2", adapter=_ADAPTER)
        split = len(input.messages)
        while split and input.messages[split - 1]["role"] == "user":
            split -= 1
        stream = ag2.MemoryStream()
        earlier = [
            events.ModelRequest([events.TextInput(message["content"])])
            if message["role"] == "user"
            else events.ModelResponse(events.ModelMessage(message["content"]))
            for message in input.messages[:split]
        ]
        if earlier:
            await stream.history.replace(earlier)
        asked = [message["content"] for message in input.messages[split:]]
        return stream, asked or [input.last_new_user_message_str()]

    async def call(self, input: AgentInput) -> AgentTurn:
        agent = self._resolve()
        stream = self._conversations.recall(input) if self._history == "session" else None
        if stream is None:
            stream, asked = await self._replay(input)
        else:
            asked = [message["content"] for message in input.new_messages if message["role"] == "user"]
        known = len(list(await stream.history.get_events()))
        reply = await agent.ask(*asked, stream=stream, **self._ask_kwargs)
        events = list(await stream.history.get_events())
        # Compaction may have shortened the history; then every event left is taken.
        new = events[known:] if len(events) >= known else events
        content = self._content_fn(reply) if self._content_fn is not None else str(getattr(reply, "body", None) or "")
        if self._history == "session":
            self._conversations.remember(input, content, stream)
        return AgentTurn(content=content, tools=_tools(new), usage=_usage(new))


def ag2_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    **ask_kwargs: Any,
) -> AgentAdapter:
    """Adapter for an AG2 1.x agent (``pip install ag2``).

    ``target`` is an ``ag2.Agent`` or a zero-argument factory returning one.
    A turn is one ``await agent.ask(*messages, stream=...)``. Extra kwargs go
    to ``ask`` unchanged: ``dependencies=...``, ``variables=...``,
    ``tools=...``, ``hitl_hook=...``. Multi-agent setups of AG2 1.x are an
    agent whose tools are other agents (``Agent.as_tool()``), so they are
    driven the same way; a worker agent runs on a stream of its own, so the
    turn shows the delegation as one tool call and counts the worker's tokens,
    but not the tools the worker called.

    An ``ag2.Agent`` holds no conversation; the events of a conversation live
    on a stream. ``history="session"`` (default) keeps one ``MemoryStream``
    per ``thread_id`` and asks only the new user messages, so the agent sees
    the tool calls and results of earlier turns. After a rollback or a
    scripted ``sim.agent("...")`` reply, and on every call with
    ``history="transcript"``, the adapter opens a new stream whose history is
    the transcript (``ModelRequest`` / ``ModelResponse`` events); earlier
    tool results are then not replayed.

    The reply is ``AgentReply.body`` (the JSON text for an agent with a
    response schema); ``content_fn`` gets the ``AgentReply`` instead. Tool
    calls are the ``ToolCallEvent`` / ``ToolResultEvent`` pairs the turn added
    to the stream, a ``ToolErrorEvent`` being an error with the exception text
    as its result, and a call without a result is recorded as not executed.
    Usage is the sum of the turn's ``UsageEvent`` records.

    >>> adapter = ag2_adapter(lambda: support_agent)
    """
    return _AG2Adapter(target, history=history, content_fn=content_fn, ask_kwargs=ask_kwargs)
