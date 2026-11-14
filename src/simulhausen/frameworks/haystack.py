"""Adapter for deepset Haystack: the ``Agent`` component, or a pipeline that answers with chat messages.

``haystack_adapter`` sends a list of ``ChatMessage`` and reads the chat
messages that come back: the reply, the tool calls with their results and the
token usage of the generators. Haystack is imported only when a call builds its
first ``ChatMessage``, so the module loads without ``haystack-ai`` installed.
"""

from __future__ import annotations

import asyncio
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
    from collections.abc import Callable, Mapping, Sequence

    from agno.models.response import ToolExecution

    from simulhausen.types import Message

__all__ = ["haystack_adapter"]

_ADAPTER = "haystack_adapter"
# Output sockets that hold chat messages: the Agent's ``messages``, a chat generator's ``replies``.
_MESSAGE_SOCKETS = ("messages", "replies", "last_message")


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role))


def _same_call(call: Any, origin: Any) -> bool:
    """Whether a ``ToolCallResult.origin`` is this ``ToolCall``: by id when either has one, else by content."""
    call_id, origin_id = getattr(call, "id", None), getattr(origin, "id", None)
    if call_id is not None or origin_id is not None:
        return call_id == origin_id
    return call is origin or (
        getattr(call, "tool_name", None) == getattr(origin, "tool_name", None)
        and getattr(call, "arguments", None) == getattr(origin, "arguments", None)
    )


def _result(value: Any) -> str | None:
    """A ``ToolCallResult.result``: a string, or a list of content parts whose text parts are joined."""
    if isinstance(value, (list, tuple)):
        texts = [part.text for part in value if isinstance(getattr(part, "text", None), str)]
        if texts:
            return "".join(texts)
    return result_text(value)


def _tools(messages: Sequence[Any]) -> list[ToolExecution] | None:
    """Tool calls of the turn: ``ChatMessage.tool_calls`` paired with ``tool_call_results``.

    Calls come in the order the model asked for them. A call without a result
    did not run. A result whose call is not among the messages (a pipeline
    that outputs only the tool messages) is reported from its ``origin``.
    """
    calls: list[Any] = []
    results: list[Any] = []
    for message in messages:
        calls.extend(getattr(message, "tool_calls", None) or [])
        results.extend(getattr(message, "tool_call_results", None) or [])
    executions: list[ToolExecution] = []
    for call in calls:
        match = next((result for result in results if _same_call(call, getattr(result, "origin", None))), None)
        if match is None:
            executions.append(tool_execution(call.tool_name, call.arguments, executed=False, call_id=call.id))
            continue
        results.remove(match)
        executions.append(
            tool_execution(
                call.tool_name, call.arguments, _result(match.result), error=bool(match.error), call_id=call.id
            )
        )
    for result in results:
        origin = result.origin
        executions.append(
            tool_execution(
                origin.tool_name, origin.arguments, _result(result.result), error=bool(result.error), call_id=origin.id
            )
        )
    return executions or None


def _usage(messages: Sequence[Any]) -> Usage | None:
    """``meta["usage"]`` summed over the assistant messages; each one that reports usage is one model request."""
    total: Usage | None = None
    for message in messages:
        meta = getattr(message, "meta", None)
        reported = meta.get("usage") if isinstance(meta, dict) else None
        usage = usage_of(reported, requests=1) if reported else None
        if usage is not None:
            total = usage if total is None else total + usage
    return total


def _reply(messages: Sequence[Any]) -> Any:
    """The message that answers the user: the last assistant message with text, else the last message."""
    for message in reversed(messages):
        if _role(message) == "assistant" and getattr(message, "text", None):
            return message
    return messages[-1] if messages else None


def _text(message: Any) -> str:
    """Text of the reply message; a tool message (the Agent stopped on a tool) gives its result."""
    if message is None:
        return ""
    text = getattr(message, "text", None)
    if text:
        return str(text)
    results = getattr(message, "tool_call_results", None) or []
    return (_result(results[-1].result) or "") if results else ""


def _produced(sent: Sequence[Any], output: Sequence[Any]) -> list[Any]:
    """The messages of ``output`` that the run added.

    The Agent returns the whole conversation (its system prompt, the messages
    it was given, what it produced), a chat generator only its replies: what
    follows the last message that was sent is new, and everything is new when
    that message is not there.
    """
    last = sent[-1] if sent else None
    for index in range(len(output) - 1, -1, -1):
        if output[index] is last or output[index] == last:
            return list(output[index + 1 :])
    return list(output)


class _HaystackAdapter(FrameworkAdapter):
    """Chat messages -> ``Agent.run_async`` / ``Pipeline.run_async`` -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        input_component: str | None,
        input_socket: str,
        output_component: str | None,
        output_socket: str | None,
        inputs: Mapping[str, Any] | None,
        run_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._input_component = input_component
        self._input_socket = input_socket
        self._output_component = output_component
        self._output_socket = output_socket
        self._inputs = dict(inputs or {})
        self._run_kwargs = run_kwargs
        # thread_id -> the ChatMessage list as it stood after the agent's reply.
        self._conversations: Conversations[list[Any]] = Conversations()

    def _is_target(self, target: Any) -> bool:
        return not isinstance(target, type) and (hasattr(target, "run") or hasattr(target, "run_async"))

    def _chat_messages(self, messages: Sequence[Message]) -> list[Any]:
        chat = require("haystack.dataclasses", package="haystack-ai", adapter=_ADAPTER).ChatMessage
        return [
            chat.from_user(message["content"]) if message["role"] == "user" else chat.from_assistant(message["content"])
            for message in messages
        ]

    async def _run(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        run_async = getattr(target, "run_async", None)
        if callable(run_async):
            return await run_async(*args, **kwargs)
        return await asyncio.to_thread(target.run, *args, **kwargs)

    async def _invoke(self, target: Any, messages: list[Any]) -> list[Any]:
        """Run the target and return the chat messages of its output."""
        if not hasattr(target, "add_component"):
            result = await self._run(target, messages=messages, **self._run_kwargs)
            outputs = [result]
        else:
            data: dict[str, Any] = dict(self._inputs)
            if self._input_component is None:
                data[self._input_socket] = messages
            else:
                data[self._input_component] = {**data.get(self._input_component, {}), self._input_socket: messages}
            result = await self._run(target, data, **self._run_kwargs)
            outputs = list(result.values()) if self._output_component is None else [result.get(self._output_component)]
        sockets = _MESSAGE_SOCKETS if self._output_socket is None else (self._output_socket,)
        for output in outputs:
            for socket in sockets:
                value = output.get(socket) if isinstance(output, dict) else None
                if value:
                    return list(value) if isinstance(value, (list, tuple)) else [value]
        return []

    async def call(self, input: AgentInput) -> AgentTurn:
        target = self._resolve()
        carried = self._conversations.recall(input) if self._history == "session" else None
        if carried is None:
            sent = self._chat_messages(input.messages)
        else:
            sent = [*carried, *self._chat_messages(input.new_messages)]
        produced = _produced(sent, await self._invoke(target, sent))
        reply = _reply(produced)
        content = self._content_fn(reply) if self._content_fn is not None else _text(reply)
        if self._history == "session":
            self._conversations.remember(input, content, [*sent, *produced])
        return AgentTurn(content=content, tools=_tools(produced), usage=_usage(produced))


def haystack_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    input_component: str | None = None,
    input_socket: str = "messages",
    output_component: str | None = None,
    output_socket: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a Haystack ``Agent`` or a pipeline with a chat interface (``pip install haystack-ai``).

    ``target`` is the ``Agent`` component (anything with
    ``run_async(messages=[ChatMessage, ...])`` returning ``{"messages": [...]}``),
    a ``Pipeline`` / ``AsyncPipeline``, or a zero-argument factory returning
    one. A turn is one ``run_async`` call; a target that only has ``run`` is
    called in a worker thread. Extra kwargs go to that call unchanged: for
    the Agent ``generation_kwargs=...``, ``tools=...`` or its state inputs,
    for a pipeline ``include_outputs_from=...``.

    A pipeline receives the messages as ``{input_socket: messages}``, which
    Haystack hands to every component with a free input of that name, or as
    ``{input_component: {input_socket: messages}}`` when ``input_component``
    is given. ``inputs`` holds whatever else the pipeline needs on every run
    (template variables, a retriever's ``top_k``). The chat messages are read
    from ``output_component`` (default: the first component whose output has
    them) at ``output_socket`` (default: ``messages``, then ``replies``, then
    ``last_message``). List the components you want to see in
    ``include_outputs_from`` when the tool messages are not in a final output.

    Haystack components keep no conversation, so the adapter does.
    ``history="session"`` (default) carries the ``ChatMessage`` list between
    turns, tool calls and tool results included, and appends the new user
    message; after a rollback or a scripted ``sim.agent("...")`` reply it
    starts again from the transcript. ``history="transcript"`` rebuilds the
    list from the transcript on every call, so earlier tool results are not
    replayed. The Agent's own system prompt is not carried: the Agent adds it
    on each run.

    The reply is the text of the last assistant message the run produced (the
    result of the last tool message when the Agent stopped on a tool);
    ``content_fn`` gets that ``ChatMessage`` instead. Tool calls are the
    ``tool_calls`` of the new messages paired with ``tool_call_results`` by
    id, or by tool name and arguments when the model gave no id, with the
    result's ``error`` flag; a call without a result is recorded as not
    executed. Usage is ``meta["usage"]`` summed over the new messages.

    >>> adapter = haystack_adapter(lambda: support_agent)
    >>> adapter = haystack_adapter(chat_pipeline, input_component="llm", inputs={"retriever": {"top_k": 3}})
    """
    return _HaystackAdapter(
        target,
        history=history,
        content_fn=content_fn,
        input_component=input_component,
        input_socket=input_socket,
        output_component=output_component,
        output_socket=output_socket,
        inputs=inputs,
        run_kwargs=run_kwargs,
    )
