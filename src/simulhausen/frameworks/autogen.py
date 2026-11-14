"""Adapter for Microsoft AutoGen AgentChat: an ``AssistantAgent`` or a team.

``autogen_adapter`` drives anything with AgentChat's task-runner interface
(``await target.run(task=...)`` returning a ``TaskResult``): a single agent,
``RoundRobinGroupChat``, ``SelectorGroupChat``, ``Swarm`` and the other group
chats. AutoGen is imported only when a conversation has to be reset or
replayed, so the module loads without ``autogen-agentchat`` installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    require,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

    from simulhausen.types import Message

__all__ = ["autogen_adapter"]

_ADAPTER = "autogen_adapter"
# The ``source`` AgentChat messages carry for the two sides of a replayed transcript.
_USER_SOURCE = "user"
_ASSISTANT_SOURCE = "assistant"


def _tools(messages: Sequence[Any]) -> list[ToolExecution] | None:
    """Tool calls of the run: ``ToolCallRequestEvent`` items paired with ``ToolCallExecutionEvent`` items.

    Both events hold a list in ``content``: ``FunctionCall`` (``id``, ``name``,
    ``arguments``) and ``FunctionExecutionResult`` (``call_id``, ``content``,
    ``is_error``). They are paired by call id and reported in request order,
    whichever agent of a team made them; handoffs appear too, since AgentChat
    runs them as ``transfer_to_<agent>`` tool calls. A request without a
    result did not run.
    """
    calls: list[Any] = []
    results: dict[str, Any] = {}
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for item in content:
            if hasattr(item, "call_id"):
                results[str(item.call_id)] = item
            elif hasattr(item, "arguments") and hasattr(item, "name"):
                calls.append(item)
    executions: list[ToolExecution] = []
    for call in calls:
        call_id = getattr(call, "id", None)
        result = results.get(str(call_id))
        if result is None:
            executions.append(tool_execution(call.name, call.arguments, executed=False, call_id=call_id))
        else:
            executions.append(
                tool_execution(
                    call.name,
                    call.arguments,
                    getattr(result, "content", None),
                    error=bool(getattr(result, "is_error", False)),
                    call_id=call_id,
                )
            )
    return executions or None


def _usage(messages: Sequence[Any]) -> Usage | None:
    """``models_usage`` summed over the messages; every message that carries one is one model request."""
    total: Usage | None = None
    for message in messages:
        usage = usage_of(getattr(message, "models_usage", None), requests=1)
        if usage is not None:
            total = usage if total is None else total + usage
    return total


def _reply(messages: Sequence[Any]) -> Any:
    """Content of the last chat message of the run.

    Chat messages (``TextMessage``, ``ToolCallSummaryMessage``,
    ``StructuredMessage``...) have ``to_model_text``; events such as tool call
    requests do not. A ``HandoffMessage`` only says who speaks next, so it is
    the reply only when the run produced no other chat message.
    """
    chat = [message for message in messages if hasattr(message, "to_model_text")]
    spoken = [message for message in chat if not hasattr(message, "target")] or chat
    return spoken[-1].content if spoken else None


class _AutoGenAdapter(FrameworkAdapter):
    """New user messages -> ``target.run(task=...)`` -> ``AgentTurn``, one target state per thread."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        run_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._run_kwargs = run_kwargs
        # thread_id -> the agent or team that serves it (a factory builds one per thread).
        self._targets: dict[str, Any] = {}
        # id(target) -> the thread whose conversation the target holds right now.
        self._loaded: dict[int, str] = {}
        # thread_id -> ``save_state()`` of a thread that had to make room on a shared target.
        self._saved: dict[str, Any] = {}
        # Threads whose target state still matches the transcript.
        self._conversations: Conversations[bool] = Conversations()

    def _is_target(self, target: Any) -> bool:
        return not isinstance(target, type) and hasattr(target, "run") and hasattr(target, "save_state")

    async def _reset(self, target: Any) -> None:
        """Back to the initial state: ``Team.reset()``, or ``ChatAgent.on_reset(...)`` for a single agent."""
        reset = getattr(target, "reset", None)
        if callable(reset):
            await reset()
            return
        core = require("autogen_core", package="autogen-agentchat", adapter=_ADAPTER)
        await target.on_reset(core.CancellationToken())

    async def _checkout(self, thread_id: str) -> tuple[Any, bool]:
        """The target holding ``thread_id``'s conversation, and whether it is in its initial state.

        A factory gives every thread its own target. A target shared between
        threads (one instance, or a factory that returns the same object) is
        switched over: the conversation it holds is put aside with
        ``save_state()`` and the wanted one comes back with ``load_state()``,
        or the target is reset when the thread is new.
        """
        target = self._targets.get(thread_id)
        if target is None:
            source = self._target
            target = source if self._is_target(source) or not callable(source) else source()
            self._targets[thread_id] = target
        holder = self._loaded.get(id(target))
        if holder == thread_id:
            return target, False
        self._loaded[id(target)] = thread_id
        if holder is None:
            return target, True
        self._saved[holder] = await target.save_state()
        saved = self._saved.pop(thread_id, None)
        if saved is None:
            await self._reset(target)
            return target, True
        await target.load_state(saved)
        return target, False

    def _messages(self, messages: Sequence[Message]) -> list[Any]:
        """Transcript messages as AgentChat ``TextMessage`` objects."""
        chat = require("autogen_agentchat.messages", package="autogen-agentchat", adapter=_ADAPTER)
        return [
            chat.TextMessage(
                content=message["content"], source=_USER_SOURCE if message["role"] == "user" else _ASSISTANT_SOURCE
            )
            for message in messages
        ]

    async def _replay(self, target: Any, messages: Sequence[Message]) -> list[Any]:
        """Put the transcript into a freshly reset target; returns the task still to send.

        An agent with a ``model_context`` (``AssistantAgent``) gets the earlier
        exchanges written into that context with their real roles, and the
        trailing user messages become the task. A team has no context of its
        own, so it receives the whole transcript as the task: its participants
        see the earlier replies as messages from a source named ``assistant``.
        """
        split = len(messages)
        while split and messages[split - 1]["role"] == "user":
            split -= 1
        context = getattr(target, "model_context", None)
        if context is None or split in (0, len(messages)):
            return self._messages(messages)
        models = require("autogen_core.models", package="autogen-agentchat", adapter=_ADAPTER)
        name = getattr(target, "name", _ASSISTANT_SOURCE)
        for message in messages[:split]:
            if message["role"] == "user":
                await context.add_message(models.UserMessage(content=message["content"], source=_USER_SOURCE))
            else:
                await context.add_message(models.AssistantMessage(content=message["content"], source=name))
        return self._messages(messages[split:])

    async def call(self, input: AgentInput) -> AgentTurn:
        target, clean = await self._checkout(input.thread_id)
        if self._history == "session" and self._conversations.recall(input):
            task = self._messages(input.new_messages)
        else:
            if not clean:
                await self._reset(target)
            task = await self._replay(target, input.messages)
        # Whatever happens in the run, the target's state has moved on from the stored transcript.
        self._conversations.forget(input.thread_id)
        result = await target.run(task=task, **{"output_task_messages": False, **self._run_kwargs})
        messages = list(getattr(result, "messages", None) or [])
        content = self._render(_reply(messages))
        if self._history == "session":
            self._conversations.remember(input, content, True)
        return AgentTurn(content=content, tools=_tools(messages), usage=_usage(messages))


def autogen_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for an AutoGen AgentChat agent or team (``pip install autogen-agentchat``).

    ``target`` is an ``AssistantAgent`` (or any ``ChatAgent``), a team
    (``RoundRobinGroupChat``, ``SelectorGroupChat``, ``Swarm``...) or a
    zero-argument factory returning one. Every turn is one
    ``await target.run(task=...)``. Extra kwargs go to ``run`` unchanged
    (``cancellation_token=...``); ``output_task_messages`` defaults to
    ``False`` so the task is not echoed back as the reply.

    AgentChat agents and teams keep the conversation inside themselves, so the
    adapter keeps one target per ``thread_id``. Pass a factory to get that for
    real: each scenario then talks to its own agent or team, and scenarios may
    run in parallel. A single instance is shared instead: when another thread
    takes over, the adapter stores the current conversation with
    ``save_state()`` and resets the target (``Team.reset()``,
    ``ChatAgent.on_reset()``), and ``load_state()`` brings a stored
    conversation back. That works for scenarios that run one after another,
    not at the same time, and anything put into the instance by hand before
    the first turn is lost at the first switch.

    ``history="session"`` (default) sends only the new user messages; the
    target's own state supplies the rest, tool calls of earlier turns
    included. After a rollback or a scripted ``sim.agent("...")`` reply that
    state no longer matches the dialogue, so the target is reset and the
    transcript is replayed into it. ``history="transcript"`` does that on
    every call: an ``AssistantAgent`` gets the earlier exchanges written into
    its ``model_context`` and the last user message as the task; a team gets
    the whole transcript as a list of ``TextMessage`` (replies from a source
    named ``assistant``). Earlier tool results are then not replayed.

    The reply is the content of the last chat message of the run (the last
    ``TextMessage`` of the agent side; a ``HandoffMessage`` only when nothing
    else was said), through ``content_fn`` when given: a ``StructuredMessage``
    holds a Pydantic model, which is rendered as JSON otherwise. Tool calls
    come from ``ToolCallRequestEvent`` / ``ToolCallExecutionEvent`` paired by
    call id, every team member's calls and ``transfer_to_<agent>`` handoffs
    included; a request without a result is recorded as not executed. Usage is
    ``models_usage`` summed over the run's messages. A team reports a failing
    participant as its own ``RuntimeError``, which propagates.

    >>> adapter = autogen_adapter(lambda: build_support_team())
    """
    return _AutoGenAdapter(target, history=history, content_fn=content_fn, run_kwargs=run_kwargs)
