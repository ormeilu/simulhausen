"""Adapter for LlamaIndex agent workflows: ``FunctionAgent``, ``ReActAgent``, ``AgentWorkflow``.

``llamaindex_adapter`` runs an agent from ``llama_index.core.agent.workflow``
inside the test process: ``agent.run(user_msg=..., chat_history=...)``, the
events of the run for the tool calls and the token usage, the awaited result
for the reply. LlamaIndex is not imported at module import time; the adapter
reads the handler, the events and the chat messages by their attributes, and
only resuming a run that waits for a human imports ``HumanResponseEvent``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    maybe_await,
    require,
    text_of,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Usage

if TYPE_CHECKING:
    from collections.abc import Callable

    from agno.models.response import ToolExecution

__all__ = ["llamaindex_adapter"]


def _field(source: Any, name: str) -> Any:
    return source.get(name) if isinstance(source, dict) else getattr(source, name, None)


def _waits_for_input(event: Any) -> bool:
    """Whether the event is an ``InputRequiredEvent`` (or a subclass): the run waits for a human."""
    return any(cls.__name__ == "InputRequiredEvent" for cls in type(event).__mro__)


def _reported_usage(raw: Any) -> Usage | None:
    """Token usage of one model call, from the provider response an ``AgentOutput`` keeps in ``raw``."""
    if raw is None:
        return None
    return usage_of(_field(raw, "usage") or _field(raw, "usage_metadata"), requests=1)


class _Run:
    """What the events of one run say: tool calls in call order, usage, the event the run waits on."""

    def __init__(self) -> None:
        # [call id, name, arguments, ToolOutput or None] per call.
        self._calls: list[list[Any]] = []
        self.usage: Usage | None = None
        self.waiting: Any = None

    def see(self, event: Any) -> None:
        output = getattr(event, "tool_output", None)
        if output is not None:
            # ToolCallResult: closes the call with the same id, or stands alone when no ToolCall was shown.
            for call in self._calls:
                if call[3] is None and call[:2] == [event.tool_id, event.tool_name]:
                    call[3] = output
                    return
            self._calls.append([event.tool_id, event.tool_name, event.tool_kwargs, output])
        elif hasattr(event, "tool_kwargs") and hasattr(event, "tool_id"):
            # ToolCall. A resumed run announces its pending call a second time.
            if not any(call[3] is None and call[:2] == [event.tool_id, event.tool_name] for call in self._calls):
                self._calls.append([event.tool_id, event.tool_name, event.tool_kwargs, None])
        elif hasattr(event, "raw") and hasattr(event, "response") and not hasattr(event, "delta"):
            # AgentOutput: one finished model call of some agent of the workflow (AgentStream is its chunks).
            reported = _reported_usage(event.raw)
            if reported is not None:
                self.usage = reported if self.usage is None else self.usage + reported

    def tools(self) -> list[ToolExecution]:
        executions: list[ToolExecution] = []
        for call_id, name, arguments, output in self._calls:
            result = getattr(output, "content", None) or getattr(output, "raw_output", None)
            executions.append(
                tool_execution(
                    name,
                    arguments,
                    result,
                    error=bool(getattr(output, "is_error", False)),
                    executed=output is not None,
                    call_id=call_id or None,
                )
            )
        return executions


class _LlamaIndexAdapter(FrameworkAdapter):
    """Transcript -> ``agent.run`` -> events and result -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        resume_fn: Callable[[AgentInput], Any] | None,
        run_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._resume_fn = resume_fn
        self._run_kwargs = run_kwargs
        # Per thread: the chat history the last run left in the agent's memory (a list),
        # or the handler of a run that waits for a human.
        self._threads: Conversations[Any] = Conversations()
        self._paused: dict[str, Any] = {}

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "run")

    async def call(self, input: AgentInput) -> AgentTurn:
        agent = self._resolve()
        session = self._history == "session"
        known = self._threads.recall(input) if session else None
        paused = self._paused.pop(input.thread_id, None)
        if known is not None and not isinstance(known, list):
            # The last turn stopped on a question to the human: the new user message answers it.
            handler = known
            handler.ctx.send_event(self._answer(input))
        else:
            if paused is not None:
                # The transcript moved away from the question (a rollback, a scripted reply).
                await maybe_await(paused.cancel_run())
            pending: list[Any] = [
                {"role": message["role"], "content": message["content"]}
                for message in (input.messages if known is None else input.new_messages)
            ]
            user_msg = pending.pop()["content"] if pending and pending[-1]["role"] == "user" else None
            chat_history = [*(known or []), *pending]
            handler = agent.run(user_msg=user_msg, chat_history=chat_history or None, **self._run_kwargs)

        run = _Run()
        async for event in handler.stream_events():
            if _waits_for_input(event):
                run.waiting = event
                break
            run.see(event)

        if run.waiting is not None:
            content = text_of(getattr(run.waiting, "prefix", None))
            if session:
                self._paused[input.thread_id] = handler
                self._threads.remember(input, content, handler)
            else:
                await maybe_await(handler.cancel_run())
            return AgentTurn(content=content, tools=run.tools() or None, usage=run.usage)

        result = await handler
        if self._content_fn is not None:
            content = self._content_fn(result)
        else:
            response = getattr(result, "response", None)
            content = text_of(getattr(response, "content", None)) if response is not None else text_of(result)
        if session:
            messages = await self._memory_of(handler)
            if messages is None:
                self._threads.forget(input.thread_id)
            else:
                self._threads.remember(input, content, messages)
        return AgentTurn(content=content, tools=run.tools() or None, usage=run.usage)

    def _answer(self, input: AgentInput) -> Any:
        if self._resume_fn is not None:
            return self._resume_fn(input)
        events = require("llama_index.core.workflow", package="llama-index-core", adapter="llamaindex_adapter")
        return events.HumanResponseEvent(response=input.last_new_user_message_str())

    @staticmethod
    async def _memory_of(handler: Any) -> list[Any] | None:
        """Every message the run left in the agent's memory, tool calls and results included."""
        store = getattr(getattr(handler, "ctx", None), "store", None)
        memory = await store.get("memory", default=None) if store is not None else None
        if memory is None:
            return None
        return list(await memory.aget_all())


def llamaindex_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    resume_fn: Callable[[AgentInput], Any] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a LlamaIndex ``FunctionAgent``, ``ReActAgent`` or ``AgentWorkflow``.

    ``target`` is the agent or workflow itself or a lazy factory
    (``lambda: build_agent()`` with the import inside, so collecting tests
    builds nothing). Every turn is one ``target.run(user_msg=...,
    chat_history=...)``; extra keyword arguments go to ``run`` unchanged
    (``max_iterations=...``, ``early_stopping_method=...``).

    The reply is the text of the final ``AgentOutput.response``. ``content_fn``
    renders it from the run's result instead, for example from
    ``structured_response`` of an agent with ``output_cls``.

    Tool calls come from the events of the run, so the tools of every agent
    of an ``AgentWorkflow`` are recorded, the ``handoff`` call between them
    included: a ``ToolCallResult`` is an executed call, ``tool_output.is_error``
    marks a failed one (LlamaIndex turns a tool exception into such a result
    and lets the model go on). Usage is read from the provider response each
    ``AgentOutput`` keeps in ``raw`` (its ``usage`` or ``usage_metadata``),
    summed over the model calls of the turn; a provider that reports none
    there gives no usage.

    ``history="session"`` (the default) keeps the agent's context between
    turns, tool calls and results included: after a run the adapter takes the
    messages the run left in the agent's memory and passes them as
    ``chat_history`` of the next one. When the transcript no longer matches
    what the agent saw (a rollback, a reply injected with ``sim.agent("...")``),
    the history is rebuilt from the transcript. ``history="transcript"`` does
    that on every call, so earlier tool results are not replayed.

    A run that waits for a human (a tool calling ``ctx.wait_for_event`` with
    an ``InputRequiredEvent``) ends the turn: the event's ``prefix`` is the
    reply and the pending call is recorded as not executed, so
    ``state.has_tool_call`` does not count it. In session mode the next user
    message resumes the same run as ``HumanResponseEvent(response=<message>)``;
    ``resume_fn`` builds another event from the ``AgentInput`` when the tool
    waits for a custom one. In transcript mode the waiting run is cancelled.

    >>> adapter = llamaindex_adapter(lambda: FunctionAgent(tools=[lookup_order], llm=llm), max_iterations=10)
    """
    return _LlamaIndexAdapter(
        target, history=history, content_fn=content_fn, resume_fn=resume_fn, run_kwargs=run_kwargs
    )
