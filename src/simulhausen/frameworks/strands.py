"""Adapter for AWS Strands Agents (``strands-agents``).

``strands_adapter`` runs a Strands ``Agent``, or a ``Swarm`` / ``Graph``, inside
the test process: ``invoke_async`` per turn, the tool calls read from the
messages the turn added, the token usage from the result.
"""

from __future__ import annotations

import copy
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    require,
    tool_execution,
    transcript_prompt,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Message, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["strands_adapter"]

_ADAPTER = "strands_adapter"
_PACKAGE = "strands-agents"

# Messages of the nodes a Swarm or Graph ran during the turn in progress, in
# execution order. A context variable, so concurrent scenarios that share one
# orchestrator (and its hook) do not see each other's nodes.
_NODE_RUNS: ContextVar[list[list[Any]] | None] = ContextVar("simulhausen_strands_node_runs", default=None)


def _after_node(event: Any) -> None:
    """Hook on ``AfterNodeCallEvent``: keep what the node's agent said before the orchestrator resets it."""
    runs = _NODE_RUNS.get()
    if runs is None:
        return
    node = (getattr(event.source, "nodes", None) or {}).get(event.node_id)
    messages = getattr(getattr(node, "executor", None), "messages", None)
    if messages is not None:
        preset = len(getattr(node, "_initial_messages", None) or [])
        runs.append(list(messages[preset:]))


def _native(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Transcript messages in Strands' shape: a role and a list of content blocks."""
    return [{"role": message["role"], "content": [{"text": message["content"]}]} for message in messages]


def _split(input: AgentInput, messages: Sequence[Message]) -> tuple[list[Message], str]:
    """``messages`` as the part that goes into ``agent.messages`` and the user message to send now."""
    if messages and messages[-1]["role"] == "user":
        return list(messages[:-1]), messages[-1]["content"]
    return list(messages), input.last_new_user_message_str()


def _result_value(content: Any) -> Any:
    """The content blocks of a ``toolResult`` as one value: joined text, the JSON of a lone block, else as is."""
    if not isinstance(content, list):
        return content
    texts = [block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)]
    if len(texts) == len(content):
        return "\n".join(texts)
    if len(content) == 1 and isinstance(content[0], dict) and "json" in content[0]:
        return content[0]["json"]
    return content


def _tools(messages: Sequence[Any]) -> list[ToolExecution]:
    """``toolUse`` blocks paired with their ``toolResult`` blocks by ``toolUseId``, in call order.

    A ``toolUse`` without a result did not run: the agent stopped on an
    interrupt before the tool was executed.
    """
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        for block in message.get("content") or []:
            use = block.get("toolUse")
            if isinstance(use, dict):
                calls[str(use.get("toolUseId"))] = {"use": use, "result": None}
            result = block.get("toolResult")
            if isinstance(result, dict):
                call = calls.setdefault(str(result.get("toolUseId")), {"use": {}, "result": None})
                call["result"] = result
    tools: list[ToolExecution] = []
    for call_id, call in calls.items():
        use, result = call["use"], call["result"]
        tools.append(
            tool_execution(
                use.get("name"),
                use.get("input"),
                _result_value(result.get("content")) if result is not None else None,
                error=result is not None and result.get("status") == "error",
                executed=result is not None,
                call_id=call_id,
            )
        )
    return tools


def _request_usage(result: Any) -> Usage | None:
    """Usage of the whole request from the result's invocation context, nested agent calls included."""
    usage = getattr(getattr(result, "invocation_context", None), "usage", None)
    if not usage:
        return None
    turns = usage.get("turns_taken") if isinstance(usage, dict) else None
    return usage_of(usage, requests=turns if isinstance(turns, int) else None)


def _usage_between(before: Any, after: Any) -> Usage | None:
    """What ``accumulated_usage`` grew by during the turn: an agent's metrics count from its creation."""
    if not isinstance(after, dict):
        return None
    start = before if isinstance(before, dict) else {}
    keys = ("inputTokens", "outputTokens", "totalTokens")
    return usage_of({key: (after.get(key) or 0) - (start.get(key) or 0) for key in keys})


def _text(result: Any) -> str:
    """The framework's own rendering of a result, without the newline it puts after every text block."""
    return str(result).rstrip("\n")


def _last_node_text(result: Any) -> str:
    """The reply of a ``Swarm`` / ``Graph`` run: what the node that ran last said."""
    order = getattr(result, "node_history", None) or getattr(result, "execution_order", None) or []
    results = getattr(result, "results", None) or {}
    last = results.get(getattr(order[-1], "node_id", None)) if order else None
    return _text(last.result) if last is not None else _text(result)


class _StrandsAdapter(FrameworkAdapter):
    """Last user message -> ``agent.invoke_async`` with the thread's messages -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        invoke_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._invoke_kwargs = invoke_kwargs
        # Per thread (a factory target) or one for all under the key None: the object and its preset messages.
        self._targets: dict[str | None, tuple[Any, list[Any]]] = {}
        self._histories: Conversations[list[Any]] = Conversations()
        self._watched: list[Any] = []

    def _is_target(self, target: Any) -> bool:
        # An Agent is callable itself, so a factory is what has no invoke_async.
        return hasattr(target, "invoke_async") or not callable(target)

    def _target_for(self, thread_id: str) -> tuple[Any, list[Any]]:
        """The agent of the thread and the messages it was built with: a factory builds one per thread."""
        key = None if self._is_target(self._target) else thread_id
        if key not in self._targets:
            target = self._target if key is None else self._target()
            self._targets[key] = (target, copy.deepcopy(list(getattr(target, "messages", None) or [])))
        return self._targets[key]

    def _watch(self, orchestrator: Any) -> None:
        """Register the node hook on a Swarm / Graph and on the orchestrators nested in its nodes, once each."""
        if any(seen is orchestrator for seen in self._watched):
            return
        self._watched.append(orchestrator)
        event = require("strands.hooks", package=_PACKAGE, adapter=_ADAPTER).AfterNodeCallEvent
        orchestrator.add_hook(_after_node, event)
        for node in (getattr(orchestrator, "nodes", None) or {}).values():
            executor = getattr(node, "executor", None)
            if hasattr(executor, "add_hook") and hasattr(executor, "nodes"):
                self._watch(executor)

    async def call(self, input: AgentInput) -> AgentTurn:
        target, preset = self._target_for(input.thread_id)
        if not hasattr(target, "messages"):
            return await self._call_orchestrator(target, input)
        stored = self._histories.recall(input) if self._history == "session" else None
        if stored is not None:
            earlier, prompt = _split(input, input.new_messages)
            if target.messages is not stored:
                # One agent object served another thread in between: put this thread's messages back.
                target.messages = stored
            target.messages.extend(_native(earlier))
        else:
            # A new thread, transcript mode, or a transcript that no longer matches (a rollback, a scripted reply).
            earlier, prompt = _split(input, input.messages)
            target.messages = [*copy.deepcopy(preset), *_native(earlier)]
        known = {id(message) for message in target.messages}
        before = copy.copy(getattr(getattr(target, "event_loop_metrics", None), "accumulated_usage", None))
        result = await target.invoke_async(prompt, **self._invoke_kwargs)
        added = [message for message in target.messages if id(message) not in known]
        content = self._content_fn(result) if self._content_fn is not None else _text(result)
        usage = _request_usage(result) or _usage_between(
            before, getattr(getattr(result, "metrics", None), "accumulated_usage", None)
        )
        if self._history == "session":
            self._histories.remember(input, content, target.messages)
        return AgentTurn(content=content, tools=_tools(added) or None, usage=usage)

    async def _call_orchestrator(self, target: Any, input: AgentInput) -> AgentTurn:
        """A Swarm or Graph resets its agents on every run, so the transcript is the task in both modes."""
        self._watch(target)
        runs: list[list[Any]] = []
        token = _NODE_RUNS.set(runs)
        try:
            result = await target.invoke_async(transcript_prompt(input.messages), **self._invoke_kwargs)
        finally:
            _NODE_RUNS.reset(token)
        content = self._content_fn(result) if self._content_fn is not None else _last_node_text(result)
        usage = _request_usage(result) or usage_of(getattr(result, "accumulated_usage", None))
        tools = [tool for messages in runs for tool in _tools(messages)]
        return AgentTurn(content=content, tools=tools or None, usage=usage)


def strands_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    **invoke_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a Strands ``Agent``, ``Swarm`` or ``Graph``.

    ``target`` is the object itself or a zero-argument factory. A Strands agent
    owns its conversation in ``agent.messages``, so the two differ: a factory
    is called once per ``thread_id`` and every scenario gets an agent of its
    own; a single agent object is shared, and the adapter swaps
    ``agent.messages`` to the thread it is serving. That keeps scenarios that
    run one after another apart, but the agent's ``state``, session manager
    and metrics stay shared, and scenarios running concurrently on one agent
    would overwrite each other's messages. Pass a factory when that matters.
    Extra kwargs go to ``invoke_async`` unchanged: ``invocation_state=...``,
    ``structured_output_model=...``.

    ``history="session"`` sends only the newest user message; the agent keeps
    its messages across turns, tool uses and results included. When the
    transcript stops matching them (a rollback, a reply scripted with
    ``sim.agent("text")``), ``agent.messages`` is rebuilt from the messages
    the agent was created with plus the transcript. ``history="transcript"``
    rebuilds it that way on every call, so earlier tool results are gone.

    Tool calls are the ``toolUse`` blocks of the messages the turn added,
    paired with their ``toolResult`` blocks by ``toolUseId``; a result with
    ``status == "error"`` (Strands catches a raising tool) is an error and a
    ``toolUse`` without a result (an interrupt) is recorded as not executed.
    The calls an agent used as a tool makes inside are not in the outer
    agent's messages and are not listed. Usage is the request's usage from
    ``result.invocation_context`` (nested agent calls included), or what
    ``result.metrics.accumulated_usage`` grew by during the turn.

    A ``Swarm`` or ``Graph`` resets its agents at the start of every run and
    has no memory of its own, so in both history modes the task is the whole
    transcript as text. The reply is what the last executed node said, tool
    calls come from every executed node's agent in execution order (handoffs
    included), usage is the run's total.

    The reply is ``str(result)``, which is Strands' own rendering (structured
    output as JSON, interrupts as a list); ``content_fn`` gets the
    ``AgentResult`` / ``MultiAgentResult`` instead.

    >>> adapter = strands_adapter(lambda: Agent(model=model, tools=[lookup_order]))
    """
    return _StrandsAdapter(target, history=history, content_fn=content_fn, invoke_kwargs=invoke_kwargs)
