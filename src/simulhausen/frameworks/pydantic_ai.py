"""Adapter for Pydantic AI agents (``pip install pydantic-ai-slim`` or ``pydantic-ai``).

``pydantic_ai_adapter`` runs ``agent.run(prompt, message_history=...)`` once
per turn and reads the reply from ``result.output``, the tool calls from
``result.new_messages()`` and the token usage from ``result.usage``.
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
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Message, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["pydantic_ai_adapter"]

_ADAPTER = "pydantic_ai_adapter"
_PACKAGE = "pydantic-ai-slim"

_CALL_KINDS = frozenset({"tool-call", "builtin-tool-call"})
# "retry-prompt" is the part older releases (and histories stored by them) use
# for a tool call the model has to repeat; current ones answer the call with a
# "tool-return" whose ``outcome`` is "retried".
_RETURN_KINDS = frozenset({"tool-return", "builtin-tool-return", "retry-prompt"})


def _split_transcript(messages: Sequence[Message]) -> tuple[list[Message], str | None]:
    """The transcript as (earlier messages, prompt): the trailing user messages are the prompt.

    Several user messages in a row are joined with a blank line. The prompt is
    ``None`` when the transcript does not end with a user message.
    """
    earlier = list(messages)
    tail: list[str] = []
    while earlier and earlier[-1]["role"] == "user":
        tail.append(earlier.pop()["content"])
    return earlier, "\n\n".join(reversed(tail)) if tail else None


def _system_parts(messages: Sequence[Any]) -> list[Any]:
    """The system prompt parts that open a native history."""
    first = messages[0] if messages else None
    return [part for part in getattr(first, "parts", None) or [] if getattr(part, "part_kind", None) == "system-prompt"]


def _native_history(messages: Sequence[Message], system_parts: Sequence[Any]) -> list[Any]:
    """The transcript as Pydantic AI messages, opened by the agent's system prompt when it is known."""
    native = require("pydantic_ai.messages", package=_PACKAGE, adapter=_ADAPTER)
    history: list[Any] = []
    for message in messages:
        if message["role"] == "user":
            history.append(native.ModelRequest(parts=[native.UserPromptPart(content=message["content"])]))
        else:
            history.append(native.ModelResponse(parts=[native.TextPart(content=message["content"])]))
    if history and system_parts:
        if getattr(history[0], "kind", None) == "request":
            history[0].parts = [*system_parts, *history[0].parts]
        else:
            history.insert(0, native.ModelRequest(parts=list(system_parts)))
    return history


def _tool_calls(result: Any) -> list[ToolExecution]:
    """The tool calls of this run, in the order the model asked for them.

    A call is paired with its return part by ``tool_call_id``. The return's
    ``outcome`` decides the status: "success" is a clean execution; "failed",
    "denied" and "retried" (and a retry prompt of an older release) are errors;
    "interrupted" and a call with no return at all (deferred, waiting for
    approval) never ran. The call of the output tool that carries a structured
    reply is not a tool call of the agent and is left out.
    """
    output_tool = getattr(result, "_output_tool_name", None)
    calls: list[Any] = []
    returns: dict[Any, Any] = {}
    for message in result.new_messages():
        for part in getattr(message, "parts", None) or []:
            kind = getattr(part, "part_kind", None)
            if kind in _CALL_KINDS:
                calls.append(part)
            elif kind in _RETURN_KINDS and getattr(part, "tool_name", None):
                returns[getattr(part, "tool_call_id", None)] = part
    executions: list[ToolExecution] = []
    for call in calls:
        if output_tool and call.tool_name == output_tool:
            continue
        call_id = getattr(call, "tool_call_id", None)
        returned = returns.get(call_id)
        if returned is None:
            executions.append(tool_execution(call.tool_name, call.args, executed=False, call_id=call_id))
            continue
        retry = getattr(returned, "part_kind", None) == "retry-prompt"
        outcome = "retried" if retry else getattr(returned, "outcome", "success")
        executions.append(
            tool_execution(
                call.tool_name,
                call.args,
                returned.content,
                error=outcome not in ("success", "interrupted"),
                executed=outcome != "interrupted",
                call_id=call_id,
            )
        )
    return executions


def _usage(result: Any) -> Usage | None:
    """Usage of the run; ``result.usage`` is a property in current releases and a method in older ones."""
    usage = getattr(result, "usage", None)
    if callable(usage) and not hasattr(usage, "input_tokens"):
        usage = usage()
    return usage_of(usage)


class _PydanticAIAdapter(FrameworkAdapter):
    """Newest user message plus the message history -> ``agent.run`` -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        run_kwargs: dict[str, Any],
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._run_kwargs = run_kwargs
        self._histories: Conversations[list[Any]] = Conversations()
        # The system prompt parts of the latest run that started without a
        # history: Pydantic AI adds them only to such a run, so a history
        # rebuilt from the transcript has to bring them along.
        self._system: list[Any] = []

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "run")

    async def call(self, input: AgentInput) -> AgentTurn:
        agent = self._resolve()
        prior = self._histories.recall(input) if self._history == "session" else None
        prompt: str | None
        if prior is not None:
            prompt = "\n\n".join(m["content"] for m in input.new_messages if m["role"] == "user") or None
        else:
            earlier, prompt = _split_transcript(input.messages)
            prior = _native_history(earlier, self._system) if earlier else None
        result = await agent.run(prompt, message_history=prior, **self._run_kwargs)
        content = self._render(result.output)
        if prior is None or self._history == "session":
            messages = list(result.all_messages())
            if prior is None:
                self._system = _system_parts(messages)
            if self._history == "session":
                self._histories.remember(input, content, messages)
        return AgentTurn(content=content, tools=_tool_calls(result) or None, usage=_usage(result))


def pydantic_ai_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a Pydantic AI ``Agent`` (or any ``AbstractAgent``, a ``WrapperAgent`` included).

    ``target`` is the agent or a zero-argument factory returning it (so
    collecting tests builds no agent). Extra kwargs go to ``agent.run``
    unchanged: ``deps=...``, ``model_settings=...``, ``usage_limits=...``,
    ``output_type=...`` and so on. Exceptions of the run, a plain exception
    raised by a tool included, propagate unchanged.

    ``history="session"`` (default) keeps ``result.all_messages()`` per
    ``thread_id`` and passes it back as ``message_history`` on the next turn,
    so the agent sees its earlier tool calls and their results. When the
    transcript no longer continues what the agent saw (a rollback, a reply
    injected with ``sim.agent("text")``), the history is rebuilt from the
    transcript as plain user and assistant messages. ``history="transcript"``
    rebuilds it that way on every call and remembers nothing, so a rollback
    reaches the agent.

    Pydantic AI adds ``system_prompt`` only to a run that starts without a
    history. The adapter keeps the system prompt parts of such a run and puts
    them in front of every history it rebuilds; when the very first call
    already has earlier messages there is nothing to copy, and only
    ``instructions`` (applied by Pydantic AI on every run) reach the model.

    ``content_fn`` renders ``result.output`` into the reply text. Without it a
    string is used as it is and a structured output (a Pydantic model, a
    dataclass, ``DeferredToolRequests``) becomes JSON.

    Tool calls come from ``result.new_messages()``: function tools, toolset and
    MCP tools, and the provider's built-in tools. A call whose return says
    ``failed``, ``denied`` or ``retried`` (``ToolFailed``, ``ModelRetry``,
    invalid arguments) is recorded as an error; a deferred call or one waiting
    for approval is recorded as not executed. The output tool of a structured
    reply is not reported. Tools that another agent runs inside a tool
    function (agent delegation) are not part of the parent's messages and stay
    invisible; only the delegating tool itself is recorded.

    >>> adapter = pydantic_ai_adapter(lambda: support_agent, deps=SupportDeps(plan="pro"))
    """
    return _PydanticAIAdapter(target, run_kwargs, history=history, content_fn=content_fn)
