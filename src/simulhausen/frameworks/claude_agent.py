"""Adapter for agents built with the Claude Agent SDK (``pip install claude-agent-sdk``).

``claude_agent_adapter`` runs ``query(prompt=..., options=...)`` once per turn
and reads the reply, the cost and the token usage from the ``ResultMessage``
and the tool calls from the tool-use and tool-result blocks of the messages the
SDK streams, subagents' included.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from simulhausen.exceptions import SimulationBackendError
from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    require,
    tool_execution,
    transcript_prompt,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["claude_agent_adapter"]

_ADAPTER = "claude_agent_adapter"
_PACKAGE = "claude-agent-sdk"

_INPUT_TOKEN_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _kind(message: Any) -> str:
    """The SDK's message class by name: the adapter never imports the SDK's types."""
    return type(message).__name__


def _blocks(message: Any) -> list[Any]:
    content = getattr(message, "content", None)
    return content if isinstance(content, list) else []


def _tool_calls(messages: Sequence[Any]) -> list[ToolExecution]:
    """Tool calls of one run in the order the model asked for them.

    A ``ToolUseBlock`` is paired with its ``ToolResultBlock`` by id; a result
    with ``is_error`` (the tool raised, the permission was denied) is an error,
    and a tool use that got no result (deferred by a hook, the run stopped)
    did not run. A ``ServerToolUseBlock`` was executed by the API, so it counts
    as run even when the SDK passes no result block for it.
    """
    results: dict[Any, Any] = {}
    for message in messages:
        for block in _blocks(message):
            if hasattr(block, "tool_use_id"):
                results[block.tool_use_id] = block
    executions: list[ToolExecution] = []
    for message in messages:
        for block in _blocks(message):
            if not (hasattr(block, "input") and hasattr(block, "name")):
                continue
            result = results.get(block.id)
            if result is not None:
                error = bool(getattr(result, "is_error", False))
                executions.append(
                    tool_execution(block.name, block.input, result.content, error=error, call_id=block.id)
                )
            else:
                hosted = _kind(block) == "ServerToolUseBlock"
                executions.append(tool_execution(block.name, block.input, executed=hosted, call_id=block.id))
    return executions


def _usage(result: Any, messages: Sequence[Any]) -> Usage | None:
    """Usage of the run from its ``ResultMessage``.

    Input tokens include the tokens read from and written to the prompt cache,
    which the API reports apart from ``input_tokens``. The model requests are
    counted from the assistant messages: one API response may arrive as
    several messages with the same ``message_id``.
    """
    reported = getattr(result, "usage", None) or {}
    tokens = {
        "input_tokens": sum(int(reported.get(key) or 0) for key in _INPUT_TOKEN_KEYS),
        "output_tokens": int(reported.get("output_tokens") or 0),
        "cost": getattr(result, "total_cost_usd", None),
    }
    responses = [message for message in messages if _kind(message) == "AssistantMessage"]
    requests = len({getattr(message, "message_id", None) or id(message) for message in responses})
    return usage_of(tokens, requests=requests)


def _last_text(messages: Sequence[Any]) -> str:
    """Text of the last assistant message of the main agent (subagents' messages have a parent tool use)."""
    for message in reversed(messages):
        if _kind(message) == "AssistantMessage" and getattr(message, "parent_tool_use_id", None) is None:
            return "".join(block.text for block in _blocks(message) if hasattr(block, "text"))
    return ""


def _session_id(result: Any, messages: Sequence[Any]) -> str | None:
    """The SDK session of the run: from the result, else from the ``init`` system message."""
    session_id = getattr(result, "session_id", None)
    if session_id:
        return str(session_id)
    for message in messages:
        data = getattr(message, "data", None)
        if _kind(message) == "SystemMessage" and isinstance(data, dict) and data.get("session_id"):
            return str(data["session_id"])
    return None


class _ClaudeAgentAdapter(FrameworkAdapter):
    """Newest user message in the resumed SDK session (or the transcript in a new one) -> ``query`` -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        transport: Callable[[Any], Any] | None,
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._transport = transport
        # thread_id -> the SDK session id of the thread's latest run.
        self._sessions: Conversations[str] = Conversations()

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "resume") or hasattr(target, "system_prompt")

    async def call(self, input: AgentInput) -> AgentTurn:
        options = self._resolve()
        sdk = require("claude_agent_sdk", package=_PACKAGE, adapter=_ADAPTER)
        session_id = self._sessions.recall(input) if self._history == "session" else None
        if session_id is not None:
            options = dataclasses.replace(options, resume=session_id)
            prompt = "\n\n".join(m["content"] for m in input.new_messages if m["role"] == "user")
        else:
            prompt = transcript_prompt(input.messages)
        kwargs: dict[str, Any] = {}
        if self._transport is not None:
            kwargs["transport"] = self._transport(options)
        messages = [message async for message in sdk.query(prompt=prompt, options=options, **kwargs)]
        result = next((message for message in reversed(messages) if _kind(message) == "ResultMessage"), None)
        if result is None:
            raise SimulationBackendError("the Claude Agent SDK run ended without a result message")
        if getattr(result, "is_error", False):
            # With the bundled CLI the SDK raises ResultError itself; this covers a transport that does not.
            detail = "; ".join(getattr(result, "errors", None) or []) or getattr(result, "result", None) or "no details"
            raise SimulationBackendError(f"the Claude Agent SDK run failed ({result.subtype}): {detail}")
        raw = getattr(result, "structured_output", None)
        if raw is None:
            raw = getattr(result, "result", None)
        if raw is None:
            raw = _last_text(messages)
        content = self._render(raw)
        if self._history == "session":
            new_session = _session_id(result, messages)
            if new_session is None:
                self._sessions.forget(input.thread_id)
            else:
                self._sessions.remember(input, content, new_session)
        return AgentTurn(content=content, tools=_tool_calls(messages) or None, usage=_usage(result, messages))


def claude_agent_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    transport: Callable[[Any], Any] | None = None,
) -> AgentAdapter:
    """Adapter for an agent defined by the Claude Agent SDK's ``ClaudeAgentOptions``.

    ``target`` is the options object or a zero-argument factory returning it
    (so collecting tests builds nothing). The options are the agent: system
    prompt, tools and MCP servers, subagents, hooks, permission mode, model,
    working directory. Each turn runs one ``query(prompt=..., options=...)``,
    which starts the ``claude`` CLI, and reads the stream to its end. ``query``
    takes no other settings, so the adapter forwards no extra keyword
    arguments. Exceptions of the SDK (``CLINotFoundError``, ``ProcessError``,
    ``ResultError`` for a run that ended with an error result) propagate
    unchanged; an error result that the SDK did not raise for becomes
    ``SimulationBackendError``.

    ``history="session"`` (default) resumes the SDK session of the thread's
    previous turn: the adapter keeps the ``session_id`` of each run per
    ``thread_id`` and sends the new user message with ``resume=<session_id>``
    set on a copy of the options, so the agent keeps its real context, tool
    calls and file edits included. When the transcript no longer continues
    what the agent saw (a rollback, a reply injected with
    ``sim.agent("text")``), the next turn starts a new session and sends the
    transcript as one prompt. ``history="transcript"`` does that on every
    call, so a rollback reaches the agent. Leave ``resume``,
    ``continue_conversation`` and ``session_id`` unset in the options:
    scenarios would share one session through them.

    ``content_fn`` renders the reply. It gets ``ResultMessage.structured_output``
    when the options ask for an output format, else ``ResultMessage.result``
    (the final text), else the text of the main agent's last message. Without
    it text is used as it is and a structured output becomes JSON.

    ``transport`` is a factory ``options -> Transport`` for the SDK's custom
    transport hook (a remote Claude Code, a scripted one in tests). It is
    called on every turn with that turn's options, because a transport serves
    one ``query``. The SDK does not apply the options to a custom transport,
    ``resume`` included, so the factory has to.

    Tool calls are the ``ToolUseBlock``s of the stream paired with their
    ``ToolResultBlock``s: built-in tools, MCP tools (``mcp__server__tool``)
    and the calls subagents make, in stream order. A result with ``is_error``
    (the tool failed, the permission was denied) is recorded as an error, a
    tool use without a result as not executed. Usage and cost come from the
    ``ResultMessage`` (``total_cost_usd`` as ``Usage.cost``; input tokens
    include prompt-cache reads and writes).

    >>> adapter = claude_agent_adapter(lambda: ClaudeAgentOptions(system_prompt="You are a support agent."))
    """
    return _ClaudeAgentAdapter(target, history=history, content_fn=content_fn, transport=transport)
