"""Shared pieces of the in-process framework adapters.

Every adapter in this package drives an agent built with another framework
inside the test process and turns what that framework reports (reply, tool
calls, token usage) into an :class:`~simulhausen.types.AgentTurn`. None of the
frameworks is a dependency: a module imports its framework only when an adapter
is called, through :func:`require`, so ``import simulhausen`` stays as light as
before and a missing framework is reported by name.
"""

from __future__ import annotations

import importlib
import inspect
import json
from typing import TYPE_CHECKING, Any, Literal

from agno.models.response import ToolExecution

from simulhausen.http_adapters import _blocks_text, _render_transcript, _to_text, _tool_args
from simulhausen.types import AgentAdapter, AgentInput, Message, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "Conversations",
    "FrameworkAdapter",
    "FrameworkHistory",
    "maybe_await",
    "require",
    "result_text",
    "text_of",
    "tool_execution",
    "transcript_prompt",
    "usage_of",
]

# How the agent under test gets the dialogue so far:
# - "session": the framework's own memory (a session, a thread, a checkpointer,
#   the message history the adapter carries between turns) plus the newest user
#   message, so tool calls of earlier turns stay in the agent's context;
# - "transcript": only what the transcript says, rebuilt on every call, so a
#   rollback really erases the removed exchange (red-team backtracking).
type FrameworkHistory = Literal["session", "transcript"]


def require(module: str, *, package: str, adapter: str) -> Any:
    """Import a framework module on first use; a missing one names the package to install.

    Returns the module as ``Any``: the frameworks are not dependencies, so the
    type checker never sees them.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"simulhausen.{adapter} needs the {package!r} package, which is not installed. "
            f"Install it next to simulhausen (pip install {package})."
        ) from exc


async def maybe_await(value: Any) -> Any:
    """The value itself, or its result when it is awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


def text_of(value: Any) -> str:
    """Reply content as text.

    Strings as they are, a list of content blocks as its text blocks, a
    Pydantic model as JSON, any other structured value as JSON.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _blocks_text(value)
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    return _to_text(value)


def result_text(value: Any) -> str | None:
    """A tool result as text (``ToolExecution.result`` is a string); ``None`` stays ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _blocks_text(value) or _to_text(value)
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def tool_execution(
    name: str | None,
    args: Any = None,
    result: Any = None,
    *,
    error: bool = False,
    executed: bool = True,
    call_id: Any = None,
) -> ToolExecution:
    """One tool call of the agent under test in agno's ``ToolExecution`` shape.

    ``args`` may be a dict or a JSON string. ``error=True`` marks a call that
    raised or was rejected. ``executed=False`` marks a call the model asked
    for that never ran (the run was interrupted, the tool waits for approval):
    it is recorded as paused, so ``state.has_tool_call`` does not count it.
    """
    return ToolExecution(
        tool_call_id=str(call_id) if call_id is not None else None,
        tool_name=name,
        tool_args=_tool_args(args),
        result=result_text(result),
        tool_call_error=True if error else None,
        requires_confirmation=None if executed else True,
    )


def usage_of(source: Any, *, requests: int | None = None) -> Usage | None:
    """Token usage from a framework's usage object or dict; ``None`` when it holds none.

    The same as :meth:`simulhausen.types.Usage.of`: it reads ``input_tokens`` /
    ``prompt_tokens`` / ``prompt_token_count`` / ``inputTokens``, the matching
    output and total names, ``requests`` and ``cost``. ``requests`` overrides
    the model call count when the caller counted it itself.
    """
    return Usage.of(source, requests=requests)


def transcript_prompt(messages: Sequence[Message]) -> str:
    """The transcript as one string, for frameworks whose input is a single task text."""
    return _render_transcript(messages)


def _key(messages: Sequence[Message]) -> tuple[tuple[str, str], ...]:
    return tuple((message["role"], message["content"]) for message in messages)


class Conversations[T]:
    """Per-thread native state (a message history, a session object) that follows the transcript.

    ``remember`` stores the state together with the transcript as it stands
    after the agent's reply; ``recall`` gives it back only while the transcript
    still begins that way. After a rollback, or a reply the script injected
    with ``sim.agent("text")``, the stored state no longer matches the
    dialogue: ``recall`` returns ``None`` and the adapter rebuilds the agent's
    context from the transcript.
    """

    def __init__(self) -> None:
        self._states: dict[str, tuple[tuple[tuple[str, str], ...], T]] = {}

    def recall(self, input: AgentInput) -> T | None:
        stored = self._states.get(input.thread_id)
        if stored is None:
            return None
        seen, state = stored
        new = len(input.new_messages)
        before = input.messages[: len(input.messages) - new] if new else input.messages
        return state if new and _key(before) == seen else None

    def remember(self, input: AgentInput, reply: str, state: T) -> None:
        self._states[input.thread_id] = ((*_key(input.messages), ("assistant", reply)), state)

    def forget(self, thread_id: str) -> None:
        self._states.pop(thread_id, None)


class FrameworkAdapter(AgentAdapter):
    """Base of the framework adapters: lazy target, history mode, reply rendering.

    ``target`` is the framework object or a zero-argument factory returning it
    (so collecting tests builds no agent). Agents of some frameworks are
    callable themselves, so a subclass says what its targets look like in
    :meth:`_is_target`; anything else that is callable is taken for a factory,
    called once and cached.
    """

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory = "session",
        content_fn: Callable[[Any], str] | None = None,
    ) -> None:
        if history not in ("session", "transcript"):
            raise ValueError(f"history must be 'session' or 'transcript', got {history!r}")
        self._target = target
        self._resolved: Any = None
        self._history: FrameworkHistory = history
        self._content_fn = content_fn

    def _is_target(self, target: Any) -> bool:
        """Whether ``target`` is the framework object itself rather than a factory."""
        return not callable(target)

    def _resolve(self) -> Any:
        if self._resolved is None:
            target = self._target
            self._resolved = target if self._is_target(target) or not callable(target) else target()
        return self._resolved

    def _render(self, raw: Any) -> str:
        """The reply as text: through ``content_fn`` when given, else :func:`text_of`."""
        return self._content_fn(raw) if self._content_fn is not None else text_of(raw)
