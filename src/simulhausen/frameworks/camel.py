"""Adapter for CAMEL-AI ``ChatAgent`` (``pip install camel-ai``).

``camel_adapter`` awaits ``agent.astep(message)`` once per turn and reads the
reply from ``response.msgs``, the tool calls from ``response.info["tool_calls"]``
and the token usage from ``response.info["usage"]``. The agent owns its memory;
the adapter resets it and writes the transcript back into it whenever the
dialogue has to be rebuilt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    maybe_await,
    require,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Message

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["camel_adapter"]

_ADAPTER = "camel_adapter"
_PACKAGE = "camel-ai"

# ChatAgent catches a tool's exception (and an unknown tool name) and hands the
# model this text as the result; a ToolCallingRecord has no error flag besides it.
_FAILED = "Tool execution failed:"


def _replay(agent: Any, messages: Sequence[Message]) -> None:
    """Write transcript messages into the agent's memory, as the user's and the agent's own."""
    if not messages:
        return
    base_message = require("camel.messages", package=_PACKAGE, adapter=_ADAPTER).BaseMessage
    roles = require("camel.types", package=_PACKAGE, adapter=_ADAPTER).OpenAIBackendRole
    name = getattr(agent, "role_name", None) or "Assistant"
    for message in messages:
        if message["role"] == "user":
            agent.update_memory(
                base_message.make_user_message(role_name="User", content=message["content"]), roles.USER
            )
        else:
            agent.update_memory(
                base_message.make_assistant_message(role_name=name, content=message["content"]), roles.ASSISTANT
            )


def _tools(info: Any) -> list[ToolExecution]:
    """Tool executions of one step: the calls the agent ran, then the ones it left to an external executor."""
    info = info if isinstance(info, dict) else {}
    executions: list[ToolExecution] = []
    for record in info.get("tool_calls") or []:
        result = record.result
        failed = isinstance(result, str) and result.startswith(_FAILED)
        executions.append(
            tool_execution(record.tool_name, record.args, result, error=failed, call_id=record.tool_call_id)
        )
    executions.extend(
        tool_execution(request.tool_name, request.args, executed=False, call_id=request.tool_call_id)
        for request in info.get("external_tool_call_requests") or []
    )
    return executions


class _CamelAdapter(FrameworkAdapter):
    """Newest user message, after the transcript when the memory is rebuilt -> ``agent.astep`` -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        step_kwargs: dict[str, Any],
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._step_kwargs = step_kwargs
        self._shared = self._is_target(target) or not callable(target)
        self._agents: dict[str, Any] = {}
        # Threads whose dialogue their agent's memory still holds.
        self._live: Conversations[bool] = Conversations()
        # The thread a shared agent instance holds now: it has one memory.
        self._owner: str | None = None

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "astep")

    def _agent(self, thread_id: str) -> Any:
        if self._shared:
            return self._target
        if thread_id not in self._agents:
            self._agents[thread_id] = self._target()
        return self._agents[thread_id]

    async def call(self, input: AgentInput) -> AgentTurn:
        agent = self._agent(input.thread_id)
        continues = (
            self._history == "session"
            and self._live.recall(input) is not None
            and (not self._shared or self._owner == input.thread_id)
        )
        if continues:
            pending = list(input.new_messages)
        else:
            agent.reset()
            pending = list(input.messages)
        # astep takes the newest user message; whatever precedes it goes into the memory.
        prompt = pending.pop()["content"] if pending and pending[-1]["role"] == "user" else ""
        _replay(agent, pending)
        # With a streaming model astep returns a response that is awaited for the final result.
        response = await maybe_await(await agent.astep(prompt, **self._step_kwargs))

        messages = getattr(response, "msgs", None) or []
        message = messages[0] if messages else None
        parsed = getattr(message, "parsed", None)
        content = self._render(parsed if parsed is not None else getattr(message, "content", None))
        info = getattr(response, "info", None)
        self._owner = input.thread_id
        self._live.remember(input, content, True)
        return AgentTurn(
            content=content,
            tools=_tools(info) or None,
            usage=usage_of(info.get("usage") if isinstance(info, dict) else None),
        )


def camel_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    **step_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a CAMEL-AI ``ChatAgent``.

    ``target`` is the agent or a zero-argument factory returning one. A
    ``ChatAgent`` owns its memory, so pass a factory when the adapter serves
    more than one scenario: every ``thread_id`` then gets its own agent. A
    single agent instance holds one dialogue at a time; a turn of another
    scenario resets it and writes that scenario's transcript into its memory,
    so scenarios that run concurrently need a factory.

    A turn is ``await agent.astep(message)``. Extra kwargs go to ``astep``
    unchanged (``response_format=...``). A streaming model's response is
    awaited for its final result.

    ``history="session"`` (default) sends the newest user message and leaves
    the rest to the agent's memory, tool calls and tool results of earlier
    turns included. When the transcript no longer continues what the agent
    answered (a rollback, a reply injected with ``sim.agent("...")``) the
    adapter rebuilds the memory as transcript mode does.
    ``history="transcript"`` rebuilds it on every call: ``agent.reset()``, then
    the transcript written into the memory with ``agent.update_memory`` as user
    and assistant messages, then ``astep`` with the newest user message. A
    rollback reaches the agent that way; earlier tool calls are not replayed.
    ``reset()`` keeps the system message and clears everything else the agent
    remembers.

    Tool calls come from ``response.info["tool_calls"]``
    (``ToolCallingRecord``: name, arguments, result, id), in call order.
    ``ChatAgent`` does not let a tool's exception out: the model gets the text
    ``"Tool execution failed: ..."`` as the result, and a record whose result
    starts with it is reported as an error. Calls of external tools, which the
    agent hands back in ``info["external_tool_call_requests"]`` without running
    them, are recorded as not executed.

    ``AgentTurn.usage`` holds the step's ``info["usage"]`` (prompt, completion
    and total tokens over every model call of the step); CAMEL does not report
    the number of model calls.

    ``content_fn`` renders the reply: it gets the first message's ``parsed``
    object when ``response_format`` produced one, else its ``content``.
    Without it a string stays as it is and a Pydantic model becomes JSON.

    ``Workforce`` and ``RolePlaying`` are not supported. ``RolePlaying.astep``
    advances a dialogue between two of its own agents, and
    ``Workforce.process_task_async`` returns a ``Task`` with a result text but
    no tool calls or usage; wrap those in ``function_adapter`` instead.

    >>> adapter = camel_adapter(lambda: ChatAgent("You are the order desk.", model=model, tools=[lookup_order]))
    """
    return _CamelAdapter(target, step_kwargs, history=history, content_fn=content_fn)
