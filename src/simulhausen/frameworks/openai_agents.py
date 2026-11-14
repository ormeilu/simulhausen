"""Adapter for agents built with the OpenAI Agents SDK (``pip install openai-agents``).

``openai_agents_adapter`` runs ``Runner.run(agent, input, ...)`` once per turn
and reads the reply from ``result.final_output``, the tool calls from
``result.new_items`` (handoff targets' tools included) and the token usage from
``result.context_wrapper.usage``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, NamedTuple

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    maybe_await,
    require,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Message, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["openai_agents_adapter"]

_ADAPTER = "openai_agents_adapter"
_PACKAGE = "openai-agents"

# Calls the SDK executes itself and answers with an output item; a call of one
# of these types without an output did not run (it waits for approval, or the
# run stopped before it). Every other call type is a hosted tool that the
# provider ran, and its call item already holds the outcome.
_LOCAL_CALL_TYPES = frozenset(
    {"function_call", "computer_call", "local_shell_call", "shell_call", "apply_patch_call", "custom_tool_call"}
)
_CALL_ITEMS = frozenset({"tool_call_item", "handoff_call_item"})
_OUTPUT_ITEMS = frozenset({"tool_call_output_item", "handoff_output_item"})
_APPROVAL_ITEMS = frozenset({"tool_approval_item", "mcp_approval_request_item"})
_FAILED_STATUSES = frozenset({"failed", "incomplete"})
_TRIPWIRES = frozenset({"InputGuardrailTripwireTriggered", "OutputGuardrailTripwireTriggered"})


class _Thread(NamedTuple):
    """What session mode keeps per thread: the run's input list and the agent that answered last.

    ``items`` is ``None`` when an SDK session holds the history instead.
    """

    items: list[Any] | None
    agent: Any


def _split_transcript(messages: Sequence[Message]) -> tuple[list[Message], str]:
    """The transcript as (earlier messages, prompt): the trailing user messages are the prompt.

    Several user messages in a row are joined with a blank line. The prompt is
    empty when the transcript does not end with a user message.
    """
    earlier = list(messages)
    tail: list[str] = []
    while earlier and earlier[-1]["role"] == "user":
        tail.append(earlier.pop()["content"])
    return earlier, "\n\n".join(reversed(tail))


def _items(messages: Sequence[Message]) -> list[Any]:
    """Transcript messages as Responses input items."""
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def _get(raw: Any, name: str) -> Any:
    return raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)


def _call_id(item: Any) -> Any:
    raw = item.raw_item
    return _get(raw, "call_id") or _get(raw, "id")


def _plain(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


def _name(item: Any) -> str | None:
    """Tool name of a call item; a hosted tool without one is named after its type (``web_search``)."""
    raw = item.raw_item
    name = getattr(item, "tool_name", None) or _get(raw, "name")
    if name:
        return str(name)
    kind = _get(raw, "type")
    return kind.removesuffix("_call") if isinstance(kind, str) else None


def _arguments(raw: Any) -> Any:
    """Arguments of a call: ``arguments``/``input`` of a function, MCP or custom call, else the hosted tool's query."""
    for key in ("arguments", "input"):
        value = _get(raw, key)
        if value is not None:
            return value
    for key in ("action", "queries", "code"):
        value = _get(raw, key)
        if value is not None:
            return {key: _plain(value)}
    return None


def _hosted_result(raw: Any) -> Any:
    for key in ("error", "output", "outputs", "results", "result"):
        value = _get(raw, key)
        if value is not None:
            return _plain(value)
    return None


def _tool_calls(items: Sequence[Any], interruptions: Sequence[Any], failed: set[str]) -> list[ToolExecution]:
    """Tool calls of one run in call order, built from its run items.

    A ``ToolCallItem`` or ``HandoffCallItem`` is paired with its output item
    by call id. ``failed`` holds the ids of the calls whose exception the SDK
    turned into a message for the model. Approval requests that have no call
    item of their own (hosted MCP) are added as calls that did not run.
    """
    outputs = {_call_id(item): item for item in items if item.type in _OUTPUT_ITEMS}
    executions: list[ToolExecution] = []
    seen: set[Any] = set()
    for item in items:
        if item.type not in _CALL_ITEMS:
            continue
        raw = item.raw_item
        call_id = _call_id(item)
        seen.add(call_id)
        output = outputs.get(call_id)
        if output is not None:
            value = getattr(output, "output", None)
            if value is None:
                value = _get(output.raw_item, "output")
            error = str(call_id) in failed
            executions.append(tool_execution(_name(item), _arguments(raw), value, error=error, call_id=call_id))
        elif item.type == "handoff_call_item" or _get(raw, "type") in _LOCAL_CALL_TYPES:
            executions.append(tool_execution(_name(item), _arguments(raw), executed=False, call_id=call_id))
        else:
            error = bool(_get(raw, "error")) or _get(raw, "status") in _FAILED_STATUSES
            result = _hosted_result(raw)
            executions.append(tool_execution(_name(item), _arguments(raw), result, error=error, call_id=call_id))
    for item in (*items, *interruptions):
        call_id = _call_id(item)
        if item.type in _APPROVAL_ITEMS and call_id not in seen:
            seen.add(call_id)
            executions.append(tool_execution(_name(item), _arguments(item.raw_item), executed=False, call_id=call_id))
    return executions


def _usage(source: Any) -> Usage | None:
    """Usage of the run from a result or from the run data of an exception."""
    return usage_of(getattr(getattr(source, "context_wrapper", None), "usage", None))


def _recording(run_config: Any, failed: set[str], config_type: Any) -> Any:
    """``run_config`` with a ``tool_error_formatter`` that notes which calls failed.

    The SDK turns a tool's exception into a message for the model and leaves
    no mark on the run items; the formatter hook is where it reports the
    failed call. The user's own formatter still decides the message, and
    without one the SDK's default message stays. An SDK release without the
    hook gets the config unchanged.
    """
    if "tool_error_formatter" not in getattr(config_type, "__dataclass_fields__", {}):
        return run_config
    is_dict = isinstance(run_config, dict)
    theirs = run_config.get("tool_error_formatter") if is_dict else getattr(run_config, "tool_error_formatter", None)

    async def record(args: Any) -> Any:
        failed.add(str(args.call_id))
        return await maybe_await(theirs(args)) if theirs is not None else None

    if run_config is None:
        return {"tool_error_formatter": record}
    if is_dict:
        return {**run_config, "tool_error_formatter": record}
    return dataclasses.replace(run_config, tool_error_formatter=record)


class _OpenAIAgentsAdapter(FrameworkAdapter):
    """Input list (or newest user message with an SDK session) -> ``Runner.run`` -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        run_kwargs: dict[str, Any],
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        session: Callable[[str], Any] | None,
        guardrail_reply: str | Callable[[Any], str] | None,
        resume_last_agent: bool,
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        if session is not None and history == "transcript":
            raise ValueError("session= keeps memory between turns; it cannot be combined with history='transcript'")
        self._run_kwargs = run_kwargs
        self._session_factory = session
        self._guardrail_reply = guardrail_reply
        self._resume_last_agent = resume_last_agent
        self._threads: Conversations[_Thread] = Conversations()
        self._sessions: dict[str, Any] = {}

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "handoffs") or hasattr(target, "tools")

    async def _session_input(self, input: AgentInput, known: _Thread | None) -> tuple[Any, Any]:
        """The run input and the SDK session of the thread, brought in line with the transcript."""
        factory = self._session_factory
        assert factory is not None
        thread_id = input.thread_id
        fresh = thread_id not in self._sessions
        if fresh:
            self._sessions[thread_id] = factory(thread_id)
        session = self._sessions[thread_id]
        if known is not None:
            return "\n\n".join(m["content"] for m in input.new_messages if m["role"] == "user"), session
        # The session no longer matches the dialogue (or never saw it): replace
        # what it holds with the transcript.
        earlier, prompt = _split_transcript(input.messages)
        if not fresh:
            await session.clear_session()
        if earlier:
            await session.add_items(_items(earlier))
        return prompt, session

    async def call(self, input: AgentInput) -> AgentTurn:
        starting = self._resolve()
        agents = require("agents", package=_PACKAGE, adapter=_ADAPTER)
        known = self._threads.recall(input) if self._history == "session" else None
        agent = known.agent if known is not None and self._resume_last_agent else starting
        kwargs = dict(self._run_kwargs)
        if self._session_factory is not None:
            run_input, kwargs["session"] = await self._session_input(input, known)
        elif known is not None and known.items is not None:
            run_input = [*known.items, *_items(input.new_messages)]
        else:
            run_input = _items(input.messages)
        failed: set[str] = set()
        kwargs["run_config"] = _recording(kwargs.get("run_config"), failed, getattr(agents, "RunConfig", None))
        try:
            result = await agents.Runner.run(agent, run_input, **kwargs)
        except Exception as exc:
            reply = self._guardrail_reply
            if reply is None or type(exc).__name__ not in _TRIPWIRES:
                raise
            # The run has no result to continue from: the next turn starts
            # from the transcript, which holds this reply.
            self._threads.forget(input.thread_id)
            data = getattr(exc, "run_data", None)
            tools = _tool_calls(getattr(data, "new_items", None) or [], [], failed)
            return AgentTurn(
                content=reply if isinstance(reply, str) else reply(exc), tools=tools or None, usage=_usage(data)
            )
        content = self._render(result.final_output)
        interruptions = list(getattr(result, "interruptions", None) or [])
        if self._history == "session":
            if interruptions:
                # An interrupted run ends with calls that have no output; it is
                # not a history another run can start from.
                self._threads.forget(input.thread_id)
            else:
                items = None if self._session_factory is not None else list(result.to_input_list())
                self._threads.remember(input, content, _Thread(items, result.last_agent))
        tools = _tool_calls(result.new_items, interruptions, failed)
        return AgentTurn(content=content, tools=tools or None, usage=_usage(result))


def openai_agents_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    session: Callable[[str], Any] | None = None,
    guardrail_reply: str | Callable[[Any], str] | None = None,
    resume_last_agent: bool = True,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for an OpenAI Agents SDK ``Agent``, with its handoffs.

    ``target`` is the starting agent or a zero-argument factory returning it
    (so collecting tests builds no agent). Extra kwargs go to ``Runner.run``
    unchanged: ``context=...``, ``max_turns=...``, ``run_config=...``,
    ``hooks=...`` and so on. Exceptions of the run propagate unchanged.

    ``history="session"`` (default) keeps ``result.to_input_list()`` per
    ``thread_id`` and sends it back with the new user message, so the agent
    sees its earlier tool calls and their results. When the transcript no
    longer continues what the agent saw (a rollback, a reply injected with
    ``sim.agent("text")``), the input is rebuilt from the transcript as plain
    user and assistant messages. ``history="transcript"`` sends the transcript
    that way on every call and remembers nothing, so a rollback reaches the
    agent.

    ``session`` switches session mode to the SDK's own session memory: a
    factory ``thread_id -> Session`` (``SQLiteSession`` itself fits), called
    once per thread. Each turn then sends only the new user message with
    ``session=...``. After a rollback or an injected reply the adapter clears
    the session and fills it from the transcript.

    ``resume_last_agent`` (default ``True``) starts the next turn of a thread
    from ``result.last_agent``, the agent a handoff left in charge, as the
    SDK's own chat loop does. With ``False``, and always in transcript mode or
    after a rebuild, every turn starts from ``target``.

    ``guardrail_reply`` decides what a tripped input or output guardrail means.
    The SDK raises ``InputGuardrailTripwireTriggered`` or
    ``OutputGuardrailTripwireTriggered`` and has no refusal text of its own, so
    by default the exception propagates like any other. Give the text your
    application shows in that case, or a function ``exception -> text``, to
    record it as the agent's reply instead; the tool calls and usage of the
    aborted run are kept.

    ``content_fn`` renders ``result.final_output`` into the reply text. Without
    it a string is used as it is and a structured output becomes JSON.

    Tool calls come from ``result.new_items`` of the whole run, so the tools
    of every agent a handoff reached are included, and the handoff itself is
    recorded as a call of its tool (``transfer_to_<agent>``). Function, MCP,
    shell and computer calls are paired with their output items; hosted tools
    (web search, file search, code interpreter, hosted MCP) are read from
    their call items. A function tool whose exception the SDK reported to the
    model is recorded as an error; the adapter learns about it through
    ``RunConfig.tool_error_formatter``, which the SDK calls only for tools
    with the default ``failure_error_function``, so a tool with its own error
    function looks like a clean call. A call waiting for approval
    (``result.interruptions``) is recorded as not executed, the reply of such
    a turn is empty, and the approval is not resumed. Tools that an agent used
    through ``Agent.as_tool()`` runs are not part of the parent's run items:
    only the call of the agent tool itself is recorded.

    >>> adapter = openai_agents_adapter(lambda: triage_agent, context=Customer(plan="pro"))
    """
    return _OpenAIAgentsAdapter(
        target,
        run_kwargs,
        history=history,
        content_fn=content_fn,
        session=session,
        guardrail_reply=guardrail_reply,
        resume_last_agent=resume_last_agent,
    )
