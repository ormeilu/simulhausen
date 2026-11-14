"""Adapter for Microsoft Agent Framework (``agent-framework`` / ``agent-framework-core``).

``agent_framework_adapter`` runs an Agent Framework agent, or a workflow built
with ``WorkflowBuilder`` or one of the orchestration builders, inside the test
process: ``agent.run(messages, session=...)`` per turn, the tool calls and the
token usage read from the response.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    require,
    text_of,
    tool_execution,
    usage_of,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Message, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["agent_framework_adapter"]

_ADAPTER = "agent_framework_adapter"
_PACKAGE = "agent-framework-core"


def _native(messages: Sequence[Message]) -> list[Any]:
    """Transcript messages as Agent Framework ``Message`` objects."""
    message_cls = require("agent_framework", package=_PACKAGE, adapter=_ADAPTER).Message
    return [message_cls(message["role"], [message["content"]]) for message in messages]


def _tools(messages: Sequence[Any]) -> list[ToolExecution]:
    """``function_call`` contents paired with ``function_result`` contents by call id, in call order.

    A result that carries an ``exception`` is an error. A call without a
    result did not run (it waits for approval, or the run stopped), unless the
    service executes it itself (``hosted``) or it is there for the record only.
    """
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        for content in getattr(message, "contents", None) or []:
            kind = getattr(content, "type", None)
            if kind == "function_approval_request":
                content, kind = content.function_call, "function_call"
            call_id = str(getattr(content, "call_id", None))
            if kind == "function_call":
                calls.setdefault(call_id, {"call": content, "result": None})
            elif kind == "function_result":
                calls.setdefault(call_id, {"call": None, "result": None})["result"] = content
    tools: list[ToolExecution] = []
    for call_id, entry in calls.items():
        call, result = entry["call"], entry["result"]
        remote = bool(getattr(call, "hosted", None) or getattr(call, "informational_only", None))
        tools.append(
            tool_execution(
                getattr(call, "name", None),
                getattr(call, "arguments", None),
                getattr(result, "result", None),
                error=getattr(result, "exception", None) is not None,
                executed=result is not None or remote,
                call_id=call_id,
            )
        )
    return tools


def _usage(details: Any) -> Usage | None:
    """``usage_details`` (``input_token_count`` and friends) as ``Usage``."""
    if not isinstance(details, dict):
        return None
    return usage_of(
        {
            "input_tokens": details.get("input_token_count"),
            "output_tokens": details.get("output_token_count"),
            "total_tokens": details.get("total_token_count"),
        }
    )


def _responses(events: Sequence[Any]) -> list[Any]:
    """The agent response of every executor run in a workflow, in execution order.

    The same response travels in several events as separate copies (a workflow
    output, a request for the next user turn, the executor's completion), and
    twice in the completion itself (the message sent on and the output). The
    completion events alone, one response each, count every model call once.
    """
    found: list[Any] = []
    for event in events:
        if getattr(event, "type", None) != "executor_completed":
            continue
        data = getattr(event, "data", None)
        for item in reversed(data if isinstance(data, list) else [data]):
            response = getattr(item, "agent_response", item)
            if hasattr(response, "messages") and hasattr(response, "usage_details"):
                found.append(response)
                break
    return found


def _output_text(output: Any) -> str:
    """A workflow output as text: an agent response, a list of messages (the last one), or any value."""
    if isinstance(output, list) and output:
        output = output[-1]
    text = getattr(output, "text", None)
    return text if isinstance(text, str) else text_of(output)


class _AgentFrameworkAdapter(FrameworkAdapter):
    """New messages -> ``agent.run`` in the thread's session (or ``workflow.run``) -> ``AgentTurn``."""

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
        self._targets: dict[str | None, Any] = {}
        self._threads: set[str] = set()
        # Per thread: the agent session, or the ids of the requests a workflow is waiting on.
        self._states: Conversations[Any] = Conversations()

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "run") or not callable(target)

    def _target_for(self, thread_id: str, *, rebuild: bool = False) -> Any:
        """The agent or workflow of the thread: a factory builds one per thread, an object serves them all."""
        factory = not self._is_target(self._target)
        key = thread_id if factory else None
        if key not in self._targets or (rebuild and factory):
            self._targets[key] = self._target() if factory else self._target
        return self._targets[key]

    async def call(self, input: AgentInput) -> AgentTurn:
        target = self._target_for(input.thread_id)
        if hasattr(target, "create_session"):
            return await self._call_agent(target, input)
        return await self._call_workflow(input)

    async def _call_agent(self, agent: Any, input: AgentInput) -> AgentTurn:
        session = self._states.recall(input) if self._history == "session" else None
        if session is not None:
            messages = input.new_messages
        else:
            # A new thread, transcript mode, or a transcript that no longer matches the session.
            messages = input.messages
            first = self._history == "session" and input.thread_id not in self._threads
            self._threads.add(input.thread_id)
            session_id = input.thread_id if first else f"{input.thread_id}-{uuid.uuid4().hex[:12]}"
            session = agent.create_session(session_id=session_id)
        response = await agent.run(_native(messages), session=session, **self._run_kwargs)
        content = self._content_fn(response) if self._content_fn is not None else text_of(response.text)
        if self._history == "session":
            self._states.remember(input, content, session)
        return AgentTurn(
            content=content,
            tools=_tools(response.messages) or None,
            usage=_usage(getattr(response, "usage_details", None)),
        )

    async def _call_workflow(self, input: AgentInput) -> AgentTurn:
        factory = not self._is_target(self._target)
        pending: list[str] | None = None
        if self._history == "transcript":
            if not factory:
                raise ValueError(
                    "agent_framework_adapter(history='transcript') needs a factory for a workflow: its executors "
                    "keep their conversation between runs, so every call has to start from a new workflow."
                )
            workflow, messages = self._target(), input.messages
        else:
            pending = self._states.recall(input)
            diverged = pending is None and input.thread_id in self._threads
            self._threads.add(input.thread_id)
            workflow = self._target_for(input.thread_id, rebuild=diverged)
            messages = input.messages if pending is None else input.new_messages
        if pending:
            # The workflow asked for the next user turn (handoff, group chat): the new messages answer it.
            answer = _native(messages)
            result = await workflow.run(responses=dict.fromkeys(pending, answer), **self._run_kwargs)
        else:
            alone = len(messages) == 1 and messages[0]["role"] == "user"
            task = messages[0]["content"] if alone else _native(messages)
            result = await workflow.run(task, **self._run_kwargs)
        events = list(result)
        outputs = result.get_outputs()
        responses = _responses(events)
        if self._content_fn is not None:
            content = self._content_fn(result)
        else:
            content = _output_text(outputs[-1]) if outputs else ""
        usage: Usage | None = None
        for response in responses:
            one = _usage(response.usage_details)
            if one is not None:
                usage = one if usage is None else usage + one
        if self._history == "session":
            requests = result.get_request_info_events()
            waiting = [event.request_id for event in requests if getattr(event, "response_type", None) is list]
            self._states.remember(input, content, waiting)
        tools = _tools([message for response in responses for message in response.messages])
        return AgentTurn(content=content, tools=tools or None, usage=usage)


def agent_framework_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a Microsoft Agent Framework agent or workflow.

    ``target`` is an agent (``Agent``, any ``BaseAgent``, ``workflow.as_agent()``),
    a ``Workflow`` (from ``WorkflowBuilder``, ``SequentialBuilder``,
    ``HandoffBuilder`` and the other orchestration builders), or a
    zero-argument factory, which is called once per ``thread_id``. Extra kwargs
    go to ``run`` unchanged: ``options=...``, ``tools=...``,
    ``function_invocation_kwargs=...``.

    For an agent, ``history="session"`` keeps one ``AgentSession``
    (``agent.create_session(session_id=thread_id)``) per scenario and sends
    only the new messages, so the agent's history provider replays the earlier
    turns, tool calls included. When the transcript stops matching the session
    (a rollback, a reply scripted with ``sim.agent("text")``), the adapter
    opens a new session and sends the whole transcript.
    ``history="transcript"`` does that on every call. Tool calls are the
    ``function_call`` contents of ``response.messages`` paired with the
    ``function_result`` contents by call id; a result with an ``exception``
    (the framework catches a raising tool) is an error, a call without a
    result (a ``function_approval_request``) is recorded as not executed.
    Usage is ``response.usage_details``; the framework does not report the
    number of model requests.

    A ``Workflow`` is run directly, and every participant's response is read
    from the run's events: the reply is the last workflow output, tool calls
    and usage cover all participants. A workflow keeps its conversation inside
    its executors, so one workflow object serves one scenario; pass a factory
    to give each scenario its own. In session mode the adapter sends the new
    message, or answers the request for the next user turn that a handoff or
    group chat workflow left pending, and after a rollback rebuilds the
    workflow from the factory and sends the transcript. Transcript mode builds
    a new workflow for every call and therefore needs a factory. An agent made
    with ``workflow.as_agent()`` works as an agent, but its response carries
    only the workflow's final output: the participants' tool calls and usage
    are not visible that way.

    The reply is ``response.text``; ``content_fn`` gets the ``AgentResponse``
    (the ``WorkflowRunResult`` for a workflow) instead.

    >>> adapter = agent_framework_adapter(lambda: Agent(client=client, tools=[lookup_order]))
    """
    return _AgentFrameworkAdapter(target, history=history, content_fn=content_fn, run_kwargs=run_kwargs)
