"""Adapter for the LangChain family: LangGraph graphs, Deep Agents, chat models and chains.

``langchain_adapter`` drives anything that is a LangChain ``Runnable`` inside
the test process:

- a compiled LangGraph graph whose state holds the dialogue under ``messages``:
  ``langchain.agents.create_agent``, ``langgraph.prebuilt.create_react_agent``,
  a hand-built ``StateGraph``, ``deepagents.create_deep_agent``, the
  supervisor and swarm graphs with their sub-agents;
- a legacy ``AgentExecutor`` (``langchain-classic``);
- a chat model or a chain whose ``ainvoke`` returns an ``AIMessage`` or a string.

Neither LangChain nor LangGraph is imported at module import time. The adapter
talks to the target through ``astream`` / ``ainvoke`` and reads the returned
messages by their attributes; only resuming an interrupted graph needs a class
from ``langgraph`` and imports it at that moment.

The HTTP adapter for a LangGraph server is a different thing and keeps its
name: ``simulhausen.langgraph_adapter``.
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
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agno.models.response import ToolExecution

    from simulhausen.types import Message

__all__ = ["langchain_adapter"]

# The key under which LangGraph reports pending interrupts in the state it streams.
_INTERRUPT = "__interrupt__"

_AI_TYPES = frozenset({"ai", "assistant", "AIMessageChunk"})


def _field(source: Any, name: str) -> Any:
    """An attribute of a message object, or the same key of a message given as a dict."""
    return source.get(name) if isinstance(source, dict) else getattr(source, name, None)


def _kind(message: Any) -> str | None:
    """``"ai"``, ``"tool"`` or another message type; ``None`` for a value that is not a message."""
    if isinstance(message, dict):
        kind = message.get("type") or message.get("role") if "content" in message else None
    else:
        kind = getattr(message, "type", None) if hasattr(message, "content") else None
    if not isinstance(kind, str):
        return None
    return "ai" if kind in _AI_TYPES else kind


def _key(message: Any) -> Any:
    """What tells one message from another across the states of a run: its id, else the object itself."""
    return _field(message, "id") or id(message)


def _as_dicts(messages: Sequence[Message]) -> list[Any]:
    """Transcript messages in the role/content form every LangChain entry point accepts."""
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def _reply_text(value: Any) -> str:
    """Reply text of a run result: a message's text blocks, the last message of a list, else the value."""
    if isinstance(value, list) and value and _kind(value[-1]) is not None:
        value = value[-1]
    return text_of(_field(value, "content")) if _kind(value) is not None else text_of(value)


def _tools_and_usage(fresh: Sequence[Any], earlier: Sequence[Any] = ()) -> tuple[list[ToolExecution], Usage | None]:
    """Tool calls and token usage of the messages one turn added.

    Every ``tool_calls`` entry of a new AI message is paired with the
    ``ToolMessage`` that carries the same id: ``status == "error"`` marks a
    failed call, a call without a ``ToolMessage`` did not run. A call an
    ``earlier`` AI message asked for counts only when its ``ToolMessage`` is
    new, which is what resuming an interrupted graph looks like. Usage is the
    sum of ``usage_metadata`` over the new AI messages; ``requests`` counts
    the messages that carry one.
    """
    results = {_field(message, "tool_call_id"): message for message in fresh if _kind(message) == "tool"}
    tools: list[ToolExecution] = []
    usage: Usage | None = None
    for new, messages in ((False, earlier), (True, fresh)):
        for message in messages:
            if _kind(message) != "ai":
                continue
            if new:
                reported = usage_of(_field(message, "usage_metadata"), requests=1)
                if reported is not None:
                    usage = reported if usage is None else usage + reported
            for call in _field(message, "tool_calls") or []:
                call_id = _field(call, "id")
                result = results.get(call_id) if call_id else None
                if result is None and not new:
                    continue
                tools.append(
                    tool_execution(
                        _field(call, "name"),
                        _field(call, "args"),
                        _field(result, "content"),
                        error=_field(result, "status") == "error",
                        executed=result is not None,
                        call_id=call_id,
                    )
                )
    return tools, usage


class _Thread:
    """What session mode remembers about one conversation between turns."""

    def __init__(self, *, thread_id: str | None = None, messages: list[Any] | None = None, interrupted: bool = False):
        # Thread of a graph with a checkpointer; ``None`` when the adapter carries the messages.
        self.thread_id = thread_id
        self.messages = messages if messages is not None else []
        self.interrupted = interrupted


class _LangChainAdapter(FrameworkAdapter):
    """Transcript -> ``astream`` (graphs) or ``ainvoke`` (everything else) -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        input_key: str | None,
        output_key: str | None,
        inputs: Mapping[str, Any] | None,
        config: Mapping[str, Any] | None,
        resume_fn: Callable[[AgentInput], Any] | None,
        run_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._input_key = input_key
        self._output_key = output_key
        self._inputs = dict(inputs or {})
        self._config = dict(config or {})
        self._resume_fn = resume_fn
        self._run_kwargs = run_kwargs
        self._threads: Conversations[_Thread] = Conversations()
        # Scenario threads that already ran on the checkpointer under their own id.
        self._started: set[str] = set()

    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "ainvoke")

    async def call(self, input: AgentInput) -> AgentTurn:
        target = self._resolve()
        if hasattr(target, "checkpointer") and hasattr(target, "astream"):
            return await self._call_graph(target, input)
        if hasattr(target, "return_intermediate_steps"):
            return await self._call_executor(target, input)
        messages = _as_dicts(input.messages)
        payload = messages if self._input_key is None else {**self._inputs, self._input_key: messages}
        return self._turn(await target.ainvoke(payload, self._run_config(None), **self._run_kwargs))

    def _run_config(self, thread_id: str | None) -> dict[str, Any] | None:
        config = dict(self._config)
        if thread_id is not None:
            config["configurable"] = {**(config.get("configurable") or {}), "thread_id": thread_id}
        return config or None

    # -- LangGraph graphs -----------------------------------------------------------

    async def _call_graph(self, graph: Any, input: AgentInput) -> AgentTurn:
        key = self._input_key or "messages"
        checkpointer = getattr(graph, "checkpointer", None)
        persistent = checkpointer is not None and not isinstance(checkpointer, bool)
        session = self._history == "session"
        known = self._threads.recall(input) if session else None
        messages: list[Any] = _as_dicts(input.messages)
        thread_id: str | None = None
        resume = False
        if persistent and known is not None:
            # The checkpointer holds the dialogue so far: only what is new goes in.
            thread_id = known.thread_id
            messages = _as_dicts(input.new_messages)
            resume = known.interrupted and self._resume_fn is not None
        elif persistent:
            # First turn of the thread, a rollback, a scripted reply or transcript mode: the
            # checkpointer must not add anything to the transcript, so it gets an unused thread.
            fresh = session and input.thread_id not in self._started
            thread_id = input.thread_id if fresh else f"{input.thread_id}-{uuid.uuid4().hex[:12]}"
            self._started.add(input.thread_id)
        elif known is not None:
            messages = [*known.messages, *_as_dicts(input.new_messages)]

        payload: Any = {**self._inputs, key: messages}
        if resume and self._resume_fn is not None:
            command = require("langgraph.types", package="langgraph", adapter="langchain_adapter").Command
            payload = command(resume=self._resume_fn(input))
        final, added, earlier = await self._stream(graph, payload, self._run_config(thread_id), key)

        state = final if isinstance(final, dict) else {}
        interrupts = list(state.get(_INTERRUPT) or [])
        tools, usage = _tools_and_usage(added, earlier)
        if self._content_fn is not None:
            content = self._content_fn(final)
        else:
            raw = state.get(self._output_key) if self._output_key else state.get(key, final)
            content = _reply_text(raw)
            if not content and interrupts:
                # The run stopped to ask a human: what it asks is the reply.
                content = text_of(getattr(interrupts[0], "value", interrupts[0]))
        if session:
            carried = state.get(key)
            if persistent or isinstance(carried, list):
                thread = _Thread(thread_id=thread_id, messages=carried, interrupted=bool(interrupts))
                self._threads.remember(input, content, thread)
            else:
                self._threads.forget(input.thread_id)
        return AgentTurn(content=content, tools=tools or None, usage=usage)

    async def _stream(self, graph: Any, payload: Any, config: Any, key: str) -> tuple[Any, list[Any], list[Any]]:
        """Run the graph; return its final state, the messages the run added and the ones it started with.

        The run is streamed as ``values`` with ``subgraphs=True`` (what
        ``ainvoke`` does, plus the subgraphs), so the messages of sub-agents
        that never reach the parent's state are seen too. The first state of
        the parent graph is the dialogue before the run; any message that
        shows up later, in the parent or in a subgraph, belongs to this turn,
        in the order it appeared.
        """
        final: Any = None
        before: dict[Any, Any] | None = None
        added: dict[Any, Any] = {}
        stream = graph.astream(payload, config, stream_mode=["values"], subgraphs=True, **self._run_kwargs)
        async for namespace, _, state in stream:
            if not namespace:
                final = state
            messages = state.get(key) if isinstance(state, dict) else None
            if not isinstance(messages, list):
                continue
            if before is None:
                before = {_key(message): message for message in messages}
                continue
            for message in messages:
                added.setdefault(_key(message), message)
        earlier = before or {}
        return final, [message for ref, message in added.items() if ref not in earlier], list(earlier.values())

    # -- AgentExecutor ----------------------------------------------------------------

    async def _call_executor(self, executor: Any, input: AgentInput) -> AgentTurn:
        session = self._history == "session"
        known = self._threads.recall(input) if session else None
        pending = _as_dicts(input.messages if known is None else input.new_messages)
        question: Any = {"role": "user", "content": ""}
        for index in range(len(pending) - 1, -1, -1):
            if pending[index]["role"] == "user":
                question = pending.pop(index)
                break
        chat_history = [*(known.messages if known is not None else []), *pending]
        payload = {**self._inputs, "chat_history": chat_history, self._input_key or "input": question["content"]}
        result = await executor.ainvoke(payload, self._run_config(None), **self._run_kwargs)
        turn = self._turn(result)
        if session:
            replayed: list[Any] = []
            for action, observation in _steps(result):
                call_id = getattr(action, "tool_call_id", None)
                if call_id is None:
                    # A text (ReAct) step has no message form; the final answer carries its outcome.
                    continue
                replayed.extend(m for m in getattr(action, "message_log", None) or [] if m not in replayed)
                replayed.append({"role": "tool", "content": text_of(observation), "tool_call_id": call_id})
            messages = [*chat_history, question, *replayed, {"role": "assistant", "content": turn.content}]
            self._threads.remember(input, turn.content, _Thread(messages=messages))
        return turn

    # -- results of ainvoke -----------------------------------------------------------

    def _turn(self, result: Any) -> AgentTurn:
        """An ``AIMessage``, a string or an executor's output dict as an ``AgentTurn``."""
        tools = [
            tool_execution(
                getattr(action, "tool", None),
                getattr(action, "tool_input", None),
                observation,
                call_id=getattr(action, "tool_call_id", None),
            )
            for action, observation in _steps(result)
        ]
        usage: Usage | None = None
        if _kind(result) == "ai":
            # A bare chat model only asks for tools; nothing runs them.
            tools, usage = _tools_and_usage([result])
        if self._content_fn is not None:
            content = self._content_fn(result)
        elif isinstance(result, dict) and _kind(result) is None:
            content = _reply_text(result.get(self._output_key or "output", result))
        else:
            content = _reply_text(result)
        return AgentTurn(content=content, tools=tools or None, usage=usage)


def _steps(result: Any) -> list[Any]:
    """``(action, observation)`` pairs of an ``AgentExecutor`` run; empty for anything else."""
    steps = result.get("intermediate_steps") if isinstance(result, dict) else None
    return list(steps) if isinstance(steps, list) else []


def langchain_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    input_key: str | None = None,
    output_key: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    resume_fn: Callable[[AgentInput], Any] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a LangChain ``Runnable``: a LangGraph graph, a Deep Agent, a chat model or a chain.

    ``target`` is the object itself or a lazy factory (``lambda: build_graph()``
    with the import inside, so collecting tests builds nothing). What a turn
    does depends on what the target is:

    - A compiled LangGraph graph (``create_agent``, ``create_react_agent``,
      ``create_deep_agent``, a ``StateGraph``, a supervisor or swarm graph) is
      streamed with ``{"messages": [...]}``. The reply is the text of the last
      message of the final state. Tool calls are the ``tool_calls`` of the AI
      messages this turn added, paired with their ``ToolMessage`` by id; the
      calls of sub-agents and subgraphs are included even when their messages
      stay out of the parent state. A ``ToolMessage`` with ``status="error"``
      is a failed call; a call that has no ``ToolMessage`` (the graph was
      interrupted before the tool ran) is recorded as not executed, so
      ``state.has_tool_call`` does not count it. A tool exception the graph
      does not catch propagates out of the adapter.
    - A legacy ``AgentExecutor`` gets ``{"input": <last user message>,
      "chat_history": [...]}``; tool calls come from ``intermediate_steps``, so
      build it with ``return_intermediate_steps=True``.
    - Any other runnable (a chat model, a chain) gets the transcript as a list
      of messages and answers with an ``AIMessage`` or a string. Tool calls a
      bare chat model asks for are recorded as not executed.

    ``history="session"`` (the default) keeps the agent's own context between
    turns, tool calls and tool results included. A graph compiled with a
    checkpointer runs on the thread ``input.thread_id`` and receives only the
    new messages. A graph without one, and an ``AgentExecutor``, have no
    memory, so the adapter carries the message list the last turn returned and
    sends it back with the new messages. When the transcript no longer matches
    what the agent saw (a rollback, a reply injected with ``sim.agent("...")``),
    the context is rebuilt from the transcript, on a new checkpointer thread
    when there is a checkpointer. ``history="transcript"`` does that on every
    call: the whole transcript as messages, no thread (a unique one when the
    graph has a checkpointer), nothing remembered.

    ``input_key`` and ``output_key`` are for a non-standard shape. For a graph
    ``input_key`` names the state key that holds the messages (default
    ``"messages"``) and ``output_key`` a state key to answer from instead of
    the last message (``"structured_response"`` and the like). For an
    ``AgentExecutor`` they name the input and output keys (``"input"``,
    ``"output"``). For another runnable ``input_key`` wraps the message list
    into ``{input_key: [...]}`` and ``output_key`` picks the answer out of a
    dict result. ``inputs`` are extra keys of the input dict (other state
    fields, prompt variables). ``content_fn`` renders the reply from what the
    run returned: the final state of a graph, else the result of ``ainvoke``.

    ``config`` is merged into the run config (``recursion_limit``, callbacks,
    ``configurable`` values); with a checkpointer the adapter sets
    ``configurable.thread_id`` itself, so scenarios never share a thread.
    Extra keyword arguments go to ``astream`` / ``ainvoke`` unchanged:
    ``context=...``, ``durability=...`` and so on.

    ``resume_fn`` makes human-in-the-loop graphs testable. When the previous
    turn of a checkpointed thread ended in an interrupt, the next call resumes
    the graph with ``Command(resume=resume_fn(input))`` instead of sending the
    user message as a new one. Without it an interrupted thread simply
    receives the next user message. An interrupted turn whose last message has
    no text answers with the interrupt's value.

    Usage is the sum of ``usage_metadata`` over the AI messages the turn
    added, with ``requests`` counting those messages.

    >>> adapter = langchain_adapter(lambda: create_agent(model, tools, checkpointer=InMemorySaver()))
    >>> adapter = langchain_adapter(graph, context={"user_id": "42"}, config={"recursion_limit": 50})
    """
    return _LangChainAdapter(
        target,
        history=history,
        content_fn=content_fn,
        input_key=input_key,
        output_key=output_key,
        inputs=inputs,
        config=config,
        resume_fn=resume_fn,
        run_kwargs=run_kwargs,
    )
