"""Adapter for Hugging Face smolagents: ``CodeAgent``, ``ToolCallingAgent`` and agents with managed agents.

smolagents is synchronous, so a turn is ``agent.run(task, reset=...)`` in a
worker thread. The run's tool calls are recorded where they happen: for the
duration of the call the entries of ``agent.tools`` and ``agent.managed_agents``
(and of every managed agent below) are swapped for recording stand-ins, which is
the only way to see the real tools a ``CodeAgent`` calls from its generated
code. The memory steps the run added supply the rest: call ids, calls that never
reached a tool, and the code of a step that called none.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    FrameworkHistory,
    tool_execution,
    transcript_prompt,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.response import ToolExecution

__all__ = ["smolagents_adapter"]

# The tool every smolagents run ends with; it carries the reply, not an action.
_FINAL_ANSWER = "final_answer"
# Keyword Tool.__call__ takes from the agent loop; not an argument of the tool.
_SANITIZE = "sanitize_inputs_outputs"


@dataclass
class _Call:
    """One call of a tool or a managed agent, recorded when it started."""

    step: int
    name: str
    args: dict[str, Any]
    result: Any = None
    error: bool = False


class _Run:
    """What one ``agent.run`` did: the recorded calls and the tokens of the model calls."""

    def __init__(self, memory: Any) -> None:
        self._memory = memory
        self.calls: list[_Call] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0

    def begin(self, name: str, args: dict[str, Any]) -> _Call:
        # The step in progress joins the top agent's memory when it ends, so its
        # future index is the current length. Calls of managed agents' tools
        # land in the top agent's step that delegated to them.
        call = _Call(step=len(self._memory.steps), name=name, args=args)
        self.calls.append(call)
        return call

    def add_usage(self, steps: Sequence[Any]) -> None:
        for step in steps:
            usage = getattr(step, "token_usage", None)
            if usage is not None:
                self.input_tokens += usage.input_tokens
                self.output_tokens += usage.output_tokens
                self.requests += 1

    def usage(self) -> Usage | None:
        usage = Usage(input_tokens=self.input_tokens, output_tokens=self.output_tokens, requests=self.requests)
        return usage if usage else None


def _call_args(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Arguments of a call by name; positional ones (generated code uses them) get the tool's input names."""
    named = {key: value for key, value in kwargs.items() if key != _SANITIZE}
    names = list(getattr(target, "inputs", None) or {})
    if len(args) == 1 and not named and isinstance(args[0], dict) and all(key in names for key in args[0]):
        return dict(args[0])
    positional = {names[index] if index < len(names) else f"arg{index}": value for index, value in enumerate(args)}
    return {**positional, **named}


class _Recorded:
    """Stands in for a tool or a managed agent during one run: records the call, then makes it."""

    def __init__(self, wrapped: Any, run: _Run, *, managed: bool) -> None:
        self._wrapped = wrapped
        self._run = run
        self._managed = managed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call = self._run.begin(self._wrapped.name, _call_args(self._wrapped, args, kwargs))
        try:
            call.result = self._wrapped(*args, **kwargs)
        except Exception as exc:
            call.result, call.error = str(exc), True
            raise
        finally:
            if self._managed:
                # A managed agent resets its memory on every call, so its steps are this call's.
                self._run.add_usage(self._wrapped.memory.steps)
        return call.result


def _instrument(agent: Any, run: _Run, restore: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    """Swap the agent's tools and managed agents for recording stand-ins, managed agents' own included."""
    for registry, managed in ((agent.tools, False), (getattr(agent, "managed_agents", None) or {}, True)):
        originals = dict(registry)
        restore.append((registry, originals))
        for name, item in originals.items():
            if managed:
                _instrument(item, run, restore)
            if name != _FINAL_ANSWER:
                registry[name] = _Recorded(item, run, managed=managed)


def _requested(step: Any) -> list[tuple[str, Any, Any]]:
    """The calls a memory step says the model asked for, as ``(name, arguments, id)``."""
    calls = getattr(step, "tool_calls", None)
    if calls:
        return [(call.name, call.arguments, call.id) for call in calls]
    # A ToolCallingAgent step whose tool raised keeps the calls only on the model's message.
    message = getattr(step, "model_output_message", None)
    return [
        (call.function.name, call.function.arguments, call.id) for call in getattr(message, "tool_calls", None) or []
    ]


def _executions(calls: list[_Call], first: int, steps: Sequence[Any]) -> list[ToolExecution]:
    """Tool executions of the steps a run added, in call order."""
    executions: list[ToolExecution] = []
    for index, step in enumerate(steps, first):
        requested = _requested(step)
        ran = [call for call in calls if call.step == index]
        ids: dict[str, list[Any]] = {}
        for name, _, call_id in requested:
            ids.setdefault(name, []).append(call_id)
        for call in ran:
            queue = ids.get(call.name)
            call_id = queue.pop(0) if queue else None
            executions.append(tool_execution(call.name, call.args, call.result, error=call.error, call_id=call_id))
        failure = getattr(step, "error", None)
        code = getattr(step, "code_action", None)
        ran_names = {call.name for call in ran}
        for name, args, call_id in requested:
            if name == _FINAL_ANSWER or name in ran_names:
                continue
            if code is not None:
                # A CodeAgent step. Its real tool calls are reported above; the interpreter call
                # itself only when the code called no tool and did more than deliver the answer.
                if ran or (getattr(step, "is_final_answer", False) and failure is None):
                    continue
                args = {"code": code}
            result = str(failure) if failure is not None else getattr(step, "observations", None)
            executions.append(tool_execution(name, args, result, error=failure is not None, call_id=call_id))
    return executions


class _SmolagentsAdapter(FrameworkAdapter):
    """Newest user message or rendered transcript -> ``agent.run`` in a thread -> ``AgentTurn``."""

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
        self._shared = self._is_target(target) or not callable(target)
        self._agents: dict[str, Any] = {}
        # Threads whose dialogue their agent's memory still holds.
        self._live: Conversations[bool] = Conversations()
        # A shared agent instance holds one dialogue at a time and runs one call at a time.
        self._owner: str | None = None
        self._lock = threading.Lock()

    def _is_target(self, target: Any) -> bool:
        # Agents are callable (that is how a manager calls a managed agent), factories have no memory.
        return hasattr(target, "run") and hasattr(target, "memory")

    def _agent(self, thread_id: str) -> Any:
        if self._shared:
            return self._target
        if thread_id not in self._agents:
            self._agents[thread_id] = self._target()
        return self._agents[thread_id]

    async def call(self, input: AgentInput) -> AgentTurn:
        return await asyncio.to_thread(self._turn, input)

    def _turn(self, input: AgentInput) -> AgentTurn:
        agent = self._agent(input.thread_id)
        with self._lock if self._shared else contextlib.nullcontext():
            continues = (
                self._history == "session"
                and self._live.recall(input) is not None
                and (not self._shared or self._owner == input.thread_id)
            )
            if continues:
                task, reset = input.last_new_user_message_str(), False
            else:
                task, reset = transcript_prompt(input.messages), True
            first = 0 if reset else len(agent.memory.steps)
            run = _Run(agent.memory)
            restore: list[tuple[dict[str, Any], dict[str, Any]]] = []
            try:
                _instrument(agent, run, restore)
                result = agent.run(task, reset=reset, **self._run_kwargs)
            finally:
                for registry, originals in restore:
                    registry.update(originals)
            steps = agent.memory.steps[first:]
            run.add_usage(steps)
            # With return_full_result the run returns a RunResult around the output.
            output = result.output if hasattr(result, "output") and hasattr(result, "token_usage") else result
            content = self._render(output)
            self._owner = input.thread_id
            self._live.remember(input, content, True)
            return AgentTurn(content=content, tools=_executions(run.calls, first, steps) or None, usage=run.usage())


def smolagents_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    **run_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a smolagents ``CodeAgent`` or ``ToolCallingAgent``, with or without managed agents.

    ``target`` is the agent or a zero-argument factory returning one. An agent
    owns its memory, so pass a factory when the adapter serves more than one
    scenario: every ``thread_id`` then gets its own agent. A single agent
    instance holds one dialogue at a time; its runs are serialised, and a turn
    of another scenario starts it over from that scenario's transcript.

    A turn is ``agent.run(task, reset=...)`` in a worker thread
    (``asyncio.to_thread``). Extra kwargs go to ``run`` unchanged
    (``max_steps=...``, ``additional_args=...``); ``reset`` belongs to the
    adapter and ``stream=True`` is not supported.

    ``history="session"`` (default) sends the newest user message with
    ``reset=False`` after the first turn, so the agent keeps its memory steps,
    tool calls and observations of earlier turns included (and a ``CodeAgent``
    its interpreter variables). When the transcript no longer continues what
    the agent answered (a rollback, a reply injected with ``sim.agent("...")``)
    the next turn runs with ``reset=True`` on the rendered transcript.
    ``history="transcript"`` does that on every call: ``reset=True`` and the
    whole transcript as one task text, so a rollback reaches the agent and
    earlier tool results are not replayed.

    Tool calls come from recording stand-ins that replace the entries of
    ``agent.tools`` and ``agent.managed_agents`` while the run lasts (the
    originals are put back afterwards, the tool objects themselves are not
    touched). That reports every tool with its arguments, result and error in
    call order: the tools a ``CodeAgent`` calls from its generated code, a
    delegation to a managed agent (under the managed agent's name, with its
    report as the result) and the tools that managed agent called. A call that
    never reached a tool (unknown tool, arguments the tool rejects) is taken
    from the memory step and marked as an error. A ``CodeAgent`` step whose
    code called no tool is reported as one ``python_interpreter`` call with the
    code as ``{"code": ...}`` and the observations as the result; that is also
    all there is when the code runs in a remote executor, where tool calls
    leave the process. ``final_answer`` only carries the reply, so it is never
    reported, and neither is a ``python_interpreter`` step that did nothing but
    deliver the answer.

    ``AgentTurn.usage`` sums ``token_usage`` over the steps the run added,
    managed agents' steps included; ``requests`` counts the steps that report
    tokens. It is ``None`` when the model reports none.

    ``content_fn`` renders the final answer into the reply text; without it a
    string stays as it is and any other value becomes JSON.

    >>> adapter = smolagents_adapter(lambda: CodeAgent(tools=[lookup_order], model=model), max_steps=6)
    """
    return _SmolagentsAdapter(target, run_kwargs, history=history, content_fn=content_fn)
