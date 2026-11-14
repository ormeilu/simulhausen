"""Adapter for DSPy modules: ``dspy.ReAct``, ``dspy.Predict``, ``dspy.ChainOfThought`` or your own ``dspy.Module``.

``dspy_adapter`` puts the user's message into one input field of the module's
signature, calls the module and reads the reply from one output field. DSPy is
imported only when a ``dspy.History`` has to be built, so the module loads
without ``dspy`` installed.
"""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any, get_args

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

    from simulhausen.types import Message

__all__ = ["dspy_adapter"]

_ADAPTER = "dspy_adapter"
# ReAct's own way out of its loop: ``submit`` (``finish`` before DSPy 3.5) carries the answer, it is not a tool call.
_FINAL_TOOLS = frozenset({"submit", "finish"})


def _signature(module: Any) -> Any:
    """The signature that names the module's inputs and outputs: its own, else its first predictor's."""
    own = getattr(module, "signature", None)
    if own is not None:
        return own
    predictors = getattr(module, "predictors", None)
    found = predictors() if callable(predictors) else []
    return getattr(found[0], "signature", None) if found else None


def _fields(signature: Any, kind: str) -> dict[str, Any]:
    fields = getattr(signature, kind, None)
    return dict(fields) if isinstance(fields, dict) else {}


def _is_history(field: Any) -> bool:
    """Whether a signature field is typed ``dspy.History`` (or ``dspy.History | None``)."""
    annotation = getattr(field, "annotation", None)
    return any(getattr(kind, "__name__", "") == "History" for kind in (annotation, *get_args(annotation)))


def _user_text(messages: Sequence[Message]) -> str:
    return "\n\n".join(message["content"] for message in messages if message["role"] == "user")


def _exchanges(messages: Sequence[Message], input_field: str, output_field: str) -> tuple[list[dict[str, str]], str]:
    """The transcript as ``dspy.History`` entries plus the text still to answer.

    Every agent reply closes one entry ``{input_field: ..., output_field: ...}``;
    the user messages after the last reply are the text to answer. A transcript
    that ends with a reply has nothing left to answer: its last user message is
    asked again.
    """
    entries: list[dict[str, str]] = []
    waiting: list[str] = []
    asked = ""
    for message in messages:
        if message["role"] == "user":
            waiting.append(message["content"])
            continue
        entry = {input_field: "\n\n".join(waiting)} if waiting else {}
        asked = entry.get(input_field, asked)
        entries.append({**entry, output_field: message["content"]})
        waiting = []
    return entries, "\n\n".join(waiting) if waiting else asked


def _history_tools(events: Sequence[Any]) -> list[ToolExecution]:
    """Tool calls from the events ReAct appends to ``prediction.history`` (DSPy 3.5 and later).

    An event may hold ``tool_calls``, a ``dspy.ToolCalls`` whose calls
    (``id``, ``name``, ``args``) carry their results in ``tool_call_results``
    (``call_id``, ``value``, ``is_error``). Results are matched by call id,
    then by position.
    """
    executions: list[ToolExecution] = []
    for event in events:
        batch = event.get("tool_calls") if isinstance(event, dict) else None
        calls = getattr(batch, "tool_calls", None) or []
        results = getattr(getattr(batch, "tool_call_results", None), "tool_call_results", None) or []
        for index, call in enumerate(calls):
            if call.name in _FINAL_TOOLS:
                continue
            call_id = getattr(call, "id", None)
            match = next((result for result in results if call_id is not None and result.call_id == call_id), None)
            if match is None and index < len(results) and getattr(results[index], "call_id", None) is None:
                match = results[index]
            if match is None:
                executions.append(tool_execution(call.name, call.args, executed=False, call_id=call_id))
            else:
                executions.append(
                    tool_execution(call.name, call.args, match.value, error=bool(match.is_error), call_id=call_id)
                )
    return executions


def _trajectory_tools(trajectory: Any) -> list[ToolExecution]:
    """Tool calls from ``prediction.trajectory`` of the older ReAct (DSPy up to 3.4).

    The trajectory is a flat dict ``tool_name_0``, ``tool_args_0``,
    ``observation_0``, ``tool_name_1``... A tool that raised leaves the
    observation ``"Execution error in <tool>: ..."``; that text is the only
    trace of the failure, so it marks the call as an error.
    """
    executions: list[ToolExecution] = []
    index = 0
    while isinstance(trajectory, dict) and f"tool_name_{index}" in trajectory:
        name = trajectory[f"tool_name_{index}"]
        key = f"observation_{index}"
        observation = trajectory.get(key)
        failed = isinstance(observation, str) and observation.startswith(f"Execution error in {name}:")
        if name not in _FINAL_TOOLS:
            executions.append(
                tool_execution(
                    name, trajectory.get(f"tool_args_{index}"), observation, error=failed, executed=key in trajectory
                )
            )
        index += 1
    return executions


def _usage(prediction: Any) -> Usage | None:
    """``prediction.get_lm_usage()`` (filled when ``dspy.settings.track_usage`` is on), summed over the models."""
    read = getattr(prediction, "get_lm_usage", None)
    report = read() if callable(read) else None
    total: Usage | None = None
    for entry in report.values() if isinstance(report, dict) else ():
        usage = usage_of(entry)
        if usage is not None:
            total = usage if total is None else total + usage
    return total


class _DSPyAdapter(FrameworkAdapter):
    """User text -> ``module.acall(**inputs)`` -> ``AgentTurn``."""

    def __init__(
        self,
        target: Any,
        *,
        history: FrameworkHistory,
        content_fn: Callable[[Any], str] | None,
        input_field: str | None,
        output_field: str | None,
        history_field: str | None,
        call_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target, history=history, content_fn=content_fn)
        self._input_field = input_field
        self._output_field = output_field
        self._history_field = history_field
        self._call_kwargs = call_kwargs
        # thread_id -> the ``dspy.History`` a ReAct module returned, as it stood after the agent's reply.
        self._conversations: Conversations[Any] = Conversations()

    def _is_target(self, target: Any) -> bool:
        # DSPy modules are callable: tell them from a factory by what a module has. A module class is a factory.
        return not isinstance(target, type) and any(
            hasattr(target, name) for name in ("predictors", "forward", "aforward")
        )

    async def call(self, input: AgentInput) -> AgentTurn:
        module = self._resolve()
        signature = _signature(module)
        inputs = _fields(signature, "input_fields")
        outputs = _fields(signature, "output_fields")
        history_field = self._history_field or next((name for name in inputs if _is_history(inputs[name])), None)
        input_field = self._input_field or next((name for name in inputs if name != history_field), "question")
        output_field = (
            self._output_field
            or next((name for name in outputs if name != "reasoning"), None)
            or next(iter(outputs), None)
        )
        # ReAct of DSPy 3.5 and later takes ``history=`` and returns it extended with what it did.
        react_inputs = _fields(getattr(getattr(module, "react", None), "signature", None), "input_fields")
        returns_history = history_field is None and "history" in react_inputs

        kwargs = dict(self._call_kwargs)
        carried = self._conversations.recall(input) if returns_history and self._history == "session" else None
        known = 0
        if carried is not None:
            # The module appends to the history it is given: hand it a copy, so a failed call leaves ours intact.
            kwargs["history"] = copy.deepcopy(carried)
            known = len(carried.messages)
            text = _user_text(input.new_messages)
        elif history_field is not None:
            entries, text = _exchanges(input.messages, input_field, output_field or "answer")
            dspy = require("dspy", package="dspy", adapter=_ADAPTER)
            kwargs[history_field] = dspy.History(messages=entries)
        elif len(input.messages) > 1:
            text = transcript_prompt(input.messages)
        else:
            text = input.last_new_user_message_str()
        kwargs[input_field] = text

        # ``Module.acall`` awaits ``aforward``; a module that only defines ``forward`` runs in a worker thread.
        if hasattr(module, "aforward"):
            prediction = await module.acall(**kwargs)
        else:
            prediction = await asyncio.to_thread(module, **kwargs)

        raw = getattr(prediction, output_field, None) if output_field is not None else prediction
        content = self._render(raw)
        native = getattr(prediction, "history", None)
        events = getattr(native, "messages", None)
        if isinstance(events, list):
            tools = _history_tools(events[known:])
            if returns_history and self._history == "session":
                self._conversations.remember(input, content, native)
        else:
            tools = _trajectory_tools(getattr(prediction, "trajectory", None))
        return AgentTurn(content=content, tools=tools or None, usage=_usage(prediction))


def dspy_adapter(
    target: Any,
    *,
    history: FrameworkHistory = "session",
    content_fn: Callable[[Any], str] | None = None,
    input_field: str | None = None,
    output_field: str | None = None,
    history_field: str | None = None,
    **call_kwargs: Any,
) -> AgentAdapter:
    """Adapter for a DSPy module (``pip install dspy``).

    ``target`` is a ``dspy.Module`` (``dspy.ReAct``, ``dspy.Predict``,
    ``dspy.ChainOfThought``, your own module) or a zero-argument factory
    returning one. A turn is one ``await module.acall(**inputs)``; a module
    that defines only ``forward`` is called in a worker thread, because
    ``acall`` needs ``aforward``. Extra kwargs go to the call unchanged: the
    other input fields of the signature, ``max_iters=...`` for ReAct. The
    language model comes from ``dspy.configure(...)`` as usual.

    The user's text goes into ``input_field`` (default: the first input field
    of the module's signature that is not a ``dspy.History``, or ``question``
    when the module shows no signature) and the reply is read from
    ``output_field`` (default: the first output field, ``reasoning`` left
    out), through ``content_fn`` when given. The signature is the module's
    own ``signature`` or that of its first predictor; name the fields for a
    module whose ``forward`` takes something else.

    DSPy modules hold no conversation, so the context is rebuilt from the
    transcript:

    - a signature with a ``dspy.History`` input (``history_field``, found by
      its type) gets the earlier exchanges as
      ``History(messages=[{input_field: ..., output_field: ...}, ...])`` and
      the new user text in ``input_field``, in both history modes;
    - any other module gets the whole transcript rendered into
      ``input_field`` once the dialogue has more than one message.

    The exception is ``dspy.ReAct`` of DSPy 3.5 and later, which accepts
    ``history=`` and returns it in ``prediction.history`` extended with its
    thoughts, tool calls and results. With ``history="session"`` (default)
    the adapter carries that history between turns, so earlier tool results
    stay in context; after a rollback or a scripted ``sim.agent("...")``
    reply, and always with ``history="transcript"``, it starts again from the
    rendered transcript.

    Tool calls come from ReAct: the events added to ``prediction.history``
    (calls paired with their results, ``is_error`` kept), or
    ``prediction.trajectory`` of DSPy up to 3.4 (``tool_name_i``,
    ``tool_args_i``, ``observation_i``, where an observation that starts with
    ``Execution error in <tool>:`` is an error). The final ``submit`` /
    ``finish`` step is not a tool call. Usage is ``prediction.get_lm_usage()``
    summed over the models, which DSPy fills only with
    ``dspy.configure(track_usage=True)``; it gives tokens, not a request
    count.

    >>> adapter = dspy_adapter(lambda: dspy.ReAct("question -> answer", tools=[lookup_order]))
    """
    return _DSPyAdapter(
        target,
        history=history,
        content_fn=content_fn,
        input_field=input_field,
        output_field=output_field,
        history_field=history_field,
        call_kwargs=call_kwargs,
    )
