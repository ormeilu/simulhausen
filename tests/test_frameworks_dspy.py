import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import dspy as module
from simulhausen.frameworks.dspy import dspy_adapter

# -- fakes shaped like dspy ----------------------------------------------------------


@dataclass
class History:
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class ToolCallResult:
    value: Any
    call_id: str | None = None
    is_error: bool = False


@dataclass
class ToolCalls:
    tool_calls: list[ToolCall]
    tool_call_results: Any = None


class Prediction:
    def __init__(self, usage: Any = None, **fields: Any) -> None:
        self.__dict__.update(fields)
        self._usage = usage

    def get_lm_usage(self) -> Any:
        return self._usage


def _signature(inputs: dict[str, Any], outputs: list[str]) -> Any:
    return SimpleNamespace(
        input_fields={name: SimpleNamespace(annotation=kind) for name, kind in inputs.items()},
        output_fields={name: SimpleNamespace(annotation=str) for name in outputs},
    )


class Predict:
    """A predictor: has a signature and ``aforward``; callable, like every DSPy module."""

    def __init__(self, signature: Any, *predictions: Any) -> None:
        self.signature = signature
        self.predictions = list(predictions)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        raise AssertionError("a module with aforward is awaited, and is not a factory")

    async def aforward(self, **kwargs: Any) -> Any:
        raise AssertionError("acall is the entry point")

    async def acall(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.predictions.pop(0) if self.predictions else Prediction(answer=f"answer {len(self.calls)}")


class ReAct(Predict):
    """ReAct of DSPy 3.5: takes ``history=``, appends its events to it and returns it."""

    def __init__(self, *turns: list[dict[str, Any]], fail_on: int | None = None) -> None:
        super().__init__(_signature({"question": str}, ["answer"]))
        self.react = SimpleNamespace(signature=_signature({"question": str, "history": History}, ["tool_calls"]))
        self.turns = list(turns)
        self.fail_on = fail_on
        self.given: list[int | None] = []

    async def acall(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        history = kwargs.get("history")
        self.given.append(None if history is None else len(history.messages))
        history = history if history is not None else History()
        history.messages.extend(self.turns.pop(0) if self.turns else [{"question": kwargs["question"]}])
        if self.fail_on == len(self.calls):
            raise RuntimeError("down")
        return Prediction(answer=f"answer {len(self.calls)}", history=history)


@pytest.fixture(autouse=True)
def framework(monkeypatch: pytest.MonkeyPatch) -> None:
    def require(name: str, *, package: str, adapter: str) -> Any:
        assert (name, package, adapter) == ("dspy", "dspy", "dspy_adapter")
        return SimpleNamespace(History=History)

    monkeypatch.setattr(module, "require", require)


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, messages: list[tuple[str, str]], **kwargs: Any) -> AgentTurn:
    turn = await adapter.call(_input(messages, **kwargs))
    assert isinstance(turn, AgentTurn)
    return turn


# -- fields --------------------------------------------------------------------------


async def test_fields_default_to_the_signature_and_extra_kwargs_reach_the_call() -> None:
    predict = Predict(
        _signature({"context": str, "question": str}, ["reasoning", "answer"]),
        Prediction(reasoning="because", answer="42", usage={"m": {"prompt_tokens": 5, "completion_tokens": 2}}),
    )

    turn = await _turn(dspy_adapter(predict, input_field="question", context="docs"), [("user", "why?")])

    assert predict.calls == [{"context": "docs", "question": "why?"}]
    assert turn.content == "42"
    assert turn.tools is None
    assert turn.usage == Usage(input_tokens=5, output_tokens=2, total_tokens=7)


async def test_named_fields_and_content_fn() -> None:
    class Answer(BaseModel):
        text: str

    predict = Predict(
        _signature({"question": str}, ["answer", "extra"]), Prediction(answer="a", extra=Answer(text="x"))
    )
    custom = Predict(_signature({"question": str}, ["reasoning"]), Prediction(reasoning=Answer(text="y")))

    plain = await _turn(dspy_adapter(predict, output_field="extra"), [("user", "q")])
    rendered = await _turn(dspy_adapter(custom, content_fn=lambda raw: raw.text), [("user", "q")])

    assert plain.content == '{"text":"x"}'
    # ``reasoning`` is the answer only when the signature has no other output.
    assert rendered.content == "y"


async def test_a_module_without_a_signature_uses_its_first_predictor_or_question() -> None:
    inner = Predict(_signature({"query": str}, ["reply"]))

    class Wrapper:
        def __init__(self, predictors: list[Any]) -> None:
            self._predictors = predictors
            self.calls: list[dict[str, Any]] = []
            self.threads: list[str] = []

        def predictors(self) -> list[Any]:
            return self._predictors

        def __call__(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            self.threads.append(threading.current_thread().name)
            return Prediction(reply="from the wrapper") if self._predictors else "a bare string"

    wrapped, bare = Wrapper([inner]), Wrapper([])

    first = await _turn(dspy_adapter(wrapped), [("user", "hi")])
    second = await _turn(dspy_adapter(bare), [("user", "hi")])

    assert (wrapped.calls, first.content) == ([{"query": "hi"}], "from the wrapper")
    # Nothing to read the fields from: "question" in, the returned value itself out, no usage.
    assert (bare.calls, second.content, second.usage) == ([{"question": "hi"}], "a bare string", None)
    # Neither has ``aforward``: both ran in a worker thread.
    assert threading.current_thread().name not in wrapped.threads + bare.threads


async def test_factories_and_module_classes_are_built_once() -> None:
    built: list[Any] = []

    class Module:
        def __init__(self) -> None:
            built.append(self)

        def forward(self, **kwargs: Any) -> Any:
            raise AssertionError("called through __call__")

        def __call__(self, **kwargs: Any) -> Any:
            return Prediction(answer="built")

    adapter = dspy_adapter(Module, output_field="answer")
    await _turn(adapter, [("user", "a")])
    turn = await _turn(adapter, [("user", "b")])

    assert (turn.content, len(built)) == ("built", 1)
    assert (await _turn(dspy_adapter(lambda: Module(), output_field="answer"), [("user", "a")])).content == "built"


# -- history -------------------------------------------------------------------------


@pytest.mark.parametrize("history", ["session", "transcript"])
async def test_a_history_input_gets_the_earlier_exchanges(history: Any) -> None:
    predict = Predict(_signature({"history": History | None, "question": str}, ["answer"]))
    adapter = dspy_adapter(predict, history=history)

    await _turn(adapter, [("user", "q1")])
    await _turn(
        adapter,
        [
            ("assistant", "welcome"),
            ("user", "q1"),
            ("user", "q1b"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("user", "q2b"),
        ],
        new=2,
    )
    await _turn(adapter, [("user", "q1"), ("assistant", "a1")], new=0)

    assert predict.calls[0] == {"history": History([]), "question": "q1"}
    assert predict.calls[1] == {
        "history": History([{"answer": "welcome"}, {"question": "q1\n\nq1b", "answer": "a1"}]),
        "question": "q2\n\nq2b",
    }
    # Nothing new to answer: the last user message is asked again.
    assert predict.calls[2] == {"history": History([{"question": "q1", "answer": "a1"}]), "question": "q1"}


async def test_a_named_history_field_without_an_output_field_uses_answer() -> None:
    module_ = Predict(SimpleNamespace(input_fields=None, output_fields=None))
    adapter = dspy_adapter(module_, history_field="chat", input_field="text")

    await _turn(adapter, [("user", "q1"), ("assistant", "a1"), ("user", "q2")])

    assert module_.calls == [{"chat": History([{"text": "q1", "answer": "a1"}]), "text": "q2"}]


async def test_a_module_without_history_gets_the_transcript_as_text() -> None:
    predict = Predict(_signature({"question": str}, ["answer"]))
    adapter = dspy_adapter(predict)

    await _turn(adapter, [("user", "q1")])
    await _turn(adapter, [("user", "q1"), ("assistant", "a1"), ("user", "q2")])

    assert predict.calls[0] == {"question": "q1"}
    assert "User: q1\nAssistant: a1\nUser: q2" in predict.calls[1]["question"]


async def test_react_history_is_carried_in_session_mode() -> None:
    lookup = ToolCall("lookup", {"id": 7}, id="c1")
    first_turn = [
        {
            "question": "q1",
            "tool_calls": ToolCalls([lookup], SimpleNamespace(tool_call_results=[ToolCallResult("ok", "c1")])),
        },
        {"tool_calls": ToolCalls([ToolCall("submit", {"answer": "a"})]), "answer": "a"},
    ]
    react = ReAct(first_turn)
    adapter = dspy_adapter(react)

    first = await _turn(adapter, [("user", "q1")])
    second = await _turn(adapter, [("user", "q1"), ("assistant", first.content), ("user", "q2"), ("user", "q3")], new=2)
    other = await _turn(adapter, [("user", "x")], thread_id="u")

    assert react.given == [None, 2, None]
    assert react.calls[1]["question"] == "q2\n\nq3"
    assert first.tools is not None
    assert [(tool.tool_name, tool.result) for tool in first.tools] == [("lookup", "ok")]
    # Only this turn's events are read.
    assert (second.tools, other.tools) == (None, None)


async def test_react_starts_from_the_transcript_after_a_rollback_and_in_transcript_mode() -> None:
    react = ReAct()
    adapter = dspy_adapter(react)
    await _turn(adapter, [("user", "q1")])
    await _turn(adapter, [("user", "other"), ("assistant", "scripted"), ("user", "q2")])

    stateless = ReAct()
    transcript = dspy_adapter(stateless, history="transcript")
    await _turn(transcript, [("user", "q1")])
    await _turn(transcript, [("user", "q1"), ("assistant", "answer 1"), ("user", "q2")])

    assert react.given == [None, None]
    assert "User: other\nAssistant: scripted\nUser: q2" in react.calls[1]["question"]
    assert stateless.given == [None, None]
    assert "User: q1\nAssistant: answer 1\nUser: q2" in stateless.calls[1]["question"]


async def test_a_failed_call_leaves_the_carried_history_intact() -> None:
    react = ReAct(fail_on=2)
    adapter = dspy_adapter(react)
    follow_up = [("user", "q1"), ("assistant", "answer 1"), ("user", "q2")]

    await _turn(adapter, [("user", "q1")])
    with pytest.raises(RuntimeError, match="down"):
        await _turn(adapter, follow_up)
    await _turn(adapter, follow_up)

    # The failed call appended to a copy: the retry starts from the same one event.
    assert react.given == [None, 1, 1]


# -- tools and usage -----------------------------------------------------------------


async def test_tool_calls_from_the_react_history() -> None:
    events: list[Any] = [
        "not an event",
        {"question": "q"},
        {
            "tool_calls": ToolCalls(
                [
                    ToolCall("a", {"n": 1}, id="c1"),
                    ToolCall("b", {"n": 2}),
                    ToolCall("c", {}, id="c3"),
                    ToolCall("d", {}),
                ],
                SimpleNamespace(
                    tool_call_results=[
                        ToolCallResult("boom", "c1", is_error=True),
                        ToolCallResult({"ok": True}),
                        ToolCallResult("for another call", "zz"),
                    ]
                ),
            )
        },
        {"tool_calls": ToolCalls([ToolCall("finish", {}), ToolCall("submit", {"answer": "x"})])},
    ]

    turn = await _turn(dspy_adapter(ReAct(events)), [("user", "q")])

    assert turn.tools is not None
    assert [
        (t.tool_name, t.tool_args, t.result, bool(t.tool_call_error), t.is_paused, t.tool_call_id) for t in turn.tools
    ] == [
        ("a", {"n": 1}, "boom", True, False, "c1"),
        ("b", {"n": 2}, '{"ok": true}', False, False, None),
        ("c", {}, None, False, True, "c3"),
        ("d", {}, None, False, True, None),
    ]


async def test_tool_calls_from_the_older_trajectory() -> None:
    trajectory = {
        "thought_0": "look",
        "tool_name_0": "lookup",
        "tool_args_0": {"id": 7},
        "observation_0": "shipped",
        "tool_name_1": "refund",
        "tool_args_1": {"id": 7},
        "observation_1": "Execution error in refund: RuntimeError('closed')",
        "tool_name_2": "wait",
        "tool_args_2": {},
        "tool_name_3": "finish",
        "tool_args_3": {},
        "observation_3": "Completed.",
    }
    predict = Predict(_signature({"question": str}, ["answer"]), Prediction(answer="done", trajectory=trajectory))

    turn = await _turn(dspy_adapter(predict), [("user", "q")])

    assert turn.tools is not None
    assert [(t.tool_name, t.tool_args, t.result, bool(t.tool_call_error), t.is_paused) for t in turn.tools] == [
        ("lookup", {"id": 7}, "shipped", False, False),
        ("refund", {"id": 7}, "Execution error in refund: RuntimeError('closed')", True, False),
        ("wait", {}, None, False, True),
    ]


async def test_usage_is_summed_over_the_models() -> None:
    report = {
        "a": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "b": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "idle": {"prompt_tokens": 0},
    }
    signature = _signature({"question": str}, ["answer"])

    summed = await _turn(dspy_adapter(Predict(signature, Prediction(answer="x", usage=report))), [("user", "q")])
    off = await _turn(dspy_adapter(Predict(signature, Prediction(answer="x", usage=None))), [("user", "q")])

    assert summed.usage == Usage(input_tokens=6, output_tokens=3, total_tokens=9)
    assert off.usage is None


def test_an_unknown_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="history must be"):
        dspy_adapter(ReAct(), history="server")  # ty: ignore[invalid-argument-type]
