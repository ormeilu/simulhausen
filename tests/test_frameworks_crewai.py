"""``crewai_adapter`` against hand-written fakes shaped like CrewAI's Agent, Crew, Flow and event bus."""

import contextvars
import itertools
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks.crewai import crewai_adapter

_clock = itertools.count()


class _ToolEvent:
    def __init__(self, tool_name: str, tool_args: Any, at: int | None = None, **fields: Any) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.timestamp = next(_clock) if at is None else at
        for name, value in fields.items():
            setattr(self, name, value)


class ToolUsageFinishedEvent(_ToolEvent):
    failure: Any = None


class ToolUsageErrorEvent(_ToolEvent):
    pass


class ToolSelectionErrorEvent(_ToolEvent):
    pass


class FakeBus:
    """Shaped like ``crewai_event_bus``: handlers per event type, called with ``(source, event)``."""

    def __init__(self) -> None:
        self.handlers: dict[type, list[Callable[[Any, Any], None]]] = {}
        self.flushes = 0

    def register_handler(self, event_type: type, handler: Callable[[Any, Any], None]) -> None:
        self.handlers.setdefault(event_type, []).append(handler)

    def off(self, event_type: type, handler: Callable[[Any, Any], None]) -> None:
        self.handlers[event_type].remove(handler)

    def flush(self) -> bool:
        self.flushes += 1
        return True

    def emit(self, event: Any) -> None:
        for handler in list(self.handlers.get(type(event), [])):
            handler(self, event)

    @property
    def listening(self) -> int:
        return sum(len(handlers) for handlers in self.handlers.values())


@pytest.fixture
def bus(monkeypatch: pytest.MonkeyPatch) -> FakeBus:
    """A fake ``crewai.events`` module: the bus and the three event classes the adapter listens to."""
    fake = FakeBus()
    module = SimpleNamespace(
        crewai_event_bus=fake,
        ToolUsageFinishedEvent=ToolUsageFinishedEvent,
        ToolUsageErrorEvent=ToolUsageErrorEvent,
        ToolSelectionErrorEvent=ToolSelectionErrorEvent,
    )
    monkeypatch.setitem(sys.modules, "crewai.events", module)
    return fake


type _Script = Callable[[Any], Any]


def _output(raw: str = "ok", **fields: Any) -> Any:
    return SimpleNamespace(raw=raw, **{"pydantic": None, **fields})


class FakeAgent:
    """Shaped like ``crewai.Agent``: a role and ``kickoff_async(messages)``."""

    role = "desk"

    def __init__(self, *scripts: _Script) -> None:
        self.scripts = list(scripts)
        self.kickoffs: list[tuple[list[Any], dict[str, Any]]] = []

    async def kickoff_async(self, messages: list[Any], **kwargs: Any) -> Any:
        self.kickoffs.append((messages, kwargs))
        return self.scripts.pop(0)(messages) if self.scripts else _output()


class FakeCrew:
    """Shaped like ``crewai.Crew``: tasks, agents, lifetime usage totals and ``akickoff(inputs=...)``."""

    def __init__(self, *scripts: _Script) -> None:
        self.tasks: list[Any] = []
        self.agents: list[Any] = []
        self.scripts = list(scripts)
        self.kickoffs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.requests = 0

    def calculate_usage_metrics(self) -> Any:
        return SimpleNamespace(
            prompt_tokens=10 * self.requests,
            completion_tokens=5 * self.requests,
            total_tokens=15 * self.requests,
            successful_requests=self.requests,
        )

    async def akickoff(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        self.kickoffs.append((inputs, kwargs))
        self.requests += 2
        result = self.scripts.pop(0)(inputs) if self.scripts else _output()
        result.token_usage = self.calculate_usage_metrics()
        return result


class OldCrew(FakeCrew):
    """A Crew of a release without ``akickoff``."""

    akickoff = None  # type: ignore[assignment]

    async def kickoff_async(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        return await FakeCrew.akickoff(self, inputs, **kwargs)


class FakeFlow:
    """Shaped like ``crewai.flow.Flow``: ``kickoff_async(inputs=...)``, state that survives and ``usage_metrics``."""

    def __init__(self, usage: Any = None) -> None:
        self.turns = 0
        self.inputs: list[dict[str, Any]] = []
        self.usage_metrics = usage

    async def kickoff_async(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        self.inputs.append(inputs)
        self.turns += 1
        return f"turn {self.turns}"


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


def _summary(turn: AgentTurn) -> list[tuple[Any, ...]]:
    return [(t.tool_name, t.tool_args, t.result, bool(t.tool_call_error)) for t in turn.tools or []]


class _Reply(BaseModel):
    answer: str


# -- tool events --------------------------------------------------------------------


async def test_tool_events_become_executions_in_event_order(bus: FakeBus) -> None:
    def script(messages: Any) -> Any:
        # Handlers run in a thread pool: the events arrive out of order and are sorted by time.
        bus.emit(ToolUsageErrorEvent("refund", {"order_id": "7"}, at=2, error=ValueError("not refundable")))
        bus.emit(ToolUsageFinishedEvent("lookup_order", '{"order_id": "7"}', at=1, output="shipped"))
        # CrewAI's own retries of the failing call: the same error again.
        bus.emit(ToolUsageErrorEvent("refund", {"order_id": "7"}, at=3, error=ValueError("not refundable")))
        bus.emit(ToolUsageErrorEvent("refund", {"order_id": "7"}, at=4, error=ValueError("still not refundable")))
        bus.emit(ToolSelectionErrorEvent("no_such_tool", {}, at=5, error="Action 'no_such_tool' don't exist"))
        bus.emit(ToolUsageFinishedEvent("ping", {}, at=6, output="Tool not found", failure=object()))
        bus.emit(ToolUsageErrorEvent("refund", {"order_id": "7"}, at=7, error=ValueError("not refundable")))
        bus.emit(ToolUsageFinishedEvent("lookup_order", {"order_id": "8"}, at=8, output={"status": "lost"}))
        bus.emit(ToolUsageErrorEvent("refund", {"order_id": "8"}, at=9, error=ValueError("not refundable")))
        return _output("done")

    turn = await _turn(crewai_adapter(FakeAgent(script)), _input([("user", "go")]))

    assert turn.content == "done"
    assert _summary(turn) == [
        ("lookup_order", {"order_id": "7"}, "shipped", False),
        ("refund", {"order_id": "7"}, "not refundable", True),
        ("refund", {"order_id": "7"}, "still not refundable", True),
        ("no_such_tool", {}, "Action 'no_such_tool' don't exist", True),
        ("ping", {}, "Tool not found", True),
        ("refund", {"order_id": "7"}, "not refundable", True),
        ("lookup_order", {"order_id": "8"}, '{"status": "lost"}', False),
        ("refund", {"order_id": "8"}, "not refundable", True),
    ]
    # The bus was flushed and the handlers are gone.
    assert bus.flushes == 1
    assert bus.listening == 0


async def test_events_of_another_context_are_not_this_calls(bus: FakeBus) -> None:
    def script(messages: Any) -> Any:
        elsewhere = contextvars.Context()
        elsewhere.run(bus.emit, ToolUsageFinishedEvent("other_scenario_tool", {}, output="x"))
        bus.emit(ToolUsageFinishedEvent("lookup_order", {}, output="shipped"))
        return _output()

    turn = await _turn(crewai_adapter(FakeAgent(script)), _input([("user", "go")]))
    assert [name for name, *_ in _summary(turn)] == ["lookup_order"]


async def test_a_failing_kickoff_propagates_and_removes_the_handlers(bus: FakeBus) -> None:
    def script(messages: Any) -> Any:
        raise RuntimeError("model is down")

    with pytest.raises(RuntimeError, match="model is down"):
        await crewai_adapter(FakeAgent(script)).call(_input([("user", "hi")]))
    assert bus.listening == 0


async def test_without_crewai_the_package_is_named() -> None:
    with pytest.raises(ImportError, match=r"simulhausen\.crewai_adapter needs the 'crewai' package"):
        await crewai_adapter(FakeAgent()).call(_input([("user", "hi")]))


# -- a single Agent -----------------------------------------------------------------


async def test_agent_session_mode_carries_the_native_messages(bus: FakeBus) -> None:
    native = [
        {"role": "system", "content": "You are desk."},
        {"role": "user", "content": "Current Task: where is it?"},
        {"role": "assistant", "content": "Action: lookup_order\nObservation: shipped"},
        {"role": "user", "content": "Analyze the tool result."},
    ]
    usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "successful_requests": 2}
    agent = FakeAgent(lambda messages: _output("it is shipped", messages=native, usage_metrics=usage))
    adapter = crewai_adapter(agent, response_format=_Reply)
    first = await _turn(adapter, _input([("user", "where is it?")]))
    await _turn(adapter, _input([("user", "where is it?"), ("assistant", "it is shipped"), ("user", "thanks")]))

    assert first.usage == Usage(input_tokens=20, output_tokens=10, total_tokens=30, requests=2)
    assert first.tools is None
    assert agent.kickoffs[0] == ([{"role": "user", "content": "where is it?"}], {"response_format": _Reply})
    # The system prompt is left to the kickoff; the answer the run did not record is appended.
    assert agent.kickoffs[1][0] == [
        *native[1:],
        {"role": "assistant", "content": "it is shipped"},
        {"role": "user", "content": "thanks"},
    ]


async def test_agent_history_that_ends_with_the_answer_is_kept_as_it_is(bus: FakeBus) -> None:
    native = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    agent = FakeAgent(lambda messages: _output("hello", messages=native, usage_metrics=None))
    adapter = crewai_adapter(agent)
    first = await _turn(adapter, _input([("user", "hi")]))
    await _turn(adapter, _input([("user", "hi"), ("assistant", "hello"), ("user", "more")]))

    assert first.usage is None
    assert agent.kickoffs[1][0] == [*native, {"role": "user", "content": "more"}]


async def test_agent_falls_back_to_the_transcript(bus: FakeBus) -> None:
    transcript = [("user", "hi"), ("assistant", "ok"), ("user", "more")]
    plain = [{"role": role, "content": content} for role, content in transcript]

    # Transcript mode never carries native messages.
    agent = FakeAgent(*[lambda messages: _output(messages=[{"role": "user", "content": "native"}])] * 2)
    adapter = crewai_adapter(agent, history="transcript")
    await _turn(adapter, _input(transcript[:1]))
    await _turn(adapter, _input(transcript))
    assert agent.kickoffs[1][0] == plain

    # Neither does session mode after a rollback, or when the kickoff returned no message list.
    agent = FakeAgent(lambda messages: _output(messages=[{"role": "user", "content": "native"}]))
    adapter = crewai_adapter(agent)
    await _turn(adapter, _input([("user", "my code is 7421")]))
    await _turn(adapter, _input([("user", "hello again")]))
    assert agent.kickoffs[1][0] == [{"role": "user", "content": "hello again"}]
    await _turn(adapter, _input([("user", "hello again"), ("assistant", "ok"), ("user", "more")]))
    assert agent.kickoffs[2][0] == [
        {"role": "user", "content": "hello again"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]


async def test_a_structured_reply_is_rendered(bus: FakeBus) -> None:
    parsed = _Reply(answer="all good")
    plain = crewai_adapter(FakeAgent(lambda messages: _output("raw", pydantic=parsed)))
    assert (await _turn(plain, _input([("user", "sum")]))).content == '{"answer":"all good"}'

    custom = crewai_adapter(
        FakeAgent(lambda messages: _output("raw", pydantic=parsed)), content_fn=lambda raw: raw.answer.upper()
    )
    assert (await _turn(custom, _input([("user", "sum")]))).content == "ALL GOOD"


# -- a Crew -------------------------------------------------------------------------


async def test_crew_gets_the_newest_message_and_the_history_and_reports_usage_per_turn(bus: FakeBus) -> None:
    crew = FakeCrew()
    adapter = crewai_adapter(crew, input_files={"a": "b"})
    first = await _turn(adapter, _input([("user", "where is it?")]))
    second = await _turn(adapter, _input([("user", "where is it?"), ("assistant", "ok"), ("user", "thanks")]))

    assert crew.kickoffs == [
        ({"message": "where is it?", "history": ""}, {"input_files": {"a": "b"}}),
        ({"message": "thanks", "history": "User: where is it?\nAssistant: ok"}, {"input_files": {"a": "b"}}),
    ]
    # The crew's totals grow over its lifetime; each turn reports its own share.
    per_turn = Usage(input_tokens=20, output_tokens=10, total_tokens=30, requests=2)
    assert (first.usage, second.usage) == (per_turn, per_turn)


async def test_crew_input_keys_inputs_fn_and_a_release_without_akickoff(bus: FakeBus) -> None:
    crew = FakeCrew()
    await _turn(crewai_adapter(crew, input_key="question", history_key=None), _input([("user", "hi")]))
    assert crew.kickoffs[0][0] == {"question": "hi"}

    crew = OldCrew()
    adapter = crewai_adapter(crew, inputs_fn=lambda agent_input: {"thread": agent_input.thread_id})
    await _turn(adapter, _input([("user", "hi")]))
    assert crew.kickoffs[0][0] == {"thread": "t"}

    # A transcript without a user message still makes a valid kickoff.
    crew = FakeCrew()
    await _turn(crewai_adapter(crew), _input([("assistant", "welcome")], new=0))
    assert crew.kickoffs[0][0] == {"message": "", "history": ""}


# -- a Flow and factories -----------------------------------------------------------


async def test_flow_reply_inputs_and_usage(bus: FakeBus) -> None:
    flow = FakeFlow(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6, successful_requests=1))
    turn = await _turn(crewai_adapter(flow), _input([("user", "hi")]))

    assert turn.content == "turn 1"
    assert flow.inputs == [{"message": "hi", "history": ""}]
    assert turn.usage == Usage(input_tokens=4, output_tokens=2, total_tokens=6, requests=1)
    assert (await _turn(crewai_adapter(FakeFlow()), _input([("user", "hi")]))).usage is None


async def test_a_factory_builds_per_thread_in_session_mode_and_per_call_in_transcript_mode(bus: FakeBus) -> None:
    built: list[FakeFlow] = []

    def factory() -> FakeFlow:
        built.append(FakeFlow())
        return built[-1]

    adapter = crewai_adapter(factory)
    replies = [
        (await _turn(adapter, _input([("user", "a")], thread_id="one"))).content,
        (await _turn(adapter, _input([("user", "b")], thread_id="two"))).content,
        (
            await _turn(adapter, _input([("user", "a"), ("assistant", "turn 1"), ("user", "c")], thread_id="one"))
        ).content,
    ]
    assert replies == ["turn 1", "turn 1", "turn 2"]
    assert len(built) == 2

    built.clear()
    adapter = crewai_adapter(factory, history="transcript")
    await _turn(adapter, _input([("user", "a")]))
    again = await _turn(adapter, _input([("user", "a"), ("assistant", "turn 1"), ("user", "c")]))
    assert again.content == "turn 1"
    assert len(built) == 2


def test_history_is_validated() -> None:
    with pytest.raises(ValueError, match="history must be"):
        crewai_adapter(FakeAgent(), history="nope")  # ty: ignore[invalid-argument-type]
