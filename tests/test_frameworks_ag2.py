from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import ag2 as module
from simulhausen.frameworks.ag2 import ag2_adapter

# -- fakes shaped like ag2 -----------------------------------------------------------


@dataclass
class TextInput:
    content: str


@dataclass
class DataInput:
    data: Any


@dataclass
class ModelRequest:
    parts: list[Any]


@dataclass
class ModelMessage:
    content: str


@dataclass
class ToolCallEvent:
    name: str
    arguments: str = "{}"
    id: str = "call"


@dataclass
class ToolCallsEvent:
    calls: list[ToolCallEvent] = field(default_factory=list)


@dataclass
class ModelResponse:
    message: ModelMessage | None = None
    tool_calls: ToolCallsEvent = field(default_factory=ToolCallsEvent)
    usage: Any = None


@dataclass
class ToolResult:
    parts: Any


@dataclass
class ToolResultEvent:
    parent_id: str
    result: Any
    name: str | None = None


@dataclass
class ToolErrorEvent(ToolResultEvent):
    error: Exception | None = None


@dataclass
class HumanMessage:
    """Has ``parent_id`` like a tool result, but is not one."""

    content: str
    parent_id: str


@dataclass
class UsageEvent:
    usage: Any
    kind: str = "model_call"


class History:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def get_events(self) -> list[Any]:
        return list(self.events)

    async def replace(self, events: list[Any]) -> None:
        self.events = list(events)


class MemoryStream:
    def __init__(self) -> None:
        self.history = History()


@pytest.fixture(autouse=True)
def framework(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "ag2": SimpleNamespace(MemoryStream=MemoryStream),
        "ag2.events": SimpleNamespace(
            ModelRequest=ModelRequest, TextInput=TextInput, ModelResponse=ModelResponse, ModelMessage=ModelMessage
        ),
    }

    def require(name: str, *, package: str, adapter: str) -> Any:
        assert (package, adapter) == ("ag2", "ag2_adapter")
        return modules[name]

    monkeypatch.setattr(module, "require", require)


class Agent:
    """An agent: ``ask`` appends the request and the turn's events to the stream it is given."""

    def __init__(self, *turns: list[Any], compact: bool = False) -> None:
        self.turns = list(turns)
        self.compact = compact
        self.calls: list[tuple[tuple[str, ...], list[Any], dict[str, Any]]] = []
        self.streams: list[Any] = []

    async def ask(self, *messages: str, stream: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, list(stream.history.events), kwargs))
        self.streams.append(stream)
        events = self.turns.pop(0) if self.turns else [ModelResponse(ModelMessage(f"reply {len(self.calls)}"))]
        if self.compact:
            stream.history.events = []
        stream.history.events.extend([ModelRequest([TextInput(text) for text in messages]), *events])
        spoken = [event.message.content for event in events if isinstance(event, ModelResponse) and event.message]
        return SimpleNamespace(body=spoken[-1] if spoken else None)


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, messages: list[tuple[str, str]], **kwargs: Any) -> AgentTurn:
    turn = await adapter.call(_input(messages, **kwargs))
    assert isinstance(turn, AgentTurn)
    return turn


# -- the turn's events -> AgentTurn --------------------------------------------------


async def test_reply_tools_and_usage_come_from_the_events_of_the_turn() -> None:
    usage = SimpleNamespace(prompt_tokens=5.0, completion_tokens=2.0, total_tokens=7.0)
    lookup, refund = ToolCallEvent("lookup", '{"id": 7}', id="c1"), ToolCallEvent("refund", '{"id": 7}', id="c2")
    approve, builtin = ToolCallEvent("approve", "{}", id="c3"), ToolCallEvent("search", "{}", id="c4")
    agent = Agent(
        [
            ModelResponse(tool_calls=ToolCallsEvent([lookup, refund, approve]), usage=usage),
            ToolCallsEvent([lookup, refund]),
            lookup,
            ToolResultEvent("c1", ToolResult([TextInput("o"), DataInput({"k": 1}), DataInput(None)])),
            ToolErrorEvent("c2", ToolResult([TextInput("Traceback...")]), error=RuntimeError("closed")),
            HumanMessage("approve?", parent_id="c3"),
            builtin,
            ToolResultEvent("c4", "plain"),
            UsageEvent(usage),
            UsageEvent(None),
            ModelResponse(ModelMessage("It is shipped."), usage=usage),
            UsageEvent(usage),
        ]
    )

    turn = await _turn(ag2_adapter(agent, variables={"plan": "pro"}), [("user", "where?")])

    assert turn.content == "It is shipped."
    assert agent.calls[0][0] == ("where?",)
    assert agent.calls[0][2] == {"variables": {"plan": "pro"}}
    assert turn.tools is not None
    assert [
        (t.tool_name, t.tool_args, t.result, bool(t.tool_call_error), t.is_paused, t.tool_call_id) for t in turn.tools
    ] == [
        ("lookup", {"id": 7}, 'o{"k": 1}', False, False, "c1"),
        ("refund", {"id": 7}, "closed", True, False, "c2"),
        # Asked for by the model, never executed.
        ("approve", {}, None, False, True, "c3"),
        ("search", {}, "plain", False, False, "c4"),
    ]
    # Only the usage events count; the same numbers on the responses are not added again.
    assert turn.usage == Usage(input_tokens=10, output_tokens=4, total_tokens=14, requests=2)


async def test_an_empty_reply_and_content_fn() -> None:
    silent = await _turn(ag2_adapter(Agent([])), [("user", "hi")])
    custom = await _turn(ag2_adapter(Agent(), content_fn=lambda reply: reply.body.upper()), [("user", "hi")])

    assert (silent.content, silent.tools, silent.usage) == ("", None, None)
    assert custom.content == "REPLY 1"


# -- history -------------------------------------------------------------------------


async def test_session_mode_keeps_one_stream_per_thread() -> None:
    agent = Agent(
        [ToolCallEvent("lookup", "{}", id="c1"), ToolResultEvent("c1", "ok"), ModelResponse(ModelMessage("a1"))]
    )
    adapter = ag2_adapter(agent)

    await _turn(adapter, [("user", "q1")])
    second = await _turn(adapter, [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("user", "q3")], new=2)
    await _turn(adapter, [("user", "x")], thread_id="u")

    assert agent.streams[0] is agent.streams[1]
    assert agent.streams[2] is not agent.streams[0]
    assert agent.calls[1][0] == ("q2", "q3")
    # The stream still holds the first turn's tool call; only this turn's events are reported.
    assert len(agent.calls[1][1]) == 4
    assert second.tools is None
    assert agent.calls[2][1] == []


async def test_a_rollback_and_transcript_mode_replay_the_transcript_on_a_new_stream() -> None:
    agent = Agent()
    adapter = ag2_adapter(agent)
    await _turn(adapter, [("user", "q1")])
    await _turn(adapter, [("user", "other"), ("assistant", "scripted"), ("user", "q2")])

    stateless = Agent()
    transcript = ag2_adapter(stateless, history="transcript")
    await _turn(transcript, [("user", "q1")])
    await _turn(transcript, [("user", "q1"), ("assistant", "reply 1"), ("user", "q2")])

    replayed = [ModelRequest([TextInput("other")]), ModelResponse(ModelMessage("scripted"))]
    assert agent.streams[1] is not agent.streams[0]
    assert agent.calls[1][:2] == (("q2",), replayed)
    assert stateless.streams[1] is not stateless.streams[0]
    assert stateless.calls[1][:2] == (
        ("q2",),
        [ModelRequest([TextInput("q1")]), ModelResponse(ModelMessage("reply 1"))],
    )


async def test_a_transcript_that_ends_with_a_reply_asks_the_last_user_message_again() -> None:
    agent = Agent()

    await _turn(ag2_adapter(agent, history="transcript"), [("user", "q1"), ("assistant", "a1")], new=0)

    assert agent.calls[0][0] == ("q1",)
    assert len(agent.calls[0][1]) == 2


async def test_a_history_that_got_shorter_is_read_whole() -> None:
    agent = Agent(compact=True)
    adapter = ag2_adapter(agent)

    await _turn(adapter, [("user", "q1")])
    agent.turns.append([ToolCallEvent("lookup", "{}", id="c1"), ToolResultEvent("c1", "ok")])
    agent.streams[0].history.events.extend([ModelResponse(), ModelResponse(), ModelResponse()])
    turn = await _turn(adapter, [("user", "q1"), ("assistant", "reply 1"), ("user", "q2")])

    assert turn.tools is not None
    assert [tool.tool_name for tool in turn.tools] == ["lookup"]


async def test_factories_and_agent_classes_are_built_once() -> None:
    built: list[Agent] = []

    def factory() -> Agent:
        built.append(Agent())
        return built[-1]

    adapter = ag2_adapter(factory)
    await _turn(adapter, [("user", "a")])
    await _turn(adapter, [("user", "a"), ("assistant", "reply 1"), ("user", "b")])

    assert len(built) == 1
    assert (await _turn(ag2_adapter(Agent), [("user", "a")])).content == "reply 1"


def test_an_unknown_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="history must be"):
        ag2_adapter(Agent(), history="server")  # ty: ignore[invalid-argument-type]
