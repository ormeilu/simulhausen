from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import llamaindex as module
from simulhausen.frameworks.llamaindex import llamaindex_adapter

# -- fakes shaped like llama_index.core.agent.workflow and its workflow handler ---------


@dataclass
class ChatMessage:
    role: str
    content: str | None


@dataclass
class ToolOutput:
    content: str = ""
    raw_output: Any = None
    is_error: bool = False


@dataclass
class ToolCall:
    tool_name: str
    tool_kwargs: dict[str, Any]
    tool_id: str


@dataclass
class ToolCallResult(ToolCall):
    tool_output: ToolOutput = field(default_factory=ToolOutput)


@dataclass
class AgentStream:
    delta: str
    response: str
    raw: Any = None


@dataclass
class AgentOutput:
    response: ChatMessage
    raw: Any = None
    structured_response: dict[str, Any] | None = None


@dataclass
class InputRequiredEvent:
    prefix: str


class ConfirmDelete(InputRequiredEvent):
    pass


@dataclass
class HumanResponseEvent:
    response: str


class Memory:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages

    async def aget_all(self) -> list[Any]:
        return self.messages


class Store:
    def __init__(self, memory: Memory | None) -> None:
        self.memory = memory

    async def get(self, key: str, default: Any = None) -> Any:
        assert key == "memory"
        return self.memory if self.memory is not None else default


class Handler:
    """A workflow handler: every ``stream_events()`` plays the next batch, awaiting it gives the result."""

    def __init__(self, batches: list[list[Any]], result: Any, memory: list[Any] | None = None) -> None:
        self.batches = list(batches)
        self.result = result
        self.sent: list[Any] = []
        self.ctx = SimpleNamespace(
            store=Store(Memory(memory) if memory is not None else None), send_event=self.sent.append
        )
        self.cancelled = 0

    async def stream_events(self) -> Any:
        for event in self.batches.pop(0):
            yield event

    def __await__(self) -> Any:
        async def result() -> Any:
            return self.result

        return result().__await__()

    async def cancel_run(self) -> None:
        self.cancelled += 1


class Agent:
    def __init__(self, *handlers: Any) -> None:
        self.handlers = list(handlers)
        self.runs: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> Any:
        self.runs.append(kwargs)
        return self.handlers.pop(0)


def _output(text: str | None, raw: Any = None) -> AgentOutput:
    return AgentOutput(response=ChatMessage("assistant", text), raw=raw)


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


FIRST = [("user", "hi")]
SECOND = [("user", "hi"), ("assistant", "hello"), ("user", "more")]


def test_an_unknown_history_is_rejected() -> None:
    assert module.__all__ == ["llamaindex_adapter"]
    with pytest.raises(ValueError, match="history must be"):
        llamaindex_adapter(Agent(), history="nope")  # ty: ignore[invalid-argument-type]


async def test_events_become_tools_and_usage_and_the_memory_is_carried_to_the_next_run() -> None:
    native = [ChatMessage("user", "hi"), ChatMessage("assistant", None), ChatMessage("tool", "found")]
    events = [
        AgentStream(delta="", response="", raw={"usage": {"prompt_tokens": 99}}),
        _output(None, raw={"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}),
        ToolCall("lookup", {"id": 1}, "c1"),
        ToolCall("refund", {"id": 1}, "c2"),
        ToolCallResult("refund", {"id": 1}, "c2", ToolOutput("closed", is_error=True)),
        ToolCallResult("lookup", {"id": 1}, "c1", ToolOutput("found")),
        ToolCallResult("handoff", {"to_agent": "b"}, "", ToolOutput("", raw_output={"moved": True})),
        _output(None, raw=SimpleNamespace(usage_metadata={"prompt_token_count": 3, "candidates_token_count": 1})),
        _output("hello", raw=None),
        _output("hello", raw={"id": "no usage here"}),
        SimpleNamespace(result="a stop event"),
    ]
    agent = Agent(Handler([events], _output("hello"), native), Handler([[]], _output("bye"), []))
    built: list[int] = []

    def factory() -> Agent:
        built.append(1)
        return agent

    adapter = llamaindex_adapter(factory, max_iterations=7)

    first = await _turn(adapter, _input(FIRST))
    second = await _turn(adapter, _input(SECOND))

    assert first.content == "hello"
    assert [(t.tool_name, t.tool_args, t.result, t.tool_call_id) for t in first.tools or []] == [
        ("lookup", {"id": 1}, "found", "c1"),
        ("refund", {"id": 1}, "closed", "c2"),
        ("handoff", {"to_agent": "b"}, '{"moved": true}', None),
    ]
    assert [bool(t.tool_call_error) for t in first.tools or []] == [False, True, False]
    assert first.usage == Usage(input_tokens=8, output_tokens=3, total_tokens=11, requests=2)
    assert (second.content, second.tools, second.usage) == ("bye", None, None)
    assert built == [1]
    assert agent.runs == [
        {"user_msg": "hi", "chat_history": None, "max_iterations": 7},
        {"user_msg": "more", "chat_history": native, "max_iterations": 7},
    ]


async def test_the_history_is_rebuilt_from_the_transcript_when_it_no_longer_matches() -> None:
    native = [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")]
    agent = Agent(*(Handler([[]], _output("hello"), native) for _ in range(4)))
    adapter = llamaindex_adapter(agent)

    await adapter.call(_input(FIRST))
    await adapter.call(_input([("user", "hi"), ("assistant", "hello"), ("user", "a"), ("user", "b")], new=2))
    await adapter.call(_input([("user", "other")]))
    await adapter.call(_input([("user", "hi"), ("assistant", "scripted")], new=0, thread_id="u"))

    assert agent.runs[1] == {"user_msg": "b", "chat_history": [*native, {"role": "user", "content": "a"}]}
    # A rollback: nothing of the old exchange is sent.
    assert agent.runs[2] == {"user_msg": "other", "chat_history": None}
    # No trailing user message: the transcript is the history and the framework picks the question itself.
    assert agent.runs[3] == {
        "user_msg": None,
        "chat_history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "scripted"}],
    }


async def test_transcript_mode_remembers_nothing() -> None:
    native = [ChatMessage("user", "hi"), ChatMessage("tool", "found"), ChatMessage("assistant", "hello")]
    agent = Agent(Handler([[]], _output("hello"), native), Handler([[]], _output("bye"), native))
    adapter = llamaindex_adapter(agent, history="transcript")

    await adapter.call(_input(FIRST))
    await adapter.call(_input(SECOND))

    assert agent.runs[1] == {
        "user_msg": "more",
        "chat_history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    }


async def test_a_run_without_a_memory_is_rebuilt_from_the_transcript_next_time() -> None:
    bare = Handler([[]], _output("hello"))
    del bare.ctx
    agent = Agent(Handler([[]], _output("hello")), Handler([[]], _output("again")), bare, Handler([[]], "text"))
    adapter = llamaindex_adapter(agent)

    await adapter.call(_input(FIRST))
    await adapter.call(_input(SECOND))
    await adapter.call(_input(FIRST, thread_id="u"))
    plain = await _turn(adapter, _input(FIRST, thread_id="v"))

    assert agent.runs[1]["chat_history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    # A workflow whose result is not an AgentOutput.
    assert plain.content == "text"


async def test_content_fn_renders_the_result() -> None:
    result = AgentOutput(response=ChatMessage("assistant", "raw"), structured_response={"answer": "42"})
    adapter = llamaindex_adapter(
        Agent(Handler([[]], result)), content_fn=lambda output: output.structured_response["answer"]
    )

    assert (await _turn(adapter, _input(FIRST))).content == "42"


async def test_a_waiting_run_pauses_the_turn_and_the_next_message_resumes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    required: list[tuple[str, str, str]] = []

    def fake_require(name: str, *, package: str, adapter: str) -> Any:
        required.append((name, package, adapter))
        return SimpleNamespace(HumanResponseEvent=HumanResponseEvent)

    monkeypatch.setattr(module, "require", fake_require)
    usage = {"usage": {"input_tokens": 4, "output_tokens": 1}}
    before = [_output(None, raw=usage), ToolCall("delete", {"user": "u"}, "c1"), ConfirmDelete("Really?"), "never read"]
    after = [
        ToolCall("delete", {"user": "u"}, "c1"),
        ToolCallResult("delete", {"user": "u"}, "c1", ToolOutput("deleted")),
        _output("done", raw=usage),
    ]
    handler = Handler([before, after], _output("done"), [ChatMessage("user", "delete me")])
    agent = Agent(handler)
    adapter = llamaindex_adapter(agent)

    asked = await _turn(adapter, _input([("user", "delete me")]))
    resumed = await _turn(adapter, _input([("user", "delete me"), ("assistant", "Really?"), ("user", "yes")]))

    assert asked.content == "Really?"
    assert [(t.tool_name, t.tool_args, t.is_paused) for t in asked.tools or []] == [("delete", {"user": "u"}, True)]
    assert asked.usage == Usage(input_tokens=4, output_tokens=1, total_tokens=5, requests=1)
    assert handler.sent == [HumanResponseEvent(response="yes")]
    assert required == [("llama_index.core.workflow", "llama-index-core", "llamaindex_adapter")]
    assert len(agent.runs) == 1
    assert resumed.content == "done"
    assert [(t.tool_name, t.result, t.is_paused) for t in resumed.tools or []] == [("delete", "deleted", False)]
    assert handler.cancelled == 0


async def test_resume_fn_builds_the_event_that_answers() -> None:
    handler = Handler([[InputRequiredEvent("Wipe?")], []], _output("wiped"), [])
    adapter = llamaindex_adapter(Agent(handler), resume_fn=lambda agent_input: ("answer", agent_input.thread_id))

    asked = await _turn(adapter, _input([("user", "wipe")]))
    await adapter.call(_input([("user", "wipe"), ("assistant", "Wipe?"), ("user", "yes")]))

    assert (asked.content, asked.tools) == ("Wipe?", None)
    assert handler.sent == [("answer", "t")]


async def test_a_waiting_run_is_cancelled_when_it_cannot_be_resumed() -> None:
    in_transcript_mode = Handler([[InputRequiredEvent("Sure?")]], None)
    adapter = llamaindex_adapter(Agent(in_transcript_mode), history="transcript")
    assert (await _turn(adapter, _input(FIRST))).content == "Sure?"
    assert in_transcript_mode.cancelled == 1

    rolled_back = Handler([[InputRequiredEvent("Sure?")]], None)
    agent = Agent(rolled_back, Handler([[]], _output("fresh"), []))
    adapter = llamaindex_adapter(agent)
    await adapter.call(_input(FIRST))
    fresh = await _turn(adapter, _input([("user", "something else")]))

    assert rolled_back.cancelled == 1
    assert rolled_back.sent == []
    assert fresh.content == "fresh"
    assert agent.runs[1] == {"user_msg": "something else", "chat_history": None}
