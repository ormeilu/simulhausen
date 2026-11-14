import threading
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import haystack as module
from simulhausen.frameworks.haystack import haystack_adapter

# -- fakes shaped like haystack.dataclasses ------------------------------------------


class ChatRole(str, Enum):  # noqa: UP042 - the real enum is a str Enum too
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class ToolCallResult:
    result: Any
    origin: ToolCall
    error: bool = False


@dataclass
class TextContent:
    text: str


@dataclass
class ChatMessage:
    role: ChatRole
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_results: list[ToolCallResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_user(cls, text: str) -> "ChatMessage":
        return cls(ChatRole.USER, text)

    @classmethod
    def from_assistant(cls, text: str | None = None, **kwargs: Any) -> "ChatMessage":
        return cls(ChatRole.ASSISTANT, text, **kwargs)

    @classmethod
    def from_tool(cls, result: Any, origin: ToolCall, error: bool = False) -> "ChatMessage":
        return cls(ChatRole.TOOL, tool_call_results=[ToolCallResult(result, origin, error)])


SYSTEM = ChatMessage(ChatRole.SYSTEM, "be nice")


@pytest.fixture(autouse=True)
def framework(monkeypatch: pytest.MonkeyPatch) -> None:
    def require(name: str, *, package: str, adapter: str) -> Any:
        assert (name, package, adapter) == ("haystack.dataclasses", "haystack-ai", "haystack_adapter")
        return SimpleNamespace(ChatMessage=ChatMessage)

    monkeypatch.setattr(module, "require", require)


def _texts(messages: list[ChatMessage]) -> list[tuple[str, str | None]]:
    return [(message.role.value, message.text) for message in messages]


class Agent:
    """The Agent component: returns its system prompt, the messages it got and what it produced."""

    def __init__(self, *produced: list[ChatMessage]) -> None:
        self.produced = list(produced)
        self.seen: list[list[ChatMessage]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def run_async(self, messages: list[ChatMessage], **kwargs: Any) -> dict[str, Any]:
        self.seen.append(list(messages))
        self.kwargs.append(kwargs)
        new = self.produced.pop(0) if self.produced else [ChatMessage.from_assistant(f"seen {len(messages)}")]
        return {"messages": [SYSTEM, *messages, *new], "last_message": new[-1] if new else None}


class SyncAgent:
    """A component without ``run_async``."""

    def __init__(self) -> None:
        self.threads: list[str] = []

    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        self.threads.append(threading.current_thread().name)
        return {"messages": [*messages, ChatMessage.from_assistant("sync")]}


class Pipeline:
    """A pipeline: ``run_async(data, **kwargs)`` returning ``{component: {socket: value}}``."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def add_component(self, name: str, instance: Any) -> None:
        raise AssertionError("not used")

    async def run_async(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append((data, kwargs))
        return self.result


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, messages: list[tuple[str, str]], **kwargs: Any) -> AgentTurn:
    turn = await adapter.call(_input(messages, **kwargs))
    assert isinstance(turn, AgentTurn)
    return turn


# -- the produced messages -> AgentTurn ----------------------------------------------


async def test_reply_tools_and_usage_come_from_the_new_messages() -> None:
    lookup = ToolCall("lookup", {"id": 7}, id="c1")
    refund = ToolCall("refund", {"id": 7})
    approve = ToolCall("approve", {})
    stray = ToolCall("stray", {"x": 1}, id="c9")
    usage = {"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    agent = Agent(
        [
            ChatMessage.from_assistant(tool_calls=[lookup, refund, approve], meta=usage),
            ChatMessage.from_tool("closed", ToolCall("refund", {"id": 7}), error=True),
            ChatMessage.from_tool([TextContent("o"), TextContent("k")], ToolCall("lookup", {"id": 7}, id="c1")),
            ChatMessage.from_tool([1, 2], stray),
            ChatMessage.from_assistant("It is shipped.", meta=usage),
            ChatMessage(ChatRole.ASSISTANT, None, meta={"usage": {}}),
        ]
    )

    turn = await _turn(haystack_adapter(agent, generation_kwargs={"temperature": 0}), [("user", "where?")])

    assert turn.content == "It is shipped."
    assert agent.kwargs == [{"generation_kwargs": {"temperature": 0}}]
    assert turn.tools is not None
    first, second, third, fourth = turn.tools
    assert (first.tool_name, first.tool_args, first.result, first.tool_call_id) == ("lookup", {"id": 7}, "ok", "c1")
    assert not first.tool_call_error
    assert (second.tool_name, second.result, second.tool_call_error) == ("refund", "closed", True)
    # Asked for, never executed.
    assert (third.tool_name, third.is_paused) == ("approve", True)
    # A result whose call is not among the messages is reported from its origin.
    assert (fourth.tool_name, fourth.tool_args, fourth.result, fourth.tool_call_id) == (
        "stray",
        {"x": 1},
        "[1, 2]",
        "c9",
    )
    assert turn.usage == Usage(input_tokens=10, output_tokens=4, total_tokens=14, requests=2)


async def test_calls_without_ids_pair_by_identity_or_content_and_never_across_ids() -> None:
    same = ToolCall("a", {"n": 1})
    agent = Agent(
        [
            ChatMessage.from_assistant(tool_calls=[same, ToolCall("a", {"n": 2}), ToolCall("a", {"n": 3}, id="x")]),
            ChatMessage.from_tool("two", ToolCall("a", {"n": 2})),
            ChatMessage.from_tool("one", same),
            ChatMessage.from_tool("other", ToolCall("a", {"n": 3}, id="y")),
        ]
    )

    turn = await _turn(haystack_adapter(agent), [("user", "go")])

    assert turn.tools is not None
    assert [(tool.tool_args, tool.result, tool.is_paused) for tool in turn.tools] == [
        ({"n": 1}, "one", False),
        ({"n": 2}, "two", False),
        ({"n": 3}, None, True),
        ({"n": 3}, "other", False),
    ]
    # No assistant text: the last message, a tool message, gives its result.
    assert turn.content == "other"


async def test_reply_of_an_empty_run_and_of_a_message_without_text() -> None:
    silent = await _turn(haystack_adapter(Agent([])), [("user", "hi")])
    blank = await _turn(haystack_adapter(Agent([ChatMessage.from_assistant(None)])), [("user", "hi")])

    assert (silent.content, silent.tools, silent.usage) == ("", None, None)
    assert blank.content == ""


async def test_content_fn_gets_the_reply_message() -> None:
    agent = Agent([ChatMessage.from_assistant("hi", meta={"finish_reason": "stop"})])

    turn = await _turn(
        haystack_adapter(agent, content_fn=lambda message: f"{message.text}/{message.meta['finish_reason']}"),
        [("user", "hi")],
    )

    assert turn.content == "hi/stop"


# -- history -------------------------------------------------------------------------


async def test_session_mode_carries_the_native_messages_without_the_system_prompt() -> None:
    call = ToolCall("lookup", {}, id="c1")
    agent = Agent(
        [
            ChatMessage.from_assistant(tool_calls=[call]),
            ChatMessage.from_tool("ok", call),
            ChatMessage.from_assistant("a1"),
        ]
    )
    adapter = haystack_adapter(agent)

    await _turn(adapter, [("user", "q1")])
    second = await _turn(adapter, [("user", "q1"), ("assistant", "a1"), ("user", "q2")])

    assert [message.role.value for message in agent.seen[1]] == ["user", "assistant", "tool", "assistant", "user"]
    assert second.content == "seen 5"
    # Only this turn's calls are reported.
    assert second.tools is None


async def test_session_mode_rebuilds_after_a_rollback_and_keeps_threads_apart() -> None:
    agent = Agent()
    adapter = haystack_adapter(agent)

    await _turn(adapter, [("user", "q1")])
    await _turn(adapter, [("user", "other")])
    await _turn(adapter, [("user", "q1")], thread_id="u")

    assert _texts(agent.seen[1]) == [("user", "other")]
    assert _texts(agent.seen[2]) == [("user", "q1")]


async def test_transcript_mode_rebuilds_from_the_transcript() -> None:
    call = ToolCall("lookup", {}, id="c1")
    agent = Agent(
        [
            ChatMessage.from_assistant(tool_calls=[call]),
            ChatMessage.from_tool("ok", call),
            ChatMessage.from_assistant("a1"),
        ]
    )
    adapter = haystack_adapter(agent, history="transcript")

    await _turn(adapter, [("user", "q1")])
    await _turn(adapter, [("user", "q1"), ("assistant", "a1"), ("user", "q2")])

    assert _texts(agent.seen[1]) == [("user", "q1"), ("assistant", "a1"), ("user", "q2")]


# -- targets -------------------------------------------------------------------------


async def test_a_sync_only_target_runs_in_a_thread_and_a_factory_is_resolved_once() -> None:
    built: list[SyncAgent] = []

    def factory() -> SyncAgent:
        built.append(SyncAgent())
        return built[-1]

    adapter = haystack_adapter(factory)

    first = await _turn(adapter, [("user", "hi")])
    await _turn(adapter, [("user", "hi"), ("assistant", "sync"), ("user", "more")])

    assert first.content == "sync"
    assert len(built) == 1
    assert threading.current_thread().name not in built[0].threads


async def test_a_pipeline_gets_flat_inputs_and_the_first_chat_output_is_read() -> None:
    pipeline = Pipeline(
        {"retriever": {"documents": ["d"]}, "notes": "text", "llm": {"replies": [ChatMessage.from_assistant("hi")]}}
    )
    adapter = haystack_adapter(pipeline, inputs={"retriever": {"top_k": 3}}, include_outputs_from={"llm"})

    turn = await _turn(adapter, [("user", "hello")])
    await _turn(adapter, [("user", "hello"), ("assistant", "hi"), ("user", "more")])

    assert turn.content == "hi"
    data, kwargs = pipeline.calls[0]
    assert kwargs == {"include_outputs_from": {"llm"}}
    assert data["retriever"] == {"top_k": 3}
    assert _texts(data["messages"]) == [("user", "hello")]
    # A generator returns only its replies: the adapter appends them to what it sent.
    assert _texts(pipeline.calls[1][0]["messages"]) == [("user", "hello"), ("assistant", "hi"), ("user", "more")]


async def test_a_pipeline_with_named_components_and_sockets() -> None:
    reply = ChatMessage.from_assistant("from the agent")
    pipeline = Pipeline({"llm": {"replies": [ChatMessage.from_assistant("draft")]}, "agent": {"last_message": reply}})
    adapter = haystack_adapter(
        pipeline,
        input_component="agent",
        input_socket="chat",
        output_component="agent",
        output_socket="last_message",
        inputs={"agent": {"mode": "fast"}},
    )

    turn = await _turn(adapter, [("user", "hello")])

    assert turn.content == "from the agent"
    data, _ = pipeline.calls[0]
    assert data == {"agent": {"mode": "fast", "chat": [ChatMessage.from_user("hello")]}}


async def test_a_pipeline_without_chat_messages_gives_an_empty_reply() -> None:
    missing = await _turn(
        haystack_adapter(Pipeline({"llm": {"replies": []}}), output_component="agent"), [("user", "x")]
    )
    empty = await _turn(haystack_adapter(Pipeline({})), [("user", "x")])

    assert (missing.content, empty.content) == ("", "")


def test_an_unknown_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="history must be"):
        haystack_adapter(Agent(), history="server")  # ty: ignore[invalid-argument-type]
