"""``camel_adapter`` against hand-written fakes shaped like CAMEL's ``ChatAgent`` and its responses."""

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks.camel import camel_adapter


class _Reply(BaseModel):
    answer: str


class FakeBaseMessage:
    """Shaped like ``camel.messages.BaseMessage`` as far as the adapter builds it."""

    def __init__(self, kind: str, role_name: str, content: str) -> None:
        self.kind = kind
        self.role_name = role_name
        self.content = content

    @classmethod
    def make_user_message(cls, role_name: str, content: str) -> "FakeBaseMessage":
        return cls("user", role_name, content)

    @classmethod
    def make_assistant_message(cls, role_name: str, content: str) -> "FakeBaseMessage":
        return cls("assistant", role_name, content)


@pytest.fixture
def camel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two CAMEL modules the adapter imports when it writes a transcript into the memory."""
    monkeypatch.setitem(sys.modules, "camel.messages", SimpleNamespace(BaseMessage=FakeBaseMessage))
    roles = SimpleNamespace(USER="backend-user", ASSISTANT="backend-assistant")
    monkeypatch.setitem(sys.modules, "camel.types", SimpleNamespace(OpenAIBackendRole=roles))


def _response(content: str | None = "ok", *, parsed: Any = None, **info: Any) -> Any:
    msgs = [] if content is None else [SimpleNamespace(content=content, parsed=parsed)]
    return SimpleNamespace(msgs=msgs, info=info)


class FakeAgent:
    """Shaped like ``ChatAgent``: a memory, ``reset``, ``update_memory`` and ``astep``."""

    def __init__(self, *responses: Any, role_name: str | None = "Desk") -> None:
        self.role_name = role_name
        self.responses = list(responses)
        self.memory: list[tuple[str, str, str]] = []
        self.resets = 0
        self.steps: list[tuple[str, dict[str, Any], list[tuple[str, str, str]]]] = []

    def reset(self) -> None:
        self.resets += 1
        self.memory.clear()

    def update_memory(self, message: FakeBaseMessage, role: str) -> None:
        self.memory.append((role, message.role_name, message.content))

    async def astep(self, message: str, **kwargs: Any) -> Any:
        self.steps.append((message, kwargs, list(self.memory)))
        self.memory.append(("backend-user", "User", message))
        response = self.responses.pop(0) if self.responses else _response()
        self.memory.append(("backend-assistant", "Desk", "reply"))
        return response


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


async def test_reply_tool_calls_and_usage() -> None:
    info = {
        "tool_calls": [
            SimpleNamespace(
                tool_name="lookup_order", args={"order_id": "7"}, result={"status": "shipped"}, tool_call_id="c1"
            ),
            SimpleNamespace(
                tool_name="refund",
                args={"order_id": "7"},
                result="Tool execution failed: Error executing async tool 'refund': not refundable",
                tool_call_id="c2",
            ),
        ],
        "external_tool_call_requests": [SimpleNamespace(tool_name="export", args={"fmt": "csv"}, tool_call_id="c3")],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    agent = FakeAgent(_response("shipped", **info))
    turn = await _turn(camel_adapter(agent, response_format=_Reply), _input([("user", "where is it?")]))

    assert turn.content == "shipped"
    assert turn.usage == Usage(input_tokens=20, output_tokens=8, total_tokens=28)
    tools = turn.tools or []
    assert [(t.tool_name, t.tool_args, t.tool_call_id) for t in tools] == [
        ("lookup_order", {"order_id": "7"}, "c1"),
        ("refund", {"order_id": "7"}, "c2"),
        ("export", {"fmt": "csv"}, "c3"),
    ]
    assert tools[0].result == '{"status": "shipped"}'
    assert [bool(t.tool_call_error) for t in tools] == [False, True, False]
    assert [t.is_paused for t in tools] == [False, False, True]
    # The first turn resets the agent and needs nothing replayed; kwargs reach astep.
    assert agent.resets == 1
    assert agent.steps == [("where is it?", {"response_format": _Reply}, [])]


async def test_session_mode_sends_only_the_new_messages(camel: None) -> None:
    agent = FakeAgent()
    adapter = camel_adapter(agent)
    await _turn(adapter, _input([("user", "one")]))
    follow_up = [("user", "one"), ("assistant", "ok"), ("user", "two"), ("user", "three")]
    turn = await _turn(adapter, _input(follow_up, new=2))

    assert turn.tools is None
    assert turn.usage is None
    assert agent.resets == 1
    prompt, _, memory = agent.steps[1]
    assert prompt == "three"
    # The agent's own memory of the first turn, then the earlier of the two new messages.
    assert memory == [
        ("backend-user", "User", "one"),
        ("backend-assistant", "Desk", "reply"),
        ("backend-user", "User", "two"),
    ]


async def test_session_mode_rebuilds_the_memory_after_a_rollback_or_an_injected_reply(camel: None) -> None:
    agent = FakeAgent(role_name=None)
    adapter = camel_adapter(agent)
    await _turn(adapter, _input([("user", "my code is 7421")]))
    await _turn(adapter, _input([("user", "hello"), ("assistant", "injected"), ("user", "again")]))

    assert agent.resets == 2
    prompt, _, memory = agent.steps[1]
    assert prompt == "again"
    assert memory == [("backend-user", "User", "hello"), ("backend-assistant", "Assistant", "injected")]


async def test_transcript_mode_resets_and_replays_on_every_call(camel: None) -> None:
    agent = FakeAgent()
    adapter = camel_adapter(agent, history="transcript")
    await _turn(adapter, _input([("user", "one")]))
    await _turn(adapter, _input([("user", "one"), ("assistant", "ok"), ("user", "two")]))
    # A transcript that does not end with a user message is replayed whole.
    await _turn(adapter, _input([("user", "one"), ("assistant", "ok")], new=0))

    assert agent.resets == 3
    assert agent.steps[1][0] == "two"
    assert agent.steps[1][2] == [("backend-user", "User", "one"), ("backend-assistant", "Desk", "ok")]
    assert agent.steps[2][0] == ""
    assert agent.steps[2][2] == [("backend-user", "User", "one"), ("backend-assistant", "Desk", "ok")]


async def test_a_factory_builds_one_agent_per_thread_and_a_shared_agent_starts_over(camel: None) -> None:
    built: list[FakeAgent] = []

    def factory() -> FakeAgent:
        built.append(FakeAgent())
        return built[-1]

    adapter = camel_adapter(factory)
    await _turn(adapter, _input([("user", "a")], thread_id="one"))
    await _turn(adapter, _input([("user", "b")], thread_id="two"))
    await _turn(adapter, _input([("user", "a"), ("assistant", "ok"), ("user", "more")], thread_id="one"))
    assert [[prompt for prompt, *_ in agent.steps] for agent in built] == [["a", "more"], ["b"]]
    assert [agent.resets for agent in built] == [1, 1]

    shared = FakeAgent()
    adapter = camel_adapter(shared)
    await _turn(adapter, _input([("user", "a")], thread_id="one"))
    await _turn(adapter, _input([("user", "b")], thread_id="two"))
    # The instance now holds thread "two": thread "one" is rebuilt from its transcript.
    await _turn(adapter, _input([("user", "a"), ("assistant", "ok"), ("user", "more")], thread_id="one"))
    assert shared.resets == 3
    assert shared.steps[2][2] == [("backend-user", "User", "a"), ("backend-assistant", "Desk", "ok")]


async def test_a_streaming_response_is_awaited_and_odd_responses_are_tolerated() -> None:
    async def final() -> Any:
        return _response("streamed")

    assert (await _turn(camel_adapter(FakeAgent(final())), _input([("user", "hi")]))).content == "streamed"

    empty = await _turn(camel_adapter(FakeAgent(SimpleNamespace(msgs=[], info=None))), _input([("user", "hi")]))
    assert (empty.content, empty.tools, empty.usage) == ("", None, None)


async def test_a_parsed_reply_is_rendered() -> None:
    parsed = _Reply(answer="all good")
    plain = camel_adapter(FakeAgent(_response('{"answer": "all good"}', parsed=parsed)))
    assert (await _turn(plain, _input([("user", "sum")]))).content == '{"answer":"all good"}'

    custom = camel_adapter(FakeAgent(_response("raw", parsed=parsed)), content_fn=lambda raw: raw.answer.upper())
    assert (await _turn(custom, _input([("user", "sum")]))).content == "ALL GOOD"


async def test_replaying_without_camel_names_the_package() -> None:
    adapter = camel_adapter(FakeAgent(), history="transcript")
    with pytest.raises(ImportError, match=r"simulhausen\.camel_adapter needs the 'camel-ai' package"):
        await adapter.call(_input([("user", "one"), ("assistant", "ok"), ("user", "two")]))


def test_history_is_validated() -> None:
    with pytest.raises(ValueError, match="history must be"):
        camel_adapter(FakeAgent(), history="nope")  # ty: ignore[invalid-argument-type]
