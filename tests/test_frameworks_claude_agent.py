from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, SimulationBackendError, Usage
from simulhausen.frameworks import claude_agent as module
from simulhausen.frameworks.claude_agent import claude_agent_adapter

# -- fakes shaped like claude_agent_sdk's types (the adapter tells them apart by class name) ----


@dataclass
class ClaudeAgentOptions:
    system_prompt: str | None = None
    resume: str | None = None


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = "sig"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ServerToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool | None = None


@dataclass
class ServerToolResultBlock:
    tool_use_id: str
    content: dict[str, Any]


@dataclass
class AssistantMessage:
    content: list[Any]
    model: str = "claude-fake"
    parent_tool_use_id: str | None = None
    message_id: str | None = None


@dataclass
class UserMessage:
    content: Any
    parent_tool_use_id: str | None = None


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]


@dataclass
class ResultMessage:
    session_id: str = "s-1"
    result: str | None = "reply"
    subtype: str = "success"
    is_error: bool = False
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    structured_output: Any = None
    errors: list[str] | None = None


@dataclass
class FakeSDK:
    """Stand-in for the ``claude_agent_sdk`` module: ``query`` replays scripted message streams."""

    streams: list[list[Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def query(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append(kwargs)
        stream = self.streams.pop(0) if self.streams else [ResultMessage(result=f"reply {len(self.calls)}")]
        for message in stream:
            if isinstance(message, Exception):
                raise message
            yield message


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    fake = FakeSDK()

    def require(name: str, *, package: str, adapter: str) -> Any:
        assert (name, package, adapter) == ("claude_agent_sdk", "claude-agent-sdk", "claude_agent_adapter")
        return fake

    monkeypatch.setattr(module, "require", require)
    return fake


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


# -- history ------------------------------------------------------------------------


async def test_session_mode_resumes_the_sdk_session_of_the_previous_turn(sdk: FakeSDK) -> None:
    options = ClaudeAgentOptions(system_prompt="be brief")
    adapter = claude_agent_adapter(options)

    first = await _turn(adapter, _input([("user", "hi")]))
    second = await _turn(
        adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "a"), ("user", "b")], new=2)
    )

    assert (first.content, second.content) == ("reply 1", "reply 2")
    assert sdk.calls[0] == {"prompt": "hi", "options": options}
    assert sdk.calls[1]["prompt"] == "a\n\nb"
    assert sdk.calls[1]["options"] == ClaudeAgentOptions(system_prompt="be brief", resume="s-1")
    # The user's options object is copied, not changed.
    assert options.resume is None
    assert first.tools is None


async def test_session_mode_starts_a_new_session_after_a_rollback(sdk: FakeSDK) -> None:
    adapter = claude_agent_adapter(ClaudeAgentOptions())
    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "secret")]))

    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "other")]))

    assert sdk.calls[2]["options"].resume is None
    assert "User: hi\nAssistant: reply 1\nUser: other" in sdk.calls[2]["prompt"]


async def test_threads_do_not_share_a_session(sdk: FakeSDK) -> None:
    adapter = claude_agent_adapter(ClaudeAgentOptions())
    await adapter.call(_input([("user", "hi")], thread_id="a"))

    await adapter.call(_input([("user", "hello")], thread_id="b"))

    assert sdk.calls[1]["prompt"] == "hello"
    assert sdk.calls[1]["options"].resume is None


async def test_transcript_mode_never_resumes(sdk: FakeSDK) -> None:
    adapter = claude_agent_adapter(ClaudeAgentOptions(), history="transcript")
    await adapter.call(_input([("user", "hi")]))

    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    assert sdk.calls[1]["options"].resume is None
    assert "User: hi\nAssistant: reply 1\nUser: more" in sdk.calls[1]["prompt"]


async def test_the_session_id_falls_back_to_the_init_message_and_is_forgotten_when_unknown(sdk: FakeSDK) -> None:
    sdk.streams += [
        [
            SystemMessage("status", {}),
            SystemMessage("init", {"session_id": "s-init"}),
            ResultMessage(session_id="", result="reply 1"),
        ],
        [ResultMessage(session_id="", result="reply 2")],
    ]
    adapter = claude_agent_adapter(ClaudeAgentOptions())

    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))
    await adapter.call(
        _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more"), ("assistant", "reply 2"), ("user", "go")])
    )

    assert [call["options"].resume for call in sdk.calls] == [None, "s-init", None]


async def test_the_transport_factory_gets_the_options_of_the_turn(sdk: FakeSDK) -> None:
    seen: list[Any] = []

    def transport(options: Any) -> str:
        seen.append(options)
        return f"transport {len(seen)}"

    adapter = claude_agent_adapter(ClaudeAgentOptions(), transport=transport)
    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    assert [call["transport"] for call in sdk.calls] == ["transport 1", "transport 2"]
    assert [options.resume for options in seen] == [None, "s-1"]


# -- reply --------------------------------------------------------------------------


async def test_the_reply_is_the_structured_output_then_the_result_then_the_last_main_text(sdk: FakeSDK) -> None:
    sdk.streams += [
        [ResultMessage(result="text", structured_output={"answer": "x"})],
        [ResultMessage(result="text", structured_output={"answer": "y"})],
        [
            AssistantMessage([TextBlock("first")]),
            AssistantMessage([ThinkingBlock("hm"), TextBlock("main "), TextBlock("answer")]),
            AssistantMessage([TextBlock("subagent notes")], parent_tool_use_id="toolu_1"),
            ResultMessage(result=None),
        ],
        [UserMessage("replayed prompt"), ResultMessage(result=None)],
    ]
    one = [("user", "hi")]

    structured = await _turn(claude_agent_adapter(ClaudeAgentOptions()), _input(one))
    custom = await _turn(claude_agent_adapter(ClaudeAgentOptions(), content_fn=lambda raw: raw["answer"]), _input(one))
    fallback = await _turn(claude_agent_adapter(ClaudeAgentOptions()), _input(one))
    nothing = await _turn(claude_agent_adapter(ClaudeAgentOptions()), _input(one))

    assert structured.content == '{"answer": "x"}'
    assert custom.content == "y"
    assert fallback.content == "main answer"
    assert nothing.content == ""


# -- tools, usage -------------------------------------------------------------------


async def test_tool_uses_are_paired_with_results_subagents_included(sdk: FakeSDK) -> None:
    sdk.streams.append(
        [
            AssistantMessage(
                [TextBlock("Let me look."), ToolUseBlock("t1", "mcp__shop__lookup", {"id": 1})], message_id="m1"
            ),
            UserMessage([ToolResultBlock("t1", [{"type": "text", "text": "found"}], is_error=False)]),
            AssistantMessage([ToolUseBlock("t2", "Agent", {"subagent_type": "researcher"})], message_id="m2"),
            AssistantMessage([ToolUseBlock("t3", "Grep", {"pattern": "x"})], parent_tool_use_id="t2", message_id="m3"),
            UserMessage([ToolResultBlock("t3", "boom", is_error=True)], parent_tool_use_id="t2"),
            UserMessage([ToolResultBlock("t2", "notes")]),
            AssistantMessage(
                [
                    ServerToolUseBlock("s1", "advisor", {"q": "?"}),
                    ServerToolResultBlock("s1", {"type": "advisor_result", "text": "ok"}),
                    ServerToolUseBlock("s2", "web_search", {"query": "x"}),
                    ToolUseBlock("t4", "Bash", {"command": "rm"}),
                ],
                message_id="m4",
            ),
            ResultMessage(result="done"),
        ]
    )

    turn = await _turn(claude_agent_adapter(ClaudeAgentOptions()), _input([("user", "hi")]))

    tools = turn.tools or []
    assert [(t.tool_name, t.tool_args, t.result, t.tool_call_id) for t in tools] == [
        ("mcp__shop__lookup", {"id": 1}, "found", "t1"),
        ("Agent", {"subagent_type": "researcher"}, "notes", "t2"),
        ("Grep", {"pattern": "x"}, "boom", "t3"),
        ("advisor", {"q": "?"}, '{"type": "advisor_result", "text": "ok"}', "s1"),
        # The API ran the search; the SDK passes no result block for it.
        ("web_search", {"query": "x"}, None, "s2"),
        ("Bash", {"command": "rm"}, None, "t4"),
    ]
    assert [bool(t.tool_call_error) for t in tools] == [False, False, True, False, False, False]
    assert [t.is_paused for t in tools] == [False, False, False, False, False, True]


async def test_usage_counts_cache_tokens_cost_and_model_requests(sdk: FakeSDK) -> None:
    usage = {"input_tokens": 10, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 5, "output_tokens": 4}
    sdk.streams += [
        [
            # One API response split into two messages, one more response, one without an id.
            AssistantMessage([TextBlock("a")], message_id="m1"),
            AssistantMessage([TextBlock("b")], message_id="m1"),
            AssistantMessage([TextBlock("c")], message_id="m2"),
            AssistantMessage([TextBlock("d")]),
            ResultMessage(usage=usage, total_cost_usd=0.25),
        ],
        [ResultMessage(usage={"input_tokens": None, "output_tokens": 2})],
        [ResultMessage()],
    ]
    one = _input([("user", "hi")])

    full = await _turn(claude_agent_adapter(ClaudeAgentOptions()), one)
    partial = await _turn(claude_agent_adapter(ClaudeAgentOptions()), one)
    silent = await _turn(claude_agent_adapter(ClaudeAgentOptions()), one)

    assert full.usage == Usage(input_tokens=16, output_tokens=4, total_tokens=20, requests=3, cost=0.25)
    assert partial.usage == Usage(output_tokens=2)
    assert silent.usage is None


# -- errors -------------------------------------------------------------------------


async def test_a_run_without_a_result_or_with_an_error_result_is_a_backend_error(sdk: FakeSDK) -> None:
    sdk.streams += [
        [AssistantMessage([TextBlock("partial")])],
        [ResultMessage(is_error=True, subtype="error_max_turns", errors=["turn limit reached", "sorry"], result=None)],
        [ResultMessage(is_error=True, subtype="success", result="API Error: 529")],
        [ResultMessage(is_error=True, subtype="error_during_execution", result=None)],
    ]
    adapter = claude_agent_adapter(ClaudeAgentOptions())
    one = _input([("user", "hi")])

    with pytest.raises(SimulationBackendError, match="ended without a result message"):
        await adapter.call(one)
    with pytest.raises(SimulationBackendError, match=r"failed \(error_max_turns\): turn limit reached; sorry"):
        await adapter.call(one)
    with pytest.raises(SimulationBackendError, match=r"failed \(success\): API Error: 529"):
        await adapter.call(one)
    with pytest.raises(SimulationBackendError, match=r"failed \(error_during_execution\): no details"):
        await adapter.call(one)


async def test_exceptions_of_the_sdk_propagate(sdk: FakeSDK) -> None:
    sdk.streams.append([AssistantMessage([TextBlock("partial")]), RuntimeError("Claude Code not found")])

    with pytest.raises(RuntimeError, match="Claude Code not found"):
        await claude_agent_adapter(ClaudeAgentOptions()).call(_input([("user", "hi")]))


# -- target -------------------------------------------------------------------------


async def test_a_factory_is_called_once(sdk: FakeSDK) -> None:
    built: list[ClaudeAgentOptions] = []

    def factory() -> ClaudeAgentOptions:
        built.append(ClaudeAgentOptions(system_prompt="built"))
        return built[-1]

    adapter = claude_agent_adapter(factory)
    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi")], thread_id="u"))

    assert len(built) == 1
    assert [call["options"] for call in sdk.calls] == [built[0], built[0]]


def test_an_unknown_history_is_rejected_and_bare_objects_are_targets() -> None:
    with pytest.raises(ValueError, match="history must be"):
        claude_agent_adapter(ClaudeAgentOptions(), history="server")  # ty: ignore[invalid-argument-type]
    # An options-like object with only one of the two marker attributes is still the target, not a factory.
    marker = SimpleNamespace(system_prompt="x")
    adapter = claude_agent_adapter(marker)
    assert adapter._resolve() is marker  # ty: ignore[unresolved-attribute]
