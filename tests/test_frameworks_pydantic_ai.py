from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import pydantic_ai as module
from simulhausen.frameworks.pydantic_ai import pydantic_ai_adapter

# -- fakes shaped like pydantic_ai.messages ------------------------------------------


@dataclass
class SystemPromptPart:
    content: str
    part_kind: str = "system-prompt"


@dataclass
class UserPromptPart:
    content: str
    part_kind: str = "user-prompt"


@dataclass
class TextPart:
    content: str
    part_kind: str = "text"


@dataclass
class ToolCallPart:
    tool_name: str
    args: Any = None
    tool_call_id: str = "call"
    part_kind: str = "tool-call"


@dataclass
class ToolReturnPart:
    tool_name: str
    content: Any
    tool_call_id: str = "call"
    outcome: str = "success"
    part_kind: str = "tool-return"


@dataclass
class RetryPromptPart:
    content: Any
    tool_name: str | None = None
    tool_call_id: str = "call"
    part_kind: str = "retry-prompt"


@dataclass
class ModelRequest:
    parts: list[Any]
    kind: str = "request"


@dataclass
class ModelResponse:
    parts: list[Any]
    kind: str = "response"


MESSAGES = SimpleNamespace(
    ModelRequest=ModelRequest, ModelResponse=ModelResponse, UserPromptPart=UserPromptPart, TextPart=TextPart
)


@dataclass
class RunUsage:
    """A property value in current releases: callable for backwards compatibility, with the numbers on it."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def __call__(self) -> "RunUsage":
        raise AssertionError("the deprecated call form must not be used when the numbers are there")


@dataclass
class Result:
    output: Any
    history: list[Any]
    new: list[Any]
    usage: Any = None
    _output_tool_name: str | None = None

    def all_messages(self) -> list[Any]:
        return self.history

    def new_messages(self) -> list[Any]:
        return self.new


@dataclass
class FakeAgent:
    """Stand-in for ``pydantic_ai.Agent``: answers from a script and keeps what ``run`` received."""

    outputs: list[Any] = field(default_factory=list)
    new_parts: list[list[Any]] = field(default_factory=list)
    system_prompt: str | None = None
    usage: Any = None
    output_tool: str | None = None
    calls: list[tuple[Any, list[Any] | None, dict[str, Any]]] = field(default_factory=list)

    async def run(self, user_prompt: Any = None, *, message_history: Any = None, **kwargs: Any) -> Result:
        self.calls.append((user_prompt, None if message_history is None else list(message_history), kwargs))
        output = self.outputs.pop(0) if self.outputs else f"reply {len(self.calls)}"
        first: list[Any] = []
        if message_history is None and self.system_prompt:
            first.append(SystemPromptPart(self.system_prompt))
        if user_prompt is not None:
            first.append(UserPromptPart(user_prompt))
        new: list[Any] = [ModelRequest(first)]
        if self.new_parts:
            parts = self.new_parts.pop(0)
            new += [
                ModelResponse([p for p in parts if p.part_kind == "tool-call"]),
                ModelRequest([p for p in parts if p.part_kind != "tool-call"]),
            ]
        new.append(ModelResponse([TextPart(str(output))]))
        return Result(output, [*(message_history or []), *new], new, self.usage, self.output_tool)


@pytest.fixture(autouse=True)
def native(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """The adapter builds histories from the fake message classes instead of importing pydantic_ai."""
    required: list[tuple[str, str, str]] = []

    def require(name: str, *, package: str, adapter: str) -> Any:
        required.append((name, package, adapter))
        return MESSAGES

    monkeypatch.setattr(module, "require", require)
    return required


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


def _shape(history: list[Any] | None) -> list[tuple[str, list[tuple[str, Any]]]]:
    return [(m.kind, [(p.part_kind, p.content) for p in m.parts]) for m in history or []]


# -- history ------------------------------------------------------------------------


async def test_session_mode_carries_the_native_history_between_turns(native: list[tuple[str, str, str]]) -> None:
    agent = FakeAgent()
    adapter = pydantic_ai_adapter(agent, deps="d")

    first = await _turn(adapter, _input([("user", "hi")]))
    second = await _turn(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    assert (first.content, second.content) == ("reply 1", "reply 2")
    assert agent.calls[0] == ("hi", None, {"deps": "d"})
    prompt, history, _ = agent.calls[1]
    assert prompt == "more"
    # The very objects the first run returned, not a rebuilt copy.
    assert _shape(history) == [("request", [("user-prompt", "hi")]), ("response", [("text", "reply 1")])]
    assert native == []
    assert first.tools is None
    assert first.usage is None


async def test_session_mode_joins_several_new_user_messages() -> None:
    agent = FakeAgent()
    adapter = pydantic_ai_adapter(agent)
    await adapter.call(_input([("user", "hi")]))

    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "a"), ("user", "b")], new=2))

    assert agent.calls[1][0] == "a\n\nb"


async def test_session_mode_rebuilds_from_the_transcript_after_a_rollback(native: list[tuple[str, str, str]]) -> None:
    agent = FakeAgent(system_prompt="be brief")
    adapter = pydantic_ai_adapter(agent)
    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "secret")]))

    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "other")]))

    prompt, history, _ = agent.calls[2]
    assert prompt == "other"
    assert _shape(history) == [
        ("request", [("system-prompt", "be brief"), ("user-prompt", "hi")]),
        ("response", [("text", "reply 1")]),
    ]
    assert native == [("pydantic_ai.messages", "pydantic-ai-slim", "pydantic_ai_adapter")]
    # The rebuilt run is remembered again.
    await adapter.call(
        _input([("user", "hi"), ("assistant", "reply 1"), ("user", "other"), ("assistant", "reply 3"), ("user", "go")])
    )
    assert agent.calls[3][0] == "go"
    assert len(agent.calls[3][1] or []) == 4


async def test_threads_do_not_share_memory() -> None:
    agent = FakeAgent()
    adapter = pydantic_ai_adapter(agent)
    await adapter.call(_input([("user", "hi")], thread_id="a"))

    await adapter.call(_input([("user", "hi")], thread_id="b"))

    assert agent.calls[1][:2] == ("hi", None)


async def test_transcript_mode_rebuilds_the_history_on_every_call() -> None:
    agent = FakeAgent(system_prompt="be brief")
    adapter = pydantic_ai_adapter(agent, history="transcript")
    await adapter.call(_input([("user", "hi")]))

    await adapter.call(_input([("user", "hi"), ("assistant", "scripted"), ("user", "more")]))

    assert agent.calls[0][:2] == ("hi", None)
    prompt, history, _ = agent.calls[1]
    assert prompt == "more"
    assert _shape(history) == [
        ("request", [("system-prompt", "be brief"), ("user-prompt", "hi")]),
        ("response", [("text", "scripted")]),
    ]


async def test_a_transcript_that_opens_with_the_agent_gets_the_system_prompt_in_its_own_request() -> None:
    agent = FakeAgent(system_prompt="be brief")
    adapter = pydantic_ai_adapter(agent, history="transcript")
    await adapter.call(_input([("user", "hi")]))

    await adapter.call(_input([("assistant", "welcome"), ("user", "hi")]))

    assert _shape(agent.calls[1][1]) == [
        ("request", [("system-prompt", "be brief")]),
        ("response", [("text", "welcome")]),
    ]


async def test_no_system_prompt_is_known_before_a_run_without_history() -> None:
    agent = FakeAgent(system_prompt="be brief")
    adapter = pydantic_ai_adapter(agent, history="transcript")

    await adapter.call(_input([("assistant", "welcome"), ("user", "hi")]))
    # Nothing ends with a user message: the agent runs on the history alone.
    await adapter.call(_input([("user", "hi"), ("assistant", "welcome")], new=0))

    assert _shape(agent.calls[0][1]) == [("response", [("text", "welcome")])]
    assert agent.calls[1][0] is None
    assert len(agent.calls[1][1] or []) == 2


def test_system_parts_of_an_empty_history() -> None:
    assert module._system_parts([]) == []


# -- reply, tools, usage ------------------------------------------------------------


class _Answer(BaseModel):
    answer: str


async def test_structured_output_is_rendered_and_the_output_tool_is_not_a_tool_call() -> None:
    parts = [
        [
            ToolCallPart("final_result", {"answer": "x"}, "o1"),
            ToolReturnPart("final_result", "Final result processed.", "o1"),
        ],
        [],
    ]
    agent = FakeAgent(outputs=[_Answer(answer="x"), _Answer(answer="y")], new_parts=parts, output_tool="final_result")

    plain = await _turn(pydantic_ai_adapter(agent), _input([("user", "hi")]))
    custom = await _turn(pydantic_ai_adapter(agent, content_fn=lambda out: out.answer), _input([("user", "hi")]))

    assert plain.content == '{"answer":"x"}'
    assert plain.tools is None
    assert custom.content == "y"


async def test_tool_calls_are_paired_with_their_returns() -> None:
    parts = [
        ToolCallPart("lookup", {"id": 1}, "c1"),
        ToolCallPart("boom", '{"x": 2}', "c2"),
        ToolCallPart("again", None, "c3"),
        ToolCallPart("legacy", {}, "c4"),
        ToolCallPart("cut", {}, "c5"),
        ToolCallPart("danger", {"id": 9}, "c6"),
        ToolCallPart("no", {}, "c7"),
        ToolReturnPart("lookup", {"found": 1}, "c1"),
        ToolReturnPart("boom", "kaput", "c2", outcome="failed"),
        ToolReturnPart("again", "try again", "c3", outcome="retried"),
        RetryPromptPart([{"type": "missing"}], tool_name="legacy", tool_call_id="c4"),
        ToolReturnPart("cut", "interrupted", "c5", outcome="interrupted"),
        ToolReturnPart("no", "not allowed", "c7", outcome="denied"),
        # A retry that answers no tool call is feedback on the output, not a tool result.
        RetryPromptPart("bad output"),
    ]
    agent = FakeAgent(new_parts=[parts])

    turn = await _turn(pydantic_ai_adapter(agent), _input([("user", "hi")]))

    tools = turn.tools or []
    assert [(t.tool_name, t.tool_args, t.result, t.tool_call_id) for t in tools] == [
        ("lookup", {"id": 1}, '{"found": 1}', "c1"),
        ("boom", {"x": 2}, "kaput", "c2"),
        ("again", {}, "try again", "c3"),
        ("legacy", {}, '[{"type": "missing"}]', "c4"),
        ("cut", {}, "interrupted", "c5"),
        ("danger", {"id": 9}, None, "c6"),
        ("no", {}, "not allowed", "c7"),
    ]
    assert [bool(t.tool_call_error) for t in tools] == [False, True, True, True, False, False, True]
    assert [t.is_paused for t in tools] == [False, False, False, False, True, True, False]


async def test_usage_is_read_from_the_property_and_from_the_method_of_older_releases() -> None:
    current = FakeAgent(usage=RunUsage(input_tokens=5, output_tokens=2, requests=1))
    older = FakeAgent(usage=lambda: SimpleNamespace(input_tokens=3, output_tokens=1, requests=2))

    assert (await _turn(pydantic_ai_adapter(current), _input([("user", "hi")]))).usage == Usage(5, 2, 7, requests=1)
    assert (await _turn(pydantic_ai_adapter(older), _input([("user", "hi")]))).usage == Usage(3, 1, 4, requests=2)


# -- target -------------------------------------------------------------------------


async def test_a_factory_is_called_once_and_an_agent_is_used_as_it_is() -> None:
    built: list[FakeAgent] = []

    def factory() -> FakeAgent:
        built.append(FakeAgent())
        return built[-1]

    adapter = pydantic_ai_adapter(factory)
    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi")], thread_id="u"))

    assert len(built) == 1
    assert len(built[0].calls) == 2


def test_an_unknown_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="history must be"):
        pydantic_ai_adapter(FakeAgent(), history="server")  # ty: ignore[invalid-argument-type]


async def test_exceptions_of_the_run_propagate() -> None:
    class Broken:
        async def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")

    with pytest.raises(RuntimeError, match="model down"):
        await pydantic_ai_adapter(Broken()).call(_input([("user", "hi")]))
