from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import autogen as module
from simulhausen.frameworks.autogen import autogen_adapter

# -- fakes shaped like autogen_agentchat / autogen_core -----------------------------


@dataclass
class TextMessage:
    content: Any
    source: str
    models_usage: Any = None

    def to_model_text(self) -> str:
        return str(self.content)


@dataclass
class HandoffMessage(TextMessage):
    target: str = "other"


@dataclass
class Event:
    """A ``BaseAgentEvent``: content and usage, no ``to_model_text``."""

    content: Any
    source: str = "desk"
    models_usage: Any = None


@dataclass
class FunctionCall:
    id: str
    name: str
    arguments: str


@dataclass
class FunctionExecutionResult:
    call_id: str
    content: str
    name: str = ""
    is_error: bool | None = False


@dataclass
class ModelMessage:
    content: str
    source: str


class UserMessage(ModelMessage):
    pass


class AssistantMessage(ModelMessage):
    pass


class CancellationToken:
    pass


FRAMEWORK = {
    "autogen_agentchat.messages": SimpleNamespace(TextMessage=TextMessage),
    "autogen_core": SimpleNamespace(CancellationToken=CancellationToken),
    "autogen_core.models": SimpleNamespace(UserMessage=UserMessage, AssistantMessage=AssistantMessage),
}


@pytest.fixture(autouse=True)
def framework(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    asked: list[tuple[str, str, str]] = []

    def require(name: str, *, package: str, adapter: str) -> Any:
        asked.append((name, package, adapter))
        return FRAMEWORK[name]

    monkeypatch.setattr(module, "require", require)
    return asked


@dataclass
class Team:
    """A team: runs, resets with ``reset()``, saves and loads a state; no model context."""

    replies: list[Any] = field(default_factory=list)
    log: list[Any] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)

    async def run(self, *, task: Any, **kwargs: Any) -> Any:
        self.log.append(("run", [(message.source, message.content) for message in task], kwargs))
        self.memory.extend(message.content for message in task)
        reply = self.replies.pop(0) if self.replies else [TextMessage(f"seen {len(self.memory)}", "desk")]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(messages=reply)

    async def reset(self) -> None:
        self.log.append("reset")
        self.memory.clear()

    async def save_state(self) -> dict[str, Any]:
        self.log.append("save")
        return {"memory": list(self.memory)}

    async def load_state(self, state: dict[str, Any]) -> None:
        self.log.append("load")
        self.memory[:] = state["memory"]


class Context:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def add_message(self, message: Any) -> None:
        self.messages.append(message)


class Agent:
    """A single agent: resets with ``on_reset(token)`` and has a model context; callable like some targets."""

    name = "desk"

    def __init__(self) -> None:
        self.model_context = Context()
        self.log: list[Any] = []

    def __call__(self) -> Any:
        raise AssertionError("an agent must not be mistaken for a factory")

    async def run(self, *, task: Any, **kwargs: Any) -> Any:
        self.log.append(("run", [(message.source, message.content) for message in task]))
        return SimpleNamespace(messages=[TextMessage("ok", "desk")])

    async def on_reset(self, token: Any) -> None:
        assert isinstance(token, CancellationToken)
        self.log.append("on_reset")
        self.model_context.messages.clear()

    async def save_state(self) -> dict[str, Any]:
        return {}

    async def load_state(self, state: dict[str, Any]) -> None:
        raise AssertionError("not used")


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, messages: list[tuple[str, str]], **kwargs: Any) -> AgentTurn:
    turn = await adapter.call(_input(messages, **kwargs))
    assert isinstance(turn, AgentTurn)
    return turn


def _runs(target: Any) -> list[Any]:
    return [entry[1] for entry in target.log if isinstance(entry, tuple)]


# -- the run's messages -> AgentTurn -------------------------------------------------


async def test_reply_tools_and_usage_come_from_the_run_messages() -> None:
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2)
    team = Team(
        replies=[
            [
                Event(
                    [FunctionCall("c1", "lookup", '{"id": 7}'), FunctionCall("c2", "refund", '{"id": 7}')],
                    models_usage=usage,
                ),
                Event([FunctionExecutionResult("c2", "closed", is_error=True), FunctionExecutionResult("c1", "ok")]),
                Event([FunctionCall("c3", "approve", "{}")], models_usage=usage),
                Event("thinking"),
                TextMessage(["multi", "modal"], "desk"),
                TextMessage("It is shipped.", "desk", models_usage=usage),
                HandoffMessage("Transferred to user.", "desk", target="user"),
            ]
        ]
    )

    turn = await _turn(autogen_adapter(team, cancellation_token="token"), [("user", "where?")])

    assert turn.content == "It is shipped."
    assert team.log == [
        ("run", [("user", "where?")], {"output_task_messages": False, "cancellation_token": "token"}),
    ]
    assert turn.tools is not None
    lookup, refund, approve = turn.tools
    assert (lookup.tool_name, lookup.tool_args, lookup.result, lookup.tool_call_id) == (
        "lookup",
        {"id": 7},
        "ok",
        "c1",
    )
    assert not lookup.tool_call_error
    assert (refund.tool_name, refund.result, refund.tool_call_error) == ("refund", "closed", True)
    # Asked for, never executed.
    assert (approve.tool_name, approve.result, approve.is_paused) == ("approve", None, True)
    assert turn.usage == Usage(input_tokens=15, output_tokens=6, total_tokens=21, requests=3)


async def test_reply_falls_back_to_a_handoff_and_to_nothing() -> None:
    team = Team(replies=[[HandoffMessage("Transferred to user.", "desk", target="user")], [Event("only an event")]])
    adapter = autogen_adapter(team, output_task_messages=True)

    handoff = await _turn(adapter, [("user", "hi")])
    silent = await _turn(adapter, [("user", "hi"), ("assistant", "Transferred to user."), ("user", "more")])

    assert handoff.content == "Transferred to user."
    assert (handoff.tools, handoff.usage) == (None, None)
    assert silent.content == ""
    assert team.log[0][2] == {"output_task_messages": True}


async def test_structured_content_is_rendered_or_goes_through_content_fn() -> None:
    class Answer(BaseModel):
        text: str

    plain = await _turn(autogen_adapter(Team(replies=[[TextMessage(Answer(text="x"), "desk")]])), [("user", "hi")])
    custom = await _turn(
        autogen_adapter(Team(replies=[[TextMessage(Answer(text="x"), "desk")]]), content_fn=lambda raw: raw.text),
        [("user", "hi")],
    )

    assert plain.content == '{"text":"x"}'
    assert custom.content == "x"


# -- session mode --------------------------------------------------------------------


async def test_session_mode_sends_only_the_new_messages() -> None:
    team = Team()
    adapter = autogen_adapter(team)

    first = await _turn(adapter, [("user", "a"), ("user", "b")], new=2)
    second = await _turn(adapter, [("user", "a"), ("user", "b"), ("assistant", first.content), ("user", "c")])

    assert _runs(team) == [[("user", "a"), ("user", "b")], [("user", "c")]]
    assert second.content == "seen 3"
    assert "reset" not in team.log


async def test_session_mode_replays_the_transcript_after_a_rollback() -> None:
    team = Team()
    adapter = autogen_adapter(team)

    first = await _turn(adapter, [("user", "a")])
    await _turn(adapter, [("user", "a"), ("assistant", first.content), ("user", "secret")])
    # The secret exchange was rolled back.
    after = await _turn(adapter, [("user", "a"), ("assistant", first.content), ("user", "c")])

    assert team.log[-2] == "reset"
    assert _runs(team)[-1] == [("user", "a"), ("assistant", "seen 1"), ("user", "c")]
    assert after.content == "seen 3"


async def test_a_failed_run_makes_the_next_call_replay() -> None:
    team = Team(replies=[[TextMessage("one", "desk")], RuntimeError("down")])
    adapter = autogen_adapter(team)
    follow_up = [("user", "a"), ("assistant", "one"), ("user", "b")]

    await _turn(adapter, [("user", "a")])
    with pytest.raises(RuntimeError, match="down"):
        await _turn(adapter, follow_up)
    await _turn(adapter, follow_up)

    assert team.log[-2] == "reset"
    assert _runs(team)[-1] == [("user", "a"), ("assistant", "one"), ("user", "b")]


# -- transcript mode -----------------------------------------------------------------


async def test_transcript_mode_resets_and_replays_on_every_call() -> None:
    team = Team()
    adapter = autogen_adapter(team, history="transcript")

    first = await _turn(adapter, [("user", "a")])
    await _turn(adapter, [("user", "a"), ("assistant", first.content), ("user", "b")])

    # The first call found the target untouched; the second reset it first.
    assert [entry if isinstance(entry, str) else "run" for entry in team.log] == ["run", "reset", "run"]
    assert _runs(team)[-1] == [("user", "a"), ("assistant", "seen 1"), ("user", "b")]


async def test_an_agent_gets_earlier_exchanges_in_its_model_context() -> None:
    agent = Agent()
    adapter = autogen_adapter(agent, history="transcript")

    await _turn(adapter, [("user", "a")])
    await _turn(adapter, [("user", "a"), ("assistant", "ok"), ("user", "b"), ("user", "c")], new=2)

    assert agent.log == [("run", [("user", "a")]), "on_reset", ("run", [("user", "b"), ("user", "c")])]
    assert agent.model_context.messages == [UserMessage("a", "user"), AssistantMessage("ok", "desk")]


async def test_a_transcript_that_ends_with_a_reply_is_sent_whole() -> None:
    agent = Agent()

    await _turn(autogen_adapter(agent, history="transcript"), [("user", "a"), ("assistant", "ok")], new=0)

    assert agent.log == [("run", [("user", "a"), ("assistant", "ok")])]
    assert agent.model_context.messages == []


# -- one target per thread -----------------------------------------------------------


async def test_a_factory_builds_one_target_per_thread() -> None:
    built: list[Team] = []

    def factory() -> Team:
        built.append(Team())
        return built[-1]

    adapter = autogen_adapter(factory)

    await _turn(adapter, [("user", "a")], thread_id="one")
    await _turn(adapter, [("user", "a"), ("assistant", "seen 1"), ("user", "b")], thread_id="one")
    other = await _turn(adapter, [("user", "x")], thread_id="two")

    assert len(built) == 2
    assert other.content == "seen 1"
    assert [team.log.count("save") + team.log.count("reset") for team in built] == [0, 0]


async def test_a_shared_target_switches_between_threads() -> None:
    team = Team()
    adapter = autogen_adapter(team)

    await _turn(adapter, [("user", "a")], thread_id="one")
    new_thread = await _turn(adapter, [("user", "x")], thread_id="two")
    back = await _turn(adapter, [("user", "a"), ("assistant", "seen 1"), ("user", "b")], thread_id="one")

    # "two" started from a reset target; "one" came back through load_state with its memory.
    assert new_thread.content == "seen 1"
    assert back.content == "seen 2"
    assert [entry if isinstance(entry, str) else "run" for entry in team.log] == [
        "run",
        "save",
        "reset",
        "run",
        "save",
        "load",
        "run",
    ]


async def test_a_restored_thread_that_was_rolled_back_is_reset() -> None:
    team = Team()
    adapter = autogen_adapter(team)

    await _turn(adapter, [("user", "a")], thread_id="one")
    await _turn(adapter, [("user", "x")], thread_id="two")
    after = await _turn(adapter, [("user", "other")], thread_id="one")

    assert team.log[-3:-1] == ["load", "reset"]
    assert after.content == "seen 1"


async def test_targets_are_told_from_factories_and_plain_objects_are_used_as_they_are(
    framework: list[tuple[str, str, str]],
) -> None:
    agent = Agent()
    await _turn(autogen_adapter(agent), [("user", "a")])
    assert agent.log == [("run", [("user", "a")])]
    assert framework == [("autogen_agentchat.messages", "autogen-agentchat", "autogen_adapter")]

    with pytest.raises(AttributeError, match="run"):
        await _turn(autogen_adapter(object()), [("user", "a")])


def test_an_unknown_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="history must be"):
        autogen_adapter(Team(), history="server")  # ty: ignore[invalid-argument-type]
