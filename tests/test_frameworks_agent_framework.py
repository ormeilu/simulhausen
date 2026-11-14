"""Unit tests of ``agent_framework_adapter`` with hand-written fakes shaped like the Agent Framework objects.

The real library is exercised in ``tests/contract/test_agent_framework_contract.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import agent_framework
from simulhausen.frameworks.agent_framework import agent_framework_adapter


@dataclass
class Message:
    role: str
    contents: list[Any]


def call(call_id: str, name: str, arguments: Any = None, **flags: Any) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments, **flags)


def answer(call_id: str, result: Any, exception: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="function_result", call_id=call_id, result=result, exception=exception)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


@dataclass
class Response:
    text: str
    messages: list[Message] = field(default_factory=list)
    usage_details: Any = None


USAGE = {"input_token_count": 5, "output_token_count": 2, "total_token_count": 7}


@pytest.fixture(autouse=True)
def framework(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require(module: str, *, package: str, adapter: str) -> Any:
        assert (module, package, adapter) == ("agent_framework", "agent-framework-core", "agent_framework_adapter")
        return SimpleNamespace(Message=Message)

    monkeypatch.setattr(agent_framework, "require", fake_require)


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _call(adapter: Any, input: AgentInput) -> AgentTurn:
    turn = await adapter.call(input)
    assert isinstance(turn, AgentTurn)
    return turn


def _sent(messages: list[Message]) -> list[str]:
    return [f"{message.role}:{message.contents[0]}" for message in messages]


def test_history_is_validated() -> None:
    with pytest.raises(ValueError, match="history must be"):
        agent_framework_adapter(SimpleNamespace(run=None), history="server")  # ty: ignore[invalid-argument-type]


# -- agents --------------------------------------------------------------------------------------------------


class Agent:
    def __init__(self, *responses: Response) -> None:
        self.responses = list(responses)
        self.runs: list[tuple[list[str], str, dict[str, Any]]] = []

    def create_session(self, *, session_id: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(session_id=session_id)

    async def run(self, messages: list[Message], *, session: Any, **kwargs: Any) -> Response:
        self.runs.append((_sent(messages), session.session_id, kwargs))
        return self.responses.pop(0) if self.responses else Response(f"reply {len(self.runs)}")


async def test_session_mode_keeps_one_session_per_thread_and_sends_the_new_messages() -> None:
    agent = Agent(Response("reply 1", usage_details=USAGE))
    adapter = agent_framework_adapter(agent, options={"temperature": 0})

    first = await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "one"), ("user", "two")], new=2))
    await _call(adapter, _input([("user", "other")], thread_id="u"))

    assert first == AgentTurn(content="reply 1", usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7))
    assert agent.runs == [
        (["user:hi"], "t", {"options": {"temperature": 0}}),
        (["user:one", "user:two"], "t", {"options": {"temperature": 0}}),
        (["user:other"], "u", {"options": {"temperature": 0}}),
    ]


async def test_session_mode_opens_a_new_session_when_the_transcript_diverged() -> None:
    agent = Agent()
    adapter = agent_framework_adapter(agent)
    await _call(adapter, _input([("user", "hi")]))

    await _call(adapter, _input([("user", "hi"), ("assistant", "scripted"), ("user", "more")]))
    await _call(
        adapter,
        _input([("user", "hi"), ("assistant", "scripted"), ("user", "more"), ("assistant", "reply 2"), ("user", "x")]),
    )

    sent, session_id, _ = agent.runs[1]
    assert sent == ["user:hi", "assistant:scripted", "user:more"]
    assert session_id.startswith("t-")
    # The new session is the thread's session from now on.
    assert agent.runs[2][:2] == (["user:x"], session_id)


async def test_transcript_mode_sends_the_transcript_in_a_fresh_session_every_time() -> None:
    agent = Agent()
    adapter = agent_framework_adapter(agent, history="transcript")

    await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    (first, first_session, _), (second, second_session, _) = agent.runs
    assert first == ["user:hi"]
    assert second == ["user:hi", "assistant:reply 1", "user:more"]
    assert first_session.startswith("t-")
    assert second_session.startswith("t-")
    assert first_session != second_session


async def test_factory_builds_one_agent_per_thread() -> None:
    built: list[Agent] = []

    def factory() -> Agent:
        built.append(Agent())
        return built[-1]

    adapter = agent_framework_adapter(factory)
    await _call(adapter, _input([("user", "a1")], thread_id="a"))
    await _call(adapter, _input([("user", "b1")], thread_id="b"))
    await _call(adapter, _input([("user", "a1"), ("assistant", "reply 1"), ("user", "a2")], thread_id="a"))

    assert [[sent for sent, _, _ in agent.runs] for agent in built] == [[["user:a1"], ["user:a2"]], [["user:b1"]]]


async def test_content_fn_gets_the_response() -> None:
    adapter = agent_framework_adapter(Agent(), content_fn=lambda response: response.text.upper())

    assert (await _call(adapter, _input([("user", "hi")]))).content == "REPLY 1"


async def test_tools_are_paired_by_call_id() -> None:
    approval = SimpleNamespace(type="function_approval_request", function_call=call("c3", "refund", {"id": 7}))
    messages: list[Any] = [
        Message("assistant", [text("Looking."), call("c1", "lookup", '{"id": 7}'), call("c2", "cancel", {"id": 7})]),
        Message("tool", [answer("c2", "Error: Function failed.", "locked"), answer("c1", '{"status": "shipped"}')]),
        Message("assistant", [call("c3", "refund", {"id": 7}), approval]),
        Message("assistant", [SimpleNamespace(type="function_approval_request", function_call=call("c4", "wipe"))]),
        Message("assistant", [call("c5", "web_search", hosted=True), call("c6", "noted", informational_only=True)]),
        Message("tool", [answer("c9", "orphan")]),
        SimpleNamespace(role="assistant"),
    ]
    agent = Agent(Response("done", messages=messages, usage_details="not a dict"))

    turn = await _call(agent_framework_adapter(agent), _input([("user", "hi")]))

    assert turn.usage is None
    assert turn.tools is not None
    seen = [(t.tool_call_id, t.tool_name, t.tool_args, t.result, t.tool_call_error, t.is_paused) for t in turn.tools]
    assert seen == [
        ("c1", "lookup", {"id": 7}, '{"status": "shipped"}', None, False),
        ("c2", "cancel", {"id": 7}, "Error: Function failed.", True, False),
        ("c3", "refund", {"id": 7}, None, None, True),
        ("c4", "wipe", {}, None, None, True),
        ("c5", "web_search", {}, None, None, False),
        ("c6", "noted", {}, None, None, False),
        ("c9", None, {}, "orphan", None, False),
    ]


# -- workflows ---------------------------------------------------------------------------------------------


class RunResult(list[Any]):
    def get_outputs(self) -> list[Any]:
        return [event.data for event in self if event.type == "output"]

    def get_request_info_events(self) -> list[Any]:
        return [event for event in self if event.type == "request_info"]


def event(kind: str, data: Any = None, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=kind, data=data, **fields)


class Workflow:
    def __init__(self, *runs: list[Any]) -> None:
        self.script = list(runs)
        self.runs: list[tuple[Any, dict[str, Any]]] = []

    async def run(self, message: Any = None, **kwargs: Any) -> RunResult:
        task = _sent(message) if isinstance(message, list) else message
        if "responses" in kwargs:
            kwargs["responses"] = {key: _sent(value) for key, value in kwargs["responses"].items()}
        self.runs.append((task, kwargs))
        return RunResult(self.script.pop(0) if self.script else [event("output", Response(f"out {len(self.runs)}"))])


async def test_workflow_events_give_the_reply_and_every_participants_tools_and_usage() -> None:
    writer = Response(
        "draft", [Message("assistant", [call("c1", "lookup")]), Message("tool", [answer("c1", "ok")])], USAGE
    )
    editor = Response("polished", [Message("assistant", [text("polished")])], USAGE)
    workflow = Workflow(
        [
            event("executor_invoked", "Write it"),
            event("executor_completed", ["plain value"]),
            event("executor_completed", [SimpleNamespace(agent_response=writer), writer]),
            event("output", editor),
            event("executor_completed", SimpleNamespace(agent_response=editor)),
            event("executor_completed", None),
        ]
    )

    turn = await _call(agent_framework_adapter(workflow, tools=["extra"]), _input([("user", "Write it")]))

    assert turn.content == "polished"
    assert turn.tools is not None
    assert [(t.tool_name, t.result) for t in turn.tools] == [("lookup", "ok")]
    assert turn.usage == Usage(input_tokens=10, output_tokens=4, total_tokens=14)
    assert workflow.runs == [("Write it", {"tools": ["extra"]})]


async def test_workflow_output_shapes_and_content_fn() -> None:
    last = SimpleNamespace(text="from a message list")
    workflow = Workflow([event("output", [SimpleNamespace(text="first"), last])], [event("output", {"n": 1})], [], [])
    adapter = agent_framework_adapter(workflow)

    assert (await _call(adapter, _input([("user", "hi")]))).content == "from a message list"
    assert (await _call(adapter, _input([("user", "hi")], thread_id="u"))).content == '{"n": 1}'
    empty = await _call(adapter, _input([("user", "hi")], thread_id="v"))
    assert (empty.content, empty.tools, empty.usage) == ("", None, None)

    counted = agent_framework_adapter(workflow, content_fn=lambda result: f"{len(result)} events")
    assert (await _call(counted, _input([("user", "hi")], thread_id="w"))).content == "0 events"


async def test_workflow_session_mode_sends_new_messages_and_answers_pending_requests() -> None:
    waiting = [
        event("output", Response("Anything else?")),
        event("request_info", None, request_id="r1", response_type=list),
        event("request_info", None, request_id="r2", response_type=str),
    ]
    workflow = Workflow([event("output", Response("Welcome back"))], waiting)
    adapter = agent_framework_adapter(workflow)

    await _call(adapter, _input([("assistant", "Welcome"), ("user", "hi")]))
    await _call(
        adapter, _input([("assistant", "Welcome"), ("user", "hi"), ("assistant", "Welcome back"), ("user", "a")])
    )
    await _call(
        adapter,
        _input(
            [
                ("assistant", "Welcome"),
                ("user", "hi"),
                ("assistant", "Welcome back"),
                ("user", "a"),
                ("assistant", "Anything else?"),
                ("user", "b"),
            ]
        ),
    )

    assert workflow.runs == [
        # What the script said first goes in with the first message.
        (["assistant:Welcome", "user:hi"], {}),
        ("a", {}),
        # Only the request that takes messages is answered.
        (None, {"responses": {"r1": ["user:b"]}}),
    ]


async def test_workflow_is_rebuilt_from_the_factory_when_the_transcript_diverged() -> None:
    built: list[Workflow] = []

    def factory() -> Workflow:
        built.append(Workflow())
        return built[-1]

    adapter = agent_framework_adapter(factory)
    await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "other")]))
    await _call(adapter, _input([("user", "other"), ("assistant", "out 1"), ("user", "more")]))

    assert [workflow.runs for workflow in built] == [[("hi", {})], [("other", {}), ("more", {})]]

    # A single workflow object cannot be rebuilt: it gets the transcript and keeps what it remembers.
    shared = Workflow()
    adapter = agent_framework_adapter(shared)
    await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "other")]))
    assert shared.runs == [("hi", {}), ("other", {})]


async def test_workflow_transcript_mode_builds_a_workflow_per_call_and_needs_a_factory() -> None:
    built: list[Workflow] = []

    def factory() -> Workflow:
        built.append(Workflow())
        return built[-1]

    adapter = agent_framework_adapter(factory, history="transcript")
    await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "hi"), ("assistant", "out 1"), ("user", "more")]))

    # One workflow to look at the target, then a new one for every call.
    assert [workflow.runs for workflow in built] == [
        [],
        [("hi", {})],
        [(["user:hi", "assistant:out 1", "user:more"], {})],
    ]

    with pytest.raises(ValueError, match="needs a factory for a workflow"):
        await agent_framework_adapter(Workflow(), history="transcript").call(_input([("user", "hi")]))
