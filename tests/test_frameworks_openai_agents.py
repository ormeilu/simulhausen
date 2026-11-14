from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import openai_agents as module
from simulhausen.frameworks.openai_agents import openai_agents_adapter

# -- fakes shaped like the SDK's objects ----------------------------------------------


@dataclass
class FakeAgent:
    name: str = "triage"
    tools: list[Any] = field(default_factory=list)
    handoffs: list[Any] = field(default_factory=list)


@dataclass
class RunConfig:
    tracing_disabled: bool = False
    tool_error_formatter: Any = None


@dataclass
class OldRunConfig:
    """The run config of an SDK release that has no ``tool_error_formatter``."""

    tracing_disabled: bool = False


class InputGuardrailTripwireTriggered(Exception):
    def __init__(self, run_data: Any = None) -> None:
        super().__init__("Guardrail InputGuardrail triggered tripwire")
        self.run_data = run_data


def _item(kind: str, raw: Any, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(type=kind, raw_item=raw, **extra)


def _call(name: str, arguments: Any, call_id: str, kind: str = "tool_call_item") -> SimpleNamespace:
    # Call items hold a Pydantic model of the Responses API, read by attribute.
    raw = SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id, id=f"fc-{call_id}")
    return _item(kind, raw)


def _output(call_id: str, output: Any, kind: str = "tool_call_output_item") -> SimpleNamespace:
    # Output items hold a plain dict with the output as text; the Python value sits on the item.
    raw = {"type": "function_call_output", "call_id": call_id, "output": str(output)}
    return _item(kind, raw, output=output) if kind == "tool_call_output_item" else _item(kind, raw)


def _result(
    output: Any = "reply",
    items: list[Any] | None = None,
    *,
    interruptions: list[Any] | None = None,
    usage: Any = None,
    last_agent: Any = None,
    input_list: list[Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        final_output=output,
        new_items=items or [],
        interruptions=interruptions or [],
        context_wrapper=SimpleNamespace(usage=usage),
        last_agent=last_agent,
        to_input_list=lambda: list(input_list or []),
    )


class FakeRunner:
    """Stand-in for ``agents.Runner``: replays results (or raises) and keeps what ``run`` received."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls: list[tuple[Any, Any, dict[str, Any]]] = []
        self.failing: list[tuple[str, str]] = []
        self.messages: list[Any] = []

    async def run(self, agent: Any, run_input: Any, **kwargs: Any) -> Any:
        self.calls.append((agent, run_input, kwargs))
        config: Any = kwargs.get("run_config")
        formatter = config.get("tool_error_formatter") if isinstance(config, dict) else config.tool_error_formatter
        for name, call_id in self.failing:
            args = SimpleNamespace(kind="tool_exception", tool_name=name, call_id=call_id, default_message="failed")
            self.messages.append(await formatter(args))
        result = self.results.pop(0) if self.results else _result(f"reply {len(self.calls)}")
        if isinstance(result, Exception):
            raise result
        if result.last_agent is None:
            result.last_agent = agent
        if not result.to_input_list():
            sent = run_input if isinstance(run_input, list) else [{"role": "user", "content": run_input}]
            carried = [*sent, {"role": "assistant", "content": result.final_output, "native": True}]
            result.to_input_list = lambda: carried
        return result


class FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.items: list[Any] = []
        self.cleared = 0

    async def clear_session(self) -> None:
        self.cleared += 1
        self.items.clear()

    async def add_items(self, items: list[Any]) -> None:
        self.items.extend(items)


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    """The adapter gets a fake ``agents`` module instead of importing the SDK."""
    fake = FakeRunner()

    def require(name: str, *, package: str, adapter: str) -> Any:
        assert (name, package, adapter) == ("agents", "openai-agents", "openai_agents_adapter")
        return SimpleNamespace(Runner=fake, RunConfig=RunConfig)

    monkeypatch.setattr(module, "require", require)
    return fake


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


def _user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


# -- history ------------------------------------------------------------------------


async def test_session_mode_carries_the_input_list_between_turns(runner: FakeRunner) -> None:
    agent = FakeAgent()
    adapter = openai_agents_adapter(agent, max_turns=3)

    first = await _turn(adapter, _input([("user", "hi")]))
    second = await _turn(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    assert (first.content, second.content) == ("reply 1", "reply 2")
    assert runner.calls[0][:2] == (agent, [_user("hi")])
    assert runner.calls[0][2]["max_turns"] == 3
    # The native items of the first run come back, followed by the new user message.
    assert runner.calls[1][1] == [_user("hi"), {**_assistant("reply 1"), "native": True}, _user("more")]
    assert first.tools is None
    assert first.usage is None


async def test_session_mode_rebuilds_from_the_transcript_after_a_rollback(runner: FakeRunner) -> None:
    adapter = openai_agents_adapter(FakeAgent())
    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "secret")]))

    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "other")]))

    assert runner.calls[2][1] == [_user("hi"), _assistant("reply 1"), _user("other")]


async def test_threads_do_not_share_memory(runner: FakeRunner) -> None:
    adapter = openai_agents_adapter(FakeAgent())
    await adapter.call(_input([("user", "hi")], thread_id="a"))

    await adapter.call(_input([("user", "hello")], thread_id="b"))

    assert runner.calls[1][1] == [_user("hello")]


async def test_transcript_mode_sends_the_transcript_and_remembers_nothing(runner: FakeRunner) -> None:
    triage, billing = FakeAgent(), FakeAgent("billing")
    runner.results.append(_result("reply 1", last_agent=billing))
    adapter = openai_agents_adapter(triage, history="transcript")
    await adapter.call(_input([("user", "hi")]))

    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    assert runner.calls[1][:2] == (triage, [_user("hi"), _assistant("reply 1"), _user("more")])


async def test_the_next_turn_starts_from_the_agent_a_handoff_left_in_charge(runner: FakeRunner) -> None:
    triage, billing = FakeAgent(), FakeAgent("billing")
    runner.results += [_result("reply 1", last_agent=billing), _result("reply 2", last_agent=billing)]
    resumed = openai_agents_adapter(triage)
    fixed = openai_agents_adapter(triage, resume_last_agent=False)
    follow_up = _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")])

    await resumed.call(_input([("user", "hi")]))
    await resumed.call(follow_up)
    await fixed.call(_input([("user", "hi")]))
    await fixed.call(follow_up)

    assert [call[0].name for call in runner.calls] == ["triage", "billing", "triage", "triage"]


async def test_an_sdk_session_gets_only_the_new_user_messages(runner: FakeRunner) -> None:
    sessions: list[FakeSession] = []

    def factory(thread_id: str) -> FakeSession:
        sessions.append(FakeSession(thread_id))
        return sessions[-1]

    adapter = openai_agents_adapter(FakeAgent(), session=factory)

    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi"), ("assistant", "reply 1"), ("user", "a"), ("user", "b")], new=2))
    await adapter.call(_input([("user", "hello")], thread_id="u"))

    assert [(call[1], call[2]["session"].session_id) for call in runner.calls] == [
        ("hi", "t"),
        ("a\n\nb", "t"),
        ("hello", "u"),
    ]
    assert [(session.session_id, session.cleared, session.items) for session in sessions] == [
        ("t", 0, []),
        ("u", 0, []),
    ]


async def test_an_sdk_session_is_refilled_from_the_transcript_after_a_rollback(runner: FakeRunner) -> None:
    session = FakeSession("t")
    adapter = openai_agents_adapter(FakeAgent(), session=lambda thread_id: session)
    await adapter.call(_input([("user", "hi")]))
    session.items.append("what the SDK stored")

    await adapter.call(_input([("user", "hi"), ("assistant", "injected"), ("user", "more")]))
    # Nothing ends with a user message and nothing precedes it: an empty prompt on a cleared session.
    await adapter.call(_input([], new=0))

    assert runner.calls[1][1] == "more"
    assert session.cleared == 2
    assert runner.calls[2][1] == ""
    assert session.items == []


async def test_a_new_sdk_session_is_filled_when_the_thread_opens_with_earlier_messages(runner: FakeRunner) -> None:
    session = FakeSession("t")
    adapter = openai_agents_adapter(FakeAgent(), session=lambda thread_id: session)

    await adapter.call(_input([("assistant", "welcome"), ("user", "hi")]))

    assert (session.cleared, session.items) == (0, [_assistant("welcome")])
    assert runner.calls[0][1] == "hi"


def test_a_session_cannot_be_combined_with_transcript_mode() -> None:
    with pytest.raises(ValueError, match="cannot be combined with history='transcript'"):
        openai_agents_adapter(FakeAgent(), session=FakeSession, history="transcript")
    with pytest.raises(ValueError, match="history must be"):
        openai_agents_adapter(FakeAgent(), history="server")  # ty: ignore[invalid-argument-type]


# -- reply, tools, usage ------------------------------------------------------------


class _Answer(BaseModel):
    answer: str


async def test_structured_output_and_usage(runner: FakeRunner) -> None:
    usage = SimpleNamespace(requests=2, input_tokens=10, output_tokens=4, total_tokens=14)
    runner.results += [_result(_Answer(answer="x"), usage=usage), _result(_Answer(answer="y"))]

    plain = await _turn(openai_agents_adapter(FakeAgent()), _input([("user", "hi")]))
    custom = await _turn(
        openai_agents_adapter(FakeAgent(), content_fn=lambda out: out.answer), _input([("user", "hi")])
    )

    assert plain.content == '{"answer":"x"}'
    assert plain.usage == Usage(10, 4, 14, requests=2)
    assert custom.content == "y"


async def test_calls_are_paired_with_outputs_and_handoffs_are_calls(runner: FakeRunner) -> None:
    items = [
        _call("lookup", '{"id": 1}', "c1"),
        _call("refund", '{"id": 1}', "c2"),
        _output("c1", {"found": 1}),
        _output("c2", "An error occurred while running the tool. Please try again."),
        _item("message_output_item", SimpleNamespace(type="message", id="m1")),
        _call("transfer_to_billing", "{}", "h1", kind="handoff_call_item"),
        _output("h1", '{"assistant": "billing"}', kind="handoff_output_item"),
        _call("invoice", '{"id": 1}', "c3"),
        _output("c3", None),
        _call("transfer_to_nobody", "{}", "h2", kind="handoff_call_item"),
    ]
    runner.results.append(_result("done", items))
    runner.failing.append(("refund", "c2"))

    turn = await _turn(openai_agents_adapter(FakeAgent()), _input([("user", "hi")]))

    tools = turn.tools or []
    assert [(t.tool_name, t.tool_args, t.result, t.tool_call_id) for t in tools] == [
        ("lookup", {"id": 1}, '{"found": 1}', "c1"),
        ("refund", {"id": 1}, "An error occurred while running the tool. Please try again.", "c2"),
        ("transfer_to_billing", {}, '{"assistant": "billing"}', "h1"),
        # The Python value of the output is None: the text of the raw item is used.
        ("invoice", {"id": 1}, "None", "c3"),
        ("transfer_to_nobody", {}, None, "h2"),
    ]
    assert [bool(t.tool_call_error) for t in tools] == [False, True, False, False, False]
    assert [t.is_paused for t in tools] == [False, False, False, False, True]
    # Without a formatter of the user's the SDK keeps its default message.
    assert runner.messages == [None]


async def test_hosted_tools_are_read_from_their_call_items(runner: FakeRunner) -> None:
    search = SimpleNamespace(
        type="web_search_call", id="ws1", status="completed", action=SimpleNamespace(model_dump=lambda: {"query": "q"})
    )
    files = {"type": "file_search_call", "id": "fs1", "status": "completed", "queries": ["a"], "results": [{"t": 1}]}
    code = {"type": "code_interpreter_call", "id": "ci1", "status": "failed", "code": "1/0", "outputs": None}
    mcp_ok = {"type": "mcp_call", "id": "m1", "name": "jira_search", "arguments": '{"q": 1}', "output": "found"}
    mcp_bad = {"type": "mcp_call", "id": "m2", "name": "jira_search", "arguments": "{}", "error": "denied"}
    image = {"type": "image_generation_call", "id": "ig1", "status": "completed", "result": "base64"}
    odd = _item("tool_call_item", {"id": "x1"})
    items = [_item("tool_call_item", raw) for raw in (search, files, code, mcp_ok, mcp_bad, image)]
    runner.results.append(_result("done", [*items, odd]))

    turn = await _turn(openai_agents_adapter(FakeAgent()), _input([("user", "hi")]))

    tools = turn.tools or []
    assert [(t.tool_name, t.tool_args, t.result, t.tool_call_id) for t in tools] == [
        ("web_search", {"action": {"query": "q"}}, None, "ws1"),
        ("file_search", {"queries": ["a"]}, '[{"t": 1}]', "fs1"),
        ("code_interpreter", {"code": "1/0"}, None, "ci1"),
        ("jira_search", {"q": 1}, "found", "m1"),
        ("jira_search", {}, "denied", "m2"),
        ("image_generation", {}, "base64", "ig1"),
        (None, {}, None, "x1"),
    ]
    assert [bool(t.tool_call_error) for t in tools] == [False, False, True, False, True, False, False]
    assert not any(t.is_paused for t in tools)


async def test_calls_waiting_for_approval_did_not_run_and_the_run_is_not_carried_on(runner: FakeRunner) -> None:
    danger = _call("delete_account", '{"user": "u"}', "d1")
    approval = _item("tool_approval_item", danger.raw_item, tool_name="delete_account")
    hosted = _item(
        "mcp_approval_request_item",
        {"type": "mcp_approval_request", "id": "ar1", "name": "jira_delete", "arguments": '{"key": "A-1"}'},
    )
    late = _item("tool_approval_item", {"type": "function_call", "call_id": "d2", "name": "wipe", "arguments": "{}"})
    runner.results.append(_result(None, [danger, approval, hosted], interruptions=[approval, late]))
    adapter = openai_agents_adapter(FakeAgent())

    turn = await _turn(adapter, _input([("user", "delete")]))
    await adapter.call(_input([("user", "delete"), ("assistant", ""), ("user", "more")]))

    tools = turn.tools or []
    assert turn.content == ""
    assert [(t.tool_name, t.tool_args, t.tool_call_id, t.is_paused) for t in tools] == [
        ("delete_account", {"user": "u"}, "d1", True),
        ("jira_delete", {"key": "A-1"}, "ar1", True),
        ("wipe", {}, "d2", True),
    ]
    # Rebuilt from the transcript, not continued from the interrupted run.
    assert runner.calls[1][1] == [_user("delete"), _assistant(""), _user("more")]


# -- the error recorder -------------------------------------------------------------


async def test_the_recorder_keeps_the_users_run_config_and_formatter(runner: FakeRunner) -> None:
    async def formatter(args: Any) -> str:
        return f"{args.tool_name} is down"

    as_object = RunConfig(tracing_disabled=True, tool_error_formatter=lambda args: "sync message")
    runner.failing.append(("refund", "c2"))
    runner.results += [
        _result("a", [_call("refund", "{}", "c2"), _output("c2", "refund is down")]),
        _result("b", [_call("refund", "{}", "c2"), _output("c2", "sync message")]),
    ]

    first = await _turn(
        openai_agents_adapter(FakeAgent(), run_config={"tracing_disabled": True, "tool_error_formatter": formatter}),
        _input([("user", "hi")]),
    )
    second = await _turn(openai_agents_adapter(FakeAgent(), run_config=as_object), _input([("user", "hi")]))

    assert runner.messages == ["refund is down", "sync message"]
    assert [bool(turn.tools and turn.tools[0].tool_call_error) for turn in (first, second)] == [True, True]
    sent_dict, sent_object = runner.calls[0][2]["run_config"], runner.calls[1][2]["run_config"]
    assert sent_dict["tracing_disabled"] is True
    assert isinstance(sent_object, RunConfig)
    assert sent_object.tracing_disabled is True
    # The user's own config object is left as it was.
    assert sent_object is not as_object


def test_an_sdk_without_the_formatter_hook_gets_the_config_unchanged() -> None:
    config = OldRunConfig()

    assert module._recording(config, set(), OldRunConfig) is config
    assert module._recording(None, set(), None) is None


# -- guardrails and errors ----------------------------------------------------------


async def test_a_tripped_guardrail_propagates_by_default(runner: FakeRunner) -> None:
    runner.results.append(InputGuardrailTripwireTriggered())

    with pytest.raises(InputGuardrailTripwireTriggered):
        await openai_agents_adapter(FakeAgent()).call(_input([("user", "hi")]))


async def test_a_tripped_guardrail_becomes_the_reply_when_asked(runner: FakeRunner) -> None:
    usage = SimpleNamespace(requests=1, input_tokens=3, output_tokens=1, total_tokens=4)
    data = SimpleNamespace(
        new_items=[_call("lookup", "{}", "c1"), _output("c1", "ok")], context_wrapper=SimpleNamespace(usage=usage)
    )
    runner.results += [
        _result("reply 1"),
        InputGuardrailTripwireTriggered(data),
        _result("reply 3"),
        InputGuardrailTripwireTriggered(),
    ]
    adapter = openai_agents_adapter(FakeAgent(), guardrail_reply="I cannot help with that.")
    dynamic = openai_agents_adapter(FakeAgent(), guardrail_reply=lambda exc: f"Refused: {exc}")
    blocked = [("user", "hi"), ("assistant", "reply 1"), ("user", "secret")]

    await adapter.call(_input([("user", "hi")]))
    refused = await _turn(adapter, _input(blocked))
    await adapter.call(_input([*blocked, ("assistant", "I cannot help with that."), ("user", "more")]))
    bare = await _turn(dynamic, _input([("user", "hi")]))

    assert refused.content == "I cannot help with that."
    assert [t.tool_name for t in refused.tools or []] == ["lookup"]
    assert refused.usage == Usage(3, 1, 4, requests=1)
    # The aborted run left nothing to continue from: plain transcript messages on the next turn.
    assert runner.calls[2][1] == [
        _user("hi"),
        _assistant("reply 1"),
        _user("secret"),
        _assistant("I cannot help with that."),
        _user("more"),
    ]
    assert bare.content == "Refused: Guardrail InputGuardrail triggered tripwire"
    assert (bare.tools, bare.usage) == (None, None)


async def test_other_exceptions_propagate_even_with_a_guardrail_reply(runner: FakeRunner) -> None:
    runner.results.append(RuntimeError("model down"))

    with pytest.raises(RuntimeError, match="model down"):
        await openai_agents_adapter(FakeAgent(), guardrail_reply="no").call(_input([("user", "hi")]))


# -- target -------------------------------------------------------------------------


async def test_a_factory_is_called_once(runner: FakeRunner) -> None:
    built: list[FakeAgent] = []

    def factory() -> FakeAgent:
        built.append(FakeAgent())
        return built[-1]

    adapter = openai_agents_adapter(factory)
    await adapter.call(_input([("user", "hi")]))
    await adapter.call(_input([("user", "hi")], thread_id="u"))

    assert len(built) == 1
    assert [call[0] for call in runner.calls] == [built[0], built[0]]
