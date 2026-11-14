"""Unit tests of ``google_adk_adapter`` with hand-written fakes shaped like the ADK objects.

The real library is exercised in ``tests/contract/test_google_adk_contract.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import google_adk
from simulhausen.frameworks.google_adk import google_adk_adapter


@dataclass
class Part:
    text: str | None = None
    function_call: Any = None
    function_response: Any = None
    thought: bool | None = None


@dataclass
class Content:
    role: str
    parts: list[Part]


@dataclass
class Event:
    author: str = "agent"
    invocation_id: str = "inv"
    content: Content | None = None
    partial: bool | None = None
    usage_metadata: Any = None
    long_running_tool_ids: set[str] | None = None
    output: Any = None

    def is_final_response(self) -> bool:
        parts = self.content.parts if self.content else []
        return not any(part.function_call or part.function_response for part in parts)


def say(text: str, **kwargs: Any) -> Event:
    return Event(content=Content("model", [Part(text=text)]), **kwargs)


def call(name: str, call_id: str | None, args: Any = None, **kwargs: Any) -> Event:
    part = Part(function_call=SimpleNamespace(id=call_id, name=name, args=args))
    return Event(content=Content("model", [part]), **kwargs)


def answer(name: str, call_id: str | None, response: Any) -> Event:
    part = Part(function_response=SimpleNamespace(id=call_id, name=name, response=response))
    return Event(content=Content("user", [part]))


class SessionService:
    def __init__(self) -> None:
        self.sessions: dict[tuple[str, str, str], SimpleNamespace] = {}
        self.deleted: list[str] = []

    async def create_session(self, *, app_name: str, user_id: str, state: Any, session_id: str) -> SimpleNamespace:
        session = SimpleNamespace(id=session_id, state=state, events=[])
        self.sessions[app_name, user_id, session_id] = session
        return session

    async def get_session(self, *, app_name: str, user_id: str, session_id: str) -> SimpleNamespace | None:
        return self.sessions.get((app_name, user_id, session_id))

    async def append_event(self, session: SimpleNamespace, event: Event) -> Event:
        session.events.append(event)
        return event

    async def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        self.deleted.append(session_id)
        del self.sessions[app_name, user_id, session_id]


class PluginManager:
    def __init__(self) -> None:
        self.plugins: list[Any] = []

    def get_plugin(self, name: str) -> Any:
        return next((plugin for plugin in self.plugins if plugin.name == name), None)

    def register_plugin(self, plugin: Any) -> None:
        self.plugins.append(plugin)


class BaseAgent:
    def __init__(self, name: str = "support") -> None:
        self.name = name
        self.root_agent = self

    async def run_async(self, parent_context: Any) -> None:
        raise AssertionError("the adapter runs agents through a Runner")


@dataclass
class Runner:
    session_service: Any
    app: Any = None
    agent: Any = None
    node: Any = None
    app_name: str | None = None
    plugin_manager: Any = field(default_factory=PluginManager)
    script: list[Any] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.app is not None:
            self.agent = self.app.root_agent
            self.app_name = self.app_name or self.app.name
        elif self.node is not None:
            self.agent = self.node
            self.app_name = self.app_name or self.node.name

    async def run_async(self, *, user_id: str, session_id: str, new_message: Content, **kwargs: Any) -> Any:
        session = await self.session_service.get_session(app_name=self.app_name, user_id=user_id, session_id=session_id)
        assert session is not None, "the adapter creates the session before the run"
        self.runs.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "text": new_message.parts[0].text,
                "role": new_message.role,
                "history": [(event.author, event.content.parts[0].text) for event in session.events],
                "state": session.state,
                "kwargs": kwargs,
            }
        )
        session.events.append(Event(author="user", content=new_message))
        items: list[Any] = self.script.pop(0) if self.script else [say("ok")]
        for item in items:
            if isinstance(item, Exception):
                raise item
            if callable(item):
                await item(self.plugin_manager.plugins[0])
                continue
            if isinstance(item, Event) and item.is_final_response() and not item.partial:
                session.events.append(Event(author=self.agent.name, content=item.content))
            yield item


class BasePlugin:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture(autouse=True)
def adk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake ADK modules behind ``require``; the default session service of the fake is exposed for assertions."""
    default_service = SessionService()
    modules = {
        "google.adk.runners": SimpleNamespace(Runner=Runner),
        "google.adk.sessions": SimpleNamespace(InMemorySessionService=lambda: default_service),
        "google.adk.agents": SimpleNamespace(BaseAgent=BaseAgent),
        "google.adk.events": SimpleNamespace(Event=Event),
        "google.adk.plugins.base_plugin": SimpleNamespace(BasePlugin=BasePlugin),
        "google.genai.types": SimpleNamespace(Content=Content, Part=Part),
    }

    def fake_require(module: str, *, package: str, adapter: str) -> Any:
        assert (package, adapter) == ("google-adk", "google_adk_adapter")
        return modules[module]

    monkeypatch.setattr(google_adk, "require", fake_require)
    return {"service": default_service}


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


def _runner(*turns: list[Any], service: SessionService | None = None) -> Runner:
    return Runner(session_service=service or SessionService(), agent=BaseAgent(), app_name="shop", script=list(turns))


async def _call(adapter: Any, input: AgentInput) -> AgentTurn:
    turn = await adapter.call(input)
    assert isinstance(turn, AgentTurn)
    return turn


# -- building the runner ------------------------------------------------------------------------------


async def test_agent_is_wrapped_in_a_runner_with_an_in_memory_session_service(adk: dict[str, Any]) -> None:
    built: list[BaseAgent] = []

    def factory() -> BaseAgent:
        built.append(BaseAgent())
        return built[-1]

    adapter: Any = google_adk_adapter(factory, state={"plan": "pro"}, run_config="cfg")

    first = await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "hi"), ("assistant", "ok"), ("user", "more")]))

    assert first == AgentTurn(content="ok")
    assert len(built) == 1
    runner = adapter._runner
    assert (runner.agent, runner.app_name, runner.session_service) == (built[0], "simulhausen", adk["service"])
    assert [plugin.name for plugin in runner.plugin_manager.plugins] == ["simulhausen_recorder"]
    assert runner.runs == [
        {
            "user_id": "simulhausen",
            "session_id": "t",
            "text": "hi",
            "role": "user",
            "history": [],
            "state": {"plan": "pro"},
            "kwargs": {"run_config": "cfg"},
        },
        {
            "user_id": "simulhausen",
            "session_id": "t",
            "text": "more",
            "role": "user",
            "history": [("user", "hi"), ("support", "ok")],
            "state": {"plan": "pro"},
            "kwargs": {"run_config": "cfg"},
        },
    ]


async def test_app_and_workflow_node_targets() -> None:
    service = SessionService()
    app = SimpleNamespace(name="shop", root_agent=BaseAgent())
    adapter: Any = google_adk_adapter(app, session_service=service, user_id="ann")
    await _call(adapter, _input([("user", "hi")]))
    assert (adapter._runner.app, adapter._runner.app_name) == (app, "shop")
    assert ("shop", "ann", "t") in service.sessions

    node = SimpleNamespace(name="pipeline")
    adapter: Any = google_adk_adapter(node, app_name="flows")
    await _call(adapter, _input([("user", "hi")]))
    assert (adapter._runner.node, adapter._runner.app_name) == (node, "flows")


async def test_ready_runner_keeps_its_own_services() -> None:
    runner = _runner()
    runner.plugin_manager = None
    adapter: Any = google_adk_adapter(runner)

    assert (await _call(adapter, _input([("user", "hi")]))).content == "ok"
    assert runner.runs[0]["session_id"] == "t"

    wrong: list[dict[str, Any]] = [{"app_name": "x"}, {"session_service": SessionService()}]
    for options in wrong:
        with pytest.raises(ValueError, match="got a ready Runner"):
            await google_adk_adapter(_runner(), **options).call(_input([("user", "hi")]))


async def test_recorder_plugin_is_registered_once_per_runner() -> None:
    runner = _runner()
    await _call(google_adk_adapter(runner), _input([("user", "hi")]))
    await _call(google_adk_adapter(runner), _input([("user", "hi")], thread_id="u"))

    assert len(runner.plugin_manager.plugins) == 1


def test_history_is_validated() -> None:
    with pytest.raises(ValueError, match="history must be"):
        google_adk_adapter(BaseAgent(), history="server")  # ty: ignore[invalid-argument-type]


# -- sessions --------------------------------------------------------------------------------------------


async def test_session_mode_seeds_a_new_session_with_what_the_script_said_first() -> None:
    runner = _runner()
    adapter: Any = google_adk_adapter(runner)

    await _call(adapter, _input([("assistant", "Welcome"), ("user", "hi")]))

    assert runner.runs[0]["history"] == [("support", "Welcome")]
    assert runner.runs[0]["state"] is None
    assert runner.runs[0]["text"] == "hi"


async def test_session_mode_uses_an_existing_session_as_it_is() -> None:
    service = SessionService()
    stored = await service.create_session(app_name="shop", user_id="simulhausen", state={"kept": 1}, session_id="t")
    stored.events.append(say("from an earlier process", author="support"))
    runner = _runner(service=service)

    await _call(google_adk_adapter(runner, state={"ignored": 1}), _input([("assistant", "Welcome"), ("user", "hi")]))

    assert runner.runs[0]["history"] == [("support", "from an earlier process")]
    assert runner.runs[0]["state"] == {"kept": 1}


async def test_session_mode_stores_extra_new_messages_before_the_newest() -> None:
    runner = _runner()
    adapter: Any = google_adk_adapter(runner)
    await _call(adapter, _input([("user", "hi")]))

    follow_up = _input([("user", "hi"), ("assistant", "ok"), ("user", "one"), ("user", "two")], new=2)
    await _call(adapter, follow_up)

    assert runner.runs[1]["session_id"] == "t"
    assert runner.runs[1]["history"] == [("user", "hi"), ("support", "ok"), ("user", "one")]
    assert runner.runs[1]["text"] == "two"


async def test_session_mode_restarts_from_the_transcript_when_it_diverged() -> None:
    runner = _runner()
    adapter: Any = google_adk_adapter(runner)
    await _call(adapter, _input([("user", "hi")]))

    # The script injected a reply the agent never gave, and the agent is called without a new user message.
    await _call(adapter, _input([("user", "hi"), ("assistant", "scripted")], new=0))

    restarted = runner.runs[1]
    assert restarted["session_id"].startswith("t-")
    assert restarted["history"] == [("user", "hi"), ("support", "scripted")]
    assert restarted["text"] == "hi"
    # The new session is the thread's session from now on.
    await _call(adapter, _input([("user", "hi"), ("assistant", "scripted"), ("assistant", "ok"), ("user", "next")]))
    assert runner.runs[2]["session_id"] == restarted["session_id"]


async def test_threads_do_not_share_sessions() -> None:
    runner = _runner()
    adapter: Any = google_adk_adapter(runner)

    await _call(adapter, _input([("user", "hi")], thread_id="a"))
    await _call(adapter, _input([("user", "hi")], thread_id="b"))

    assert [(run["session_id"], run["history"]) for run in runner.runs] == [("a", []), ("b", [])]


async def test_transcript_mode_runs_every_call_in_a_seeded_throwaway_session() -> None:
    service = SessionService()
    runner = _runner(service=service)
    runner.agent.name = ""
    adapter: Any = google_adk_adapter(runner, history="transcript", state={"plan": "pro"})

    await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "hi"), ("assistant", "ok"), ("user", "more")]))

    first, second = runner.runs
    assert first["history"] == []
    assert second["history"] == [("user", "hi"), ("model", "ok")]
    assert second["state"] == {"plan": "pro"}
    assert first["session_id"] != second["session_id"]
    assert service.deleted == [first["session_id"], second["session_id"]]
    assert service.sessions == {}


async def test_exception_from_the_run_propagates_and_the_throwaway_session_is_deleted() -> None:
    service = SessionService()
    runner = _runner([RuntimeError("tool failed")], service=service)

    with pytest.raises(RuntimeError, match="tool failed"):
        await google_adk_adapter(runner, history="transcript").call(_input([("user", "hi")]))

    assert service.sessions == {}


# -- reply ------------------------------------------------------------------------------------------------


async def test_reply_is_the_last_final_text_without_thoughts_and_partials() -> None:
    thinking = Event(content=Content("model", [Part(text="hmm", thought=True), Part(text="first")]))
    runner = _runner([say("stre", partial=True), thinking, say("second"), say("am", partial=True), Event()])

    assert (await _call(google_adk_adapter(runner), _input([("user", "hi")]))).content == "second"


async def test_reply_falls_back_to_the_node_output() -> None:
    runner = _runner(
        [say("draft"), Event(output={"text": "DRAFT"})],
        [SimpleNamespace(content=None, output=None)],
    )
    adapter: Any = google_adk_adapter(runner)

    assert (await _call(adapter, _input([("user", "hi")]))).content == '{"text": "DRAFT"}'
    # An event without is_final_response is never the reply.
    assert (await _call(adapter, _input([("user", "hi")], thread_id="u"))).content == ""


async def test_content_fn_gets_the_events_of_the_turn() -> None:
    runner = _runner([say("a"), say("b", partial=True), say("c")])
    adapter: Any = google_adk_adapter(
        runner, content_fn=lambda events: "+".join(e.content.parts[0].text for e in events)
    )

    assert (await _call(adapter, _input([("user", "hi")]))).content == "a+c"


# -- tools and usage ------------------------------------------------------------------------------------


async def test_tool_calls_are_paired_with_their_responses() -> None:
    runner = _runner(
        [
            call("lookup", "c1", {"id": 7}),
            answer("lookup", "c1", {"result": "shipped"}),
            call("cancel", "c2", {"id": 7}),
            answer("cancel", "c2", {"error": "locked"}),
            call("details", "c3"),
            answer("details", "c3", {"status": "ok", "error": None}),
            call("flaky", "c4"),
            answer("flaky", "c4", {"error": "partial", "data": 1}),
            call("raw", "c5"),
            answer("raw", "c5", "plain"),
            call("unanswered", "c6"),
            say("done"),
        ]
    )

    turn = await _call(google_adk_adapter(runner), _input([("user", "hi")]))

    assert turn.tools is not None
    assert [(t.tool_call_id, t.tool_name, t.tool_args, t.result, t.tool_call_error) for t in turn.tools] == [
        ("c1", "lookup", {"id": 7}, "shipped", None),
        ("c2", "cancel", {"id": 7}, "locked", True),
        ("c3", "details", {}, '{"status": "ok", "error": null}', None),
        ("c4", "flaky", {}, '{"error": "partial", "data": 1}', True),
        ("c5", "raw", {}, "plain", None),
        ("c6", "unanswered", {}, None, None),
    ]
    assert not any(tool.is_paused for tool in turn.tools)


async def test_calls_without_ids_are_paired_by_name_in_order() -> None:
    runner = _runner(
        [
            call("lookup", None, {"id": 1}),
            call("lookup", None, {"id": 2}),
            answer("lookup", None, {"result": "one"}),
            answer("lookup", None, {"result": "two"}),
            answer("orphan", "c9", {"result": "no call seen"}),
        ]
    )

    turn = await _call(google_adk_adapter(runner), _input([("user", "hi")]))

    assert turn.tools is not None
    assert [(t.tool_name, t.tool_args, t.result) for t in turn.tools] == [
        ("lookup", {"id": 1}, "one"),
        ("lookup", {"id": 2}, "two"),
        ("orphan", {}, "no call seen"),
    ]


async def test_held_and_long_running_calls_are_not_executed() -> None:
    confirmation = call("adk_request_confirmation", "h1", {"originalFunctionCall": {"id": "c1"}})
    credentials = call("adk_request_credential", "h2", {"functionCallId": "c2"})
    runner = _runner(
        [
            call("refund", "c1", {"id": 7}),
            confirmation,
            answer("refund", "c1", {"error": "This tool call requires confirmation, please approve or reject."}),
            call("calendar", "c2"),
            credentials,
            call("adk_request_confirmation", "h3", {}),
            answer("adk_request_confirmation", "h3", {"confirmed": True}),
            call("export", "c3", long_running_tool_ids={"c3"}),
            answer("export", "c3", {"result": "started"}),
        ]
    )

    turn = await _call(google_adk_adapter(runner), _input([("user", "hi")]))

    assert turn.tools is not None
    assert [(t.tool_name, t.is_paused, t.tool_call_error, t.result) for t in turn.tools] == [
        ("refund", True, None, "This tool call requires confirmation, please approve or reject."),
        ("calendar", True, None, None),
        ("export", True, None, "started"),
    ]


async def test_usage_is_summed_over_the_events() -> None:
    usage = {"prompt_token_count": 5, "candidates_token_count": 2, "total_token_count": 7}
    runner = _runner(
        [
            call("lookup", "c1", usage_metadata=usage),
            say("par", partial=True, usage_metadata=usage),
            answer("lookup", "c1", {"result": 1}),
            say("done", usage_metadata=SimpleNamespace(**usage)),
        ]
    )

    turn = await _call(google_adk_adapter(runner), _input([("user", "hi")]))

    assert turn.usage == Usage(input_tokens=10, output_tokens=4, total_tokens=14, requests=2)


async def test_plugin_records_add_what_a_nested_runner_did() -> None:
    usage = {"prompt_token_count": 5, "candidates_token_count": 2, "total_token_count": 7}
    tool = SimpleNamespace(name="lookup")
    inner = SimpleNamespace(invocation_id="nested", function_call_id="n1")
    outer = SimpleNamespace(invocation_id="inv", function_call_id="c1")

    async def nested_run(plugin: Any) -> None:
        response = SimpleNamespace(usage_metadata=usage, partial=None)
        await plugin.after_model_callback(callback_context=inner, llm_response=response)
        await plugin.after_model_callback(callback_context=inner, llm_response=SimpleNamespace(usage_metadata=None))
        partial = SimpleNamespace(usage_metadata=usage, partial=True)
        await plugin.after_model_callback(callback_context=inner, llm_response=partial)
        # A nested tool that raised and whose error a callback turned into a response: two records, one call.
        await plugin.on_tool_error_callback(tool=tool, tool_args={"id": 7}, tool_context=inner, error=KeyError("7"))
        await plugin.after_tool_callback(tool=tool, tool_args={"id": 7}, tool_context=inner, result={"error": "gone"})
        other = SimpleNamespace(invocation_id="nested", function_call_id=None)
        await plugin.after_tool_callback(tool=tool, tool_args={"id": 8}, tool_context=other, result={"result": "ok"})

    async def outer_callbacks(plugin: Any) -> None:
        # The outer run's own model and tool calls are already in the events.
        response = SimpleNamespace(usage_metadata=usage, partial=None)
        await plugin.after_model_callback(callback_context=outer, llm_response=response)
        helper = SimpleNamespace(name="helper")
        await plugin.after_tool_callback(tool=helper, tool_args={}, tool_context=outer, result={"result": "done"})

    runner = _runner(
        [
            call("helper", "c1", {"request": "order 7"}, usage_metadata=usage),
            nested_run,
            outer_callbacks,
            answer("helper", "c1", {"result": "done"}),
            say("finished", usage_metadata=usage),
        ]
    )
    adapter: Any = google_adk_adapter(runner)

    turn = await _call(adapter, _input([("user", "hi")]))

    assert turn.tools is not None
    assert [(t.tool_call_id, t.tool_name, t.tool_args, t.result, t.tool_call_error) for t in turn.tools] == [
        ("c1", "helper", {"request": "order 7"}, "done", None),
        ("n1", "lookup", {"id": 7}, "gone", True),
        (None, "lookup", {"id": 8}, "ok", None),
    ]
    assert turn.usage == Usage(input_tokens=15, output_tokens=6, total_tokens=21, requests=3)

    # Outside a call of the adapter the plugin records nothing.
    plugin = runner.plugin_manager.plugins[0]
    await plugin.after_tool_callback(tool=tool, tool_args={}, tool_context=inner, result={})
    assert (await _call(adapter, _input([("user", "hi")], thread_id="u"))).tools is None
