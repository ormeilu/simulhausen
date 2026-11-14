from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import simulhausen as sim
from simulhausen import AgentInput, AgentTurn, Usage, http_adapters
from simulhausen.frameworks import langchain as module
from simulhausen.frameworks.langchain import langchain_adapter

# -- fakes shaped like langchain_core.messages and a compiled LangGraph graph -----------


@dataclass
class Msg:
    type: str
    content: Any = ""
    id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    status: str | None = None
    usage_metadata: dict[str, int] | None = None


def ai(
    content: Any = "", *calls: tuple[str, dict[str, Any], str | None], id: str | None = None, tokens: int = 0
) -> Msg:
    usage = {"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1} if tokens else None
    tool_calls = [{"name": name, "args": args, "id": call_id} for name, args, call_id in calls]
    return Msg("ai", content, id=id, tool_calls=tool_calls, usage_metadata=usage)


def tool(content: Any, call_id: str | None, *, status: str = "success", id: str | None = None) -> Msg:
    return Msg("tool", content, id=id, tool_call_id=call_id, status=status)


class Command:
    def __init__(self, resume: Any) -> None:
        self.resume = resume


type Step = tuple[tuple[str, ...], list[Msg]]


class Graph:
    """A compiled graph: an optional checkpointer, ``astream`` yielding ``(namespace, mode, state)``.

    ``script`` gets the root messages after the input was applied and returns
    the steps of the run: messages a node of the root graph adds, or the state
    of a subgraph under its namespace. ``interrupt`` ends the run with pending
    interrupts in the final state.
    """

    def __init__(self, script: Any, *, checkpointer: Any = None, key: str = "messages", interrupt: Any = None) -> None:
        self.checkpointer = checkpointer
        self.script = script
        self.key = key
        self.interrupt = interrupt
        self.threads: dict[str, list[Msg]] = {}
        self.calls: list[SimpleNamespace] = []
        self._ids = 0

    def _coerce(self, message: Any) -> Msg:
        if isinstance(message, Msg):
            return message
        self._ids += 1
        kind = {"user": "human", "assistant": "ai"}[message["role"]]
        return Msg(kind, message["content"], id=f"in-{self._ids}")

    async def astream(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls.append(SimpleNamespace(payload=payload, config=config, kwargs=kwargs))
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        messages = list(self.threads.get(thread_id, [])) if self.checkpointer not in (None, True) else []
        if isinstance(payload, dict):
            messages += [self._coerce(message) for message in payload[self.key]]
        yield (), "values", {self.key: list(messages)}
        for namespace, new in self.script(messages):
            if namespace:
                yield namespace, "values", {self.key: list(new)}
            else:
                messages += new
                yield (), "values", {self.key: list(messages)}
        if self.interrupt is not None:
            yield (), "values", {self.key: list(messages), "__interrupt__": [self.interrupt]}
        if thread_id is not None:
            self.threads[thread_id] = messages


def _echo(messages: list[Msg]) -> list[Step]:
    """One model call that reports the human messages it was shown and how many messages it saw."""
    users = "|".join(str(m.content) for m in messages if m.type == "human")
    return [((), [ai(f"{users} ({len(messages)})", tokens=5)])]


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


FIRST = [("user", "hi")]
SECOND = [("user", "hi"), ("assistant", "hi (1)"), ("user", "more")]


# -- names -----------------------------------------------------------------------------


def test_the_in_process_factory_does_not_shadow_the_http_langgraph_adapter() -> None:
    assert module.__all__ == ["langchain_adapter"]
    assert not hasattr(module, "langgraph_adapter")
    assert sim.langgraph_adapter is http_adapters.langgraph_adapter


def test_an_unknown_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="history must be"):
        langchain_adapter(Graph(_echo), history="nope")  # ty: ignore[invalid-argument-type]


# -- message helpers ---------------------------------------------------------------------


def test_kind_tells_messages_from_other_values() -> None:
    assert module._kind(ai("x")) == "ai"
    assert module._kind(Msg("AIMessageChunk")) == "ai"
    assert module._kind({"role": "assistant", "content": "x"}) == "ai"
    assert module._kind({"type": "tool", "content": "x"}) == "tool"
    assert module._kind({"role": "assistant"}) is None
    assert module._kind(SimpleNamespace(type="ai")) is None
    assert module._kind(SimpleNamespace(type=3, content="x")) is None
    assert module._kind("text") is None


def test_reply_text() -> None:
    assert module._reply_text([Msg("human", "q"), ai([{"type": "text", "text": "a"}, {"type": "tool_use"}])]) == "a"
    assert module._reply_text(ai("plain")) == "plain"
    assert module._reply_text("text") == "text"
    assert module._reply_text({"answer": 1}) == '{"answer": 1}'
    assert module._reply_text([]) == ""
    assert module._reply_text(None) == ""


def test_tools_are_paired_by_id_and_usage_is_summed() -> None:
    fresh = [
        ai(
            "",
            ("lookup", {"id": 1}, "c1"),
            ("refund", {"id": 1}, "c2"),
            ("ask", {}, "c3"),
            ("anon", {}, None),
            tokens=5,
        ),
        tool("found", "c1"),
        tool("closed", "c2", status="error"),
        tool("stray", None),
        Msg("human", "ignored"),
        ai("done", tokens=7),
        ai("synthetic hand-back"),
    ]

    tools, usage = module._tools_and_usage(fresh)

    assert [(t.tool_name, t.tool_args, t.result, t.tool_call_id) for t in tools] == [
        ("lookup", {"id": 1}, "found", "c1"),
        ("refund", {"id": 1}, "closed", "c2"),
        ("ask", {}, None, "c3"),
        ("anon", {}, None, None),
    ]
    assert [bool(t.tool_call_error) for t in tools] == [False, True, False, False]
    assert [t.is_paused for t in tools] == [False, False, True, True]
    # Only the two messages that carry usage_metadata count as requests.
    assert usage == Usage(input_tokens=12, output_tokens=2, total_tokens=14, requests=2)
    assert module._tools_and_usage([Msg("human", "hi")]) == ([], None)


def test_an_earlier_call_counts_only_when_its_result_is_new() -> None:
    earlier = [ai("", ("delete", {"user": "u"}, "c1"), ("other", {}, "c9"), tokens=5), Msg("human", "old")]

    tools, usage = module._tools_and_usage([tool("deleted", "c1"), ai("done")], earlier)

    assert [(t.tool_name, t.result, t.is_paused) for t in tools] == [("delete", "deleted", False)]
    # The earlier message's usage belongs to the turn that produced it.
    assert usage is None


# -- graphs ------------------------------------------------------------------------------


async def test_a_graph_with_a_checkpointer_runs_on_the_scenario_thread_and_gets_only_new_messages() -> None:
    graph = Graph(_echo, checkpointer=object())
    built: list[int] = []

    def factory() -> Graph:
        built.append(1)
        return graph

    adapter = langchain_adapter(
        factory, config={"recursion_limit": 9, "configurable": {"tenant": "acme"}}, context="ctx"
    )

    first = await _turn(adapter, _input(FIRST))
    second = await _turn(adapter, _input(SECOND))

    assert (first.content, second.content) == ("hi (1)", "hi|more (3)")
    assert first.tools is None
    assert first.usage == Usage(input_tokens=5, output_tokens=1, total_tokens=6, requests=1)
    assert built == [1]
    assert [call.payload for call in graph.calls] == [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "more"}]},
    ]
    expected = {"recursion_limit": 9, "configurable": {"tenant": "acme", "thread_id": "t"}}
    assert [call.config for call in graph.calls] == [expected, expected]
    assert graph.calls[0].kwargs == {"stream_mode": ["values"], "subgraphs": True, "context": "ctx"}


async def test_a_rollback_moves_a_checkpointed_graph_to_a_new_thread_with_the_whole_transcript() -> None:
    graph = Graph(_echo, checkpointer=object())
    adapter = langchain_adapter(graph)

    await adapter.call(_input(FIRST))
    await adapter.call(_input(SECOND))
    rolled_back = await _turn(adapter, _input([("user", "hi"), ("assistant", "hi (1)"), ("user", "other")]))
    after = await _turn(
        adapter,
        _input(
            [*FIRST, ("assistant", "hi (1)"), ("user", "other"), ("assistant", rolled_back.content), ("user", "last")]
        ),
    )

    assert rolled_back.content == "hi|other (3)"
    thread_ids = [call.config["configurable"]["thread_id"] for call in graph.calls]
    assert thread_ids[:2] == ["t", "t"]
    assert thread_ids[2].startswith("t-")
    assert len(graph.calls[2].payload["messages"]) == 3
    # The new thread is the session from then on.
    assert thread_ids[3] == thread_ids[2]
    assert graph.calls[3].payload == {"messages": [{"role": "user", "content": "last"}]}
    assert after.content == "hi|other|last (5)"


async def test_transcript_mode_uses_a_unique_thread_only_when_there_is_a_checkpointer() -> None:
    saved = Graph(_echo, checkpointer=object())
    plain = Graph(_echo)

    for graph in (saved, plain):
        adapter = langchain_adapter(graph, history="transcript")
        await adapter.call(_input(FIRST))
        second = await _turn(adapter, _input(SECOND))
        assert second.content == "hi|more (3)"
        assert len(graph.calls[1].payload["messages"]) == 3

    first_thread, second_thread = (call.config["configurable"]["thread_id"] for call in saved.calls)
    assert first_thread.startswith("t-")
    assert second_thread.startswith("t-")
    assert first_thread != second_thread
    assert [call.config for call in plain.calls] == [None, None]


@pytest.mark.parametrize("checkpointer", [None, True])
async def test_a_graph_without_a_checkpointer_gets_the_carried_messages_back(checkpointer: Any) -> None:
    def script(messages: list[Msg]) -> list[Step]:
        if messages[-1].content == "hi":
            return [((), [ai("", ("lookup", {"id": 1}, "c1"))]), ((), [tool("found", "c1")]), ((), [ai("hi (1)")])]
        return _echo(messages)

    graph = Graph(script, checkpointer=checkpointer)
    adapter = langchain_adapter(graph)

    first = await _turn(adapter, _input(FIRST))
    second = await _turn(adapter, _input(SECOND))
    rebuilt = await _turn(adapter, _input([("user", "other")]))
    other_thread = await _turn(adapter, _input(SECOND, thread_id="u"))

    assert [t.tool_name for t in first.tools or []] == ["lookup"]
    # The native messages of the first turn, tool call and result included, plus the new one.
    assert [getattr(m, "type", m) for m in graph.calls[1].payload["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        {"role": "user", "content": "more"},
    ]
    assert second.content == "hi|more (5)"
    assert second.tools is None
    assert rebuilt.content == "other (1)"
    assert other_thread.content == "hi|more (3)"
    assert all(call.config is None for call in graph.calls)


async def test_messages_of_subgraphs_are_collected_once_in_order_of_appearance() -> None:
    def script(messages: list[Msg]) -> list[Step]:
        handoff = ai("", ("transfer", {}, "c0"), id="a0", tokens=1)
        moved = tool("moved", "c0", id="t0")
        inner_call = ai("", ("lookup", {"id": 1}, "c1"), id="a1", tokens=2)
        inner_result = tool("found", "c1", id="t1")
        inner_reply = ai("inner done", id="a2", tokens=3)
        seen_by_sub = [*messages, handoff, moved]
        return [
            (("boss:1",), [*messages, handoff]),
            ((), [handoff, moved]),
            (("worker:2",), [*seen_by_sub, inner_call]),
            (("worker:2",), [*seen_by_sub, inner_call, inner_result, inner_reply]),
            ((), [inner_reply, ai("all done", id="a3", tokens=4)]),
        ]

    turn = await _turn(langchain_adapter(Graph(script)), _input(FIRST))

    assert turn.content == "all done"
    assert [(t.tool_name, t.result) for t in turn.tools or []] == [("transfer", "moved"), ("lookup", "found")]
    assert turn.usage == Usage(input_tokens=10, output_tokens=4, total_tokens=14, requests=4)


async def test_an_interrupt_is_the_reply_and_resume_fn_resumes_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    required: list[tuple[str, str, str]] = []

    def fake_require(name: str, *, package: str, adapter: str) -> Any:
        required.append((name, package, adapter))
        return SimpleNamespace(Command=Command)

    monkeypatch.setattr(module, "require", fake_require)

    def script(messages: list[Msg]) -> list[Step]:
        if messages[-1].type == "human":
            return [((), [ai("", ("delete", {"user": "u"}, "c1"), id="a1", tokens=5)])]
        return [((), [tool("deleted", "c1", id="t1")]), ((), [ai("done", id="a2", tokens=5)])]

    graph = Graph(script, checkpointer=object(), interrupt=SimpleNamespace(value={"ask": "delete?"}))
    adapter = langchain_adapter(graph, resume_fn=lambda agent_input: agent_input.last_new_user_message_str().upper())

    asked = await _turn(adapter, _input([("user", "delete me")]))
    graph.interrupt = None
    resumed = await _turn(adapter, _input([("user", "delete me"), ("assistant", asked.content), ("user", "yes")]))

    assert asked.content == '{"ask": "delete?"}'
    assert [(t.tool_name, t.is_paused) for t in asked.tools or []] == [("delete", True)]
    command = graph.calls[1].payload
    assert isinstance(command, Command)
    assert command.resume == "YES"
    assert required == [("langgraph.types", "langgraph", "langchain_adapter")]
    assert resumed.content == "done"
    assert [(t.tool_name, t.result, t.is_paused) for t in resumed.tools or []] == [("delete", "deleted", False)]
    assert resumed.usage == Usage(input_tokens=5, output_tokens=1, total_tokens=6, requests=1)


async def test_without_resume_fn_an_interrupted_thread_gets_the_next_message() -> None:
    graph = Graph(
        lambda messages: [((), [ai("thinking", id=f"a{len(messages)}")])], checkpointer=object(), interrupt="stop"
    )
    adapter = langchain_adapter(graph)

    asked = await _turn(adapter, _input([("user", "go")]))
    await adapter.call(_input([("user", "go"), ("assistant", "thinking"), ("user", "yes")]))

    # The last message has text, so the interrupt does not replace it.
    assert asked.content == "thinking"
    assert graph.calls[1].payload == {"messages": [{"role": "user", "content": "yes"}]}


async def test_a_bare_interrupt_value_is_rendered() -> None:
    graph = Graph(lambda messages: [((), [ai("")])], interrupt="approve?")

    assert (await _turn(langchain_adapter(graph), _input(FIRST))).content == "approve?"


async def test_input_key_inputs_output_key_and_content_fn() -> None:
    graph = Graph(_echo, key="dialogue")

    by_key = await _turn(langchain_adapter(graph, input_key="dialogue", inputs={"plan": "pro"}), _input(FIRST))
    other_key = await _turn(langchain_adapter(graph, input_key="dialogue", output_key="missing"), _input(FIRST))
    custom = langchain_adapter(
        graph, input_key="dialogue", content_fn=lambda final: f"{len(final['dialogue'])} messages"
    )

    assert by_key.content == "hi (1)"
    assert graph.calls[0].payload == {"plan": "pro", "dialogue": [{"role": "user", "content": "hi"}]}
    assert other_key.content == ""
    assert (await _turn(custom, _input(FIRST))).content == "2 messages"


async def test_a_graph_whose_state_holds_no_message_list() -> None:
    class Plain:
        checkpointer = None

        def __init__(self, final: Any) -> None:
            self.final = final
            self.payloads: list[Any] = []

        async def astream(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
            self.payloads.append(payload)
            yield (), "values", self.final

    text = Plain("just text")
    adapter = langchain_adapter(text)
    first = await _turn(adapter, _input(FIRST))
    await adapter.call(_input([("user", "hi"), ("assistant", "just text"), ("user", "more")]))

    assert first.content == "just text"
    # Nothing to carry: the next turn is rebuilt from the transcript.
    assert len(text.payloads[1]["messages"]) == 3
    assert (await _turn(langchain_adapter(Plain({"answer": 42})), _input(FIRST))).content == '{"answer": 42}'


# -- AgentExecutor -------------------------------------------------------------------------


class Executor:
    return_intermediate_steps = True

    def __init__(self, steps: list[Any] | None = None) -> None:
        self.steps = steps
        self.calls: list[SimpleNamespace] = []

    async def ainvoke(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls.append(SimpleNamespace(payload=payload, config=config, kwargs=kwargs))
        result = {**payload, "output": f"{payload['input']} ({len(payload['chat_history'])})"}
        if self.steps is not None and len(self.calls) == 1:
            result["intermediate_steps"] = self.steps
        return result


async def test_an_executor_gets_the_question_and_the_carried_history_with_its_tool_steps() -> None:
    asked = ai("", ("lookup", {"id": 1}, "c1"), ("stock", {}, "c2"))
    steps = [
        (SimpleNamespace(tool="lookup", tool_input={"id": 1}, tool_call_id="c1", message_log=[asked]), "found"),
        (SimpleNamespace(tool="stock", tool_input="A", tool_call_id="c2", message_log=[asked]), {"left": 3}),
        (SimpleNamespace(tool="think", tool_input="hmm", log="Thought: hmm"), "nothing"),
    ]
    executor = Executor(steps)
    adapter = langchain_adapter(executor, inputs={"tone": "brief"}, config={"tags": ["sim"]}, max_concurrency=1)

    first = await _turn(adapter, _input(FIRST))
    second = await _turn(adapter, _input([("user", "hi"), ("assistant", "hi (0)"), ("user", "more")]))

    assert first.content == "hi (0)"
    assert [(t.tool_name, t.tool_args, t.result, t.tool_call_id) for t in first.tools or []] == [
        ("lookup", {"id": 1}, "found", "c1"),
        ("stock", {"arguments": "A"}, '{"left": 3}', "c2"),
        ("think", {"arguments": "hmm"}, "nothing", None),
    ]
    assert first.usage is None
    assert executor.calls[0].payload == {"tone": "brief", "chat_history": [], "input": "hi"}
    assert (executor.calls[0].config, executor.calls[0].kwargs) == ({"tags": ["sim"]}, {"max_concurrency": 1})
    # The first turn as messages: the question, the model's tool calls once, both results, the answer.
    assert executor.calls[1].payload["chat_history"] == [
        {"role": "user", "content": "hi"},
        asked,
        {"role": "tool", "content": "found", "tool_call_id": "c1"},
        {"role": "tool", "content": '{"left": 3}', "tool_call_id": "c2"},
        {"role": "assistant", "content": "hi (0)"},
    ]
    assert second.content == "more (5)"
    assert second.tools is None


async def test_an_executor_in_transcript_mode_and_with_other_keys() -> None:
    executor = Executor()
    adapter = langchain_adapter(executor, history="transcript")

    await adapter.call(_input(FIRST))
    second = await _turn(adapter, _input([("user", "hi"), ("assistant", "ignored"), ("user", "more")]))
    no_question = await _turn(adapter, _input([("assistant", "welcome")], new=0))

    assert second.content == "more (2)"
    assert executor.calls[1].payload["chat_history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ignored"},
    ]
    assert no_question.content == " (1)"

    class Renamed(Executor):
        async def ainvoke(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
            return {"answer": payload["question"].upper()}

    renamed = langchain_adapter(Renamed(), input_key="question", output_key="answer")
    assert (await _turn(renamed, _input(FIRST))).content == "HI"


# -- chat models and chains ----------------------------------------------------------------


class Model:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[SimpleNamespace] = []

    async def ainvoke(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls.append(SimpleNamespace(payload=payload, config=config, kwargs=kwargs))
        return self.reply


async def test_a_chat_model_gets_the_transcript_and_its_tool_calls_are_not_executed() -> None:
    model = Model(ai([{"type": "text", "text": "let me check"}], ("lookup", {"id": 1}, "c1"), tokens=9))
    adapter = langchain_adapter(model, config={"tags": ["sim"]}, stop=["\n"])

    turn = await _turn(adapter, _input(SECOND))

    assert turn.content == "let me check"
    assert [(t.tool_name, t.tool_args, t.is_paused) for t in turn.tools or []] == [("lookup", {"id": 1}, True)]
    assert turn.usage == Usage(input_tokens=9, output_tokens=1, total_tokens=10, requests=1)
    call = model.calls[0]
    assert call.payload == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi (1)"},
        {"role": "user", "content": "more"},
    ]
    assert (call.config, call.kwargs) == ({"tags": ["sim"]}, {"stop": ["\n"]})


async def test_a_chain_answers_with_a_string_or_a_dict() -> None:
    text = Model("plain text")
    wrapped = Model({"output": "from output", "other": [ai("last message")]})

    assert (await _turn(langchain_adapter(text), _input(FIRST))).content == "plain text"
    assert text.calls[0].config is None
    keyed = langchain_adapter(wrapped, input_key="history", inputs={"tone": "brief"})
    assert (await _turn(keyed, _input(FIRST))).content == "from output"
    assert wrapped.calls[0].payload == {"tone": "brief", "history": [{"role": "user", "content": "hi"}]}
    assert (await _turn(langchain_adapter(wrapped, output_key="other"), _input(FIRST))).content == "last message"
    assert (await _turn(langchain_adapter(Model({"n": 1})), _input(FIRST))).content == '{"n": 1}'
    custom = langchain_adapter(wrapped, content_fn=lambda result: result["output"].upper())
    assert (await _turn(custom, _input(FIRST))).content == "FROM OUTPUT"
