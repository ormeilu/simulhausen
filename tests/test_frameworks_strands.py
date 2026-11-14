"""Unit tests of ``strands_adapter`` with hand-written fakes shaped like the Strands objects.

The real library is exercised in ``tests/contract/test_strands_contract.py``.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks import strands
from simulhausen.frameworks.strands import strands_adapter


def text(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": [{"text": content}]}


def use(call_id: str, name: str, args: Any = None) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": call_id, "name": name, "input": args}}]}


def tool_result(call_id: str, content: Any, status: str = "success") -> dict[str, Any]:
    return {"role": "user", "content": [{"toolResult": {"toolUseId": call_id, "status": status, "content": content}}]}


class Result:
    def __init__(self, reply: str, **fields: Any) -> None:
        self.reply = reply
        self.__dict__.update(fields)

    def __str__(self) -> str:
        return f"{self.reply}\n"


class Agent:
    """Appends the prompt, the scripted messages of the turn and the reply to ``messages``, like a Strands agent."""

    def __init__(self, *turns: list[dict[str, Any]], messages: list[dict[str, Any]] | None = None) -> None:
        self.messages: list[dict[str, Any]] = messages or []
        self.turns = list(turns)
        self.prompts: list[tuple[str, list[str], dict[str, Any]]] = []
        self.event_loop_metrics: Any = SimpleNamespace(accumulated_usage={"inputTokens": 0, "outputTokens": 0})

    def __call__(self, prompt: str) -> Any:
        raise AssertionError("an agent must not be mistaken for a factory")

    async def invoke_async(self, prompt: str, **kwargs: Any) -> Result:
        seen = [f"{m['role']}:{m['content'][0].get('text', '<tool>')}" for m in self.messages]
        self.prompts.append((prompt, seen, kwargs))
        self.messages.append(text("user", prompt))
        self.messages.extend(self.turns.pop(0) if self.turns else [])
        reply = f"reply {len(self.prompts)}"
        self.messages.append(text("assistant", reply))
        totals = self.event_loop_metrics.accumulated_usage
        totals["inputTokens"] += 5
        totals["outputTokens"] += 2
        return Result(reply, metrics=SimpleNamespace(accumulated_usage=totals))


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


async def _call(adapter: Any, input: AgentInput) -> AgentTurn:
    turn = await adapter.call(input)
    assert isinstance(turn, AgentTurn)
    return turn


def test_history_is_validated() -> None:
    with pytest.raises(ValueError, match="history must be"):
        strands_adapter(Agent(), history="server")  # ty: ignore[invalid-argument-type]


# -- one agent ---------------------------------------------------------------------------------------------


async def test_session_mode_sends_the_newest_message_and_keeps_the_agents_messages() -> None:
    agent = Agent()
    adapter = strands_adapter(agent, invocation_state={"tenant": "a"})

    first = await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    assert first == AgentTurn(content="reply 1", usage=Usage(input_tokens=5, output_tokens=2))
    assert agent.prompts == [
        ("hi", [], {"invocation_state": {"tenant": "a"}}),
        ("more", ["user:hi", "assistant:reply 1"], {"invocation_state": {"tenant": "a"}}),
    ]


async def test_session_mode_keeps_preset_messages_and_seeds_what_the_script_said_first() -> None:
    agent = Agent(messages=[text("user", "example"), text("assistant", "ok")])

    await _call(strands_adapter(agent), _input([("assistant", "Welcome"), ("user", "hi")]))

    assert agent.prompts[0][:2] == ("hi", ["user:example", "assistant:ok", "assistant:Welcome"])


async def test_session_mode_adds_extra_new_messages_before_the_newest() -> None:
    agent = Agent()
    adapter = strands_adapter(agent)
    await _call(adapter, _input([("user", "hi")]))

    await _call(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "one"), ("user", "two")], new=2))

    assert agent.prompts[1][:2] == ("two", ["user:hi", "assistant:reply 1", "user:one"])


async def test_session_mode_rebuilds_the_messages_when_the_transcript_diverged() -> None:
    agent = Agent([use("c1", "lookup"), tool_result("c1", [{"text": "found"}])], messages=[text("user", "example")])
    adapter = strands_adapter(agent)
    await _call(adapter, _input([("user", "hi")]))

    # A scripted reply the agent never gave, and no new user message: the last one is sent again.
    await _call(adapter, _input([("user", "hi"), ("assistant", "scripted")], new=0))

    assert agent.prompts[1][:2] == ("hi", ["user:example", "user:hi", "assistant:scripted"])


async def test_one_agent_object_keeps_threads_apart_by_swapping_messages() -> None:
    agent = Agent()
    adapter = strands_adapter(agent)

    await _call(adapter, _input([("user", "a1")], thread_id="a"))
    await _call(adapter, _input([("user", "b1")], thread_id="b"))
    await _call(adapter, _input([("user", "a1"), ("assistant", "reply 1"), ("user", "a2")], thread_id="a"))

    assert [seen for _, seen, _ in agent.prompts] == [[], [], ["user:a1", "assistant:reply 1"]]


async def test_factory_builds_one_agent_per_thread() -> None:
    built: list[Agent] = []

    def factory() -> Agent:
        built.append(Agent())
        return built[-1]

    adapter = strands_adapter(factory)
    await _call(adapter, _input([("user", "a1")], thread_id="a"))
    await _call(adapter, _input([("user", "b1")], thread_id="b"))
    await _call(adapter, _input([("user", "a1"), ("assistant", "reply 1"), ("user", "a2")], thread_id="a"))

    assert len(built) == 2
    assert [prompt for prompt, _, _ in built[0].prompts] == ["a1", "a2"]
    assert [prompt for prompt, _, _ in built[1].prompts] == ["b1"]


async def test_transcript_mode_rebuilds_the_messages_on_every_call() -> None:
    agent = Agent([use("c1", "lookup"), tool_result("c1", [{"text": "found"}])])
    adapter = strands_adapter(agent, history="transcript")

    await _call(adapter, _input([("user", "hi")]))
    await _call(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    # The tool use of the first turn is not replayed: only the transcript is.
    assert agent.prompts[1][:2] == ("more", ["user:hi", "assistant:reply 1"])


async def test_content_fn_gets_the_result() -> None:
    adapter = strands_adapter(Agent(), content_fn=lambda result: result.reply.upper())

    assert (await _call(adapter, _input([("user", "hi")]))).content == "REPLY 1"


async def test_tools_are_the_ones_of_this_turn_paired_by_id() -> None:
    first: list[dict[str, Any]] = [
        use("c1", "lookup", {"id": 7}),
        tool_result("c1", [{"text": "shipped"}, {"text": "today"}]),
        {
            "role": "assistant",
            "content": [
                {"text": "Two more."},
                {"toolUse": {"toolUseId": "c2", "name": "cancel", "input": '{"id": 7}'}},
                {"toolUse": {"toolUseId": "c3", "name": "details", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "c3", "status": "success", "content": [{"json": {"n": 1}}]}},
                {"toolResult": {"toolUseId": "c2", "status": "error", "content": [{"text": "Error: locked"}]}},
            ],
        },
        use("c4", "mixed"),
        tool_result("c4", [{"text": "a"}, {"image": "..."}]),
        use("c5", "plain"),
        tool_result("c5", "raw"),
        tool_result("c9", [{"text": "orphan"}]),
        use("c6", "approve"),
    ]
    agent = Agent(first)
    adapter = strands_adapter(agent)

    turn = await _call(adapter, _input([("user", "hi")]))
    second = await _call(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))

    assert turn.tools is not None
    assert [
        (t.tool_call_id, t.tool_name, t.tool_args, t.result, t.tool_call_error, t.is_paused) for t in turn.tools
    ] == [
        ("c1", "lookup", {"id": 7}, "shipped\ntoday", None, False),
        ("c2", "cancel", {"id": 7}, "Error: locked", True, False),
        ("c3", "details", {}, '{"n": 1}', None, False),
        ("c4", "mixed", {}, "a", None, False),
        ("c5", "plain", {}, "raw", None, False),
        ("c9", None, {}, "orphan", None, False),
        ("c6", "approve", {}, None, None, True),
    ]
    assert second.tools is None


async def test_usage_prefers_the_invocation_context_and_falls_back_to_the_metrics_delta() -> None:
    class Contextual(Agent):
        async def invoke_async(self, prompt: str, **kwargs: Any) -> Result:
            usage = {"inputTokens": 9, "outputTokens": 3, "totalTokens": 12, "turns_taken": 2}
            return Result("contextual", invocation_context=SimpleNamespace(usage=usage))

    turn = await _call(strands_adapter(Contextual()), _input([("user", "hi")]))
    assert turn.usage == Usage(input_tokens=9, output_tokens=3, total_tokens=12, requests=2)

    agent = Agent()
    adapter = strands_adapter(agent)
    await _call(adapter, _input([("user", "hi")]))
    second = await _call(adapter, _input([("user", "hi"), ("assistant", "reply 1"), ("user", "more")]))
    # The agent's metrics count from its creation; the turn gets what they grew by.
    assert second.usage == Usage(input_tokens=5, output_tokens=2)

    class Bare(Agent):
        async def invoke_async(self, prompt: str, **kwargs: Any) -> Result:
            return Result("no metrics", invocation_context=SimpleNamespace(usage={}))

    assert (await _call(strands_adapter(Bare()), _input([("user", "hi")]))).usage is None

    class FreshMetrics(Agent):
        async def invoke_async(self, prompt: str, **kwargs: Any) -> Result:
            usage = {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5}
            return Result("fresh", metrics=SimpleNamespace(accumulated_usage=usage))

    fresh = FreshMetrics()
    fresh.event_loop_metrics = None
    assert (await _call(strands_adapter(fresh), _input([("user", "hi")]))).usage == Usage(4, 1, 5)


# -- Swarm / Graph -----------------------------------------------------------------------------------------


class Orchestrator:
    """A Swarm-like object: nodes with executors, hooks, a fresh run per ``invoke_async``."""

    def __init__(self, nodes: dict[str, Any], plan: list[tuple[str, list[dict[str, Any]], str]]) -> None:
        self.nodes = nodes
        self.plan = plan
        self.hooks: list[tuple[Any, Any]] = []
        self.tasks: list[tuple[str, dict[str, Any]]] = []

    def add_hook(self, callback: Any, event_type: Any) -> None:
        self.hooks.append((callback, event_type))

    async def invoke_async(self, task: str, **kwargs: Any) -> Any:
        self.tasks.append((task, kwargs))
        results: dict[str, Any] = {}
        history: list[Any] = []
        for node_id, messages, reply in self.plan:
            node = self.nodes[node_id]
            if hasattr(node.executor, "messages"):
                node.executor.messages = [*copy.deepcopy(node._initial_messages), *messages]
            for callback, _ in self.hooks:
                callback(SimpleNamespace(source=self, node_id=node_id))
            results[node_id] = SimpleNamespace(result=Result(reply))
            history.append(SimpleNamespace(node_id=node_id))
        usage = {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14}
        return Result("all nodes", results=results, node_history=history, accumulated_usage=usage)


def _node(executor: Any, preset: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(executor=executor, _initial_messages=preset or [])


@pytest.fixture
def hook_event(monkeypatch: pytest.MonkeyPatch) -> type:
    class AfterNodeCallEvent:
        pass

    def fake_require(module: str, *, package: str, adapter: str) -> Any:
        assert (module, package, adapter) == ("strands.hooks", "strands-agents", "strands_adapter")
        return SimpleNamespace(AfterNodeCallEvent=AfterNodeCallEvent)

    monkeypatch.setattr(strands, "require", fake_require)
    return AfterNodeCallEvent


async def test_orchestrator_gets_the_transcript_and_reports_every_nodes_tools(hook_event: type) -> None:
    nested = Orchestrator({}, [])
    nodes = {
        "front": _node(SimpleNamespace(messages=[]), preset=[use("old", "preset_tool")]),
        "orders": _node(SimpleNamespace(messages=[])),
        "review": _node(nested),
    }
    plan = [
        ("front", [use("h1", "handoff_to_agent", {"agent_name": "orders"}), tool_result("h1", [{"text": "ok"}])], "a"),
        ("orders", [use("c1", "lookup", {"id": 7}), tool_result("c1", [{"text": "shipped"}])], "Order 7 shipped."),
        ("front", [use("c2", "close"), tool_result("c2", [{"text": "closed"}])], "b"),
        ("review", [], "Reviewed."),
    ]
    swarm = Orchestrator(nodes, plan)
    adapter = strands_adapter(lambda: swarm, invocation_state={"tenant": "a"})

    turn = await _call(adapter, _input([("user", "Where is order 7?")]))
    await _call(adapter, _input([("user", "Where is order 7?"), ("assistant", "Reviewed."), ("user", "Sure?")]))

    assert turn.content == "Reviewed."
    assert turn.tools is not None
    assert [(t.tool_name, t.result) for t in turn.tools] == [
        ("handoff_to_agent", "ok"),
        ("lookup", "shipped"),
        ("close", "closed"),
    ]
    assert turn.usage == Usage(input_tokens=10, output_tokens=4, total_tokens=14)
    assert swarm.tasks[0] == ("Where is order 7?", {"invocation_state": {"tenant": "a"}})
    assert "User: Where is order 7?\nAssistant: Reviewed.\nUser: Sure?" in swarm.tasks[1][0]
    # One hook per orchestrator, the nested one included, however many turns ran.
    assert swarm.hooks == [(strands._after_node, hook_event)]
    assert nested.hooks == [(strands._after_node, hook_event)]


async def test_orchestrator_reply_falls_back_to_the_whole_result_and_content_fn(hook_event: type) -> None:
    class Plain:
        async def invoke_async(self, task: str, **kwargs: Any) -> Any:
            usage = {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2, "turns_taken": 1}
            return Result("no node order", invocation_context=SimpleNamespace(usage=usage))

        def add_hook(self, callback: Any, event_type: Any) -> None:
            pass

    turn = await _call(strands_adapter(Plain()), _input([("user", "hi")]))
    assert (turn.content, turn.tools, turn.usage) == ("no node order", None, Usage(1, 1, 2, requests=1))

    adapter = strands_adapter(Plain(), content_fn=lambda result: result.reply.upper())
    assert (await _call(adapter, _input([("user", "hi")]))).content == "NO NODE ORDER"


def test_node_hook_ignores_runs_outside_the_adapter_and_unknown_nodes() -> None:
    event = SimpleNamespace(source=SimpleNamespace(nodes={}), node_id="x")
    strands._after_node(event)

    token = strands._NODE_RUNS.set([])
    try:
        strands._after_node(event)
        assert strands._NODE_RUNS.get() == []
    finally:
        strands._NODE_RUNS.reset(token)
