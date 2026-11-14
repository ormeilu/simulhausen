"""``smolagents_adapter`` against hand-written fakes shaped like smolagents agents, tools and memory steps."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from simulhausen import AgentInput, AgentTurn, Usage
from simulhausen.frameworks.smolagents import smolagents_adapter


class FakeTool:
    """Shaped like ``smolagents.Tool``: a name, named inputs and a call that takes the loop's keyword."""

    def __init__(self, name: str, fn: Callable[..., Any], inputs: tuple[str, ...] = ("order_id",)) -> None:
        self.name = name
        self.inputs = {key: {"type": "string"} for key in inputs}
        self.description = f"the {name} tool"
        self._fn = fn

    def __call__(self, *args: Any, sanitize_inputs_outputs: bool = False, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


def _call(name: str, arguments: Any, call_id: str = "c1") -> Any:
    return SimpleNamespace(name=name, arguments=arguments, id=call_id)


def _tokens(input_tokens: int = 10, output_tokens: int = 5) -> Any:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class FakeAgent:
    """Shaped like a smolagents agent: ``run`` plays one scripted plan per call and fills ``memory.steps``."""

    def __init__(
        self,
        *plans: Callable[["FakeAgent"], Any],
        tools: tuple[FakeTool, ...] = (),
        managed: tuple["FakeAgent", ...] = (),
        name: str | None = None,
    ) -> None:
        self.tools: dict[str, Any] = {tool.name: tool for tool in tools}
        self.tools["final_answer"] = FakeTool("final_answer", lambda answer: answer, inputs=("answer",))
        self.managed_agents: dict[str, Any] = {str(agent.name): agent for agent in managed}
        self.memory = SimpleNamespace(steps=[])
        self.name = name
        self.inputs = {"task": {}, "additional_args": {}}
        self.plans = list(plans)
        self.runs: list[tuple[str, bool, dict[str, Any]]] = []

    def run(self, task: str, reset: bool = True, **kwargs: Any) -> Any:
        self.runs.append((task, reset, kwargs))
        if reset:
            self.memory.steps = []
        self.memory.steps.append(SimpleNamespace(task=task))
        return self.plans.pop(0)(self)

    def __call__(self, task: str, **kwargs: Any) -> str:
        return f"report: {self.run(task)}"

    def step(self, **fields: Any) -> None:
        """Append an ``ActionStep``-shaped memory step."""
        defaults: dict[str, Any] = {
            "tool_calls": None,
            "error": None,
            "model_output_message": None,
            "code_action": None,
            "observations": None,
            "is_final_answer": False,
            "token_usage": None,
        }
        self.memory.steps.append(SimpleNamespace(**{**defaults, **fields}))

    def answer(self, text: str, **fields: Any) -> str:
        self.tools["final_answer"](answer=text)
        self.step(tool_calls=[_call("final_answer", {"answer": text}, "final")], is_final_answer=True, **fields)
        return text


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


def _lookup() -> FakeTool:
    return FakeTool("lookup_order", lambda order_id: f"order {order_id} is shipped")


def _summary(turn: AgentTurn) -> list[tuple[Any, ...]]:
    return [(t.tool_name, t.tool_args, t.result, bool(t.tool_call_error), t.tool_call_id) for t in turn.tools or []]


async def _turn(adapter: Any, agent_input: AgentInput) -> AgentTurn:
    turn = await adapter.call(agent_input)
    assert isinstance(turn, AgentTurn)
    return turn


async def test_a_tool_call_is_recorded_with_arguments_result_id_and_usage() -> None:
    def plan(agent: FakeAgent) -> str:
        result = agent.tools["lookup_order"](order_id="1042", sanitize_inputs_outputs=True)
        agent.step(tool_calls=[_call("lookup_order", {"order_id": "1042"})], observations=result, token_usage=_tokens())
        return agent.answer("shipped", token_usage=_tokens(20, 7))

    agent = FakeAgent(plan, tools=(_lookup(),))
    originals = dict(agent.tools)
    turn = await _turn(smolagents_adapter(agent, max_steps=4), _input([("user", "where is it?")]))

    assert turn.content == "shipped"
    assert _summary(turn) == [("lookup_order", {"order_id": "1042"}, "order 1042 is shipped", False, "c1")]
    assert turn.usage == Usage(input_tokens=30, output_tokens=12, total_tokens=42, requests=2)
    assert agent.runs == [("where is it?", True, {"max_steps": 4})]
    # The stand-ins are gone and final_answer never was one.
    assert agent.tools == originals


async def test_session_mode_keeps_the_memory_and_reports_only_the_new_steps() -> None:
    def first(agent: FakeAgent) -> str:
        agent.tools["lookup_order"]("1042")
        agent.step(tool_calls=[_call("lookup_order", {"order_id": "1042"})])
        return agent.answer("shipped")

    def second(agent: FakeAgent) -> str:
        return agent.answer("you are welcome")

    agent = FakeAgent(first, second, tools=(_lookup(),))
    adapter = smolagents_adapter(agent)
    opening = await _turn(adapter, _input([("user", "where is it?")]))
    follow_up = await _turn(adapter, _input([("user", "where is it?"), ("assistant", "shipped"), ("user", "thanks")]))

    assert [name for name, *_ in _summary(opening)] == ["lookup_order"]
    assert follow_up.tools is None
    assert follow_up.usage is None
    assert agent.runs == [("where is it?", True, {}), ("thanks", False, {})]
    assert len(agent.memory.steps) == 5


async def test_session_mode_starts_over_after_a_rollback_and_transcript_mode_always_does() -> None:
    agent = FakeAgent(*[lambda a: a.answer("ok")] * 2)
    adapter = smolagents_adapter(agent)
    await _turn(adapter, _input([("user", "my code is 7421")]))
    await _turn(adapter, _input([("user", "hello again")]))
    assert agent.runs == [("my code is 7421", True, {}), ("hello again", True, {})]

    agent = FakeAgent(*[lambda a: a.answer("ok")] * 2)
    adapter = smolagents_adapter(agent, history="transcript")
    await _turn(adapter, _input([("user", "one")]))
    await _turn(adapter, _input([("user", "one"), ("assistant", "ok"), ("user", "two")]))
    task, reset, _ = agent.runs[1]
    assert reset is True
    assert "User: one\nAssistant: ok\nUser: two" in task


async def test_a_factory_builds_one_agent_per_thread_and_a_shared_agent_starts_over() -> None:
    built: list[FakeAgent] = []

    def factory() -> FakeAgent:
        built.append(FakeAgent(*[lambda a: a.answer("ok")] * 2))
        return built[-1]

    adapter = smolagents_adapter(factory)
    await _turn(adapter, _input([("user", "a")], thread_id="one"))
    await _turn(adapter, _input([("user", "b")], thread_id="two"))
    await _turn(adapter, _input([("user", "a"), ("assistant", "ok"), ("user", "more")], thread_id="one"))
    assert [agent.runs for agent in built] == [[("a", True, {}), ("more", False, {})], [("b", True, {})]]

    shared = FakeAgent(*[lambda a: a.answer("ok")] * 3)
    adapter = smolagents_adapter(shared)
    await _turn(adapter, _input([("user", "a")], thread_id="one"))
    await _turn(adapter, _input([("user", "b")], thread_id="two"))
    # The instance now holds thread "two": thread "one" is rebuilt from its transcript.
    await _turn(adapter, _input([("user", "a"), ("assistant", "ok"), ("user", "more")], thread_id="one"))
    assert [reset for _, reset, _ in shared.runs] == [True, True, True]
    assert "User: a\nAssistant: ok\nUser: more" in shared.runs[2][0]


async def test_a_raising_tool_is_an_error_and_a_call_that_reached_no_tool_comes_from_the_step() -> None:
    def refund(order_id: str) -> str:
        raise ValueError(f"order {order_id} is not refundable")

    def plan(agent: FakeAgent) -> str:
        requested = SimpleNamespace(function=SimpleNamespace(name="refund", arguments={"order_id": "7"}), id="c7")
        try:
            agent.tools["refund"](order_id="7")
        except ValueError as exc:
            # The real ToolCallingAgent leaves tool_calls unset when a tool raises.
            agent.step(error=exc, model_output_message=SimpleNamespace(tool_calls=[requested]))
        agent.step(tool_calls=[_call("no_such_tool", '{"x": 1}', "c8")], error=RuntimeError("Unknown tool"))
        agent.step(error=RuntimeError("could not parse the model output"))
        return agent.answer("sorry")

    turn = await _turn(
        smolagents_adapter(FakeAgent(plan, tools=(FakeTool("refund", refund),))), _input([("user", "refund")])
    )

    assert _summary(turn) == [
        ("refund", {"order_id": "7"}, "order 7 is not refundable", True, "c7"),
        ("no_such_tool", {"x": 1}, "Unknown tool", True, "c8"),
    ]


async def test_code_steps_report_real_tools_or_the_interpreter_call() -> None:
    def plan(agent: FakeAgent) -> str:
        # The stand-in passes every attribute of the tool through.
        assert agent.tools["lookup_order"].description == "the lookup_order tool"
        # Generated code calls tools positionally, with a dict, and with more arguments than inputs.
        agent.tools["lookup_order"]("1042")
        agent.tools["lookup_order"]({"order_id": "7"})
        agent.tools["echo"]("a", "b", flag=True)
        agent.step(tool_calls=[_call("python_interpreter", "code-1", "call_1")], code_action="code-1")
        agent.step(
            tool_calls=[_call("python_interpreter", "print(6 * 7)", "call_2")],
            code_action="print(6 * 7)",
            observations="Execution logs:\n42",
        )
        agent.step(
            tool_calls=[_call("python_interpreter", "1 / 0", "call_3")],
            code_action="1 / 0",
            error=ZeroDivisionError("division by zero"),
        )
        agent.step(
            tool_calls=[_call("python_interpreter", "final_answer('done')", "call_4")],
            code_action="final_answer('done')",
            is_final_answer=True,
        )
        return "done"

    echo = FakeTool("echo", lambda *args, **kwargs: {"args": list(args), **kwargs}, inputs=("text",))
    turn = await _turn(smolagents_adapter(FakeAgent(plan, tools=(_lookup(), echo))), _input([("user", "go")]))

    assert _summary(turn) == [
        ("lookup_order", {"order_id": "1042"}, "order 1042 is shipped", False, None),
        ("lookup_order", {"order_id": "7"}, "order {'order_id': '7'} is shipped", False, None),
        ("echo", {"text": "a", "arg1": "b", "flag": True}, '{"args": ["a", "b"], "flag": true}', False, None),
        ("python_interpreter", {"code": "print(6 * 7)"}, "Execution logs:\n42", False, "call_2"),
        ("python_interpreter", {"code": "1 / 0"}, "division by zero", True, "call_3"),
    ]


async def test_a_managed_agent_is_a_call_and_its_tools_and_tokens_count() -> None:
    def clerk_plan(agent: FakeAgent) -> str:
        agent.tools["lookup_order"](order_id="1042")
        agent.step(tool_calls=[_call("lookup_order", {"order_id": "1042"}, "k1")], token_usage=_tokens(3, 1))
        return agent.answer("found", token_usage=_tokens(4, 2))

    def manager_plan(agent: FakeAgent) -> str:
        agent.managed_agents["clerk"](task="find it")
        agent.step(tool_calls=[_call("clerk", {"task": "find it"}, "m1")], token_usage=_tokens())
        return agent.answer("done", token_usage=_tokens())

    clerk = FakeAgent(clerk_plan, tools=(_lookup(),), name="clerk")
    manager = FakeAgent(manager_plan, managed=(clerk,))
    turn = await _turn(smolagents_adapter(lambda: manager), _input([("user", "go")]))

    assert _summary(turn) == [
        ("clerk", {"task": "find it"}, "report: found", False, "m1"),
        ("lookup_order", {"order_id": "1042"}, "order 1042 is shipped", False, None),
    ]
    assert turn.usage == Usage(input_tokens=27, output_tokens=13, total_tokens=40, requests=4)
    assert not hasattr(manager.managed_agents["clerk"], "_wrapped")
    assert not hasattr(clerk.tools["lookup_order"], "_wrapped")


async def test_a_run_result_is_unwrapped_and_content_fn_renders_the_output() -> None:
    def plan(agent: FakeAgent) -> Any:
        return SimpleNamespace(output={"total": 3}, token_usage=None)

    adapter = smolagents_adapter(FakeAgent(plan), content_fn=lambda output: f"total={output['total']}")
    assert (await _turn(adapter, _input([("user", "sum")]))).content == "total=3"

    plain = smolagents_adapter(FakeAgent(lambda a: {"total": 3}))
    assert (await _turn(plain, _input([("user", "sum")]))).content == '{"total": 3}'


async def test_a_failing_run_propagates_and_puts_the_tools_back() -> None:
    def plan(agent: FakeAgent) -> str:
        raise RuntimeError("model is down")

    agent = FakeAgent(plan, tools=(_lookup(),))
    originals = dict(agent.tools)
    with pytest.raises(RuntimeError, match="model is down"):
        await smolagents_adapter(agent).call(_input([("user", "hi")]))
    assert agent.tools == originals


def test_history_is_validated() -> None:
    with pytest.raises(ValueError, match="history must be"):
        smolagents_adapter(FakeAgent(), history="nope")  # ty: ignore[invalid-argument-type]
