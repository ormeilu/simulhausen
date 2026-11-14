"""``dspy_adapter`` against the real DSPy, with a scripted language model built on ``dspy.utils.DummyLM``.

DSPy is not a dependency of the project, so the module is skipped in a plain
test run. Run it with the library overlaid:

    uv run --no-group harbor --with dspy pytest tests/contract/test_dspy_contract.py --no-cov -q

The scripted model plugs into the engine of ``DummyLM`` (DSPy 3.5 and later), and the ReAct assertions are about
the ``prediction.history`` that release introduced.
"""

import dataclasses
import re
import threading
from typing import Any

import pytest

pytest.importorskip("dspy")
pytest.importorskip("dspy.clients.engines.dummy_engine")

import dspy
from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
from dspy.lm15 import Usage
from dspy.utils import DummyLM

import simulhausen as sim
from simulhausen import ScenarioState
from simulhausen.frameworks.dspy import dspy_adapter

QUESTION = re.compile(r"<question>\n(.*?)\n</question>", re.DOTALL)
OUTPUT = re.compile(r"<(\w+)>\n\{\1\}")


class _DeskEngine(DummyEngine):
    """Decides the answer from the request, lets ``DummyLM`` format it and reports 5 + 2 tokens."""

    def _complete_messages(self, messages: list[dict[str, Any]]) -> Any:
        owner = self.owner
        owner.seen.append(messages)
        last = messages[-1]["content"]
        asked = [
            question
            for message in messages
            if message["role"] == "user"
            for question in QUESTION.findall(message["content"])
        ]
        answer = "users: " + " | ".join(asked)
        if "<tool_calls>" not in last:
            # Whatever the signature asks for besides the answer (ChainOfThought's reasoning or plan) gets a filler.
            fields: dict[str, Any] = {name: "thinking" for name in OUTPUT.findall(last) if name != "answer"}
            fields["answer"] = answer
        else:
            current = QUESTION.findall(last)
            prompt = current[-1] if current else ""
            if "order" in prompt and "User:" not in prompt:
                call = {"name": "lookup_order", "args": {"order_id": 7}}
            elif "refund" in prompt:
                call = {"name": "refund", "args": {"order_id": 7}}
            else:
                call = {"name": "submit", "args": {"answer": answer}}
            fields = {"next_thought": "next", "tool_calls": {"tool_calls": [call]}}
        owner.answers = iter([fields])
        response = super()._complete_messages(messages)
        return dataclasses.replace(response, usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7))


class Desk(DummyLM):
    """A scripted model: ReAct gets a tool call when the user names one, everything else gets what it was shown."""

    def __init__(self) -> None:
        super().__init__([])
        self.seen: list[list[dict[str, Any]]] = []
        self._engine_spec = _DeskEngine(self)
        self._async_engine_spec = AsyncDummyEngine(self._engine_spec)


def lookup_order(order_id: int) -> str:
    """Look an order up."""
    return f"order {order_id} is shipped"


def refund(order_id: int) -> str:
    """Refund an order."""
    raise RuntimeError("refunds are closed")


def _react() -> dspy.ReAct:
    return dspy.ReAct("question -> answer", tools=[lookup_order, refund])


def _tool_results(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message["role"] == "tool")


async def _play(model: Desk, adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    with dspy.context(lm=model, track_usage=True):
        return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


async def test_reply_tool_call_memory_and_usage_in_session_mode() -> None:
    model = Desk()
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(model, dspy_adapter(_react), script)

    state = states[0]
    assert result.messages[1]["content"] == "users: where is my order?"
    assert state.has_tool_call("lookup_order")
    call = state.tool_calls("lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "order 7 is shipped", 0)
    # The submit step that carries the answer is not a tool call.
    assert state.tool_call_names() == ["lookup_order"]
    # Second turn: ReAct is given the history of the first one, its tool call and tool result included.
    assert result.messages[3]["content"] == "users: where is my order? | thanks"
    assert [message["role"] for message in model.seen[-1]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert state.tool_calls(turn=1) == []
    # Three model requests over the two turns, 5 + 2 tokens each; DSPy reports tokens, not requests.
    assert result.usage == sim.Usage(input_tokens=15, output_tokens=6, total_tokens=21)


async def test_a_raising_tool_is_an_error() -> None:
    states: list[ScenarioState] = []

    await _play(Desk(), dspy_adapter(_react()), [sim.user("refund it"), sim.agent(), states.append])

    state = states[0]
    assert not state.has_tool_call("refund")
    assert state.tool_calls() == []
    (failed,) = state.tool_calls(include_errors=True)
    assert (failed.name, failed.input, failed.status) == ("refund", {"order_id": 7}, "error")
    assert failed.output is not None
    assert failed.output.startswith("Execution error in refund:")
    assert "refunds are closed" in failed.output


def _forget_the_secret(state: ScenarioState) -> None:
    state.rollback_messages_to(2)


ROLLBACK_SCRIPT = [
    sim.user("hello"),
    sim.agent(),
    sim.user("my secret is 42"),
    sim.agent(),
    _forget_the_secret,
    sim.user("what do you know?"),
    sim.agent(),
]


@pytest.mark.parametrize("history", ["transcript", "session"])
async def test_a_rollback_reaches_the_agent(history: Any) -> None:
    model = Desk()

    result = await _play(model, dspy_adapter(_react(), history=history), ROLLBACK_SCRIPT)

    assert "secret" not in str(model.seen[-1])
    # The transcript that is left is rendered into the question.
    assert "User: hello\nAssistant: users: hello\nUser: what do you know?" in result.messages[-1]["content"]


async def test_transcript_mode_does_not_replay_tool_results() -> None:
    model = Desk()
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent()]

    await _play(model, dspy_adapter(_react(), history="transcript"), script)

    assert _tool_results(model.seen[-1]) == 0
    assert "where is my order?" in str(model.seen[-1])


async def test_one_adapter_serves_scenarios_without_sharing_memory() -> None:
    model = Desk()
    adapter = dspy_adapter(_react())

    await _play(model, adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(model, adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "users: what do you know?"
    assert "secret" not in str(model.seen[-1])


class Chat(dspy.Signature):
    """Answer the customer."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


@pytest.mark.parametrize("history", ["transcript", "session"])
async def test_a_history_input_is_filled_from_the_transcript(history: Any) -> None:
    model = Desk()
    script = [sim.user("hello"), sim.agent(), sim.user("more"), sim.agent()]

    result = await _play(model, dspy_adapter(dspy.Predict(Chat), history=history), script)

    assert result.messages[1]["content"] == "users: hello"
    # The earlier exchange arrives as real chat turns, not as text inside the question.
    assert result.messages[3]["content"] == "users: hello | more"
    assert [message["role"] for message in model.seen[-1]] == ["system", "user", "assistant", "user"]
    assert result.usage == sim.Usage(input_tokens=10, output_tokens=4, total_tokens=14)


async def test_chain_of_thought_answers_with_its_answer_field() -> None:
    model = Desk()
    script = [sim.user("hello"), sim.agent(), sim.user("more"), sim.agent()]

    result = await _play(model, dspy_adapter(dspy.ChainOfThought("question -> answer")), script)

    assert result.messages[1]["content"] == "users: hello"
    # No history input: the second question holds the rendered transcript.
    assert "User: hello\nAssistant: users: hello\nUser: more" in result.messages[3]["content"]


async def test_a_module_with_only_forward_runs_in_a_worker_thread() -> None:
    threads: list[str] = []

    class Wrapper(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question: str) -> dspy.Prediction:
            threads.append(threading.current_thread().name)
            return self.predict(question=question)

    result = await _play(Desk(), dspy_adapter(Wrapper), [sim.user("hello"), sim.agent()])

    assert result.messages[-1]["content"] == "users: hello"
    assert threads != [threading.current_thread().name]
