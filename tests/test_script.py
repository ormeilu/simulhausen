import dataclasses

import pytest

import simulhausen as sim
from simulhausen.script import AgentStep, FailStep, JudgeStep, MessageStep, ProceedStep, SucceedStep, UserStep


def test_user_and_agent_steps() -> None:
    assert sim.user() == UserStep(content=None)
    assert sim.user("hi") == UserStep(content="hi")
    assert sim.agent() == AgentStep(content=None)
    assert sim.agent("fixed") == AgentStep(content="fixed")


def test_message_step() -> None:
    assert sim.message("assistant", "earlier reply") == MessageStep(role="assistant", content="earlier reply")


def test_judge_step_without_criteria_is_final() -> None:
    assert sim.judge() == JudgeStep(criteria=None, additional_context=None)


def test_judge_step_with_criteria_is_a_checkpoint() -> None:
    step = sim.judge(["a", "b"], additional_context="db row: 42")
    assert step.criteria == ("a", "b")
    assert step.additional_context == "db row: 42"


def test_judge_rejects_empty_criteria() -> None:
    with pytest.raises(ValueError, match="nothing to check"):
        sim.judge([])


def test_proceed_step() -> None:
    def on_turn(state: sim.ScenarioState) -> None: ...

    assert sim.proceed() == ProceedStep(turns=None, on_turn=None, on_step=None)
    assert sim.proceed(3, on_turn=on_turn).on_turn is on_turn
    assert sim.proceed(turns=3).turns == 3


@pytest.mark.parametrize("turns", [0, -1])
def test_proceed_rejects_non_positive_turns(turns: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        sim.proceed(turns)


def test_succeed_and_fail_steps() -> None:
    assert sim.succeed() == SucceedStep(reasoning=None)
    assert sim.succeed("done") == SucceedStep(reasoning="done")
    assert sim.fail("broken") == FailStep(reasoning="broken")


def test_steps_are_immutable() -> None:
    step = sim.user("hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.content = "other"  # ty: ignore[invalid-assignment]
