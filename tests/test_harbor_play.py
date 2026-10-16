from typing import Any, cast

import pytest
from agno.models.base import Model
from harbor_support import TARGET, ScriptedSimulator, goal_reached

from simulhausen import SimulationStructuredOutputError, UserSimulatorAgent, collector
from simulhausen.harbor import ConfigError, ScenarioSpec
from simulhausen.harbor import play as play_module
from simulhausen.harbor.play import build_simulator, play


def _spec(script: list[dict[str, Any]] | None = None, **fields: Any) -> ScenarioSpec:
    payload: dict[str, Any] = {"id": "refund", "name": "refund", "description": "Charged twice.", "target": TARGET}
    if script is not None:
        payload["script"] = script
    return ScenarioSpec.model_validate({**payload, **fields})


@pytest.fixture
def no_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(spec: ScenarioSpec, model: str | None) -> UserSimulatorAgent:
        raise AssertionError("a fully scripted scenario needs no simulator")

    monkeypatch.setattr(play_module, "build_simulator", refuse)


@pytest.fixture
def simulator(monkeypatch: pytest.MonkeyPatch) -> list[ScriptedSimulator]:
    """Makes play() build ScriptedSimulator(*turns) for the turns the test appends first."""
    built: list[ScriptedSimulator] = []

    def build(spec: ScenarioSpec, model: str | None) -> UserSimulatorAgent:
        return built[0]

    monkeypatch.setattr(play_module, "build_simulator", build)
    return built


async def test_a_scripted_dialogue_records_messages_checkpoints_and_tools(no_simulator: None) -> None:
    spec = _spec(
        [
            {"user": "i was charged twice"},
            {"agent": True},
            {"judge": {"criteria": ["asks for the order"]}},
            {"message": {"role": "assistant", "content": "seeded"}},
            {"check": {"type": "tool_called", "tool": "lookup_order"}},
            {"user": "thanks"},
            {"agent": "injected reply"},
            {"judge": True},
        ]
    )
    target = {**TARGET, "prefix": "bot"}

    transcript = await play(spec, target)

    assert [(m.role, m.content) for m in transcript.messages] == [
        ("user", "i was charged twice"),
        ("assistant", "bot reply 1: i was charged twice"),
        ("assistant", "seeded"),
        ("user", "thanks"),
        ("assistant", "injected reply"),
    ]
    assert [(c.kind, c.step, c.message_count, c.criteria) for c in transcript.checkpoints] == [
        ("judge", 2, 2, []),
        ("check", 4, 3, []),
    ]
    assert transcript.stop_reason == "judge_step"
    assert transcript.turns == 2
    assert [turn.message_index for turn in transcript.agent_turns] == [1]
    assert transcript.agent_turns[0].tools[0]["tool_name"] == "lookup_order"
    assert transcript.agent_turns[0].references[0]["query"] == "i was charged twice"
    assert transcript.target == target
    assert transcript.scenario.id == "refund"
    assert transcript.finished
    assert transcript.total_time >= transcript.agent_time > 0
    # The unjudged play result does not stay in the report collector.
    assert collector.records == []


async def test_the_target_is_recorded_without_secrets(no_simulator: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_KEY", "from-env")
    spec = _spec([{"user": "hi"}, {"agent": True}])

    literal = await play(spec, {**TARGET, "api_key": "s3cret"})
    indirect = await play(spec, {**TARGET, "api_key_env": "AGENT_KEY"})

    assert literal.messages[1].content == "agent (s3cret) reply 1: hi"
    assert literal.target == TARGET
    assert indirect.messages[1].content == "agent (from-env) reply 1: hi"
    assert indirect.target == {**TARGET, "api_key_env": "AGENT_KEY"}
    assert literal.stop_reason == "script_end"


async def test_without_a_script_the_dialogue_runs_until_the_goal_is_reached(simulator: list[ScriptedSimulator]) -> None:
    simulator.append(ScriptedSimulator("first", "second", goal_reached()))

    transcript = await play(_spec(persona="terse"), TARGET)

    assert [m.content for m in transcript.messages if m.role == "user"] == ["first", "second"]
    assert transcript.turns == 2
    assert transcript.stop_reason == "goal_achieved"


async def test_proceed_that_hits_max_turns(simulator: list[ScriptedSimulator]) -> None:
    simulator.append(ScriptedSimulator())

    transcript = await play(_spec(max_turns=2), TARGET)

    assert transcript.turns == 2
    assert transcript.stop_reason == "max_turns"


async def test_proceed_that_plays_its_turns_ends_with_the_script(simulator: list[ScriptedSimulator]) -> None:
    simulator.append(ScriptedSimulator())

    transcript = await play(_spec([{"proceed": {"turns": 2}}]), TARGET)

    assert transcript.turns == 2
    assert transcript.stop_reason == "script_end"


async def test_steps_after_a_proceed_decide_the_stop_reason(simulator: list[ScriptedSimulator]) -> None:
    simulator.append(ScriptedSimulator("a", goal_reached(), "b", goal_reached()))

    checked = await play(_spec([{"proceed": True}, {"check": {"type": "tool_called", "tool": "x"}}]), TARGET)
    continued = await play(_spec([{"proceed": True}, {"user": "one more thing"}, {"agent": True}]), TARGET)

    assert checked.stop_reason == "goal_achieved"
    assert continued.stop_reason == "script_end"


async def test_a_simulated_user_step_builds_the_simulator(simulator: list[ScriptedSimulator]) -> None:
    simulator.append(ScriptedSimulator("written by the simulator"))

    transcript = await play(_spec([{"user": True}, {"agent": True}]), TARGET)

    assert transcript.messages[0].content == "written by the simulator"


async def test_an_explicit_simulator_wins(no_simulator: None) -> None:
    transcript = await play(_spec([{"user": True}]), TARGET, simulator=ScriptedSimulator("mine"))

    assert transcript.messages[0].content == "mine"


async def test_a_harness_failure_is_recorded_with_the_dialogue_so_far(simulator: list[ScriptedSimulator]) -> None:
    simulator.append(ScriptedSimulator("first", SimulationStructuredOutputError("no structured output")))

    transcript = await play(_spec(), TARGET)

    assert transcript.harness_error
    assert transcript.stop_reason == "harness_error"
    assert transcript.error == "SimulationStructuredOutputError: no structured output"
    assert [m.content for m in transcript.messages] == ["first", "agent reply 1: first"]
    assert not transcript.finished
    assert collector.records == []


async def test_an_error_of_the_agent_under_test_is_recorded(no_simulator: None) -> None:
    spec = _spec([{"user": "hello"}, {"agent": True}, {"user": "explode"}, {"agent": True}])

    transcript = await play(spec, {**TARGET, "fail_on": "explode"})

    assert not transcript.harness_error
    assert transcript.stop_reason == "error"
    assert transcript.error == "RuntimeError: agent crashed on 'explode'"
    assert transcript.turns == 1
    assert len(transcript.messages) == 3


async def test_configuration_errors_raise_before_anything_is_played() -> None:
    with pytest.raises(ConfigError, match="unknown target type"):
        await play(_spec([{"user": "x"}]), {"type": "nope"})


def test_build_simulator_uses_the_scenario_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    model = cast("Model", object())
    asked: list[tuple[str, str | None, str | None]] = []

    def fake_resolve(role: str, cli_value: str | None, scenario_value: str | None) -> Model:
        asked.append((role, cli_value, scenario_value))
        return model

    monkeypatch.setattr(play_module, "resolve_model", fake_resolve)
    spec = _spec(persona="terse", simulator_prompt="You are a customer.", simulator_model="openai:gpt-4.1-mini")

    built = build_simulator(spec, "cli:model")

    assert asked == [("simulator", "cli:model", "openai:gpt-4.1-mini")]
    assert built.persona == "terse"
    assert built.system_prompt == "You are a customer."
    assert built._model is model
