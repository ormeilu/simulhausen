"""Agent phase: play a scenario's dialogue with the agent under test, without grading it.

The dialogue loop is :func:`simulhausen.run` itself, with the agent under test and the
user simulator but no judge: the criteria stay with the verifier, which grades the
recorded dialogue later. The scenario's script is translated into simulhausen script
steps, and a few callables between them watch the run: one keeps the live
:class:`~simulhausen.ScenarioState` (the runner mutates that one object throughout, so
it survives an exception), checkpoint steps note how many messages the dialogue had when
they were reached, and the proceed steps note why they stopped, which becomes the
transcript's ``stop_reason``.

Without a judge nothing ends a ``proceed`` early except the simulator reporting the goal
reached, so a dialogue ends there, at ``max_turns``, at a ``judge = true`` step or at the
end of the script.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from simulhausen import __version__
from simulhausen.agents import UserSimulatorAgent
from simulhausen.config import get_config
from simulhausen.exceptions import SimulationHarnessError
from simulhausen.harbor.models import resolve_model
from simulhausen.harbor.targets import build_target, redact_target
from simulhausen.harbor.transcript import Checkpoint, ScenarioRef, build_transcript
from simulhausen.reporting import collector
from simulhausen.runner import run
from simulhausen.script import agent, message, proceed, user

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simulhausen.harbor.spec import ScenarioSpec, ScriptStepSpec
    from simulhausen.harbor.transcript import StopReason, Transcript
    from simulhausen.script import ScriptStep
    from simulhausen.types import ScenarioState

__all__ = ["build_simulator", "play"]


def build_simulator(spec: ScenarioSpec, model: str | None) -> UserSimulatorAgent:
    """The user simulator for ``spec``, on the model ``--model`` or the environment names.

    Module-level so tests (and embedders) can swap in a simulator that needs no model.
    """
    return UserSimulatorAgent(
        resolve_model("simulator", model, spec.simulator_model),
        persona=spec.persona,
        system_prompt=spec.simulator_prompt,
    )


class _Recorder:
    """Builds the runner script for a scenario and records what happens while it runs."""

    def __init__(self, spec: ScenarioSpec, max_turns: int) -> None:
        self.spec = spec
        self.max_turns = max_turns
        self.state: ScenarioState | None = None
        self.checkpoints: list[Checkpoint] = []
        self.stop_reason: StopReason = "script_end"

    def script(self) -> list[ScriptStep]:
        steps: list[ScriptStep] = [self._capture]
        for index, step in enumerate(self.spec.steps):
            steps.extend(self._translate(index, step))
        return steps

    def _capture(self, state: ScenarioState) -> None:
        self.state = state

    def _message_added(self, state: ScenarioState) -> None:
        # A later message step means the dialogue did not end where a proceed stopped.
        self.stop_reason = "script_end"

    def _translate(self, index: int, step: ScriptStepSpec) -> list[ScriptStep]:
        if step.user is not None:
            return [user(None if step.user is True else step.user), self._message_added]
        if step.agent is not None:
            return [agent(None if step.agent is True else step.agent), self._message_added]
        if step.message is not None:
            return [message(step.message.role, step.message.content), self._message_added]
        if step.proceed is not None:
            return self._proceed(None if step.proceed is True else step.proceed.turns)
        if step.is_final_judge:
            # Always the last step (the loader enforces it), so the script ends right here.
            return [self._final_judge]
        return [self._checkpoint(index, "judge" if step.judge is not None else "check")]

    def _proceed(self, turns: int | None) -> list[ScriptStep]:
        start = 0

        def before(state: ScenarioState) -> None:
            nonlocal start
            start = state.current_turn

        def after(state: ScenarioState) -> None:
            if turns is not None and state.current_turn - start >= turns:
                self.stop_reason = "script_end"
            elif state.current_turn >= self.max_turns:
                self.stop_reason = "max_turns"
            else:
                # No judge in play, so only the simulator's goal signal stops a proceed early.
                self.stop_reason = "goal_achieved"

        return [before, proceed(turns), after]

    def _checkpoint(self, index: int, kind: Literal["judge", "check"]) -> ScriptStep:
        def record(state: ScenarioState) -> None:
            self.checkpoints.append(Checkpoint(kind=kind, step=index, message_count=len(state.messages)))

        return record

    def _final_judge(self, state: ScenarioState) -> None:
        self.stop_reason = "judge_step"


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def play(
    spec: ScenarioSpec,
    target: Mapping[str, Any],
    *,
    model: str | None = None,
    simulator: UserSimulatorAgent | None = None,
) -> Transcript:
    """Play ``spec`` against the agent ``target`` describes and return the transcript.

    The simulator is built only when the script needs one (``user = true`` or a
    ``proceed`` step), on ``model`` or the model the environment or the scenario names;
    pass ``simulator`` to supply your own. Configuration problems (the target, the
    model) raise :class:`~simulhausen.harbor.ConfigError` before anything is played.
    Once the dialogue runs, nothing raises: a harness failure or an error of the agent
    under test ends up in the transcript (``harness_error``, ``error``), next to the
    messages played until then.
    """
    adapter = build_target(target)
    if simulator is None and spec.uses_simulator():
        simulator = build_simulator(spec, model)
    recorder = _Recorder(spec, spec.max_turns or get_config().max_turns)
    agents: list[object] = [adapter] if simulator is None else [adapter, simulator]
    first_record = len(collector.records)
    harness_error = False
    error: str | None = None
    started = time.perf_counter()
    try:
        await run(
            spec.name,
            spec.description,
            agents,
            recorder.script(),
            max_turns=spec.max_turns,
            min_turns=spec.min_turns,
        )
    except SimulationHarnessError as exc:
        harness_error, error = True, _describe(exc)
        recorder.stop_reason = "harness_error"
    except Exception as exc:  # noqa: BLE001 - the agent under test may raise anything; the transcript records it
        error = _describe(exc)
        recorder.stop_reason = "error"
    total_time = time.perf_counter() - started
    # run() reports every scenario to the collector. This one has no verdict yet: drop it,
    # the judge phase reports the graded result.
    agent_time = sum(record.result.agent_time for record in collector.records[first_record:])
    del collector.records[first_record:]
    return build_transcript(
        recorder.state,
        scenario=ScenarioRef(id=spec.id, name=spec.name, description=spec.description),
        simulhausen_version=__version__,
        checkpoints=recorder.checkpoints,
        stop_reason=recorder.stop_reason,
        harness_error=harness_error,
        error=error,
        total_time=total_time,
        agent_time=agent_time,
        target=redact_target(target),
    )
