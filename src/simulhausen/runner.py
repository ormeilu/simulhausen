"""Scenario runner: plays the dialogue from a script or on its own, then judges it."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from simulhausen.agents import JudgeAgent, UserSimulatorAgent, UserTurn
from simulhausen.config import get_config
from simulhausen.exceptions import SimulationHarnessError
from simulhausen.reporting import collector
from simulhausen.script import (
    AgentStep,
    FailStep,
    JudgeStep,
    MessageStep,
    ProceedStep,
    SucceedStep,
    UserStep,
    proceed,
)
from simulhausen.types import AgentAdapter, AgentInput, AgentTurn, ScenarioResult, ScenarioState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from simulhausen.script import ScriptStep, StateCallback
    from simulhausen.types import CriterionVerdict, JudgeScore

__all__ = ["run"]

logger = logging.getLogger(__name__)


async def _call(callback: StateCallback | None, state: ScenarioState) -> None:
    if callback is None:
        return
    outcome = callback(state)
    if inspect.isawaitable(outcome):
        await outcome


class _Simulation:
    """One scenario run: the participants and the dialogue state."""

    def __init__(
        self,
        name: str,
        description: str,
        agents: Sequence[object],
        *,
        max_turns: int,
        min_turns: int,
        verbose: bool,
        debug: bool,
    ) -> None:
        adapters = [a for a in agents if isinstance(a, AgentAdapter)]
        if len(adapters) != 1:
            raise ValueError(f"agents needs exactly one AgentAdapter (the agent under test), got {len(adapters)}")
        self.adapter = adapters[0]
        provided_simulator = next((a for a in agents if isinstance(a, UserSimulatorAgent)), None)
        self.simulator_provided = provided_simulator is not None
        self.simulator = provided_simulator or UserSimulatorAgent()
        self.judge = next((a for a in agents if isinstance(a, JudgeAgent)), None)
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.verbose = verbose
        self.debug = debug
        self.state = ScenarioState(thread_id=str(uuid.uuid4()), name=name, description=description)
        # Checkpoint verdicts accumulate; the final verdict is replaced each time
        # the judge rules on its own criteria, so it always reflects one dialogue.
        self.checkpoint_verdicts: list[CriterionVerdict] = []
        self.final_verdicts: list[CriterionVerdict] = []
        self.judge_score: JudgeScore | None = None
        self.reasoning: str | None = None
        # Set by succeed()/fail() and by a failed checkpoint: the scenario is over.
        self.forced_outcome: bool | None = None
        self.failure_reason: str | None = None
        # Transcript length at the last final verdict; None until there is one.
        self.final_judged_at: int | None = None
        self.agent_time = 0.0
        # Length of the transcript the last time the agent spoke (for new_messages).
        self._seen = 0

    @property
    def ended(self) -> bool:
        return self.forced_outcome is not None

    @property
    def verdicts(self) -> list[CriterionVerdict]:
        return self.checkpoint_verdicts + self.final_verdicts

    # -- single steps -----------------------------------------------------

    def _log(self, event: str, text: str) -> None:
        if self.verbose:
            logger.info("[%s] %s: %s", self.state.name, event, text)

    async def _human_message(self) -> str | None:
        """Debug mode: ask on stdin for the next user message; ``None`` lets the simulator write it."""
        last = self.state.last_agent_message_str()
        prompt = f"\n[{self.state.name}] agent: {last}\n" if last else f"\n[{self.state.name}] (dialogue start)\n"
        prompt += "your message (Enter = simulator): "
        typed = await asyncio.to_thread(input, prompt)
        return typed.strip() or None

    async def _simulated_turn(self) -> UserTurn:
        if self.debug and (typed := await self._human_message()) is not None:
            return UserTurn(message=typed, goal_achieved=False)
        return await self.simulator.next_turn(self.state)

    async def user_turn(self, content: str | None) -> None:
        if content is None:
            typed = await self._human_message() if self.debug else None
            content = typed or await self.simulator.next_message(self.state)
        self.state.add_user_message(content)
        self._log("user", content)

    async def agent_turn(self, content: str | None = None) -> None:
        if content is not None:
            turn = AgentTurn(content=content)
        else:
            agent_input = AgentInput(
                thread_id=self.state.thread_id,
                messages=list(self.state.messages),
                new_messages=list(self.state.messages[self._seen :]),
                scenario_state=self.state,
            )
            started = time.perf_counter()
            try:
                response = await self.adapter.call(agent_input)
            finally:
                self.agent_time += time.perf_counter() - started
            turn = response if isinstance(response, AgentTurn) else AgentTurn(content=str(response or ""))
        self.state.add_agent_message(turn.content, tools=turn.tools, references=turn.references)
        self.state.current_turn += 1
        self._seen = len(self.state.messages)
        self._log("agent", turn.content)

    def _require_judge(self) -> JudgeAgent:
        if self.judge is None:
            # Inline checkpoints work without an explicit judge: build a default one.
            self.judge = JudgeAgent()
        return self.judge

    def _log_verdicts(self, verdicts: list[CriterionVerdict]) -> None:
        for verdict in verdicts:
            self._log(f"judge {verdict.status}", f"{verdict.criterion} ({verdict.reasoning})")

    async def checkpoint(self, criteria: Sequence[str], additional_context: str | None) -> None:
        """Inline criteria: fail right here if any fails, carry on otherwise."""
        judged = await self._require_judge().judge(self.state, criteria, additional_context=additional_context)
        self.checkpoint_verdicts.extend(judged.verdicts)
        self._log_verdicts(judged.verdicts)
        if not all(v.passed for v in judged.verdicts):
            self.reasoning = judged.summary
            self.forced_outcome = False

    async def final_judge(self, additional_context: str | None = None) -> None:
        judge = self._require_judge()
        if not judge.criteria:
            raise ValueError(
                f"scenario {self.state.name!r} has a final judge() step, but the JudgeAgent has no criteria"
            )
        judged = await judge.judge(self.state, additional_context=additional_context)
        self._record_final(judged.verdicts, judged.summary)
        self.judge_score = await judge.score(self.state)

    def _record_final(self, verdicts: list[CriterionVerdict], summary: str) -> None:
        self.final_verdicts = list(verdicts)
        self.reasoning = summary
        self.final_judged_at = len(self.state.messages)
        self._log_verdicts(verdicts)

    async def run_proceed(self, step: ProceedStep) -> None:
        """Simulated user and agent take turns until a stop signal.

        Stops after ``step.turns`` turns or at ``max_turns``, when the simulator
        reports the goal as achieved, or when the judge concludes. A conclusion
        whose verdict still has inconclusive criteria is not final while turns
        remain: the dialogue goes on to gather the missing evidence.
        """
        played = 0
        while (step.turns is None or played < step.turns) and self.state.current_turn < self.max_turns:
            user_turn = await self._simulated_turn()
            # Goal reached: no polite filler message, the dialogue is judged as is.
            if user_turn.goal_achieved and self.state.current_turn > 0:
                self._log("user", "(goal achieved)")
                return
            if not user_turn.message:
                raise SimulationHarnessError(f"user simulator returned an empty message (scenario {self.state.name!r})")
            self.state.add_user_message(user_turn.message)
            self._log("user", user_turn.message)
            await _call(step.on_step, self.state)
            await self.agent_turn()
            await _call(step.on_step, self.state)
            played += 1
            await _call(step.on_turn, self.state)
            if await self._judge_concludes():
                return

    async def _judge_concludes(self) -> bool:
        judge = self.judge
        if judge is None or not judge.criteria or self.state.current_turn < self.min_turns:
            return False
        if not await judge.decide(self.state):
            return False
        judged = await judge.judge(self.state)
        last_turn = self.state.current_turn >= self.max_turns
        if any(v.inconclusive for v in judged.verdicts) and not last_turn:
            self._log("judge", "concluded with inconclusive criteria, continuing the dialogue")
            return False
        self._record_final(judged.verdicts, judged.summary)
        self.judge_score = await judge.score(self.state)
        return True

    # -- whole script -----------------------------------------------------

    def _warn_if_simulator_unused(self, script: Sequence[ScriptStep]) -> None:
        """Warn that the UserSimulatorAgent passed in never runs.

        In a script the simulator only runs for ``user()`` without text and for
        ``proceed()``. When every user message is spelled out, the simulator in
        ``agents`` is dead weight that suggests a model drives the dialogue when
        none does. Silently ignoring it loses scenarios meant to be live.
        """
        if not self.simulator_provided:
            return
        if any((isinstance(s, UserStep) and s.content is None) or isinstance(s, ProceedStep) for s in script):
            return
        logger.warning(
            "[%s] UserSimulatorAgent is never called: every user message in the script is fixed text. "
            "Drop it from agents, or use user() without text where the model should write the message.",
            self.state.name,
        )

    async def run_step(self, step: ScriptStep) -> None:
        match step:
            case UserStep(content=content):
                await self.user_turn(content)
            case AgentStep(content=content):
                await self.agent_turn(content)
            case MessageStep(role=role, content=content):
                self.state.add_message(role, content)
            case JudgeStep(criteria=None, additional_context=context):
                await self.final_judge(context)
            case JudgeStep(criteria=criteria, additional_context=context) if criteria is not None:
                await self.checkpoint(criteria, context)
            case ProceedStep():
                await self.run_proceed(step)
            case SucceedStep(reasoning=reasoning):
                self.reasoning = reasoning
                self.forced_outcome = True
            case FailStep(reasoning=reasoning):
                self.reasoning = reasoning
                self.failure_reason = f"fail(): {reasoning}" if reasoning else "fail() step"
                self.forced_outcome = False
            case _ if callable(step):
                await _call(step, self.state)
            case _:
                raise TypeError(f"unknown script step: {step!r}")

    async def run_script(self, script: Sequence[ScriptStep]) -> None:
        self._warn_if_simulator_unused(script)
        for step in script:
            await self.run_step(step)
            if self.ended:
                return
        # A judge with criteria and no final verdict yet, or one that the dialogue
        # has moved past (e.g. proceed() concluded, then more steps ran): judge at the end.
        stale = self.final_judged_at is None or self.final_judged_at < len(self.state.messages)
        if self.judge is not None and self.judge.criteria and stale:
            await self.final_judge()

    def result(
        self, *, failure_reason: str | None = None, harness_error: bool = False, total_time: float
    ) -> ScenarioResult:
        failure_reason = failure_reason or self.failure_reason
        if self.forced_outcome is not None:
            success = self.forced_outcome and failure_reason is None
        else:
            criteria_ok = all(v.passed for v in self.verdicts)
            score_ok = self.judge_score is None or self.judge_score.passed
            success = failure_reason is None and criteria_ok and score_ok
        return ScenarioResult(
            name=self.state.name,
            description=self.state.description,
            success=success,
            verdicts=self.verdicts,
            judge_score=self.judge_score,
            messages=list(self.state.messages),
            tool_calls=self.state.tool_call_names(),
            turns=self.state.current_turn,
            reasoning=self.reasoning,
            failure_reason=failure_reason,
            harness_error=harness_error,
            total_time=total_time,
            agent_time=self.agent_time,
        )


async def run(
    name: str,
    description: str,
    agents: Sequence[object],
    script: Sequence[ScriptStep] | None = None,
    *,
    max_turns: int | None = None,
    min_turns: int | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
) -> ScenarioResult:
    """Play one scenario and return its result; every result also lands in the report collector.

    ``agents`` holds exactly one :class:`AgentAdapter` (the agent under test)
    and optionally a :class:`UserSimulatorAgent` and a :class:`JudgeAgent`.
    Without ``script`` the run is ``[proceed()]`` followed by the judge's
    verdict. Unset keyword arguments fall back to :func:`configure`.

    An ``AssertionError`` from a script callable is recorded on the result and
    re-raised, so pytest reports it as a normal failure. A
    :class:`SimulationHarnessError` (no structured output from the judge or the
    simulator, a backend failure) is recorded and re-raised as well, but marks
    the result ``harness_error=True``, which reports show as WARN, not FAIL.
    """
    for label, value in (("max_turns", max_turns), ("min_turns", min_turns)):
        if value is not None and value < 1:
            raise ValueError(f"{label} must be at least 1, got {value}")
    config = get_config()
    simulation = _Simulation(
        name=name,
        description=description,
        agents=agents,
        max_turns=config.max_turns if max_turns is None else max_turns,
        min_turns=config.min_turns if min_turns is None else min_turns,
        verbose=config.verbose if verbose is None else verbose,
        debug=config.debug if debug is None else debug,
    )
    started = time.perf_counter()

    def finish(**kwargs: Any) -> ScenarioResult:
        result = simulation.result(total_time=time.perf_counter() - started, **kwargs)
        collector.add(result)
        return result

    try:
        await simulation.run_script(script if script is not None else [proceed()])
    except AssertionError as exc:
        finish(failure_reason=f"assertion in script: {exc}")
        raise
    except SimulationHarnessError as exc:
        # The harness broke, not the product: WARN in reports, so judge,
        # simulator and gateway failures do not drag the pass rate down. Still
        # re-raised: a plain pytest run must see the scenario did not finish.
        logger.warning("[%s] harness failure: %r", name, exc)
        finish(failure_reason=f"harness failure: {exc!r}", harness_error=True)
        raise
    except Exception as exc:
        # Full traceback to the log, a short reason in the report.
        logger.exception("[%s] scenario run failed", name)
        finish(failure_reason=f"runtime error: {exc!r}")
        raise
    return finish()
