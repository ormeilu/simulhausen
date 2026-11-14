"""Scenario runner: plays the dialogue from a script or on its own, then judges it."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

from simulhausen._cache import cached_acall, caching_context, scenario_context
from simulhausen.agents import JudgeAgent, UserSimulatorAgent, UserTurn
from simulhausen.config import check_cache_key, get_config
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
from simulhausen.types import (
    AgentAdapter,
    AgentInput,
    AgentTurn,
    JudgeCall,
    ScenarioResult,
    ScenarioState,
    TrialsResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from simulhausen.agents import JudgeVerdict
    from simulhausen.script import ScriptStep, StateCallback
    from simulhausen.types import CriterionVerdict, JudgeScore

__all__ = ["run", "run_trials"]

logger = logging.getLogger(__name__)


async def _call(callback: StateCallback | None, state: ScenarioState) -> object:
    if callback is None:
        return None
    outcome = callback(state)
    if inspect.isawaitable(outcome):
        return await outcome
    return outcome


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
        # Set when a callable ended the dialogue with judge(): the verdict decides.
        self.stopped = False
        self.failure_reason: str | None = None
        # Transcript length at the last final verdict; None until there is one.
        self.final_judged_at: int | None = None
        self.agent_time = 0.0
        self.judge_time = 0.0
        self.simulator_time = 0.0
        self.judge_calls: list[JudgeCall] = []

    @property
    def ended(self) -> bool:
        return self.forced_outcome is not None or self.stopped

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

    async def _helper_call[T](
        self,
        agent: UserSimulatorAgent | JudgeAgent,
        method: str,
        call: Callable[[], Awaitable[T]],
        **arguments: object,
    ) -> T:
        """Run a call to a helper agent; with a cache key set, replay a stored answer instead.

        The key covers the agent's class and settings, the call's own arguments and the
        dialogue so far. Agents with ``cacheable = False`` (a stateful simulator, say) are
        always called.
        """
        started = time.perf_counter()
        try:
            context = caching_context()
            if context is None or not agent.cacheable:
                return await call()
            identity = f"{type(agent).__module__}.{type(agent).__qualname__}.{method}"
            return await cached_acall(
                context, identity, {"agent": agent.cache_fingerprint(), "state": self.state, **arguments}, call
            )
        finally:
            elapsed = time.perf_counter() - started
            if isinstance(agent, JudgeAgent):
                self.judge_time += elapsed
            else:
                self.simulator_time += elapsed

    async def _simulated_turn(self) -> UserTurn:
        if self.debug and (typed := await self._human_message()) is not None:
            return UserTurn(message=typed, goal_achieved=False)
        return await self._helper_call(self.simulator, "next_turn", lambda: self.simulator.next_turn(self.state))

    async def user_turn(self, content: str | None) -> None:
        if content is None:
            typed = await self._human_message() if self.debug else None
            content = typed or await self._helper_call(
                self.simulator, "next_message", lambda: self.simulator.next_message(self.state)
            )
        self.state.add_user_message(content)
        self._log("user", content)

    async def agent_turn(self, content: str | None = None) -> None:
        if content is not None:
            turn = AgentTurn(content=content)
        else:
            agent_input = AgentInput(
                thread_id=self.state.thread_id,
                messages=list(self.state.messages),
                new_messages=list(self.state.messages[self.state._agent_seen :]),
                scenario_state=self.state,
            )
            started = time.perf_counter()
            try:
                response = await self.adapter.call(agent_input)
            finally:
                self.agent_time += time.perf_counter() - started
            turn = response if isinstance(response, AgentTurn) else AgentTurn(content=str(response or ""))
        self.state.add_agent_message(turn.content, tools=turn.tools, references=turn.references, usage=turn.usage)
        self.state.current_turn += 1
        self.state._agent_seen = len(self.state.messages)
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
        judged = await self._judge_call(self._require_judge(), criteria, additional_context)
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
        judged = await self._judge_call(judge, None, additional_context)
        self._record_final(judged.verdicts, judged.summary)
        self.judge_score = await self._score(judge)

    async def _judge_call(
        self, judge: JudgeAgent, criteria: Sequence[str] | None, additional_context: str | None
    ) -> JudgeVerdict:
        """``judge.judge`` keyed by the criteria it will really apply (its own when ``None``).

        Each call lands in ``judge_calls`` with the evidence the judge saw.
        """
        applied = list(judge.criteria if criteria is None else criteria)
        started = time.perf_counter()
        judged = await self._helper_call(
            judge,
            "judge",
            lambda: judge.judge(self.state, criteria, additional_context=additional_context),
            criteria=applied,
            additional_context=additional_context,
        )
        self.judge_calls.append(
            JudgeCall(
                criteria=applied,
                verdicts=list(judged.verdicts),
                summary=judged.summary,
                additional_context=additional_context,
                messages_count=len(self.state.messages),
                references=list(self.state.references),
                tool_executions=list(self.state.tool_executions),
                time=time.perf_counter() - started,
            )
        )
        return judged

    async def _decide(self, judge: JudgeAgent) -> bool:
        return await self._helper_call(judge, "decide", lambda: judge.decide(self.state), criteria=judge.criteria)

    async def _score(self, judge: JudgeAgent) -> JudgeScore | None:
        return await self._helper_call(judge, "score", lambda: judge.score(self.state), criteria=judge.criteria)

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
        if not await self._decide(judge):
            return False
        judged = await self._judge_call(judge, None, None)
        last_turn = self.state.current_turn >= self.max_turns
        if any(v.inconclusive for v in judged.verdicts) and not last_turn:
            self._log("judge", "concluded with inconclusive criteria, continuing the dialogue")
            return False
        self._record_final(judged.verdicts, judged.summary)
        self.judge_score = await self._score(judge)
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
                # A callable may end the scenario by returning succeed()/fail(), or
                # judge() to end the dialogue and let the verdict decide (the
                # red-team early exit does that once its scorer sees the attack land).
                outcome = await _call(step, self.state)
                if isinstance(outcome, SucceedStep | FailStep):
                    await self.run_step(outcome)
                elif isinstance(outcome, JudgeStep):
                    await self.run_step(outcome)
                    self.stopped = True
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
            judge_time=self.judge_time,
            simulator_time=self.simulator_time,
            judge_calls=list(self.judge_calls),
            usage=self.state.usage,
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
    cache_key: str | None = None,
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

    ``cache_key`` (default: the one from :func:`configure`) turns on the response
    cache for this run: the simulator's and the judge's answers, and every function
    decorated with :func:`simulhausen.cache`, are replayed from disk when the same
    call was seen before. Change the key to start from scratch.
    """
    result, error = await _play(
        name,
        description,
        agents,
        script,
        max_turns=max_turns,
        min_turns=min_turns,
        verbose=verbose,
        debug=debug,
        cache_key=cache_key,
    )
    if error is not None:
        raise error
    return result


async def _play(
    name: str,
    description: str,
    agents: Sequence[object],
    script: Sequence[ScriptStep] | None,
    *,
    max_turns: int | None,
    min_turns: int | None,
    verbose: bool | None,
    debug: bool | None,
    cache_key: str | None,
    trial: int | None = None,
    trials: int | None = None,
) -> tuple[ScenarioResult, Exception | None]:
    """Play one scenario; the result always comes back, next to the error that ended the run early."""
    for label, value in (("max_turns", max_turns), ("min_turns", min_turns)):
        if value is not None and value < 1:
            raise ValueError(f"{label} must be at least 1, got {value}")
    if cache_key is not None:
        check_cache_key(cache_key)
    config = get_config()
    effective_key = config.cache_key if cache_key is None else cache_key
    if effective_key is not None and trial is not None:
        # Each trial is its own draw: sharing a key would replay trial 1 into the others.
        effective_key = f"{effective_key}#trial-{trial}"
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
        result.trial, result.trials = trial, trials
        collector.add(result)
        return result

    try:
        with scenario_context(name, description, effective_key):
            await simulation.run_script(script if script is not None else [proceed()])
    except AssertionError as exc:
        return finish(failure_reason=f"assertion in script: {exc}"), exc
    except SimulationHarnessError as exc:
        # The harness broke, not the product: WARN in reports, so judge,
        # simulator and gateway failures do not drag the pass rate down. run()
        # still re-raises: a plain pytest run must see the scenario did not finish.
        logger.warning("[%s] harness failure: %r", name, exc)
        return finish(failure_reason=f"harness failure: {exc!r}", harness_error=True), exc
    except Exception as exc:
        # Full traceback to the log, a short reason in the report.
        logger.exception("[%s] scenario run failed", name)
        return finish(failure_reason=f"runtime error: {exc!r}"), exc
    return finish(), None


async def run_trials(
    name: str,
    description: str,
    agents: Sequence[object] | Callable[[], Sequence[object]],
    script: Sequence[ScriptStep] | Callable[[Sequence[object]], Sequence[ScriptStep]] | None = None,
    *,
    trials: int | None = None,
    min_pass_rate: float = 1.0,
    concurrency: int = 1,
    max_turns: int | None = None,
    min_turns: int | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
    cache_key: str | None = None,
) -> TrialsResult:
    """Play the same scenario ``trials`` times and report how reliably it passes.

    An agent that passes once may fail the next time on the same input, so one
    green run says little. The :class:`TrialsResult` holds every trial's
    :class:`ScenarioResult` and the reliability numbers over them: the pass
    rate, ``pass_hat(k)`` (the chance that k trials in a row all pass, the
    pass^k of tau-bench) and whether the scenario is flaky.

    Nothing is raised for a failed trial: a script assertion, a harness
    failure or a runtime error ends that trial (recorded on its result, the
    exception kept in ``TrialsResult.errors``) and the next one starts. Assert
    on the outcome instead::

        outcome = await sim.run_trials(name=..., description=..., agents=make_agents, trials=5)
        assert outcome.success, outcome.failure_summary()

    ``agents`` is the usual list, or a zero-argument factory called once per
    trial. Use the factory for anything that keeps state between calls: a
    :class:`RedTeamAgent`, an adapter around a stateful agent object. ``script``
    may be a function of the trial's agents for the same reason
    (``lambda agents: agents[1].marathon_script()``).

    ``trials`` falls back to :func:`configure` (``--sim-trials``), which starts
    at 3. ``min_pass_rate`` is the share of judged trials (PASS + FAIL) that has
    to pass for ``success``; WARN trials are left out, as in every pass rate
    here. ``concurrency`` runs that many trials at a time; keep it at 1 when the
    agent or its clients are process-wide singletons. With a cache key each
    trial gets its own entries, so a rerun replays every trial as it went.
    """
    config = get_config()
    count = config.trials if trials is None else trials
    if count < 1:
        raise ValueError(f"trials must be at least 1, got {count}")
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ValueError(f"min_pass_rate must be between 0 and 1, got {min_pass_rate}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    gate = asyncio.Semaphore(concurrency)

    async def one(trial: int) -> tuple[ScenarioResult, Exception | None]:
        async with gate:
            trial_agents = cast("Sequence[object]", agents() if callable(agents) else agents)
            trial_script = cast("Sequence[ScriptStep] | None", script(trial_agents) if callable(script) else script)
            return await _play(
                name,
                description,
                trial_agents,
                trial_script,
                max_turns=max_turns,
                min_turns=min_turns,
                verbose=verbose,
                debug=debug,
                cache_key=cache_key,
                trial=trial,
                trials=count,
            )

    played = await asyncio.gather(*(one(trial) for trial in range(1, count + 1)))
    return TrialsResult(
        name=name,
        results=[result for result, _ in played],
        errors=[error for _, error in played],
        min_pass_rate=min_pass_rate,
    )
