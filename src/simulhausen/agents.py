"""Helper LLM agents: the user simulator and the per-criterion judge.

Both are plain agno agents with an ``output_schema``. Prompts are adapted from
LangWatch Scenario's simulator and judge (Apache-2.0, see NOTICE).
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Literal

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.scorer.judge import JudgeScorer
from pydantic import BaseModel, Field

from simulhausen.config import require_model, resolve_parser_model
from simulhausen.exceptions import SimulationStructuredOutputError
from simulhausen.prompts import (
    JUDGE_DECISION_PROMPT,
    JUDGE_VERDICT_PROMPT,
    PER_CRITERION_RULES,
    STRUCTURED_RETRY_NUDGE,
    USER_SIMULATOR_PROMPT,
)
from simulhausen.types import AgentRole, CriterionVerdict, JudgeScore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agno.models.base import Model

    from simulhausen.types import ScenarioState

__all__ = ["JudgeAgent", "JudgeVerdict", "UserSimulatorAgent", "UserTurn"]

logger = logging.getLogger(__name__)

# Models now and then answer in free text instead of the schema. A parser model
# fixes most of it (see SimulationConfig.parser_model); the retry covers what
# remains of the nondeterminism.
STRUCTURED_ATTEMPTS = 3


def _fence(label: str, text: str) -> str:
    """Wrap untrusted data in a tag with a random per-call suffix.

    User text, RAG hits and tool results may contain instructions or closing
    XML tags. A random suffix makes the closing tag unguessable from inside the
    data, the same idea as agno's hardened judge prompt.
    """
    tag = f"untrusted_{label}_{secrets.token_hex(16)}"
    return f"<{tag}>\n{text}\n</{tag}>"


async def _arun_structured[T](agent: Agent, prompt: str, schema: type[T], *, scenario: str, what: str) -> T:
    """Run the agent and demand a schema object; retry with a format nudge if it is not one."""
    for attempt in range(1, STRUCTURED_ATTEMPTS + 1):
        attempt_prompt = prompt if attempt == 1 else prompt + STRUCTURED_RETRY_NUDGE
        response = await agent.arun(attempt_prompt)
        content = response.content
        if isinstance(content, schema):
            return content
        logger.warning(
            "%s returned %s instead of %s (scenario %r, attempt %d/%d)",
            what,
            type(content).__name__,
            schema.__name__,
            scenario,
            attempt,
            STRUCTURED_ATTEMPTS,
        )
    raise SimulationStructuredOutputError(
        f"{what} returned no structured output in {STRUCTURED_ATTEMPTS} attempts (scenario {scenario!r})"
    )


def _numbered(criteria: Sequence[str]) -> str:
    return "\n".join(f"{index}. {criterion}" for index, criterion in enumerate(criteria, 1))


# -- user simulator ------------------------------------------------------------


class UserTurn(BaseModel):
    """One simulated user turn: the message plus a "goal reached" signal."""

    message: str = Field(description="The user's next message, without quotes or commentary")
    goal_achieved: bool = Field(
        description="true when the scenario's goal is already reached and the dialogue should stop"
    )


class UserSimulatorAgent:
    """An LLM playing the user: writes the next user message from the scenario and transcript.

    ``next_turn`` also returns ``goal_achieved``, so unscripted runs stop as
    soon as the scenario's goal is reached instead of padding the dialogue
    with polite filler up to ``max_turns``.

    ``system_prompt`` replaces the built-in prompt entirely (the scenario
    description is then up to you to include); ``persona`` is appended to
    whichever prompt is used.
    """

    role: AgentRole = AgentRole.USER

    def __init__(
        self,
        model: Model | None = None,
        *,
        parser_model: Model | None = None,
        system_prompt: str | None = None,
        persona: str | None = None,
    ) -> None:
        self._model = model
        self._parser_model = parser_model
        self.system_prompt = system_prompt
        self.persona = persona

    def instructions(self, state: ScenarioState) -> str:
        prompt = self.system_prompt or USER_SIMULATOR_PROMPT.format(description=state.description)
        if self.persona:
            prompt += f"\n<persona>\n{self.persona}\n</persona>\n"
        return prompt

    async def next_turn(self, state: ScenarioState) -> UserTurn:
        agent = Agent(
            model=require_model(self._model),
            instructions=self.instructions(state),
            output_schema=UserTurn,
            parser_model=resolve_parser_model(self._parser_model),
            telemetry=False,
        )
        transcript = state.transcript() or "(the dialogue has not started yet)"
        turn = await _arun_structured(
            agent,
            f"Dialogue so far:\n{transcript}\n\nThe user's next message:",
            UserTurn,
            scenario=state.name,
            what="user simulator",
        )
        message = turn.message.strip()
        if not (message or turn.goal_achieved):
            raise SimulationStructuredOutputError(f"user simulator returned an empty turn (scenario {state.name!r})")
        return UserTurn(message=message, goal_achieved=turn.goal_achieved)

    async def next_message(self, state: ScenarioState) -> str:
        """A message for a script ``user()`` step without content."""
        turn = await self.next_turn(state)
        if not turn.message:
            raise SimulationStructuredOutputError(f"user simulator returned an empty message (scenario {state.name!r})")
        return turn.message


# -- judge ---------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """The judge's structured answer: one verdict per criterion plus a summary."""

    verdicts: list[CriterionVerdict] = Field(description="Exactly one verdict per criterion, in the given order")
    summary: str = Field(description="One or two sentences on the overall outcome")


class _TurnDecision(BaseModel):
    """Mid-dialogue decision: keep talking or conclude.

    No reasoning field on purpose: writing one pushes the judge to commit to a
    pass or fail before the evidence is in, and the text is wasted tokens for a
    binary transition (the same choice LangWatch Scenario made).
    """

    decision: Literal["continue", "conclude"] = Field(
        description="continue: the dialogue needs another user turn; conclude: there is enough to judge"
    )


class JudgeAgent:
    """LLM judge with per-criterion verdicts and an optional numeric grade.

    Each criterion gets ``passed``, ``failed`` or ``inconclusive`` with its own
    reasoning. ``criteria`` may be empty when the scenario only uses inline
    ``judge(criteria=[...])`` checkpoints.

    ``score_threshold`` (1-10) additionally runs agno's ``JudgeScorer`` over the
    whole dialogue and fails the scenario below the threshold. It is opt-in so a
    numeric cut-off never changes results before it is calibrated.

    ``system_prompt`` replaces the built-in verdict prompt; the dialogue and the
    evidence are still sent as the message, fenced as untrusted data.
    """

    role: AgentRole = AgentRole.JUDGE

    def __init__(
        self,
        criteria: Sequence[str] = (),
        model: Model | None = None,
        *,
        parser_model: Model | None = None,
        system_prompt: str | None = None,
        score_threshold: int | None = None,
    ) -> None:
        if score_threshold is not None and not 1 <= score_threshold <= 10:
            raise ValueError("score_threshold must be between 1 and 10")
        self.criteria = list(criteria)
        self._model = model
        self._parser_model = parser_model
        self.system_prompt = system_prompt
        self.score_threshold = score_threshold

    def _agent(self, instructions: str, schema: type[BaseModel]) -> Agent:
        return Agent(
            model=require_model(self._model),
            instructions=instructions,
            output_schema=schema,
            parser_model=resolve_parser_model(self._parser_model),
            telemetry=False,
        )

    @staticmethod
    def evidence(state: ScenarioState, additional_context: str | None = None) -> str:
        """The dialogue plus what the agent saw and did, each fenced as untrusted data.

        The judge sees the same reference material (RAG) as the agent, so
        grounding criteria are checked against the real retrieval rather than
        a hand-written answer key. Tool executions let it tell facts that came
        from a tool apart from hallucinations.
        """
        prompt = "Dialogue:\n" + _fence("transcript", state.transcript() or "(empty)")
        if references := state.references_text():
            prompt += "\n\nReference material the agent saw:\n" + _fence("references", references)
        if tool_evidence := state.tool_evidence_text():
            prompt += "\n\nTool executions:\n" + _fence("tools", tool_evidence)
        if additional_context:
            prompt += "\n\nAdditional context from the test:\n" + _fence("context", additional_context)
        return prompt

    async def judge(
        self,
        state: ScenarioState,
        criteria: Sequence[str] | None = None,
        *,
        additional_context: str | None = None,
    ) -> JudgeVerdict:
        """Verdict on ``criteria`` (the judge's own when ``None``), one per criterion.

        ``additional_context`` is extra evidence the dialogue does not show, for
        example the state of a database or a file the agent should have changed.
        """
        criteria = list(self.criteria if criteria is None else criteria)
        if not criteria:
            raise ValueError("the judge has no criteria: pass JudgeAgent(criteria=[...]) or judge(criteria=[...])")
        instructions = self.system_prompt or JUDGE_VERDICT_PROMPT.format(
            description=state.description,
            criteria=_numbered(criteria),
            per_criterion_rules=PER_CRITERION_RULES,
        )
        judged = await _arun_structured(
            self._agent(instructions, JudgeVerdict),
            self.evidence(state, additional_context),
            JudgeVerdict,
            scenario=state.name,
            what="judge",
        )
        return JudgeVerdict(verdicts=self._align(state, criteria, judged.verdicts), summary=judged.summary)

    async def evaluate(
        self,
        state: ScenarioState,
        criteria: Sequence[str] | None = None,
        *,
        additional_context: str | None = None,
    ) -> list[CriterionVerdict]:
        """Just the per-criterion verdicts of :meth:`judge`."""
        return (await self.judge(state, criteria, additional_context=additional_context)).verdicts

    @staticmethod
    def _align(state: ScenarioState, criteria: list[str], verdicts: list[CriterionVerdict]) -> list[CriterionVerdict]:
        """One verdict per criterion, in order, with the criterion text restored verbatim.

        Models paraphrase, drop or duplicate criteria. With a matching count the
        order is trusted; otherwise verdicts are matched by exact text and a
        criterion left without one becomes inconclusive.
        """
        if len(verdicts) == len(criteria):
            return [v.model_copy(update={"criterion": c}) for c, v in zip(criteria, verdicts, strict=True)]
        logger.warning(
            "judge returned %d verdicts for %d criteria (scenario %r)", len(verdicts), len(criteria), state.name
        )
        by_text = {v.criterion: v for v in verdicts}
        return [
            by_text.get(c)
            or CriterionVerdict(criterion=c, status="inconclusive", reasoning="the judge returned no verdict for it")
            for c in criteria
        ]

    async def decide(self, state: ScenarioState) -> bool:
        """Unscripted runs: ``True`` when the dialogue holds enough for a verdict."""
        if not self.criteria:
            return False
        agent = self._agent(
            JUDGE_DECISION_PROMPT.format(description=state.description, criteria=_numbered(self.criteria)),
            _TurnDecision,
        )
        response = await agent.arun(self.evidence(state))
        decision = response.content
        if not isinstance(decision, _TurnDecision):
            # Unparsed decision: keep talking until max_turns, the final verdict decides.
            logger.warning("judge decision could not be parsed (scenario %r)", state.name)
            return False
        return decision.decision == "conclude"

    async def score(self, state: ScenarioState) -> JudgeScore | None:
        """Grade the whole dialogue 1-10 with agno's hardened ``JudgeScorer`` (when enabled)."""
        if self.score_threshold is None:
            return None
        scorer = JudgeScorer(
            model=require_model(self._model),
            criteria=(
                f"Scenario:\n{state.description}\n\n"
                "The dialogue must satisfy all of the following criteria:\n"
                f"{_numbered(self.criteria)}"
            ),
            mode="numeric",
            threshold=self.score_threshold,
        )
        # agno fences both the run output and the reference answer with nonce
        # blocks, so dialogue text and RAG data cannot rewrite the judge's task.
        score = await scorer.ascore(RunOutput(content=state.transcript()), expected=state.references_text() or None)
        raw_score = int((score.detail or {}).get("raw_score", round(score.value * 9 + 1)))
        return JudgeScore(
            value=score.value,
            raw_score=raw_score,
            threshold=self.score_threshold,
            passed=score.passed,
            reasoning=score.reason,
        )
