"""Helper LLM agents: the user simulator and the per-criterion judge.

Both are plain agno agents with an ``output_schema``. Prompts are adapted from
LangWatch Scenario's simulator and judge (Apache-2.0, see NOTICE).
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.scorer.judge import JudgeScorer
from pydantic import BaseModel, Field

from simulhausen import transcript_tools
from simulhausen.config import get_config, require_model, resolve_parser_model
from simulhausen.exceptions import SimulationStructuredOutputError
from simulhausen.prompts import (
    JUDGE_DECISION_PROMPT,
    JUDGE_VERDICT_PROMPT,
    PER_CRITERION_RULES,
    STRUCTURED_RETRY_NUDGE,
    TRANSCRIPT_DISCOVERY_RULES,
    TRANSCRIPT_SKELETON_INTRO,
    USER_SIMULATOR_PROMPT,
)
from simulhausen.types import AgentRole, CriterionVerdict, JudgeScore

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agno.models.base import Model

    from simulhausen.types import Message, ScenarioState

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


def _model_id(explicit: Model | None, default: Model | None) -> object:
    """A model as it enters cache keys: by ``id``, since its repr carries client handles and addresses."""
    return getattr(explicit or default, "id", None)


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
    cacheable: ClassVar[bool] = True

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

    def cache_fingerprint(self) -> dict[str, object]:
        """The settings that shape this agent's answers; part of every cache key.

        A subclass that adds such a setting extends this, or reruns with a
        ``cache_key`` replay stale answers. One with hidden state (like the
        red-team attacker) sets ``cacheable = False`` instead.
        """
        config = get_config()
        return {
            "model": _model_id(self._model, config.model),
            "parser_model": _model_id(self._parser_model, config.parser_model),
            "system_prompt": self.system_prompt,
            "persona": self.persona,
        }

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


class _Discovery:
    """The two transcript lookups of one judge call, bound to the dialogue as it stands.

    agno turns each public method into a tool: the name, the docstring and the
    ``Args`` section are what the model sees. Results go back fenced as
    untrusted data, since a transcript can carry injected instructions. ``calls``
    counts the lookups that ran; agno refuses the ones past ``tool_call_limit``
    before they get here.
    """

    def __init__(self, messages: Sequence[Message]) -> None:
        self._messages = list(messages)
        self.calls = 0

    @property
    def tools(self) -> list[Callable[..., str]]:
        return [self.expand_transcript, self.grep_transcript]

    def expand_transcript(self, indices: list[int]) -> str:
        """Read whole messages of the dialogue. Use the index shown in brackets in the skeleton.

        Ask for every message you need in one call.

        Args:
            indices: 0-based message indices to read in full.
        """
        self.calls += 1
        return _fence("excerpt", transcript_tools.expand_transcript(self._messages, indices))

    def grep_transcript(self, pattern: str) -> str:
        """Find the messages of the dialogue that contain some text, with their full content.

        Args:
            pattern: Text to look for, matched case-insensitively as plain text. It is not a regular expression.
        """
        self.calls += 1
        return _fence("excerpt", transcript_tools.grep_transcript(self._messages, pattern))


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

    A dialogue over ``token_threshold`` estimated tokens (UTF-8 bytes / 4) is not
    sent whole. The judge gets a skeleton of it (one line per message) and two
    tools, ``expand_transcript`` and ``grep_transcript``, to read what it needs
    before ruling, at most ``max_discovery_steps`` calls per run. This applies to
    the verdict and to the continue/conclude decision. ``score_threshold``
    grading is not affected: agno's scorer always reads the whole dialogue.
    """

    role: AgentRole = AgentRole.JUDGE
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        criteria: Sequence[str] = (),
        model: Model | None = None,
        *,
        parser_model: Model | None = None,
        system_prompt: str | None = None,
        score_threshold: int | None = None,
        token_threshold: int = transcript_tools.DEFAULT_TOKEN_THRESHOLD,
        max_discovery_steps: int = 10,
    ) -> None:
        if score_threshold is not None and not 1 <= score_threshold <= 10:
            raise ValueError("score_threshold must be between 1 and 10")
        for name, value in (("token_threshold", token_threshold), ("max_discovery_steps", max_discovery_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.criteria = list(criteria)
        self._model = model
        self._parser_model = parser_model
        self.system_prompt = system_prompt
        self.score_threshold = score_threshold
        self.token_threshold = token_threshold
        self.max_discovery_steps = max_discovery_steps

    def cache_fingerprint(self) -> dict[str, object]:
        """The settings that shape this judge's answers; part of every cache key.

        The criteria are keyed per call (inline checkpoints differ from the
        judge's own). A subclass that adds an answer-shaping setting extends this.
        """
        config = get_config()
        return {
            "model": _model_id(self._model, config.model),
            "parser_model": _model_id(self._parser_model, config.parser_model),
            "system_prompt": self.system_prompt,
            "score_threshold": self.score_threshold,
            "token_threshold": self.token_threshold,
            "max_discovery_steps": self.max_discovery_steps,
        }

    def _agent(self, instructions: str, schema: type[BaseModel], discovery: _Discovery | None = None) -> Agent:
        extra: dict[str, Any] = {}
        if discovery is not None:
            # agno's limit is soft: it answers calls past the limit with an error result and
            # asks the model again, so the run ends when the model stops calling tools.
            extra = {"tools": discovery.tools, "tool_call_limit": self.max_discovery_steps}
        return Agent(
            model=require_model(self._model),
            instructions=instructions,
            output_schema=schema,
            parser_model=resolve_parser_model(self._parser_model),
            telemetry=False,
            **extra,
        )

    def _discovery(self, state: ScenarioState) -> _Discovery | None:
        """The lookups for a dialogue over ``token_threshold``; ``None`` while it fits."""
        if transcript_tools.estimate_tokens(state.transcript()) <= self.token_threshold:
            return None
        return _Discovery(state.messages)

    def _with_discovery_rules(self, instructions: str, discovery: _Discovery | None) -> str:
        if discovery is None:
            return instructions
        return instructions.rstrip() + "\n\n" + TRANSCRIPT_DISCOVERY_RULES.format(max_steps=self.max_discovery_steps)

    @staticmethod
    def evidence(state: ScenarioState, additional_context: str | None = None, *, skeleton: bool = False) -> str:
        """The dialogue plus what the agent saw and did, each fenced as untrusted data.

        The judge sees the same reference material (RAG) as the agent, so
        grounding criteria are checked against the real retrieval rather than
        a hand-written answer key. Tool executions let it tell facts that came
        from a tool apart from hallucinations.

        With ``skeleton=True`` the dialogue is replaced by its skeleton (index,
        speaker, size and a short preview per message), for a dialogue the judge
        reads through ``expand_transcript`` and ``grep_transcript``. References
        and tool executions keep their own caps either way.
        """
        if skeleton:
            shown = transcript_tools.build_transcript_skeleton(state.messages)
            prompt = TRANSCRIPT_SKELETON_INTRO + "\n" + _fence("transcript", shown)
        else:
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
        discovery = self._discovery(state)
        judged = await _arun_structured(
            self._agent(self._with_discovery_rules(instructions, discovery), JudgeVerdict, discovery),
            self.evidence(state, additional_context, skeleton=discovery is not None),
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
        """Unscripted runs: ``True`` when the dialogue holds enough for a verdict.

        A long dialogue is read through the discovery tools here too. A decision
        that never came after the judge spent its whole step budget concludes
        the dialogue instead of continuing, as in LangWatch Scenario: the
        verdict is then the next step, and the discovery would only start over
        on every following turn.
        """
        if not self.criteria:
            return False
        discovery = self._discovery(state)
        instructions = JUDGE_DECISION_PROMPT.format(description=state.description, criteria=_numbered(self.criteria))
        agent = self._agent(self._with_discovery_rules(instructions, discovery), _TurnDecision, discovery)
        response = await agent.arun(self.evidence(state, skeleton=discovery is not None))
        decision = response.content
        if not isinstance(decision, _TurnDecision):
            if discovery is not None and discovery.calls >= self.max_discovery_steps:
                logger.warning(
                    "judge spent its %d discovery steps without a decision (scenario %r): concluding",
                    self.max_discovery_steps,
                    state.name,
                )
                return True
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
