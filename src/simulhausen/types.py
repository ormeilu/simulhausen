"""Core types: adapter input and output, dialogue state, verdicts, results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agno.models.message import MessageReferences
    from agno.models.response import ToolExecution

__all__ = [
    "AgentAdapter",
    "AgentInput",
    "AgentReturnTypes",
    "AgentRole",
    "AgentTurn",
    "CriterionStatus",
    "CriterionVerdict",
    "JudgeScore",
    "Message",
    "Role",
    "ScenarioResult",
    "ScenarioState",
    "ScenarioStatus",
]

type Role = Literal["user", "assistant"]


class Message(TypedDict):
    """One transcript entry."""

    role: Role
    content: str


class AgentRole(StrEnum):
    """Who is speaking or acting in a scenario."""

    USER = "user"
    AGENT = "agent"
    JUDGE = "judge"


@dataclass
class AgentTurn:
    """What the agent under test produced in one turn: text, tool calls, references.

    ``tools`` back assertions such as ``state.has_tool_call(...)``. ``references``
    (the RAG hits from ``RunOutput.references``) are passed on to the judge so it
    grades the reply against the same material the agent saw. ``agno_adapter``
    fills all three.
    """

    content: str
    tools: list[ToolExecution] | None = None
    references: list[MessageReferences] | None = None


# An adapter may return plain text or text with tools and references.
type AgentReturnTypes = str | AgentTurn


@dataclass
class AgentInput:
    """Adapter input for one turn."""

    thread_id: str
    # Whole transcript so far, oldest first.
    messages: list[Message]
    # Messages added since the agent last spoke (usually one user message).
    new_messages: list[Message] = field(default_factory=list)
    # Live scenario state, for adapters that need more than the transcript.
    scenario_state: ScenarioState | None = None

    def last_new_user_message_str(self) -> str:
        """Latest user message: the text the agent should answer now."""
        for message in reversed(self.new_messages or self.messages):
            if message["role"] == "user":
                return message["content"]
        return ""


class AgentAdapter(ABC):
    """Bridge between the runner and the agent under test.

    ``agno_adapter`` covers agno Agent, Team and Workflow targets and
    ``function_adapter`` covers a plain function; subclass this only for
    custom calling logic or a non-agno runtime. Keep one conversation per
    ``input.thread_id`` so the whole scenario runs in a single session.
    """

    role: AgentRole = AgentRole.AGENT

    @abstractmethod
    async def call(self, input: AgentInput) -> AgentReturnTypes: ...


def _reference_block(ref: dict[str, Any] | str) -> str:
    """One RAG hit for the judge: its text plus its URL from ``meta_data`` when present."""
    if isinstance(ref, str):
        return ref
    text = str(ref.get("content") or ref)
    meta = ref.get("meta_data") or ref.get("metadata") or {}
    url = meta.get("url") if isinstance(meta, dict) else None
    return f"{text}\nSource URL: {url}" if url else text


def _is_clean(tool: ToolExecution) -> bool:
    """Executed without error and not paused for confirmation or user input."""
    return bool(tool.tool_name) and not tool.tool_call_error and not tool.is_paused


@dataclass
class ScenarioState:
    """Dialogue state: transcript, accumulated tool executions and references."""

    thread_id: str
    name: str
    description: str
    messages: list[Message] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    references: list[MessageReferences] = field(default_factory=list)
    # Agent turns played so far.
    current_turn: int = 0

    # -- mutation ---------------------------------------------------------

    def add_message(self, role: Role, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_agent_message(
        self,
        content: str,
        tools: list[ToolExecution] | None = None,
        references: list[MessageReferences] | None = None,
    ) -> None:
        self.add_message("assistant", content)
        if tools:
            self.tool_executions.extend(tools)
        if references:
            self.references.extend(references)

    # -- messages ---------------------------------------------------------

    def last_message(self) -> Message:
        """Most recent message of any role; raises ``ValueError`` on an empty transcript."""
        if not self.messages:
            raise ValueError("the transcript is empty")
        return self.messages[-1]

    def last_user_message(self) -> Message:
        """Most recent user message; raises ``ValueError`` when there is none."""
        return self._last_with_role("user")

    def last_agent_message(self) -> Message:
        """Most recent agent message; raises ``ValueError`` when there is none."""
        return self._last_with_role("assistant")

    def first_user_message_str(self) -> str:
        """Text of the opening user message, ``""`` when there is none."""
        return next((m["content"] for m in self.messages if m["role"] == "user"), "")

    def last_user_message_str(self) -> str:
        """Text of the most recent user message, ``""`` when there is none."""
        return self._last_content("user")

    def last_agent_message_str(self) -> str:
        """Text of the most recent agent message, ``""`` when there is none."""
        return self._last_content("assistant")

    def _last_with_role(self, role: Role) -> Message:
        for message in reversed(self.messages):
            if message["role"] == role:
                return message
        raise ValueError(f"no {role} message in the transcript")

    def _last_content(self, role: Role) -> str:
        for message in reversed(self.messages):
            if message["role"] == role:
                return message["content"]
        return ""

    def transcript(self, user_name: str = "User", agent_name: str = "Assistant") -> str:
        """The dialogue as plain text, for simulator and judge prompts."""
        names = {"user": user_name, "assistant": agent_name}
        return "\n".join(f"{names[m['role']]}: {m['content']}" for m in self.messages)

    # -- tools ------------------------------------------------------------

    def has_tool_call(self, tool_name: str) -> bool:
        """Whether ``tool_name`` ran cleanly at least once.

        A call that was requested but rejected, errored or paused proves
        nothing about the agent's behaviour. Same semantics as
        ``agno.scorer.ToolCallScorer`` since agno 2.8.
        """
        return any(tool.tool_name == tool_name and _is_clean(tool) for tool in self.tool_executions)

    def tool_calls(self, tool_name: str | None = None) -> list[ToolExecution]:
        """Clean executions of ``tool_name`` (of every tool when ``None``), in call order."""
        return [t for t in self.tool_executions if _is_clean(t) and (tool_name is None or t.tool_name == tool_name)]

    def last_tool_call(self, tool_name: str | None = None) -> ToolExecution | None:
        """Most recent clean execution of ``tool_name`` (of any tool when ``None``)."""
        for tool in reversed(self.tool_executions):
            if _is_clean(tool) and (tool_name is None or tool.tool_name == tool_name):
                return tool
        return None

    def tool_call_names(self) -> list[str]:
        """Names of cleanly executed tools in call order, repeats included."""
        return [tool.tool_name for tool in self.tool_executions if tool.tool_name and _is_clean(tool)]

    def has_tool_calls_in_order(self, tool_names: list[str]) -> bool:
        """Whether the tools ran in this order (a subsequence; other calls may sit between)."""
        it = iter(self.tool_call_names())
        return all(name in it for name in tool_names)

    # -- judge evidence ---------------------------------------------------

    def references_text(self) -> str:
        """The reference material (RAG) the agent saw, as text for the judge prompt.

        Each article's URL follows its text: agents often cite articles by URL,
        and without it the judge took such a link for a hallucination.
        """
        blocks: list[str] = []
        for message_refs in self.references:
            blocks.extend(_reference_block(ref) for ref in message_refs.references or [])
        return "\n---\n".join(blocks)

    def tool_evidence_text(self, max_result_chars: int = 4000, max_total_chars: int = 12000) -> str:
        """Every tool execution with its status and (truncated) result, for the judge.

        Errored calls are included on purpose: stopping with an authorization
        error can be exactly what an auth scenario expects. Sizes are capped so
        one huge tool result cannot push the criteria and the transcript out of
        the judge's context.
        """
        blocks: list[str] = []
        for tool in self.tool_executions:
            status = "paused" if tool.is_paused else "error" if tool.tool_call_error else "success"
            block = f"Tool: {tool.tool_name or 'unknown'}\nStatus: {status}"
            if tool.tool_args:
                block += f"\nArguments: {tool.tool_args}"
            result = (tool.result or "").strip()
            if result:
                block += f"\nResult:\n{result[:max_result_chars]}"
            blocks.append(block)
        return "\n---\n".join(blocks)[:max_total_chars]


type CriterionStatus = Literal["passed", "failed", "inconclusive"]


class CriterionVerdict(BaseModel):
    """The judge's verdict on one criterion.

    Field order is the order the judge fills them in: restate the criterion as
    a positive requirement, reason about it, and only then pick a status.
    """

    criterion: str = Field(description="The criterion, copied verbatim")
    requirement: str = Field(
        default="",
        description="The criterion restated as a positive requirement the agent has to meet",
    )
    reasoning: str = Field(description="Short justification grounded in the dialogue and the evidence")
    status: CriterionStatus = Field(
        description="passed, failed, or inconclusive when the evidence to decide is not available"
    )

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def inconclusive(self) -> bool:
        return self.status == "inconclusive"


class JudgeScore(BaseModel):
    """Normalised numeric grade from ``agno.scorer.JudgeScorer``."""

    value: float = Field(ge=0.0, le=1.0, description="Grade normalised from the 1-10 scale to 0-1")
    raw_score: int = Field(ge=1, le=10, description="The judge's raw 1-10 grade")
    threshold: int = Field(ge=1, le=10, description="Minimum raw grade that passes")
    passed: bool
    reasoning: str | None = None


class ScenarioStatus(StrEnum):
    """Scenario outcome in reports.

    WARN means the scenario could not be verified, which is kept apart from
    FAIL on purpose. Either the harness broke (the judge or the simulator
    returned no structured output, the gateway did not answer), or the judge
    found no evidence to decide some criteria and none of the others failed.
    In both cases the ability to check failed, not the check; counting it as
    FAIL makes the pass rate measure helper-LLM reliability and scenario
    design instead of agent quality.
    """

    PASS = "pass"  # nosec B105 - a status, not a password
    FAIL = "fail"
    WARN = "warn"


@dataclass
class ScenarioResult:
    """Outcome of one scenario: overall success plus a per-criterion breakdown."""

    name: str
    description: str
    success: bool
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    judge_score: JudgeScore | None = None
    messages: list[Message] = field(default_factory=list)
    # Names of cleanly executed tools in call order; every report shows them.
    tool_calls: list[str] = field(default_factory=list)
    turns: int = 0
    # Why the scenario ended the way it did: succeed()/fail() reasoning or the
    # judge's summary. None when only the per-criterion verdicts explain it.
    reasoning: str | None = None
    # Assertion from a script step or a runtime error; None when only criteria failed.
    failure_reason: str | None = None
    # The scenario did not finish because the harness broke (SimulationHarnessError).
    harness_error: bool = False
    # Wall-clock seconds for the whole scenario and for the agent's turns alone.
    total_time: float = 0.0
    agent_time: float = 0.0

    @property
    def status(self) -> ScenarioStatus:
        if self.success:
            return ScenarioStatus.PASS
        if self.harness_error:
            return ScenarioStatus.WARN
        score_ok = self.judge_score is None or self.judge_score.passed
        if self.failure_reason is None and score_ok and self.inconclusive_criteria and not self.failed_criteria:
            return ScenarioStatus.WARN
        return ScenarioStatus.FAIL

    @property
    def passed_criteria(self) -> list[str]:
        return [v.criterion for v in self.verdicts if v.passed]

    @property
    def failed_criteria(self) -> list[str]:
        return [v.criterion for v in self.verdicts if v.failed]

    @property
    def inconclusive_criteria(self) -> list[str]:
        return [v.criterion for v in self.verdicts if v.inconclusive]

    def failure_summary(self) -> str:
        """Readable failure explanation, for ``assert result.success, result.failure_summary()``."""
        if self.success:
            return ""
        if self.harness_error:
            headline = "did not finish: harness failure"
        elif self.status is ScenarioStatus.WARN:
            headline = "could not be verified"
        else:
            headline = "failed"
        parts = [f"scenario {self.name!r} {headline}"]
        if self.failure_reason:
            parts.append(f"reason: {self.failure_reason}")
        if self.reasoning and self.reasoning != self.failure_reason:
            parts.append(f"judge: {self.reasoning}")
        if self.judge_score is not None and not self.judge_score.passed:
            parts.append(f"judge score {self.judge_score.raw_score}/10 below {self.judge_score.threshold}")
        parts.extend(f"criterion {v.status}: {v.criterion} ({v.reasoning})" for v in self.verdicts if not v.passed)
        return "; ".join(parts)
