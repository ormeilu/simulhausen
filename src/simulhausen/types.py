"""Core types: adapter input and output, dialogue state, verdicts, results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import comb
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from agno.models.message import MessageReferences
    from agno.models.response import ToolExecution

    from simulhausen.trajectory import ArgsMatch, ExpectedCall, TrajectoryMatch, TrajectoryMode

__all__ = [
    "AgentAdapter",
    "AgentInput",
    "AgentReturnTypes",
    "AgentRole",
    "AgentTurn",
    "CriterionStatus",
    "CriterionVerdict",
    "JudgeCall",
    "JudgeScore",
    "Message",
    "Role",
    "ScenarioResult",
    "ScenarioState",
    "ScenarioStatus",
    "ToolCall",
    "ToolCallStatus",
    "ToolCalls",
    "TrialsResult",
    "Usage",
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
class Usage:
    """Model usage of the agent under test: tokens, model requests and cost.

    Adapters fill it per turn from what the framework or the server reports;
    the state and the result hold the sum over the scenario. ``cost`` stays
    ``None`` until some turn reports one. Usages add up with ``+``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Model calls behind the turn; 0 when the source does not say.
    requests: int = 0
    cost: float | None = None

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        cost = None if self.cost is None and other.cost is None else (self.cost or 0.0) + (other.cost or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            requests=self.requests + other.requests,
            cost=cost,
        )

    def __bool__(self) -> bool:
        return bool(self.total_tokens or self.requests or self.cost)

    @classmethod
    def of(cls, source: object, *, requests: int | None = None) -> Usage | None:
        """Usage read from a framework's or a server's usage object or dict; ``None`` when it holds none.

        Understands the names the same three numbers go by: ``input_tokens`` /
        ``prompt_tokens`` / ``prompt_token_count`` / ``inputTokens``,
        ``output_tokens`` / ``completion_tokens`` / ``candidates_token_count`` /
        ``outputTokens`` and ``total_tokens`` / ``total_token_count`` /
        ``totalTokens``, plus ``requests`` and ``cost``. ``requests`` overrides
        the model call count when the caller counted it itself.
        """
        if source is None:
            return None

        def read(*names: str) -> float | None:
            for name in names:
                value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return value
            return None

        cost = read("cost", "total_cost_usd")
        usage = cls(
            input_tokens=int(
                read("input_tokens", "prompt_tokens", "prompt_token_count", "input_token_count", "inputTokens") or 0
            ),
            output_tokens=int(
                read(
                    "output_tokens", "completion_tokens", "candidates_token_count", "output_token_count", "outputTokens"
                )
                or 0
            ),
            total_tokens=int(read("total_tokens", "total_token_count", "totalTokens") or 0),
            requests=requests if requests is not None else int(read("requests", "successful_requests") or 0),
            cost=float(cost) if cost is not None else None,
        )
        return usage if usage else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Usage:
        cost = data.get("cost")
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
            requests=int(data.get("requests") or 0),
            cost=float(cost) if cost is not None else None,
        )


@dataclass
class AgentTurn:
    """What the agent under test produced in one turn: text, tool calls, references, usage.

    ``tools`` back assertions such as ``state.has_tool_call(...)``. ``references``
    (the RAG hits from ``RunOutput.references``) are passed on to the judge so it
    grades the reply against the same material the agent saw. ``usage`` is what
    the turn cost in tokens and model requests. ``agno_adapter`` fills all four.
    """

    content: str
    tools: list[ToolExecution] | None = None
    references: list[MessageReferences] | None = None
    usage: Usage | None = None


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


def _tool_status(tool: ToolExecution) -> ToolCallStatus:
    return "paused" if tool.is_paused else "error" if tool.tool_call_error else "success"


type ToolCallStatus = Literal["success", "error", "paused"]


@dataclass(frozen=True)
class ToolCall:
    """One tool execution, as ``state.tool_calls()`` returns it.

    ``turn`` is the 0-based agent turn the call belongs to (``None`` when the
    execution was added to the state by hand). ``status`` is ``"success"`` for
    a clean execution; ``"error"`` and ``"paused"`` calls appear only with
    ``state.tool_calls(include_errors=True)``. The empty call that
    ``ToolCalls.first``/``.last`` return for a tool that never ran is falsy.
    """

    name: str
    input: dict[str, Any] | None = None
    output: str | None = None
    turn: int | None = None
    execution: ToolExecution | None = None
    status: ToolCallStatus = "success"

    def __bool__(self) -> bool:
        return self.execution is not None

    @property
    def ok(self) -> bool:
        """Executed without error and not paused."""
        return self.status == "success"

    @property
    def error(self) -> bool:
        """The execution ended in an error (its text is in ``output``)."""
        return self.status == "error"

    @property
    def paused(self) -> bool:
        """Requested but not run: waiting for confirmation, user input or an external executor."""
        return self.status == "paused"


class ToolCalls(Sequence[ToolCall]):
    """Calls of one tool (or of all tools) in call order.

    A sequence with the picks ``first`` and ``last`` and the columns
    ``inputs`` and ``outputs``, so assertions read naturally::

        assert state.tool_calls("lookup_order").last.input == {"order_id": "1042"}
    """

    def __init__(self, calls: Sequence[ToolCall], name: str | None = None) -> None:
        self._calls = list(calls)
        self._name = name

    @overload
    def __getitem__(self, index: int) -> ToolCall: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ToolCall]: ...

    def __getitem__(self, index: int | slice) -> ToolCall | Sequence[ToolCall]:
        return self._calls[index]

    def __len__(self) -> int:
        return len(self._calls)

    def __iter__(self) -> Iterator[ToolCall]:
        return iter(self._calls)

    def __repr__(self) -> str:
        return f"ToolCalls({self._calls!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ToolCalls):
            return self._calls == other._calls
        if isinstance(other, list):
            return self._calls == other
        return NotImplemented

    def _empty(self) -> ToolCall:
        return ToolCall(name=self._name or "")

    @property
    def first(self) -> ToolCall:
        """The first call; an empty, falsy call when the tool never ran."""
        return self._calls[0] if self._calls else self._empty()

    @property
    def last(self) -> ToolCall:
        """The last call; an empty, falsy call when the tool never ran."""
        return self._calls[-1] if self._calls else self._empty()

    @property
    def inputs(self) -> list[dict[str, Any] | None]:
        """The arguments of every call, in order."""
        return [call.input for call in self._calls]

    @property
    def outputs(self) -> list[str | None]:
        """The result of every call, in order."""
        return [call.output for call in self._calls]


@dataclass
class ScenarioState:
    """Dialogue state: transcript, accumulated tool executions and references."""

    thread_id: str
    name: str
    description: str
    messages: list[Message] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    references: list[MessageReferences] = field(default_factory=list)
    # Agent turns played so far. A rollback does not lower it: rolled-back
    # turns still count against max_turns.
    current_turn: int = 0
    # Usage of the agent under test summed over its turns; None until an adapter
    # reports one. A rollback does not lower it either: the tokens were spent.
    usage: Usage | None = None
    # id() of each tool execution / reference -> index of the agent message that
    # brought it, so a rollback drops what the removed turns produced.
    _origins: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    # Transcript length when the agent last spoke; AgentInput.new_messages starts here.
    _agent_seen: int = field(default=0, repr=False, compare=False)

    @classmethod
    def from_report(
        cls, scenario: Mapping[str, Any], judge_call: int | Mapping[str, Any] | None = None
    ) -> ScenarioState:
        """Rebuild the state a judge call saw, from one scenario of a ``--sim-report`` JSON file.

        ``judge_call`` is an entry of the scenario's ``judge_calls`` (written with
        ``--sim-report-evidence``) or its index there: the state then holds the
        dialogue as it stood at that call, plus its references and tool
        executions, so ``JudgeAgent.judge(state, call["criteria"],
        additional_context=call["additional_context"])`` sends the judge the same
        evidence again. Without ``judge_call`` the state holds the whole
        dialogue and no evidence, since a report without evidence keeps only
        tool names.
        """
        from agno.models.message import MessageReferences
        from agno.models.response import ToolExecution

        messages: list[Message] = [
            {"role": "assistant" if m["role"] == "assistant" else "user", "content": str(m["content"])}
            for m in scenario.get("messages") or []
        ]
        call: Mapping[str, Any] = {}
        if isinstance(judge_call, int):
            call = (scenario.get("judge_calls") or [])[judge_call]
        elif judge_call is not None:
            call = judge_call
        if call:
            messages = messages[: int(call.get("messages_count", len(messages)))]
        return cls(
            thread_id=f"replay-{scenario.get('name', '')}",
            name=str(scenario.get("name") or ""),
            description=str(scenario.get("description") or ""),
            messages=messages,
            references=[MessageReferences.model_validate(r) for r in call.get("references") or []],
            tool_executions=[ToolExecution.from_dict(t) for t in call.get("tool_executions") or []],
            current_turn=sum(1 for m in messages if m["role"] == "assistant"),
        )

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
        usage: Usage | None = None,
    ) -> None:
        self.add_message("assistant", content)
        if usage is not None:
            self.usage = usage if self.usage is None else self.usage + usage
        index = len(self.messages) - 1
        for item in [*(tools or []), *(references or [])]:
            self._origins[id(item)] = index
        if tools:
            self.tool_executions.extend(tools)
        if references:
            self.references.extend(references)

    def rollback_messages_to(self, index: int) -> list[Message]:
        """Remove every message from ``index`` on and return the removed ones.

        Tool executions and references brought by the removed agent messages
        go too. ``index`` is clamped to the transcript length; a negative one
        raises ``ValueError``. ``current_turn`` stays: rolled-back turns still
        count against ``max_turns``. The runner only resends what is left, so
        an agent forgets the removed exchange only if it rebuilds its context
        from the transcript (``agno_adapter(..., history="transcript")``).
        """
        if index < 0:
            raise ValueError(f"rollback index must be >= 0, got {index}")
        index = min(index, len(self.messages))
        removed = self.messages[index:]
        del self.messages[index:]

        def kept(item: object) -> bool:
            return self._origins.get(id(item), -1) < index

        self.tool_executions[:] = [tool for tool in self.tool_executions if kept(tool)]
        self.references[:] = [ref for ref in self.references if kept(ref)]
        self._origins = {key: origin for key, origin in self._origins.items() if origin < index}
        self._agent_seen = min(self._agent_seen, index)
        return removed

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

    def tool_calls(
        self, tool_name: str | None = None, *, include_errors: bool = False, turn: int | None = None
    ) -> ToolCalls:
        """Calls of ``tool_name`` (of every tool when ``None``), in call order.

        Only clean executions by default, like ``has_tool_call``. With
        ``include_errors=True`` the calls that errored or are paused come too,
        told apart by ``ToolCall.status`` (``.ok``, ``.error``, ``.paused``).
        ``turn`` keeps the calls of one 0-based agent turn::

            retried = state.tool_calls("open_login_form", include_errors=True, turn=1)
            assert not retried, "the agent showed the login form again"
        """
        calls = [
            ToolCall(
                name=tool.tool_name or "",
                input=tool.tool_args,
                output=tool.result,
                turn=self.turn_of(tool),
                execution=tool,
                status=_tool_status(tool),
            )
            for tool in self.tool_executions
            if tool.tool_name
            and (include_errors or _is_clean(tool))
            and (tool_name is None or tool.tool_name == tool_name)
        ]
        if turn is not None:
            calls = [call for call in calls if call.turn == turn]
        return ToolCalls(calls, name=tool_name)

    def turn_of(self, item: object) -> int | None:
        """0-based agent turn that brought ``item``, a tool execution or a reference of this state.

        Works for failed and paused executions too, which ``tool_calls()``
        leaves out by default. ``None`` when the item was added to the state by
        hand or is not part of it.
        """
        origin = self._origins.get(id(item))
        if origin is None:
            return None
        return sum(1 for message in self.messages[:origin] if message["role"] == "assistant")

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

    def match_trajectory(
        self,
        expected: Sequence[str | ExpectedCall | tuple[str, Mapping[str, Any] | None]],
        *,
        mode: TrajectoryMode = "in_order",
        args: ArgsMatch = "subset",
        include_errors: bool = False,
    ) -> TrajectoryMatch:
        """Compare the tool calls so far with the expected ones; see :mod:`simulhausen.trajectory`.

        ``mode`` is ``"exact"``, ``"in_order"``, ``"any_order"``, ``"unordered"``
        or ``"subset"``. The result is truthy on a match and explains a
        mismatch::

            match = state.match_trajectory(["search", sim.expect_call("refund", order_id="1042")], mode="exact")
            assert match, match.explanation
        """
        from simulhausen.trajectory import match_trajectory

        return match_trajectory(self.tool_calls(include_errors=include_errors), expected, mode=mode, args=args)

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
            status = _tool_status(tool)
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
    a positive requirement, reason about it, and only then pick a status. With a
    parser model the judge writes free text and the parser fills the fields, so
    the prompt asks for that order in the text as well.

    A ``requirement`` of a single word is no restatement: it is the status the
    parser copied there for want of one ("passed", "fail", "выполнено"), and it
    is stored as ``""``.
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

    @field_validator("requirement")
    @classmethod
    def _drop_status_word(cls, value: str) -> str:
        value = value.strip()
        return value if len(value.split()) > 1 else ""

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


@dataclass
class JudgeCall:
    """One verdict call of the judge and the evidence it ruled on.

    Kept on :class:`ScenarioResult` so a verdict can be debugged or replayed
    later (``--sim-report-evidence`` writes it to the JSON report, and
    :meth:`ScenarioState.from_report` rebuilds the state from it). The judge saw
    ``messages[:messages_count]`` of the scenario's dialogue.
    """

    criteria: list[str]
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    summary: str = ""
    additional_context: str | None = None
    messages_count: int = 0
    references: list[MessageReferences] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    # Wall-clock seconds of the call (a cache hit takes next to none).
    time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready data: the shape ``--sim-report-evidence`` writes."""
        return {
            "criteria": list(self.criteria),
            "verdicts": [verdict.model_dump(mode="json") for verdict in self.verdicts],
            "summary": self.summary,
            "additional_context": self.additional_context,
            "messages_count": self.messages_count,
            "references": [ref.model_dump(mode="json") for ref in self.references],
            "tool_executions": [tool.to_dict() for tool in self.tool_executions],
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JudgeCall:
        from agno.models.message import MessageReferences
        from agno.models.response import ToolExecution

        context = data.get("additional_context")
        return cls(
            criteria=[str(c) for c in data.get("criteria") or []],
            verdicts=[CriterionVerdict.model_validate(v) for v in data.get("verdicts") or []],
            summary=str(data.get("summary") or ""),
            additional_context=str(context) if context is not None else None,
            messages_count=int(data.get("messages_count") or 0),
            references=[MessageReferences.model_validate(r) for r in data.get("references") or []],
            tool_executions=[ToolExecution.from_dict(t) for t in data.get("tool_executions") or []],
            time=float(data.get("time") or 0.0),
        )


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
    # Wall-clock seconds for the whole scenario, the agent's turns, the judge's calls
    # (verdicts, continue/conclude decisions, score) and the user simulator's calls.
    total_time: float = 0.0
    agent_time: float = 0.0
    judge_time: float = 0.0
    simulator_time: float = 0.0
    # Every verdict call of the judge with its evidence, in call order.
    judge_calls: list[JudgeCall] = field(default_factory=list)
    # Tokens, model requests and cost of the agent under test; None when the adapter reports none.
    usage: Usage | None = None
    # 1-based trial number and the number of trials when the scenario came from run_trials().
    trial: int | None = None
    trials: int | None = None

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


@dataclass
class TrialsResult:
    """The same scenario played several times (``run_trials``): every trial and the reliability over them.

    Rates are fractions from 0 to 1 over the judged trials (PASS + FAIL). WARN
    trials say nothing about the agent and stay out, as in every pass rate
    here.
    """

    name: str
    results: list[ScenarioResult] = field(default_factory=list)
    # The exception that ended each trial early (None for a trial that ran to its end), by trial.
    errors: list[Exception | None] = field(default_factory=list, repr=False)
    # Share of judged trials that has to pass for ``success``.
    min_pass_rate: float = 1.0

    def _count(self, status: ScenarioStatus) -> int:
        return sum(result.status is status for result in self.results)

    @property
    def trials(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return self._count(ScenarioStatus.PASS)

    @property
    def failed(self) -> int:
        return self._count(ScenarioStatus.FAIL)

    @property
    def warned(self) -> int:
        return self._count(ScenarioStatus.WARN)

    @property
    def judged(self) -> int:
        """Trials that got a verdict at all (PASS + FAIL)."""
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        """PASS / (PASS + FAIL), 0 when no trial was judged."""
        return self.passed / self.judged if self.judged else 0.0

    def pass_hat(self, k: int | None = None) -> float:
        """pass^k: the chance that ``k`` independent trials all pass, estimated from these trials.

        The unbiased estimate C(passed, k) / C(judged, k) that tau-bench
        reports; ``k`` defaults to the number of judged trials, where it is 1
        only when every one of them passed. It falls as k grows unless the
        agent never fails, which is what tells a reliable agent from a lucky
        one. 0 when no trial was judged; ``k`` above the judged count raises
        ``ValueError``.
        """
        if not self.judged:
            return 0.0
        k = self.judged if k is None else k
        if not 1 <= k <= self.judged:
            raise ValueError(f"k must be between 1 and the {self.judged} judged trial(s), got {k}")
        return comb(self.passed, k) / comb(self.judged, k)

    @property
    def flaky(self) -> bool:
        """Some trials passed and some failed: the outcome depends on the draw."""
        return bool(self.passed and self.failed)

    @property
    def success(self) -> bool:
        """At least one trial was judged and the pass rate reaches ``min_pass_rate``."""
        return bool(self.judged) and self.pass_rate >= self.min_pass_rate

    @property
    def total_time(self) -> float:
        return sum(result.total_time for result in self.results)

    @property
    def usage(self) -> Usage | None:
        """Usage of the agent under test summed over the trials; ``None`` when no trial reported one."""
        total: Usage | None = None
        for result in self.results:
            if result.usage is not None:
                total = result.usage if total is None else total + result.usage
        return total

    def failure_summary(self) -> str:
        """Readable explanation, for ``assert outcome.success, outcome.failure_summary()``."""
        if self.success:
            return ""
        if not self.judged:
            headline = f"scenario {self.name!r} could not be verified in any of {self.trials} trial(s)"
        else:
            headline = (
                f"scenario {self.name!r} passed {self.passed} of {self.judged} judged trial(s) "
                f"({self.pass_rate:.0%}, needs {self.min_pass_rate:.0%})"
            )
            if self.warned:
                headline += f", {self.warned} not verified"
        parts = [headline]
        parts.extend(
            f"trial {result.trial or index}: {result.failure_summary()}"
            for index, result in enumerate(self.results, 1)
            if not result.success
        )
        return "; ".join(parts)
