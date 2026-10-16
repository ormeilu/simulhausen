"""Fakes for the Harbor tests: a python target factory, a scripted simulator and judge, scenario files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from agno.models.message import MessageReferences
from agno.models.response import ToolExecution

import simulhausen as sim
from simulhausen import (
    AgentInput,
    AgentTurn,
    CriterionStatus,
    CriterionVerdict,
    JudgeAgent,
    JudgeScore,
    JudgeVerdict,
    ScenarioState,
    UserSimulatorAgent,
    UserTurn,
)

if TYPE_CHECKING:
    from pathlib import Path

# The target every scenario in these tests talks to: support_agent() below.
TARGET: dict[str, Any] = {"type": "python", "factory": "harbor_support:support_agent"}


def support_agent(
    prefix: str = "agent",
    tools: Sequence[str] = ("lookup_order",),
    fail_on: str | None = None,
    api_key: str | None = None,
):
    """A python target: replies "<prefix> reply N: <user text>", runs ``tools`` on its first turn.

    The first turn also brings a RAG reference. ``fail_on`` makes it raise when the user
    message contains that text, like an agent that crashes. ``api_key`` shows up in the
    replies, so tests can see it arrived.
    """
    if api_key is not None:
        prefix = f"{prefix} ({api_key})"
    calls: list[AgentInput] = []

    def reply(agent_input: AgentInput) -> AgentTurn:
        calls.append(agent_input)
        text = agent_input.last_new_user_message_str()
        if fail_on is not None and fail_on in text:
            raise RuntimeError(f"agent crashed on {text!r}")
        executions = None
        references = None
        if len(calls) == 1:
            executions = [
                ToolExecution(tool_call_id=f"call-{name}", tool_name=name, tool_args={"order": "1042"}, result="ok")
                for name in tools
            ]
            references = [MessageReferences(query=text, references=[{"content": "refund policy"}])]
        return AgentTurn(content=f"{prefix} reply {len(calls)}: {text}", tools=executions, references=references)

    return sim.function_adapter(reply)


def broken_factory(**options: Any) -> str:
    """A python target factory that returns something that is not an adapter."""
    return "not an adapter"


class AgnoLike:
    """Stands in for an agno Agent: has ``arun``."""

    async def arun(self, input: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(content=f"agno says: {input}", status=None, tools=None, references=None)


agno_like_instance = AgnoLike()
adapter_instance = support_agent()
not_callable = 42


def agno_like_factory() -> AgnoLike:
    return AgnoLike()


class ScriptedSimulator(UserSimulatorAgent):
    """Replays turns (plain strings are ordinary messages), then invents "user message N"."""

    def __init__(self, *turns: UserTurn | str | Exception) -> None:
        super().__init__()
        self.turns = list(turns)
        self.calls = 0

    async def next_turn(self, state: ScenarioState) -> UserTurn:
        self.calls += 1
        if self.turns:
            turn = self.turns.pop(0)
            if isinstance(turn, Exception):
                raise turn
            return UserTurn(message=turn, goal_achieved=False) if isinstance(turn, str) else turn
        return UserTurn(message=f"user message {self.calls}", goal_achieved=False)


def goal_reached() -> UserTurn:
    return UserTurn(message="", goal_achieved=True)


class ScriptedJudge(JudgeAgent):
    """Answers from a script: one status per criterion (default all passed), per call in order.

    ``statuses`` maps a criterion to its status; an ``Exception`` in ``errors`` is raised
    on the call with that index. Every call is recorded with the transcript size it saw.
    """

    def __init__(
        self,
        criteria: Sequence[str] = (),
        *,
        statuses: dict[str, CriterionStatus] | None = None,
        score: int | None = None,
        errors: dict[int, Exception] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(criteria, **options)
        self.statuses = statuses or {}
        self.raw_score = score
        self.errors = errors or {}
        self.calls: list[tuple[list[str], int, str | None]] = []
        self.score_calls = 0

    async def judge(
        self,
        state: ScenarioState,
        criteria: Sequence[str] | None = None,
        *,
        additional_context: str | None = None,
    ) -> JudgeVerdict:
        used = list(self.criteria if criteria is None else criteria)
        if (error := self.errors.get(len(self.calls))) is not None:
            raise error
        self.calls.append((used, len(state.messages), additional_context))
        verdicts = [
            CriterionVerdict(criterion=c, reasoning=f"reason for {c}", status=self.statuses.get(c, "passed"))
            for c in used
        ]
        return JudgeVerdict(verdicts=verdicts, summary=f"summary of {len(used)} criteria")

    async def score(self, state: ScenarioState) -> JudgeScore | None:
        self.score_calls += 1
        if self.score_threshold is None or self.raw_score is None:
            return None
        raw = self.raw_score
        return JudgeScore(
            value=(raw - 1) / 9, raw_score=raw, threshold=self.score_threshold, passed=raw >= self.score_threshold
        )


def write_scenarios(path: Path, payload: dict[str, Any]) -> Path:
    """Write a scenario file as JSON (the loader treats it like the TOML form)."""
    path.write_text(json.dumps(payload))
    return path
