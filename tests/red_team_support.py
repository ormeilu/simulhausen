"""Fakes for the red-team tests: a scripted stand-in for ``agno.Agent`` and a scripted target."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

from simulhausen.red_team import AttackerTurn
from simulhausen.red_team.agent import ResponseScore

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from agno.models.base import Model

MODEL = cast("Model", object())
PLANNER = cast("Model", object())
PARSER = cast("Model", object())

type Role = Literal["plan", "score", "attack"]


@dataclass
class LLMCall:
    """One call to the fake agent: how it was built, and what it was asked."""

    role: Role
    built: dict[str, Any]
    prompt: str

    @property
    def instructions(self) -> str:
        return str(self.built.get("instructions", ""))

    @property
    def history(self) -> list[tuple[str, str]]:
        """The attacker's messages as the model got them: ``(role, text)``, the prompt last."""
        earlier = [(m.role, m.get_content_string()) for m in self.built.get("additional_input", [])]
        return [*earlier, ("user", self.prompt)]


@dataclass
class FakeLLM:
    """Plays every agno agent the red-team agent builds: planner, scorer and attacker.

    The agent's role follows from how it was built. Queued answers are used first,
    then the defaults: a plan "PLAN", a score of 2, "attack N" (plain) or a
    GOAT-style structured turn. ``scorer`` decides a score from the prompt instead
    of the queue.
    """

    plans: list[object] = field(default_factory=list)
    scores: list[object] = field(default_factory=list)
    attacks: list[object] = field(default_factory=list)
    scorer: Callable[[str], object] | None = None
    calls: list[LLMCall] = field(default_factory=list)

    def of(self, role: Role) -> list[LLMCall]:
        return [call for call in self.calls if call.role == role]

    def attack_messages(self) -> list[str]:
        """The latest user message of every attacker call: what the attacker was last shown."""
        return [call.prompt for call in self.of("attack")]

    def _answer(self, role: Role, built: dict[str, Any], prompt: str) -> object:
        if role == "plan":
            return self.plans.pop(0) if self.plans else "PLAN"
        if role == "score":
            if self.scorer is not None:
                return self.scorer(prompt)
            return self.scores.pop(0) if self.scores else ResponseScore(score=2, adaptation="keep probing")
        if self.attacks:
            return self.attacks.pop(0)
        number = len(self.of("attack"))
        if built.get("output_schema") is AttackerTurn:
            return AttackerTurn(observation=f"obs {number}", strategy="HYPOTHETICAL FRAMING", reply=f"attack {number}")
        return f"attack {number}"

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = self

        class FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                self.built = kwargs

            async def arun(self, prompt: str) -> SimpleNamespace:
                schema = self.built.get("output_schema")
                role: Role = (
                    "score" if schema is ResponseScore else "attack" if "instructions" in self.built else "plan"
                )
                llm.calls.append(LLMCall(role, self.built, prompt))
                return SimpleNamespace(content=llm._answer(role, self.built, prompt))

        monkeypatch.setattr("simulhausen.red_team.agent.Agent", FakeAgent)
