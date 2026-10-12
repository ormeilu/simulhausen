"""The strategy interface of the red-team agent and the attacker's structured turn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

__all__ = ["AttackerTurn", "RedTeamStrategy"]


class AttackerTurn(BaseModel):
    """One attacker turn: the message the target sees plus the attacker's reasoning.

    Strategies that ask for structured output (GOAT) get the model to fill all
    three fields, in this order: observing first and choosing a technique
    second makes the attacker commit to its reasoning before it writes the
    message. Only ``reply`` reaches the target.
    """

    observation: str = Field(
        description=(
            "One sentence on what the target's last response revealed "
            '(defenses, partial compliance, tone). Use "" on turn 1'
        )
    )
    strategy: str = Field(
        description=(
            "One sentence naming the technique(s) used THIS turn and why, referring to the catalogue where applicable"
        )
    )
    reply: str = Field(
        description="The actual message to send to the target agent. This is the ONLY field the target sees"
    )


class RedTeamStrategy(ABC):
    """How an attack unfolds over the turns: the attacker's system prompt for each turn.

    A strategy only builds prompts. The agent owns everything else: the
    attacker's private history, scoring, backtracking, encodings and early
    exit. Score feedback, adaptation hints and backtrack notes reach the
    attacker through that history, not through the system prompt.

    ``attack_number`` is 1-based and counts attacks, not agent replies: it is
    ``state.current_turn + 1``. A backtracked attack and its retry share one
    number, since ``current_turn`` is not lowered by a rollback.
    """

    # Reported as "strategy" in ``RedTeamAgent.report()``.
    name: ClassVar[str] = "custom"

    @abstractmethod
    def build_system_prompt(
        self,
        *,
        target: str,
        attack_number: int,
        total_turns: int,
        scenario_description: str,
        attack_plan: str = "",
    ) -> str:
        """The attacker's system prompt for this attack."""

    @abstractmethod
    def get_phase_name(self, attack_number: int, total_turns: int) -> str:
        """The label of the phase this attack falls in, for the report and the logs."""

    def template_variables(self, total_turns: int) -> dict[str, object]:
        """Extra fields for the attack-plan template (Crescendo adds its phase boundaries)."""
        return {}

    @property
    def needs_metaprompt_plan(self) -> bool:
        """Whether the agent must generate an attack plan before the first attack.

        Staged strategies depend on a plan tailored to the target. Strategies
        that choose a technique from a catalogue each turn do not, so the
        extra model call is skipped and ``attack_plan`` stays unused.
        """
        return True

    @property
    def structured_output(self) -> bool:
        """Whether the attacker answers with an :class:`AttackerTurn` instead of plain text."""
        return False

    @property
    def phase_kind(self) -> Literal["staged", "progress"]:
        """What :meth:`get_phase_name` means: real phases ("staged") or a coarse progress bucket."""
        return "staged"

    def chosen_technique_ids(self, strategy_text: str) -> list[str]:
        """Catalogue technique ids found in the attacker's ``strategy`` field (none by default)."""
        return []
