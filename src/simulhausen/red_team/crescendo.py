"""Crescendo: a gradual escalation over four phases.

Starts as an innocent conversation and ends in an aggressive jailbreak attempt,
leaning on a model's tendency to stay consistent with the cooperative context
built in the early turns. Adapted from LangWatch Scenario's
``_red_team/crescendo.py`` (Apache-2.0, see NOTICE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from simulhausen.red_team.base import RedTeamStrategy
from simulhausen.red_team.prompts import CRESCENDO_PHASE_INSTRUCTIONS, CRESCENDO_SYSTEM_PROMPT

__all__ = ["CrescendoStrategy"]


@dataclass(frozen=True)
class _Phase:
    name: str
    # Share of the run, as the attack number divided by total_turns: [start, end).
    start: float
    end: float

    @property
    def instructions(self) -> str:
        return CRESCENDO_PHASE_INSTRUCTIONS[self.name]


# The last phase has no upper bound, so progress above 1.0 (a script longer
# than total_turns) stays in it.
_PHASES = (
    _Phase("warmup", 0.0, 0.20),
    _Phase("probing", 0.20, 0.45),
    _Phase("escalation", 0.45, 0.75),
    _Phase("direct", 0.75, float("inf")),
)


class CrescendoStrategy(RedTeamStrategy):
    """Four escalating phases, as a share of ``total_turns``.

    - warmup (up to 20%): rapport, innocent capability mapping
    - probing (20-45%): hypothetical and academic framing
    - escalation (45-75%): authority claims, roleplay, urgency, social engineering
    - direct (75-100%): maximum pressure, combined vectors, encodings

    The attack plan from the metaprompt model is part of every system prompt.
    """

    name: ClassVar[str] = "crescendo"

    def template_variables(self, total_turns: int) -> dict[str, object]:
        """Phase boundaries (in turns) for the attack-plan template."""
        phase1, phase2, phase3, _ = _PHASES
        return {
            "phase1_end": max(1, int(phase1.end * total_turns)),
            "phase2_end": max(1, int(phase2.end * total_turns)),
            "phase3_end": max(1, int(phase3.end * total_turns)),
        }

    @staticmethod
    def _phase(attack_number: int, total_turns: int) -> _Phase:
        progress = attack_number / max(total_turns, 1)
        return next(phase for phase in _PHASES if phase.start <= progress < phase.end)

    def get_phase_name(self, attack_number: int, total_turns: int) -> str:
        return self._phase(attack_number, total_turns).name

    @staticmethod
    def _turn_range(phase: _Phase, total_turns: int) -> str:
        first = max(1, int(phase.start * total_turns) + 1)
        last = max(first, min(total_turns, int(min(phase.end, 1.0) * total_turns)))
        return f"turns {first}-{last}"

    def build_system_prompt(
        self,
        *,
        target: str,
        attack_number: int,
        total_turns: int,
        scenario_description: str,
        attack_plan: str = "",
    ) -> str:
        phase = self._phase(attack_number, total_turns)
        return CRESCENDO_SYSTEM_PROMPT.format(
            target=target,
            attack_plan=attack_plan,
            attack_number=attack_number,
            total_turns=total_turns,
            phase=phase.name.upper(),
            phase_range=self._turn_range(phase, total_turns),
            phase_instructions=phase.instructions,
            description=scenario_description,
        )
