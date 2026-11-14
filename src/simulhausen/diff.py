"""Compare two JSON reports: what regressed, what got fixed, what is new.

One run of LLM-judged scenarios says how the agent does today. A change to
the agent, its prompt or its model is judged against the run before it::

    simulhausen report diff reports/main.json reports/pr.json --md reports/diff.md --fail-on-regression

Scenarios are matched by test id and name. A scenario played several times
(``run_trials``) is compared by its pass rate over the judged trials, so a
flaky scenario that goes from 4/5 to 3/5 shows up as a regression of 20
points, and ``tolerance`` says how large a drop has to be to count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from simulhausen.types import Usage

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = ["DIFF_MARKER", "ReportDiff", "ScenarioDelta", "ScenarioSide", "diff_reports", "load_report"]

# Hidden marker at the top of the markdown diff, like the report's own marker.
DIFF_MARKER = "<!-- simulhausen-diff -->"

_EPSILON = 1e-9

# - regressed / improved: the pass rate moved by more than the tolerance;
# - unchanged: both runs have a verdict and the pass rate held;
# - new / removed: the scenario is in one run only;
# - unverified: the head run has no verdict (every trial WARN), so nothing can be said;
# - verified: the base run had no verdict and the head run has one.
type DeltaKind = Literal["regressed", "improved", "unchanged", "new", "removed", "unverified", "verified"]


@dataclass
class ScenarioSide:
    """One scenario in one report, its trials folded together."""

    passed: int = 0
    failed: int = 0
    warned: int = 0
    turns: int = 0
    total_time: float = 0.0
    usage: Usage | None = None
    # Why it did not pass, first trial that says: failed criteria and the failure reason.
    failed_criteria: list[str] = field(default_factory=list)
    failure_reason: str | None = None

    @property
    def judged(self) -> int:
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float | None:
        """PASS / (PASS + FAIL) from 0 to 1; ``None`` when no trial was judged."""
        return self.passed / self.judged if self.judged else None

    @property
    def label(self) -> str:
        """``PASS``, ``FAIL`` or ``WARN`` for one run; ``3/5 passed`` for trials."""
        runs = self.judged + self.warned
        if runs == 1:
            return "PASS" if self.passed else "FAIL" if self.failed else "WARN"
        text = f"{self.passed}/{self.judged} passed" if self.judged else "not verified"
        return f"{text} (+{self.warned} WARN)" if self.warned and self.judged else text

    def add(self, scenario: Mapping[str, Any]) -> None:
        status = str(scenario.get("status") or ("pass" if scenario.get("success") else "fail")).lower()
        if status == "pass":
            self.passed += 1
        elif status == "warn":
            self.warned += 1
        else:
            self.failed += 1
        self.turns += int(scenario.get("turns") or 0)
        self.total_time += float(scenario.get("total_time") or 0.0)
        usage = scenario.get("usage")
        if isinstance(usage, dict):
            read = Usage.from_dict(usage)
            self.usage = read if self.usage is None else self.usage + read
        if status != "pass":
            for verdict in scenario.get("verdicts") or []:
                criterion = str(verdict.get("criterion") or "")
                if verdict.get("status") == "failed" and criterion not in self.failed_criteria:
                    self.failed_criteria.append(criterion)
            if self.failure_reason is None and scenario.get("failure_reason"):
                self.failure_reason = str(scenario["failure_reason"])


@dataclass
class ScenarioDelta:
    """One scenario across the two reports."""

    name: str
    test_id: str | None
    kind: DeltaKind
    base: ScenarioSide | None
    head: ScenarioSide | None

    @property
    def change(self) -> str:
        """``PASS → FAIL``, ``4/5 passed → 2/5 passed``, ``new: FAIL``."""
        if self.base is None:
            return f"new: {self.head.label}" if self.head else "new"
        if self.head is None:
            return f"removed (was {self.base.label})"
        return f"{self.base.label} → {self.head.label}"


def _sides(report: Mapping[str, Any]) -> dict[tuple[str | None, str], ScenarioSide]:
    sides: dict[tuple[str | None, str], ScenarioSide] = {}
    for scenario in report.get("scenarios") or []:
        test_id = scenario.get("test_id")
        key = (str(test_id) if test_id is not None else None, str(scenario.get("name") or ""))
        sides.setdefault(key, ScenarioSide()).add(scenario)
    return sides


def _kind(base: ScenarioSide | None, head: ScenarioSide | None, tolerance: float) -> DeltaKind:
    if base is None:
        return "new"
    if head is None:
        return "removed"
    before, after = base.pass_rate, head.pass_rate
    if after is None:
        return "unverified"
    if before is None:
        return "verified"
    # The epsilon keeps a move of exactly the tolerance (0.8 -> 0.6 at 0.2) inside it despite float rounding.
    if after < before - tolerance - _EPSILON:
        return "regressed"
    if after > before + tolerance + _EPSILON:
        return "improved"
    return "unchanged"


def _total_usage(sides: dict[tuple[str | None, str], ScenarioSide]) -> Usage | None:
    total: Usage | None = None
    for side in sides.values():
        if side.usage is not None:
            total = side.usage if total is None else total + side.usage
    return total


def _rate(sides: dict[tuple[str | None, str], ScenarioSide]) -> float | None:
    passed = sum(side.passed for side in sides.values())
    judged = sum(side.judged for side in sides.values())
    return passed / judged if judged else None


def _percent(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.0%}"


def _signed(value: float, unit: str = "", digits: int = 0) -> str:
    return f"{value:+,.{digits}f}{unit}"


@dataclass
class ReportDiff:
    """Two reports side by side: per-scenario deltas and the totals of both runs."""

    deltas: list[ScenarioDelta]
    base_pass_rate: float | None
    head_pass_rate: float | None
    base_time: float
    head_time: float
    base_usage: Usage | None
    head_usage: Usage | None
    tolerance: float = 0.0

    def of_kind(self, *kinds: DeltaKind) -> list[ScenarioDelta]:
        return [delta for delta in self.deltas if delta.kind in kinds]

    @property
    def regressions(self) -> list[ScenarioDelta]:
        """Scenarios whose pass rate fell by more than the tolerance. A new scenario has no baseline to fall from."""
        return self.of_kind("regressed")

    @property
    def improvements(self) -> list[ScenarioDelta]:
        return self.of_kind("improved")

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    def summary(self) -> str:
        """One line for a terminal: counts per kind and the pass rate of both runs."""
        counts = ", ".join(
            f"{len(self.of_kind(kind))} {kind}"
            for kind in ("regressed", "improved", "new", "removed", "unverified", "verified", "unchanged")
            if self.of_kind(kind)
        )
        return (
            f"pass rate {_percent(self.base_pass_rate)} -> {_percent(self.head_pass_rate)}; {counts or 'no scenarios'}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready data."""

        def side(value: ScenarioSide | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {
                "passed": value.passed,
                "failed": value.failed,
                "warned": value.warned,
                "pass_rate": value.pass_rate,
                "turns": value.turns,
                "total_time": value.total_time,
                "usage": value.usage.to_dict() if value.usage else None,
                "failed_criteria": value.failed_criteria,
                "failure_reason": value.failure_reason,
            }

        return {
            "tolerance": self.tolerance,
            "base": {
                "pass_rate": self.base_pass_rate,
                "total_time": self.base_time,
                "usage": self.base_usage.to_dict() if self.base_usage else None,
            },
            "head": {
                "pass_rate": self.head_pass_rate,
                "total_time": self.head_time,
                "usage": self.head_usage.to_dict() if self.head_usage else None,
            },
            "counts": {
                kind: len(self.of_kind(kind))
                for kind in ("regressed", "improved", "unchanged", "new", "removed", "unverified", "verified")
            },
            "scenarios": [
                {
                    "name": delta.name,
                    "test_id": delta.test_id,
                    "kind": delta.kind,
                    "base": side(delta.base),
                    "head": side(delta.head),
                }
                for delta in self.deltas
            ],
        }

    def to_markdown(self) -> str:
        """Markdown for a pull/merge request comment: regressions first, each with what failed."""
        regressions, improvements = self.regressions, self.improvements
        mark = "🔴" if regressions else "🟢"
        lines = [
            DIFF_MARKER,
            (
                f"## 🤖 Simulations vs baseline: {mark} {len(regressions)} regressed · {len(improvements)} improved"
                f" · pass rate {_percent(self.base_pass_rate)} → {_percent(self.head_pass_rate)}"
            ),
            "",
        ]
        sections: tuple[tuple[str, tuple[DeltaKind, ...]], ...] = (
            ("❌ Regressed", ("regressed",)),
            ("✅ Improved", ("improved",)),
            ("🆕 New", ("new",)),
            ("⚠️ Not verified in this run", ("unverified",)),
            ("🗑️ Removed", ("removed",)),
        )
        for title, kinds in sections:
            deltas = self.of_kind(*kinds)
            if not deltas:
                continue
            lines += [f"### {title} ({len(deltas)})", ""]
            for delta in deltas:
                lines.append(f"- **{_md(delta.name)}**: {delta.change}")
                head = delta.head
                if head is not None and delta.kind in ("regressed", "new", "unverified"):
                    lines += [f"  - failed criterion: {_md(criterion)}" for criterion in head.failed_criteria]
                    if head.failure_reason:
                        lines.append(f"  - {_md(head.failure_reason[:300])}")
            lines.append("")
        quiet = len(self.of_kind("unchanged", "verified"))
        lines += [
            "| | Baseline | This run | Change |",
            "|---|--:|--:|--:|",
            (
                f"| Pass rate | {_percent(self.base_pass_rate)} | {_percent(self.head_pass_rate)} | "
                f"{_rate_change(self.base_pass_rate, self.head_pass_rate)} |"
            ),
            (
                f"| Time | {self.base_time:.1f}s | {self.head_time:.1f}s | "
                f"{_signed(self.head_time - self.base_time, 's', 1)} |"
            ),
        ]
        if self.base_usage or self.head_usage:
            before = self.base_usage.total_tokens if self.base_usage else 0
            after = self.head_usage.total_tokens if self.head_usage else 0
            lines.append(f"| Agent tokens | {before:,} | {after:,} | {_signed(after - before)} |")
        lines += ["", f"{quiet} scenario(s) unchanged.", ""]
        return "\n".join(lines)


def _md(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


def _rate_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "-"
    return _signed((after - before) * 100, " pt")


def diff_reports(base: Mapping[str, Any], head: Mapping[str, Any], *, tolerance: float = 0.0) -> ReportDiff:
    """Compare two reports (the parsed ``--sim-report`` JSON): ``base`` is the run to measure against.

    ``tolerance`` is how far a scenario's pass rate (0 to 1) may move before it
    counts as regressed or improved. With single runs any flip counts; with
    trials, ``tolerance=0.2`` lets one trial out of five go the other way.
    """
    if not 0.0 <= tolerance < 1.0:
        raise ValueError(f"tolerance must be at least 0 and below 1, got {tolerance}")
    base_sides, head_sides = _sides(base), _sides(head)
    order: dict[tuple[str | None, str], None] = dict.fromkeys([*head_sides, *base_sides])
    deltas = [
        ScenarioDelta(
            name=key[1],
            test_id=key[0],
            kind=_kind(base_sides.get(key), head_sides.get(key), tolerance),
            base=base_sides.get(key),
            head=head_sides.get(key),
        )
        for key in order
    ]
    return ReportDiff(
        deltas=deltas,
        base_pass_rate=_rate(base_sides),
        head_pass_rate=_rate(head_sides),
        base_time=sum(side.total_time for side in base_sides.values()),
        head_time=sum(side.total_time for side in head_sides.values()),
        base_usage=_total_usage(base_sides),
        head_usage=_total_usage(head_sides),
        tolerance=tolerance,
    )


def load_report(path: Path) -> dict[str, Any]:
    """Read a ``--sim-report`` JSON file; anything else raises ``ValueError`` naming the file."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the report {path}: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("scenarios"), list):
        raise ValueError(f"{path} is not a simulhausen JSON report (no 'scenarios' list)")  # noqa: TRY004 - bad file content, not a bad argument type
    return report
