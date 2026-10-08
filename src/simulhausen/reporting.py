"""Result collection and reports: rich console table, markdown (PR/MR comment), JSON."""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from simulhausen.config import get_config
from simulhausen.types import CriterionVerdict, JudgeScore, ScenarioResult, ScenarioStatus

if TYPE_CHECKING:
    from simulhausen.types import Message

__all__ = [
    "REPORT_MARKER",
    "Collector",
    "ScenarioRecord",
    "SummaryStats",
    "collector",
    "deserialize_records",
    "render_console",
    "serialize_records",
    "summarize",
    "to_json",
    "to_markdown",
]

# Hidden marker at the top of the markdown report, so bots and scripts can find
# (or filter) simulation comments on a pull/merge request.
REPORT_MARKER = "<!-- simulhausen-report -->"

_current_test: ContextVar[str | None] = ContextVar("simulhausen_current_test", default=None)


@dataclass
class ScenarioRecord:
    """One scenario result, tied to the pytest test that produced it."""

    result: ScenarioResult
    test_id: str | None = None


@dataclass
class Collector:
    """Accumulates results over a session; ``run()`` feeds it."""

    records: list[ScenarioRecord] = field(default_factory=list)

    def set_current_test(self, nodeid: str | None) -> None:
        _current_test.set(nodeid)

    def drop_test_records(self, nodeid: str) -> None:
        """Forget a test's records before it reruns (pytest-rerunfailures):
        the report keeps only the last attempt, without duplicates."""
        self.records = [r for r in self.records if r.test_id != nodeid]

    def add(self, result: ScenarioResult) -> None:
        self.records.append(ScenarioRecord(result=result, test_id=_current_test.get()))

    def clear(self) -> None:
        self.records = []

    @property
    def passed(self) -> list[ScenarioRecord]:
        return [r for r in self.records if r.result.status is ScenarioStatus.PASS]

    @property
    def failed(self) -> list[ScenarioRecord]:
        """Real failures; WARN (not verified) is not among them."""
        return [r for r in self.records if r.result.status is ScenarioStatus.FAIL]

    @property
    def warned(self) -> list[ScenarioRecord]:
        return [r for r in self.records if r.result.status is ScenarioStatus.WARN]


collector = Collector()

_CONSOLE_STATUS = {
    ScenarioStatus.PASS: "[green]PASS[/green]",
    ScenarioStatus.FAIL: "[red]FAIL[/red]",
    ScenarioStatus.WARN: "[yellow]WARN[/yellow]",
}

_MD_STATUS = {
    ScenarioStatus.PASS: "✅ PASS",
    ScenarioStatus.FAIL: "❌ FAIL",
    ScenarioStatus.WARN: "⚠️ WARN",
}

_CONSOLE_MARK = {"passed": "[green]✓[/green]", "failed": "[red]✗[/red]", "inconclusive": "[yellow]?[/yellow]"}
_MD_MARK = {"passed": "✅", "failed": "❌", "inconclusive": "❔"}

# Pass-rate bands for the headline emoji. A boolean headline (any failure is
# red) cannot tell a regression from noise once a suite has dozens of
# LLM-judged scenarios, so the colour follows the pass rate instead.
PASS_RATE_GREEN = 90.0
PASS_RATE_YELLOW = 75.0

# Progress bar cells: ▓ passed, ▒ not verified (WARN), ░ failed.
_BAR_WIDTH = 10
_BAR_PASS = "▓"  # nosec B105 - a bar cell, not a password
_BAR_WARN = "▒"
_BAR_FAIL = "░"


@dataclass
class SummaryStats:
    """Run totals shared by the console and markdown reports."""

    total: int
    passed: int
    warned: int
    turns_total: int
    tool_calls_total: int
    unique_tools: list[str]
    criteria_total: int
    criteria_passed: int
    criteria_inconclusive: int
    script_errors: int
    total_time: float
    agent_time: float

    @property
    def failed(self) -> int:
        return self.total - self.passed - self.warned

    @property
    def judged(self) -> int:
        """Scenarios that got a verdict at all (PASS + FAIL)."""
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        """Share of passed among judged scenarios. WARN is neither numerator nor
        denominator: a scenario that could not be verified says nothing about the agent."""
        return self.passed / self.judged * 100 if self.judged else 0.0


def summarize(records: list[ScenarioRecord]) -> SummaryStats:
    results = [r.result for r in records]
    seen: dict[str, None] = {}
    for result in results:
        for tool in result.tool_calls:
            seen.setdefault(tool)
    return SummaryStats(
        total=len(results),
        passed=sum(r.status is ScenarioStatus.PASS for r in results),
        warned=sum(r.status is ScenarioStatus.WARN for r in results),
        turns_total=sum(r.turns for r in results),
        tool_calls_total=sum(len(r.tool_calls) for r in results),
        unique_tools=list(seen),
        criteria_total=sum(len(r.verdicts) for r in results),
        criteria_passed=sum(1 for r in results for v in r.verdicts if v.passed),
        criteria_inconclusive=sum(1 for r in results for v in r.verdicts if v.inconclusive),
        script_errors=sum(1 for r in results if r.failure_reason and not r.harness_error),
        total_time=sum(r.total_time for r in results),
        agent_time=sum(r.agent_time for r in results),
    )


def progress_bar(stats: SummaryStats) -> str:
    """PASS/WARN/FAIL shares as exactly ``_BAR_WIDTH`` cells of ▓▒░.

    The bar counts ALL scenarios while the pass rate next to it counts only
    judged ones (WARN excluded). The mismatch is deliberate: the rate answers
    "how well does the agent do", the bar "how much of the suite did we check".
    """
    parts = [(_BAR_PASS, stats.passed), (_BAR_WARN, stats.warned), (_BAR_FAIL, stats.failed)]
    if stats.total <= 0:
        return _BAR_FAIL * _BAR_WIDTH
    exact = {char: count / stats.total * _BAR_WIDTH for char, count in parts if count}
    # Every non-empty category gets at least one cell: otherwise a single
    # failure rounds down to zero and the bar shows a spotless run.
    cells = {char: max(1, int(share)) for char, share in exact.items()}
    # Largest remainder: hand out the cells lost to rounding down, or take back
    # the ones the one-cell minimum overspent, by distance from the exact share.
    while sum(cells.values()) < _BAR_WIDTH:
        cells[max(cells, key=lambda char: exact[char] - cells[char])] += 1
    while sum(cells.values()) > _BAR_WIDTH:
        cells[min((c for c in cells if cells[c] > 1), key=lambda char: exact[char] - cells[char])] -= 1
    return "".join(char * cells[char] for char, _ in parts if char in cells)


def status_emoji(stats: SummaryStats) -> str:
    """Headline colour from the pass rate; 🟢 also needs full coverage.

    Green with WARNs present would lie: a run where part of the suite was not
    verified and the rest passed would show 🟢 100%.
    """
    if stats.pass_rate >= PASS_RATE_GREEN and not stats.warned:
        return "🟢"
    return "🟡" if stats.pass_rate >= PASS_RATE_YELLOW else "🔴"


def _chat_history(messages: list[Message]) -> str:
    """The dialogue as ``User: ...`` / ``Agent: ...`` lines (names from configure())."""
    config = get_config()
    names = {"user": config.user_name, "assistant": config.agent_name}
    return "\n".join(f"{names.get(m['role'], m['role'])}: {m['content']}" for m in messages)


def _criteria_count(result: ScenarioResult) -> str:
    return f"{len(result.passed_criteria)}/{len(result.verdicts)}"


def render_console(records: list[ScenarioRecord]) -> None:
    """Final rich table plus a summary line (called from pytest_terminal_summary)."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    stats = summarize(records)
    warn_title = f", WARN {stats.warned}" if stats.warned else ""
    table = Table(
        title=f"Simulations: {stats.passed}/{stats.judged} passed ({stats.pass_rate:.0f}%){warn_title}",
        show_lines=True,
        title_style="bold",
    )
    table.add_column("Scenario", overflow="fold")
    table.add_column("Status", justify="center")
    table.add_column("Turns", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Tools (in order)", overflow="fold")
    table.add_column("Criteria and judge verdicts", overflow="fold")
    table.add_column("Dialogue", overflow="fold", max_width=48)

    for record in records:
        result = record.result
        details: list[str] = []
        if result.failure_reason:
            details.append(f"[red]{escape(result.failure_reason)}[/red]")
        details.extend(
            f"{_CONSOLE_MARK[v.status]} {escape(v.criterion)} [dim]({escape(v.reasoning)})[/dim]"
            for v in result.verdicts
        )
        if result.judge_score is not None:
            score = result.judge_score
            mark = _CONSOLE_MARK["passed" if score.passed else "failed"]
            details.append(
                f"{mark} judge score {score.raw_score}/10 (threshold {score.threshold}) "
                f"[dim]({escape(score.reasoning or '-')})[/dim]"
            )
        if result.reasoning:
            details.append(f"[dim]{escape(result.reasoning)}[/dim]")
        tools = "\n".join(f"{i}. {escape(name)}" for i, name in enumerate(result.tool_calls, 1)) or "-"
        table.add_row(
            escape(result.name),
            _CONSOLE_STATUS[result.status],
            str(result.turns),
            f"{result.total_time:.1f}s",
            tools,
            "\n".join(details) or "-",
            escape(_chat_history(result.messages)) or "-",
        )

    console.print()
    console.print(table)
    console.print(
        f"[bold]Summary:[/bold] pass rate {stats.pass_rate:.0f}% "
        f"([green]{stats.passed} passed[/green] / [red]{stats.failed} failed[/red] of {stats.judged} judged); "
        f"[yellow]{stats.warned} not verified (WARN)[/yellow] of {stats.total} total; "
        f"criteria {stats.criteria_passed}/{stats.criteria_total} passed "
        f"({stats.criteria_inconclusive} inconclusive); "
        f"turns {stats.turns_total}; tool calls {stats.tool_calls_total} "
        f"({len(stats.unique_tools)} unique); script/runtime errors {stats.script_errors}; "
        f"time {stats.total_time:.1f}s (agent {stats.agent_time:.1f}s)"
    )


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Cap on an error's length in the markdown report: the gist of the assertion
# shows, while huge payloads (whole agent replies inside an assert message) do
# not bloat the comment; the full dialogue is in the section below anyway.
FAILURE_REASON_LIMIT = 600


def _failure_block(reason: str, harness_error: bool = False) -> list[str]:
    """The error as <pre><code>: line breaks survive and the markdown stays intact."""
    text = reason if len(reason) <= FAILURE_REASON_LIMIT else reason[:FAILURE_REASON_LIMIT] + "\n… (truncated)"
    label = "**Harness failure (WARN, not an agent failure):**" if harness_error else "**Script/runtime error:**"
    return [label, "", f"<pre><code>{_html_escape(text)}</code></pre>", ""]


def _ci_job_url() -> str:
    """Link to the CI job, from GitLab or GitHub Actions environment variables."""
    if url := os.environ.get("CI_JOB_URL"):
        return url
    server, repo, run_id = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return ""


def to_markdown(records: list[ScenarioRecord]) -> str:
    """Markdown for a pull/merge request comment (GitHub and GitLab flavours).

    An overview table, every scenario's details (criteria, tools, dialogue)
    folded under one <details> block, and a summary table at the end.
    """
    stats = summarize(records)
    warn_part = f" · ⚠️ WARN {stats.warned}" if stats.warned else ""
    lines = [
        REPORT_MARKER,
        (
            f"## 🤖 Simulations: {status_emoji(stats)} {stats.pass_rate:.0f}% {progress_bar(stats)}"
            f" · PASS {stats.passed} / FAIL {stats.failed}{warn_part}"
        ),
        "",
    ]
    if job_url := _ci_job_url():
        lines += [f"CI job: {job_url}", ""]

    lines += [
        "<details>",
        f"<summary>📋 <b>Scenario overview</b> ({stats.total})</summary>",
        "",
        "| Scenario | Status | Criteria | Turns | Time | Tools (in order) |",
        "|---|:-:|:-:|--:|--:|---|",
    ]
    for record in records:
        result = record.result
        tools = " → ".join(f"`{name}`" for name in result.tool_calls) or "-"
        lines.append(
            f"| {_md_escape(result.name)} | {_MD_STATUS[result.status]} | {_criteria_count(result)} "
            f"| {result.turns} | {result.total_time:.1f}s | {tools} |"
        )
    lines += ["", "</details>", ""]

    # Per-scenario sections go under one more <details>: with 40+ scenarios the
    # comment unfolded into a wall of headings that buried the overview and summary.
    lines += ["<details>", f"<summary>🔍 <b>Detailed results</b> ({stats.total})</summary>", ""]
    for record in records:
        result = record.result
        lines += [
            "<details>",
            (
                f"<summary>{_MD_STATUS[result.status]} ({_criteria_count(result)}) · "
                f"<b>{_html_escape(result.name)}</b> · {result.turns} turn(s)</summary>"
            ),
            "",
        ]
        if result.failure_reason:
            lines += _failure_block(result.failure_reason, harness_error=result.harness_error)
        if result.reasoning:
            lines += [f"**Judge summary:** _{_md_escape(result.reasoning)}_", ""]
        if result.judge_score is not None:
            score = result.judge_score
            score_reason = f" (_{score.reasoning}_)" if score.reasoning else ""
            lines += [
                (
                    f"**Judge score:** {_MD_MARK['passed' if score.passed else 'failed']} "
                    f"{score.raw_score}/10 (threshold {score.threshold}){score_reason}"
                ),
                "",
            ]
        if result.verdicts:
            lines.append("**Criteria:**")
            lines += [f"- {_MD_MARK[v.status]} {v.criterion}: _{v.reasoning}_" for v in result.verdicts]
        if result.tool_calls:
            lines += ["", "**Tools:** " + " → ".join(f"`{name}`" for name in result.tool_calls)]
        if history := _chat_history(result.messages):
            lines += ["", "**Dialogue:**", "", "```text", history, "```"]
        lines += ["", "</details>", ""]
    lines += ["</details>", ""]

    lines += [
        "### 📊 Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        (
            f"| Pass rate | **{stats.pass_rate:.0f}%** "
            f"(PASS {stats.passed} / FAIL {stats.failed} of {stats.judged} judged) |"
        ),
        (
            f"| ⚠️ WARN (not verified) | {stats.warned} of {stats.total}: harness failure or inconclusive "
            "criteria; not counted in the pass rate |"
        ),
        (
            f"| Judge criteria | {stats.criteria_passed}/{stats.criteria_total} passed, "
            f"{stats.criteria_inconclusive} inconclusive |"
        ),
        f"| Dialogue turns | {stats.turns_total} |",
        (f"| Tool calls | {stats.tool_calls_total} ({', '.join(f'`{t}`' for t in stats.unique_tools) or '-'}) |"),
        f"| Script/runtime errors | {stats.script_errors} |",
        f"| Time | {stats.total_time:.1f}s total, {stats.agent_time:.1f}s in the agent |",
        "",
    ]
    return "\n".join(lines)


def to_json(records: list[ScenarioRecord]) -> str:
    """JSON report (a CI artifact): full results, transcripts included."""
    stats = summarize(records)
    payload = {
        "total": stats.total,
        "passed": stats.passed,
        "failed": stats.failed,
        "warned": stats.warned,
        "pass_rate": round(stats.pass_rate, 2),
        "scenarios": serialize_records(records),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def serialize_records(records: list[ScenarioRecord]) -> list[dict[str, Any]]:
    """Plain, execnet-safe payload (the pytest-xdist worker -> controller hop)."""
    return [
        {
            "test_id": record.test_id,
            "name": record.result.name,
            "description": record.result.description,
            "success": record.result.success,
            "status": record.result.status.value,
            "harness_error": record.result.harness_error,
            "turns": record.result.turns,
            "reasoning": record.result.reasoning,
            "failure_reason": record.result.failure_reason,
            "tool_calls": record.result.tool_calls,
            "total_time": record.result.total_time,
            "agent_time": record.result.agent_time,
            "judge_score": record.result.judge_score.model_dump(mode="json") if record.result.judge_score else None,
            "verdicts": [verdict.model_dump(mode="json") for verdict in record.result.verdicts],
            "messages": [dict(message) for message in record.result.messages],
        }
        for record in records
    ]


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def deserialize_records(payload: list[dict[str, Any]]) -> list[ScenarioRecord]:
    """Rebuild records received from a pytest-xdist worker."""
    records: list[ScenarioRecord] = []
    for item in payload:
        judge_score = item.get("judge_score")
        messages: list[Message] = [
            {"role": "assistant" if message["role"] == "assistant" else "user", "content": str(message["content"])}
            for message in item.get("messages") or []
        ]
        records.append(
            ScenarioRecord(
                test_id=_optional_str(item.get("test_id")),
                result=ScenarioResult(
                    name=str(item["name"]),
                    description=str(item.get("description") or ""),
                    success=bool(item["success"]),
                    turns=int(item.get("turns") or 0),
                    reasoning=_optional_str(item.get("reasoning")),
                    failure_reason=_optional_str(item.get("failure_reason")),
                    harness_error=bool(item.get("harness_error")),
                    tool_calls=[str(name) for name in item.get("tool_calls") or []],
                    total_time=float(item.get("total_time") or 0.0),
                    agent_time=float(item.get("agent_time") or 0.0),
                    judge_score=JudgeScore.model_validate(judge_score) if judge_score is not None else None,
                    verdicts=[CriterionVerdict.model_validate(verdict) for verdict in item.get("verdicts") or []],
                    messages=messages,
                ),
            )
        )
    return records
