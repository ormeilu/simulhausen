import json
import re

import pytest

import simulhausen as sim
from simulhausen.reporting import (
    FAILURE_REASON_LIMIT,
    REPORT_MARKER,
    Collector,
    ScenarioRecord,
    SummaryStats,
    collector,
    deserialize_records,
    progress_bar,
    render_console,
    serialize_records,
    status_emoji,
    summarize,
    to_json,
    to_markdown,
)
from simulhausen.types import CriterionVerdict, JudgeScore, ScenarioResult, ScenarioStatus


@pytest.fixture(autouse=True)
def _no_ci_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CI_JOB_URL", "GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
        monkeypatch.delenv(name, raising=False)


def _verdict(status: str, criterion: str = "criterion") -> CriterionVerdict:
    return CriterionVerdict.model_validate({"criterion": criterion, "reasoning": "because", "status": status})


def _result(
    name: str = "scenario",
    *,
    success: bool = True,
    failure_reason: str | None = None,
    harness_error: bool = False,
    verdicts: list[CriterionVerdict] | None = None,
    judge_score: JudgeScore | None = None,
    reasoning: str | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        name=name,
        description="description",
        success=success,
        verdicts=[_verdict("passed" if success else "failed")] if verdicts is None else verdicts,
        judge_score=judge_score,
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tool_calls=["tool_a", "tool_b"],
        turns=1,
        reasoning=reasoning,
        failure_reason=failure_reason,
        harness_error=harness_error,
        total_time=2.5,
        agent_time=1.0,
    )


def _records(*results: ScenarioResult) -> list[ScenarioRecord]:
    return [
        ScenarioRecord(result=result, test_id=f"test_file.py::test_{index}") for index, result in enumerate(results)
    ]


def _stats(passed: int, warned: int, failed: int) -> SummaryStats:
    return SummaryStats(
        total=passed + warned + failed,
        passed=passed,
        warned=warned,
        turns_total=0,
        tool_calls_total=0,
        unique_tools=[],
        criteria_total=0,
        criteria_passed=0,
        criteria_inconclusive=0,
        script_errors=0,
        total_time=0.0,
        agent_time=0.0,
    )


MIXED = (
    _result("passes"),
    _result("fails", success=False, failure_reason="assertion in script: x | y"),
    _result("harness", success=False, failure_reason="harness failure: boom", harness_error=True),
    _result("unverified", success=False, verdicts=[_verdict("inconclusive")]),
)


# -- summarize ----------------------------------------------------------------


def test_summarize_counts() -> None:
    stats = summarize(_records(_result(), _result(success=False, failure_reason="assertion in script: x")))

    assert (stats.total, stats.passed, stats.failed, stats.warned) == (2, 1, 1, 0)
    assert stats.pass_rate == 50.0
    assert (stats.criteria_total, stats.criteria_passed, stats.criteria_inconclusive) == (2, 1, 0)
    assert stats.tool_calls_total == 4
    assert stats.unique_tools == ["tool_a", "tool_b"]
    assert stats.script_errors == 1
    assert stats.turns_total == 2
    assert (stats.total_time, stats.agent_time) == (5.0, 2.0)


def test_summarize_counts_inconclusive_criteria_and_excludes_warn_from_the_pass_rate() -> None:
    stats = summarize(_records(*MIXED))

    assert (stats.total, stats.passed, stats.failed, stats.warned) == (4, 1, 1, 2)
    assert stats.judged == 2
    assert stats.pass_rate == 50.0
    assert stats.criteria_inconclusive == 1
    assert stats.criteria_total == 4
    assert stats.script_errors == 1, "a harness failure is not a script error"


def test_summarize_keeps_unique_tools_in_first_seen_order() -> None:
    first, second = _result(), _result()
    first.tool_calls, second.tool_calls = ["b", "a", "b"], ["c", "a"]

    assert summarize(_records(first, second)).unique_tools == ["b", "a", "c"]


def test_summarize_empty_records() -> None:
    stats = summarize([])

    assert (stats.total, stats.judged, stats.pass_rate) == (0, 0, 0.0)


def test_only_warnings_give_a_zero_pass_rate() -> None:
    assert summarize(_records(_result(success=False, harness_error=True))).pass_rate == 0.0


# -- progress bar and emoji ---------------------------------------------------


@pytest.mark.parametrize(
    ("passed", "warned", "failed", "expected"),
    [
        (175, 0, 0, "▓▓▓▓▓▓▓▓▓▓"),
        (154, 0, 21, "▓▓▓▓▓▓▓▓▓░"),
        (2, 0, 1, "▓▓▓▓▓▓▓░░░"),
        (174, 0, 1, "▓▓▓▓▓▓▓▓▓░"),
        (1, 0, 174, "▓░░░░░░░░░"),
        (0, 0, 175, "░░░░░░░░░░"),
        (1, 1, 1, "▓▓▓▓▒▒▒░░░"),
        (100, 50, 25, "▓▓▓▓▓▓▒▒▒░"),
        (1, 1, 98, "▓▒░░░░░░░░"),
        (0, 4, 0, "▒▒▒▒▒▒▒▒▒▒"),
        (0, 0, 0, "░░░░░░░░░░"),
    ],
)
def test_progress_bar_shape(passed: int, warned: int, failed: int, expected: str) -> None:
    assert progress_bar(_stats(passed, warned, failed)) == expected


def test_progress_bar_is_always_ten_cells_and_never_hides_a_category() -> None:
    for passed in range(0, 14):
        for warned in range(0, 14):
            for failed in range(0, 14):
                if passed + warned + failed == 0:
                    continue
                bar = progress_bar(_stats(passed, warned, failed))
                assert len(bar) == 10, (passed, warned, failed)
                for char, count in (("▓", passed), ("▒", warned), ("░", failed)):
                    assert (char in bar) == (count > 0), (passed, warned, failed)


def test_progress_bar_counts_one_in_a_thousand() -> None:
    assert progress_bar(_stats(999, 0, 1)) == "▓▓▓▓▓▓▓▓▓░"


@pytest.mark.parametrize(
    ("passed", "warned", "failed", "expected"),
    [
        (175, 0, 0, "🟢"),
        (174, 0, 1, "🟢"),
        (158, 0, 17, "🟢"),
        (154, 0, 21, "🟡"),
        (140, 0, 35, "🟡"),
        (108, 0, 67, "🔴"),
        (0, 0, 3, "🔴"),
        (10, 5, 0, "🟡"),
        (56, 113, 6, "🟡"),
        (3, 9, 7, "🔴"),
    ],
)
def test_status_emoji_bands(passed: int, warned: int, failed: int, expected: str) -> None:
    assert status_emoji(_stats(passed, warned, failed)) == expected


# -- markdown -----------------------------------------------------------------


def test_markdown_structure() -> None:
    md = to_markdown(_records(_result("passes"), _result("fails", success=False, failure_reason="boom | bang")))

    assert md.startswith(REPORT_MARKER + "\n")
    assert "## 🤖 Simulations: 🔴 50% ▓▓▓▓▓░░░░░ · PASS 1 / FAIL 1\n" in md
    assert "<summary>📋 <b>Scenario overview</b> (2)</summary>" in md
    assert "| passes | ✅ PASS | 1/1 | 1 | 2.5s | `tool_a` → `tool_b` |" in md
    assert "<summary>✅ PASS (1/1) · <b>passes</b> · 1 turn(s)</summary>" in md
    assert "<summary>❌ FAIL (0/1) · <b>fails</b> · 1 turn(s)</summary>" in md
    assert "<pre><code>boom | bang</code></pre>" in md
    assert "**Script/runtime error:**" in md
    assert "- ✅ criterion: _because_" in md
    assert "- ❌ criterion: _because_" in md
    assert "**Tools:** `tool_a` → `tool_b`" in md
    assert "```text\nUser: hi\nAgent: hello\n```" in md
    assert "### 📊 Summary" in md
    assert "| Pass rate | **50%** (PASS 1 / FAIL 1 of 2 judged) |" in md
    assert "| Time | 5.0s total, 2.0s in the agent |" in md
    assert "<details open>" not in md


def test_markdown_details_are_nested_and_balanced() -> None:
    md = to_markdown(_records(_result("a"), _result("b", success=False)))

    assert md.count("<details>") == md.count("</details>") == 4
    assert "<summary>🔍 <b>Detailed results</b> (2)</summary>" in md
    assert md.index("</details>\n\n### 📊 Summary") > md.index("<b>b</b>")


def test_markdown_warn_is_kept_apart_from_fail() -> None:
    md = to_markdown(_records(_result("passes"), *MIXED[2:3]))

    assert "🟡 100% ▓▓▓▓▓▒▒▒▒▒ · PASS 1 / FAIL 0 · ⚠️ WARN 1" in md
    assert "⚠️ WARN (0/1) · <b>harness</b>" in md
    assert "**Harness failure (WARN, not an agent failure):**" in md
    assert (
        "| ⚠️ WARN (not verified) | 1 of 2: harness failure or inconclusive criteria; not counted in the pass rate |"
        in md
    )


def test_markdown_shows_inconclusive_criteria() -> None:
    md = to_markdown(_records(MIXED[3]))

    assert "- ❔ criterion: _because_" in md
    assert "| Judge criteria | 0/1 passed, 1 inconclusive |" in md
    assert "⚠️ WARN" in md


def test_markdown_failure_block_escapes_html_and_truncates() -> None:
    reason = "<b>&" + "x" * (FAILURE_REASON_LIMIT + 100)
    md = to_markdown(_records(_result(success=False, failure_reason=reason)))

    assert "<pre><code>&lt;b&gt;&amp;" in md
    assert "<b>&x" not in md
    assert "x" * (FAILURE_REASON_LIMIT - 4) in md
    assert "x" * (FAILURE_REASON_LIMIT - 3) not in md
    assert "\n… (truncated)</code></pre>" in md


def test_markdown_failure_block_keeps_a_short_reason_intact() -> None:
    md = to_markdown(_records(_result(success=False, failure_reason="a" * FAILURE_REASON_LIMIT)))

    assert "a" * FAILURE_REASON_LIMIT + "</code></pre>" in md
    assert "truncated" not in md


def test_markdown_escapes_names_and_pipes() -> None:
    md = to_markdown(_records(_result("a | <b> & c")))

    assert "| a \\| <b> & c |" in md
    assert "<b>a | &lt;b&gt; &amp; c</b>" in md


def test_markdown_shows_judge_summary_and_score() -> None:
    score = JudgeScore(value=0.3, raw_score=3, threshold=7, passed=False, reasoning="too vague")
    md = to_markdown(_records(_result(success=False, reasoning="line one\nline two", judge_score=score)))

    assert "**Judge summary:** _line one line two_" in md
    assert "**Judge score:** ❌ 3/10 (threshold 7) (_too vague_)" in md


def test_markdown_judge_score_without_reasoning() -> None:
    score = JudgeScore(value=1.0, raw_score=10, threshold=7, passed=True)
    md = to_markdown(_records(_result(judge_score=score)))

    assert "**Judge score:** ✅ 10/10 (threshold 7)\n" in md


def test_markdown_uses_the_configured_role_names() -> None:
    sim.configure(user_name="Customer", agent_name="Ellie")

    md = to_markdown(_records(_result()))

    assert "Customer: hi\nEllie: hello" in md


def test_markdown_without_tools_or_verdicts_or_messages() -> None:
    bare = ScenarioResult(name="bare", description="", success=True)

    md = to_markdown(_records(bare))

    assert "| bare | ✅ PASS | 0/0 | 0 | 0.0s | - |" in md
    assert "**Criteria:**" not in md
    assert "**Dialogue:**" not in md
    assert "| Tool calls | 0 (-) |" in md


def test_markdown_ci_job_url_from_gitlab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI_JOB_URL", "https://gitlab.example.com/g/p/-/jobs/7")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")

    assert "\nCI job: https://gitlab.example.com/g/p/-/jobs/7\n" in to_markdown(_records(_result()))


def test_markdown_ci_job_url_from_github(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ormeilu/simulhausen")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")

    assert "\nCI job: https://github.com/ormeilu/simulhausen/actions/runs/99\n" in to_markdown(_records(_result()))


def test_markdown_ci_job_url_needs_all_github_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")

    assert "CI job" not in to_markdown(_records(_result()))


# -- json and serialization ---------------------------------------------------


def test_to_json_shape() -> None:
    payload = json.loads(to_json(_records(*MIXED)))

    assert {key: payload[key] for key in ("total", "passed", "failed", "warned", "pass_rate")} == {
        "total": 4,
        "passed": 1,
        "failed": 1,
        "warned": 2,
        "pass_rate": 50.0,
    }
    assert [s["status"] for s in payload["scenarios"]] == ["pass", "fail", "warn", "warn"]
    first = payload["scenarios"][0]
    assert first["tool_calls"] == ["tool_a", "tool_b"]
    assert first["verdicts"][0]["status"] == "passed"
    assert first["messages"][0] == {"role": "user", "content": "hi"}
    assert first["test_id"] == "test_file.py::test_0"


def test_to_json_keeps_non_ascii_text_and_rounds_the_pass_rate() -> None:
    records = _records(_result("café ✓"), _result("naïve", success=False), _result("Zürich", success=False))

    text = to_json(records)

    assert "café ✓" in text
    assert json.loads(text)["pass_rate"] == 33.33


def test_to_json_of_nothing() -> None:
    assert json.loads(to_json([])) == {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "warned": 0,
        "pass_rate": 0.0,
        "scenarios": [],
    }


def test_records_survive_the_xdist_round_trip() -> None:
    score = JudgeScore(value=0.5, raw_score=6, threshold=7, passed=False, reasoning="meh")
    records = _records(*MIXED, _result("scored", success=False, judge_score=score, reasoning="why"))
    records.append(ScenarioRecord(result=_result("untied"), test_id=None))

    payload = serialize_records(records)

    assert deserialize_records(json.loads(json.dumps(payload))) == records


def test_deserialize_accepts_a_minimal_payload() -> None:
    [record] = deserialize_records([{"name": "n", "success": True}])

    assert record.test_id is None
    assert record.result == ScenarioResult(name="n", description="", success=True)


def test_deserialize_normalises_message_roles() -> None:
    [record] = deserialize_records(
        [
            {
                "name": "n",
                "success": True,
                "messages": [{"role": "assistant", "content": 1}, {"role": "tool", "content": "x"}],
            }
        ]
    )

    assert record.result.messages == [{"role": "assistant", "content": "1"}, {"role": "user", "content": "x"}]


# -- collector ----------------------------------------------------------------


def test_collector_ties_results_to_the_current_test() -> None:
    local = Collector()
    local.add(_result("outside"))
    local.set_current_test("test_file.py::test_a")
    local.add(_result("inside"))

    assert [r.test_id for r in local.records] == [None, "test_file.py::test_a"]


def test_collector_drops_a_tests_records_before_a_rerun() -> None:
    local = Collector()
    local.set_current_test("node1")
    local.add(_result("first attempt", success=False))
    local.set_current_test("node2")
    local.add(_result("other test"))

    local.drop_test_records("node1")
    local.set_current_test("node1")
    local.add(_result("second attempt"))

    assert [r.result.name for r in local.records] == ["other test", "second attempt"]


def test_collector_groups_by_status() -> None:
    local = Collector()
    for result in MIXED:
        local.add(result)

    assert [r.result.name for r in local.passed] == ["passes"]
    assert [r.result.name for r in local.failed] == ["fails"]
    assert [r.result.name for r in local.warned] == ["harness", "unverified"]
    assert all(r.result.status is ScenarioStatus.WARN for r in local.warned)

    local.clear()
    assert local.records == []


def test_the_global_collector_is_the_one_run_feeds() -> None:
    assert collector.records == []


# -- console ------------------------------------------------------------------


def test_render_console_prints_the_table_and_summary(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "400")
    score = JudgeScore(value=0.2, raw_score=2, threshold=7, passed=False)

    render_console(_records(*MIXED, _result("scored", success=False, judge_score=score, reasoning="judge says no")))

    out = capsys.readouterr().out
    assert "Simulations: 1/3 passed (33%), WARN 2" in out
    for name in ("passes", "fails", "harness", "unverified", "scored"):
        assert name in out
    for label in ("PASS", "FAIL", "WARN"):
        assert label in out
    assert "assertion in script: x | y" in out
    assert "judge score 2/10 (threshold 7)" in out
    assert "judge says no" in out
    assert "1. tool_a" in out
    assert "User: hi" in out
    assert re.search(
        r"Summary: pass rate 33% \(1 passed / 2 failed of 3 judged\); 2 not verified \(WARN\) of 5 total", out
    )
    assert "criteria 1/5 passed (1 inconclusive)" in out


def test_render_console_with_bare_results(capsys: pytest.CaptureFixture[str]) -> None:
    render_console(_records(ScenarioResult(name="bare", description="", success=True)))

    out = capsys.readouterr().out
    assert "bare" in out
    assert "Simulations: 1/1 passed (100%)" in out
    assert ", WARN" not in out


def test_render_console_keeps_square_brackets_in_user_text(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "400")
    result = _result(
        "[smoke] refund",
        success=False,
        failure_reason="expected [a, b] but got [/c]",
        verdicts=[_verdict("failed", "mentions [search] results")],
    )

    render_console(_records(result))

    out = capsys.readouterr().out
    assert "[smoke] refund" in out
    assert "expected [a, b] but got [/c]" in out
    assert "mentions [search] results" in out
