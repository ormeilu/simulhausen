import json
from pathlib import Path
from typing import Any

import pytest

from simulhausen.cli import EXIT_OK, EXIT_REGRESSION, EXIT_USAGE, main
from simulhausen.diff import DIFF_MARKER, ScenarioSide, diff_reports, load_report


def _scenario(name: str, status: str = "pass", *, test_id: str | None = "t.py::test", **extra: Any) -> dict[str, Any]:
    return {"name": name, "test_id": test_id, "status": status, "success": status == "pass", "turns": 2, **extra}


def _report(*scenarios: dict[str, Any]) -> dict[str, Any]:
    return {"scenarios": list(scenarios)}


def _kinds(diff: Any) -> dict[str, str]:
    return {delta.name: delta.kind for delta in diff.deltas}


def test_every_kind_of_change() -> None:
    base = _report(
        _scenario("stays green"),
        _scenario("breaks"),
        _scenario("gets fixed", "fail"),
        _scenario("goes away"),
        _scenario("loses its verdict"),
        _scenario("gets a verdict", "warn"),
        _scenario("still red", "fail"),
    )
    head = _report(
        _scenario("stays green"),
        _scenario("breaks", "fail", verdicts=[{"criterion": "asks for the order", "status": "failed"}]),
        _scenario("gets fixed"),
        _scenario("loses its verdict", "warn", failure_reason="harness failure: boom"),
        _scenario("gets a verdict"),
        _scenario("still red", "fail"),
        _scenario("brand new", "fail", failure_reason="assertion in script: nope"),
    )

    diff = diff_reports(base, head)

    assert _kinds(diff) == {
        "stays green": "unchanged",
        "breaks": "regressed",
        "gets fixed": "improved",
        "loses its verdict": "unverified",
        "gets a verdict": "verified",
        "still red": "unchanged",
        "brand new": "new",
        "goes away": "removed",
    }
    assert [delta.name for delta in diff.regressions] == ["breaks"]
    assert [delta.name for delta in diff.improvements] == ["gets fixed"]
    assert diff.has_regressions
    assert diff.base_pass_rate == pytest.approx(4 / 6)
    assert diff.head_pass_rate == pytest.approx(3 / 6)
    assert diff.summary() == (
        "pass rate 67% -> 50%; 1 regressed, 1 improved, 1 new, 1 removed, 1 unverified, 1 verified, 2 unchanged"
    )
    changes = {delta.name: delta.change for delta in diff.deltas}
    assert changes["breaks"] == "PASS → FAIL"
    assert changes["brand new"] == "new: FAIL"
    assert changes["goes away"] == "removed (was PASS)"
    assert changes["loses its verdict"] == "PASS → WARN"


def test_markdown_lists_regressions_with_what_failed() -> None:
    base = _report(_scenario("breaks", total_time=1.0, usage={"total_tokens": 100}), _scenario("gone"))
    head = _report(
        _scenario(
            "breaks",
            "fail",
            total_time=3.5,
            usage={"input_tokens": 200, "output_tokens": 50},
            failure_reason="assertion in script: a | b\nsecond line",
            verdicts=[
                {"criterion": "asks for the order", "status": "failed"},
                {"criterion": "is polite", "status": "passed"},
            ],
        ),
        _scenario("fresh"),
        _scenario("shaky", "warn"),
    )

    text = diff_reports(base, head).to_markdown()

    assert text.startswith(DIFF_MARKER)
    assert "🔴 1 regressed · 0 improved · pass rate 100% → 50%" in text
    assert "### ❌ Regressed (1)" in text
    assert "- **breaks**: PASS → FAIL" in text
    assert "  - failed criterion: asks for the order" in text
    assert "is polite" not in text
    assert "  - assertion in script: a \\| b second line" in text
    assert "### 🆕 New (2)" in text
    assert "### 🗑️ Removed (1)" in text
    assert "| Pass rate | 100% | 50% | -50 pt |" in text
    assert "| Time | 1.0s | 3.5s | +2.5s |" in text
    assert "| Agent tokens | 100 | 250 | +150 |" in text
    assert "0 scenario(s) unchanged." in text


def test_a_clean_diff_is_green_and_has_no_token_row_without_usage() -> None:
    report = _report(_scenario("a"), _scenario("b", "warn"))

    diff = diff_reports(report, report)
    text = diff.to_markdown()

    assert not diff.has_regressions
    assert "🟢 0 regressed" in text
    assert "### ⚠️ Not verified in this run (1)" in text
    assert "Agent tokens" not in text
    assert "1 scenario(s) unchanged." in text


def test_empty_reports() -> None:
    diff = diff_reports({}, {})

    assert diff.summary() == "pass rate - -> -; no scenarios"
    assert "| Pass rate | - | - | - |" in diff.to_markdown()
    assert diff.to_dict()["scenarios"] == []


def test_trials_are_compared_by_pass_rate_with_a_tolerance() -> None:
    def trials(passed: int, total: int = 5) -> list[dict[str, Any]]:
        return [_scenario("flaky", "pass" if index < passed else "fail") for index in range(total)]

    base, head = _report(*trials(4)), _report(*trials(3))

    assert _kinds(diff_reports(base, head)) == {"flaky": "regressed"}
    assert _kinds(diff_reports(base, head, tolerance=0.2)) == {"flaky": "unchanged"}
    assert _kinds(diff_reports(head, base, tolerance=0.1)) == {"flaky": "improved"}
    assert diff_reports(base, head).deltas[0].change == "4/5 passed → 3/5 passed"
    with pytest.raises(ValueError, match="tolerance must be"):
        diff_reports(base, head, tolerance=1.0)


def test_side_labels() -> None:
    side = ScenarioSide()
    for status in ("pass", "warn", "warn"):
        side.add(_scenario("s", status))
    assert side.label == "1/1 passed (+2 WARN)"
    assert side.pass_rate == 1.0

    unverified = ScenarioSide()
    unverified.add(_scenario("s", "warn"))
    unverified.add(_scenario("s", "warn"))
    assert unverified.label == "not verified"
    assert unverified.pass_rate is None


def test_scenarios_are_matched_by_test_and_name_and_old_reports_without_status_still_work() -> None:
    base = _report(
        {"name": "same name", "test_id": "a.py::one", "success": True},
        {"name": "same name", "test_id": "b.py::two", "success": True},
        {"name": "no test", "success": False},
    )
    head = _report(
        {"name": "same name", "test_id": "a.py::one", "success": False},
        {"name": "same name", "test_id": "b.py::two", "success": True},
        {"name": "no test", "success": False},
    )

    diff = diff_reports(base, head)

    assert [(delta.test_id, delta.kind) for delta in diff.deltas] == [
        ("a.py::one", "regressed"),
        ("b.py::two", "unchanged"),
        (None, "unchanged"),
    ]


def test_to_dict_is_json_ready() -> None:
    base = _report(_scenario("a", usage={"total_tokens": 5}))
    head = _report(_scenario("a", "fail", failure_reason="x"), _scenario("b", usage={"total_tokens": 7}))

    data = json.loads(json.dumps(diff_reports(base, head, tolerance=0.1).to_dict()))

    assert data["tolerance"] == 0.1
    assert data["counts"]["regressed"] == 1
    assert data["base"]["usage"]["total_tokens"] == 5
    assert data["head"]["usage"]["total_tokens"] == 7
    first, second = data["scenarios"]
    assert first["kind"] == "regressed"
    assert first["head"]["failure_reason"] == "x"
    assert first["head"]["usage"] is None
    assert second["base"] is None
    assert second["head"]["usage"]["total_tokens"] == 7


# -- files and the command line -------------------------------------------------------


def _write(path: Path, report: Any) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_load_report_rejects_what_is_not_a_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read the report"):
        load_report(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read the report"):
        load_report(broken)
    with pytest.raises(ValueError, match="is not a simulhausen JSON report"):
        load_report(_write(tmp_path / "list.json", [1]))
    with pytest.raises(ValueError, match="is not a simulhausen JSON report"):
        load_report(_write(tmp_path / "other.json", {"scenarios": "no"}))
    assert load_report(_write(tmp_path / "ok.json", _report())) == {"scenarios": []}


def test_cli_writes_the_comparison_and_reports_regressions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _write(tmp_path / "base.json", _report(_scenario("a"), _scenario("b")))
    head = _write(tmp_path / "head.json", _report(_scenario("a", "fail"), _scenario("b")))
    md, data = tmp_path / "out" / "diff.md", tmp_path / "out" / "diff.json"

    code = main(["report", "diff", str(base), str(head), "--md", str(md), "--json", str(data)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "pass rate 100% -> 50%; 1 regressed, 1 unchanged" in out
    assert "regressed: a: PASS → FAIL" in out
    assert md.read_text(encoding="utf-8").startswith(DIFF_MARKER)
    assert json.loads(data.read_text(encoding="utf-8"))["counts"]["regressed"] == 1

    assert main(["report", "diff", str(base), str(head), "--fail-on-regression"]) == EXIT_REGRESSION
    assert main(["report", "diff", str(base), str(base), "--fail-on-regression"]) == EXIT_OK


def test_cli_reports_a_bad_file_as_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _write(tmp_path / "base.json", _report())

    assert main(["report", "diff", str(base), str(tmp_path / "missing.json")]) == EXIT_USAGE
    assert "cannot read the report" in capsys.readouterr().err
    assert main(["report", "diff", str(base), str(base), "--tolerance", "2"]) == EXIT_USAGE
    assert "tolerance must be" in capsys.readouterr().err
