import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

import simulhausen as sim
from simulhausen import ScenarioResult, collector
from simulhausen.pytest_plugin import (
    pytest_runtest_protocol,
    pytest_sessionfinish,
    pytest_terminal_summary,
    pytest_testnodedown,
)
from simulhausen.reporting import REPORT_MARKER, ScenarioRecord, serialize_records

SCENARIOS = """
import pytest
import simulhausen as sim


def agent_under_test():
    return sim.function_adapter(lambda agent_input: "pong")


@pytest.mark.simulation
@pytest.mark.asyncio
async def test_good_scenario():
    result = await sim.run(
        "good scenario", "pings the agent", [agent_under_test()], [sim.user("ping"), sim.agent(), sim.succeed("pong")]
    )
    assert result.success


@pytest.mark.simulation
@pytest.mark.asyncio
async def test_bad_scenario():
    result = await sim.run(
        "bad scenario", "pings the agent", [agent_under_test()], [sim.user("ping"), sim.agent(), sim.fail("no pong")]
    )
    assert result.success, result.failure_summary()
"""

PLAIN_TEST = "def test_plain():\n    assert True\n"

INI = "[pytest]\nasyncio_mode = auto\nasyncio_default_fixture_loop_scope = function\n"


@pytest.fixture
def project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    monkeypatch.setenv("COLUMNS", "250")
    pytester.makeini(INI)
    pytester.makepyfile(test_scenarios=SCENARIOS)
    return pytester


# -- end to end, in a subprocess ----------------------------------------------


def test_reports_and_terminal_summary(project: pytest.Pytester) -> None:
    result = project.runpytest_subprocess("--sim-report=out/nested/sim.json", "--sim-report-md=out/sim.md")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=1, failed=1)

    out = result.stdout.str()
    assert "Simulations: 1/2 passed (50%)" in out
    assert "Summary: pass rate 50% (1 passed / 1 failed of 2 judged)" in out
    assert "good scenario" in out
    assert "fail(): no pong" in out

    payload = json.loads((project.path / "out/nested/sim.json").read_text(encoding="utf-8"))
    assert (payload["total"], payload["passed"], payload["failed"], payload["warned"]) == (2, 1, 1, 0)
    scenarios = {s["name"]: s for s in payload["scenarios"]}
    assert scenarios["good scenario"]["status"] == "pass"
    assert scenarios["bad scenario"]["failure_reason"] == "fail(): no pong"
    assert scenarios["bad scenario"]["test_id"] == "test_scenarios.py::test_bad_scenario"
    assert scenarios["good scenario"]["messages"] == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]

    markdown = (project.path / "out/sim.md").read_text(encoding="utf-8")
    assert markdown.startswith(REPORT_MARKER)
    assert "PASS 1 / FAIL 1" in markdown
    assert "<b>bad scenario</b>" in markdown


def test_benchmark_mode_turns_failures_into_exit_code_zero(project: pytest.Pytester) -> None:
    result = project.runpytest_subprocess("--sim-benchmark", "--sim-report=sim.json")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1, failed=1)
    assert "Simulations: 1/2 passed (50%)" in result.stdout.str()
    assert json.loads((project.path / "sim.json").read_text(encoding="utf-8"))["failed"] == 1


def test_benchmark_mode_keeps_other_exit_codes(pytester: pytest.Pytester) -> None:
    pytester.makeini(INI)

    assert pytester.runpytest_subprocess("--sim-benchmark").ret == pytest.ExitCode.NO_TESTS_COLLECTED

    pytester.makepyfile(test_syntax="def oops(:\n")
    assert pytester.runpytest_subprocess("--sim-benchmark").ret == pytest.ExitCode.INTERRUPTED


def test_plain_tests_leave_no_report_and_no_summary(pytester: pytest.Pytester) -> None:
    pytester.makeini(INI)
    pytester.makepyfile(test_plain=PLAIN_TEST)

    result = pytester.runpytest_subprocess("--sim-report=sim.json", "--sim-report-md=sim.md")

    result.assert_outcomes(passed=1)
    assert "Simulations:" not in result.stdout.str()
    assert not (pytester.path / "sim.json").exists()
    assert not (pytester.path / "sim.md").exists()


def test_debug_without_dash_s_is_a_usage_error(project: pytest.Pytester) -> None:
    result = project.runpytest_subprocess("--sim-debug")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "--sim-debug reads your messages from stdin: run it with -s" in result.stderr.str()
    assert "passed" not in result.stdout.str()


def test_debug_with_dash_s_switches_the_config_on(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_debug=("import simulhausen as sim\n\ndef test_debug_flag():\n    assert sim.get_config().debug is True\n"),
        test_nodebug=(
            "import simulhausen as sim\n\ndef test_debug_flag():\n    assert sim.get_config().debug is False\n"
        ),
    )

    assert pytester.runpytest_subprocess("--sim-debug", "-s", "test_debug.py").ret == pytest.ExitCode.OK
    assert pytester.runpytest_subprocess("test_nodebug.py").ret == pytest.ExitCode.OK


def test_the_simulation_marker_is_registered(pytester: pytest.Pytester) -> None:
    pytester.makeini(INI)
    pytester.makepyfile(test_marked="import pytest\n\n@pytest.mark.simulation\ndef test_marked():\n    pass\n")

    markers = pytester.runpytest_subprocess("--markers")
    strict = pytester.runpytest_subprocess("--strict-markers")

    markers.stdout.fnmatch_lines(["@pytest.mark.simulation: scenario simulation test that calls real LLMs"])
    strict.assert_outcomes(passed=1)


def test_options_are_listed_in_help(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest_subprocess("--help")

    result.stdout.fnmatch_lines(
        [
            "*scenario simulations (simulhausen):*",
            "*--sim-benchmark*",
            "*--sim-report=PATH*",
            "*--sim-report-md=PATH*",
            "*--sim-debug*",
        ]
    )


# -- hooks, in process --------------------------------------------------------


def _result(name: str, *, success: bool = True) -> ScenarioResult:
    return ScenarioResult(name=name, description="", success=success, turns=1)


def _config(pytester: pytest.Pytester, *args: str) -> pytest.Config:
    pytester.makeini(INI)
    return pytester.parseconfig(*args)


def _session(config: pytest.Config, exitstatus: int = pytest.ExitCode.OK) -> pytest.Session:
    return cast(pytest.Session, SimpleNamespace(config=config, exitstatus=exitstatus))


def _as_xdist_worker(config: pytest.Config) -> dict[str, Any]:
    """Mark the config the way pytest-xdist does on a worker; returns its ``workeroutput``."""
    output: dict[str, Any] = {}
    vars(config).update(workerinput={}, workeroutput=output)
    return output


def _fill(*items: tuple[str | None, ScenarioResult]) -> None:
    for test_id, result in items:
        collector.set_current_test(test_id)
        collector.add(result)
    collector.set_current_test(None)


def test_session_finish_writes_reports_in_a_stable_order(pytester: pytest.Pytester) -> None:
    config = _config(pytester, "--sim-report=deep/er/sim.json", "--sim-report-md=sim.md")
    _fill(
        ("t.py::test_b", _result("b")),
        ("t.py::test_a", _result("z")),
        ("t.py::test_a", _result("a")),
        (None, _result("orphan")),
    )

    pytest_sessionfinish(_session(config), 0)

    payload = json.loads((pytester.path / "deep/er/sim.json").read_text(encoding="utf-8"))
    assert [(s["test_id"], s["name"]) for s in payload["scenarios"]] == [
        (None, "orphan"),
        ("t.py::test_a", "a"),
        ("t.py::test_a", "z"),
        ("t.py::test_b", "b"),
    ]
    assert (pytester.path / "sim.md").read_text(encoding="utf-8").startswith(REPORT_MARKER)


def test_session_finish_without_results_writes_nothing(pytester: pytest.Pytester) -> None:
    config = _config(pytester, "--sim-report=sim.json", "--sim-report-md=sim.md")

    pytest_sessionfinish(_session(config), 0)

    assert not (pytester.path / "sim.json").exists()
    assert not (pytester.path / "sim.md").exists()


@pytest.mark.parametrize(
    ("benchmark", "exitstatus", "expected"),
    [
        (True, pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.OK),
        (True, pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.INTERNAL_ERROR),
        (True, pytest.ExitCode.NO_TESTS_COLLECTED, pytest.ExitCode.NO_TESTS_COLLECTED),
        (False, pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.TESTS_FAILED),
    ],
)
def test_session_finish_benchmark_only_masks_failed_tests(
    pytester: pytest.Pytester, benchmark: bool, exitstatus: pytest.ExitCode, expected: pytest.ExitCode
) -> None:
    config = _config(pytester, *(["--sim-benchmark"] if benchmark else []))
    session = _session(config, exitstatus)

    pytest_sessionfinish(session, exitstatus)

    assert session.exitstatus == expected


def test_xdist_worker_ships_its_records_instead_of_writing_reports(pytester: pytest.Pytester) -> None:
    config = _config(pytester, "--sim-report=sim.json")
    output = _as_xdist_worker(config)
    _fill(("t.py::test_b", _result("b")), ("t.py::test_a", _result("a", success=False)))

    pytest_sessionfinish(_session(config), 1)

    shipped = output["simulhausen_records"]
    assert [item["name"] for item in shipped] == ["a", "b"]
    assert not (pytester.path / "sim.json").exists()


def test_xdist_controller_merges_worker_records() -> None:
    worker_records = [ScenarioRecord(result=_result("remote", success=False), test_id="t.py::test_remote")]
    node = SimpleNamespace(workeroutput={"simulhausen_records": serialize_records(worker_records)})

    pytest_testnodedown(node, None)

    assert collector.records == worker_records


@pytest.mark.parametrize(
    "node",
    [SimpleNamespace(), SimpleNamespace(workeroutput={}), SimpleNamespace(workeroutput={"simulhausen_records": []})],
)
def test_xdist_controller_ignores_workers_without_records(node: SimpleNamespace) -> None:
    pytest_testnodedown(node, None)

    assert collector.records == []


def test_terminal_summary_prints_only_with_results(
    pytester: pytest.Pytester, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(pytester)
    pytest_terminal_summary(object(), 0, config)
    assert capsys.readouterr().out == ""

    _fill(("t.py::test_a", _result("shown")))
    pytest_terminal_summary(object(), 0, config)
    assert "shown" in capsys.readouterr().out


def test_terminal_summary_is_silent_on_xdist_workers(
    pytester: pytest.Pytester, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(pytester)
    _as_xdist_worker(config)
    _fill(("t.py::test_a", _result("shown")))

    pytest_terminal_summary(object(), 0, config)

    assert capsys.readouterr().out == ""


def test_runtest_protocol_ties_results_to_the_test_and_drops_earlier_attempts() -> None:
    item = cast(pytest.Item, SimpleNamespace(nodeid="t.py::test_flaky"))
    _fill(("t.py::test_flaky", _result("first attempt", success=False)), ("t.py::test_other", _result("other")))

    wrapper = pytest_runtest_protocol(item, None)
    next(wrapper)
    collector.add(_result("second attempt"))
    with pytest.raises(StopIteration) as finished:
        wrapper.send(True)

    assert finished.value.value is True
    assert [(r.test_id, r.result.name) for r in collector.records] == [
        ("t.py::test_other", "other"),
        ("t.py::test_flaky", "second attempt"),
    ]
    collector.add(_result("after the test"))
    assert collector.records[-1].test_id is None


def test_runtest_protocol_clears_the_current_test_when_the_test_raises() -> None:
    item = cast(pytest.Item, SimpleNamespace(nodeid="t.py::test_x"))

    wrapper = pytest_runtest_protocol(item, None)
    next(wrapper)
    with pytest.raises(RuntimeError, match="boom"):
        wrapper.throw(RuntimeError("boom"))

    collector.add(_result("later"))
    assert collector.records[-1].test_id is None


def test_configure_registers_the_marker_and_accepts_debug_with_dash_s(pytester: pytest.Pytester) -> None:
    pytester.makeini(INI)
    config = pytester.parseconfigure("--sim-debug", "-s")

    assert sim.get_config().debug is True
    assert any(line.startswith("simulation:") for line in config.getini("markers"))


def test_configure_rejects_debug_without_dash_s(pytester: pytest.Pytester) -> None:
    pytester.makeini(INI)
    with pytest.raises(pytest.UsageError, match="run it with -s"):
        pytester.parseconfigure("--sim-debug")

    assert sim.get_config().debug is False


def test_options_default_to_off(pytester: pytest.Pytester) -> None:
    config = _config(pytester)

    assert config.getoption("--sim-benchmark") is False
    assert config.getoption("--sim-debug") is False
    assert config.getoption("--sim-report") is None
    assert config.getoption("--sim-report-md") is None
