"""pytest plugin, registered through the ``pytest11`` entry point.

Two ways to run simulation tests:

- plain pytest: failing scenarios fail the run like any other test;
- benchmark (``--sim-benchmark``): failures land in the report and the exit
  code is 0, so the CI job stays green and the outcome goes to a PR/MR comment.

Either way a rich summary is printed after the run, and markdown/JSON reports
are written when their paths are given. Without simulation results it stays quiet.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from simulhausen.config import configure
from simulhausen.reporting import (
    ScenarioRecord,
    collector,
    deserialize_records,
    render_console,
    serialize_records,
    to_json,
    to_markdown,
)

if TYPE_CHECKING:
    from collections.abc import Generator

_WORKER_RECORDS_KEY = "simulhausen_records"


def _is_xdist_worker(config: pytest.Config) -> bool:
    return hasattr(config, "workerinput")


def _ordered_records() -> list[ScenarioRecord]:
    """Stable order no matter which xdist worker finished first."""
    return sorted(collector.records, key=lambda record: (record.test_id or "", record.result.name))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("simulhausen", "scenario simulations (simulhausen)")
    group.addoption(
        "--sim-benchmark",
        action="store_true",
        default=False,
        help="benchmark mode: collect the scenario report and exit 0 even when scenarios fail",
    )
    group.addoption("--sim-report", default=None, metavar="PATH", help="write the JSON scenario report to PATH")
    group.addoption(
        "--sim-report-md",
        default=None,
        metavar="PATH",
        help="write the markdown scenario report (for a PR/MR comment) to PATH",
    )
    group.addoption(
        "--sim-debug",
        action="store_true",
        default=False,
        help="debug mode: type the simulated user's messages yourself (needs -s)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "simulation: scenario simulation test that calls real LLMs")
    if config.getoption("--sim-debug"):
        # Debug mode reads stdin, and capturing starts before plugins configure,
        # so it cannot be switched off from here.
        if config.getoption("capture") != "no":
            raise pytest.UsageError("--sim-debug reads your messages from stdin: run it with -s")
        configure(debug=True)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, object, object]:
    # Tie scenario results (run() -> collector.add) to the test's nodeid.
    # pytest-rerunfailures runs a test again: keep only its last attempt.
    collector.drop_test_records(item.nodeid)
    collector.set_current_test(item.nodeid)
    try:
        return (yield)
    finally:
        collector.set_current_test(None)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if _is_xdist_worker(config):
        config.workeroutput[_WORKER_RECORDS_KEY] = serialize_records(_ordered_records())  # ty: ignore[unresolved-attribute]
    elif collector.records:
        records = _ordered_records()
        for option, render in (("--sim-report", to_json), ("--sim-report-md", to_markdown)):
            if target := config.getoption(option):
                path = Path(target)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render(records), encoding="utf-8")
    # Benchmark mode: failed tests do not fail the run; the report matters, not the exit code.
    if config.getoption("--sim-benchmark") and exitstatus == pytest.ExitCode.TESTS_FAILED:
        session.exitstatus = pytest.ExitCode.OK


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: object, error: object) -> None:
    """Merge worker-local results into the pytest-xdist controller."""
    workeroutput = getattr(node, "workeroutput", {})
    if payload := workeroutput.get(_WORKER_RECORDS_KEY, []):
        collector.records.extend(deserialize_records(payload))


def pytest_terminal_summary(terminalreporter: object, exitstatus: int, config: pytest.Config) -> None:
    if collector.records and not _is_xdist_worker(config):
        render_console(_ordered_records())
