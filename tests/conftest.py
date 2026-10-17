"""Shared fixtures: the process-global collector and config are reset around every test, the environment
variables that steer simulhausen are cleared or pointed at a temporary directory, and the package looks installed
from PyPI."""

from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import pytest
from harbor_support import installed_from
from red_team_support import FakeLLM

from simulhausen import SimulationConfig, collector, get_config

pytest_plugins = ["pytester"]


def _reset_globals() -> None:
    defaults = SimulationConfig()
    config = get_config()
    for field in fields(SimulationConfig):
        setattr(config, field.name, getattr(defaults, field.name))
    collector.clear()
    collector.set_current_test(None)


@pytest.fixture(autouse=True)
def _isolated_globals() -> Iterator[None]:
    _reset_globals()
    yield
    _reset_globals()


# Variables that pick the Harbor commands' target and models: a developer's shell must not redirect the tests.
HARBOR_VARIABLES = ("SIMULHAUSEN_TARGET", "SIMULHAUSEN_MODEL", "SIMULHAUSEN_SIMULATOR_MODEL", "SIMULHAUSEN_JUDGE_MODEL")


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in HARBOR_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own response-cache directory, so none of them can touch the real one."""
    path = tmp_path / "response-cache"
    monkeypatch.setenv("SIMULHAUSEN_CACHE_DIR", str(path))
    return path


@pytest.fixture(autouse=True)
def _installed_from_an_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """The simulhausen under test looks installed from PyPI: how the checkout itself is installed (editable, from an
    archive) must not change the requirement that export and the Harbor agents put into task images."""
    installed_from(monkeypatch, None)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Replaces the agno agents the red-team agent builds with a scripted fake."""
    fake = FakeLLM()
    fake.install(monkeypatch)
    return fake
