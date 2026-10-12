"""Shared fixtures: the process-global collector and config are reset around every test."""

from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import pytest

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


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own response-cache directory, so none of them can touch the real one."""
    path = tmp_path / "response-cache"
    monkeypatch.setenv("SIMULHAUSEN_CACHE_DIR", str(path))
    return path
