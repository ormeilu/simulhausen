"""Shared fixtures: a recording Harbor environment and an agent factory. No container, no network."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from support import FakeEnvironment

from simulhausen_harbor.agents import SimulhausenAgent


@pytest.fixture(autouse=True)
def _no_harbor_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARBOR_TELEMETRY", "0")


@pytest.fixture
def environment(tmp_path: Path) -> FakeEnvironment:
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()
    return FakeEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="refund-duplicated-charge",
        session_id="refund-duplicated-charge__abc1234__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
    )


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "agent-logs"
    path.mkdir()
    return path


@pytest.fixture
def make_agent(logs_dir: Path) -> Callable[..., SimulhausenAgent]:
    """Build an agent the way Harbor's AgentFactory does: keyword arguments only."""

    def build(cls: type[SimulhausenAgent], model_name: str | None = None, **kwargs: Any) -> SimulhausenAgent:
        return cls(logs_dir=logs_dir, model_name=model_name, **kwargs)

    return build
