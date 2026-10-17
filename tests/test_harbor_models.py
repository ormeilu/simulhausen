import os
from typing import cast

import pytest
from agno.models.base import Model

from simulhausen.harbor import ConfigError
from simulhausen.harbor import models as models_module
from simulhausen.harbor.models import MODEL_ENV, build_model, model_string, resolve_model
from simulhausen.harbor.targets import TARGET_ENV


def test_the_shell_cannot_redirect_the_tests() -> None:
    """conftest clears the variables that pick models and targets, whatever the developer's shell exports."""
    variables = {*MODEL_ENV["simulator"], *MODEL_ENV["judge"], TARGET_ENV}
    assert sorted(name for name in variables if name in os.environ) == []


def test_precedence_cli_then_role_variable_then_shared_variable_then_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    assert model_string("judge", None, "scenario:model") == "scenario:model"
    monkeypatch.setenv("SIMULHAUSEN_MODEL", "shared:model")
    assert model_string("judge", None, "scenario:model") == "shared:model"
    monkeypatch.setenv("SIMULHAUSEN_JUDGE_MODEL", "judge:model")
    monkeypatch.setenv("SIMULHAUSEN_SIMULATOR_MODEL", "simulator:model")
    assert model_string("judge", None, "scenario:model") == "judge:model"
    assert model_string("simulator", None, "scenario:model") == "simulator:model"
    assert model_string("judge", "cli:model", "scenario:model") == "cli:model"


def test_empty_variables_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exported tasks pass the variables through as "${VAR:-}", which is empty when unset on the host.
    monkeypatch.setenv("SIMULHAUSEN_JUDGE_MODEL", "")
    monkeypatch.setenv("SIMULHAUSEN_MODEL", "")

    assert model_string("judge", None, "scenario:model") == "scenario:model"


def test_no_model_anywhere_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="no simulator model: pass --model, set \\$SIMULHAUSEN_SIMULATOR_MODEL or"):
        model_string("simulator", None, None)


def test_build_model_reports_a_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_sdk(value: str) -> Model:
        raise ImportError("`openai` not installed. Please install using `pip install openai`")

    monkeypatch.setattr(models_module, "get_model", missing_sdk)

    with pytest.raises(ConfigError, match=r"the judge model 'openai:gpt-4\.1-mini' needs a Python package") as info:
        build_model("judge", "openai:gpt-4.1-mini")
    assert "`openai` not installed" in str(info.value)
    assert "--pip" in str(info.value)


def test_build_model_reports_a_bad_model_string() -> None:
    with pytest.raises(ConfigError, match="the simulator model 'gpt-4' is not a usable agno model string"):
        build_model("simulator", "gpt-4")
    with pytest.raises(ConfigError, match="not supported"):
        build_model("simulator", "carrier-pigeon:coo")


def test_resolve_model_builds_the_chosen_string(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []
    model = cast("Model", object())

    def fake_get_model(value: str) -> Model:
        built.append(value)
        return model

    monkeypatch.setattr(models_module, "get_model", fake_get_model)
    monkeypatch.setenv("SIMULHAUSEN_MODEL", "openai:gpt-4.1-mini")

    assert resolve_model("judge", None, None) is model
    assert built == ["openai:gpt-4.1-mini"]
