from collections.abc import Callable
from typing import cast

import pytest
from agno.models.base import Model

import simulhausen as sim
from simulhausen import SimulationConfig, configure, get_config
from simulhausen.config import require_model, resolve_parser_model


def _model() -> Model:
    return cast(Model, object())


def test_defaults() -> None:
    assert get_config() == SimulationConfig()
    config = get_config()
    assert (config.model, config.parser_model) == (None, None)
    assert (config.max_turns, config.min_turns) == (10, 1)
    assert (config.verbose, config.debug) == (True, False)
    assert (config.agent_name, config.user_name) == ("Agent", "User")
    assert config.cache_key is None


def test_get_config_returns_the_live_object() -> None:
    assert get_config() is get_config()
    configure(max_turns=3)
    assert get_config().max_turns == 3


def test_configure_updates_only_the_given_fields() -> None:
    model, parser = _model(), _model()
    configure(model=model, max_turns=4, user_name="Customer")
    configure(parser_model=parser, verbose=False, debug=True, min_turns=2, agent_name="Ellie", cache_key="suite-v1")

    assert get_config() == SimulationConfig(
        model=model,
        parser_model=parser,
        max_turns=4,
        min_turns=2,
        verbose=False,
        debug=True,
        agent_name="Ellie",
        user_name="Customer",
        cache_key="suite-v1",
    )


def test_configure_without_arguments_changes_nothing() -> None:
    configure(max_turns=5)
    configure()

    assert get_config().max_turns == 5


@pytest.mark.parametrize(
    ("field", "call"),
    [
        ("max_turns", lambda value: configure(max_turns=value)),
        ("min_turns", lambda value: configure(min_turns=value)),
    ],
)
@pytest.mark.parametrize("value", [0, -3])
def test_configure_rejects_turn_limits_below_one(field: str, call: Callable[[int], None], value: int) -> None:
    with pytest.raises(ValueError, match=f"{field} must be at least 1"):
        call(value)
    assert getattr(get_config(), field) == getattr(SimulationConfig(), field)


def test_configure_keeps_the_cache_key_when_it_is_not_passed() -> None:
    configure(cache_key="suite-v1")
    configure(max_turns=3)

    assert get_config().cache_key == "suite-v1"


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_configure_rejects_a_blank_cache_key(value: str) -> None:
    with pytest.raises(ValueError, match="cache_key must be a non-empty string"):
        configure(cache_key=value)
    assert get_config().cache_key is None


def test_configure_rejects_a_cache_key_that_is_not_a_string() -> None:
    with pytest.raises(ValueError, match="cache_key must be a non-empty string"):
        configure(cache_key=cast("str", 42))
    assert get_config().cache_key is None


def test_require_model_prefers_the_explicit_model() -> None:
    explicit, configured = _model(), _model()
    configure(model=configured)

    assert require_model(explicit) is explicit
    assert require_model() is configured


def test_require_model_error_explains_how_to_configure() -> None:
    with pytest.raises(RuntimeError, match=r"no simulation model configured.*simulhausen\.configure\(model=\.\.\.\)"):
        require_model()


def test_resolve_parser_model_never_raises() -> None:
    explicit, configured = _model(), _model()
    assert resolve_parser_model() is None

    configure(parser_model=configured)
    assert resolve_parser_model() is configured
    assert resolve_parser_model(explicit) is explicit


def test_public_api_names_resolve() -> None:
    assert all(hasattr(sim, name) for name in sim.__all__)
    assert sim.__version__
