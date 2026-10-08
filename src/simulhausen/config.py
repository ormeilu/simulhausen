"""Process-wide simulation defaults (the counterpart of ``scenario.configure``).

Nothing here imports a model at module level: building a model often reads
secrets or credentials, and that must not happen while pytest merely collects
tests. Set the model from a session fixture with :func:`configure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agno.models.base import Model

__all__ = ["SimulationConfig", "configure", "get_config"]


@dataclass
class SimulationConfig:
    """Defaults shared by every scenario in the process."""

    # Model behind the user simulator and the judge (any agno Model).
    model: Model | None = None
    # Optional parser model for the helper agents' output_schema.
    #
    # Some backends (vLLM with a reasoning parser, for one) return broken or
    # empty content when response_format is combined with thinking mode: the
    # reasoning is filled in, the content is not. With parser_model set, agno
    # sends response_format to the parser call only, so the main call keeps
    # thinking and the parser (thinking off) produces the structured object.
    # use_json_mode=True is not a substitute: it swaps json_schema for
    # json_object but keeps response_format on the same call.
    parser_model: Model | None = None
    # Ceiling on user -> agent turns for unscripted runs and proceed().
    max_turns: int = 10
    # Floor on agent turns before the judge may end an unscripted dialogue
    # early; explicit judge() steps and the end of max_turns ignore it.
    min_turns: int = 1
    # Log every user/agent message and judge verdict at INFO level.
    verbose: bool = True
    # Debug mode: before each simulated user turn, ask on stdin for the message
    # (Enter lets the simulator write it). Needs pytest -s, or --sim-debug.
    debug: bool = False
    # Label of the agent under test in reports and transcripts.
    agent_name: str = "Agent"
    # Label of the simulated user in reports and transcripts.
    user_name: str = "User"


_config = SimulationConfig()


def configure(
    *,
    model: Model | None = None,
    parser_model: Model | None = None,
    max_turns: int | None = None,
    min_turns: int | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
    agent_name: str | None = None,
    user_name: str | None = None,
) -> None:
    """Set simulation defaults; call once, e.g. from a session-scoped fixture.

    Only the arguments you pass change; the rest keep their current values.
    """
    if model is not None:
        _config.model = model
    if parser_model is not None:
        _config.parser_model = parser_model
    if max_turns is not None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        _config.max_turns = max_turns
    if min_turns is not None:
        if min_turns < 1:
            raise ValueError("min_turns must be at least 1")
        _config.min_turns = min_turns
    if verbose is not None:
        _config.verbose = verbose
    if debug is not None:
        _config.debug = debug
    if agent_name is not None:
        _config.agent_name = agent_name
    if user_name is not None:
        _config.user_name = user_name


def get_config() -> SimulationConfig:
    """The live config object (mutations through :func:`configure` show up here)."""
    return _config


def resolve_parser_model(explicit: Model | None = None) -> Model | None:
    """Parser model for schema-bound helper agents; ``None`` leaves parsing to the backend.

    Unlike :func:`require_model` this never raises: a simulation still works
    without a parser model, only with a higher risk of off-schema replies.
    """
    return explicit or _config.parser_model


def require_model(explicit: Model | None = None) -> Model:
    """Model for the helper agents: explicit argument first, then the global default."""
    model = explicit or _config.model
    if model is None:
        raise RuntimeError(
            "no simulation model configured: call simulhausen.configure(model=...) "
            "(for example from a session-scoped conftest fixture) or pass model= explicitly"
        )
    return model
