"""Models of the user simulator and the judge, from agno model strings.

A Harbor task cannot construct a Python model object, so the scenario file and the
command line name models as agno model strings, ``"<provider>:<model_id>"`` (for example
``"openai:gpt-4.1-mini"``), which ``agno.models.utils.get_model`` turns into a model.

Precedence, highest first: ``--model``, then ``$SIMULHAUSEN_SIMULATOR_MODEL`` or
``$SIMULHAUSEN_JUDGE_MODEL``, then ``$SIMULHAUSEN_MODEL``, then the scenario's
``simulator_model`` or ``judge_model``. An empty variable counts as unset: the task files
``export`` writes pass these variables through with an empty default.

agno imports a provider's SDK (``openai``, ``anthropic``...) only when the model is
built, and simulhausen does not depend on any of them. A missing SDK is reported as a
configuration error naming the package to install in the task image.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final, Literal

from agno.models.utils import get_model

from simulhausen.harbor.errors import ConfigError

if TYPE_CHECKING:
    from agno.models.base import Model

__all__ = ["MODEL_ENV", "HelperRole", "build_model", "model_string", "resolve_model"]

type HelperRole = Literal["simulator", "judge"]

# Environment variables consulted for each helper, most specific first.
MODEL_ENV: Final[dict[HelperRole, tuple[str, ...]]] = {
    "simulator": ("SIMULHAUSEN_SIMULATOR_MODEL", "SIMULHAUSEN_MODEL"),
    "judge": ("SIMULHAUSEN_JUDGE_MODEL", "SIMULHAUSEN_MODEL"),
}


def model_string(role: HelperRole, cli_value: str | None, scenario_value: str | None) -> str:
    """The model string for ``role`` by precedence; raises :class:`ConfigError` when none is set."""
    if cli_value:
        return cli_value
    for variable in MODEL_ENV[role]:
        if value := os.environ.get(variable):
            return value
    if scenario_value:
        return scenario_value
    variables = " or ".join(f"${name}" for name in MODEL_ENV[role])
    raise ConfigError(
        f"no {role} model: pass --model, set {variables}, or set `{role}_model` in the scenario file "
        "(an agno model string such as 'openai:gpt-4.1-mini')"
    )


def build_model(role: HelperRole, value: str) -> Model:
    """Build the agno model ``value`` names, with errors addressed to the scenario author."""
    try:
        model = get_model(value)
    except ImportError as exc:
        raise ConfigError(
            f"the {role} model {value!r} needs a Python package that is not installed here ({exc}); install it next "
            "to simulhausen, e.g. `simulhausen harbor export --pip <package>` for a task image"
        ) from exc
    except ValueError as exc:
        raise ConfigError(f"the {role} model {value!r} is not a usable agno model string: {exc}") from exc
    if model is None:  # pragma: no cover - get_model returns None only for None, and value is a str
        raise ConfigError(f"the {role} model {value!r} resolved to nothing")
    return model


def resolve_model(role: HelperRole, cli_value: str | None, scenario_value: str | None) -> Model:
    """:func:`model_string` followed by :func:`build_model`."""
    return build_model(role, model_string(role, cli_value, scenario_value))
