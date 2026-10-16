"""Target configs: how the Harbor commands reach the agent under test.

A target is a table such as ``{type = "agentos", base_url = "http://agent-api:7777",
agent_id = "support", api_key_env = "AGENTOS_API_KEY"}``. ``type`` picks an adapter
factory and every other key becomes a keyword argument of it. Factories are resolved by
import string when a target is built, never at import time, so this module does not
depend on any adapter module and an unused protocol costs nothing.

Secrets never belong in a scenario file, which ends up baked into a task image. A key
ending in ``_env`` names an environment variable instead: ``api_key_env = "AGENTOS_KEY"``
becomes ``api_key=os.environ["AGENTOS_KEY"]`` when the target is built, inside the
container, where Harbor puts the variables a task or a run passes in.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from simulhausen.adapters import agno_adapter
from simulhausen.harbor.errors import ConfigError
from simulhausen.types import AgentAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from simulhausen.harbor.spec import ScenarioSpec

__all__ = [
    "FACTORIES",
    "TARGET_ENV",
    "build_target",
    "import_object",
    "redact_target",
    "resolve_target_config",
]

# Target type -> "module:attribute" of the factory that builds the adapter.
FACTORIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "agentos": "simulhausen.http_adapters:agentos_adapter",
        "openai-chat": "simulhausen.http_adapters:openai_chat_adapter",
        "openai-responses": "simulhausen.http_adapters:openai_responses_adapter",
        "a2a": "simulhausen.http_adapters:a2a_adapter",
        "langgraph": "simulhausen.http_adapters:langgraph_adapter",
    }
)

# A target as JSON, for a run that points existing tasks at another agent.
TARGET_ENV: Final = "SIMULHAUSEN_TARGET"

_ENV_SUFFIX: Final = "_env"
# Option names that may carry a secret; the transcript records the target without them.
_SECRET_KEY = re.compile(r"key|secret|token|password|credential|auth|header", re.IGNORECASE)


def import_object(reference: str, *, what: str) -> object:
    """The object a ``"package.module:attribute"`` string names; ``attribute`` may be dotted."""
    module_name, sep, attribute = reference.partition(":")
    if not sep or not module_name or not attribute:
        raise ConfigError(f"{what}: {reference!r} is not an import string of the form 'package.module:attribute'")
    try:
        found: object = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"{what}: cannot import {module_name!r} ({exc})") from exc
    for part in attribute.split("."):
        try:
            found = getattr(found, part)
        except AttributeError:
            raise ConfigError(f"{what}: {module_name!r} has no attribute {attribute!r}") from None
    return found


def _resolve_env(options: dict[str, Any]) -> dict[str, Any]:
    """Replace every ``<name>_env = "VAR"`` option with ``<name> = $VAR``."""
    resolved: dict[str, Any] = {}
    for key, value in options.items():
        if not key.endswith(_ENV_SUFFIX) or key == _ENV_SUFFIX:
            resolved[key] = value
            continue
        name = key.removesuffix(_ENV_SUFFIX)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"target option {key} must name an environment variable, got {value!r}")
        if name in options:
            raise ConfigError(f"target sets both {name} and {key}; keep one")
        secret = os.environ.get(value)
        if not secret:
            raise ConfigError(
                f"target option {key} names the environment variable {value}, which is not set in this container; "
                f"pass it in (agent phase: `--ae {value}=...`; verifier: `[verifier.env]`, export --env {value})"
            )
        resolved[name] = secret
    return resolved


def _as_adapter(produced: object, factory: str) -> AgentAdapter:
    if isinstance(produced, AgentAdapter):
        return produced
    if hasattr(produced, "arun"):
        # An agno Agent, Team or Workflow (or anything with the same arun).
        return agno_adapter(produced)  # ty: ignore[invalid-argument-type]
    raise ConfigError(
        f"target factory {factory!r} returned {type(produced).__name__}; it must return an AgentAdapter "
        "(simulhausen.function_adapter(...), a subclass) or an agno Agent, Team or Workflow"
    )


def _build_python_target(options: dict[str, Any]) -> AgentAdapter:
    reference = options.pop("factory", None)
    if not isinstance(reference, str):
        raise ConfigError("a target of type 'python' needs `factory = \"package.module:attribute\"`")
    factory = import_object(reference, what="target factory")
    if isinstance(factory, AgentAdapter) or hasattr(factory, "arun"):
        if options:
            raise ConfigError(
                f"target factory {reference!r} is not callable, so it takes no options: {sorted(options)}"
            )
        return _as_adapter(factory, reference)
    if not callable(factory):
        raise ConfigError(f"target factory {reference!r} is a {type(factory).__name__}, not a callable")
    try:
        produced = factory(**options)
    except TypeError as exc:
        raise ConfigError(f"target factory {reference!r} rejected the options {sorted(options)}: {exc}") from exc
    return _as_adapter(produced, reference)


def build_target(config: Mapping[str, Any]) -> AgentAdapter:
    """Build the adapter for the agent under test from a target table.

    ``type`` selects the factory (see :data:`FACTORIES`; ``"python"`` takes ``factory =
    "package.module:attribute"``, a callable returning an ``AgentAdapter`` or an agno
    Agent, Team or Workflow). Every other key is passed as a keyword argument, after the
    ``*_env`` keys are replaced by the variables they name. Raises
    :class:`~simulhausen.harbor.ConfigError` with what to fix.
    """
    options = dict(config)
    kind = options.pop("type", None)
    if not isinstance(kind, str) or not kind:
        raise ConfigError(f"a target needs a `type` ({', '.join([*FACTORIES, 'python'])}), got {dict(config)!r}")
    options = _resolve_env(options)
    if kind == "python":
        return _build_python_target(options)
    if kind not in FACTORIES:
        raise ConfigError(f"unknown target type {kind!r}; known types: {', '.join([*FACTORIES, 'python'])}")
    factory = import_object(FACTORIES[kind], what=f"target type {kind!r}")
    if not callable(factory):
        raise ConfigError(f"target type {kind!r}: {FACTORIES[kind]} is not callable")
    try:
        produced = factory(**options)
    except TypeError as exc:
        raise ConfigError(f"target type {kind!r} rejected the options {sorted(options)}: {exc}") from exc
    return _as_adapter(produced, FACTORIES[kind])


def redact_target(config: Mapping[str, Any]) -> dict[str, Any]:
    """The target as the transcript records it: options that may hold a secret are dropped.

    ``*_env`` options stay, since they hold only the name of a variable.
    """
    return {key: value for key, value in config.items() if key.endswith(_ENV_SUFFIX) or not _SECRET_KEY.search(key)}


def _parse_target_json(text: str, source: str) -> dict[str, Any]:
    try:
        target = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(target, dict):
        raise ConfigError(f'{source} must be a JSON object such as {{"type": "agentos", ...}}')
    return target


def resolve_target_config(
    scenario: ScenarioSpec, *, target_json: str | None = None, target_file: Path | None = None
) -> dict[str, Any]:
    """The target to play against: ``--target-json`` first, then ``--target-file``, then
    ``$SIMULHAUSEN_TARGET`` (JSON), then the scenario's ``target`` (or the file's default).
    """
    if target_json:
        return _parse_target_json(target_json, "--target-json")
    if target_file is not None:
        try:
            text = target_file.read_text()
        except OSError as exc:
            raise ConfigError(f"cannot read --target-file {target_file}: {exc.strerror or exc}") from exc
        return _parse_target_json(text, f"--target-file {target_file}")
    if from_env := os.environ.get(TARGET_ENV):
        return _parse_target_json(from_env, f"${TARGET_ENV}")
    if scenario.target is not None:
        return dict(scenario.target)
    raise ConfigError(
        f"no target for scenario {scenario.id!r}: set `target` in the scenario file (or in [defaults]), "
        f"${TARGET_ENV}, --target-json or --target-file"
    )
