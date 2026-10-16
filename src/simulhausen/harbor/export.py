"""``simulhausen harbor export``: one Harbor task directory per scenario.

This is Harbor's adapter pattern: a generator that renders benchmark items into task
directories, here one per scenario. Each task gets ``task.toml``, ``instruction.md`` (the
scenario description, which Harbor hands to the agent as its instruction), an
``environment/Dockerfile`` with simulhausen and the model SDKs preinstalled, and
``tests/`` with the full scenario and a ``test.sh`` that writes the reward.

Two modes decide which phase plays the dialogue:

* ``agent`` (default): the agent phase plays it, through a Harbor agent that runs
  ``simulhausen harbor play`` in the task container (``simulhausen-harbor``'s agents), and
  the verifier grades the transcript with ``simulhausen harbor judge``. The image carries
  the scenario without its criteria; the full scenario reaches the container only with
  the verifier's ``tests/``.
* ``verifier``: the verifier plays and grades in one go (``simulhausen harbor run``) and
  the agent phase does nothing (``harbor run -a nop``). No Harbor-side package needed,
  but Harbor's agent column then says ``nop``.

Secrets never go into the files: ``[verifier.env]`` passes host variables through by
name (``"${OPENAI_API_KEY}"``), resolved by Harbor when the job starts.
"""

from __future__ import annotations

import logging
import math
import re
import shlex
import shutil
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from simulhausen import __version__
from simulhausen.harbor.errors import ConfigError
from simulhausen.harbor.paths import IMAGE_SCENARIO_PATH, TESTS_SCENARIO_PATH, VERIFIER_LOGS_DIR
from simulhausen.harbor.spec import load_scenarios
from simulhausen.harbor.targets import TARGET_ENV
from simulhausen.harbor.transcript import write_json

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from simulhausen.harbor.judge import RewardMode, WarnReward
    from simulhausen.harbor.spec import ScenarioSpec

__all__ = [
    "DEFAULT_ENV",
    "DEFAULT_ORG",
    "DEFAULT_PIP",
    "EXPORT_MODES",
    "PYTHON_IMAGE",
    "UV_IMAGE",
    "ExportMode",
    "ExportOptions",
    "default_simulhausen_spec",
    "export_tasks",
]

logger = logging.getLogger(__name__)

type ExportMode = Literal["agent", "verifier"]
EXPORT_MODES: Final[tuple[ExportMode, ...]] = ("agent", "verifier")

PYTHON_IMAGE: Final = "python:3.12-slim"
# Pinned, so a task builds the same image next year.
UV_IMAGE: Final = "ghcr.io/astral-sh/uv:0.11.32"
TASK_SCHEMA_VERSION: Final = "1.4"
GENERATOR: Final = "simulhausen"

DEFAULT_ORG: Final = "simulhausen"
DEFAULT_ENV: Final[tuple[str, ...]] = ("OPENAI_API_KEY",)
DEFAULT_PIP: Final[tuple[str, ...]] = ("openai",)
DEFAULT_AGENT_TIMEOUT_SEC: Final = 900.0
DEFAULT_VERIFIER_TIMEOUT_SEC: Final = 600.0
# Verifier mode plays the whole dialogue inside the verifier's time budget.
DEFAULT_RUN_VERIFIER_TIMEOUT_SEC: Final = 1800.0
BUILD_TIMEOUT_SEC: Final = 600.0
CPUS: Final = 1
MEMORY_MB: Final = 2048

# Variables passed through to the verifier when the host has them, empty otherwise
# (simulhausen treats an empty variable as unset).
_JUDGE_PASSTHROUGH: Final = ("SIMULHAUSEN_JUDGE_MODEL", "SIMULHAUSEN_MODEL")
_RUN_PASSTHROUGH: Final = (*_JUDGE_PASSTHROUGH, "SIMULHAUSEN_SIMULATOR_MODEL", TARGET_ENV)

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ORG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TOML_ESCAPES: Final = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


@dataclass(frozen=True)
class ExportOptions:
    """What ``export`` renders; the defaults match the command line's."""

    output_dir: Path
    mode: ExportMode = "agent"
    # The Harbor task name is "<org>/<scenario id>".
    org: str = DEFAULT_ORG
    # pip requirement for simulhausen in the image; None: default_simulhausen_spec().
    simulhausen_spec: str | None = None
    # Extra requirements for the image, such as the model SDKs.
    pip: Sequence[str] = DEFAULT_PIP
    # Host variables the verifier gets, as NAME = "${NAME}".
    env: Sequence[str] = DEFAULT_ENV
    # Copied to environment/docker-compose.yaml: an agent-under-test sidecar bound to the task.
    compose: Path | None = None
    # None: the mode's default.
    agent_timeout: float | None = None
    verifier_timeout: float | None = None
    # Only these scenario ids; None: every scenario.
    ids: Sequence[str] | None = None
    overwrite: bool = False
    # Passed on to the judge in tests/test.sh.
    rewards: RewardMode = "binary"
    warn_reward: WarnReward = "skip"


def default_simulhausen_spec() -> str:
    """``simulhausen==<this version>``, or plain ``simulhausen`` when this version is not on PyPI."""
    if "+" in __version__:
        # A source tree ("0+unknown") or a local build: no such release to pin.
        logger.warning(
            "simulhausen %s is not a released version; the task image installs the latest release instead. "
            "Pass --simulhausen-spec to pin one.",
            __version__,
        )
        return "simulhausen"
    return f"simulhausen=={__version__}"


def _toml_string(value: str) -> str:
    escaped = "".join(
        _TOML_ESCAPES.get(char) or (f"\\u{ord(char):04x}" if ord(char) < 0x20 or ord(char) == 0x7F else char)
        for char in value
    )
    return f'"{escaped}"'


def _toml_strings(values: Sequence[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _timeouts(options: ExportOptions) -> tuple[float, float]:
    agent = DEFAULT_AGENT_TIMEOUT_SEC if options.agent_timeout is None else options.agent_timeout
    default_verifier = DEFAULT_RUN_VERIFIER_TIMEOUT_SEC if options.mode == "verifier" else DEFAULT_VERIFIER_TIMEOUT_SEC
    verifier = default_verifier if options.verifier_timeout is None else options.verifier_timeout
    return float(agent), float(verifier)


def _verifier_env(options: ExportOptions) -> list[tuple[str, str]]:
    names = list(dict.fromkeys(options.env))
    passthrough = _RUN_PASSTHROUGH if options.mode == "verifier" else _JUDGE_PASSTHROUGH
    return [(name, f"${{{name}}}") for name in names] + [
        (name, f"${{{name}:-}}") for name in passthrough if name not in names
    ]


def render_task_toml(spec: ScenarioSpec, options: ExportOptions) -> str:
    """``task.toml`` for ``spec`` (Harbor's TaskConfig, schema 1.4)."""
    agent_timeout, verifier_timeout = _timeouts(options)
    lines = [
        f"# Generated by simulhausen {__version__} from scenario {spec.id!r}. Edit the scenario file and export again.",
        f"schema_version = {_toml_string(TASK_SCHEMA_VERSION)}",
        "",
        "[task]",
        f"name = {_toml_string(f'{options.org}/{spec.id}')}",
        f"description = {_toml_string(spec.name)}",
        f"keywords = {_toml_strings(spec.tags)}",
        "",
        "[metadata]",
        f"generator = {_toml_string(GENERATOR)}",
        f"simulhausen_version = {_toml_string(__version__)}",
        f"scenario_id = {_toml_string(spec.id)}",
        f"mode = {_toml_string(options.mode)}",
        "",
        "[agent]",
        f"timeout_sec = {agent_timeout!r}",
        "",
        "[verifier]",
        f"timeout_sec = {verifier_timeout!r}",
        "",
        "[verifier.env]",
        *(f"{name} = {_toml_string(value)}" for name, value in _verifier_env(options)),
        "",
        "[environment]",
        f"build_timeout_sec = {BUILD_TIMEOUT_SEC!r}",
        f"cpus = {CPUS}",
        f"memory_mb = {MEMORY_MB}",
    ]
    return "\n".join(lines) + "\n"


def render_dockerfile(options: ExportOptions, simulhausen_spec: str) -> str:
    """``environment/Dockerfile``: Python, uv, simulhausen and the extra requirements."""
    requirements = " ".join(shlex.quote(requirement) for requirement in [simulhausen_spec, *options.pip])
    lines = [
        f"# Generated by simulhausen {__version__}: the task image, with simulhausen preinstalled.",
        f"FROM {PYTHON_IMAGE}",
        f"COPY --from={UV_IMAGE} /uv /uvx /bin/",
        "ENV PYTHONUNBUFFERED=1",
        f"RUN uv pip install --system --no-cache {requirements}",
    ]
    if options.mode == "agent":
        lines.append(f"COPY scenario.json {IMAGE_SCENARIO_PATH}")
    return "\n".join(lines) + "\n"


def render_test_sh(options: ExportOptions) -> str:
    """``tests/test.sh``: the verifier, which must always get to write the reward."""
    if options.mode == "agent":
        purpose = "Grades the dialogue the agent phase recorded in /logs/agent/simulhausen/."
        command = ["simulhausen", "harbor", "judge"]
    else:
        purpose = "Plays the dialogue with the agent under test, then grades it (run with `harbor run -a nop`)."
        command = ["simulhausen", "harbor", "run"]
    command += ["--scenario", TESTS_SCENARIO_PATH, "--out", VERIFIER_LOGS_DIR]
    if options.rewards != "binary":
        command += ["--rewards", options.rewards]
    if options.warn_reward != "skip":
        command += ["--warn-reward", options.warn_reward]
    lines = [
        "#!/bin/bash",
        f"# Generated by simulhausen {__version__}. {purpose}",
        "# No `set -e`: whatever happens before, simulhausen must get to write reward.json.",
        f"mkdir -p {VERIFIER_LOGS_DIR}",
        shlex.join(command),
    ]
    return "\n".join(lines) + "\n"


def _validate(options: ExportOptions) -> None:
    if not _ORG.fullmatch(options.org) or ".." in options.org:
        raise ConfigError(
            f"--org {options.org!r}: use letters, digits, '.', '_' and '-', starting with a letter or a digit"
        )
    for name in options.env:
        if not _ENV_NAME.fullmatch(name):
            raise ConfigError(f"--env {name!r} is not an environment variable name")
    for requirement in [options.simulhausen_spec or "x", *options.pip]:
        if not requirement.strip() or any(ord(char) < 0x20 for char in requirement):
            raise ConfigError(f"{requirement!r} is not a usable pip requirement")
    for flag, value in (("--agent-timeout", options.agent_timeout), ("--verifier-timeout", options.verifier_timeout)):
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ConfigError(f"{flag} must be a positive number of seconds, got {value}")
    if options.compose is not None and not options.compose.is_file():
        raise ConfigError(f"--compose {options.compose} is not a file")


def _generated_task(directory: Path) -> bool:
    """Whether ``directory`` holds a task this command generated (and may replace)."""
    try:
        config = tomllib.loads((directory / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return config.get("metadata", {}).get("generator") == GENERATOR


def _select(scenario_files: Sequence[Path], ids: Sequence[str] | None) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []
    origin: dict[str, Path] = {}
    for path in scenario_files:
        for spec in load_scenarios(path):
            if spec.id in origin:
                raise ConfigError(
                    f"scenario id {spec.id!r} is in both {origin[spec.id]} and {path}; task directories need unique ids"
                )
            origin[spec.id] = path
            specs.append(spec)
    if ids is not None:
        unknown = [scenario_id for scenario_id in ids if scenario_id not in origin]
        if unknown:
            raise ConfigError(f"--ids {unknown} not found; the scenario ids are: {', '.join(origin)}")
        specs = [spec for spec in specs if spec.id in ids]
    for spec in specs:
        spec.require_gradable()
    return specs


def export_tasks(scenario_files: Sequence[Path], options: ExportOptions) -> list[Path]:
    """Render one Harbor task directory per scenario into ``options.output_dir``.

    Everything is checked before anything is written: the scenario files, the options,
    and that no task directory would be overwritten without ``overwrite`` (which only
    replaces directories this command generated). Returns the task directories.
    """
    _validate(options)
    specs = _select(scenario_files, options.ids)
    simulhausen_spec = options.simulhausen_spec or default_simulhausen_spec()
    for spec in specs:
        task_dir = options.output_dir / spec.id
        if task_dir.exists() and not options.overwrite:
            raise ConfigError(f"{task_dir} already exists; pass --overwrite to replace it")
        if task_dir.exists() and not _generated_task(task_dir):
            raise ConfigError(f"{task_dir} exists and was not generated by simulhausen; move it away first")
    written: list[Path] = []
    for spec in specs:
        task_dir = options.output_dir / spec.id
        if task_dir.exists():
            shutil.rmtree(task_dir)
        environment, tests = task_dir / "environment", task_dir / "tests"
        environment.mkdir(parents=True)
        tests.mkdir()
        (task_dir / "task.toml").write_text(render_task_toml(spec, options))
        (task_dir / "instruction.md").write_text(spec.description.rstrip() + "\n")
        (environment / "Dockerfile").write_text(render_dockerfile(options, simulhausen_spec))
        if options.mode == "agent":
            write_json(environment / "scenario.json", spec.for_agent_phase().to_json_dict())
        if options.compose is not None:
            shutil.copyfile(options.compose, environment / "docker-compose.yaml")
        write_json(tests / "scenario.json", spec.to_json_dict())
        test_sh = tests / "test.sh"
        test_sh.write_text(render_test_sh(options))
        test_sh.chmod(0o755)
        written.append(task_dir)
    return written
