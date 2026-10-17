"""Scenario files for the Harbor commands: TOML or JSON data instead of Python.

A Harbor task is data, so the scenario it runs is too. The file mirrors
:func:`simulhausen.run`: a scenario has a name, a description for the user simulator,
criteria for the judge, an optional script and a ``target`` that says how to reach the
agent under test. Python callables cannot live in a data file, so script assertions
become declarative checks (a tool was called, was not called, ran in order, a reply
contains a text), evaluated on the recorded dialogue in the judge phase. A check's
``evaluate`` takes the state and, optionally, the indices of the assistant messages the
script injected (``Transcript.injected``), which reply checks skip.

``[defaults]`` holds keys shared by every scenario of the file. A scenario key replaces
the default as a whole: tables such as ``target`` are not merged key by key, so a
scenario that overrides ``target`` spells out the whole table.

Unknown keys are errors, because a typo such as ``critera`` must not silently drop the
criteria, and every error names the scenario and the key it is about.
"""

from __future__ import annotations

import json
import re
import tomllib
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from simulhausen.harbor.errors import ConfigError

if TYPE_CHECKING:
    import os
    from collections.abc import Collection, Sequence

    from simulhausen.types import ScenarioState

__all__ = [
    "CheckSpec",
    "JudgeCheckpointSpec",
    "MessageSpec",
    "ProceedSpec",
    "ReplyContainsCheck",
    "ReplyNotContainsCheck",
    "ScenarioSpec",
    "ScriptStepSpec",
    "ToolCalledCheck",
    "ToolNotCalledCheck",
    "ToolsInOrderCheck",
    "load_scenarios",
    "select_scenario",
    "slugify",
]

# A scenario id becomes the second half of the Harbor task name "<org>/<id>" and the
# task directory name, so it follows Harbor's package-name rule.
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def slugify(name: str) -> str:
    """The default scenario id: the name in lowercase ASCII, other characters folded into ``-``."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def _contains(text: str, needle: str, case_sensitive: bool) -> bool:
    return needle in text if case_sensitive else needle.casefold() in text.casefold()


def _agent_replies(state: ScenarioState, injected: Collection[int]) -> list[str]:
    """The replies of the agent under test: the assistant messages the script did not inject."""
    return [
        message["content"]
        for index, message in enumerate(state.messages)
        if message["role"] == "assistant" and index not in injected
    ]


class _Data(BaseModel):
    """A table of the scenario file: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


# -- checks --------------------------------------------------------------------


class ToolCalledCheck(_Data):
    """Passes when ``tool`` ran cleanly at least once (``ScenarioState.has_tool_call``)."""

    type: Literal["tool_called"]
    tool: str = Field(min_length=1)

    def describe(self) -> str:
        return f"tool {self.tool!r} was called"

    def evaluate(self, state: ScenarioState, injected: Collection[int] = ()) -> bool:
        return state.has_tool_call(self.tool)


class ToolNotCalledCheck(_Data):
    """Passes when ``tool`` never ran cleanly; a call that errored or is paused does not count."""

    type: Literal["tool_not_called"]
    tool: str = Field(min_length=1)

    def describe(self) -> str:
        return f"tool {self.tool!r} was not called"

    def evaluate(self, state: ScenarioState, injected: Collection[int] = ()) -> bool:
        return not state.has_tool_call(self.tool)


class ToolsInOrderCheck(_Data):
    """Passes when the tools ran in this order; other calls may sit between them."""

    type: Literal["tools_in_order"]
    tools: list[str] = Field(min_length=1)

    def describe(self) -> str:
        return "tools ran in order: " + " -> ".join(self.tools)

    def evaluate(self, state: ScenarioState, injected: Collection[int] = ()) -> bool:
        return state.has_tool_calls_in_order(self.tools)


class ReplyContainsCheck(_Data):
    """Passes when any reply of the agent under test contains ``text`` (case-insensitive
    unless ``case_sensitive``); replies at the ``injected`` indices do not count."""

    type: Literal["reply_contains"]
    text: str = Field(min_length=1)
    case_sensitive: bool = False

    def describe(self) -> str:
        return f"an agent reply contains {self.text!r}"

    def evaluate(self, state: ScenarioState, injected: Collection[int] = ()) -> bool:
        return any(_contains(reply, self.text, self.case_sensitive) for reply in _agent_replies(state, injected))


class ReplyNotContainsCheck(_Data):
    """Passes when no reply of the agent under test contains ``text`` (case-insensitive
    unless ``case_sensitive``); replies at the ``injected`` indices do not count."""

    type: Literal["reply_not_contains"]
    text: str = Field(min_length=1)
    case_sensitive: bool = False

    def describe(self) -> str:
        return f"no agent reply contains {self.text!r}"

    def evaluate(self, state: ScenarioState, injected: Collection[int] = ()) -> bool:
        return not any(_contains(reply, self.text, self.case_sensitive) for reply in _agent_replies(state, injected))


CheckSpec = Annotated[
    ToolCalledCheck | ToolNotCalledCheck | ToolsInOrderCheck | ReplyContainsCheck | ReplyNotContainsCheck,
    Field(discriminator="type"),
]


# -- script --------------------------------------------------------------------


class ProceedSpec(_Data):
    """``proceed = { turns = 3 }``: the simulator and the agent talk on their own."""

    turns: int | None = Field(default=None, ge=1)


class JudgeCheckpointSpec(_Data):
    """``judge = { criteria = [...] }``: criteria graded on the dialogue up to this step.

    The agent-phase copy of a scenario keeps the step with its criteria removed, so the
    dialogue still records where the checkpoint sits; the judge phase reads the criteria
    from the full scenario file.
    """

    criteria: list[str] = Field(default_factory=list)
    additional_context: str | None = None


class MessageSpec(_Data):
    """``message = { role = "assistant", content = "..." }``: a message appended as is."""

    role: Literal["user", "assistant"]
    content: str


STEP_KINDS: tuple[str, ...] = ("user", "agent", "proceed", "judge", "check", "message")


class ScriptStepSpec(_Data):
    """One script step: a table with exactly one of ``user``, ``agent``, ``proceed``,
    ``judge``, ``check`` or ``message``.

    ``user = "text"`` is a fixed user message and ``user = true`` one the simulator
    writes; ``agent = true`` calls the agent under test and ``agent = "text"`` injects a
    reply. ``judge = true`` is the final verdict: the dialogue stops there.
    """

    user: str | Literal[True] | None = None
    agent: str | Literal[True] | None = None
    proceed: ProceedSpec | Literal[True] | None = None
    judge: JudgeCheckpointSpec | Literal[True] | None = None
    check: CheckSpec | None = None
    message: MessageSpec | None = None

    @model_validator(mode="after")
    def _exactly_one_action(self) -> Self:
        given = [kind for kind in STEP_KINDS if getattr(self, kind) is not None]
        if len(given) != 1:
            raise ValueError(
                f"a script step needs exactly one of {', '.join(STEP_KINDS)}, got {', '.join(given) or 'none'}"
            )
        return self

    @property
    def kind(self) -> str:
        return next(kind for kind in STEP_KINDS if getattr(self, kind) is not None)

    @property
    def is_final_judge(self) -> bool:
        return self.judge is True

    @property
    def is_checkpoint(self) -> bool:
        """A ``judge = { criteria }`` or ``check`` step, graded on the dialogue up to here."""
        return isinstance(self.judge, JudgeCheckpointSpec) or self.check is not None


# -- scenario ------------------------------------------------------------------


class ScenarioSpec(_Data):
    """One scenario of a scenario file, with ``[defaults]`` already merged in.

    The fields map onto :func:`simulhausen.run`, :class:`~simulhausen.UserSimulatorAgent`
    (``persona``, ``simulator_prompt``) and :class:`~simulhausen.JudgeAgent` (``criteria``,
    ``score_threshold``, ``additional_context``). ``target`` is the agent under test (see
    :func:`simulhausen.harbor.build_target`), and the two model fields are agno model
    strings such as ``"openai:gpt-4.1-mini"``.
    """

    # Defaults to the slugified name; the Harbor task is named "<org>/<id>".
    id: str = ""
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    persona: str | None = None
    simulator_prompt: str | None = None
    criteria: list[str] = Field(default_factory=list)
    score_threshold: int | None = Field(default=None, ge=1, le=10)
    additional_context: str | None = None
    # None falls back to simulhausen's defaults (10 and 1).
    max_turns: int | None = Field(default=None, ge=1)
    min_turns: int | None = Field(default=None, ge=1)
    tags: list[str] = Field(default_factory=list)
    checks: list[CheckSpec] = Field(default_factory=list)
    # None: a single proceed step, the simulator and the agent talk freely.
    script: list[ScriptStepSpec] | None = Field(default=None, min_length=1)
    target: dict[str, Any] | None = None
    simulator_model: str | None = None
    judge_model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("id") and isinstance(data.get("name"), str):
            return {**data, "id": slugify(data["name"])}
        return data

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value) or ".." in value:
            raise ValueError(
                f"{value!r} is not a usable id: it becomes the Harbor task name, so use letters, digits, '.', '_' "
                "and '-', starting with a letter or a digit (set `id` when the name has no such characters)"
            )
        return value

    @field_validator("target")
    @classmethod
    def _target_has_type(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None and not (isinstance(value.get("type"), str) and value["type"]):
            raise ValueError('a target needs a `type`, such as `type = "agentos"` or `type = "python"`')
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for index, step in enumerate(self.script or []):
            if step.is_final_judge and index != len(self.script or []) - 1:
                raise ValueError(
                    f"script[{index}] is `judge = true`, the final verdict, and the dialogue stops there, so no step "
                    "may follow it; checks that should see the whole dialogue go into `checks`"
                )
        if self.score_threshold is not None and not self.criteria:
            raise ValueError("score_threshold grades the dialogue against the scenario's criteria; add `criteria`")
        return self

    # -- what the phases need -------------------------------------------------

    @property
    def steps(self) -> list[ScriptStepSpec]:
        """The script as played: a single ``proceed`` step when the file has none."""
        return self.script if self.script is not None else [ScriptStepSpec(proceed=True)]

    def uses_simulator(self) -> bool:
        """Whether any step needs the user simulator (``user = true`` or ``proceed``)."""
        return any(step.user is True or step.proceed is not None for step in self.steps)

    def checkpoint_steps(self) -> list[int]:
        """Script indices of the ``judge = { criteria }`` and ``check`` steps."""
        return [index for index, step in enumerate(self.steps) if step.is_checkpoint]

    def grading_problems(self) -> list[str]:
        """Why the judge phase could not grade this scenario; empty when it can.

        Kept out of validation on purpose: the agent-phase copy of a scenario has no
        criteria at all, and must still load.
        """
        problems: list[str] = []
        has_checkpoint = False
        for index, step in enumerate(self.steps):
            if step.is_final_judge and not self.criteria:
                problems.append(f"script[{index}] is `judge = true` (the final verdict), but there are no `criteria`")
            if isinstance(step.judge, JudgeCheckpointSpec):
                has_checkpoint = True
                if not step.judge.criteria:
                    problems.append(f"script[{index}] is a judge checkpoint without `criteria`")
            has_checkpoint = has_checkpoint or step.check is not None
        if not (self.criteria or self.checks or has_checkpoint):
            problems.append("there is nothing to grade: add `criteria`, `checks`, or judge/check steps to the script")
        return problems

    def require_gradable(self) -> None:
        """Raise :class:`ConfigError` when :meth:`grading_problems` finds anything."""
        if problems := self.grading_problems():
            raise ConfigError(f"scenario {self.id!r} cannot be graded: " + "; ".join(problems))

    def for_agent_phase(self) -> ScenarioSpec:
        """A copy without anything only the judge needs: criteria, checks, context, judge model.

        Checkpoint steps stay, with their criteria removed, so the dialogue still records
        where they sit. ``check`` steps stay whole: they name tools and texts, not criteria.
        """
        script = None
        if self.script is not None:
            script = [
                step.model_copy(update={"judge": JudgeCheckpointSpec()})
                if isinstance(step.judge, JudgeCheckpointSpec)
                else step
                for step in self.script
            ]
        return self.model_copy(
            update={
                "criteria": [],
                "checks": [],
                "additional_context": None,
                "score_threshold": None,
                "judge_model": None,
                "script": script,
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        """The scenario as a single JSON object, as ``export`` writes it into a task."""
        return self.model_dump(mode="json", exclude_none=True)


# -- loading -------------------------------------------------------------------


def _location(loc: Sequence[int | str]) -> str:
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text


def _format_errors(exc: ValidationError) -> list[str]:
    return [
        f"{_location(error['loc'])}: {error['msg']}" if error["loc"] else error["msg"]
        for error in exc.errors(include_url=False)
    ]


def _read_document(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read scenario file {path}: {exc.strerror or exc}") from exc
    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            data = tomllib.loads(raw.decode())
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            raise ConfigError(f"scenario file {path} must end in .toml or .json")
    except (tomllib.TOMLDecodeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"scenario file {path} is not valid {suffix[1:].upper()}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"scenario file {path} must hold an object with `scenarios`, or a single scenario")
    return data


def _entries(path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    """The scenario tables of a document, each with ``[defaults]`` merged in."""
    if "scenarios" not in data:
        if "defaults" in data:
            raise ConfigError(f"{path}: [defaults] applies to [[scenarios]], and the file has none")
        if not data:
            raise ConfigError(f"{path} holds no scenario")
        # A file with one scenario at the top level, as `export` writes scenario.json.
        return [data]
    unknown = sorted(set(data) - {"defaults", "scenarios"})
    if unknown:
        raise ConfigError(f"{path}: unknown top-level keys {unknown}; expected [defaults] and [[scenarios]]")
    defaults = data.get("defaults", {})
    scenarios = data["scenarios"]
    if not isinstance(defaults, dict):
        raise ConfigError(f"{path}: [defaults] must be a table")
    if not isinstance(scenarios, list) or not scenarios or not all(isinstance(s, dict) for s in scenarios):
        raise ConfigError(f"{path}: `scenarios` must be a non-empty array of tables ([[scenarios]])")
    unknown = sorted(set(defaults) - set(ScenarioSpec.model_fields))
    if unknown:
        raise ConfigError(f"{path}: unknown keys in [defaults]: {unknown}")
    if "id" in defaults:
        raise ConfigError(f"{path}: `id` cannot be a default, every scenario needs its own")
    return [{**defaults, **scenario} for scenario in scenarios]


def load_scenarios(path: str | os.PathLike[str]) -> list[ScenarioSpec]:
    """Read a scenario file (``.toml`` or ``.json``) and return its scenarios in file order.

    The file holds ``[defaults]`` and ``[[scenarios]]`` (JSON: ``{"defaults": {...},
    "scenarios": [...]}``), or a single scenario at the top level. Raises
    :class:`~simulhausen.harbor.ConfigError` listing every problem found, each with the
    scenario and the key it concerns.
    """
    path = Path(path)
    entries = _entries(path, _read_document(path))
    specs: list[ScenarioSpec] = []
    problems: list[str] = []
    for index, entry in enumerate(entries):
        label = f"scenarios[{index}]" + (f" ({entry['name']!r})" if isinstance(entry.get("name"), str) else "")
        try:
            specs.append(ScenarioSpec.model_validate(entry))
        except ValidationError as exc:
            problems.extend(f"{label}: {message}" for message in _format_errors(exc))
    if problems:
        raise ConfigError(f"scenario file {path} has errors:\n  " + "\n  ".join(problems))
    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise ConfigError(
                f"{path}: two scenarios have the id {spec.id!r}; ids must be unique, set `id` to tell them apart"
            )
        seen.add(spec.id)
    return specs


def select_scenario(
    specs: Sequence[ScenarioSpec], scenario_id: str | None, source: object = "the file"
) -> ScenarioSpec:
    """The scenario ``--id`` names; a file with a single scenario needs no id."""
    ids = [spec.id for spec in specs]
    if scenario_id is None:
        if len(specs) == 1:
            return specs[0]
        raise ConfigError(f"{source} holds {len(specs)} scenarios, pick one with --id: {', '.join(ids)}")
    for spec in specs:
        if spec.id == scenario_id:
            return spec
    raise ConfigError(f"no scenario with id {scenario_id!r} in {source}; its ids are: {', '.join(ids)}")
