"""Harbor agents that play a simulhausen scenario against an HTTP agent under test.

Harbor scores an *agent* on *tasks*. Here the system under test is an HTTP service (an agno AgentOS app, an
OpenAI-compatible endpoint, an A2A server, a LangGraph deployment), usually a compose sidecar next to the task
container, and the Harbor agent is a thin driver: it execs ``simulhausen harbor play`` inside the task container.
There an LLM user simulator talks to the agent under test over HTTP, and the transcript lands in
``/logs/agent/simulhausen/``. The task's verifier (``tests/test.sh``) then runs ``simulhausen harbor judge`` on that
transcript and writes the reward. Criteria live in ``tests/``, so play never sees them.

These are ``BaseInstalledAgent`` subclasses rather than host-side agents because sidecars are reachable only from
inside the compose network, and because the task's agent-phase network policy should apply to the dialogue.

Identity: Harbor groups results by ``agent__model__dataset`` and ``name()`` is fixed per class, so pass the identity
of the agent under test as the model: ``-m support-agent@v3``. The agents report it whole, a ``/`` included, and only
``OpenAIChatAgent`` reads it, as the default ``model`` of the endpoint.

Secrets never go into kwargs. Harbor records kwargs in the job's ``config.json``, and the target is written to a
file inside the container. Name environment variables instead (``api_key_env=AGENTOS_API_KEY``) and pass their
values with ``--ae AGENTOS_API_KEY=...``; simulhausen resolves ``*_env`` keys inside the container.

This is the only module of simulhausen that imports Harbor (and through it litellm), so it needs the extra:
``pip install 'simulhausen[harbor]'``. ``simulhausen.harbor`` re-exports the agent classes lazily, which makes
``-a simulhausen.harbor:AgentOSAgent`` the import path to pass to ``harbor run``.
"""

import inspect
import json
import math
import re
import shlex
import shutil
import tempfile
from collections.abc import Callable, Collection, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal, Self, cast, override

from agno.models.response import ToolExecution
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.result import AgentInfo, ModelInfo
from pydantic import Field, field_validator, model_validator

from simulhausen.harbor.export import UV_VERSION, default_simulhausen_spec
from simulhausen.harbor.paths import AGENT_LOGS_SUBDIR, IMAGE_SCENARIO_PATH, TRAJECTORY_FILE, TRANSCRIPT_FILE
from simulhausen.harbor.targets import _ENV_SUFFIX, FACTORIES, import_object
from simulhausen.harbor.transcript import TRANSCRIPT_FORMAT, write_json
from simulhausen.types import _is_clean

DEFAULT_SCENARIO_PATH = IMAGE_SCENARIO_PATH
"""Where ``simulhausen harbor export`` bakes the play-side scenario into the task image."""

TARGET_FILE = "/installed-agent/simulhausen/target.json"
"""Container path of the target config handed to ``simulhausen harbor play --target-file``."""

PLAY_LOG_FILE = "play.log"
INSTALL_LOG_FILE = "simulhausen-install.log"

# uv puts tools and managed Pythons under the invoking user's home by default. Installing as root into /root would
# leave a non-root agent user with a dangling interpreter symlink, so everything goes under /opt and /usr/local/bin.
_TOOL_DIR = "/opt/simulhausen/tools"
_PYTHON_DIR = "/opt/simulhausen/python"
_BIN_DIR = "/usr/local/bin"

# play_timeout_sec sends SIGINT first, so play can unwind and still write what it has, then SIGKILL.
_PLAY_KILL_AFTER_SEC = 30
_NO_TIMEOUT_WARNING = "simulhausen: timeout(1) not found in the container; play_timeout_sec is ignored"

_API_KEY_MESSAGE = (
    "a literal api_key is not accepted: Harbor stores kwargs in the job's config.json. "
    "Pass the name of an environment variable instead (--ak api_key_env=AGENT_API_KEY) "
    "and its value with --ae AGENT_API_KEY=..."
)

# Names that usually hold credentials. A literal value under such a name only gets a warning, while api_key is
# refused: the others are free-form (AgentOS form fields, headers), where a literal may be meant, and `<name>_env`
# works for every target key.
_SENSITIVE_NAME = re.compile(r"(^|[_-])(api[_-]?key|token|secret|password|authorization|cookie)($|[_-])", re.IGNORECASE)
_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)+[0-9A-Za-z.+-]*")
_PINNED_SPEC = re.compile(r"^\s*simulhausen\s*==\s*([^\s;,]+)\s*$")
_AGENTOS_IDS = ("agent_id", "team_id", "workflow_id")


class SimulhausenOptions(InstalledAgentOptions):
    """``--ak`` kwargs every simulhausen agent accepts; ``harbor agent schema <import path>`` lists them.

    Subclasses add the flat target fields of one protocol (``base_url``, ``agent_id``...). Every field a subclass
    declares beyond these is copied into the target dict when set; an explicit ``target`` wins key by key. The
    resulting target is checked against its adapter (:func:`check_target`), so Harbor's preflight refuses a target
    that cannot work before any container starts.
    """

    target_type: ClassVar[str | None] = None
    """Default ``target["type"]``; ``None`` means the target must come complete (or not at all)."""

    harbor_target_keys: ClassVar[tuple[str, ...]] = ()
    """Target keys the agent can fill from Harbor's own arguments (``-m``), which the kwargs check does not see."""

    # Inherited from Harbor's InstalledAgentOptions; redeclared for descriptions that fit this agent.
    version: str | None = Field(
        default=None,
        description=(
            "simulhausen version installed (simulhausen==<version>) when the image has no simulhausen CLI. "
            "Default: see simulhausen_spec."
        ),
    )
    prompt_template_path: str | None = Field(
        default=None, description="Ignored: play reads the scenario file, not Harbor's instruction."
    )
    config: Any = Field(default=None, description="Not supported.")

    target: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Target config for play, e.g. "
            '{"type": "agentos", "base_url": "http://agent-api:7777", "agent_id": "support"}; '
            "a JSON string works too. Unset on SimulhausenAgent: play falls back to SIMULHAUSEN_TARGET "
            "(pass it with --ae) and then to the scenario's own target."
        ),
    )
    simulator_model: str | None = Field(
        default=None,
        description=(
            'agno model string for the user simulator, e.g. "openai:gpt-4.1-mini". Unset: '
            "SIMULHAUSEN_SIMULATOR_MODEL, SIMULHAUSEN_MODEL (via --ae), then the scenario's simulator_model."
        ),
    )
    scenario_path: str = Field(default=DEFAULT_SCENARIO_PATH, description="Scenario file inside the container.")
    simulhausen_spec: str | None = Field(
        default=None,
        description=(
            "pip requirement installed when the image has no simulhausen CLI, e.g. "
            '"simulhausen @ https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz". '
            "Default: simulhausen==<version> when version is set, else the simulhausen installed next to Harbor, "
            "as simulhausen harbor export also puts it into task images: the archive or git commit it was "
            "installed from, else simulhausen==<its version>."
        ),
    )
    pip: list[str] = Field(
        default=["openai"],
        description=(
            "Extra requirements installed next to simulhausen, typically the simulator model's SDK. Only when the "
            "agent installs simulhausen: an image that already has the CLI (an exported one) gets nothing, and an "
            "explicit pip only logs a warning. Add them to such an image with simulhausen harbor export --pip."
        ),
    )
    play_timeout_sec: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Interrupt play after this many seconds. The verifier still runs on whatever was written.",
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_literal_api_key(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "api_key" in data:
            raise ValueError(_API_KEY_MESSAGE)
        return data

    @field_validator("target", mode="before")
    @classmethod
    def _parse_target_json(cls, value: Any) -> Any:
        # `--ak target={...}` arrives as a dict, but a job YAML may well quote it as a string.
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"target is not valid JSON: {exc}") from exc
            if not isinstance(value, dict):
                # ValueError, not TypeError: pydantic turns only ValueError into a validation error.
                raise ValueError("target must be a JSON object")  # noqa: TRY004
        return value

    @field_validator("target")
    @classmethod
    def _check_target(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if "api_key" in value:
            raise ValueError(_API_KEY_MESSAGE)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target must be plain JSON: {exc}") from exc
        return value

    @field_validator("pip", mode="before")
    @classmethod
    def _listify_pip(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @model_validator(mode="after")
    def _check_target_complete(self) -> Self:
        if self.target is not None and self.target_type is None and "type" not in self.target:
            raise ValueError('target needs a "type", e.g. "agentos", "openai-chat", "a2a" or "langgraph"')
        # The target as run() builds it, short of what the agent fills from Harbor's own arguments.
        target = self.build_target()
        if target is not None:
            check_target(target, flat=self.target_fields(), filled_later=self.harbor_target_keys)
        return self

    @classmethod
    def target_fields(cls) -> list[str]:
        """Flat kwargs of this protocol: the fields a subclass declares on top of the shared ones."""
        return [name for name in cls.model_fields if name not in SimulhausenOptions.model_fields]

    def build_target(self, defaults: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """Merge ``defaults`` < flat kwargs < ``target``; ``None`` when nothing describes a target at all."""
        flat = {name: getattr(self, name) for name in self.target_fields() if getattr(self, name) is not None}
        explicit = dict(self.target or {})
        if self.target_type is None and not flat and not explicit:
            return None
        base: dict[str, Any] = {} if self.target_type is None else {"type": self.target_type}
        base.update({key: value for key, value in (defaults or {}).items() if value is not None})
        return {**base, **flat, **explicit}


class _HTTPOptions(SimulhausenOptions):
    """Knobs every HTTP adapter of simulhausen takes."""

    headers: dict[str, str] | None = Field(
        default=None,
        description="Extra HTTP headers. Recorded in the job config: keep credentials in *_env kwargs.",
    )
    # No infinity or NaN: target.json must stay plain JSON.
    timeout: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Per-request timeout in seconds."
    )
    retries: int | None = Field(default=None, ge=0, description="Retries on connect errors, timeouts, 5xx and 429.")
    retry_delay: float | None = Field(default=None, ge=0, allow_inf_nan=False, description="Seconds between retries.")


class AgentOSOptions(_HTTPOptions):
    """Target fields of an agno AgentOS app (``simulhausen.http_adapters.agentos_adapter``)."""

    target_type: ClassVar[str | None] = "agentos"

    base_url: str | None = Field(default=None, description="AgentOS base URL, e.g. http://agent-api:7777.")
    # Exactly one of the three, in the kwargs or in target.
    agent_id: str | None = Field(default=None, description="Agent to run (POST /agents/{agent_id}/runs).")
    team_id: str | None = Field(default=None, description="Team to run instead of an agent.")
    workflow_id: str | None = Field(default=None, description="Workflow to run instead of an agent.")
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding the AgentOS security key (pass it with --ae)."
    )
    user_id: str | None = Field(default=None, description="user_id sent with every run.")
    session_state: dict[str, Any] | None = Field(default=None, description="session_state sent with every run.")
    history: Literal["session", "transcript"] | None = Field(
        default=None,
        description="session: AgentOS keeps the history (needs a db); transcript: resend the dialogue every turn.",
    )


class OpenAIChatOptions(_HTTPOptions):
    """Target fields of an OpenAI-compatible Chat Completions endpoint."""

    target_type: ClassVar[str | None] = "openai-chat"
    harbor_target_keys: ClassVar[tuple[str, ...]] = ("model",)

    base_url: str | None = Field(default=None, description="API base URL, e.g. http://agent-api:8000/v1.")
    model: str | None = Field(default=None, description="Model name sent to the endpoint. Default: Harbor's -m.")
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding the endpoint's API key (pass it with --ae)."
    )
    system_prompt: str | None = Field(default=None, description="System message prepended to every request.")
    extra_body: dict[str, Any] | None = Field(default=None, description="Extra request body fields.")


class A2AOptions(_HTTPOptions):
    """Target fields of an A2A server."""

    target_type: ClassVar[str | None] = "a2a"

    url: str | None = Field(default=None, description="A2A endpoint URL.")
    api_key_env: str | None = Field(
        default=None,
        description=(
            "Environment variable holding the A2A server's API key, sent as Authorization: Bearer (pass it with "
            "--ae). Other schemes go in headers."
        ),
    )


class LangGraphOptions(_HTTPOptions):
    """Target fields of a LangGraph server (LangGraph Platform / ``langgraph dev``)."""

    target_type: ClassVar[str | None] = "langgraph"

    base_url: str | None = Field(default=None, description="LangGraph server URL, e.g. http://agent-api:2024.")
    assistant_id: str | None = Field(default=None, description="Assistant (graph) id.")
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding the LangGraph API key (pass it with --ae)."
    )
    input_key: str | None = Field(default=None, description='State key the messages go into (default "messages").')


class SimulhausenAgent(BaseInstalledAgent):
    """Plays the task's scenario against the target from the kwargs, the env or the scenario file.

    ``harbor run -a simulhausen.harbor:SimulhausenAgent --ak 'target={"type": "agentos", ...}'``. The
    protocol-specific subclasses below take flat kwargs instead and give results a protocol-specific agent name.
    """

    options_model = SimulhausenOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        options = self.options
        if not isinstance(options, SimulhausenOptions):
            raise TypeError(f"{type(self).__name__}.options_model must subclass SimulhausenOptions")
        self.simulhausen_options = options
        self.target = options.build_target(self._target_defaults())
        if self.target is not None:
            self._validate_target(self.target)
            # Again with the defaults, which the kwargs check could not see.
            check_target(self.target, flat=options.target_fields())
            self._warn_about_literal_secrets(self.target)
        self._simulhausen_spec = options.simulhausen_spec or (
            f"simulhausen=={self._version}"
            if self._version
            else default_simulhausen_spec(option="--ak simulhausen_spec=...")
        )
        self._detected_version: str | None = None

    @staticmethod
    @override
    def name() -> str:
        return "simulhausen"

    @override
    def to_agent_info(self) -> AgentInfo:
        """Harbor's agent info, with ``-m`` kept whole as the model name.

        ``-m`` labels the agent under test (``acme/support@v3``), not a provider's model. Harbor would split it at
        the first ``/`` into a provider and a name and then group results by the name alone.
        """
        info = super().to_agent_info()
        if not self.model_name:
            return info
        return info.model_copy(update={"model_info": ModelInfo(name=self.model_name)})

    @override
    def version(self) -> str | None:
        """simulhausen version in the container: detected during setup, else the one that would be installed."""
        if self._detected_version:
            return self._detected_version
        if self._version:
            return self._version
        pinned = _PINNED_SPEC.match(self.simulhausen_spec)
        return pinned.group(1) if pinned else None

    @property
    def simulhausen_spec(self) -> str:
        """Requirement installed when the container lacks the CLI: by default the simulhausen version here."""
        return self._simulhausen_spec

    @property
    def environment_out_dir(self) -> PurePosixPath:
        """``/logs/agent/simulhausen``: play's ``--out``, where the verifier looks for the transcript."""
        return PurePosixPath(self.environment_logs_dir) / AGENT_LOGS_SUBDIR

    def _target_defaults(self) -> dict[str, Any]:
        """Values a subclass derives from Harbor's own arguments; flat kwargs and ``target`` override them."""
        return {}

    def _validate_target(self, target: dict[str, Any]) -> None:
        """Validation that needs more than the kwargs (subclasses: the ``-m`` model name)."""
        declared = target.get("type")
        expected = self.simulhausen_options.target_type
        if expected is not None and declared != expected:
            self.logger.warning(
                "%s expects a %r target but got type %r; results are still reported as %r",
                type(self).__name__,
                expected,
                declared,
                self.name(),
            )

    def _warn_about_literal_secrets(self, target: Mapping[str, Any]) -> None:
        for key, value in target.items():
            if value is not None and not key.endswith("_env") and _SENSITIVE_NAME.search(key):
                self.logger.warning(
                    "target key %r holds a literal value; it ends up in the job's config.json and in %s. "
                    "Prefer %s_env with --ae.",
                    key,
                    TARGET_FILE,
                    key,
                )
        headers = target.get("headers")
        if isinstance(headers, Mapping):
            for header in headers:
                if _SENSITIVE_NAME.search(str(header)):
                    self.logger.warning(
                        "header %r looks like a credential; it ends up in the job's config.json and in %s. "
                        "Use api_key_env where the protocol supports it.",
                        header,
                        TARGET_FILE,
                    )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        """Install simulhausen only when the image lacks it: exported task images ship it preinstalled."""
        found = await environment.exec(command="command -v simulhausen")
        if found.return_code == 0:
            self.logger.debug("simulhausen CLI found at %s, skipping install", (found.stdout or "").strip())
            self._warn_about_skipped_pip()
        else:
            await self._install_cli(environment)
        self._detected_version = await self._detect_version(environment)

    def _warn_about_skipped_pip(self) -> None:
        options = self.simulhausen_options
        if "pip" in options.model_fields_set and options.pip:
            flags = " ".join(f"--pip {shlex.quote(requirement)}" for requirement in options.pip)
            self.logger.warning(
                "pip requirements %s were not installed: the image already ships simulhausen, and the agent "
                "installs nothing into such an image. Add them to the task image instead: "
                "simulhausen harbor export ... %s",
                options.pip,
                flags,
            )

    async def _install_cli(self, environment: BaseEnvironment) -> None:
        has_uv = (await environment.exec(command="command -v uv", user="root")).return_code == 0
        packages: list[str] = []
        if not has_uv:
            # The installer needs curl and CA certificates; slim images have neither.
            packages += ["curl", "ca_certificates"]
        if any("git+" in requirement for requirement in [self.simulhausen_spec, *self.simulhausen_options.pip]):
            # uv needs the git executable for a git+ requirement, and slim images have none.
            packages.append("git")
        await self.ensure_system_dependencies(environment, tuple(packages))
        log = PurePosixPath(self.environment_logs_dir) / "setup" / INSTALL_LOG_FILE
        script = self.install_script(install_uv=not has_uv)
        await self.exec_as_root(
            environment,
            command=f"mkdir -p {shlex.quote(str(log.parent))} && {{ {script}; }} 2>&1 | tee {shlex.quote(str(log))}",
        )

    def install_script(self, *, install_uv: bool) -> str:
        """Shell commands that install the CLI as root, usable by any user afterwards."""
        steps: list[str] = []
        uv = "uv"
        if install_uv:
            uv = f"{_BIN_DIR}/uv"
            steps.append(
                f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh"
                f" | env UV_INSTALL_DIR={_BIN_DIR} UV_NO_MODIFY_PATH=1 sh"
            )
        tool_install = [uv, "tool", "install", "--python", ">=3.12", self.simulhausen_spec]
        for requirement in self.simulhausen_options.pip:
            tool_install += ["--with", requirement]
        steps.append(
            f"env UV_TOOL_DIR={_TOOL_DIR} UV_TOOL_BIN_DIR={_BIN_DIR} UV_PYTHON_INSTALL_DIR={_PYTHON_DIR} "
            + shlex.join(tool_install)
        )
        steps.append(f"{_BIN_DIR}/simulhausen --version")
        return " && ".join(steps)

    async def _detect_version(self, environment: BaseEnvironment) -> str | None:
        try:
            result = await environment.exec(command="simulhausen --version")
        except Exception:  # the version is informational, never worth failing setup over
            self.logger.debug("simulhausen --version failed", exc_info=True)
            return None
        if result.return_code != 0:
            return None
        return parse_version(result.stdout or "")

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Play the dialogue. ``instruction`` is the scenario description, which play reads from the scenario file.

        Play exits 0 once the dialogue is recorded, also when the agent under test errored: the judge grades that
        FAIL, and Harbor's ``--max-retries`` does not re-roll it. A non-zero exit means the harness failed (2, the
        transcript still written), the configuration was wrong (1) or ``play_timeout_sec`` ran out, and raises
        Harbor's ``NonZeroAgentExitCodeError``. Harbor records it and still runs the verifier, which grades the
        scenario WARN with the reason. ``context`` is left empty on purpose: Harbor then calls
        ``populate_context_post_run`` once the logs are on the host.
        """
        if self.target is not None:
            await self._upload_target(environment)
        await self.exec_as_agent(environment, command=self.play_command(), env={"PYTHONUNBUFFERED": "1"})

    async def _upload_target(self, environment: BaseEnvironment) -> None:
        # A file rather than --target-json on the command line: no quoting to get wrong, and the command line
        # (which Harbor logs on failure) stays short. There are no secrets in it, so it is world-readable.
        await self.exec_as_root(environment, command=f"mkdir -p {shlex.quote(str(PurePosixPath(TARGET_FILE).parent))}")
        with tempfile.TemporaryDirectory(prefix="simulhausen-target-") as tmp:
            source = Path(tmp) / "target.json"
            write_json(source, self.target)
            await environment.upload_file(source, TARGET_FILE)
        await self.exec_as_root(environment, command=f"chmod 0644 {shlex.quote(TARGET_FILE)}")

    def play_command(self) -> str:
        """The shell command run as the agent user; its output is kept in ``play.log`` next to the transcript."""
        options = self.simulhausen_options
        out_dir = self.environment_out_dir
        args = ["simulhausen", "harbor", "play", "--scenario", options.scenario_path]
        if self.target is not None:
            args += ["--target-file", TARGET_FILE]
        args += ["--out", str(out_dir)]
        if options.simulator_model:
            args += ["--model", options.simulator_model]
        play = shlex.join(args)
        if options.play_timeout_sec is not None:
            # Inside the container rather than exec(timeout_sec=...): Harbor's exec timeout raises a plain
            # RuntimeError, which would skip the verifier. timeout(1) exits 124 instead, a non-zero exit that is
            # handled like a harness failure. -s/-k work with both GNU coreutils and BusyBox.
            seconds = math.ceil(options.play_timeout_sec)
            play = (
                "if command -v timeout >/dev/null 2>&1; then "
                f"timeout -s INT -k {_PLAY_KILL_AFTER_SEC} {seconds} {play}; "
                f"else echo {shlex.quote(_NO_TIMEOUT_WARNING)} >&2; {play}; fi"
            )
        log = shlex.quote(str(out_dir / PLAY_LOG_FILE))
        # Harbor runs commands with `set -o pipefail`, so play's exit code survives the tee.
        return f"mkdir -p {shlex.quote(str(out_dir))} && {{ {play}; }} 2>&1 | tee {log}"

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Summarise ``transcript.json`` into ``context.metadata`` (lands in the trial's ``result.json``).

        Also mirrors play's ATIF ``trajectory.json`` to the agent log root, where ``harbor view`` and Harbor's
        token-usage backfill look for it. Both steps tolerate a missing or broken transcript: play may have died
        before writing one, and the verifier reports that case on its own.
        """
        log_dir = self.logs_dir / AGENT_LOGS_SUBDIR
        self._mirror_trajectory(log_dir / TRAJECTORY_FILE)
        transcript_path = log_dir / TRANSCRIPT_FILE
        try:
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.logger.warning("no transcript at %s: play did not finish", transcript_path)
            metadata: dict[str, Any] = {"transcript_error": "missing"}
        except (OSError, ValueError) as exc:
            self.logger.warning("unreadable transcript at %s: %s", transcript_path, exc)
            metadata = {"transcript_error": f"unreadable: {exc}"}
        else:
            if isinstance(data, dict):
                metadata = summarize_transcript(data)
                transcript_format = data.get("format")
                if transcript_format != TRANSCRIPT_FORMAT:
                    # The judge refuses it: simulhausen.harbor.transcript.read_transcript.
                    self.logger.warning("unexpected transcript format %r in %s", transcript_format, transcript_path)
            else:
                self.logger.warning("transcript at %s is not a JSON object", transcript_path)
                metadata = {"transcript_error": "not a JSON object"}
        context.metadata = {**(context.metadata or {}), **metadata}

    def _mirror_trajectory(self, source: Path) -> None:
        destination = self.logs_dir / TRAJECTORY_FILE
        if not source.is_file() or destination.exists():
            return
        try:
            shutil.copyfile(source, destination)
        except OSError as exc:
            self.logger.warning("could not copy %s to %s: %s", source, destination, exc)


class AgentOSAgent(SimulhausenAgent):
    """An agno AgentOS app: ``--ak base_url=http://agent-api:7777 --ak agent_id=support``."""

    options_model = AgentOSOptions

    @staticmethod
    @override
    def name() -> str:
        return "simulhausen-agentos"


class OpenAIChatAgent(SimulhausenAgent):
    """An OpenAI-compatible Chat Completions endpoint: ``--ak base_url=http://agent-api:8000/v1``.

    The endpoint's ``model`` defaults to Harbor's ``-m`` value, used verbatim.
    """

    options_model = OpenAIChatOptions

    @staticmethod
    @override
    def name() -> str:
        return "simulhausen-openai-chat"

    @override
    def _target_defaults(self) -> dict[str, Any]:
        return {"model": self.model_name}

    @override
    def _validate_target(self, target: dict[str, Any]) -> None:
        super()._validate_target(target)
        if not target.get("model"):
            raise ValueError("OpenAIChatAgent needs a model: pass -m <model> or --ak model=<model>")


class A2AAgent(SimulhausenAgent):
    """An A2A server: ``--ak url=http://agent-api:9999``."""

    options_model = A2AOptions

    @staticmethod
    @override
    def name() -> str:
        return "simulhausen-a2a"


class LangGraphAgent(SimulhausenAgent):
    """A LangGraph server: ``--ak base_url=http://agent-api:2024 --ak assistant_id=agent``."""

    options_model = LangGraphOptions

    @staticmethod
    @override
    def name() -> str:
        return "simulhausen-langgraph"


def check_target(target: Mapping[str, Any], *, flat: Collection[str] = (), filled_later: Collection[str] = ()) -> None:
    """Raise ``ValueError`` for a target that play would refuse, before any container starts.

    A built-in target needs a known ``type`` and keys that bind to the parameters of its adapter factory
    (:data:`simulhausen.harbor.targets.FACTORIES`), each ``<name>_env`` standing in for ``<name>``; an AgentOS
    target also needs exactly one of ``agent_id``, ``team_id`` and ``workflow_id``. AgentOS passes keys it does not
    know on as form fields, so there a misspelt optional key goes through. ``flat`` names the keys a kwarg sets,
    for the hints; ``filled_later`` the keys the agent may still add (``harbor_target_keys``). A ``python`` target
    is not checked: its factory may exist only in the task image.
    """
    kind = target.get("type")
    if kind == "python":
        return
    if not isinstance(kind, str) or kind not in FACTORIES:
        raise ValueError(f"unknown target type {kind!r}; known types: {', '.join([*FACTORIES, 'python'])}")
    arguments: dict[str, Any] = {}
    for key, value in target.items():
        # As simulhausen.harbor.targets resolves them in the container: `<name>_env = "VAR"` becomes `<name> = $VAR`.
        name = key.removesuffix(_ENV_SUFFIX) if key.endswith(_ENV_SUFFIX) and key != _ENV_SUFFIX else key
        if name != key and name in target:
            raise ValueError(f"target sets both {name} and {key}; keep one")
        if key != "type":
            arguments[name] = value
    factory = cast("Callable[..., object]", import_object(FACTORIES[kind], what=f"target type {kind!r}"))
    signature = inspect.signature(factory)
    parameters = signature.parameters
    arguments = {**{key: None for key in filled_later if key in parameters}, **arguments}
    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is parameter.empty
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and name not in arguments
    ]
    if missing:
        hints = ", ".join(f"--ak {name}=..." for name in missing if name in flat)
        raise ValueError(f"missing {', '.join(missing)}: {f'pass {hints} or ' if hints else ''}put it into target")
    if kind == "agentos":
        chosen = [key for key in _AGENTOS_IDS if arguments.get(key) is not None]
        if not chosen:
            hint = "pass --ak agent_id=... (or team_id, workflow_id) or " if "agent_id" in flat else ""
            raise ValueError(f"an agentos target needs one of {', '.join(_AGENTOS_IDS)}: {hint}put it into target")
        if len(chosen) > 1:
            raise ValueError(f"an agentos target takes one of {', '.join(_AGENTOS_IDS)}, got {' and '.join(chosen)}")
    try:
        signature.bind(**arguments)
    except TypeError as exc:
        raise ValueError(f"target type {kind!r} ({FACTORIES[kind]}): {exc}") from None


def parse_version(output: str) -> str | None:
    """The version in ``simulhausen --version`` output (``simulhausen 0.3.0`` or a bare ``0.3.0``)."""
    matches = _VERSION_TOKEN.findall(output)
    return matches[-1] if matches else None


def summarize_transcript(data: Mapping[str, Any]) -> dict[str, Any]:
    """The run facts of a ``simulhausen-transcript/1`` document, tolerant of missing or odd fields.

    ``tool_calls`` counts clean executions only (no error, not paused), like ``ScenarioState.has_tool_call``;
    the rest are listed under ``failed_tool_calls``.
    """
    scenario = _mapping(data.get("scenario"))
    target = _mapping(data.get("target"))
    messages = data.get("messages")
    clean: list[str] = []
    failed: list[str] = []
    for record in _tool_records(data.get("agent_turns")):
        name = record.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        (clean if _clean_record(record) else failed).append(name)
    return {
        "scenario_id": scenario.get("id"),
        "scenario_name": scenario.get("name"),
        "target_type": target.get("type"),
        "turns": data.get("turns"),
        "messages": len(messages) if isinstance(messages, list) else 0,
        "tool_calls": clean,
        "failed_tool_calls": failed,
        "stop_reason": data.get("stop_reason"),
        "harness_error": bool(data.get("harness_error")),
        "error": data.get("error"),
        "agent_time": data.get("agent_time"),
        "total_time": data.get("total_time"),
        "simulhausen_version": data.get("simulhausen_version"),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _tool_records(agent_turns: Any) -> list[Mapping[str, Any]]:
    if not isinstance(agent_turns, list):
        return []
    tools: list[Mapping[str, Any]] = []
    for turn in agent_turns:
        turn_tools = turn.get("tools") if isinstance(turn, Mapping) else None
        if isinstance(turn_tools, list):
            tools.extend(tool for tool in turn_tools if isinstance(tool, Mapping))
    return tools


def _clean_record(record: Mapping[str, Any]) -> bool:
    """``has_tool_call``'s test on a ``ToolExecution.to_dict()`` record; one agno cannot read counts as failed."""
    try:
        tool = ToolExecution.from_dict(dict(record))
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
    return _is_clean(tool)
