"""The Harbor agents (SimulhausenAgent and the protocol subclasses) against a recording environment.

No container, no network: install, play and the context summary run against ``FakeEnvironment``.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.job.result import JobStats
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import ModelInfo, TrialResult
from harbor_agent_support import FakeEnvironment, write_json
from harbor_support import installed_from

import simulhausen
from simulhausen.harbor import export as export_module
from simulhausen.harbor.agents import (
    TARGET_FILE,
    A2AAgent,
    AgentOSAgent,
    LangGraphAgent,
    OpenAIChatAgent,
    SimulhausenAgent,
    parse_version,
    summarize_transcript,
)
from simulhausen.harbor.export import UV_VERSION, default_simulhausen_spec
from simulhausen.harbor.paths import AGENT_LOGS_DIR

MakeAgent = Callable[..., SimulhausenAgent]

PIPEFAIL = "set -o pipefail; "
INSTALLED_AGENT_DIR = "[ -d /installed-agent ] || mkdir -p /installed-agent"
MISSING = ExecResult(stdout="", stderr="", return_code=1)


def found(stdout: str) -> ExecResult:
    return ExecResult(stdout=stdout, stderr="", return_code=0)


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
def make_agent(logs_dir: Path) -> MakeAgent:
    """Build an agent the way Harbor's AgentFactory does: keyword arguments only."""

    def build(cls: type[SimulhausenAgent], model_name: str | None = None, **kwargs: Any) -> SimulhausenAgent:
        return cls(logs_dir=logs_dir, model_name=model_name, **kwargs)

    return build


# --- identity ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "kwargs", "name"),
    [
        (SimulhausenAgent, {}, "simulhausen"),
        (AgentOSAgent, {"base_url": "http://agent-api:7777", "agent_id": "support"}, "simulhausen-agentos"),
        (OpenAIChatAgent, {"base_url": "http://agent-api:8000/v1", "model": "m"}, "simulhausen-openai-chat"),
        (A2AAgent, {"url": "http://agent-api:9999"}, "simulhausen-a2a"),
        (LangGraphAgent, {"base_url": "http://agent-api:2024", "assistant_id": "agent"}, "simulhausen-langgraph"),
    ],
)
def test_names_are_fixed_per_class(make_agent: MakeAgent, cls: type[SimulhausenAgent], kwargs: dict, name: str):
    agent = make_agent(cls, model_name="support-agent@v1", **kwargs)
    assert cls.name() == name
    info = agent.to_agent_info()
    assert info.name == name
    # -m carries the identity of the agent under test into Harbor's agent__model__dataset key.
    assert info.model_info is not None
    assert info.model_info.name == "support-agent@v1"


def test_model_label_keeps_its_slash(make_agent: MakeAgent):
    # Harbor's own to_agent_info would record provider "acme" and name "support-agent@v3", and group by the name.
    agent = make_agent(AgentOSAgent, model_name="acme/support-agent@v3", base_url="http://x", agent_id="support")
    info = agent.to_agent_info()
    assert info.model_info == ModelInfo(name="acme/support-agent@v3", provider=None)
    # What Harbor groups the job's statistics (mean, pass@k, the metric) by.
    stats = JobStats.from_trial_results([TrialResult.model_construct(trial_name="t", agent_info=info, source="tasks")])
    assert list(stats.evals) == ["simulhausen-agentos__acme/support-agent@v3__tasks"]


def test_no_model_label_means_no_model_info(make_agent: MakeAgent):
    assert make_agent(SimulhausenAgent).to_agent_info().model_info is None


def test_version_defaults_to_the_simulhausen_installed_here(make_agent: MakeAgent):
    # The same requirement `simulhausen harbor export` bakes into task images.
    agent = make_agent(SimulhausenAgent)
    assert agent.simulhausen_spec == default_simulhausen_spec() == f"simulhausen=={simulhausen.__version__}"
    assert agent.version() == simulhausen.__version__


def test_unreleased_version_installs_the_latest_release(
    make_agent: MakeAgent, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(export_module, "__version__", "0+unknown")
    with caplog.at_level(logging.WARNING):
        agent = make_agent(SimulhausenAgent)
    assert agent.simulhausen_spec == "simulhausen"
    assert agent.version() is None
    assert "Pass --ak simulhausen_spec=... to pin one" in caplog.text


def test_default_spec_follows_an_archive_install_of_the_host(make_agent: MakeAgent, monkeypatch: pytest.MonkeyPatch):
    # An archive install reports the last release as its version; pinning that would install a release without
    # the CLI, so the container gets the same archive.
    archive = "https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz"
    installed_from(monkeypatch, {"url": archive, "archive_info": {}})
    agent = make_agent(SimulhausenAgent)
    assert agent.simulhausen_spec == f"simulhausen @ {archive}"
    assert agent.version() is None
    assert f"'simulhausen @ {archive}' --with openai" in agent.install_script(install_uv=False)


def test_local_install_of_the_host_names_the_agent_option(
    make_agent: MakeAgent, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    installed_from(monkeypatch, {"url": "file:///home/me/simulhausen", "dir_info": {"editable": True}})
    with caplog.at_level(logging.WARNING):
        agent = make_agent(SimulhausenAgent)
    assert agent.simulhausen_spec == f"simulhausen=={simulhausen.__version__}"
    assert "installed from file:///home/me/simulhausen" in caplog.text
    assert "Pass --ak simulhausen_spec=... to install the same code" in caplog.text


def test_version_kwarg_selects_the_release_to_install(make_agent: MakeAgent):
    agent = make_agent(SimulhausenAgent, version="0.3.1")
    assert agent.version() == "0.3.1"
    assert agent.simulhausen_spec == "simulhausen==0.3.1"


def test_unpinned_spec_has_no_version_until_detected(make_agent: MakeAgent):
    agent = make_agent(
        SimulhausenAgent, simulhausen_spec="simulhausen @ git+https://github.com/ormeilu/simulhausen@main"
    )
    assert agent.version() is None


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("simulhausen 0.3.0\n", "0.3.0"),
        ("0.3.1.dev4+g1a2b3c\n", "0.3.1.dev4+g1a2b3c"),
        ("simulhausen, version 1.0.0rc1", "1.0.0rc1"),
        ("no version here", None),
    ],
)
def test_parse_version(output: str, expected: str | None):
    assert parse_version(output) == expected


# --- install ----------------------------------------------------------------------------------------------------


async def test_setup_skips_install_when_the_cli_is_present(make_agent: MakeAgent, environment: FakeEnvironment):
    environment.answer("command -v simulhausen", found("/usr/local/bin/simulhausen\n"))
    environment.answer("simulhausen --version", found("simulhausen 0.3.2\n"))
    agent = make_agent(AgentOSAgent, base_url="http://agent-api:7777", agent_id="support")

    with environment.with_default_user("agent"):
        await agent.setup(environment)

    assert environment.commands == [INSTALLED_AGENT_DIR, "command -v simulhausen", "simulhausen --version"]
    assert environment.calls[1].user == "agent"
    # The version the container actually has wins over the pinned default.
    assert agent.version() == "0.3.2"
    assert agent.to_agent_info().version == "0.3.2"


@pytest.mark.parametrize(
    ("kwargs", "warned"),
    [({"pip": ["anthropic", "google-genai>=1"]}, True), ({}, False), ({"pip": []}, False)],
    ids=["explicit pip", "default pip", "explicitly none"],
)
async def test_explicit_pip_is_not_dropped_silently_when_the_cli_is_present(
    make_agent: MakeAgent,
    environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
    kwargs: dict[str, Any],
    warned: bool,
):
    environment.answer("command -v simulhausen", found("/usr/local/bin/simulhausen\n"))
    agent = make_agent(SimulhausenAgent, **kwargs)

    with caplog.at_level(logging.WARNING):
        await agent.setup(environment)

    assert not any("uv tool install" in command for command in environment.commands)
    if warned:
        assert "pip requirements ['anthropic', 'google-genai>=1'] were not installed" in caplog.text
        assert "the image already ships simulhausen" in caplog.text
        assert "simulhausen harbor export ... --pip anthropic --pip 'google-genai>=1'" in caplog.text
    else:
        assert caplog.text == ""


async def test_setup_installs_with_existing_uv(make_agent: MakeAgent, environment: FakeEnvironment):
    environment.answer("command -v simulhausen", MISSING)
    environment.answer("command -v uv", found("/usr/bin/uv\n"))
    agent = make_agent(SimulhausenAgent)

    await agent.setup(environment)

    install = environment.calls[3]
    assert environment.commands[:3] == [INSTALLED_AGENT_DIR, "command -v simulhausen", "command -v uv"]
    assert environment.calls[2].user == "root"
    assert install.user == "root"
    assert install.command == (
        PIPEFAIL + "mkdir -p /logs/agent/setup && { "
        "env UV_TOOL_DIR=/opt/simulhausen/tools UV_TOOL_BIN_DIR=/usr/local/bin "
        "UV_PYTHON_INSTALL_DIR=/opt/simulhausen/python uv tool install --python '>=3.12' "
        f"simulhausen=={simulhausen.__version__} --with openai && /usr/local/bin/simulhausen --version"
        "; } 2>&1 | tee /logs/agent/setup/simulhausen-install.log"
    )
    assert environment.commands[4:] == ["simulhausen --version"]


async def test_setup_fetches_uv_when_missing(make_agent: MakeAgent, environment: FakeEnvironment):
    environment.answer("command -v simulhausen", MISSING)
    environment.answer("command -v uv", MISSING)
    environment.answer("command -v curl >/dev/null 2>&1", MISSING)
    environment.answer("command -v git >/dev/null 2>&1", MISSING)
    environment.answer("apt-get", found("apt-get"), contains=True)
    agent = make_agent(
        SimulhausenAgent,
        simulhausen_spec="simulhausen @ git+https://github.com/ormeilu/simulhausen@main",
        pip=["openai", "anthropic>=0.40"],
    )

    await agent.setup(environment)

    packages = [c for c in environment.calls if "apt-get install" in c.command]
    assert len(packages) == 1
    assert "curl" in packages[0].command
    assert "ca-certificates" in packages[0].command
    # A git+ requirement needs the git executable, which slim images lack.
    assert " git" in packages[0].command
    install = next(c for c in environment.calls if "uv tool install" in c.command)
    assert install.user == "root"
    assert (
        f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh"
        " | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh" in install.command
    )
    assert (
        "/usr/local/bin/uv tool install --python '>=3.12' "
        "'simulhausen @ git+https://github.com/ormeilu/simulhausen@main' --with openai --with 'anthropic>=0.40'"
    ) in install.command


async def test_setup_installs_git_for_a_git_requirement(make_agent: MakeAgent, environment: FakeEnvironment):
    environment.answer("command -v simulhausen", MISSING)
    environment.answer("command -v uv", found("/usr/bin/uv\n"))
    environment.answer("command -v git >/dev/null 2>&1", MISSING)
    environment.answer("apt-get", found("apt-get"), contains=True)
    agent = make_agent(SimulhausenAgent, pip=["openai", "tools @ git+https://example.com/tools.git"])

    await agent.setup(environment)

    packages = [c for c in environment.calls if "apt-get install" in c.command]
    assert len(packages) == 1
    assert " git" in packages[0].command
    assert "curl" not in packages[0].command


async def test_install_failure_fails_setup(make_agent: MakeAgent, environment: FakeEnvironment):
    environment.answer("command -v simulhausen", MISSING)
    environment.answer(
        "uv tool install", ExecResult(stdout="", stderr="No solution found", return_code=1), contains=True
    )
    agent = make_agent(SimulhausenAgent)

    with pytest.raises(NonZeroAgentExitCodeError, match="No solution found"):
        await agent.setup(environment)


@pytest.mark.parametrize(
    "version_result",
    [MISSING, RuntimeError("container went away")],
    ids=["non-zero exit", "exec raises"],
)
async def test_version_detection_is_best_effort(
    make_agent: MakeAgent, environment: FakeEnvironment, version_result: ExecResult | Exception
):
    environment.answer("command -v simulhausen", found("/usr/local/bin/simulhausen"))
    environment.answer("simulhausen --version", version_result)
    agent = make_agent(SimulhausenAgent)

    await agent.setup(environment)

    assert agent.version() == simulhausen.__version__


# --- run --------------------------------------------------------------------------------------------------------


async def test_run_uploads_the_target_and_plays(make_agent: MakeAgent, environment: FakeEnvironment):
    agent = make_agent(
        AgentOSAgent,
        model_name="support-agent@v1",
        base_url="http://agent-api:7777",
        agent_id="support",
        api_key_env="AGENTOS_API_KEY",
        simulator_model="openai:gpt-4.1-mini",
        extra_env={"OPENAI_API_KEY": "sk-test", "AGENTOS_API_KEY": "os-key"},
    )

    # What Trial does around run(): the task's agent user and the --ae variables on every exec.
    with environment.with_default_user("agent"), environment.scoped_exec_env(agent.extra_env):
        await agent.run("The user was charged twice.", environment, AgentContext())

    mkdir, chmod, play = environment.calls
    assert mkdir.command == PIPEFAIL + "mkdir -p /installed-agent/simulhausen"
    assert mkdir.user == chmod.user == "root"
    assert chmod.command == PIPEFAIL + f"chmod 0644 {TARGET_FILE}"
    [upload] = environment.uploads
    assert upload.target_path == TARGET_FILE
    assert json.loads(upload.content) == {
        "type": "agentos",
        "base_url": "http://agent-api:7777",
        "agent_id": "support",
        "api_key_env": "AGENTOS_API_KEY",
    }
    assert "os-key" not in upload.content
    assert play.user == "agent"
    assert play.command == (
        PIPEFAIL
        + "mkdir -p /logs/agent/simulhausen && { simulhausen harbor play --scenario /simulhausen/scenario.json "
        f"--target-file {TARGET_FILE} --out /logs/agent/simulhausen --model openai:gpt-4.1-mini; }} "
        "2>&1 | tee /logs/agent/simulhausen/play.log"
    )
    assert play.env == {"PYTHONUNBUFFERED": "1", "OPENAI_API_KEY": "sk-test", "AGENTOS_API_KEY": "os-key"}


async def test_run_without_target_leaves_it_to_the_scenario(make_agent: MakeAgent, environment: FakeEnvironment):
    agent = make_agent(SimulhausenAgent, scenario_path="/data/my scenario.json")

    await agent.run("instruction", environment, AgentContext())

    assert environment.uploads == []
    assert environment.commands == [
        PIPEFAIL + "mkdir -p /logs/agent/simulhausen && { simulhausen harbor play --scenario '/data/my scenario.json' "
        "--out /logs/agent/simulhausen; } 2>&1 | tee /logs/agent/simulhausen/play.log"
    ]


async def test_play_timeout_runs_inside_the_container(make_agent: MakeAgent, environment: FakeEnvironment):
    agent = make_agent(SimulhausenAgent, play_timeout_sec=90.5)

    await agent.run("instruction", environment, AgentContext())

    [play] = environment.calls
    assert play.timeout_sec is None
    assert (
        "if command -v timeout >/dev/null 2>&1; then timeout -s INT -k 30 91 simulhausen harbor play "
        "--scenario /simulhausen/scenario.json --out /logs/agent/simulhausen; else echo "
        "'simulhausen: timeout(1) not found in the container; play_timeout_sec is ignored' >&2; "
        "simulhausen harbor play --scenario /simulhausen/scenario.json --out /logs/agent/simulhausen; fi"
    ) in play.command


async def test_failed_play_raises_the_error_that_still_runs_the_verifier(
    make_agent: MakeAgent, environment: FakeEnvironment
):
    # Play exits non-zero only when the harness failed (an error of the agent under test exits 0 and grades FAIL).
    # SingleStepTrial catches NonZeroAgentExitCodeError (and AgentTimeoutError), records it and verifies anyway;
    # any other exception from run() would skip the verifier and lose the WARN report.
    environment.answer(
        "simulhausen harbor play",
        ExecResult(stdout="harness error: agent-api unreachable", stderr="", return_code=2),
        contains=True,
    )
    agent = make_agent(AgentOSAgent, base_url="http://agent-api:7777", agent_id="support")

    with pytest.raises(NonZeroAgentExitCodeError, match="exit 2"):
        await agent.run("instruction", environment, AgentContext())


def test_out_dir_follows_the_environment_log_dir(make_agent: MakeAgent):
    # Where the exported verifier's judge reads the transcript by default.
    agent = make_agent(SimulhausenAgent)
    assert agent.environment_out_dir.as_posix() == AGENT_LOGS_DIR


# --- target -----------------------------------------------------------------------------------------------------


def test_flat_kwargs_build_the_target(make_agent: MakeAgent):
    agent = make_agent(
        AgentOSAgent,
        base_url="http://agent-api:7777",
        team_id="desk",
        session_state={"tier": "gold"},
        history="transcript",
        headers={"X-Tenant": "acme"},
        timeout=30,
        retries=0,
    )
    assert agent.target == {
        "type": "agentos",
        "base_url": "http://agent-api:7777",
        "team_id": "desk",
        "session_state": {"tier": "gold"},
        "history": "transcript",
        "headers": {"X-Tenant": "acme"},
        "timeout": 30.0,
        "retries": 0,
    }


def test_explicit_target_wins_key_by_key(make_agent: MakeAgent):
    agent = make_agent(
        AgentOSAgent,
        base_url="http://agent-api:7777",
        agent_id="support",
        history="session",
        target={"agent_id": "billing", "history": "transcript", "metadata": {"run": "nightly"}},
    )
    assert agent.target == {
        "type": "agentos",
        "base_url": "http://agent-api:7777",
        "agent_id": "billing",
        "history": "transcript",
        "metadata": {"run": "nightly"},  # no flat kwarg: an extra AgentOS form field
    }


def test_target_may_be_a_json_string(make_agent: MakeAgent):
    agent = make_agent(SimulhausenAgent, target='{"type": "openai-responses", "base_url": "http://x/v1", "model": "m"}')
    assert agent.target == {"type": "openai-responses", "base_url": "http://x/v1", "model": "m"}


def test_explicit_none_target_means_no_target(make_agent: MakeAgent):
    # `--ak target=None` parses to None, like an omitted kwarg.
    assert make_agent(SimulhausenAgent, target=None).target is None


def test_target_type_mismatch_is_allowed_but_logged(make_agent: MakeAgent, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        agent = make_agent(AgentOSAgent, target={"type": "a2a", "url": "http://agent-api:9999"})
    assert agent.target is not None
    assert agent.target["type"] == "a2a"
    assert "expects a 'agentos' target" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": "{not json"}, "target is not valid JSON"),
        ({"target": "[1, 2]"}, "target must be a JSON object"),
        ({"target": {"base_url": "http://x"}}, 'target needs a "type"'),
        ({"target": {"type": "agentos", "x": float("nan")}}, "target must be plain JSON"),
        ({"base_url": "http://x"}, "Unknown option 'base_url'"),
        ({"play_timeout_sec": 0}, "play_timeout_sec"),
        ({"play_timeout_sec": float("inf")}, "play_timeout_sec"),
        ({"play_timeout_sec": float("nan")}, "play_timeout_sec"),
    ],
)
def test_invalid_kwargs_are_rejected(make_agent: MakeAgent, kwargs: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        make_agent(SimulhausenAgent, **kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs", "missing"),
    [
        (AgentOSAgent, {"agent_id": "support"}, "missing base_url: pass --ak base_url=... or put it into target"),
        (A2AAgent, {}, "missing url"),
        (LangGraphAgent, {"base_url": "http://agent-api:2024"}, "missing assistant_id"),
        (LangGraphAgent, {}, "missing base_url, assistant_id"),
        (SimulhausenAgent, {"target": {"type": "openai-chat", "base_url": "http://x/v1"}}, "missing model: put it"),
    ],
)
def test_required_target_keys(make_agent: MakeAgent, cls: type[SimulhausenAgent], kwargs: dict, missing: str):
    with pytest.raises(ValueError, match=missing):
        make_agent(cls, **kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs", "message"),
    [
        (
            AgentOSAgent,
            {"base_url": "http://x"},
            r"agentos target needs one of agent_id, team_id, workflow_id: pass --ak agent_id=\.\.\. \(or team_id",
        ),
        (
            SimulhausenAgent,
            {"target": {"type": "agentos", "base_url": "http://x"}},
            "needs one of agent_id, team_id, workflow_id: put it into target",
        ),
        (
            AgentOSAgent,
            {"base_url": "http://x", "agent_id": "support", "target": {"workflow_id": "refunds"}},
            "takes one of agent_id, team_id, workflow_id, got agent_id and workflow_id",
        ),
        (AgentOSAgent, {"base_url": "http://x", "team_id": "a", "workflow_id": "b"}, "got team_id and workflow_id"),
        (AgentOSAgent, {"target": {"base_url": "http://x", "agnet_id": "support"}}, "needs one of agent_id"),
        (
            A2AAgent,
            {"url": "http://x", "target": {"protocol_versoin": "1.0"}},
            "target type 'a2a' .*a2a_adapter.*unexpected keyword argument 'protocol_versoin'",
        ),
        (
            LangGraphAgent,
            {"base_url": "http://x", "assistant_id": "agent", "target": {"input_kye_env": "KEY"}},
            "unexpected keyword argument 'input_kye'",
        ),
        (SimulhausenAgent, {"target": {"type": "agentoss", "base_url": "http://x"}}, "unknown target type 'agentoss'"),
        (SimulhausenAgent, {"target": {"type": ["agentos"]}}, r"unknown target type \['agentos'\]"),
        (
            A2AAgent,
            {"url": "http://x", "headers": {"X-Tenant": "acme"}, "target": {"headers_env": "HEADERS"}},
            "sets both headers and headers_env",
        ),
    ],
    ids=[
        "agentos without an id",
        "agentos target without an id",
        "agentos id from target and kwargs",
        "agentos two flat ids",
        "agentos misspelt id",
        "a2a misspelt key",
        "langgraph misspelt env key",
        "unknown type",
        "unhashable type",
        "key and its env",
    ],
)
def test_target_that_cannot_work_is_refused_by_the_options(
    cls: type[SimulhausenAgent], kwargs: dict[str, Any], message: str
):
    # The options model alone, as Harbor's preflight validates it: no agent, no container.
    with pytest.raises(ValueError, match=message):
        cls.parse_options(kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (AgentOSAgent, {"base_url": "http://x", "target": {"agent_id_env": "AGENT_ID"}}),
        (AgentOSAgent, {"workflow_id": "refunds", "target": {"base_url_env": "AGENTOS_URL"}}),
        (SimulhausenAgent, {"target": {"type": "python", "factory": "app.targets:build", "anything": 1}}),
        (OpenAIChatAgent, {"base_url": "http://x/v1"}),  # model comes from -m, which the options do not see
    ],
    ids=["env id", "env base_url", "python target", "model from -m"],
)
def test_target_checks_accept(cls: type[SimulhausenAgent], kwargs: dict[str, Any]):
    cls.parse_options(kwargs)


def test_defaults_are_checked_once_the_agent_has_them(make_agent: MakeAgent):
    # -m becomes the target's model, which an a2a target does not take.
    with pytest.raises(ValueError, match="unexpected keyword argument 'model'"):
        make_agent(OpenAIChatAgent, model_name="m", target={"type": "a2a", "url": "http://x"})


def test_a2a_takes_an_api_key_env(make_agent: MakeAgent, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        agent = make_agent(A2AAgent, url="http://agent-api:9999", api_key_env="A2A_KEY")
    assert agent.target == {"type": "a2a", "url": "http://agent-api:9999", "api_key_env": "A2A_KEY"}
    assert caplog.text == ""


def test_required_keys_may_come_from_target(make_agent: MakeAgent):
    agent = make_agent(A2AAgent, target={"url": "http://agent-api:9999"})
    assert agent.target == {"type": "a2a", "url": "http://agent-api:9999"}


@pytest.mark.parametrize(
    ("pip", "expected"),
    [("anthropic", ["anthropic"]), ("", []), (None, []), (["openai", "google-genai"], ["openai", "google-genai"])],
)
def test_pip_accepts_a_single_requirement(make_agent: MakeAgent, pip: Any, expected: list[str]):
    agent = make_agent(SimulhausenAgent, pip=pip)
    assert agent.simulhausen_options.pip == expected


def test_options_model_must_be_simulhausen_options(logs_dir: Path):
    class Broken(SimulhausenAgent):
        options_model = InstalledAgentOptions

    with pytest.raises(TypeError, match="must subclass SimulhausenOptions"):
        Broken(logs_dir=logs_dir)


def test_native_config_is_refused(make_agent: MakeAgent):
    # Inherited from Harbor's InstalledAgentOptions; the agent declares no native config format.
    with pytest.raises(ValueError, match="does not support config"):
        make_agent(SimulhausenAgent, config={"x": 1})


# --- secrets ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "http://agent-api:7777", "api_key": "sk-live"},
        {"target": {"base_url": "http://agent-api:7777", "api_key": "sk-live"}},
        {"target": '{"base_url": "http://agent-api:7777", "api_key": "sk-live"}'},
    ],
    ids=["flat", "target", "json target"],
)
def test_literal_api_key_is_refused(make_agent: MakeAgent, kwargs: dict[str, Any]):
    with pytest.raises(ValueError, match="api_key_env"):
        make_agent(AgentOSAgent, **kwargs)


def test_credential_looking_values_are_warned_about(make_agent: MakeAgent, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        make_agent(
            AgentOSAgent,
            base_url="http://agent-api:7777",
            agent_id="support",
            headers={"Authorization": "Bearer abc", "X-Tenant": "acme"},
            # AgentOS form fields
            target={"auth_token": "abc", "token_env": "SESSION_TOKEN"},
        )
    assert "header 'Authorization' looks like a credential" in caplog.text
    assert "target key 'auth_token' holds a literal value" in caplog.text
    assert "X-Tenant" not in caplog.text
    assert "target key 'token_env'" not in caplog.text


def test_env_indirections_raise_no_warning(make_agent: MakeAgent, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        make_agent(LangGraphAgent, base_url="http://x", assistant_id="agent", api_key_env="LG_KEY", input_key="msgs")
    assert caplog.text == ""


# --- OpenAIChatAgent model ----------------------------------------------------------------------------------------


def test_openai_chat_model_defaults_to_harbor_model_name(make_agent: MakeAgent):
    agent = make_agent(OpenAIChatAgent, model_name="openai/gpt-4.1-mini", base_url="http://agent-api:8000/v1")
    assert agent.target == {
        "type": "openai-chat",
        "model": "openai/gpt-4.1-mini",
        "base_url": "http://agent-api:8000/v1",
    }


def test_openai_chat_model_kwarg_wins(make_agent: MakeAgent):
    agent = make_agent(OpenAIChatAgent, model_name="support@v1", base_url="http://x/v1", model="support-agent")
    assert agent.target is not None
    assert agent.target["model"] == "support-agent"


def test_openai_chat_needs_a_model(make_agent: MakeAgent):
    with pytest.raises(ValueError, match="OpenAIChatAgent needs a model"):
        make_agent(OpenAIChatAgent, base_url="http://x/v1")


# --- populate_context_post_run ----------------------------------------------------------------------------------

TRANSCRIPT = {
    "format": "simulhausen-transcript/1",
    "simulhausen_version": "0.3.0",
    "scenario": {"id": "refund-duplicated-charge", "name": "refund: duplicated charge", "description": "..."},
    "thread_id": "t-1",
    "messages": [
        {"role": "user", "content": "i was charged twice"},
        {"role": "assistant", "content": "Let me check order 1042."},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Refunded 49.00."},
    ],
    "agent_turns": [
        {"message_index": 1, "tools": [{"tool_name": "lookup_order", "tool_call_error": False}], "references": []},
        {
            "message_index": 3,
            "tools": [
                {"tool_name": "issue_refund", "tool_call_error": True},
                {"tool_name": "issue_refund", "requires_confirmation": True},
                {"tool_name": "issue_refund"},
                {"tool_name": "notify", "user_input_schema": "not a list"},  # agno cannot read it
                {"tool_args": {}},
                "not a tool",
            ],
        },
        "not a turn",
    ],
    "checkpoints": [],
    "turns": 2,
    "stop_reason": "max_turns",
    "harness_error": False,
    "error": None,
    "total_time": 12.3,
    "agent_time": 8.1,
    "target": {"type": "agentos", "base_url": "http://agent-api:7777"},
}


def test_context_gets_the_transcript_summary(make_agent: MakeAgent, logs_dir: Path):
    write_json(logs_dir / "simulhausen" / "transcript.json", TRANSCRIPT)
    context = AgentContext()

    make_agent(AgentOSAgent, base_url="http://agent-api:7777", agent_id="support").populate_context_post_run(context)

    assert context.metadata == {
        "scenario_id": "refund-duplicated-charge",
        "scenario_name": "refund: duplicated charge",
        "target_type": "agentos",
        "turns": 2,
        "messages": 4,
        "tool_calls": ["lookup_order", "issue_refund"],
        "failed_tool_calls": ["issue_refund", "issue_refund", "notify"],
        "stop_reason": "max_turns",
        "harness_error": False,
        "error": None,
        "agent_time": 8.1,
        "total_time": 12.3,
        "simulhausen_version": "0.3.0",
    }


def test_context_keeps_existing_metadata(make_agent: MakeAgent, logs_dir: Path):
    write_json(logs_dir / "simulhausen" / "transcript.json", {**TRANSCRIPT, "harness_error": True, "error": "boom"})
    context = AgentContext(metadata={"owner": "kept"})

    make_agent(SimulhausenAgent).populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["owner"] == "kept"
    assert context.metadata["harness_error"] is True
    assert context.metadata["error"] == "boom"


def test_missing_transcript_is_reported_not_raised(
    make_agent: MakeAgent, logs_dir: Path, caplog: pytest.LogCaptureFixture
):
    context = AgentContext()
    with caplog.at_level(logging.WARNING):
        make_agent(SimulhausenAgent).populate_context_post_run(context)
    assert context.metadata == {"transcript_error": "missing"}
    assert "play did not finish" in caplog.text


@pytest.mark.parametrize(
    ("content", "error"),
    [("{truncated", "unreadable: "), ("[1, 2]", "not a JSON object"), (b"\xff\xfe", "unreadable: ")],
    ids=["broken json", "json array", "not utf-8"],
)
def test_corrupt_transcript_is_reported_not_raised(
    make_agent: MakeAgent, logs_dir: Path, content: str | bytes, error: str
):
    path = logs_dir / "simulhausen" / "transcript.json"
    path.parent.mkdir(parents=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    context = AgentContext()

    make_agent(SimulhausenAgent).populate_context_post_run(context)

    assert context.metadata is not None
    assert str(context.metadata["transcript_error"]).startswith(error)


@pytest.mark.parametrize("transcript_format", ["other/9", "simulhausen-transcript/2"])
def test_unknown_transcript_format_is_summarised_with_a_warning(
    make_agent: MakeAgent, logs_dir: Path, caplog: pytest.LogCaptureFixture, transcript_format: str
):
    # A format the judge refuses, newer versions of this one included.
    write_json(logs_dir / "simulhausen" / "transcript.json", {"format": transcript_format, "turns": 1})
    context = AgentContext()
    with caplog.at_level(logging.WARNING):
        make_agent(SimulhausenAgent).populate_context_post_run(context)
    assert context.metadata is not None
    assert context.metadata["turns"] == 1
    assert context.metadata["messages"] == 0
    assert context.metadata["tool_calls"] == []
    assert f"unexpected transcript format {transcript_format!r}" in caplog.text


def test_trajectory_is_mirrored_for_harbor_view(make_agent: MakeAgent, logs_dir: Path):
    trajectory = {"schema_version": "ATIF-v1.6", "steps": []}
    write_json(logs_dir / "simulhausen" / "trajectory.json", trajectory)

    make_agent(SimulhausenAgent).populate_context_post_run(AgentContext())

    assert json.loads((logs_dir / "trajectory.json").read_text()) == trajectory


def test_existing_root_trajectory_is_not_overwritten(make_agent: MakeAgent, logs_dir: Path):
    write_json(logs_dir / "simulhausen" / "trajectory.json", {"from": "play"})
    write_json(logs_dir / "trajectory.json", {"from": "elsewhere"})

    make_agent(SimulhausenAgent).populate_context_post_run(AgentContext())

    assert json.loads((logs_dir / "trajectory.json").read_text()) == {"from": "elsewhere"}


def test_trajectory_copy_failure_is_only_logged(
    make_agent: MakeAgent, logs_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    write_json(logs_dir / "simulhausen" / "trajectory.json", {})

    def deny(*args: object) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr("simulhausen.harbor.agents.shutil.copyfile", deny)
    with caplog.at_level(logging.WARNING):
        make_agent(SimulhausenAgent).populate_context_post_run(AgentContext())
    assert "could not copy" in caplog.text


def test_summary_tolerates_odd_field_types():
    summary = summarize_transcript({"scenario": "x", "target": None, "messages": {}, "agent_turns": {}})
    assert summary["scenario_id"] is None
    assert summary["target_type"] is None
    assert summary["messages"] == 0
    assert summary["tool_calls"] == []
    assert summary["harness_error"] is False
