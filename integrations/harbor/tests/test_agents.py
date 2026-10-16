"""SimulhausenAgent and the protocol subclasses against a recording environment: install, play, context."""

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
from harbor.models.trial.paths import EnvironmentPaths
from support import FakeEnvironment, write_json

from simulhausen_harbor import (
    DEFAULT_SIMULHAUSEN_SPEC,
    SIMULHAUSEN_VERSION,
    A2AAgent,
    AgentOSAgent,
    LangGraphAgent,
    OpenAIChatAgent,
    SimulhausenAgent,
)
from simulhausen_harbor.agents import TARGET_FILE, parse_version, summarize_transcript

MakeAgent = Callable[..., SimulhausenAgent]

PIPEFAIL = "set -o pipefail; "
INSTALLED_AGENT_DIR = "[ -d /installed-agent ] || mkdir -p /installed-agent"
MISSING = ExecResult(stdout="", stderr="", return_code=1)


def found(stdout: str) -> ExecResult:
    return ExecResult(stdout=stdout, stderr="", return_code=0)


# --- identity ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "kwargs", "name"),
    [
        (SimulhausenAgent, {}, "simulhausen"),
        (AgentOSAgent, {"base_url": "http://agent-api:7777"}, "simulhausen-agentos"),
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


def test_version_defaults_to_the_pinned_release(make_agent: MakeAgent):
    agent = make_agent(SimulhausenAgent)
    assert agent.version() == SIMULHAUSEN_VERSION
    assert agent.simulhausen_spec == DEFAULT_SIMULHAUSEN_SPEC


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
    agent = make_agent(AgentOSAgent, base_url="http://agent-api:7777")

    with environment.with_default_user("agent"):
        await agent.setup(environment)

    assert environment.commands == [INSTALLED_AGENT_DIR, "command -v simulhausen", "simulhausen --version"]
    assert environment.calls[1].user == "agent"
    # The version the container actually has wins over the pinned default.
    assert agent.version() == "0.3.2"
    assert agent.to_agent_info().version == "0.3.2"


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
        f"{DEFAULT_SIMULHAUSEN_SPEC} --with openai && /usr/local/bin/simulhausen --version"
        "; } 2>&1 | tee /logs/agent/setup/simulhausen-install.log"
    )
    assert environment.commands[4:] == ["simulhausen --version"]


async def test_setup_fetches_uv_when_missing(make_agent: MakeAgent, environment: FakeEnvironment):
    environment.answer("command -v simulhausen", MISSING)
    environment.answer("command -v uv", MISSING)
    environment.answer("command -v curl >/dev/null 2>&1", MISSING)
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
    install = next(c for c in environment.calls if "uv tool install" in c.command)
    assert install.user == "root"
    assert (
        "curl -LsSf https://astral.sh/uv/0.12.21/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"
        in install.command
    )
    assert (
        "/usr/local/bin/uv tool install --python '>=3.12' "
        "'simulhausen @ git+https://github.com/ormeilu/simulhausen@main' --with openai --with 'anthropic>=0.40'"
    ) in install.command


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

    assert agent.version() == SIMULHAUSEN_VERSION


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
        "'simulhausen-harbor: timeout(1) not found in the container; play_timeout_sec is ignored' >&2; "
        "simulhausen harbor play --scenario /simulhausen/scenario.json --out /logs/agent/simulhausen; fi"
    ) in play.command


async def test_failed_play_raises_the_error_that_still_runs_the_verifier(
    make_agent: MakeAgent, environment: FakeEnvironment
):
    # SingleStepTrial catches NonZeroAgentExitCodeError (and AgentTimeoutError), records it and verifies anyway;
    # any other exception from run() would skip the verifier and lose the WARN report.
    environment.answer(
        "simulhausen harbor play",
        ExecResult(stdout="harness error: agent-api unreachable", stderr="", return_code=2),
        contains=True,
    )
    agent = make_agent(AgentOSAgent, base_url="http://agent-api:7777")

    with pytest.raises(NonZeroAgentExitCodeError, match="exit 2"):
        await agent.run("instruction", environment, AgentContext())


def test_out_dir_follows_the_environment_log_dir(make_agent: MakeAgent):
    agent = make_agent(SimulhausenAgent)
    assert agent.environment_out_dir.as_posix() == f"{EnvironmentPaths.agent_dir}/simulhausen"


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
        target={"agent_id": "billing", "workflow_id": "refunds"},
    )
    assert agent.target == {
        "type": "agentos",
        "base_url": "http://agent-api:7777",
        "agent_id": "billing",
        "workflow_id": "refunds",
    }


def test_target_may_be_a_json_string(make_agent: MakeAgent):
    agent = make_agent(SimulhausenAgent, target='{"type": "openai-responses", "base_url": "http://x/v1", "model": "m"}')
    assert agent.target == {"type": "openai-responses", "base_url": "http://x/v1", "model": "m"}


def test_explicit_none_target_means_no_target(make_agent: MakeAgent):
    # `--ak target=None` parses to None, like an omitted kwarg.
    assert make_agent(SimulhausenAgent, target=None).target is None


def test_target_type_mismatch_is_allowed_but_logged(make_agent: MakeAgent, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        agent = make_agent(AgentOSAgent, target={"type": "a2a", "url": "http://agent-api:9999", "base_url": "x"})
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
    ],
)
def test_invalid_kwargs_are_rejected(make_agent: MakeAgent, kwargs: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        make_agent(SimulhausenAgent, **kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs", "missing"),
    [
        (AgentOSAgent, {"agent_id": "support"}, "missing base_url"),
        (A2AAgent, {}, "missing url"),
        (LangGraphAgent, {"base_url": "http://agent-api:2024"}, "missing assistant_id"),
        (LangGraphAgent, {}, "missing base_url, assistant_id"),
    ],
)
def test_required_target_keys(make_agent: MakeAgent, cls: type[SimulhausenAgent], kwargs: dict, missing: str):
    with pytest.raises(ValueError, match=missing):
        make_agent(cls, **kwargs)


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
            A2AAgent,
            url="http://agent-api:9999",
            headers={"Authorization": "Bearer abc", "X-Tenant": "acme"},
            target={"auth_token": "abc", "token_env": "A2A_TOKEN"},
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

    make_agent(AgentOSAgent, base_url="http://agent-api:7777").populate_context_post_run(context)

    assert context.metadata == {
        "scenario_id": "refund-duplicated-charge",
        "scenario_name": "refund: duplicated charge",
        "target_type": "agentos",
        "turns": 2,
        "messages": 4,
        "tool_calls": ["lookup_order", "issue_refund"],
        "failed_tool_calls": ["issue_refund", "issue_refund"],
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


def test_unknown_transcript_format_is_summarised_with_a_warning(
    make_agent: MakeAgent, logs_dir: Path, caplog: pytest.LogCaptureFixture
):
    write_json(logs_dir / "simulhausen" / "transcript.json", {"format": "other/9", "turns": 1})
    context = AgentContext()
    with caplog.at_level(logging.WARNING):
        make_agent(SimulhausenAgent).populate_context_post_run(context)
    assert context.metadata is not None
    assert context.metadata["turns"] == 1
    assert context.metadata["messages"] == 0
    assert context.metadata["tool_calls"] == []
    assert "unexpected transcript format 'other/9'" in caplog.text


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

    monkeypatch.setattr("simulhausen_harbor.agents.shutil.copyfile", deny)
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
