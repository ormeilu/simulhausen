"""Harbor itself loads the agents: its kwargs parser, AgentFactory, preflight and `harbor run --dry-run`.

These go through Harbor's real code paths in-process; nothing builds an image or starts a container (the docker
provider's preflight, which only checks for a running daemon, is stubbed out). Also: importing simulhausen never
imports Harbor, and the lazy re-exports in ``simulhausen.harbor`` say what to install when Harbor is missing.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.cli.main import app
from harbor.cli.utils import parse_kwargs
from harbor.environments.docker.docker import DockerEnvironment
from harbor.metrics.uv_script import UvScript
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig
from typer.testing import CliRunner

import simulhausen.harbor
from simulhausen.harbor import agents
from simulhausen.harbor.agents import A2AAgent, AgentOSAgent, LangGraphAgent, OpenAIChatAgent, SimulhausenAgent
from simulhausen.harbor.export import ExportMode, ExportOptions, export_tasks

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "harbor" / "agentos"


@pytest.fixture(autouse=True)
def _no_harbor_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARBOR_TELEMETRY", "0")


@pytest.fixture(autouse=True)
def _no_docker_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DockerEnvironment, "preflight", classmethod(lambda cls: None))


# --- importing ----------------------------------------------------------------------------------------------------


def test_plain_imports_load_neither_harbor_nor_litellm():
    # A fresh interpreter: this process has Harbor loaded already.
    code = (
        "import sys, simulhausen, simulhausen.harbor, simulhausen.cli\n"
        "print(sorted({name.split('.')[0] for name in sys.modules} & {'harbor', 'litellm'}))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("name", ["SimulhausenAgent", "AgentOSAgent", "OpenAIChatAgent", "A2AAgent", "LangGraphAgent"])
def test_package_reexports_the_agent_classes(name: str):
    assert getattr(simulhausen.harbor, name) is getattr(agents, name)


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchAgent'"):
        _ = simulhausen.harbor.NoSuchAgent


def test_missing_harbor_says_what_to_install():
    # What a plain `pip install simulhausen` sees: the agents module cannot import Harbor.
    with pytest.MonkeyPatch.context() as patch:
        patch.delitem(sys.modules, "simulhausen.harbor.agents")
        patch.setitem(sys.modules, "harbor.agents.installed.base", None)
        with pytest.raises(ImportError, match=r"pip install 'simulhausen\[harbor\]'"):
            _ = simulhausen.harbor.AgentOSAgent
        # The failed import left no half-initialised module behind.
        assert "simulhausen.harbor.agents" not in sys.modules
    assert importlib.import_module("simulhausen.harbor.agents") is agents


# --- AgentFactory -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("import_path", "cls", "ak", "model"),
    [
        (
            "simulhausen.harbor:AgentOSAgent",
            AgentOSAgent,
            ["base_url=http://agent-api:7777", "agent_id=support", 'session_state={"tier": "gold"}', "retries=3"],
            "support-agent@v1",
        ),
        ("simulhausen.harbor.agents:OpenAIChatAgent", OpenAIChatAgent, ["base_url=http://agent-api:8000/v1"], "my-llm"),
        ("simulhausen.harbor:A2AAgent", A2AAgent, ["url=http://agent-api:9999"], "a2a-agent@v2"),
        (
            "simulhausen.harbor:LangGraphAgent",
            LangGraphAgent,
            ["base_url=http://agent-api:2024", "assistant_id=agent", "api_key_env=LANGSMITH_API_KEY"],
            "graph@main",
        ),
        (
            "simulhausen.harbor:SimulhausenAgent",
            SimulhausenAgent,
            [
                'target={"type": "agentos", "base_url": "http://agent-api:7777", "agent_id": "support"}',
                'pip=["openai", "anthropic"]',
            ],
            None,
        ),
    ],
)
def test_agent_factory_builds_the_agents_from_cli_kwargs(
    tmp_path: Path, import_path: str, cls: type[SimulhausenAgent], ak: list[str], model: str | None
):
    config = AgentConfig(
        name=import_path,  # `-a module:Class` arrives as the agent name
        model_name=model,
        kwargs=parse_kwargs(["simulator_model=openai:gpt-4.1-mini", *ak]),
        env={"OPENAI_API_KEY": "sk-test"},
    )

    AgentFactory.run_preflight(config)
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)

    assert type(agent) is cls
    assert isinstance(agent, SimulhausenAgent)
    assert agent.extra_env == {"OPENAI_API_KEY": "sk-test"}
    assert agent.simulhausen_options.simulator_model == "openai:gpt-4.1-mini"
    assert agent.target is not None
    assert agent.target["type"] in {"agentos", "openai-chat", "a2a", "langgraph"}
    assert agent.to_agent_info().name == cls.name()


def test_parsed_kwargs_keep_their_json_types(tmp_path: Path):
    config = AgentConfig(
        import_path="simulhausen.harbor:AgentOSAgent",
        kwargs=parse_kwargs(["base_url=http://agent-api:7777", 'session_state={"tier": "gold"}', "agent_id=42"]),
    )
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(agent, AgentOSAgent)
    assert agent.target == {
        "type": "agentos",
        "base_url": "http://agent-api:7777",
        "agent_id": "42",  # a number on the command line, coerced back to the str the field wants
        "session_state": {"tier": "gold"},
    }


@pytest.mark.parametrize(
    ("import_path", "ak", "message"),
    [
        ("simulhausen.harbor:AgentOSAgent", ["base_ur=http://x"], "Unknown option 'base_ur'"),
        ("simulhausen.harbor:AgentOSAgent", ["base_url=http://x"], "needs one of agent_id, team_id, workflow_id"),
        (
            "simulhausen.harbor:AgentOSAgent",
            ["base_url=http://x", "agent_id=support", "team_id=desk"],
            "got agent_id and team_id",
        ),
        ("simulhausen.harbor:SimulhausenAgent", ['target={"type": "a2a", "ulr": "http://x"}'], "missing url"),
        (
            "simulhausen.harbor:A2AAgent",
            ["url=http://x", 'target={"histroy": "transcript"}'],
            "unexpected keyword argument 'histroy'",
        ),
        ("simulhausen.harbor:SimulhausenAgent", ["play_timeout_sec=Infinity"], "play_timeout_sec"),
        ("simulhausen.harbor:SimulhausenAgent", ["play_timeout_sec=NaN"], "play_timeout_sec"),
        ("simulhausen.harbor:A2AAgent", ["url=http://x", "timeout=Infinity"], "timeout"),
    ],
)
def test_preflight_rejects_bad_kwargs_before_any_trial(import_path: str, ak: list[str], message: str):
    config = AgentConfig(name=import_path, kwargs=parse_kwargs(ak))
    with pytest.raises(ValueError, match=message):
        AgentFactory.run_preflight(config)


def export_example(output_dir: Path, *, mode: ExportMode = "agent") -> Path:
    """The example's scenarios as `simulhausen harbor export` writes them: one task per scenario."""
    export_tasks([EXAMPLE_DIR / "scenarios.toml"], ExportOptions(output_dir=output_dir, mode=mode))
    return output_dir


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    return export_example(tmp_path / "tasks")


def harbor(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, result.output


# The commands of README.md and examples/harbor/agentos/README.md, verbatim apart from paths and the expanded key.
def test_readme_agent_mode_command_passes_harbor_dry_run(tasks_dir: Path):
    code, output = harbor(
        "run",
        "-p",
        str(tasks_dir),
        "-a",
        "simulhausen.harbor:AgentOSAgent",
        "--ak",
        "base_url=http://agent-api:7777",
        "--ak",
        "agent_id=support",
        "-m",
        "support-agent@v1",
        "--extra-docker-compose",
        str(EXAMPLE_DIR / "agentos-sidecar.yaml"),
        "--ae",
        "OPENAI_API_KEY=sk-test",
        "-y",
        "--dry-run",
    )
    assert code == 0, output
    assert "Dry run OK — 3 trial(s)" in output


def test_readme_verifier_mode_command_passes_harbor_dry_run(tmp_path: Path):
    tasks = export_example(tmp_path / "tasks-verifier", mode="verifier")
    code, output = harbor(
        "run",
        "-p",
        str(tasks),
        "-a",
        "nop",
        "--extra-docker-compose",
        str(EXAMPLE_DIR / "agentos-sidecar.yaml"),
        "--job-name",
        "support-agent-v1",
        "-y",
        "--dry-run",
    )
    assert code == 0, output
    assert "Dry run OK — 3 trial(s)" in output


@pytest.mark.parametrize(
    ("ak", "message"),
    [
        (["agent_id=support"], "missing base_url"),
        (["base_url=http://agent-api:7777"], "needs one of agent_id, team_id, workflow_id"),
        (["base_url=http://agent-api:7777", 'target={"agent_id": "support", "type": "agentoss"}'], "agentoss"),
    ],
    ids=["missing base_url", "no agentos id", "misspelt type"],
)
def test_dry_run_reports_a_bad_kwarg(tasks_dir: Path, ak: list[str], message: str):
    flags = [arg for value in ak for arg in ("--ak", value)]
    code, output = harbor("run", "-p", str(tasks_dir), "-a", "simulhausen.harbor:AgentOSAgent", *flags, "--dry-run")
    assert code != 0
    assert message in " ".join(output.split())  # Harbor wraps its error to the terminal width


def test_example_job_config_passes_harbor_dry_run(tmp_path: Path, tasks_dir: Path, monkeypatch: pytest.MonkeyPatch):
    # job.yaml paths are relative to the repository root; mirror that layout around the exported tasks.
    (tmp_path / "examples").symlink_to(REPO_ROOT / "examples", target_is_directory=True)
    (tmp_path / "src").symlink_to(REPO_ROOT / "src", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    code, output = harbor("run", "-c", "examples/harbor/agentos/job.yaml", "-y", "--dry-run")

    assert code == 0, output
    assert "Dry run OK — 3 trial(s)" in output


def test_example_job_config_points_at_real_files(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # the agent env template is resolved during preflight
    config = JobConfig.model_validate(yaml.safe_load((EXAMPLE_DIR / "job.yaml").read_text()))
    [agent] = config.agents
    assert agent.import_path == "simulhausen.harbor:AgentOSAgent"
    assert agent.model_name == "support-agent@v1"
    AgentFactory.run_preflight(agent)
    assert [path.as_posix() for path in config.environment.extra_docker_compose] == [
        "examples/harbor/agentos/agentos-sidecar.yaml"
    ]
    assert (REPO_ROOT / config.environment.extra_docker_compose[0]).is_file()
    [metric] = config.metrics
    script = REPO_ROOT / metric.kwargs["script_path"]
    assert script == REPO_ROOT / "src" / "simulhausen" / "harbor" / "metric.py"
    assert UvScript(script_path=script)  # the constructor checks that the script exists


def test_sidecar_overlay_matches_the_documented_service():
    overlay = yaml.safe_load((EXAMPLE_DIR / "agentos-sidecar.yaml").read_text())
    service = overlay["services"]["agent-api"]
    assert service["image"] == "simulhausen-agentos-demo:latest"
    assert "OPENAI_API_KEY=${OPENAI_API_KEY}" in service["environment"]
    assert "http://127.0.0.1:7777/health" in " ".join(service["healthcheck"]["test"])
    assert overlay["services"]["main"]["depends_on"]["agent-api"]["condition"] == "service_healthy"
