"""Harbor itself loads the agents: its kwargs parser, AgentFactory, preflight and `harbor run --dry-run`.

These go through Harbor's real code paths in-process; nothing builds an image or starts a container (the docker
provider's preflight, which only checks for a running daemon, is stubbed out).
"""

import json
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

from simulhausen_harbor import A2AAgent, AgentOSAgent, LangGraphAgent, OpenAIChatAgent, SimulhausenAgent

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PACKAGE_ROOT / "examples" / "agentos"


@pytest.fixture(autouse=True)
def _no_docker_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DockerEnvironment, "preflight", classmethod(lambda cls: None))


@pytest.mark.parametrize(
    ("import_path", "cls", "ak", "model"),
    [
        (
            "simulhausen_harbor:AgentOSAgent",
            AgentOSAgent,
            ["base_url=http://agent-api:7777", "agent_id=support", 'session_state={"tier": "gold"}', "retries=3"],
            "support-agent@v1",
        ),
        ("simulhausen_harbor.agents:OpenAIChatAgent", OpenAIChatAgent, ["base_url=http://agent-api:8000/v1"], "my-llm"),
        ("simulhausen_harbor:A2AAgent", A2AAgent, ["url=http://agent-api:9999"], "a2a-agent@v2"),
        (
            "simulhausen_harbor:LangGraphAgent",
            LangGraphAgent,
            ["base_url=http://agent-api:2024", "assistant_id=agent", "api_key_env=LANGSMITH_API_KEY"],
            "graph@main",
        ),
        (
            "simulhausen_harbor:SimulhausenAgent",
            SimulhausenAgent,
            ['target={"type": "agentos", "base_url": "http://agent-api:7777"}', 'pip=["openai", "anthropic"]'],
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
        import_path="simulhausen_harbor:AgentOSAgent",
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


def test_preflight_rejects_bad_kwargs_before_any_trial():
    config = AgentConfig(name="simulhausen_harbor:AgentOSAgent", kwargs=parse_kwargs(["base_ur=http://x"]))
    with pytest.raises(ValueError, match="Unknown option 'base_ur'"):
        AgentFactory.run_preflight(config)


def write_exported_task(tasks_dir: Path, scenario_id: str, *, mode: str = "agent") -> Path:
    """A task directory shaped like `simulhausen harbor export` output."""
    task = tasks_dir / scenario_id
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.4"',
                "",
                "[task]",
                f'name = "simulhausen/{scenario_id}"',
                f'description = "{scenario_id}"',
                'keywords = ["support"]',
                "",
                "[metadata]",
                'generator = "simulhausen"',
                "",
                "[agent]",
                "timeout_sec = 900.0",
                "",
                "[verifier]",
                "timeout_sec = 600.0",
                "",
                "[verifier.env]",
                'OPENAI_API_KEY = "${OPENAI_API_KEY}"',
                'SIMULHAUSEN_JUDGE_MODEL = "${SIMULHAUSEN_JUDGE_MODEL:-}"',
                "",
                "[environment]",
                "build_timeout_sec = 600.0",
                "cpus = 1",
                "memory_mb = 2048",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("You were charged twice for order 1042.\n", encoding="utf-8")
    scenario = {"id": scenario_id, "name": scenario_id, "description": "You were charged twice for order 1042."}
    dockerfile = [
        "FROM python:3.12-slim",
        "COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/",
        "RUN uv pip install --system simulhausen==0.3.0 openai",
    ]
    if mode == "agent":
        (task / "environment" / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")
        dockerfile.append("COPY scenario.json /simulhausen/scenario.json")
    (task / "environment" / "Dockerfile").write_text("\n".join(dockerfile) + "\n", encoding="utf-8")
    (task / "tests" / "scenario.json").write_text(json.dumps({**scenario, "criteria": ["c"]}), encoding="utf-8")
    command = "judge" if mode == "agent" else "run"
    (task / "tests" / "test.sh").write_text(
        f"#!/bin/bash\nmkdir -p /logs/verifier\nsimulhausen harbor {command} --scenario /tests/scenario.json\n",
        encoding="utf-8",
    )
    return task


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    path = tmp_path / "tasks"
    write_exported_task(path, "refund-duplicated-charge")
    write_exported_task(path, "refund-missing-order-number")
    return path


def harbor(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, result.output


# The commands of README.md and examples/agentos/README.md, verbatim apart from paths and the expanded key.
def test_readme_option_c_command_passes_harbor_dry_run(tasks_dir: Path):
    code, output = harbor(
        "run",
        "-p",
        str(tasks_dir),
        "-a",
        "simulhausen_harbor:AgentOSAgent",
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
    assert "Dry run OK — 2 trial(s)" in output


def test_readme_option_a_command_passes_harbor_dry_run(tmp_path: Path):
    tasks = tmp_path / "tasks-verifier"
    write_exported_task(tasks, "refund-duplicated-charge", mode="verifier")
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
    assert "Dry run OK — 1 trial(s)" in output


def test_dry_run_reports_a_bad_kwarg(tasks_dir: Path):
    code, output = harbor(
        "run", "-p", str(tasks_dir), "-a", "simulhausen_harbor:AgentOSAgent", "--ak", "agent_id=support", "--dry-run"
    )
    assert code != 0
    assert "missing base_url" in output


def test_example_job_config_passes_harbor_dry_run(tmp_path: Path, tasks_dir: Path, monkeypatch: pytest.MonkeyPatch):
    # job.yaml paths are relative to integrations/harbor; mirror that layout around the exported tasks.
    (tmp_path / "examples").symlink_to(PACKAGE_ROOT / "examples", target_is_directory=True)
    (tmp_path / "src").symlink_to(PACKAGE_ROOT / "src", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    code, output = harbor("run", "-c", "examples/agentos/job.yaml", "-y", "--dry-run")

    assert code == 0, output
    assert "Dry run OK — 2 trial(s)" in output


def test_example_job_config_points_at_real_files(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # the agent env template is resolved during preflight
    config = JobConfig.model_validate(yaml.safe_load((EXAMPLE_DIR / "job.yaml").read_text()))
    [agent] = config.agents
    assert agent.import_path == "simulhausen_harbor:AgentOSAgent"
    assert agent.model_name == "support-agent@v1"
    AgentFactory.run_preflight(agent)
    assert [path.as_posix() for path in config.environment.extra_docker_compose] == [
        "examples/agentos/agentos-sidecar.yaml"
    ]
    assert (PACKAGE_ROOT / config.environment.extra_docker_compose[0]).is_file()
    [metric] = config.metrics
    script = PACKAGE_ROOT / metric.kwargs["script_path"]
    assert UvScript(script_path=script)  # the constructor checks that the script exists


def test_sidecar_overlay_matches_the_documented_service():
    overlay = yaml.safe_load((EXAMPLE_DIR / "agentos-sidecar.yaml").read_text())
    service = overlay["services"]["agent-api"]
    assert service["image"] == "simulhausen-agentos-demo:latest"
    assert "OPENAI_API_KEY=${OPENAI_API_KEY}" in service["environment"]
    assert "http://127.0.0.1:7777/health" in " ".join(service["healthcheck"]["test"])
    assert overlay["services"]["main"]["depends_on"]["agent-api"]["condition"] == "service_healthy"
