import json
import logging
import shutil
import stat
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
from harbor_support import installed_from

from simulhausen import __version__
from simulhausen.harbor import ConfigError, load_scenarios
from simulhausen.harbor import export as export_module
from simulhausen.harbor.export import UV_IMAGE, ExportOptions, default_simulhausen_spec, export_tasks

SCENARIOS = """
[defaults]
target = { type = "agentos", base_url = "http://agent-api:7777", agent_id = "support", api_key_env = "AGENTOS_KEY" }
simulator_model = "openai:gpt-4.1-mini"
judge_model = "openai:gpt-4.1"

[[scenarios]]
id = "refund"
name = "refund: \\"duplicated\\" charge"
description = "The user was charged twice.\\nThey want the extra charge back."
criteria = ["The agent looks the order up first"]
additional_context = "order 1042 exists"
score_threshold = 7
tags = ["support", "billing"]
checks = [{ type = "tool_not_called", tool = "issue_refund" }]
[[scenarios.script]]
user = "i was charged twice"
[[scenarios.script]]
agent = true
[[scenarios.script]]
judge = { criteria = ["The agent asks for the order number"] }
[[scenarios.script]]
proceed = { turns = 2 }

[[scenarios]]
id = "greeting"
name = "greeting"
description = "Says hello."
criteria = ["The agent greets back"]
"""


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "scenarios.toml"
    path.write_text(SCENARIOS)
    return path


def _toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def test_agent_mode_renders_a_complete_task(scenario_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "tasks"

    written = export_tasks([scenario_file], ExportOptions(output_dir=out, org="acme"))

    assert written == [out / "refund", out / "greeting"]
    task = out / "refund"
    assert sorted(str(p.relative_to(task)) for p in task.rglob("*") if p.is_file()) == [
        "environment/Dockerfile",
        "environment/scenario.json",
        "instruction.md",
        "task.toml",
        "tests/scenario.json",
        "tests/test.sh",
    ]
    config = _toml(task / "task.toml")
    assert config["schema_version"] == "1.4"
    assert config["task"] == {
        "name": "acme/refund",
        "description": 'refund: "duplicated" charge',
        "keywords": ["support", "billing"],
    }
    assert config["metadata"] == {
        "generator": "simulhausen",
        "simulhausen_version": __version__,
        "scenario_id": "refund",
        "mode": "agent",
    }
    assert config["agent"] == {"timeout_sec": 900.0}
    assert config["verifier"]["timeout_sec"] == 600.0
    assert config["verifier"]["env"] == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "SIMULHAUSEN_JUDGE_MODEL": "${SIMULHAUSEN_JUDGE_MODEL:-}",
        "SIMULHAUSEN_MODEL": "${SIMULHAUSEN_MODEL:-}",
    }
    assert config["environment"] == {"build_timeout_sec": 600.0, "cpus": 1, "memory_mb": 2048}
    assert (task / "instruction.md").read_text() == "The user was charged twice.\nThey want the extra charge back.\n"

    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "FROM python:3.12-slim\n" in dockerfile
    assert f"COPY --from={UV_IMAGE} /uv /uvx /bin/\n" in dockerfile
    assert f"RUN uv pip install --system --no-cache simulhausen=={__version__} openai\n" in dockerfile
    assert dockerfile.endswith("COPY scenario.json /simulhausen/scenario.json\n")

    test_sh = task / "tests" / "test.sh"
    assert test_sh.stat().st_mode & stat.S_IXUSR
    script = test_sh.read_text()
    assert script.startswith("#!/bin/bash\n")
    assert "set -e" not in script.replace("No `set -e`", "")
    assert "mkdir -p /logs/verifier\n" in script
    assert script.endswith("simulhausen harbor judge --scenario /tests/scenario.json --out /logs/verifier\n")


def test_a_git_requirement_gets_git_in_the_image(scenario_file: Path, tmp_path: Path) -> None:
    spec = "simulhausen @ git+https://github.com/ormeilu/simulhausen@main"
    export_tasks([scenario_file], ExportOptions(output_dir=tmp_path, simulhausen_spec=spec))

    dockerfile = (tmp_path / "refund" / "environment" / "Dockerfile").read_text()
    git_line = dockerfile.index("apt-get install -y --no-install-recommends git")
    assert git_line < dockerfile.index("RUN uv pip install")
    assert f"RUN uv pip install --system --no-cache '{spec}' openai\n" in dockerfile


def test_test_sh_says_so_when_the_image_lacks_the_cli(scenario_file: Path, tmp_path: Path) -> None:
    export_tasks([scenario_file], ExportOptions(output_dir=tmp_path))

    script = tmp_path / "refund" / "tests" / "test.sh"
    bash = shutil.which("bash")
    assert bash is not None
    # An empty PATH: neither simulhausen nor anything else is found, as in an image without the CLI.
    missing = subprocess.run([bash, str(script)], env={"PATH": ""}, capture_output=True, text=True, check=False)

    assert missing.returncode == 1
    assert "simulhausen: command not found in the task image" in missing.stderr
    assert "--simulhausen-spec" in missing.stderr


def test_the_image_gets_the_scenario_without_its_criteria(scenario_file: Path, tmp_path: Path) -> None:
    export_tasks([scenario_file], ExportOptions(output_dir=tmp_path))

    image = (tmp_path / "refund" / "environment" / "scenario.json").read_text()
    full = json.loads((tmp_path / "refund" / "tests" / "scenario.json").read_text())

    for secret in ["looks the order up", "asks for the order number", "order 1042 exists", "issue_refund", 'gpt-4.1"']:
        assert secret not in image
    assert '"simulator_model": "openai:gpt-4.1-mini"' in image
    assert '"api_key_env": "AGENTOS_KEY"' in image
    # Both are single scenarios with the defaults merged in, so `--id` is not needed.
    assert load_scenarios(tmp_path / "refund" / "environment" / "scenario.json")[0].steps[2].judge is not None
    [reloaded] = load_scenarios(tmp_path / "refund" / "tests" / "scenario.json")
    assert reloaded == load_scenarios(scenario_file)[0]
    assert full["judge_model"] == "openai:gpt-4.1"


def test_verifier_mode_plays_in_the_verifier(scenario_file: Path, tmp_path: Path) -> None:
    compose = tmp_path / "sidecar.yaml"
    compose.write_text("services:\n  agent-api:\n    image: my-agent\n")
    options = ExportOptions(
        output_dir=tmp_path / "tasks",
        mode="verifier",
        env=["OPENAI_API_KEY", "AGENTOS_KEY", "OPENAI_API_KEY", "SIMULHAUSEN_MODEL"],
        pip=["openai>=1.0", "anthropic"],
        simulhausen_spec="simulhausen>=0.3",
        compose=compose,
        ids=["greeting"],
        rewards="detailed",
        warn_reward="zero",
    )

    [task] = export_tasks([scenario_file], options)

    assert task.name == "greeting"
    assert not (task / "environment" / "scenario.json").exists()
    assert (task / "environment" / "docker-compose.yaml").read_text() == compose.read_text()
    config = _toml(task / "task.toml")
    assert config["verifier"]["timeout_sec"] == 1800.0
    assert config["metadata"]["mode"] == "verifier"
    assert config["verifier"]["env"] == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "AGENTOS_KEY": "${AGENTOS_KEY}",
        "SIMULHAUSEN_MODEL": "${SIMULHAUSEN_MODEL}",
        "SIMULHAUSEN_JUDGE_MODEL": "${SIMULHAUSEN_JUDGE_MODEL:-}",
        "SIMULHAUSEN_SIMULATOR_MODEL": "${SIMULHAUSEN_SIMULATOR_MODEL:-}",
        "SIMULHAUSEN_TARGET": "${SIMULHAUSEN_TARGET:-}",
    }
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "RUN uv pip install --system --no-cache 'simulhausen>=0.3' 'openai>=1.0' anthropic\n" in dockerfile
    assert "COPY scenario.json" not in dockerfile
    assert (
        (task / "tests" / "test.sh")
        .read_text()
        .endswith(
            "simulhausen harbor run --scenario /tests/scenario.json --out /logs/verifier --rewards detailed "
            "--warn-reward zero\n"
        )
    )


def test_timeouts_can_be_set(scenario_file: Path, tmp_path: Path) -> None:
    options = ExportOptions(output_dir=tmp_path, agent_timeout=60, verifier_timeout=120.5, ids=["greeting"])

    export_tasks([scenario_file], options)

    config = _toml(tmp_path / "greeting" / "task.toml")
    assert (config["agent"]["timeout_sec"], config["verifier"]["timeout_sec"]) == (60.0, 120.5)


def test_strings_survive_toml_escaping(tmp_path: Path) -> None:
    tricky = 'quote " backslash \\ tab \t bell \x07 del \x7f ünïcode 🙂'
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"id": "t", "name": tricky, "description": "d", "criteria": ["c"], "tags": [tricky]}))

    export_tasks([path], ExportOptions(output_dir=tmp_path / "out"))

    task = _toml(tmp_path / "out" / "t" / "task.toml")["task"]
    assert task["description"] == tricky
    assert task["keywords"] == [tricky]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"org": "-bad"}, "--org '-bad'"),
        ({"org": "a..b"}, "--org 'a..b'"),
        ({"env": ["OPENAI-KEY"]}, "--env 'OPENAI-KEY' is not an environment variable name"),
        ({"pip": [" "]}, "' ' is not a usable pip requirement"),
        ({"simulhausen_spec": "simulhausen\nRUN evil"}, "is not a usable pip requirement"),
        ({"agent_timeout": 0}, "--agent-timeout must be a positive number of seconds"),
        ({"verifier_timeout": float("inf")}, "--verifier-timeout must be a positive number of seconds"),
        ({"compose": Path("/no/such/compose.yaml")}, "--compose /no/such/compose.yaml is not a file"),
        ({"ids": ["refund", "nope"]}, "--ids ['nope'] not found; the scenario ids are: refund, greeting"),
    ],
)
def test_option_errors(scenario_file: Path, tmp_path: Path, options: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError) as info:
        export_tasks([scenario_file], ExportOptions(output_dir=tmp_path / "out", **options))
    assert message in str(info.value)
    assert not (tmp_path / "out").exists()


def test_scenario_errors_stop_the_export_before_anything_is_written(scenario_file: Path, tmp_path: Path) -> None:
    ungradable = tmp_path / "ungradable.json"
    ungradable.write_text(json.dumps({"id": "u", "name": "u", "description": "d"}))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"id": "greeting", "name": "g", "description": "d", "criteria": ["c"]}))

    with pytest.raises(ConfigError, match="scenario 'u' cannot be graded"):
        export_tasks([scenario_file, ungradable], ExportOptions(output_dir=tmp_path / "out"))
    with pytest.raises(ConfigError, match="scenario id 'greeting' is in both"):
        export_tasks([scenario_file, duplicate], ExportOptions(output_dir=tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_ids_that_differ_only_in_case_are_rejected(scenario_file: Path, tmp_path: Path) -> None:
    """On macOS and Windows "Refund" and "refund" are one task directory: the second export would replace the first."""
    other_file = tmp_path / "other.json"
    other_file.write_text(json.dumps({"id": "Refund", "name": "r", "description": "d", "criteria": ["c"]}))
    same_file = tmp_path / "same.json"
    same_file.write_text(
        json.dumps(
            {"scenarios": [{"id": f"a{suffix}", "name": "a", "description": "d", "criteria": ["c"]} for suffix in "bB"]}
        )
    )

    with pytest.raises(
        ConfigError,
        match=r"scenario ids 'refund' \(.*scenarios\.toml\) and 'Refund' \(.*other\.json\) differ only in case",
    ):
        export_tasks([scenario_file, other_file], ExportOptions(output_dir=tmp_path / "out"))
    with pytest.raises(ConfigError, match=r"'ab' .* and 'aB' .* differ only in case"):
        export_tasks([same_file], ExportOptions(output_dir=tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_overwrite_replaces_only_generated_tasks(scenario_file: Path, tmp_path: Path) -> None:
    options = ExportOptions(output_dir=tmp_path, ids=["greeting"])
    export_tasks([scenario_file], options)
    (tmp_path / "greeting" / "leftover.txt").write_text("old")

    with pytest.raises(ConfigError, match="already exists; pass --overwrite"):
        export_tasks([scenario_file], options)
    export_tasks([scenario_file], ExportOptions(output_dir=tmp_path, ids=["greeting"], overwrite=True))

    assert not (tmp_path / "greeting" / "leftover.txt").exists()
    assert (tmp_path / "greeting" / "task.toml").exists()


def test_overwrite_never_deletes_a_foreign_directory(scenario_file: Path, tmp_path: Path) -> None:
    foreign = tmp_path / "greeting"
    foreign.mkdir()
    (foreign / "task.toml").write_text('[metadata]\ngenerator = "someone else"\n')
    broken = tmp_path / "refund"
    broken.mkdir()
    (broken / "task.toml").write_text("not toml [")

    for scenario_id in ("greeting", "refund"):
        with pytest.raises(ConfigError, match="was not generated by simulhausen"):
            export_tasks([scenario_file], ExportOptions(output_dir=tmp_path, ids=[scenario_id], overwrite=True))
    assert (foreign / "task.toml").exists()


def test_default_simulhausen_spec(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(export_module, "__version__", "1.2.3")
    assert default_simulhausen_spec() == "simulhausen==1.2.3"

    monkeypatch.setattr(export_module, "__version__", "0+unknown")
    assert default_simulhausen_spec() == "simulhausen"
    assert "0+unknown is not a released version" in caplog.text


ARCHIVE = "https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz"
COMMIT = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        ({"url": ARCHIVE, "archive_info": {"hashes": {"sha256": "ab12"}}}, f"simulhausen @ {ARCHIVE}"),
        (
            {
                "url": "https://github.com/ormeilu/simulhausen.git",
                "vcs_info": {"vcs": "git", "commit_id": COMMIT, "requested_revision": "main"},
            },
            f"simulhausen @ git+https://github.com/ormeilu/simulhausen.git@{COMMIT}",
        ),
        (
            {
                "url": "git+ssh://git@github.com/ormeilu/simulhausen.git",
                "vcs_info": {"vcs": "git", "commit_id": COMMIT},
            },
            f"simulhausen @ git+ssh://git@github.com/ormeilu/simulhausen.git@{COMMIT}",
        ),
    ],
    ids=["archive", "git", "git url with its prefix"],
)
def test_default_spec_repeats_a_remote_install(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, direct_url: dict[str, Any], expected: str
) -> None:
    # Such an install reports the last release (0.2.0 has no CLI): pinning that would break the task image.
    installed_from(monkeypatch, direct_url)
    monkeypatch.setattr(export_module, "__version__", "0.2.0")
    with caplog.at_level(logging.WARNING):
        assert default_simulhausen_spec() == expected
    assert caplog.text == ""


@pytest.mark.parametrize(
    "direct_url",
    [
        {"url": "file:///home/me/simulhausen", "dir_info": {"editable": True}},
        {"url": "file:///home/me/simulhausen", "dir_info": {}},
        {"url": "file:///tmp/simulhausen-0.3.0-py3-none-any.whl", "archive_info": {}},
        {"url": "file:///home/me/simulhausen", "vcs_info": {"vcs": "git", "commit_id": COMMIT}},
        {"url": "https://hg.example.com/simulhausen", "vcs_info": {"vcs": "hg", "commit_id": COMMIT}},
        {"url": "https://github.com/ormeilu/simulhausen.git", "vcs_info": {"vcs": "git"}},
    ],
    ids=["editable checkout", "local directory", "local wheel", "local repository", "mercurial", "no commit"],
)
def test_default_spec_of_an_unreachable_install_falls_back_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, direct_url: dict[str, Any]
) -> None:
    installed_from(monkeypatch, direct_url)
    monkeypatch.setattr(export_module, "__version__", "0.2.0")
    with caplog.at_level(logging.WARNING):
        assert default_simulhausen_spec() == "simulhausen==0.2.0"
    assert f"installed from {direct_url['url']}, which a task container cannot install from" in caplog.text
    assert "installs simulhausen==0.2.0 instead" in caplog.text
    assert "Pass --simulhausen-spec to install the same code" in caplog.text


def test_unreachable_install_of_an_unreleased_version_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    installed_from(monkeypatch, {"url": "file:///home/me/simulhausen", "dir_info": {"editable": True}})
    monkeypatch.setattr(export_module, "__version__", "0+unknown")
    with caplog.at_level(logging.WARNING):
        assert default_simulhausen_spec(option="--ak simulhausen_spec=...") == "simulhausen"
    assert len(caplog.records) == 1
    assert "Pass --ak simulhausen_spec=... to install the same code" in caplog.text


@pytest.mark.parametrize(
    "direct_url",
    [None, "", "{truncated", '["a", "list"]', '{"archive_info": {}}', '{"url": 42}'],
    ids=["index", "empty", "broken json", "json array", "no url", "url not a string"],
)
def test_default_spec_pins_the_version_without_a_usable_direct_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, direct_url: str | None
) -> None:
    installed_from(monkeypatch, direct_url)
    monkeypatch.setattr(export_module, "__version__", "1.2.3")
    with caplog.at_level(logging.WARNING):
        assert default_simulhausen_spec() == "simulhausen==1.2.3"
    assert caplog.text == ""


def test_default_spec_without_installed_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_installed(name: str) -> None:
        raise export_module.PackageNotFoundError(name)

    monkeypatch.setattr(export_module, "distribution", not_installed)
    monkeypatch.setattr(export_module, "__version__", "1.2.3")
    assert default_simulhausen_spec() == "simulhausen==1.2.3"


def test_export_bakes_the_host_archive_into_the_image(
    scenario_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_from(monkeypatch, {"url": ARCHIVE, "archive_info": {}})
    [task] = export_tasks([scenario_file], ExportOptions(output_dir=tmp_path / "tasks", ids=["greeting"]))
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert f"RUN uv pip install --system --no-cache 'simulhausen @ {ARCHIVE}' openai\n" in dockerfile
