import asyncio
import json
import logging
import os
import runpy
import signal
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from harbor_support import TARGET, ScriptedJudge, ScriptedSimulator, write_scenarios

from simulhausen import (
    AgentAdapter,
    AgentInput,
    JudgeAgent,
    SimulationBackendError,
    UserSimulatorAgent,
    __version__,
    function_adapter,
)
from simulhausen.cli import EXIT_OK, EXIT_UNVERIFIED, EXIT_USAGE, main
from simulhausen.harbor import ScenarioSpec
from simulhausen.harbor import judge as judge_module
from simulhausen.harbor import play as play_module

SCENARIO: dict[str, Any] = {
    "id": "refund",
    "name": "refund: duplicated charge",
    "description": "Charged twice.",
    "target": TARGET,
    "criteria": ["The agent is polite"],
    "checks": [{"type": "tool_called", "tool": "lookup_order"}],
    "script": [{"user": "i was charged twice"}, {"agent": True}, {"user": True}, {"agent": True}],
}


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    return write_scenarios(tmp_path / "scenarios.json", {"scenarios": [SCENARIO]})


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake simulator and judge for every command; tests change their behaviour through the dict."""
    state: dict[str, Any] = {"simulator": ScriptedSimulator("order 1042"), "statuses": {}, "built": []}

    def build_simulator(spec: ScenarioSpec, model: str | None) -> UserSimulatorAgent:
        state["built"].append(("simulator", model))
        return state["simulator"]

    def build_judge(spec: ScenarioSpec, model: str | None) -> JudgeAgent:
        state["built"].append(("judge", model))
        return ScriptedJudge(spec.criteria, statuses=state["statuses"])

    monkeypatch.setattr(play_module, "build_simulator", build_simulator)
    monkeypatch.setattr(judge_module, "build_judge", build_judge)
    return state


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


# -- play ----------------------------------------------------------------------


def test_play_writes_the_transcript(
    scenario_file: Path, tmp_path: Path, helpers: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "agent"

    code = main(["harbor", "play", "--scenario", str(scenario_file), "--out", str(out), "--model", "openai:x"])

    assert code == EXIT_OK
    assert sorted(path.name for path in out.iterdir()) == ["trajectory.json", "transcript.json", "transcript.md"]
    transcript = _json(out / "transcript.json")
    assert [m["content"] for m in transcript["messages"] if m["role"] == "user"] == [
        "i was charged twice",
        "order 1042",
    ]
    assert helpers["built"] == [("simulator", "openai:x")]
    captured = capsys.readouterr()
    assert f"refund: 2 turn(s), stop reason script_end; transcript written to {out / 'transcript.json'}" in captured.out
    # The dialogue is logged to stderr while it runs; the handler goes away afterwards.
    assert "user: i was charged twice" in captured.err
    assert not any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        for h in logging.getLogger("simulhausen").handlers
    )


def test_play_exits_2_when_the_dialogue_does_not_finish(
    scenario_file: Path, tmp_path: Path, helpers: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    helpers["simulator"] = ScriptedSimulator(SimulationBackendError("gateway timeout"))

    code = main(["harbor", "play", "--scenario", str(scenario_file), "--out", str(tmp_path)])

    assert code == EXIT_UNVERIFIED
    assert _json(tmp_path / "transcript.json")["harness_error"] is True
    assert "the dialogue did not finish (harness failure): SimulationBackendError: gateway timeout" in (
        capsys.readouterr().err
    )


def test_play_reports_an_agent_error_on_stderr(
    tmp_path: Path, helpers: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_scenarios(tmp_path / "s.json", {**SCENARIO, "script": [{"user": "explode"}, {"agent": True}]})

    target = json.dumps({**TARGET, "fail_on": "explode"})
    code = main(["harbor", "play", "--scenario", str(path), "--out", str(tmp_path / "o"), "--target-json", target])

    assert code == EXIT_UNVERIFIED
    assert "the dialogue did not finish (error): RuntimeError" in capsys.readouterr().err


def test_play_takes_the_target_from_a_file(tmp_path: Path, helpers: dict[str, Any]) -> None:
    path = write_scenarios(tmp_path / "s.json", {**SCENARIO, "target": None})
    target_file = tmp_path / "target.json"
    target_file.write_text(json.dumps({**TARGET, "prefix": "from file"}))

    code = main(["harbor", "play", "--scenario", str(path), "--target-file", str(target_file), "--out", str(tmp_path)])

    assert code == EXIT_OK
    assert _json(tmp_path / "transcript.json")["messages"][1]["content"].startswith("from file reply 1")


def test_configuration_errors_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_scenarios(tmp_path / "s.json", {**SCENARIO, "target": None})

    code = main(["harbor", "play", "--scenario", str(path), "--out", str(tmp_path / "out")])

    assert code == EXIT_USAGE
    assert "simulhausen: error: no target for scenario 'refund'" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_play_removes_an_old_dialogue_before_anything_can_fail(
    scenario_file: Path, tmp_path: Path, helpers: dict[str, Any]
) -> None:
    out = tmp_path / "agent"
    _play(scenario_file, out)
    (out / "notes.txt").write_text("mine")
    broken = write_scenarios(tmp_path / "broken.json", {**SCENARIO, "critera": ["typo"]})

    code = main(["harbor", "play", "--scenario", str(broken), "--out", str(out)])

    # A later judge finds no transcript (WARN) instead of grading the old dialogue as new.
    assert code == EXIT_USAGE
    assert [path.name for path in out.iterdir()] == ["notes.txt"]


def interrupted_agent() -> AgentAdapter:
    """A python target that stalls on "hang" after SIGINT arrives, as `timeout -s INT` stops a play that runs long."""

    async def reply(agent_input: AgentInput) -> str:
        if "hang" in agent_input.last_new_user_message_str():
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        return "ok"

    return function_adapter(reply)


@pytest.mark.parametrize("command", ["play", "run"])
def test_sigint_keeps_the_dialogue_played_so_far(
    tmp_path: Path, helpers: dict[str, Any], capsys: pytest.CaptureFixture[str], command: str
) -> None:
    script = [{"user": "hello"}, {"agent": True}, {"user": "hang"}, {"agent": True}]
    path = write_scenarios(tmp_path / "s.json", {**SCENARIO, "script": script})
    target = json.dumps({"type": "python", "factory": "test_harbor_cli:interrupted_agent"})

    code = main(["harbor", command, "--scenario", str(path), "--out", str(tmp_path), "--target-json", target])

    assert code == EXIT_UNVERIFIED
    transcript = _json(tmp_path / "transcript.json")
    assert [m["content"] for m in transcript["messages"]] == ["hello", "ok", "hang"]
    assert transcript["error"] == "interrupted (SIGINT or timeout) before the dialogue finished"
    assert "the dialogue did not finish (harness failure): interrupted" in capsys.readouterr().err
    if command == "run":
        assert _json(tmp_path / "simulhausen-result.json")["status"] == "warn"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["harbor"],
        ["harbor", "play"],
        ["harbor", "play", "--scenario", "s.json", "--target-json", "{}", "--target-file", "t.json"],
        ["harbor", "judge", "--scenario", "s.json", "--rewards", "fancy"],
    ],
)
def test_usage_errors_exit_1(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(argv)

    assert info.value.code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])

    assert info.value.code == 0
    assert capsys.readouterr().out == f"simulhausen {__version__}\n"


def test_python_dash_m_runs_the_same_command_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["simulhausen", "harbor", "judge", "--scenario", str(tmp_path / "none.toml")])

    with pytest.raises(SystemExit) as info:
        runpy.run_module("simulhausen", run_name="__main__")

    assert info.value.code == EXIT_USAGE


# -- judge ---------------------------------------------------------------------


def _play(scenario_file: Path, out: Path) -> Path:
    assert main(["harbor", "play", "--scenario", str(scenario_file), "--out", str(out)]) == EXIT_OK
    return out / "transcript.json"


def test_judge_writes_the_reward(
    scenario_file: Path, tmp_path: Path, helpers: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    transcript = _play(scenario_file, tmp_path / "agent")
    out = tmp_path / "verifier"
    argv = ["harbor", "judge", "--scenario", str(scenario_file), "--transcript", str(transcript), "--out", str(out)]

    code = main([*argv, "--rewards", "detailed"])

    assert code == EXIT_OK
    assert _json(out / "reward.json") == {
        "reward": 1,
        "criteria_pass_rate": 1.0,
        "criterion_1": 1,
        "check_1": 1,
    }
    assert sorted(path.name for path in out.iterdir()) == ["reward.json", "simulhausen-result.json", "transcript.md"]
    printed = capsys.readouterr().out
    assert "PASS refund\n" in printed
    assert "  criterion 1 passed: The agent is polite\n" in printed
    assert "  check 1 passed: tool 'lookup_order' was called\n" in printed
    assert 'reward.json: {"reward": 1, "criteria_pass_rate": 1.0, "criterion_1": 1, "check_1": 1}' in printed


def test_judge_fail_still_writes_a_reward(
    scenario_file: Path, tmp_path: Path, helpers: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    transcript = _play(scenario_file, tmp_path / "agent")
    helpers["statuses"] = {"The agent is polite": "failed"}
    argv = ["harbor", "judge", "--scenario", str(scenario_file), "--transcript", str(transcript)]

    code = main([*argv, "--out", str(tmp_path / "v")])

    assert code == EXIT_OK
    assert _json(tmp_path / "v" / "reward.json") == {"reward": 0}
    printed = capsys.readouterr().out
    assert "FAIL refund\n  scenario 'refund: duplicated charge' failed" in printed
    assert "  criterion 1 failed: The agent is polite" in printed


def test_judge_without_a_transcript_is_a_warn(
    scenario_file: Path, tmp_path: Path, helpers: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "verifier"
    out.mkdir()
    (out / "reward.json").write_text('{"reward": 1}')
    argv = ["harbor", "judge", "--scenario", str(scenario_file), "--transcript", str(tmp_path / "none.json")]

    skipped = main([*argv, "--out", str(out)])

    assert skipped == EXIT_UNVERIFIED
    assert not (out / "reward.json").exists()
    assert _json(out / "simulhausen-result.json")["status"] == "warn"
    printed = capsys.readouterr().out
    assert "WARN refund\n" in printed
    assert "no reward.json written: WARN is not a verdict" in printed
    assert helpers["built"] == []

    zero = main([*argv, "--out", str(out), "--warn-reward", "zero"])

    assert zero == EXIT_OK
    assert _json(out / "reward.json") == {"reward": 0}


def test_judge_refuses_a_scenario_with_nothing_to_grade(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_scenarios(tmp_path / "s.json", {**SCENARIO, "criteria": [], "checks": []})

    code = main(["harbor", "judge", "--scenario", str(path), "--out", str(tmp_path / "v")])

    assert code == EXIT_USAGE
    assert "nothing to grade" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["judge", "run"])
def test_a_configuration_error_leaves_no_old_reward_behind(
    scenario_file: Path, tmp_path: Path, helpers: dict[str, Any], command: str
) -> None:
    out = tmp_path / "verifier"
    assert main(["harbor", "run", "--scenario", str(scenario_file), "--out", str(out)]) == EXIT_OK
    assert (out / "reward.json").exists()
    broken = write_scenarios(tmp_path / "broken.json", {**SCENARIO, "critera": ["typo"]})

    code = main(["harbor", command, "--scenario", str(broken), "--out", str(out)])

    assert code == EXIT_USAGE
    left = sorted(path.name for path in out.iterdir())
    # run also plays, so it clears the old dialogue too; judge leaves what play wrote there.
    assert left == ([] if command == "run" else ["trajectory.json", "transcript.json", "transcript.md"])


def test_judge_writes_its_files_when_the_judge_breaks(
    scenario_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helpers: dict[str, Any]
) -> None:
    transcript = _play(scenario_file, tmp_path / "agent")
    monkeypatch.setattr(
        judge_module,
        "build_judge",
        lambda spec, model: ScriptedJudge(spec.criteria, errors={0: RuntimeError("connection reset")}),
    )
    out = tmp_path / "verifier"
    argv = ["harbor", "judge", "--scenario", str(scenario_file), "--transcript", str(transcript), "--out", str(out)]

    skipped = main(argv)
    zero = main([*argv, "--warn-reward", "zero"])

    assert (skipped, zero) == (EXIT_UNVERIFIED, EXIT_OK)
    result = _json(out / "simulhausen-result.json")
    assert result["status"] == "warn"
    assert "harness failure: RuntimeError('connection reset')" in result["reason"]
    assert _json(out / "reward.json") == {"reward": 0}


# -- run -----------------------------------------------------------------------


def test_run_plays_and_judges_in_one_go(scenario_file: Path, tmp_path: Path, helpers: dict[str, Any]) -> None:
    out = tmp_path / "verifier"

    code = main(["harbor", "run", "--scenario", str(scenario_file), "--out", str(out), "--model", "m:x"])

    assert code == EXIT_OK
    assert sorted(path.name for path in out.iterdir()) == [
        "reward.json",
        "simulhausen-result.json",
        "trajectory.json",
        "transcript.json",
        "transcript.md",
    ]
    assert _json(out / "reward.json") == {"reward": 1}
    # The judge is built before the dialogue, so a broken judge model costs no dialogue.
    assert helpers["built"] == [("judge", "m:x"), ("simulator", "m:x")]
    assert "## Outcome: PASS" in (out / "transcript.md").read_text()


def test_run_fails_fast_on_a_judge_model_problem(
    scenario_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["harbor", "run", "--scenario", str(scenario_file), "--out", str(tmp_path / "out")])

    assert code == EXIT_USAGE
    assert "no judge model" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# -- export --------------------------------------------------------------------


def test_export(scenario_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "tasks"

    code = main(["harbor", "export", str(scenario_file), "--output-dir", str(out), "--org", "acme"])

    assert code == EXIT_OK
    dockerfile = (out / "refund" / "environment" / "Dockerfile").read_text()
    assert dockerfile.split("--no-cache ")[1].startswith(f"simulhausen=={__version__} openai\n")
    assert '"OPENAI_API_KEY" = "${OPENAI_API_KEY}"' not in (out / "refund" / "task.toml").read_text()
    assert 'OPENAI_API_KEY = "${OPENAI_API_KEY}"' in (out / "refund" / "task.toml").read_text()
    printed = capsys.readouterr().out
    assert f"{out / 'refund'}\n" in printed
    # The scenario's target is a python factory, which only the generic agent runs.
    assert f"1 task(s) written; run them with: harbor run -p {out} -a simulhausen_harbor:SimulhausenAgent" in printed


def test_export_verifier_mode_with_options(
    scenario_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tasks"
    argv = ["harbor", "export", str(scenario_file), "--output-dir", str(out), "--mode", "verifier"]

    code = main([*argv, "--pip", "openai", "--pip", "anthropic", "--env", "AGENTOS_KEY", "--ids", "refund"])

    assert code == EXIT_OK
    task_toml = (out / "refund" / "task.toml").read_text()
    assert 'AGENTOS_KEY = "${AGENTOS_KEY}"' in task_toml
    assert "OPENAI_API_KEY" not in task_toml
    assert "openai anthropic\n" in (out / "refund" / "environment" / "Dockerfile").read_text()
    assert f"-p {out} -a nop" in capsys.readouterr().out


def test_export_flags_take_one_value_each(scenario_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "tasks"

    code = main(["harbor", "export", "--env", "AGENTOS_KEY", str(scenario_file), "--output-dir", str(out)])

    assert code == EXIT_OK
    assert 'AGENTOS_KEY = "${AGENTOS_KEY}"' in (out / "refund" / "task.toml").read_text()


@pytest.mark.parametrize(
    ("target", "hint"),
    [
        ({"type": "openai-chat", "base_url": "http://llm:8000/v1", "model": "m"}, "OpenAIChatAgent --ak base_url=..."),
        ({"type": "agentos", "base_url": "http://agent-api:7777", "agent_id": "a"}, "AgentOSAgent --ak base_url="),
        ({"type": "a2a", "url": "http://agent-api:7777/a2a"}, "A2AAgent --ak url=..."),
        ({"type": "openai-responses", "base_url": "http://llm:8000/v1", "model": "m"}, "SimulhausenAgent --ak target="),
        (None, "SimulhausenAgent --ak target=..."),
    ],
)
def test_export_hint_names_the_harbor_agent_for_the_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], target: dict[str, str] | None, hint: str
) -> None:
    scenario: dict[str, object] = {"name": "refund", "description": "d", "criteria": ["c"]}
    if target is not None:
        scenario["target"] = target
    scenario_file = tmp_path / "scenarios.json"
    scenario_file.write_text(json.dumps(scenario))

    code = main(["harbor", "export", str(scenario_file), "--output-dir", str(tmp_path / "tasks")])

    assert code == EXIT_OK
    assert f"-a simulhausen_harbor:{hint}" in capsys.readouterr().out


def test_files_are_utf8_with_lf_whatever_the_locale(tmp_path: Path) -> None:
    """Under an ASCII locale, without UTF-8 mode or locale coercion, Python's default encoding cannot hold
    what the commands write; every file must still come out as UTF-8 with LF line ends."""
    scenario = {
        "id": "unicode",
        "name": "ünïcode ✅",
        "description": "Größe 🙂\nzweite Zeile",
        "target": TARGET,
        "checks": [{"type": "reply_contains", "text": "größe"}],
        "script": [{"user": "Größe?"}, {"agent": True}],
    }
    path = write_scenarios(tmp_path / "s.json", scenario)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(Path(__file__).parent),
        "LC_ALL": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        # Only stdout and stderr: the files get the locale's default unless the code says otherwise.
        "PYTHONIOENCODING": "utf-8",
    }

    for argv in (
        ["run", "--scenario", str(path), "--out", str(tmp_path / "run")],
        ["export", str(path), "--output-dir", str(tmp_path / "tasks")],
    ):
        done = subprocess.run(
            [sys.executable, "-m", "simulhausen", "harbor", *argv],
            env=env,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        assert done.returncode == EXIT_OK, done.stderr

    written = [*(tmp_path / "run").iterdir(), *(p for p in (tmp_path / "tasks").rglob("*") if p.is_file())]
    assert len(written) == 11
    for file in written:
        data = file.read_bytes()
        data.decode("utf-8")
        assert b"\r" not in data, file
    assert "- ✅ 1. an agent reply contains 'größe'" in (tmp_path / "run" / "transcript.md").read_text(encoding="utf-8")
    task = tomllib.loads((tmp_path / "tasks" / "unicode" / "task.toml").read_text(encoding="utf-8"))
    assert task["task"]["description"] == "ünïcode ✅"
    instruction = (tmp_path / "tasks" / "unicode" / "instruction.md").read_bytes()
    assert instruction == "Größe 🙂\nzweite Zeile\n".encode()


def test_export_errors_exit_1(scenario_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["harbor", "export", str(scenario_file), "--output-dir", str(tmp_path), "--env", "BAD-NAME"])

    assert code == EXIT_USAGE
    assert "--env 'BAD-NAME' is not an environment variable name" in capsys.readouterr().err
