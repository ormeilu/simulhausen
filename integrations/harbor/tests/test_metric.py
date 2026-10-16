"""The uv-script metric: simulhausen's pass rate over Harbor rewards, in-process and the way Harbor runs it."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from harbor.metrics.uv_script import UvScript

from simulhausen_harbor import metric

METRIC_SCRIPT = Path(metric.__file__)


def test_pass_rate_excludes_warn_rows():
    result = metric.compute([{"reward": 1}, {"reward": 0}, None, {"reward": 1}, None, {"reward": 1.0}])
    assert result == {"pass_rate": 0.75, "n_pass": 3, "n_fail": 1, "n_warn": 2}


def test_rows_without_a_numeric_reward_count_as_warn():
    result = metric.compute([{}, {"reward": None}, {"reward": True}, {"reward": "1"}, [1], 7, {"reward": 0.5}])
    assert result == {"pass_rate": 0.0, "n_pass": 0, "n_fail": 1, "n_warn": 6}


def test_other_numeric_keys_are_averaged_over_the_rows_that_have_them():
    rewards = [
        {"reward": 1, "judge_score": 0.9, "criterion_1": 1, "criterion_2": 1},
        {"reward": 0, "judge_score": 0.5, "criterion_1": 0},  # criterion_2 inconclusive: key omitted
        None,
        {"reward": 1, "judge_score": float("nan"), "note": "text", "flag": True},
    ]
    result = metric.compute(rewards)
    assert result["pass_rate"] == pytest.approx(2 / 3)
    assert result["judge_score"] == pytest.approx(0.7)
    assert result["criterion_1"] == 0.5
    assert result["criterion_2"] == 1.0
    assert "note" not in result
    assert "flag" not in result
    assert "reward" not in result


def test_keys_clashing_with_the_summary_get_a_prefix():
    result = metric.compute([{"reward": 1, "pass_rate": 0.25, "n_warn": 3}])
    assert result["pass_rate"] == 1.0
    assert result["n_warn"] == 0
    assert result["mean_pass_rate"] == 0.25
    assert result["mean_n_warn"] == 3


def test_empty_input_reports_zero():
    assert metric.compute([]) == {"pass_rate": 0.0, "n_pass": 0, "n_fail": 0, "n_warn": 0}


def write_rewards(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_main_writes_the_output_file(tmp_path: Path):
    rewards = write_rewards(tmp_path / "rewards.jsonl", ['{"reward": 1}', "", "null", '{"reward": 0}'])
    output = tmp_path / "metric.json"

    assert metric.main(["-i", str(rewards), "-o", str(output)]) == 0

    assert json.loads(output.read_text()) == {"pass_rate": 0.5, "n_pass": 1, "n_fail": 1, "n_warn": 1}


def test_main_prints_without_an_output_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rewards = write_rewards(tmp_path / "rewards.jsonl", ['{"reward": 1, "judge_score": 0.8}'])

    assert metric.main(["--input-path", str(rewards)]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "pass_rate": 1.0,
        "n_pass": 1,
        "n_fail": 0,
        "n_warn": 0,
        "judge_score": 0.8,
    }


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "simulhausen_harbor.metric"],
        # -I: the script must not depend on the package (or anything next to it) being importable.
        [sys.executable, "-I", str(METRIC_SCRIPT)],
    ],
    ids=["python -m", "standalone script"],
)
def test_runs_as_a_command(tmp_path: Path, command: list[str]):
    rewards = write_rewards(tmp_path / "rewards.jsonl", ['{"reward": 1}', "null"])
    output = tmp_path / "metric.json"

    subprocess.run([*command, "-i", str(rewards), "-o", str(output)], check=True, cwd=tmp_path)

    assert json.loads(output.read_text()) == {"pass_rate": 1.0, "n_pass": 1, "n_fail": 0, "n_warn": 1}


@pytest.mark.skipif(shutil.which("uv") is None, reason="Harbor's uv-script metric shells out to uv")
def test_harbor_runs_it_as_a_uv_script(monkeypatch: pytest.MonkeyPatch):
    # Harbor's own UvScript metric: writes rewards.jsonl and runs `uv run <script> -i ... -o ...`.
    # The PEP 723 header declares no dependencies, so uv needs no index.
    monkeypatch.setenv("UV_OFFLINE", "1")

    result = UvScript(script_path=METRIC_SCRIPT).compute([{"reward": 1, "judge_score": 0.6}, None, {"reward": 0}])

    assert result == {"pass_rate": 0.5, "n_pass": 1, "n_fail": 1, "n_warn": 1, "judge_score": 0.6}
