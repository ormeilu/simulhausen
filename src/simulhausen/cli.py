"""The ``simulhausen`` command line (also ``python -m simulhausen``).

So far it holds one command group, ``simulhausen harbor``, which runs scenarios inside
Harbor task containers (see :mod:`simulhausen.harbor`). The exit codes tell apart what
Harbor and the Harbor agents need to know:

* 0: done; for ``judge`` and ``run``, a reward was written;
* 1: a usage or configuration error, nothing was played or graded;
* 2: the scenario could not be verified: ``play`` hit a harness failure (the transcript
  is still written), or ``judge``/``run`` came to WARN and wrote no reward.

An error of the agent under test is not a harness failure: ``play`` exits 0 and the judge
grades it FAIL. Harbor records a failed agent phase as an errored trial and retries those
under ``--max-retries``, which would replace a real FAIL with whichever attempt passes.

Harbor itself ignores the verifier's exit code (only ``reward.json`` counts), so these
codes are for people reading the logs and for the agent that runs ``play``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn

from simulhausen import __version__
from simulhausen.harbor.errors import ConfigError
from simulhausen.harbor.export import DEFAULT_ENV, DEFAULT_ORG, DEFAULT_PIP, EXPORT_MODES, ExportOptions, export_tasks
from simulhausen.harbor.judge import (
    REWARD_MODES,
    WARN_REWARDS,
    judge_transcript,
    prepare_judge,
    remove_stale_results,
    unreadable_transcript,
    write_judge_outputs,
)
from simulhausen.harbor.paths import AGENT_LOGS_DIR, DEFAULT_TRANSCRIPT, REWARD_FILE, TRANSCRIPT_FILE, VERIFIER_LOGS_DIR
from simulhausen.harbor.play import play
from simulhausen.harbor.spec import load_scenarios, select_scenario
from simulhausen.harbor.targets import resolve_target_config
from simulhausen.harbor.transcript import TranscriptError, read_transcript, remove_play_outputs, write_play_outputs

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from simulhausen.harbor.judge import Judgement
    from simulhausen.harbor.spec import ScenarioSpec
    from simulhausen.harbor.transcript import Transcript

__all__ = ["EXIT_OK", "EXIT_UNVERIFIED", "EXIT_USAGE", "build_parser", "main"]

EXIT_OK: Final = 0
EXIT_USAGE: Final = 1
EXIT_UNVERIFIED: Final = 2


class _Parser(argparse.ArgumentParser):
    """argparse exits with 2 on a usage error; here 2 means "not verified", so use 1."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


# -- commands ------------------------------------------------------------------


def _load(args: argparse.Namespace) -> ScenarioSpec:
    return select_scenario(load_scenarios(args.scenario), args.id, args.scenario)


def _report_play(out: Path, transcript: Transcript) -> None:
    print(
        f"{transcript.scenario.id}: {transcript.turns} turn(s), stop reason {transcript.stop_reason}; "
        f"transcript written to {out / TRANSCRIPT_FILE}"
    )
    if not transcript.finished:
        kind = "harness failure" if transcript.harness_error else "error"
        print(f"the dialogue did not finish ({kind}): {transcript.error}", file=sys.stderr)


def _cmd_play(args: argparse.Namespace) -> int:
    # First, so that even a configuration error leaves no old dialogue for a later judge to grade as new.
    remove_play_outputs(args.out)
    spec = _load(args)
    target = resolve_target_config(spec, target_json=args.target_json, target_file=args.target_file)
    transcript = asyncio.run(
        play(
            spec,
            target,
            model=args.model,
            on_progress=lambda partial: write_play_outputs(args.out, partial),
            keep_dialogue_on_cancel=True,
        )
    )
    write_play_outputs(args.out, transcript)
    _report_play(args.out, transcript)
    # Only a harness failure is a failed agent phase; an agent's own error is graded FAIL (see above).
    return EXIT_UNVERIFIED if transcript.harness_error else EXIT_OK


def _finish_judgement(args: argparse.Namespace, judgement: Judgement) -> int:
    reward = write_judge_outputs(args.out, judgement, args.rewards, args.warn_reward)
    result = judgement.result
    print(f"{result.status.value.upper()} {judgement.spec.id}")
    if not result.success:
        print(f"  {result.failure_summary()}")
    for graded in judgement.criteria:
        print(f"  criterion {graded.number} {graded.verdict.status}: {graded.verdict.criterion}")
    for check in judgement.checks:
        print(f"  check {check.number} {'passed' if check.passed else 'failed'}: {check.label}")
    if reward is None:
        print(f"no {REWARD_FILE} written: WARN is not a verdict (--warn-reward zero writes a reward of 0)")
        return EXIT_UNVERIFIED
    print(f"{REWARD_FILE}: {json.dumps(reward)}")
    return EXIT_OK


def _cmd_judge(args: argparse.Namespace) -> int:
    # First, so that even a configuration error (exit 1) leaves no old reward behind.
    remove_stale_results(args.out)
    spec = _load(args)
    spec.require_gradable()
    try:
        transcript = read_transcript(args.transcript)
    except TranscriptError as exc:
        judgement = unreadable_transcript(spec, str(exc))
    else:
        judgement = asyncio.run(judge_transcript(spec, transcript, model=args.model))
    return _finish_judgement(args, judgement)


def _cmd_run(args: argparse.Namespace) -> int:
    # First, so that even a configuration error (exit 1) leaves no old reward or dialogue behind.
    remove_stale_results(args.out)
    remove_play_outputs(args.out)
    spec = _load(args)
    spec.require_gradable()
    target = resolve_target_config(spec, target_json=args.target_json, target_file=args.target_file)
    judge = prepare_judge(spec, args.model)

    async def play_and_judge() -> Judgement:
        transcript = await play(spec, target, model=args.model, keep_dialogue_on_cancel=True)
        write_play_outputs(args.out, transcript)
        _report_play(args.out, transcript)
        return await judge_transcript(spec, transcript, judge=judge)

    return _finish_judgement(args, asyncio.run(play_and_judge()))


def _cmd_export(args: argparse.Namespace) -> int:
    options = ExportOptions(
        output_dir=args.output_dir,
        mode=args.mode,
        org=args.org,
        simulhausen_spec=args.simulhausen_spec,
        pip=args.pip or DEFAULT_PIP,
        env=args.env or DEFAULT_ENV,
        compose=args.compose,
        agent_timeout=args.agent_timeout,
        verifier_timeout=args.verifier_timeout,
        ids=args.ids,
        overwrite=args.overwrite,
        rewards=args.rewards,
        warn_reward=args.warn_reward,
    )
    written = export_tasks(args.scenario_files, options)
    for task_dir in written:
        print(task_dir)
    agent = _harbor_agent_hint(written) if options.mode == "agent" else "nop"
    print(f"{len(written)} task(s) written; run them with: harbor run -p {options.output_dir} -a {agent}")
    return EXIT_OK


# The Harbor agent (simulhausen.harbor.agents) for each target type, with its required options;
# any other type (or a mix) runs with the generic SimulhausenAgent.
_HARBOR_AGENTS: Final = {
    "agentos": "AgentOSAgent --ak base_url=...",
    "openai-chat": "OpenAIChatAgent --ak base_url=...",
    "a2a": "A2AAgent --ak url=...",
    "langgraph": "LangGraphAgent --ak base_url=... --ak assistant_id=...",
}


def _harbor_agent_hint(task_dirs: list[Path]) -> str:
    """The Harbor agent for agent-mode tasks, picked by the target type their scenarios name."""
    kinds = {(load_scenarios(task / "environment" / "scenario.json")[0].target or {}).get("type") for task in task_dirs}
    kind = kinds.pop() if len(kinds) == 1 else None
    agent = _HARBOR_AGENTS.get(kind) if isinstance(kind, str) else None
    return f"simulhausen.harbor:{agent or 'SimulhausenAgent --ak target=...'}"


# -- parser --------------------------------------------------------------------


def _scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", required=True, type=Path, metavar="PATH", help="scenario file (.toml or .json)")
    parser.add_argument("--id", metavar="ID", help="scenario to use when the file holds several")
    parser.add_argument("--model", metavar="M", help="agno model string for the simulator and the judge")


def _target_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target-json", metavar="JSON", help="the agent under test, as a JSON target object")
    group.add_argument("--target-file", type=Path, metavar="PATH", help="a JSON file with the target object")


def _reward_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rewards", choices=REWARD_MODES, default="binary", help="reward.json shape (default: %(default)s)"
    )
    parser.add_argument(
        "--warn-reward",
        choices=WARN_REWARDS,
        default="skip",
        help="what a WARN writes: no reward.json, or a reward of 0 (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="simulhausen", description="simulhausen: simulation testing for agno agents.")
    parser.add_argument("--version", action="version", version=f"simulhausen {__version__}")
    commands = parser.add_subparsers(metavar="COMMAND", required=True)
    harbor = commands.add_parser(
        "harbor",
        help="run scenarios as Harbor tasks",
        description="Run simulhausen scenarios inside Harbor task containers, or export them as Harbor tasks.",
    )
    harbor_commands = harbor.add_subparsers(metavar="COMMAND", required=True)

    play_parser = harbor_commands.add_parser("play", help="play the dialogue with the agent under test (agent phase)")
    _scenario_arguments(play_parser)
    _target_arguments(play_parser)
    play_parser.add_argument("--out", type=Path, default=Path(AGENT_LOGS_DIR), metavar="DIR")
    play_parser.set_defaults(handler=_cmd_play)

    judge_parser = harbor_commands.add_parser("judge", help="grade a recorded dialogue and write reward.json")
    _scenario_arguments(judge_parser)
    judge_parser.add_argument("--transcript", type=Path, default=Path(DEFAULT_TRANSCRIPT), metavar="PATH")
    judge_parser.add_argument("--out", type=Path, default=Path(VERIFIER_LOGS_DIR), metavar="DIR")
    _reward_arguments(judge_parser)
    judge_parser.set_defaults(handler=_cmd_judge)

    run_parser = harbor_commands.add_parser("run", help="play and grade in one go (verifier-only tasks)")
    _scenario_arguments(run_parser)
    _target_arguments(run_parser)
    run_parser.add_argument("--out", type=Path, default=Path(VERIFIER_LOGS_DIR), metavar="DIR")
    _reward_arguments(run_parser)
    run_parser.set_defaults(handler=_cmd_run)

    export_parser = harbor_commands.add_parser("export", help="render scenario files into Harbor task directories")
    export_parser.add_argument("scenario_files", nargs="+", type=Path, metavar="SCENARIO_FILE")
    export_parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR")
    export_parser.add_argument(
        "--mode",
        choices=EXPORT_MODES,
        default="agent",
        help="agent: a Harbor agent plays, the verifier judges; verifier: the verifier does both "
        "(default: %(default)s)",
    )
    export_parser.add_argument(
        "--org", default=DEFAULT_ORG, help="Harbor task names are ORG/<id> (default: %(default)s)"
    )
    export_parser.add_argument("--simulhausen-spec", metavar="REQ", help="pip requirement for simulhausen in the image")
    # One value per flag, repeated: a flag taking several values would swallow the scenario files after it.
    export_parser.add_argument(
        "--pip", action="append", metavar="REQ", help="extra requirement for the image, repeatable (default: openai)"
    )
    export_parser.add_argument(
        "--env",
        action="append",
        metavar="NAME",
        help="host variable the verifier gets, repeatable (default: OPENAI_API_KEY)",
    )
    export_parser.add_argument("--compose", type=Path, metavar="FILE", help="docker-compose.yaml for the task")
    export_parser.add_argument("--agent-timeout", type=float, metavar="SEC")
    export_parser.add_argument("--verifier-timeout", type=float, metavar="SEC")
    export_parser.add_argument("--ids", action="append", metavar="ID", help="export only this scenario, repeatable")
    export_parser.add_argument("--overwrite", action="store_true", help="replace task directories export wrote before")
    _reward_arguments(export_parser)
    export_parser.set_defaults(handler=_cmd_export)
    return parser


@contextlib.contextmanager
def _console_logging() -> Generator[None]:
    """Log simulhausen's progress (every message, every warning) to stderr while a command runs."""
    package_logger = logging.getLogger("simulhausen")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    previous_level = package_logger.level
    package_logger.addHandler(handler)
    if package_logger.getEffectiveLevel() > logging.INFO:
        package_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(previous_level)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code (see the module docstring)."""
    args = build_parser().parse_args(argv)
    with _console_logging():
        try:
            return args.handler(args)
        except ConfigError as exc:
            print(f"simulhausen: error: {exc}", file=sys.stderr)
            return EXIT_USAGE
