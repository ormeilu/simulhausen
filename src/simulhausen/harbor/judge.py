"""Verifier: grade a recorded dialogue and turn the outcome into Harbor's ``reward.json``.

The judge phase rebuilds the :class:`~simulhausen.ScenarioState` from ``transcript.json``
and grades it with the same pieces a live :func:`simulhausen.run` uses:
:meth:`JudgeAgent.judge` for the scenario's criteria on the whole dialogue and for every
checkpoint's criteria on the dialogue up to that checkpoint, :meth:`JudgeAgent.score` when
``score_threshold`` is set, and the declarative checks. The outcome is an ordinary
:class:`~simulhausen.ScenarioResult`, so PASS, FAIL and WARN follow the usual rules.

Harbor knows only numbers, and only a reward file counts: the script's exit code is
ignored, and a missing ``reward.json`` is recorded as a verifier error. That is how WARN
stays out of the mean by default (``--warn-reward skip``), the way it stays out of
simulhausen's pass rate. Everything else, from the verdicts and their reasoning to the
times, goes to ``simulhausen-result.json`` next to it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

from simulhausen import __version__
from simulhausen.agents import JudgeAgent
from simulhausen.exceptions import SimulationHarnessError
from simulhausen.harbor.models import resolve_model
from simulhausen.harbor.paths import RESULT_FILE, REWARD_FILE, TRANSCRIPT_MARKDOWN_FILE
from simulhausen.harbor.spec import JudgeCheckpointSpec
from simulhausen.harbor.transcript import transcript_markdown, write_json
from simulhausen.reporting import ScenarioRecord, collector, serialize_records
from simulhausen.types import ScenarioResult, ScenarioStatus

if TYPE_CHECKING:
    from pathlib import Path

    from simulhausen.harbor.spec import ScenarioSpec
    from simulhausen.harbor.transcript import Transcript
    from simulhausen.types import CriterionVerdict, JudgeScore, ScenarioState

__all__ = [
    "RESULT_FORMAT",
    "REWARD_MODES",
    "WARN_REWARDS",
    "CheckOutcome",
    "GradedCriterion",
    "Judgement",
    "RewardMode",
    "WarnReward",
    "build_judge",
    "judge_transcript",
    "needs_judge",
    "prepare_judge",
    "remove_stale_reward",
    "unreadable_transcript",
    "write_judge_outputs",
]

RESULT_FORMAT: Final = "simulhausen-result/1"

# binary: {"reward": 1 | 0}, which keeps Harbor's mean and pass@k working.
# detailed: also the judge score, the criteria pass rate and one key per criterion and check.
type RewardMode = Literal["binary", "detailed"]
# What a WARN writes: skip = no reward.json (a verifier error in Harbor), zero = {"reward": 0}.
type WarnReward = Literal["skip", "zero"]
REWARD_MODES: Final[tuple[RewardMode, ...]] = ("binary", "detailed")
WARN_REWARDS: Final[tuple[WarnReward, ...]] = ("skip", "zero")

_MARK: Final = {"passed": "✅", "failed": "❌", "inconclusive": "❔"}


@dataclass(frozen=True)
class GradedCriterion:
    """A verdict with its place in the scenario file.

    ``number`` is 1-based in file order: the scenario's ``criteria`` first, then each
    checkpoint's criteria in script order. ``step`` is the checkpoint's script index,
    ``None`` for the scenario's own criteria.
    """

    number: int
    verdict: CriterionVerdict
    step: int | None = None


@dataclass(frozen=True)
class CheckOutcome:
    """A declarative check and whether it held; numbered like :class:`GradedCriterion`."""

    number: int
    type: str
    label: str
    passed: bool
    step: int | None = None


@dataclass
class Judgement:
    """Everything the judge phase found, and what it writes for Harbor."""

    spec: ScenarioSpec
    result: ScenarioResult
    criteria: list[GradedCriterion] = field(default_factory=list)
    checks: list[CheckOutcome] = field(default_factory=list)
    transcript: Transcript | None = None
    judge_time: float = 0.0

    @property
    def status(self) -> ScenarioStatus:
        return self.result.status

    def reward(self, mode: RewardMode = "binary", warn: WarnReward = "skip") -> dict[str, int | float] | None:
        """The ``reward.json`` payload; ``None`` for a WARN with ``warn="skip"``.

        ``reward`` is 1 for PASS and 0 otherwise. ``detailed`` adds ``judge_score`` (0-1,
        when scored), ``criteria_pass_rate`` (passed among the criteria that got a passed
        or failed verdict) and ``criterion_<n>`` / ``check_<n>`` (1 or 0). Inconclusive
        criteria get no key: Harbor takes only numbers, and 0 would call them failed.
        """
        status = self.status
        if status is ScenarioStatus.WARN and warn == "skip":
            return None
        payload: dict[str, int | float] = {"reward": 1 if status is ScenarioStatus.PASS else 0}
        if mode == "detailed":
            if self.result.judge_score is not None:
                payload["judge_score"] = round(self.result.judge_score.value, 4)
            judged = [graded for graded in self.criteria if not graded.verdict.inconclusive]
            if judged:
                payload["criteria_pass_rate"] = round(sum(g.verdict.passed for g in judged) / len(judged), 4)
            for graded in judged:
                payload[f"criterion_{graded.number}"] = int(graded.verdict.passed)
            for check in self.checks:
                payload[f"check_{check.number}"] = int(check.passed)
        return payload

    def to_json_dict(self, reward: dict[str, int | float] | None) -> dict[str, Any]:
        """``simulhausen-result.json``: the outcome with its reasons, and the reward written."""
        result = self.result
        return {
            "format": RESULT_FORMAT,
            "simulhausen_version": __version__,
            "scenario": {"id": self.spec.id, "name": self.spec.name, "description": self.spec.description},
            "status": result.status.value,
            "reason": (result.reasoning or "") if result.success else result.failure_summary(),
            "reward": reward,
            "verdicts": [
                {"number": graded.number, "step": graded.step, **graded.verdict.model_dump(mode="json")}
                for graded in self.criteria
            ],
            "score": result.judge_score.model_dump(mode="json") if result.judge_score else None,
            "checks": [asdict(check) for check in self.checks],
            "turns": result.turns,
            "stop_reason": self.transcript.stop_reason if self.transcript else None,
            "total_time": result.total_time,
            "agent_time": result.agent_time,
            "judge_time": self.judge_time,
            "record": serialize_records([ScenarioRecord(result=result)])[0],
        }

    def markdown(self) -> str:
        """The outcome as markdown, for the top of the verifier's ``transcript.md``."""
        result = self.result
        lines = [f"## Outcome: {result.status.value.upper()}", ""]
        if reason := (result.reasoning if result.success else result.failure_summary()):
            lines += [reason, ""]
        if self.criteria:
            lines.append("### Criteria")
            for graded in self.criteria:
                where = f" (checkpoint script[{graded.step}])" if graded.step is not None else ""
                verdict = graded.verdict
                lines.append(
                    f"- {_MARK[verdict.status]} {graded.number}. {verdict.criterion}{where}: {verdict.reasoning}"
                )
            lines.append("")
        if (score := result.judge_score) is not None:
            mark = _MARK["passed" if score.passed else "failed"]
            summary = f"{mark} {score.raw_score}/10, threshold {score.threshold}: {score.reasoning or '-'}"
            lines += ["### Judge score", "", summary, ""]
        if self.checks:
            lines.append("### Checks")
            for check in self.checks:
                where = f" (script[{check.step}])" if check.step is not None else ""
                lines.append(f"- {_MARK['passed' if check.passed else 'failed']} {check.number}. {check.label}{where}")
        return "\n".join(lines).rstrip() + "\n"


def build_judge(spec: ScenarioSpec, model: str | None) -> JudgeAgent:
    """The judge for ``spec``, on the model ``--model`` or the environment names.

    Module-level so tests (and embedders) can swap in a judge that needs no model.
    """
    return JudgeAgent(
        spec.criteria,
        resolve_model("judge", model, spec.judge_model),
        score_threshold=spec.score_threshold,
    )


def _mismatch(spec: ScenarioSpec, transcript: Transcript) -> str | None:
    """Why ``transcript`` cannot be graded against ``spec``; ``None`` when it can."""
    if transcript.scenario.id != spec.id:
        return f"the transcript is of scenario {transcript.scenario.id!r}, not {spec.id!r}"
    if not transcript.finished:
        # An interrupted dialogue never reached its later checkpoints; it is not graded anyway.
        return None
    expected = [
        (index, "judge" if isinstance(step.judge, JudgeCheckpointSpec) else "check")
        for index, step in enumerate(spec.steps)
        if step.is_checkpoint
    ]
    recorded = [(checkpoint.step, checkpoint.kind) for checkpoint in transcript.checkpoints]
    if recorded != expected:
        return (
            f"the transcript has checkpoints at {recorded}, the scenario file at {expected} "
            "(script index, kind): it was played from another version of the scenario"
        )
    return None


def _checks(spec: ScenarioSpec, transcript: Transcript, state: ScenarioState) -> list[CheckOutcome]:
    outcomes = [
        CheckOutcome(number, check.type, check.describe(), check.evaluate(state))
        for number, check in enumerate(spec.checks, 1)
    ]
    reached = {checkpoint.step: checkpoint.message_count for checkpoint in transcript.checkpoints}
    for index, step in enumerate(spec.steps):
        if step.check is not None:
            prefix = transcript.to_state(reached[index])
            outcomes.append(
                CheckOutcome(
                    len(outcomes) + 1, step.check.type, step.check.describe(), step.check.evaluate(prefix), index
                )
            )
    return outcomes


class _Grading:
    """The judge's verdicts and score, collected in file order until the harness breaks."""

    def __init__(self) -> None:
        self.criteria: list[GradedCriterion] = []
        self.score: JudgeScore | None = None
        self.summaries: list[str] = []

    def add(self, verdicts: list[CriterionVerdict], summary: str, step: int | None = None) -> None:
        start = len(self.criteria) + 1
        self.criteria += [GradedCriterion(start + offset, verdict, step) for offset, verdict in enumerate(verdicts)]
        self.summaries.append(summary)

    async def run(self, judge: JudgeAgent, spec: ScenarioSpec, transcript: Transcript, state: ScenarioState) -> None:
        if spec.criteria:
            final = await judge.judge(state, None, additional_context=spec.additional_context)
            self.add(final.verdicts, final.summary)
        reached = {checkpoint.step: checkpoint.message_count for checkpoint in transcript.checkpoints}
        for index, step in enumerate(spec.steps):
            if isinstance(step.judge, JudgeCheckpointSpec):
                prefix = transcript.to_state(reached[index])
                verdict = await judge.judge(
                    prefix, step.judge.criteria, additional_context=step.judge.additional_context
                )
                self.add(verdict.verdicts, verdict.summary, index)
        if spec.score_threshold is not None:
            self.score = await judge.score(state)


def needs_judge(spec: ScenarioSpec) -> bool:
    """Whether grading ``spec`` asks the judge anything (criteria or judge checkpoints)."""
    return bool(spec.criteria) or any(isinstance(step.judge, JudgeCheckpointSpec) for step in spec.steps)


def prepare_judge(spec: ScenarioSpec, model: str | None) -> JudgeAgent | None:
    """The judge grading ``spec`` will need, built now; ``None`` when only checks are graded.

    ``run`` calls it before the dialogue, so a missing model or SDK fails before anything
    is spent on playing.
    """
    return build_judge(spec, model) if needs_judge(spec) else None


def _conclude(
    spec: ScenarioSpec,
    transcript: Transcript | None,
    started: float,
    *,
    grading: _Grading | None = None,
    checks: list[CheckOutcome] | None = None,
    failure: str | None = None,
    harness: str | None = None,
) -> Judgement:
    """Build the :class:`ScenarioResult` from what was found and report it to the collector."""
    grading = grading or _Grading()
    checks = checks or []
    verdicts = [graded.verdict for graded in grading.criteria]
    reasons = [failure] if failure else []
    reasons += [f"check failed: {check.label}" for check in checks if not check.passed]
    score_ok = grading.score is None or grading.score.passed
    # A failure found before the harness broke is certain, so it stays a FAIL; without one
    # the scenario could not be verified (WARN). A live run behaves alike: a failed
    # checkpoint ends it before a later judge call can break.
    certain_failure = bool(reasons) or any(v.failed for v in verdicts) or not score_ok
    if harness:
        reasons.append(harness)
    failure_reason = "; ".join(reasons) or None
    judge_time = time.perf_counter() - started
    played = transcript.total_time if transcript else 0.0
    result = ScenarioResult(
        name=spec.name,
        description=spec.description,
        success=failure_reason is None and all(v.passed for v in verdicts) and score_ok,
        verdicts=verdicts,
        judge_score=grading.score,
        messages=[{"role": m.role, "content": m.content} for m in transcript.messages] if transcript else [],
        tool_calls=transcript.to_state().tool_call_names() if transcript else [],
        turns=transcript.turns if transcript else 0,
        reasoning=grading.summaries[0] if grading.summaries else None,
        failure_reason=failure_reason,
        harness_error=harness is not None and not certain_failure,
        total_time=played + judge_time,
        agent_time=transcript.agent_time if transcript else 0.0,
    )
    collector.add(result)
    return Judgement(spec, result, grading.criteria, checks, transcript, judge_time)


async def judge_transcript(
    spec: ScenarioSpec,
    transcript: Transcript,
    *,
    model: str | None = None,
    judge: JudgeAgent | None = None,
) -> Judgement:
    """Grade ``transcript`` against ``spec``'s criteria, checkpoints, score and checks.

    The judge is built (on ``model`` or the model the environment or the scenario names)
    only when there are criteria to grade; pass ``judge`` to supply your own. A dialogue
    the harness broke is WARN without asking the judge, one the agent under test broke
    (a runtime error) is FAIL, and a judge that fails to answer
    (:class:`~simulhausen.SimulationHarnessError`) makes the scenario WARN unless a
    failure was already certain. The result is also added to :data:`simulhausen.collector`.
    """
    spec.require_gradable()
    started = time.perf_counter()
    if problem := _mismatch(spec, transcript):
        return _conclude(spec, transcript, started, harness=problem)
    if transcript.harness_error:
        return _conclude(spec, transcript, started, harness=f"the dialogue did not finish: {transcript.error}")
    if transcript.stop_reason == "error":
        return _conclude(spec, transcript, started, failure=f"runtime error: {transcript.error}")
    state = transcript.to_state()
    checks = _checks(spec, transcript, state)
    grading = _Grading()
    harness: str | None = None
    judge = judge or prepare_judge(spec, model)
    if judge is not None:
        try:
            await grading.run(judge, spec, transcript, state)
        except SimulationHarnessError as exc:
            harness = f"harness failure: {exc!r}"
    return _conclude(spec, transcript, started, grading=grading, checks=checks, harness=harness)


def unreadable_transcript(spec: ScenarioSpec, reason: str) -> Judgement:
    """The WARN judgement for a transcript that is missing or cannot be read."""
    return _conclude(spec, None, time.perf_counter(), harness=reason)


def remove_stale_reward(out_dir: Path) -> None:
    """Delete a ``reward.json`` left from an earlier run, so a WARN never shows an old reward."""
    (out_dir / REWARD_FILE).unlink(missing_ok=True)


def write_judge_outputs(
    out_dir: Path, judgement: Judgement, mode: RewardMode = "binary", warn: WarnReward = "skip"
) -> dict[str, int | float] | None:
    """Write ``simulhausen-result.json``, ``transcript.md`` and, last, ``reward.json``.

    Returns the reward written, ``None`` when a WARN wrote none.
    """
    remove_stale_reward(out_dir)
    reward = judgement.reward(mode, warn)
    write_json(out_dir / RESULT_FILE, judgement.to_json_dict(reward))
    if judgement.transcript is not None:
        markdown = transcript_markdown(judgement.transcript, judgement.markdown())
        (out_dir / TRANSCRIPT_MARKDOWN_FILE).write_text(markdown)
    if reward is not None:
        # Last: Harbor reads this file as the verdict.
        write_json(out_dir / REWARD_FILE, reward)
    return reward
