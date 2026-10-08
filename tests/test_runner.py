import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from agno.models.message import MessageReferences
from agno.models.response import ToolExecution

import simulhausen as sim
from simulhausen import (
    AgentAdapter,
    AgentInput,
    AgentTurn,
    CriterionStatus,
    CriterionVerdict,
    JudgeAgent,
    JudgeScore,
    JudgeVerdict,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    UserSimulatorAgent,
    UserTurn,
    collector,
)
from simulhausen.script import ScriptStep

type Round = CriterionStatus | list[CriterionStatus]


class ScriptedUser(UserSimulatorAgent):
    """Replays the given turns (plain strings are ordinary messages), then invents "user message N"."""

    def __init__(self, *turns: UserTurn | str) -> None:
        super().__init__()
        self.turns = [UserTurn(message=t, goal_achieved=False) if isinstance(t, str) else t for t in turns]
        self.transcript_sizes: list[int] = []

    async def next_turn(self, state: ScenarioState) -> UserTurn:
        self.transcript_sizes.append(len(state.messages))
        if self.turns:
            return self.turns.pop(0)
        return UserTurn(message=f"user message {len(self.transcript_sizes)}", goal_achieved=False)


class ReplyAdapter(AgentAdapter):
    """Answers with the given replies, then "agent reply N", and remembers every input."""

    def __init__(self, *replies: str | AgentTurn, delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.delay = delay
        self.inputs: list[AgentInput] = []

    async def call(self, input: AgentInput) -> str | AgentTurn:
        self.inputs.append(input)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.replies.pop(0) if self.replies else f"agent reply {len(self.inputs)}"


@dataclass
class JudgeCall:
    criteria: list[str]
    additional_context: str | None
    transcript_size: int


class FakeJudge(JudgeAgent):
    """A judge that answers from a script instead of a model.

    ``rounds`` holds one entry per ``judge`` call (the last repeats): a status for
    all criteria, or one status per criterion. ``decisions`` and ``grades`` do the
    same for ``decide`` and ``score``.
    """

    def __init__(
        self,
        criteria: Sequence[str] = ("is polite",),
        *,
        rounds: Sequence[Round] = ("passed",),
        decisions: Sequence[bool] = (False,),
        grades: Sequence[JudgeScore | None] = (None,),
        summary: str = "judge summary",
    ) -> None:
        super().__init__(criteria)
        self.rounds = list(rounds)
        self.decisions = list(decisions)
        self.grades = list(grades)
        self.summary = summary
        self.judge_calls: list[JudgeCall] = []
        self.decide_sizes: list[int] = []
        self.score_calls = 0

    @staticmethod
    def _next[T](queue: list[T]) -> T:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def judge(
        self,
        state: ScenarioState,
        criteria: Sequence[str] | None = None,
        *,
        additional_context: str | None = None,
    ) -> JudgeVerdict:
        used = list(self.criteria if criteria is None else criteria)
        self.judge_calls.append(JudgeCall(used, additional_context, len(state.messages)))
        spec = self._next(self.rounds)
        statuses = list(spec) if isinstance(spec, list) else [spec] * len(used)
        verdicts = [
            CriterionVerdict(criterion=c, reasoning=f"{s} reasoning", status=s)
            for c, s in zip(used, statuses, strict=True)
        ]
        return JudgeVerdict(verdicts=verdicts, summary=self.summary)

    async def decide(self, state: ScenarioState) -> bool:
        self.decide_sizes.append(len(state.messages))
        return self._next(self.decisions)

    async def score(self, state: ScenarioState) -> JudgeScore | None:
        self.score_calls += 1
        return self._next(self.grades)


async def play(*agents: object, script: Sequence[ScriptStep] | None = None, **kwargs: Any) -> ScenarioResult:
    return await sim.run("scenario", "A user tests the agent.", list(agents), script, **kwargs)


def _verdict_pairs(result: ScenarioResult) -> list[tuple[str, str]]:
    return [(v.criterion, v.status) for v in result.verdicts]


def _roles(result: ScenarioResult) -> list[str]:
    return [m["role"] for m in result.messages]


# -- auto mode and proceed ----------------------------------------------------


async def test_auto_mode_proceeds_until_max_turns_then_judges(caplog: pytest.LogCaptureFixture) -> None:
    adapter, user, judge = ReplyAdapter(), ScriptedUser(), FakeJudge()

    with caplog.at_level(logging.WARNING, logger="simulhausen.runner"):
        result = await play(adapter, user, judge, max_turns=3)

    assert result.success
    assert result.turns == 3
    assert _roles(result) == ["user", "assistant"] * 3
    assert result.messages[0]["content"] == "user message 1"
    assert len(judge.judge_calls) == 1
    assert judge.judge_calls[0].transcript_size == 6
    assert _verdict_pairs(result) == [("is polite", "passed")]
    assert result.reasoning == "judge summary"
    assert caplog.records == []


async def test_auto_mode_without_a_judge_passes_without_verdicts() -> None:
    result = await play(ReplyAdapter(), ScriptedUser(), max_turns=2)

    assert result.success
    assert result.verdicts == []
    assert result.turns == 2


async def test_proceed_with_turns_stops_early() -> None:
    adapter = ReplyAdapter()

    result = await play(adapter, ScriptedUser(), script=[sim.proceed(turns=2)], max_turns=10)

    assert result.turns == 2
    assert len(adapter.inputs) == 2


async def test_proceed_stops_at_max_turns_across_scripted_turns() -> None:
    result = await play(
        ReplyAdapter(),
        ScriptedUser(),
        script=[sim.user("opening"), sim.agent(), sim.proceed(turns=5)],
        max_turns=3,
    )

    assert result.turns == 3
    assert result.messages[0]["content"] == "opening"


async def test_two_proceed_steps_share_the_turn_budget() -> None:
    result = await play(
        ReplyAdapter(), ScriptedUser(), script=[sim.proceed(turns=2), sim.proceed(turns=2)], max_turns=3
    )

    assert result.turns == 3


async def test_proceed_stops_when_the_goal_is_achieved_after_an_agent_turn() -> None:
    adapter = ReplyAdapter()
    user = ScriptedUser("order status?", UserTurn(message="", goal_achieved=True))
    judge = FakeJudge()

    result = await play(adapter, user, judge, max_turns=10)

    assert result.turns == 1
    assert len(adapter.inputs) == 1
    assert result.success
    assert len(judge.judge_calls) == 1, "the dialogue is judged as it stands"


async def test_goal_achieved_is_ignored_before_the_first_agent_turn() -> None:
    adapter = ReplyAdapter()
    user = ScriptedUser(UserTurn(message="one and done", goal_achieved=True))

    result = await play(adapter, user, max_turns=10, script=[sim.proceed(turns=1)])

    assert result.turns == 1
    assert result.messages[0]["content"] == "one and done"


async def test_empty_user_turn_is_a_harness_failure() -> None:
    user = ScriptedUser(UserTurn(message="", goal_achieved=False))

    with pytest.raises(sim.SimulationHarnessError, match="empty message"):
        await play(ReplyAdapter(), user)

    assert collector.records[0].result.status is ScenarioStatus.WARN


async def test_proceed_stops_when_the_judge_concludes() -> None:
    grade = JudgeScore(value=1.0, raw_score=10, threshold=7, passed=True)
    judge = FakeJudge(decisions=[False, True], grades=[grade])

    result = await play(ReplyAdapter(), ScriptedUser(), judge, max_turns=10)

    assert result.turns == 2
    assert judge.decide_sizes == [2, 4]
    assert len(judge.judge_calls) == 1, "a conclusion is judged once, not again at the end"
    assert judge.score_calls == 1
    assert result.judge_score == grade
    assert result.success


async def test_judge_conclusion_waits_for_min_turns() -> None:
    judge = FakeJudge(decisions=[True])

    result = await play(ReplyAdapter(), ScriptedUser(), judge, max_turns=10, min_turns=3)

    assert result.turns == 3
    assert judge.decide_sizes == [6]


async def test_min_turns_comes_from_the_config() -> None:
    sim.configure(min_turns=2)
    judge = FakeJudge(decisions=[True])

    result = await play(ReplyAdapter(), ScriptedUser(), judge)

    assert result.turns == 2


async def test_a_judge_without_criteria_is_never_asked_to_decide() -> None:
    judge = FakeJudge(criteria=(), decisions=[True])

    result = await play(ReplyAdapter(), ScriptedUser(), judge, max_turns=2)

    assert result.turns == 2
    assert judge.decide_sizes == []
    assert judge.judge_calls == []


async def test_inconclusive_conclusion_continues_while_turns_remain() -> None:
    judge = FakeJudge(rounds=["inconclusive", "passed"], decisions=[True])

    result = await play(ReplyAdapter(), ScriptedUser(), judge, max_turns=5)

    assert result.turns == 2
    assert [call.transcript_size for call in judge.judge_calls] == [2, 4]
    assert _verdict_pairs(result) == [("is polite", "passed")]
    assert result.status is ScenarioStatus.PASS


async def test_inconclusive_conclusion_is_final_on_the_last_turn() -> None:
    judge = FakeJudge(rounds=["inconclusive"], decisions=[True])

    result = await play(ReplyAdapter(), ScriptedUser(), judge, max_turns=2)

    assert result.turns == 2
    assert len(judge.judge_calls) == 2
    assert _verdict_pairs(result) == [("is polite", "inconclusive")]
    assert not result.success
    assert result.status is ScenarioStatus.WARN


async def test_proceed_callbacks_see_every_step_and_turn() -> None:
    steps: list[str] = []
    turns: list[int] = []

    def on_step(state: ScenarioState) -> None:
        steps.append(state.last_message()["role"])

    async def on_turn(state: ScenarioState) -> None:
        await asyncio.sleep(0)
        turns.append(state.current_turn)

    await play(ReplyAdapter(), ScriptedUser(), script=[sim.proceed(2, on_step=on_step, on_turn=on_turn)])

    assert steps == ["user", "assistant", "user", "assistant"]
    assert turns == [1, 2]


async def test_async_on_step_is_awaited() -> None:
    seen: list[int] = []

    async def on_step(state: ScenarioState) -> None:
        await asyncio.sleep(0)
        seen.append(len(state.messages))

    await play(ReplyAdapter(), ScriptedUser(), script=[sim.proceed(1, on_step=on_step)])

    assert seen == [1, 2]


# -- script steps -------------------------------------------------------------


async def test_message_and_fixed_agent_reply_do_not_call_the_adapter() -> None:
    adapter = ReplyAdapter()

    result = await play(
        adapter,
        script=[
            sim.message("user", "earlier question"),
            sim.message("assistant", "earlier answer"),
            sim.user("now"),
            sim.agent("fixed"),
        ],
    )

    assert adapter.inputs == []
    assert [m["content"] for m in result.messages] == ["earlier question", "earlier answer", "now", "fixed"]
    assert result.turns == 1, "only agent() steps count as turns, message() does not"


async def test_adapter_input_carries_the_thread_state_and_new_messages_only() -> None:
    adapter = ReplyAdapter()

    await play(
        adapter,
        script=[
            sim.message("user", "context"),
            sim.user("first"),
            sim.agent(),
            sim.user("second"),
            sim.user("third"),
            sim.agent(),
        ],
    )

    first, second = adapter.inputs
    assert [m["content"] for m in first.new_messages] == ["context", "first"]
    assert [m["content"] for m in second.new_messages] == ["second", "third"]
    assert [m["content"] for m in second.messages] == [
        "context",
        "first",
        "agent reply 1",
        "second",
        "third",
    ]
    assert first.thread_id == second.thread_id
    assert first.scenario_state is second.scenario_state
    assert first.scenario_state is not None
    assert first.scenario_state.thread_id == first.thread_id
    assert first.scenario_state.name == "scenario"
    assert first.last_new_user_message_str() == "first"
    assert second.last_new_user_message_str() == "third"


async def test_a_fixed_reply_counts_as_the_agent_having_spoken() -> None:
    adapter = ReplyAdapter()

    await play(adapter, script=[sim.user("a"), sim.agent("fixed"), sim.user("b"), sim.agent()])

    assert [m["content"] for m in adapter.inputs[0].new_messages] == ["b"]


async def test_each_run_gets_its_own_thread() -> None:
    first, second = ReplyAdapter(), ReplyAdapter()

    await play(first, script=[sim.user("a"), sim.agent()])
    await play(second, script=[sim.user("a"), sim.agent()])

    assert first.inputs[0].thread_id != second.inputs[0].thread_id


async def test_user_step_without_text_asks_the_simulator() -> None:
    user = ScriptedUser("written by the simulator")

    result = await play(ReplyAdapter(), user, script=[sim.user(), sim.agent()])

    assert result.messages[0]["content"] == "written by the simulator"


async def test_user_step_without_text_and_without_a_simulator_needs_a_model() -> None:
    with pytest.raises(RuntimeError, match="no simulation model configured"):
        await play(ReplyAdapter(), script=[sim.user(), sim.agent()])

    assert collector.records[0].result.failure_reason is not None
    assert collector.records[0].result.failure_reason.startswith("runtime error: RuntimeError(")


async def test_adapter_replies_become_turns_with_tools_and_references() -> None:
    lookup = ToolExecution(tool_name="lookup")
    failed = ToolExecution(tool_name="refund", tool_call_error=True)
    refs = MessageReferences(query="q", references=["policy"])
    adapter = ReplyAdapter("plain text", AgentTurn(content="rich", tools=[lookup, failed], references=[refs]))
    seen: list[ScenarioState] = []

    result = await play(
        adapter,
        script=[sim.user("a"), sim.agent(), sim.user("b"), sim.agent(), seen.append],
    )

    assert [m["content"] for m in result.messages if m["role"] == "assistant"] == ["plain text", "rich"]
    assert result.tool_calls == ["lookup"]
    assert seen[0].references == [refs]
    assert seen[0].tool_executions == [lookup, failed]


async def test_state_callbacks_run_in_order_sync_and_async() -> None:
    order: list[str] = []

    def sync_step(state: ScenarioState) -> None:
        order.append(f"sync:{len(state.messages)}")

    async def async_step(state: ScenarioState) -> None:
        await asyncio.sleep(0)
        order.append(f"async:{len(state.messages)}")

    await play(ReplyAdapter(), script=[sim.user("a"), sync_step, sim.agent(), async_step])

    assert order == ["sync:1", "async:2"]


async def test_unknown_step_is_a_runtime_error() -> None:
    with pytest.raises(TypeError, match="unknown script step: 42"):
        await play(ReplyAdapter(), script=[cast("ScriptStep", 42)])

    assert collector.records[0].result.failure_reason == "runtime error: TypeError('unknown script step: 42')"


# -- succeed, fail ------------------------------------------------------------


async def test_succeed_ends_the_run_early_and_skips_the_judge() -> None:
    judge = FakeJudge()
    later: list[str] = []

    result = await play(
        ReplyAdapter(),
        judge,
        script=[sim.user("hi"), sim.agent(), sim.succeed("the agent answered"), lambda state: later.append("ran")],
    )

    assert result.success
    assert result.reasoning == "the agent answered"
    assert result.failure_reason is None
    assert later == []
    assert judge.judge_calls == []


async def test_fail_ends_the_run_early_with_a_reason() -> None:
    judge = FakeJudge()

    result = await play(
        ReplyAdapter(), judge, script=[sim.user("hi"), sim.agent(), sim.fail("wrong tool"), sim.agent()]
    )

    assert not result.success
    assert result.status is ScenarioStatus.FAIL
    assert result.failure_reason == "fail(): wrong tool"
    assert result.reasoning == "wrong tool"
    assert result.turns == 1
    assert judge.judge_calls == []
    assert "reason: fail(): wrong tool" in result.failure_summary()


async def test_fail_without_reasoning() -> None:
    result = await play(ReplyAdapter(), script=[sim.fail()])

    assert result.failure_reason == "fail() step"
    assert result.reasoning is None


async def test_a_failed_checkpoint_ends_the_run_before_a_later_succeed() -> None:
    judge = FakeJudge(criteria=(), rounds=["failed"])

    result = await play(ReplyAdapter(), judge, script=[sim.judge(["c"]), sim.succeed()])

    assert not result.success


# -- judge steps --------------------------------------------------------------


async def test_passing_checkpoint_lets_the_script_continue() -> None:
    judge = FakeJudge(criteria=(), rounds=["passed"])
    later: list[str] = []

    result = await play(
        ReplyAdapter(),
        judge,
        script=[
            sim.user("hi"),
            sim.agent(),
            sim.judge(["is polite"], additional_context="db: ok"),
            lambda s: later.append("ran"),
        ],
    )

    assert later == ["ran"]
    assert result.success
    assert _verdict_pairs(result) == [("is polite", "passed")]
    assert judge.judge_calls == [JudgeCall(["is polite"], "db: ok", 2)]
    assert result.judge_score is None


async def test_failing_checkpoint_ends_the_run() -> None:
    judge = FakeJudge(criteria=("final criterion",), rounds=["failed"], summary="not polite")
    later: list[str] = []

    result = await play(
        ReplyAdapter(),
        judge,
        script=[sim.user("hi"), sim.agent(), sim.judge(["is polite"]), lambda s: later.append("ran"), sim.agent()],
    )

    assert later == []
    assert not result.success
    assert result.status is ScenarioStatus.FAIL
    assert result.reasoning == "not polite"
    assert result.failure_reason is None
    assert result.turns == 1
    assert _verdict_pairs(result) == [("is polite", "failed")]
    assert len(judge.judge_calls) == 1, "no final judging after a failed checkpoint"


async def test_inconclusive_checkpoint_also_ends_the_run_as_unverified() -> None:
    judge = FakeJudge(criteria=(), rounds=["inconclusive"])

    result = await play(
        ReplyAdapter(), judge, script=[sim.user("hi"), sim.agent(), sim.judge(["c"]), sim.fail("not reached")]
    )

    assert result.status is ScenarioStatus.WARN
    assert result.failure_reason is None


async def test_checkpoint_without_a_judge_agent_builds_a_default_one(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def judge(
        self: JudgeAgent, state: ScenarioState, criteria: Sequence[str] | None = None, **_: Any
    ) -> JudgeVerdict:
        calls.append(list(criteria or []))
        return JudgeVerdict(verdicts=[CriterionVerdict(criterion="c", reasoning="r", status="passed")], summary="s")

    monkeypatch.setattr(JudgeAgent, "judge", judge)

    result = await play(ReplyAdapter(), script=[sim.user("hi"), sim.agent(), sim.judge(["c"])])

    assert calls == [["c"]]
    assert result.success


async def test_checkpoint_without_a_judge_agent_or_model_explains_what_to_configure() -> None:
    with pytest.raises(RuntimeError, match="no simulation model configured"):
        await play(ReplyAdapter(), script=[sim.user("hi"), sim.agent(), sim.judge(["c"])])

    assert collector.records[0].result.failure_reason is not None
    assert collector.records[0].result.failure_reason.startswith("runtime error:")


async def test_final_judge_step_runs_the_judges_own_criteria_and_the_score() -> None:
    grade = JudgeScore(value=1.0, raw_score=10, threshold=7, passed=True)
    judge = FakeJudge(["a", "b"], rounds=[["passed", "failed"]], grades=[grade])

    result = await play(
        ReplyAdapter(), judge, script=[sim.user("hi"), sim.agent(), sim.judge(additional_context="rows: 3")]
    )

    assert judge.judge_calls == [JudgeCall(["a", "b"], "rows: 3", 2)]
    assert _verdict_pairs(result) == [("a", "passed"), ("b", "failed")]
    assert result.judge_score == grade
    assert not result.success


async def test_steps_after_a_final_judge_still_run_and_do_not_trigger_a_second_verdict() -> None:
    judge = FakeJudge()
    later: list[str] = []

    result = await play(
        ReplyAdapter(), judge, script=[sim.user("hi"), sim.agent(), sim.judge(), lambda s: later.append("ran")]
    )

    assert later == ["ran"]
    assert len(judge.judge_calls) == 1
    assert len(result.verdicts) == 1


async def test_implicit_final_judge_at_the_end_of_the_script() -> None:
    judge = FakeJudge(grades=[JudgeScore(value=0.1, raw_score=2, threshold=7, passed=False)])

    result = await play(ReplyAdapter(), judge, script=[sim.user("hi"), sim.agent()])

    assert len(judge.judge_calls) == 1
    assert result.reasoning == "judge summary"
    assert not result.success, "a judge score below the threshold fails the scenario"
    assert result.failure_reason is None
    assert result.status is ScenarioStatus.FAIL
    assert "judge score 2/10 below 7" in result.failure_summary()


async def test_no_implicit_judge_without_criteria() -> None:
    judge = FakeJudge(criteria=())

    result = await play(ReplyAdapter(), judge, script=[sim.user("hi"), sim.agent()])

    assert judge.judge_calls == []
    assert result.success


@pytest.mark.parametrize("with_judge", [True, False])
async def test_final_judge_step_needs_criteria(with_judge: bool) -> None:
    agents: list[object] = [ReplyAdapter()]
    if with_judge:
        agents.append(JudgeAgent())

    with pytest.raises(ValueError, match="has a final judge\\(\\) step, but the JudgeAgent has no criteria"):
        await play(*agents, script=[sim.user("hi"), sim.agent(), sim.judge()])

    assert collector.records[0].result.failure_reason is not None
    assert collector.records[0].result.failure_reason.startswith("runtime error: ValueError(")


async def test_failed_verdict_fails_the_scenario() -> None:
    result = await play(ReplyAdapter(), FakeJudge(rounds=["failed"]), script=[sim.user("hi"), sim.agent()])

    assert not result.success
    assert result.status is ScenarioStatus.FAIL
    assert result.failure_summary().endswith("criterion failed: is polite (failed reasoning)")


async def test_a_passing_score_keeps_the_scenario_green() -> None:
    judge = FakeJudge(grades=[JudgeScore(value=1.0, raw_score=9, threshold=7, passed=True)])

    assert (await play(ReplyAdapter(), judge, script=[sim.user("hi"), sim.agent()])).success


# -- final verdict replacement ------------------------------------------------


async def test_judge_step_after_a_proceed_conclusion_does_not_duplicate_verdicts() -> None:
    judge = FakeJudge(["a", "b"], decisions=[True])

    result = await play(ReplyAdapter(), ScriptedUser(), judge, script=[sim.proceed(), sim.judge()])

    assert [v.criterion for v in result.verdicts] == ["a", "b"]
    assert result.turns == 1


async def test_judging_again_after_more_dialogue_keeps_only_the_latest_final_verdicts() -> None:
    judge = FakeJudge(rounds=["failed", "passed"])

    result = await play(
        ReplyAdapter(),
        judge,
        script=[sim.user("one"), sim.agent(), sim.judge(), sim.user("two"), sim.agent()],
    )

    assert [call.transcript_size for call in judge.judge_calls] == [2, 4]
    assert _verdict_pairs(result) == [("is polite", "passed")]
    assert result.success, "the earlier failed verdict no longer counts"


async def test_explicit_judge_step_twice_replaces_the_verdicts() -> None:
    judge = FakeJudge(rounds=["failed", "passed"])

    result = await play(ReplyAdapter(), judge, script=[sim.user("a"), sim.agent(), sim.judge(), sim.judge()])

    assert len(judge.judge_calls) == 2
    assert _verdict_pairs(result) == [("is polite", "passed")]
    assert result.reasoning == "judge summary"


async def test_dialogue_after_a_proceed_conclusion_is_judged_again_at_the_end() -> None:
    judge = FakeJudge(rounds=["failed", "passed"], decisions=[True])

    result = await play(
        ReplyAdapter(), ScriptedUser(), judge, script=[sim.proceed(), sim.user("one more thing"), sim.agent()]
    )

    assert [call.transcript_size for call in judge.judge_calls] == [2, 4]
    assert _verdict_pairs(result) == [("is polite", "passed")]


async def test_a_judge_step_at_the_end_is_not_repeated_by_the_implicit_judge() -> None:
    judge = FakeJudge()

    await play(ReplyAdapter(), judge, script=[sim.user("a"), sim.agent(), sim.judge()])

    assert len(judge.judge_calls) == 1


async def test_checkpoint_and_final_verdicts_are_both_reported() -> None:
    judge = FakeJudge(["final criterion"], rounds=["passed"])

    result = await play(
        ReplyAdapter(),
        judge,
        script=[
            sim.user("a"),
            sim.agent(),
            sim.judge(["checkpoint criterion"]),
            sim.user("b"),
            sim.agent(),
            sim.judge(),
        ],
    )

    assert _verdict_pairs(result) == [("checkpoint criterion", "passed"), ("final criterion", "passed")]
    assert result.success


async def test_checkpoints_accumulate_while_the_final_verdict_is_replaced() -> None:
    judge = FakeJudge(["final"], rounds=["passed"])

    result = await play(
        ReplyAdapter(),
        judge,
        script=[
            sim.user("a"),
            sim.agent(),
            sim.judge(["first checkpoint"]),
            sim.judge(),
            sim.user("b"),
            sim.agent(),
            sim.judge(["second checkpoint"]),
        ],
    )

    assert [v.criterion for v in result.verdicts] == ["first checkpoint", "second checkpoint", "final"]


async def test_a_replaced_final_verdict_also_replaces_the_score() -> None:
    judge = FakeJudge(grades=[JudgeScore(value=0.0, raw_score=1, threshold=7, passed=False), None])

    result = await play(
        ReplyAdapter(), judge, script=[sim.user("a"), sim.agent(), sim.judge(), sim.user("b"), sim.agent()]
    )

    assert judge.score_calls == 2
    assert result.judge_score is None
    assert result.success


# -- errors -------------------------------------------------------------------


async def test_assertion_in_a_callable_is_recorded_and_reraised() -> None:
    def check(state: ScenarioState) -> None:
        assert state.has_tool_call("refund"), "the refund tool was never called"

    with pytest.raises(AssertionError, match="refund tool"):
        await play(ReplyAdapter(), script=[sim.user("hi"), sim.agent(), check])

    result = collector.records[0].result
    assert result.status is ScenarioStatus.FAIL
    assert result.failure_reason is not None
    assert result.failure_reason.startswith("assertion in script: the refund tool was never called")
    assert not result.harness_error
    assert result.turns == 1


async def test_assertion_in_an_async_callable_is_recorded_and_reraised() -> None:
    async def check(state: ScenarioState) -> None:
        await asyncio.sleep(0)
        raise AssertionError("async check failed")

    with pytest.raises(AssertionError):
        await play(ReplyAdapter(), script=[check])

    assert collector.records[0].result.failure_reason == "assertion in script: async check failed"


async def test_assertion_in_a_proceed_callback_is_recorded_and_reraised() -> None:
    def on_turn(state: ScenarioState) -> None:
        raise AssertionError("too chatty")

    with pytest.raises(AssertionError):
        await play(ReplyAdapter(), ScriptedUser(), script=[sim.proceed(on_turn=on_turn)])

    assert collector.records[0].result.failure_reason == "assertion in script: too chatty"


@pytest.mark.parametrize("error", [sim.SimulationBackendError("504"), sim.SimulationStructuredOutputError("no schema")])
async def test_harness_errors_are_recorded_as_warn_and_reraised(
    error: sim.SimulationHarnessError, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken(AgentAdapter):
        async def call(self, input: AgentInput) -> str:
            raise error

    with caplog.at_level(logging.WARNING, logger="simulhausen.runner"), pytest.raises(sim.SimulationHarnessError):
        await play(Broken(), script=[sim.user("hi"), sim.agent()])

    result = collector.records[0].result
    assert result.status is ScenarioStatus.WARN
    assert result.harness_error
    assert result.failure_reason == f"harness failure: {error!r}"
    assert not result.success
    assert result.failure_summary().startswith("scenario 'scenario' did not finish: harness failure")
    assert any("harness failure" in message for message in caplog.messages)


async def test_simulator_failure_is_a_harness_warning() -> None:
    class Mute(UserSimulatorAgent):
        async def next_turn(self, state: ScenarioState) -> UserTurn:
            raise sim.SimulationStructuredOutputError("the model answered in prose")

    with pytest.raises(sim.SimulationStructuredOutputError):
        await play(ReplyAdapter(), Mute(), script=[sim.user(), sim.agent()])

    assert collector.records[0].result.status is ScenarioStatus.WARN


async def test_generic_exception_is_recorded_as_a_runtime_error_and_reraised(caplog: pytest.LogCaptureFixture) -> None:
    class Boom(AgentAdapter):
        async def call(self, input: AgentInput) -> str:
            raise KeyError("boom")

    with caplog.at_level(logging.ERROR, logger="simulhausen.runner"), pytest.raises(KeyError):
        await play(Boom(), script=[sim.user("hi"), sim.agent()])

    result = collector.records[0].result
    assert result.status is ScenarioStatus.FAIL
    assert result.failure_reason == "runtime error: KeyError('boom')"
    assert not result.harness_error
    assert caplog.records[-1].exc_info is not None
    assert "scenario run failed" in caplog.records[-1].message


async def test_agent_time_is_counted_even_when_the_adapter_raises() -> None:
    class Slow(AgentAdapter):
        async def call(self, input: AgentInput) -> str:
            await asyncio.sleep(0.01)
            raise RuntimeError("late")

    with pytest.raises(RuntimeError):
        await play(Slow(), script=[sim.user("hi"), sim.agent()])

    assert collector.records[0].result.agent_time > 0


@pytest.mark.parametrize(
    "agents", [[], [FakeJudge()], [ReplyAdapter(), ReplyAdapter()]], ids=["none", "judge only", "two"]
)
async def test_exactly_one_adapter_is_required(agents: list[object]) -> None:
    with pytest.raises(ValueError, match="exactly one AgentAdapter"):
        await play(*agents)

    assert collector.records == []


async def test_non_positive_turn_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_turns must be at least 1"):
        await sim.run("limits", "desc", [ReplyAdapter()], script=[sim.user("hi")], max_turns=0)
    with pytest.raises(ValueError, match="min_turns must be at least 1"):
        await sim.run("limits", "desc", [ReplyAdapter()], script=[sim.user("hi")], min_turns=-1)

    assert collector.records == []


async def test_function_adapter_works_as_the_agent_under_test() -> None:
    result = await play(
        sim.function_adapter(lambda agent_input: f"echo {agent_input.last_new_user_message_str()}"),
        script=[sim.user("ping"), sim.agent()],
    )

    assert result.messages[1]["content"] == "echo ping"


# -- unused simulator warning -------------------------------------------------


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == "simulhausen.runner"]


async def test_warns_when_every_user_message_is_fixed_text(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="simulhausen.runner"):
        await play(ReplyAdapter(), UserSimulatorAgent(), script=[sim.user("hi"), sim.agent()])

    [warning] = _warnings(caplog)
    assert warning.startswith("[scenario] UserSimulatorAgent is never called")


@pytest.mark.parametrize(
    "script",
    [
        [sim.user("hi"), sim.agent(), sim.user(), sim.agent()],
        [sim.user("hi"), sim.agent(), sim.proceed(1)],
        None,
    ],
    ids=["user()", "proceed()", "auto mode"],
)
async def test_no_warning_when_the_simulator_is_used(
    caplog: pytest.LogCaptureFixture, script: list[ScriptStep] | None
) -> None:
    with caplog.at_level(logging.WARNING, logger="simulhausen.runner"):
        await play(ReplyAdapter(), ScriptedUser(), script=script, max_turns=3)

    assert _warnings(caplog) == []


async def test_no_warning_without_a_simulator(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="simulhausen.runner"):
        await play(ReplyAdapter(), script=[sim.user("hi"), sim.agent()])

    assert _warnings(caplog) == []


# -- debug mode ---------------------------------------------------------------


class Keyboard:
    def __init__(self, *lines: str) -> None:
        self.lines = list(lines)
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        return self.lines.pop(0)


async def test_debug_user_step_takes_typed_text_and_falls_back_to_the_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyboard = Keyboard("  typed question  ", "")
    monkeypatch.setattr("builtins.input", keyboard)
    user = ScriptedUser("simulated follow-up")

    result = await play(ReplyAdapter(), user, script=[sim.user(), sim.agent(), sim.user(), sim.agent()], debug=True)

    assert [m["content"] for m in result.messages if m["role"] == "user"] == ["typed question", "simulated follow-up"]
    assert "(dialogue start)" in keyboard.prompts[0]
    assert "agent: agent reply 1" in keyboard.prompts[1]
    assert "Enter = simulator" in keyboard.prompts[1]
    assert user.transcript_sizes == [2], "the simulator is not asked when the user typed"


async def test_debug_proceed_takes_typed_text_and_falls_back_to_the_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", Keyboard("typed", "   "))
    user = ScriptedUser("simulated")

    result = await play(ReplyAdapter(), user, script=[sim.proceed(turns=2)], debug=True)

    assert [m["content"] for m in result.messages if m["role"] == "user"] == ["typed", "simulated"]
    assert user.transcript_sizes == [2]


async def test_debug_comes_from_the_config_and_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    sim.configure(debug=True)
    keyboard = Keyboard("from the keyboard")
    monkeypatch.setattr("builtins.input", keyboard)

    result = await play(ReplyAdapter(), ScriptedUser(), script=[sim.user(), sim.agent()])
    assert result.messages[0]["content"] == "from the keyboard"

    monkeypatch.setattr("builtins.input", Keyboard())
    result = await play(ReplyAdapter(), ScriptedUser("simulated"), script=[sim.user(), sim.agent()], debug=False)
    assert result.messages[0]["content"] == "simulated"


# -- config fallbacks, timing, collector, logging ----------------------------


async def test_max_turns_comes_from_the_config_unless_overridden() -> None:
    sim.configure(max_turns=2)

    assert (await play(ReplyAdapter(), ScriptedUser())).turns == 2
    assert (await play(ReplyAdapter(), ScriptedUser(), max_turns=1)).turns == 1


async def test_timings_are_populated() -> None:
    result = await play(ReplyAdapter(delay=0.02), script=[sim.user("hi"), sim.agent()])

    assert result.agent_time >= 0.015
    assert result.total_time >= result.agent_time


async def test_fixed_replies_cost_no_agent_time() -> None:
    result = await play(ReplyAdapter(), script=[sim.user("hi"), sim.agent("fixed")])

    assert result.agent_time == 0.0
    assert result.total_time > 0


async def test_every_result_lands_in_the_collector_with_its_test_id() -> None:
    collector.set_current_test("tests/test_x.py::test_case")

    result = await play(ReplyAdapter(), script=[sim.user("hi"), sim.agent()])

    [record] = collector.records
    assert record.result is result
    assert record.test_id == "tests/test_x.py::test_case"
    assert result.name == "scenario"
    assert result.description == "A user tests the agent."


async def test_verbose_runs_log_the_dialogue_and_the_verdicts(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="simulhausen.runner"):
        await play(ReplyAdapter("hello!"), FakeJudge(), script=[sim.user("hi"), sim.agent()], verbose=True)

    assert caplog.messages == [
        "[scenario] user: hi",
        "[scenario] agent: hello!",
        "[scenario] judge passed: is polite (passed reasoning)",
    ]


async def test_quiet_runs_log_nothing(caplog: pytest.LogCaptureFixture) -> None:
    sim.configure(verbose=True)

    with caplog.at_level(logging.INFO, logger="simulhausen.runner"):
        await play(ReplyAdapter(), FakeJudge(), script=[sim.user("hi"), sim.agent()], verbose=False)

    assert caplog.records == []


async def test_verbose_proceed_logs_goal_achieved_and_ignored_conclusions(caplog: pytest.LogCaptureFixture) -> None:
    user = ScriptedUser("question", UserTurn(message="", goal_achieved=True))
    judge = FakeJudge(rounds=["inconclusive", "passed"], decisions=[True])

    with caplog.at_level(logging.INFO, logger="simulhausen.runner"):
        await play(ReplyAdapter(), user, judge, max_turns=5, verbose=True)

    assert "[scenario] user: (goal achieved)" in caplog.messages
    assert "[scenario] judge: concluded with inconclusive criteria, continuing the dialogue" in caplog.messages
