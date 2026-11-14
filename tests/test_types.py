from typing import Any, cast

import pytest
from agno.models.message import MessageReferences
from agno.models.response import ToolExecution
from pydantic import ValidationError

import simulhausen as sim
from simulhausen import AgentAdapter, AgentInput, CriterionVerdict, JudgeScore, ScenarioResult, ScenarioState
from simulhausen.types import ScenarioStatus

PAUSE_FLAGS = ["requires_confirmation", "requires_user_input", "external_execution_required"]


def _state() -> ScenarioState:
    return ScenarioState(thread_id="thread", name="scenario", description="description")


def _tool(name: str | None, *, error: bool | None = None, paused: str | None = None) -> ToolExecution:
    tool = ToolExecution(tool_name=name, tool_call_error=error)
    if paused:
        setattr(tool, paused, True)
    return tool


def _verdict(status: str, criterion: str = "criterion") -> CriterionVerdict:
    return CriterionVerdict.model_validate({"criterion": criterion, "reasoning": f"why {status}", "status": status})


def _score(raw: int, threshold: int = 7) -> JudgeScore:
    return JudgeScore(value=(raw - 1) / 9, raw_score=raw, threshold=threshold, passed=raw >= threshold)


# -- messages -----------------------------------------------------------------


def test_message_helpers_on_a_dialogue() -> None:
    state = _state()
    state.add_user_message("first question")
    state.add_agent_message("first answer")
    state.add_user_message("second question")

    assert state.last_message() == {"role": "user", "content": "second question"}
    assert state.last_user_message()["content"] == "second question"
    assert state.last_agent_message()["content"] == "first answer"
    assert state.first_user_message_str() == "first question"
    assert state.last_user_message_str() == "second question"
    assert state.last_agent_message_str() == "first answer"


def test_message_helpers_on_an_empty_transcript() -> None:
    state = _state()

    with pytest.raises(ValueError, match="transcript is empty"):
        state.last_message()
    with pytest.raises(ValueError, match="no user message"):
        state.last_user_message()
    with pytest.raises(ValueError, match="no assistant message"):
        state.last_agent_message()
    assert state.first_user_message_str() == ""
    assert state.last_user_message_str() == ""
    assert state.last_agent_message_str() == ""


def test_role_specific_helpers_when_the_role_never_spoke() -> None:
    state = _state()
    state.add_user_message("hello")

    assert state.last_agent_message_str() == ""
    with pytest.raises(ValueError, match="no assistant message"):
        state.last_agent_message()


def test_add_agent_message_accumulates_tools_and_references() -> None:
    state = _state()
    refs = MessageReferences(query="q", references=["text"])
    state.add_agent_message("a", tools=[_tool("one")], references=[refs])
    state.add_agent_message("b", tools=[], references=None)
    state.add_agent_message("c", tools=[_tool("two")])

    assert [t.tool_name for t in state.tool_executions] == ["one", "two"]
    assert state.references == [refs]


def test_transcript_names() -> None:
    state = _state()
    assert state.transcript() == ""
    state.add_user_message("hi")
    state.add_agent_message("hello")

    assert state.transcript() == "User: hi\nAssistant: hello"
    assert state.transcript(user_name="Customer", agent_name="Ellie") == "Customer: hi\nEllie: hello"


# -- tools --------------------------------------------------------------------


def test_tool_assertions_require_clean_execution() -> None:
    state = _state()
    state.tool_executions = [
        _tool("refused", error=True),
        _tool("pending", paused="requires_confirmation"),
        _tool("completed"),
        _tool(None),
    ]

    assert not state.has_tool_call("refused")
    assert not state.has_tool_call("pending")
    assert state.has_tool_call("completed")
    assert state.tool_call_names() == ["completed"]
    assert not state.has_tool_call("missing")


@pytest.mark.parametrize("flag", PAUSE_FLAGS)
def test_a_paused_tool_is_not_a_clean_call(flag: str) -> None:
    tool = _tool("search", paused=flag)
    state = _state()
    state.tool_executions = [tool]

    assert tool.is_paused
    assert not state.has_tool_call("search")
    assert state.tool_calls() == []


def test_tool_call_error_false_still_counts() -> None:
    state = _state()
    state.tool_executions = [_tool("search", error=False)]

    assert state.has_tool_call("search")


def test_tool_calls_and_last_tool_call() -> None:
    first, failed, second, other = _tool("search"), _tool("search", error=True), _tool("search"), _tool("book")
    state = _state()
    state.tool_executions = [first, failed, second, other]

    assert [call.execution for call in state.tool_calls("search")] == [first, second]
    assert [call.name for call in state.tool_calls()] == ["search", "search", "book"]
    assert state.tool_calls("absent") == []
    assert state.last_tool_call("search") is second
    assert state.last_tool_call() is other
    assert state.last_tool_call("absent") is None
    assert _state().last_tool_call() is None


def test_tool_calls_in_order_skips_failed_calls_between() -> None:
    state = _state()
    state.tool_executions = [_tool("first"), _tool("failed", error=True), _tool("second")]

    assert state.has_tool_calls_in_order(["first", "second"])
    assert not state.has_tool_calls_in_order(["second", "first"])
    assert not state.has_tool_calls_in_order(["first", "failed"])
    assert state.has_tool_calls_in_order([])


def test_tool_calls_in_order_is_a_subsequence_and_counts_repeats() -> None:
    state = _state()
    state.tool_executions = [_tool("a"), _tool("x"), _tool("a"), _tool("b")]

    assert state.has_tool_calls_in_order(["a", "b"])
    assert state.has_tool_calls_in_order(["a", "a", "b"])
    assert not state.has_tool_calls_in_order(["a", "a", "a"])
    assert state.tool_call_names() == ["a", "x", "a", "b"]


# -- judge evidence -----------------------------------------------------------


def test_references_text_keeps_article_url_for_judge() -> None:
    state = _state()
    state.add_agent_message(
        "answer",
        references=[
            MessageReferences(
                query="honey",
                references=[
                    {
                        "name": "Honey",
                        "content": "Honey is allowed in hand luggage",
                        "meta_data": {"url": "https://example.com/honey"},
                    },
                    {"name": "No link", "content": "An article without an address", "meta_data": {}},
                    {"content": "Metadata key variant", "metadata": {"url": "https://example.com/alt"}},
                    {"content": "Metadata is not a dict", "meta_data": "oops"},
                    "raw text",
                ],
            ),
            MessageReferences(query="empty", references=None),
        ],
    )

    assert state.references_text().split("\n---\n") == [
        "Honey is allowed in hand luggage\nSource URL: https://example.com/honey",
        "An article without an address",
        "Metadata key variant\nSource URL: https://example.com/alt",
        "Metadata is not a dict",
        "raw text",
    ]


def test_references_text_falls_back_to_the_whole_dict_without_content() -> None:
    state = _state()
    state.add_agent_message("a", references=[MessageReferences(query="q", references=[{"title": "only a title"}])])

    assert "only a title" in state.references_text()


def test_references_text_is_empty_without_references() -> None:
    assert _state().references_text() == ""


def test_tool_evidence_includes_success_error_and_paused() -> None:
    state = _state()
    state.tool_executions = [
        ToolExecution(tool_name="lookup", tool_args={"id": 7}, result='  {"status":"ok"}  '),
        ToolExecution(tool_name="get_status", tool_call_error=True, result="AuthorizationRequiredError"),
        ToolExecution(tool_name="refund", requires_confirmation=True),
        ToolExecution(tool_name=None),
    ]

    blocks = state.tool_evidence_text().split("\n---\n")

    assert blocks[0] == 'Tool: lookup\nStatus: success\nArguments: {\'id\': 7}\nResult:\n{"status":"ok"}'
    assert blocks[1] == "Tool: get_status\nStatus: error\nResult:\nAuthorizationRequiredError"
    assert blocks[2] == "Tool: refund\nStatus: paused"
    assert blocks[3] == "Tool: unknown\nStatus: success"


def test_tool_evidence_caps_each_result_and_the_total() -> None:
    state = _state()
    state.tool_executions = [ToolExecution(tool_name="big", result="x" * 500) for _ in range(3)]

    capped_result = state.tool_evidence_text(max_result_chars=10).split("\n---\n")[0]
    assert capped_result.endswith("Result:\n" + "x" * 10)

    total = state.tool_evidence_text(max_total_chars=100)
    assert len(total) == 100


def test_tool_evidence_is_empty_without_tools() -> None:
    assert _state().tool_evidence_text() == ""


# -- AgentInput ---------------------------------------------------------------


def test_last_new_user_message_prefers_new_messages() -> None:
    agent_input = AgentInput(
        thread_id="t",
        messages=[{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}],
        new_messages=[{"role": "user", "content": "fresh"}],
    )

    assert agent_input.last_new_user_message_str() == "fresh"


def test_last_new_user_message_falls_back_to_the_whole_dialogue() -> None:
    agent_input = AgentInput(
        thread_id="t",
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "fixed reply"},
        ],
    )

    assert agent_input.last_new_user_message_str() == "question"


def test_last_new_user_message_is_empty_without_user_messages() -> None:
    assert AgentInput(thread_id="t", messages=[]).last_new_user_message_str() == ""
    only_agent = AgentInput(thread_id="t", messages=[{"role": "assistant", "content": "hi"}])
    assert only_agent.last_new_user_message_str() == ""


# -- verdicts and scores ------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "passed", "failed", "inconclusive"),
    [("passed", True, False, False), ("failed", False, True, False), ("inconclusive", False, False, True)],
)
def test_criterion_verdict_flags(status: str, passed: bool, failed: bool, inconclusive: bool) -> None:
    verdict = _verdict(status)

    assert (verdict.passed, verdict.failed, verdict.inconclusive) == (passed, failed, inconclusive)
    assert verdict.requirement == ""


@pytest.mark.parametrize(
    ("requirement", "stored"),
    [
        ("passed", ""),
        ("  fail ", ""),
        ("выполнено", ""),
        ("", ""),
        ("  The agent offers a refund  ", "The agent offers a refund"),
    ],
)
def test_criterion_verdict_drops_a_one_word_requirement(requirement: str, stored: str) -> None:
    verdict = CriterionVerdict(criterion="c", requirement=requirement, reasoning="r", status="passed")

    assert verdict.requirement == stored


def test_criterion_verdict_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        _verdict("maybe")


@pytest.mark.parametrize("kwargs", [{"value": 1.2}, {"raw_score": 0}, {"raw_score": 11}, {"threshold": 0}])
def test_judge_score_validates_ranges(kwargs: dict[str, float]) -> None:
    base: dict[str, float | bool] = {"value": 0.5, "raw_score": 5, "threshold": 5, "passed": True}
    with pytest.raises(ValidationError):
        JudgeScore.model_validate({**base, **kwargs})


def test_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        cast("type[AgentAdapter]", AgentAdapter)()
    assert AgentAdapter.role is sim.AgentRole.AGENT


# -- ScenarioResult -----------------------------------------------------------


def _result(
    *,
    success: bool = False,
    verdicts: list[CriterionVerdict] | None = None,
    judge_score: JudgeScore | None = None,
    reasoning: str | None = None,
    failure_reason: str | None = None,
    harness_error: bool = False,
) -> ScenarioResult:
    return ScenarioResult(
        name="refund",
        description="d",
        success=success,
        verdicts=verdicts or [],
        judge_score=judge_score,
        reasoning=reasoning,
        failure_reason=failure_reason,
        harness_error=harness_error,
    )


def test_status_pass() -> None:
    result = _result(success=True, verdicts=[_verdict("passed")])

    assert result.status is ScenarioStatus.PASS
    assert result.failure_summary() == ""


def test_status_fail_on_failed_criterion() -> None:
    result = _result(verdicts=[_verdict("passed", "a"), _verdict("failed", "b"), _verdict("inconclusive", "c")])

    assert result.status is ScenarioStatus.FAIL
    assert result.passed_criteria == ["a"]
    assert result.failed_criteria == ["b"]
    assert result.inconclusive_criteria == ["c"]


def test_status_fail_without_any_explanation() -> None:
    assert _result().status is ScenarioStatus.FAIL


def test_status_warn_on_harness_error() -> None:
    result = _result(harness_error=True, failure_reason="harness failure: boom", verdicts=[_verdict("failed")])

    assert result.status is ScenarioStatus.WARN
    assert result.failure_summary().startswith("scenario 'refund' did not finish: harness failure")


def test_status_warn_when_only_inconclusive_criteria_remain() -> None:
    result = _result(verdicts=[_verdict("passed", "a"), _verdict("inconclusive", "b")])

    assert result.status is ScenarioStatus.WARN
    summary = result.failure_summary()
    assert summary.startswith("scenario 'refund' could not be verified")
    assert "criterion inconclusive: b (why inconclusive)" in summary
    assert "criterion passed" not in summary


def test_inconclusive_stays_fail_with_a_failure_reason() -> None:
    result = _result(verdicts=[_verdict("inconclusive")], failure_reason="assertion in script: x")

    assert result.status is ScenarioStatus.FAIL


def test_inconclusive_stays_fail_when_the_score_is_below_threshold() -> None:
    assert _result(verdicts=[_verdict("inconclusive")], judge_score=_score(3)).status is ScenarioStatus.FAIL


def test_inconclusive_with_a_passing_score_is_warn() -> None:
    assert _result(verdicts=[_verdict("inconclusive")], judge_score=_score(9)).status is ScenarioStatus.WARN


def test_failure_summary_lists_reason_judge_score_and_criteria() -> None:
    result = _result(
        failure_reason="assertion in script: no refund",
        reasoning="the agent never refunded",
        judge_score=_score(3),
        verdicts=[_verdict("failed", "refunds the order"), _verdict("passed", "greets")],
    )

    assert result.failure_summary() == (
        "scenario 'refund' failed; reason: assertion in script: no refund; "
        "judge: the agent never refunded; judge score 3/10 below 7; "
        "criterion failed: refunds the order (why failed)"
    )


def test_failure_summary_does_not_repeat_identical_reasoning() -> None:
    result = _result(failure_reason="fail(): nope", reasoning="fail(): nope")

    assert result.failure_summary() == "scenario 'refund' failed; reason: fail(): nope"


# -- tool call view and rollback -------------------------------------------------


def _executed(name: str, args: dict[str, object], result: str) -> ToolExecution:
    return ToolExecution(tool_name=name, tool_args=args, result=result)


def test_tool_calls_view_exposes_inputs_outputs_and_turns() -> None:
    state = _state()
    state.add_user_message("find order 1")
    state.add_agent_message("found", tools=[_executed("lookup", {"id": "1"}, "order 1")])
    state.add_user_message("and order 2?")
    state.add_agent_message("found too", tools=[_executed("lookup", {"id": "2"}, "order 2")])

    calls = state.tool_calls("lookup")

    assert len(calls) == 2
    assert calls.inputs == [{"id": "1"}, {"id": "2"}]
    assert calls.outputs == ["order 1", "order 2"]
    assert [call.turn for call in calls] == [0, 1]
    assert calls.first.input == {"id": "1"}
    assert calls.last.output == "order 2"
    assert calls[1].name == "lookup"
    assert list(calls[:1]) == [calls.first]
    assert "lookup" in repr(calls)
    assert calls == state.tool_calls("lookup")
    assert calls != "lookup"


def test_empty_tool_calls_give_falsy_picks() -> None:
    calls = _state().tool_calls("missing")

    assert not calls.first
    assert not calls.last
    assert calls.last.name == "missing"
    assert calls.inputs == []
    assert calls.outputs == []


def test_hand_added_tool_calls_have_no_turn() -> None:
    state = _state()
    state.tool_executions = [_executed("lookup", {}, "x")]

    assert state.tool_calls().first.turn is None


def test_rollback_drops_messages_and_what_their_turns_produced() -> None:
    kept_tool, removed_tool = _executed("lookup", {}, "a"), _executed("refund", {}, "b")
    kept_ref = MessageReferences(query="q1", references=["first"])
    removed_ref = MessageReferences(query="q2", references=["second"])
    state = _state()
    state.add_user_message("one")
    state.add_agent_message("first answer", tools=[kept_tool], references=[kept_ref])
    state.add_user_message("two")
    state.add_agent_message("second answer", tools=[removed_tool], references=[removed_ref])
    state.current_turn = 2
    hand_added = _executed("manual", {}, "c")
    state.tool_executions.append(hand_added)
    executions = state.tool_executions

    removed = state.rollback_messages_to(2)

    assert removed == [
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second answer"},
    ]
    assert [m["content"] for m in state.messages] == ["one", "first answer"]
    assert state.tool_executions is executions
    assert state.tool_executions == [kept_tool, hand_added]
    assert state.references == [kept_ref]
    assert state.current_turn == 2


def test_rollback_index_is_clamped_and_must_not_be_negative() -> None:
    state = _state()
    state.add_user_message("one")

    assert state.rollback_messages_to(10) == []
    assert len(state.messages) == 1
    with pytest.raises(ValueError, match=">= 0"):
        state.rollback_messages_to(-1)


def test_turns_after_a_rollback_count_remaining_agent_messages() -> None:
    state = _state()
    state.add_user_message("one")
    state.add_agent_message("a1")
    state.add_user_message("two")
    state.add_agent_message("a2")
    state.rollback_messages_to(2)
    state.add_user_message("two again")
    state.add_agent_message("a2 again", tools=[_executed("lookup", {}, "x")])

    assert state.tool_calls().first.turn == 1


# -- replay from a report -------------------------------------------------------


REPORTED: dict[str, Any] = {
    "name": "refund",
    "description": "A user asks for a refund",
    "tool_calls": ["lookup"],
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "checking"},
        {"role": "user", "content": "and?"},
        {"role": "assistant", "content": "done"},
    ],
    "judge_calls": [
        {
            "criteria": ["looks it up"],
            "additional_context": "db: row 1",
            "messages_count": 2,
            "references": [{"query": "q", "references": [{"content": "policy"}]}],
            "tool_executions": [{"tool_name": "lookup", "tool_args": {"id": 1}, "result": "found"}],
        }
    ],
}


def test_from_report_rebuilds_the_state_a_judge_call_saw() -> None:
    by_index = ScenarioState.from_report(REPORTED, 0)
    by_entry = ScenarioState.from_report(REPORTED, REPORTED["judge_calls"][0])

    for state in (by_index, by_entry):
        assert (state.name, state.description) == ("refund", "A user asks for a refund")
        assert state.messages == REPORTED["messages"][:2]
        assert state.current_turn == 1
        assert state.references_text() == "policy"
        assert state.tool_call_names() == ["lookup"]
        assert state.tool_calls("lookup").last.input == {"id": 1}


def test_from_report_without_a_judge_call_holds_the_whole_dialogue_and_no_evidence() -> None:
    state = ScenarioState.from_report(REPORTED)

    assert state.messages == REPORTED["messages"]
    assert state.current_turn == 2
    assert (state.references, state.tool_executions) == ([], [])


def test_from_report_accepts_a_minimal_scenario() -> None:
    state = ScenarioState.from_report({"messages": [{"role": "tool", "content": 1}]}, {})

    assert (state.name, state.description) == ("", "")
    assert state.messages == [{"role": "user", "content": "1"}]


# -- failed and paused calls (tool_calls(include_errors=True), turn_of) ------


def _login_dialogue() -> tuple[ScenarioState, ToolExecution, ToolExecution]:
    state = _state()
    failed = ToolExecution(tool_name="show_my_bookings", tool_args={}, result="login required", tool_call_error=True)
    paused = ToolExecution(tool_name="open_login_form", tool_args={"reason": "bookings"}, requires_confirmation=True)
    state.add_user_message("show my bookings")
    state.add_agent_message("please log in", tools=[failed, paused])
    state.add_user_message("no thanks")
    state.add_agent_message("ok", tools=[_executed("search", {"q": "public"}, "found")])
    return state, failed, paused


def test_tool_calls_leave_failed_and_paused_calls_out_by_default() -> None:
    state, _, _ = _login_dialogue()

    assert [call.name for call in state.tool_calls()] == ["search"]
    assert state.tool_calls("show_my_bookings") == []


def test_include_errors_returns_failed_and_paused_calls_with_their_status_and_turn() -> None:
    state, failed, paused = _login_dialogue()

    calls = state.tool_calls(include_errors=True)

    assert [(call.name, call.status, call.turn) for call in calls] == [
        ("show_my_bookings", "error", 0),
        ("open_login_form", "paused", 0),
        ("search", "success", 1),
    ]
    assert [call.ok for call in calls] == [False, False, True]
    assert (calls[0].error, calls[0].paused) == (True, False)
    assert (calls[1].error, calls[1].paused) == (False, True)
    assert calls[0].output == "login required"
    assert calls[0].execution is failed
    assert state.tool_calls("open_login_form", include_errors=True).last.execution is paused


def test_the_turn_filter_answers_the_issue_18_question() -> None:
    state, _, _ = _login_dialogue()
    login_tools = {"show_my_bookings", "open_login_form"}

    again = [call for call in state.tool_calls(include_errors=True, turn=1) if call.name in login_tools]

    assert again == []
    assert len(state.tool_calls(include_errors=True, turn=0)) == 2
    assert [call.name for call in state.tool_calls(turn=1)] == ["search"]
    assert state.tool_calls("search", turn=0) == []


def test_turn_of_is_public_and_covers_failed_executions_and_references() -> None:
    state, failed, paused = _login_dialogue()
    reference = MessageReferences(query="q", references=["doc"])
    state.add_user_message("more")
    state.add_agent_message("here", references=[reference])

    assert state.turn_of(failed) == 0
    assert state.turn_of(paused) == 0
    assert state.turn_of(reference) == 2
    assert state.turn_of(_executed("stranger", {}, "x")) is None


def test_an_execution_without_a_name_is_never_a_call() -> None:
    state = _state()
    state.tool_executions = [ToolExecution(tool_name=None, tool_call_error=True)]

    assert state.tool_calls(include_errors=True) == []


# -- usage --------------------------------------------------------------------


def test_usage_adds_up_and_fills_the_total() -> None:
    first = sim.Usage(input_tokens=10, output_tokens=5, requests=1)
    second = sim.Usage(input_tokens=1, output_tokens=2, total_tokens=4, requests=2, cost=0.5)

    assert first.total_tokens == 15
    assert first + second == sim.Usage(11, 7, 19, requests=3, cost=0.5)
    assert (second + sim.Usage(cost=0.25)).cost == 0.75
    assert (first + first).cost is None
    assert first.__add__(3) is NotImplemented  # ty: ignore[invalid-argument-type]
    assert not sim.Usage()
    assert sim.Usage(requests=1)
    assert sim.Usage.from_dict(second.to_dict()) == second
    assert sim.Usage.from_dict({}) == sim.Usage()


def test_usage_of_reads_objects_mappings_and_costs() -> None:
    assert sim.Usage.of(None) is None
    assert sim.Usage.of({"nothing": 1}) is None
    assert sim.Usage.of({"prompt_tokens": 2, "completion_tokens": 3}, requests=1) == sim.Usage(2, 3, 5, requests=1)
    assert sim.Usage.of({"total_cost_usd": 0.1}) == sim.Usage(cost=0.1)


def test_the_state_sums_usage_over_agent_turns_and_keeps_it_through_a_rollback() -> None:
    state = _state()
    assert state.usage is None
    state.add_user_message("one")
    state.add_agent_message("a", usage=sim.Usage(input_tokens=3, output_tokens=1))
    state.add_user_message("two")
    state.add_agent_message("b")
    state.add_agent_message("c", usage=sim.Usage(input_tokens=1, output_tokens=1, requests=1))

    assert state.usage == sim.Usage(4, 2, 6, requests=1)
    state.rollback_messages_to(0)
    assert state.usage == sim.Usage(4, 2, 6, requests=1)


# -- trials -------------------------------------------------------------------


def _trial(status: str, trial: int | None = None) -> ScenarioResult:
    verdict = CriterionVerdict(criterion="c", reasoning="r", status="inconclusive" if status == "warn" else "failed")
    return ScenarioResult(
        name="s",
        description="d",
        success=status == "pass",
        verdicts=[] if status == "pass" else [verdict],
        total_time=1.5,
        trial=trial,
    )


def test_trials_result_counts_and_rates() -> None:
    outcome = sim.TrialsResult(name="s", results=[_trial("pass"), _trial("pass"), _trial("fail"), _trial("warn")])

    assert (outcome.trials, outcome.passed, outcome.failed, outcome.warned, outcome.judged) == (4, 2, 1, 1, 3)
    assert outcome.pass_rate == pytest.approx(2 / 3)
    assert outcome.flaky
    assert not outcome.success
    assert outcome.total_time == 6.0
    assert outcome.usage is None
    # C(2, k) / C(3, k)
    assert outcome.pass_hat(1) == pytest.approx(2 / 3)
    assert outcome.pass_hat(2) == pytest.approx(1 / 3)
    assert outcome.pass_hat() == 0.0
    with pytest.raises(ValueError, match="k must be between 1 and the 3 judged"):
        outcome.pass_hat(4)


def test_trials_result_min_pass_rate_and_usage() -> None:
    results = [_trial("pass"), _trial("fail")]
    results[0].usage = sim.Usage(1, 1)
    results[1].usage = sim.Usage(2, 2)
    outcome = sim.TrialsResult(name="s", results=results, min_pass_rate=0.5)

    assert outcome.success
    assert outcome.failure_summary() == ""
    assert outcome.usage == sim.Usage(3, 3)
    assert sim.TrialsResult(name="s", results=[_trial("pass")] * 3).pass_hat() == 1.0


def test_trials_result_without_a_judged_trial_is_not_a_success() -> None:
    outcome = sim.TrialsResult(name="s", results=[_trial("warn"), _trial("warn")], min_pass_rate=0.0)

    assert not outcome.success
    assert not outcome.flaky
    assert outcome.pass_rate == 0.0
    assert outcome.pass_hat() == 0.0
    assert outcome.failure_summary().startswith("scenario 's' could not be verified in any of 2 trial(s)")


def test_trials_failure_summary_names_the_failed_trials() -> None:
    outcome = sim.TrialsResult(name="s", results=[_trial("pass", 1), _trial("fail", 2), _trial("warn")])

    summary = outcome.failure_summary()

    assert summary.startswith("scenario 's' passed 1 of 2 judged trial(s) (50%, needs 100%), 1 not verified")
    assert "trial 2: scenario 's' failed" in summary
    assert "trial 3: scenario 's' could not be verified" in summary
    assert "trial 1:" not in summary
