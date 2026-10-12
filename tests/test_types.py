from typing import cast

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
