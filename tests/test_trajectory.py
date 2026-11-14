import re
from typing import Any

import pytest
from agno.models.response import ToolExecution

import simulhausen as sim
from simulhausen import ScenarioState, ToolCall
from simulhausen.trajectory import ExpectedCall, match_trajectory


def _call(name: str, status: Any = "success", **args: Any) -> ToolCall:
    return ToolCall(name=name, input=args, execution=ToolExecution(tool_name=name, tool_args=args), status=status)


SEARCH = _call("search", q="order 1042")
LOOKUP = _call("lookup_order", order_id="1042", verbose=True)
REFUND = _call("refund", order_id="1042", amount=30)
RUN = [SEARCH, LOOKUP, REFUND]


@pytest.mark.parametrize(
    ("mode", "expected", "matched"),
    [
        ("exact", ["search", "lookup_order", "refund"], True),
        ("exact", ["search", "refund", "lookup_order"], False),
        ("exact", ["search", "lookup_order"], False),
        ("exact", ["search", "lookup_order", "refund", "notify"], False),
        ("in_order", ["search", "refund"], True),
        ("in_order", ["refund", "search"], False),
        ("in_order", [], True),
        ("any_order", ["refund", "search"], True),
        ("any_order", ["refund", "notify"], False),
        ("any_order", ["search", "search"], False),
        ("unordered", ["refund", "search", "lookup_order"], True),
        ("unordered", ["refund", "search"], False),
        ("subset", ["search", "lookup_order", "refund", "notify"], True),
        ("subset", ["search", "lookup_order"], False),
    ],
)
def test_modes(mode: Any, expected: list[str], matched: bool) -> None:
    match = match_trajectory(RUN, expected, mode=mode)

    assert bool(match) is matched
    assert match.matched is matched
    assert (match.explanation == "") is matched


def test_arguments_match_as_a_subset_by_default() -> None:
    assert match_trajectory(RUN, [sim.expect_call("lookup_order", order_id="1042")])
    assert not match_trajectory(RUN, [sim.expect_call("lookup_order", order_id="7")])
    assert not match_trajectory(RUN, [sim.expect_call("lookup_order", missing=1)])
    assert match_trajectory(RUN, [("refund", {"amount": 30}), ("search", None)], mode="any_order")


def test_exact_arguments_and_ignored_arguments() -> None:
    expected = [sim.expect_call("lookup_order", order_id="1042")]

    assert not match_trajectory(RUN, expected, args="exact")
    assert match_trajectory(RUN, [sim.expect_call("lookup_order", order_id="1042", verbose=True)], args="exact")
    assert match_trajectory(RUN, [sim.expect_call("lookup_order", order_id="nope")], args="ignore")


def test_argument_values_may_be_predicates_and_patterns() -> None:
    def small(amount: int) -> bool:
        return amount <= 50

    assert match_trajectory(RUN, [sim.expect_call("refund", amount=small)])
    assert not match_trajectory(RUN, [sim.expect_call("refund", amount=lambda amount: amount > 50)])
    assert match_trajectory(RUN, [sim.expect_call("search", q=re.compile(r"order \d+"))])
    assert not match_trajectory(RUN, [sim.expect_call("refund", amount=re.compile("30"))])
    explanation = match_trajectory([], [sim.expect_call("refund", amount=small, note=re.compile("x"))]).explanation
    assert "refund(amount=<small>, note=/x/)" in explanation


def test_a_call_without_arguments_matches_only_an_empty_expectation() -> None:
    bare = ToolCall(name="ping", input=None, execution=ToolExecution(tool_name="ping"))

    assert match_trajectory([bare], ["ping"])
    assert match_trajectory([bare], [ExpectedCall("ping", {})], args="exact")
    assert not match_trajectory([bare], [sim.expect_call("ping", host="a")])


def test_pairing_is_not_greedy() -> None:
    """A loose expectation must not take the only call a strict one accepts."""
    calls = [_call("lookup", id="2"), _call("lookup", id="1")]
    expected = ["lookup", sim.expect_call("lookup", id="2")]

    assert match_trajectory(calls, expected, mode="any_order")
    assert match_trajectory(calls, expected, mode="unordered")
    assert not match_trajectory(
        calls, [sim.expect_call("lookup", id="2"), sim.expect_call("lookup", id="2")], mode="any_order"
    )


def test_explanations_name_what_is_missing_and_what_is_unexpected() -> None:
    missing = match_trajectory(RUN, ["search", sim.expect_call("notify", channel="sms")], mode="any_order")
    assert missing.missing == (ExpectedCall("notify", {"channel": "sms"}),)
    assert "missing: notify(channel='sms')" in missing.explanation
    assert "actual: search(q='order 1042') -> lookup_order(" in missing.explanation

    extra = match_trajectory(RUN, ["search", "lookup_order"], mode="subset")
    assert extra.unexpected == (REFUND,)
    assert "unexpected: refund(order_id='1042', amount=30)" in extra.explanation

    order = match_trajectory(RUN, ["refund", "search", "lookup_order"], mode="exact")
    assert not order.missing
    assert not order.unexpected
    assert "the calls are there, in a different order" in order.explanation

    both = match_trajectory(RUN, ["search", "notify"], mode="unordered")
    assert [str(call) for call in both.missing] == ["notify"]
    assert [call.name for call in both.unexpected] == ["lookup_order", "refund"]

    none = match_trajectory([], ["search"], mode="in_order")
    assert "actual: no calls" in none.explanation
    assert "expected: no calls" in match_trajectory(RUN, [], mode="exact").explanation


def test_in_order_reports_every_expected_call_it_could_not_place() -> None:
    match = match_trajectory(RUN, ["refund", "search", "lookup_order", "notify"], mode="in_order")

    assert [str(call) for call in match.missing] == ["search", "lookup_order", "notify"]


def test_failed_calls_are_marked_in_the_explanation() -> None:
    failed = _call("refund", status="error", order_id="1")

    assert "refund(order_id='1') [error]" in match_trajectory([failed], ["search"]).explanation


def test_unknown_mode_and_args_match_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown trajectory mode 'strict'"):
        match_trajectory(RUN, [], mode="strict")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="unknown args match 'fuzzy'"):
        match_trajectory(RUN, [], args="fuzzy")  # ty: ignore[invalid-argument-type]


def test_state_match_trajectory_uses_clean_calls_unless_told_otherwise() -> None:
    state = ScenarioState(thread_id="t", name="n", description="d")
    state.add_user_message("refund order 1042")
    state.add_agent_message(
        "done",
        tools=[
            ToolExecution(tool_name="search", tool_args={"q": "1042"}, result="found"),
            ToolExecution(tool_name="refund", tool_args={"order_id": "1042"}, tool_call_error=True),
        ],
    )

    assert state.match_trajectory(["search"], mode="exact")
    assert not state.match_trajectory(["search", "refund"])
    assert state.match_trajectory(["search", sim.expect_call("refund", order_id="1042")], include_errors=True)
    assert isinstance(state.match_trajectory([]), sim.TrajectoryMatch)
    assert sim.match_trajectory is match_trajectory
