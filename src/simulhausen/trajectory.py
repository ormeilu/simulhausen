"""Tool trajectory matching: compare the calls the agent made with the calls you expect.

``state.has_tool_calls_in_order([...])`` answers one question about names.
:func:`match_trajectory` answers the usual five (the modes below), looks at
arguments too, and explains a mismatch::

    match = state.match_trajectory(
        ["search_orders", sim.expect_call("issue_refund", order_id="1042")],
        mode="exact",
    )
    assert match, match.explanation

Modes, with the agent's calls on one side and the expected calls on the other:

- ``"exact"``: the same calls in the same order, nothing else;
- ``"in_order"`` (default): the expected calls happened in this order; other
  calls may sit between them;
- ``"any_order"``: every expected call happened, in any order; other calls
  are allowed;
- ``"unordered"``: the same calls in any order, nothing else;
- ``"subset"``: the agent called nothing outside the expected calls, which
  work as an allow-list here (not all of them have to happen, repeats are fine).

Only clean executions take part unless ``include_errors=True``, as everywhere
in ``ScenarioState``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from simulhausen.types import ToolCall

__all__ = ["ArgsMatch", "ExpectedCall", "TrajectoryMatch", "TrajectoryMode", "expect_call", "match_trajectory"]

type TrajectoryMode = Literal["exact", "in_order", "any_order", "unordered", "subset"]

# How the arguments of an expected call are compared with the actual ones:
# - "subset": every expected argument is there with a matching value; the call may carry more;
# - "exact": the same argument names, each value matching;
# - "ignore": names only.
type ArgsMatch = Literal["subset", "exact", "ignore"]


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, re.Pattern):
        return isinstance(actual, str) and expected.search(actual) is not None
    if callable(expected):
        return bool(expected(actual))
    return expected == actual


@dataclass(frozen=True)
class ExpectedCall:
    """One expected tool call: a name and, optionally, arguments.

    ``args=None`` accepts any arguments. An argument value is compared with
    ``==``; a compiled regular expression is searched in a string value, and a
    callable is a predicate on the actual value
    (``expect_call("refund", amount=lambda a: a <= 100)``).
    """

    name: str
    args: Mapping[str, Any] | None = None

    def matches(self, call: ToolCall, args: ArgsMatch = "subset") -> bool:
        if call.name != self.name:
            return False
        if self.args is None or args == "ignore":
            return True
        actual = call.input or {}
        if args == "exact" and set(actual) != set(self.args):
            return False
        return all(key in actual and _value_matches(value, actual[key]) for key, value in self.args.items())

    def __str__(self) -> str:
        if self.args is None:
            return self.name
        return f"{self.name}({', '.join(f'{key}={_show(value)}' for key, value in self.args.items())})"


def _show(value: Any) -> str:
    if isinstance(value, re.Pattern):
        return f"/{value.pattern}/"
    if callable(value):
        return f"<{getattr(value, '__name__', 'predicate')}>"
    return repr(value)


def _show_call(call: ToolCall) -> str:
    arguments = ", ".join(f"{key}={value!r}" for key, value in (call.input or {}).items())
    suffix = "" if call.ok else f" [{call.status}]"
    return f"{call.name}({arguments}){suffix}"


def expect_call(name: str, /, **args: Any) -> ExpectedCall:
    """An expected call of ``name`` with these arguments: ``expect_call("lookup_order", order_id="1042")``.

    Without keyword arguments any arguments are accepted. For an argument whose
    name is not a Python identifier, build ``ExpectedCall(name, {...})``.
    """
    return ExpectedCall(name, args or None)


def _expected(item: str | ExpectedCall | tuple[str, Mapping[str, Any] | None]) -> ExpectedCall:
    if isinstance(item, ExpectedCall):
        return item
    if isinstance(item, str):
        return ExpectedCall(item)
    name, args = item
    return ExpectedCall(name, args)


@dataclass(frozen=True)
class TrajectoryMatch:
    """Outcome of :func:`match_trajectory`; truthy when the trajectory matches.

    ``missing`` are the expected calls without a counterpart, ``unexpected``
    the agent's calls the mode does not allow.
    """

    matched: bool
    mode: TrajectoryMode
    expected: tuple[ExpectedCall, ...]
    actual: tuple[ToolCall, ...]
    missing: tuple[ExpectedCall, ...] = ()
    unexpected: tuple[ToolCall, ...] = ()

    def __bool__(self) -> bool:
        return self.matched

    @property
    def explanation(self) -> str:
        """Why the trajectory does not match, for an assertion message; ``""`` when it does."""
        if self.matched:
            return ""
        parts = [f"tool trajectory does not match (mode {self.mode!r})"]
        if self.missing:
            parts.append("missing: " + ", ".join(str(call) for call in self.missing))
        if self.unexpected:
            parts.append("unexpected: " + ", ".join(_show_call(call) for call in self.unexpected))
        if not self.missing and not self.unexpected:
            parts.append("the calls are there, in a different order")
        parts.append("expected: " + (" -> ".join(str(call) for call in self.expected) or "no calls"))
        parts.append("actual: " + (" -> ".join(_show_call(call) for call in self.actual) or "no calls"))
        return "; ".join(parts)


def _pair_up(
    expected: Sequence[ExpectedCall], actual: Sequence[ToolCall], args: ArgsMatch
) -> tuple[list[int | None], list[int | None]]:
    """A largest one-to-one pairing of expected and actual calls (Kuhn's algorithm).

    Returns, for each expected call, the index of its actual call (or ``None``)
    and the reverse for each actual call. Greedy pairing is not enough: with a
    loose and a strict expectation for the same tool, the loose one can take
    the only call the strict one accepts.
    """
    actual_to_expected: list[int | None] = [None] * len(actual)

    def assign(index: int, seen: set[int]) -> bool:
        for candidate, call in enumerate(actual):
            if candidate in seen or not expected[index].matches(call, args):
                continue
            seen.add(candidate)
            holder = actual_to_expected[candidate]
            if holder is None or assign(holder, seen):
                actual_to_expected[candidate] = index
                return True
        return False

    for index in range(len(expected)):
        assign(index, set())
    expected_to_actual: list[int | None] = [None] * len(expected)
    for candidate, index in enumerate(actual_to_expected):
        if index is not None:
            expected_to_actual[index] = candidate
    return expected_to_actual, actual_to_expected


def match_trajectory(
    calls: Sequence[ToolCall],
    expected: Sequence[str | ExpectedCall | tuple[str, Mapping[str, Any] | None]],
    *,
    mode: TrajectoryMode = "in_order",
    args: ArgsMatch = "subset",
) -> TrajectoryMatch:
    """Compare ``calls`` (``state.tool_calls()``) with the expected calls; see the module docstring for the modes.

    An expected call is a tool name, an :class:`ExpectedCall`
    (:func:`expect_call`) or a ``(name, args)`` pair.
    """
    if mode not in get_args(TrajectoryMode.__value__):
        raise ValueError(f"unknown trajectory mode {mode!r}")
    if args not in get_args(ArgsMatch.__value__):
        raise ValueError(f"unknown args match {args!r}")
    wanted = tuple(_expected(item) for item in expected)
    actual = tuple(calls)

    def outcome(
        missing: Sequence[ExpectedCall], unexpected: Sequence[ToolCall], *, ordered: bool = True
    ) -> TrajectoryMatch:
        return TrajectoryMatch(
            matched=ordered and not missing and not unexpected,
            mode=mode,
            expected=wanted,
            actual=actual,
            missing=tuple(missing),
            unexpected=tuple(unexpected),
        )

    if mode == "subset":
        return outcome([], [call for call in actual if not any(item.matches(call, args) for item in wanted)])

    if mode == "in_order":
        position = 0
        missing: list[ExpectedCall] = []
        for item in wanted:
            found = next((i for i in range(position, len(actual)) if item.matches(actual[i], args)), None)
            if found is None:
                missing.append(item)
            else:
                position = found + 1
        return outcome(missing, [])

    expected_to_actual, actual_to_expected = _pair_up(wanted, actual, args)
    missing = [item for item, paired in zip(wanted, expected_to_actual, strict=True) if paired is None]
    if mode == "any_order":
        return outcome(missing, [])
    unexpected = [call for call, paired in zip(actual, actual_to_expected, strict=True) if paired is None]
    if mode == "unordered":
        return outcome(missing, unexpected)
    # exact: the same calls, and each expected call at its own position.
    in_place = len(wanted) == len(actual) and all(
        item.matches(call, args) for item, call in zip(wanted, actual, strict=True)
    )
    return outcome(missing, unexpected, ordered=in_place)
