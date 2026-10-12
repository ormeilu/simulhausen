"""The response cache: keys, storage and the ``cache`` decorator (the runner side lives in test_runner.py)."""

import asyncio
import inspect
import logging
import os
import pickle
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, assert_type, cast

import pytest
from agno.models.base import Model
from agno.models.message import MessageReferences
from agno.models.response import ToolExecution

import simulhausen as sim
from simulhausen import AgentInput, CriterionVerdict, JudgeVerdict, ScenarioState, UserTurn
from simulhausen._cache import (
    ScenarioContext,
    _canonical,
    cache_dir,
    caching_context,
    current_context,
    scenario_context,
)
from simulhausen.agents import _TurnDecision

KEY = "suite-v1"


def running(
    key: str | None = KEY, name: str = "scenario", description: str = "A user tests the agent."
) -> AbstractContextManager[ScenarioContext]:
    """A scenario context with a cache key, standing in for ``run()``."""
    return scenario_context(name, description, key)


def entries(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pkl"))


def untouched(directory: Path) -> bool:
    """Nothing was ever written: not even the directory exists."""
    return not directory.exists()


class Calls:
    """Counts calls so a test can tell a hit from a miss."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, *call: Any) -> None:
        self.calls.append(call if len(call) != 1 else call[0])

    def __len__(self) -> int:
        return len(self.calls)


# -- off unless a scenario runs with a key -----------------------------------------------


def test_outside_a_run_the_function_is_called_straight_through(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    assert [double(2), double(2)] == [4, 4]
    assert seen.calls == [2, 2]
    assert not cache_dir.exists()


def test_without_a_cache_key_the_function_is_called_straight_through(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    with running(key=None):
        assert [double(2), double(2)] == [4, 4]

    assert seen.calls == [2, 2]
    assert not cache_dir.exists()


async def test_outside_a_run_an_async_function_is_called_straight_through(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    async def double(x: int) -> int:
        seen(x)
        return x * 2

    assert [await double(2), await double(2)] == [4, 4]
    with running(key=None):
        assert await double(2) == 4

    assert len(seen) == 3
    assert untouched(cache_dir)


# -- hits and misses ------------------------------------------------------------------------


def test_sync_function_hits_and_misses(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    with running():
        assert [double(2), double(2), double(3), double(2)] == [4, 4, 6, 4]

    assert seen.calls == [2, 3]
    assert len(entries(cache_dir)) == 2


async def test_async_function_hits_and_misses(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    async def double(x: int) -> int:
        seen(x)
        return x * 2

    with running():
        assert [await double(2), await double(2), await double(3), await double(2)] == [4, 4, 6, 4]

    assert seen.calls == [2, 3]
    assert len(entries(cache_dir)) == 2


def test_a_later_run_with_the_same_key_replays_the_earlier_one() -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    with running():
        double(2)
    with running():
        assert double(2) == 4

    assert seen.calls == [2]


def test_none_is_a_cacheable_result() -> None:
    seen = Calls()

    @sim.cache()
    def nothing(x: int) -> None:
        seen(x)

    with running():
        assert nothing(1) is None
        assert nothing(1) is None

    assert seen.calls == [1]


def test_exceptions_are_not_cached(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    def flaky(x: int) -> int:
        seen(x)
        if len(seen) == 1:
            raise RuntimeError("first call fails")
        return x

    with running():
        with pytest.raises(RuntimeError):
            flaky(1)
        assert flaky(1) == 1
        assert flaky(1) == 1

    assert seen.calls == [1, 1]
    assert len(entries(cache_dir)) == 1


def test_a_hit_returns_a_fresh_copy() -> None:
    @sim.cache()
    def numbers() -> list[int]:
        return [1, 2]

    with running():
        first = numbers()
        first.append(3)
        assert numbers() == [1, 2]


def test_pydantic_results_survive_the_round_trip() -> None:
    seen = Calls()

    @sim.cache()
    def turn(text: str) -> UserTurn:
        seen(text)
        return UserTurn(message=text, goal_achieved=True)

    with running():
        first = turn("hi")
    with running():
        second = turn("hi")

    assert second == first == UserTurn(message="hi", goal_achieved=True)
    assert second is not first
    assert len(seen) == 1


def test_hits_and_misses_are_logged_at_debug_level(caplog: pytest.LogCaptureFixture) -> None:
    @sim.cache()
    def double(x: int) -> int:
        return x * 2

    with caplog.at_level(logging.DEBUG, logger="simulhausen._cache"), running():
        double(1)
        double(1)

    assert [r.message.split(" (")[0].rsplit(" ", 1)[0] for r in caplog.records] == ["cache miss", "cache hit"]


# -- what the key contains ---------------------------------------------------------------


def test_keyword_arguments_and_defaults_are_part_of_the_key() -> None:
    seen = Calls()

    @sim.cache()
    def combine(a: int, b: int = 1, *, c: int = 0) -> int:
        seen(a, b, c)
        return a + b + c

    with running():
        assert combine(1) == 2
        assert combine(1, 1) == 2  # the same bound arguments as combine(1)
        assert combine(1, b=1, c=0) == 2
        assert combine(1, c=5) == 7  # only a keyword differs
        assert combine(1, b=2) == 3
        assert combine(a=1, c=5) == 7  # spelled differently, bound the same

    assert seen.calls == [(1, 1, 0), (1, 1, 5), (1, 2, 0)]


def test_variadic_arguments_are_part_of_the_key() -> None:
    seen = Calls()

    @sim.cache()
    def collect(*args: int, **kwargs: int) -> tuple[tuple[int, ...], dict[str, int]]:
        seen(args, kwargs)
        return args, kwargs

    with running():
        collect(1, 2, x=3)
        collect(1, 2, x=3)
        collect(1, 2, x=4)
        collect(1, 2)
        collect(1)

    assert seen.calls == [((1, 2), {"x": 3}), ((1, 2), {"x": 4}), ((1, 2), {}), ((1,), {})]


def test_different_functions_with_the_same_arguments_do_not_collide() -> None:
    @sim.cache()
    def first(x: int) -> str:
        return f"first {x}"

    @sim.cache()
    def second(x: int) -> str:
        return f"second {x}"

    with running():
        assert first(1) == "first 1"
        assert second(1) == "second 1"
        assert first(1) == "first 1"


def test_the_same_function_name_in_another_module_does_not_collide() -> None:
    def make(module: str) -> Callable[[int], str]:
        def answer(x: int) -> str:
            return f"{module} {x}"

        answer.__module__ = module
        return sim.cache()(answer)

    one, other = make("package.one"), make("package.two")

    assert inspect.unwrap(one).__qualname__ == inspect.unwrap(other).__qualname__
    with running():
        assert one(1) == "package.one 1"
        assert other(1) == "package.two 1"


def test_ignore_leaves_arguments_out_of_the_key() -> None:
    seen = Calls()

    @sim.cache(ignore=["request_id"])
    def ask(prompt: str, request_id: str) -> str:
        seen(prompt, request_id)
        return prompt.upper()

    with running():
        ask("a", "r1")
        ask("a", "r2")
        ask("b", "r1")

    assert seen.calls == [("a", "r1"), ("b", "r1")]


def test_ignore_rejects_names_that_are_not_parameters() -> None:
    def ask(prompt: str) -> str:
        return prompt

    with pytest.raises(ValueError, match=r"\['promt'\].*not parameters of .*ask"):
        sim.cache(ignore=["promt"])(ask)


@pytest.mark.parametrize("ignore", ["request_id", lambda: None], ids=["a string", "the decorated function"])
def test_ignore_must_be_a_sequence_of_names_and_the_decorator_needs_parentheses(ignore: Any) -> None:
    with pytest.raises(TypeError, match=r"with parentheses: @cache\(\)"):
        sim.cache(ignore)


def test_methods_leave_self_out_of_the_key() -> None:
    class Backend:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.calls = 0

        @sim.cache()
        def ask(self, question: str) -> str:
            self.calls += 1
            return f"{self.tag}: {question}"

    one, other = Backend("one"), Backend("other")
    with running():
        assert one.ask("hi") == "one: hi"
        assert other.ask("hi") == "one: hi"  # what self holds is not in the key: pass it as an argument
        assert other.ask("bye") == "other: bye"

    assert (one.calls, other.calls) == (1, 1)


async def test_async_methods_and_the_upstream_ignore_self_spelling() -> None:
    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        @sim.cache(ignore=["self"])
        async def ask(self, question: str) -> str:
            self.calls += 1
            return question

    one, other = Backend(), Backend()
    with running():
        await one.ask("hi")
        await other.ask("hi")

    assert (one.calls, other.calls) == (1, 0)


def test_classmethods_leave_cls_out_of_the_key() -> None:
    seen = Calls()

    class Factory:
        @classmethod
        @sim.cache()
        def build(cls, size: int) -> str:
            seen(cls.__name__, size)
            return f"{size}"

    class Sub(Factory):
        pass

    with running():
        Factory.build(1)
        Sub.build(1)  # a hit: the class is not part of the key
        Sub.build(2)

    assert seen.calls == [("Factory", 1), ("Sub", 2)]


def test_only_a_first_parameter_named_self_is_dropped() -> None:
    seen = Calls()

    @sim.cache()
    def pair(x: int, self: int) -> int:
        seen(x, self)
        return x + self

    with running():
        pair(1, 1)
        pair(1, 2)

    assert len(seen) == 2


def test_the_cache_key_scenario_name_and_description_are_part_of_the_key() -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    for key, name, description in [
        ("suite-v1", "refund", "wants money back"),
        ("suite-v1", "refund", "wants money back"),
        ("suite-v2", "refund", "wants money back"),
        ("suite-v1", "refund two", "wants money back"),
        ("suite-v1", "refund", "wants the money back"),
    ]:
        with running(key, name, description):
            double(1)

    assert len(seen) == 4


def _input(thread_id: str = "t1", messages: list[str] | None = None, new: list[str] | None = None) -> AgentInput:
    def turns(texts: list[str]) -> Any:
        return [{"role": "user", "content": text} for text in texts]

    return AgentInput(thread_id=thread_id, messages=turns(messages or ["hi"]), new_messages=turns(new or []))


def test_agent_input_ignores_the_thread_id_and_the_state_but_not_the_messages() -> None:
    seen = Calls()

    @sim.cache()
    def answer(input: AgentInput) -> str:
        seen(input.thread_id)
        return input.last_new_user_message_str()

    other_state = ScenarioState(thread_id="x", name="other", description="other")
    inputs = [
        _input("t1"),
        _input("t2"),  # another thread: a hit
        AgentInput(thread_id="t3", messages=_input().messages, scenario_state=other_state),  # a hit
        _input("t4", messages=["hi", "there"]),  # other messages: a miss
        _input("t5", new=["hi"]),  # other new_messages: a miss
    ]
    with running():
        for item in inputs:
            answer(item)

    assert seen.calls == ["t1", "t4", "t5"]


def _state(**changes: Any) -> ScenarioState:
    state = ScenarioState(thread_id="t1", name="refund", description="wants money back")
    state.add_user_message("hi")
    state.add_agent_message("hello")
    for field, value in changes.items():
        setattr(state, field, value)
    return state


def test_scenario_state_ignores_the_thread_id_and_private_fields() -> None:
    seen = Calls()

    @sim.cache()
    def judge(state: ScenarioState) -> str:
        seen(state.thread_id)
        return state.name

    other = _state(thread_id="t2")
    other._agent_seen = 2
    other._origins = {1: 0}
    with running():
        judge(_state())
        judge(other)

    assert len(seen) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "other name"},
        {"description": "other description"},
        {"messages": [{"role": "user", "content": "other"}]},
        {"current_turn": 3},
        {"tool_executions": [ToolExecution(tool_name="lookup", tool_args={"id": 1}, result="found")]},
        {"references": [MessageReferences(query="q", references=[{"content": "a document"}])]},
    ],
    ids=["name", "description", "messages", "current_turn", "tools", "references"],
)
def test_scenario_state_changes_that_the_judge_sees_change_the_key(changes: dict[str, Any]) -> None:
    seen = Calls()

    @sim.cache()
    def judge(state: ScenarioState) -> str:
        seen(state.name)
        return state.name

    with running():
        judge(_state())
        judge(_state(**changes))

    assert len(seen) == 2


def test_dict_order_and_set_order_do_not_matter() -> None:
    seen = Calls()

    @sim.cache()
    def take(value: object) -> str:
        seen(value)
        return "ok"

    first, second = set(), set()
    first.update([0, 8, 16])
    second.update([16, 8, 0])
    if list(first) == list(second):  # pragma: no cover - CPython orders colliding ints by insertion
        pytest.skip("this interpreter iterates equal sets in the same order")

    with running():
        take({"a": 1, "b": 2})
        take({"b": 2, "a": 1})
        take(first)
        take(second)

    assert len(seen) == 2


def test_dict_keys_of_different_types_do_not_collide() -> None:
    seen = Calls()

    @sim.cache()
    def take(value: dict[object, str]) -> str:
        seen(value)
        return "ok"

    with running():
        take({1: "a", "1": "b"})
        take({"1": "b", 1: "a"})  # the same dict, another order
        take({1: "b", "1": "a"})  # same keys and values, paired differently

    assert len(seen) == 2


@dataclass
class Options:
    temperature: float
    tags: list[str]


class Plain:
    """An object with a stable repr."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Plain({self.value})"


class Opaque:
    """An object whose repr carries a memory address."""


def test_dataclasses_tuples_and_objects_with_a_stable_repr_are_keyed_by_content() -> None:
    seen = Calls()

    @sim.cache()
    def take(*values: object) -> str:
        seen(values)
        return "ok"

    with running():
        take(Options(0.1, ["a"]), ("x", 1), Plain(1))
        take(Options(0.1, ["a"]), ("x", 1), Plain(1))
        take(Options(0.2, ["a"]), ("x", 1), Plain(1))
        take(Options(0.1, ["a"]), ("x", 2), Plain(1))
        take(Options(0.1, ["a"]), ("x", 1), Plain(2))

    assert len(seen) == 4


def test_a_repr_with_a_memory_address_defeats_the_cache() -> None:
    seen = Calls()

    @sim.cache()
    def take(value: object) -> str:
        seen(value)
        return "ok"

    shared = Opaque()
    with running():
        take(Opaque())
        take(Opaque())  # another address: a miss, as the docs warn
        take(shared)
        take(shared)

    assert len(seen) == 3


def test_a_repr_is_not_mistaken_for_a_string_or_a_user_dict() -> None:
    assert _canonical(Decimal(1)) != _canonical("Decimal('1')")
    assert _canonical({1, 2}) != _canonical({"__set__": ["1", "2"]})
    assert _canonical({"__repr__": 1}) != _canonical(Opaque())


# -- storage ---------------------------------------------------------------------------------


def _stored_entry(cache_dir: Path) -> Path:
    [entry] = entries(cache_dir)
    return entry


@pytest.mark.parametrize(
    "garbage",
    [
        b"",
        b"not a pickle",
        pickle.dumps({"value": 1})[:-3],
        b"cnonexistent_module\nThing\n.",
        pickle.dumps(42),
        pickle.dumps((0, "an older entry layout")),
    ],
    ids=["empty", "garbage", "truncated", "missing class", "wrong shape", "other layout"],
)
def test_a_damaged_entry_is_a_miss_and_gets_overwritten(
    cache_dir: Path, garbage: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    with running():
        double(2)
    entry = _stored_entry(cache_dir)
    entry.write_bytes(garbage)

    with caplog.at_level(logging.WARNING, logger="simulhausen._cache"), running():
        assert double(2) == 4  # recomputed
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert entry.name in caplog.records[0].message

    with running():
        assert double(2) == 4  # served from the rewritten entry
    assert len(seen) == 2
    assert pickle.loads(entry.read_bytes())[1] == 4


def test_a_write_leaves_only_the_finished_entry_and_goes_through_an_atomic_replace(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moves: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(source: Any, target: Any) -> None:
        moves.append((Path(source), Path(target)))
        real_replace(source, target)

    @sim.cache()
    def double(x: int) -> int:
        return x * 2

    with monkeypatch.context() as patch, running():
        patch.setattr(os, "replace", spy)
        double(1)
    with running():
        double(2)

    assert [path.suffix for path in cache_dir.iterdir()] == [".pkl", ".pkl"]
    [(source, target)] = moves
    assert source.parent == target.parent == cache_dir  # same directory: the replace stays on one filesystem
    assert source.name != target.name


def test_a_failed_write_is_logged_leaves_no_temporary_file_and_keeps_the_result(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(source: Any, target: Any) -> None:
        raise OSError("disk on fire")

    @sim.cache()
    def double(x: int) -> int:
        return x * 2

    with monkeypatch.context() as patch, caplog.at_level(logging.WARNING, logger="simulhausen._cache"), running():
        patch.setattr(os, "replace", broken)
        assert double(1) == 2

    assert list(cache_dir.iterdir()) == []
    assert "disk on fire" in caplog.text


def test_an_unpicklable_result_is_returned_and_logged_not_stored(
    cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    @sim.cache()
    def make(x: int) -> Callable[[], int]:
        return lambda: x

    with caplog.at_level(logging.WARNING, logger="simulhausen._cache"), running():
        assert make(1)() == 1

    assert not cache_dir.exists()
    assert "could not store cache entry" in caplog.text


def test_a_cache_directory_that_cannot_be_used_never_fails_the_call(cache_dir: Path) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.write_text("a file where the directory should be")
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    with running():
        assert double(2) == 4
        assert double(2) == 4

    assert len(seen) == 2
    assert cache_dir.read_text() == "a file where the directory should be"


def test_an_entry_path_that_is_a_directory_is_a_miss_and_the_write_fails_quietly(cache_dir: Path) -> None:
    @sim.cache()
    def double(x: int) -> int:
        return x * 2

    with running():
        double(2)
    entry = _stored_entry(cache_dir)
    entry.unlink()
    entry.mkdir()

    with running():
        assert double(2) == 4

    assert entry.is_dir()
    assert [path.name for path in cache_dir.iterdir()] == [entry.name]


def test_clear_cache_removes_the_entries_and_nothing_else(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    with running():
        double(1)
        double(2)
    foreign = [cache_dir / "notes.pkl", cache_dir / "data.txt", cache_dir / f"{'g' * 64}.pkl"]
    for path in foreign:
        path.write_text("keep me")
    (cache_dir / f"{'a' * 64}.x1y2z3.tmp").write_text("a leftover of an interrupted write")
    os.close(tempfile.mkstemp(dir=cache_dir, prefix=f"{'b' * 64}.", suffix=".tmp")[0])  # a real one

    assert sim.clear_cache() == 4
    assert sorted(path.name for path in cache_dir.iterdir()) == sorted(path.name for path in foreign)

    with running():
        double(1)
    assert seen.calls == [1, 2, 1]


def test_clear_cache_without_a_directory_does_nothing(cache_dir: Path) -> None:
    assert not cache_dir.exists()
    assert sim.clear_cache() == 0
    assert not cache_dir.exists()


def test_cache_dir_prefers_the_simulhausen_variable_then_xdg_then_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIMULHAUSEN_CACHE_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    assert cache_dir() == tmp_path / "explicit"

    monkeypatch.delenv("SIMULHAUSEN_CACHE_DIR")
    assert cache_dir() == tmp_path / "xdg" / "simulhausen"

    monkeypatch.delenv("XDG_CACHE_HOME")
    assert cache_dir() == tmp_path / "home" / ".cache" / "simulhausen"


def test_an_empty_variable_counts_as_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULHAUSEN_CACHE_DIR", "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "simulhausen"

    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert cache_dir() == tmp_path / "home" / ".cache" / "simulhausen"


def test_a_tilde_in_the_variables_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("SIMULHAUSEN_CACHE_DIR", "~/explicit")
    assert cache_dir() == tmp_path / "explicit"

    monkeypatch.delenv("SIMULHAUSEN_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", "~/xdg")
    assert cache_dir() == tmp_path / "xdg" / "simulhausen"


def test_the_directory_is_read_on_every_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = Calls()

    @sim.cache()
    def double(x: int) -> int:
        seen(x)
        return x * 2

    for name in ("one", "two"):
        monkeypatch.setenv("SIMULHAUSEN_CACHE_DIR", str(tmp_path / name))
        with running():
            double(1)

    assert len(seen) == 2
    assert len(entries(tmp_path / "one")) == len(entries(tmp_path / "two")) == 1


# -- the decorator itself ------------------------------------------------------------------


def test_the_wrapper_keeps_the_name_docstring_and_signature() -> None:
    def add(a: int, b: int = 2) -> int:
        """Add two numbers."""
        return a + b

    wrapped = sim.cache()(add)

    for attribute in ("__name__", "__qualname__", "__module__", "__doc__"):
        assert getattr(wrapped, attribute) == getattr(add, attribute)
    assert wrapped.__doc__ == "Add two numbers."
    assert inspect.unwrap(wrapped) is add
    assert inspect.signature(wrapped) == inspect.signature(add)
    assert not inspect.iscoroutinefunction(wrapped)


async def test_the_async_wrapper_stays_a_coroutine_function() -> None:
    async def fetch(url: str) -> str:
        return url

    wrapped = sim.cache()(fetch)

    assert inspect.iscoroutinefunction(wrapped)
    assert inspect.signature(wrapped) == inspect.signature(fetch)
    assert await wrapped("a") == "a"


async def test_the_decorated_function_keeps_its_types() -> None:
    @sim.cache(ignore=["flag"])
    def add(a: int, b: int = 2, flag: bool = False) -> int:
        return a + b

    @sim.cache()
    async def fetch(url: str) -> str:
        return url

    assert_type(add(1), int)
    assert_type(await fetch("a"), str)
    assert (add(1), await fetch("a")) == (3, "a")


# -- the scenario context ------------------------------------------------------------------


def test_scenario_context_restores_the_previous_one_on_exit() -> None:
    assert current_context() is None
    assert caching_context() is None

    with scenario_context("outer", "outer description", "key") as outer:
        assert current_context() is outer
        assert caching_context() is outer
        with scenario_context("inner", "inner description", None) as inner:
            assert current_context() is inner
            assert caching_context() is None  # no key: not caching
        assert current_context() is outer

    assert current_context() is None
    assert (outer.name, outer.description, outer.cache_key) == ("outer", "outer description", "key")


def test_scenario_context_is_reset_after_an_error() -> None:
    with pytest.raises(RuntimeError), scenario_context("boom", "d", "key"):
        raise RuntimeError("inside the scenario")

    assert current_context() is None


async def test_scenarios_running_at_the_same_time_keep_their_own_keys(cache_dir: Path) -> None:
    seen = Calls()

    @sim.cache()
    async def slow(x: int) -> int:
        seen(x)
        await asyncio.sleep(0)
        return x

    async def scenario(key: str | None) -> None:
        with scenario_context("scenario", "d", key):
            await slow(1)
            context = current_context()
            assert context is not None
            assert context.cache_key == key

    await asyncio.gather(scenario("a"), scenario("b"), scenario(None))
    assert len(seen) == 3
    assert len(entries(cache_dir)) == 2

    await asyncio.gather(scenario("a"), scenario("b"), scenario(None))
    assert len(seen) == 4  # only the run without a key called the function again
    assert current_context() is None


# -- the real helper agents, with a stubbed model call --------------------------------------


async def test_the_real_helper_agents_replay_a_dialogue_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[type] = []

    class StubAgent:
        def __init__(self, **kwargs: Any) -> None:
            self.schema: type = kwargs["output_schema"]

        async def arun(self, prompt: str) -> SimpleNamespace:
            asked.append(self.schema)
            answers: dict[type, object] = {
                UserTurn: UserTurn(message="I was charged twice", goal_achieved=False),
                _TurnDecision: _TurnDecision(decision="conclude"),
                JudgeVerdict: JudgeVerdict(
                    verdicts=[CriterionVerdict(criterion="x", reasoning="polite enough", status="passed")],
                    summary="fine",
                ),
            }
            return SimpleNamespace(content=answers[self.schema])

    monkeypatch.setattr("simulhausen.agents.Agent", StubAgent)
    sim.configure(model=cast("Model", SimpleNamespace(id="stub-model")), cache_key=KEY)

    async def reply(input: AgentInput) -> str:
        return "Let me check that for you."

    async def play_once() -> sim.ScenarioResult:
        return await sim.run(
            "refund",
            "The user was charged twice.",
            [sim.function_adapter(reply), sim.UserSimulatorAgent(), sim.JudgeAgent(criteria=["is polite"])],
            max_turns=3,
        )

    first = await play_once()
    model_calls = list(asked)
    second = await play_once()

    assert model_calls == [UserTurn, _TurnDecision, JudgeVerdict]
    assert asked == model_calls  # the second run asked no model
    assert first.success
    assert [(v.criterion, v.status, v.reasoning) for v in second.verdicts] == [("is polite", "passed", "polite enough")]
    assert second.messages == first.messages
    assert second.reasoning == first.reasoning == "fine"
