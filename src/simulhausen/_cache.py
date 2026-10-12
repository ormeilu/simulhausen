"""Response cache: replay model answers from disk, so a re-run is fast, cheap and deterministic.

Caching is on while a scenario runs with a cache key (``configure(cache_key=...)`` or
``run(..., cache_key=...)``). Then two things are cached:

* the answers of the user simulator and the judge, at the runner's call sites;
* your own functions and methods decorated with :func:`cache`.

Without a key, or outside :func:`simulhausen.run`, a decorated function is called straight
through. The agent under test is never cached by the runner.

The idea and the ``cache`` / ``cache_key`` API follow LangWatch Scenario's ``cache.py``
(Apache-2.0, see NOTICE). This version uses the standard library only: one pickle file per
entry instead of joblib. It also fixes two weaknesses of the original key: keyword arguments
and defaults take part in it, and so does the identity of the function, so two functions called
with the same arguments no longer collide.

Cache key
=========

The key is a SHA-256 over canonical JSON of:

* the cache key string, the scenario name and the scenario description;
* the function's ``__module__`` and ``__qualname__``;
* every argument as the function sees it (positional, keyword and defaults alike), except
  ``self`` / ``cls`` (a first parameter named so) and the names listed in ``ignore``.

Arguments are turned into JSON by structure: ``None``, booleans, numbers, strings, lists,
tuples, mappings (key order does not matter), sets and dataclasses are walked recursively.
:class:`~simulhausen.types.AgentInput` counts as ``messages`` plus ``new_messages`` (no
``thread_id``, no ``scenario_state``), and :class:`~simulhausen.types.ScenarioState` as its name,
description, messages, ``current_turn`` and the tool-evidence and reference texts the judge
sees (no ``thread_id``, no private fields). Any other object is keyed by its ``repr``. **A
``repr`` that contains a memory address (``<Foo object at 0x7f...>``, a function, an open
client) differs on every run and defeats caching for that call**: pass plain data, or leave
that argument out with ``ignore``. State held on ``self`` is not part of the key either, so
pass whatever should change the answer as an argument.

Storage
=======

Entries live in ``$SIMULHAUSEN_CACHE_DIR``, else ``$XDG_CACHE_HOME/simulhausen``, else
``~/.cache/simulhausen``: one pickle file per entry, written to a temporary file and moved into
place with ``os.replace``, so a reader never sees half an entry. An entry that cannot be read
(corrupt, written by another version of a class) counts as a miss and is overwritten. A result
that cannot be pickled, or a directory that cannot be written, logs a warning and the call
still returns its result: the cache never fails a test.

Pickle runs code while loading. The cache directory must hold only what this user wrote:
never point ``SIMULHAUSEN_CACHE_DIR`` at a shared or untrusted location. Invalidate by
changing the cache key, or call :func:`clear_cache`.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import json
import logging
import os
import pickle  # nosec B403 - a local cache, written and read by the same user
import re
import tempfile
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from simulhausen.types import AgentInput, ScenarioState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Sequence

__all__ = [
    "ScenarioContext",
    "cache",
    "cache_dir",
    "cached_acall",
    "cached_call",
    "caching_context",
    "clear_cache",
    "current_context",
    "scenario_context",
]

logger = logging.getLogger(__name__)

# Bumped when the key layout or the entry layout changes: old entries then simply never match.
_FORMAT: Final = 1
_MISS: Final = object()
# Names of the files this module creates; clear_cache() touches nothing else in the directory.
_OWN_FILE: Final = re.compile(r"[0-9a-f]{64}\.(?:pkl|\w+\.tmp)")


# -- scenario context -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioContext:
    """The scenario that is running right now, as the cache sees it."""

    name: str
    description: str
    # None: no caching, decorated functions are called straight through.
    cache_key: str | None


_context: ContextVar[ScenarioContext | None] = ContextVar("simulhausen_scenario_context", default=None)


def current_context() -> ScenarioContext | None:
    """The running scenario's context, ``None`` outside :func:`simulhausen.run`."""
    return _context.get()


def caching_context() -> ScenarioContext | None:
    """The running scenario's context when it has a cache key; ``None`` means "do not cache"."""
    context = _context.get()
    return context if context is not None and context.cache_key is not None else None


@contextlib.contextmanager
def scenario_context(name: str, description: str, cache_key: str | None) -> Generator[ScenarioContext]:
    """Make the scenario current inside the block; the previous context returns afterwards, errors included.

    Each ``asyncio`` task copies the context it starts in, so scenarios run concurrently
    (``asyncio.gather``) never see each other's cache key.
    """
    context = ScenarioContext(name=name, description=description, cache_key=cache_key)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


# -- the key ------------------------------------------------------------------------


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical(value: object) -> object:
    """A JSON-ready form of ``value`` that does not depend on dict order, set order or process.

    Tagged forms (``__set__``, ``__repr__``, ...) never collide with user dicts: the keys of a
    user dict are JSON-encoded, so they always carry quotes.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, AgentInput):
        return {"__agent_input__": _canonical({"messages": value.messages, "new_messages": value.new_messages})}
    if isinstance(value, ScenarioState):
        return {
            "__scenario_state__": _canonical(
                {
                    "name": value.name,
                    "description": value.description,
                    "messages": value.messages,
                    "current_turn": value.current_turn,
                    "tool_evidence": value.tool_evidence_text(),
                    "references": value.references_text(),
                }
            )
        }
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {_dump(_canonical(key)): _canonical(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return {"__set__": sorted(_dump(_canonical(item)) for item in value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _canonical(getattr(value, field.name)) for field in fields(value)},
        }
    return {"__repr__": [f"{type(value).__module__}.{type(value).__qualname__}", repr(value)]}


def _digest(context: ScenarioContext, identity: str, arguments: Mapping[str, object]) -> str:
    payload = {
        "format": _FORMAT,
        "cache_key": context.cache_key,
        "scenario": {"name": context.name, "description": context.description},
        "function": identity,
        "arguments": _canonical(arguments),
    }
    return hashlib.sha256(_dump(payload).encode()).hexdigest()


# -- storage ------------------------------------------------------------------------


def cache_dir() -> Path:
    """Where entries live.

    ``$SIMULHAUSEN_CACHE_DIR`` wins, then ``$XDG_CACHE_HOME/simulhausen``, then ``~/.cache/simulhausen``.
    Read on every call, so changing the environment takes effect at once.
    """
    if explicit := os.environ.get("SIMULHAUSEN_CACHE_DIR"):
        return Path(explicit).expanduser()
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg).expanduser() / "simulhausen"
    return Path.home() / ".cache" / "simulhausen"


def _entry_path(context: ScenarioContext, identity: str, arguments: Mapping[str, object]) -> Path:
    return cache_dir() / f"{_digest(context, identity, arguments)}.pkl"


def _load(path: Path) -> object:
    """The stored value, or ``_MISS`` when there is no usable entry."""
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)  # nosec B301 - a local cache, written and read by the same user
    except FileNotFoundError:
        return _MISS
    except Exception as exc:  # noqa: BLE001 - whatever a damaged pickle raises, it is a miss
        logger.warning("ignoring unreadable cache entry %s: %r", path.name, exc)
        return _MISS
    if not (isinstance(envelope, tuple) and len(envelope) == 2 and envelope[0] == _FORMAT):
        logger.warning("ignoring cache entry %s: unexpected layout", path.name)
        return _MISS
    return envelope[1]


def _save(path: Path, value: object) -> None:
    """Write the entry atomically; a failure is logged and never raised."""
    try:
        payload = pickle.dumps((_FORMAT, value))
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, path)
        finally:
            # Gone after a successful replace; removes the leftover when writing failed.
            Path(temporary).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - an unpicklable result or a read-only disk must not fail the call
        logger.warning("could not store cache entry %s: %r", path.name, exc)


def cached_call[T](
    context: ScenarioContext, identity: str, arguments: Mapping[str, object], produce: Callable[[], T]
) -> T:
    """Return the stored result of ``identity(arguments)``, or run ``produce`` and store what it returns."""
    path = _entry_path(context, identity, arguments)
    stored = _load(path)
    if stored is not _MISS:
        logger.debug("cache hit %s (%s)", path.stem[:12], identity)
        return cast("T", stored)
    logger.debug("cache miss %s (%s)", path.stem[:12], identity)
    value = produce()
    _save(path, value)
    return value


async def cached_acall[T](
    context: ScenarioContext, identity: str, arguments: Mapping[str, object], produce: Callable[[], Awaitable[T]]
) -> T:
    """:func:`cached_call` for a coroutine; the file work runs in a thread, off the event loop."""
    path = _entry_path(context, identity, arguments)
    stored = await asyncio.to_thread(_load, path)
    if stored is not _MISS:
        logger.debug("cache hit %s (%s)", path.stem[:12], identity)
        return cast("T", stored)
    logger.debug("cache miss %s (%s)", path.stem[:12], identity)
    value = await produce()
    await asyncio.to_thread(_save, path, value)
    return value


def clear_cache() -> int:
    """Delete every cache entry and return how many files went.

    Only files this module wrote are removed (entries and leftover temporary files), so a
    directory shared with something else stays intact. The directory itself is kept.
    """
    directory = cache_dir()
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.iterdir():
        if _OWN_FILE.fullmatch(path.name):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


# -- the decorator ------------------------------------------------------------------


class _Decorator(Protocol):
    def __call__[**P, R](self, function: Callable[P, R], /) -> Callable[P, R]: ...


def cache(ignore: Sequence[str] = ()) -> _Decorator:
    """Cache a sync or async function (or method) while a scenario runs with a cache key.

    The first call with a given key computes and stores the result; later calls, in this run
    or in a later one, return the stored result without calling the function. Outside
    :func:`simulhausen.run`, or without a cache key, the function is called as usual. The
    wrapper keeps the signature, name and docstring, so it works on methods and static
    analysis sees the original types. Exceptions are not cached. Calls with the same key that
    run at the same time may each compute the result once.

    ``ignore`` lists parameter names left out of the key, for arguments that vary without
    changing the answer (a client, a request id). A name that is not a parameter raises
    ``ValueError`` at decoration time. A first parameter called ``self`` or ``cls`` is always
    left out; see the module docstring for what the rest of the key consists of and for the
    pickle and ``repr`` caveats.

    A scenario is found through a ``ContextVar``, so the decorated function must be called
    from the task running the scenario: a script step, an adapter, or code they await. A
    thread started without copying the context runs uncached.

    Example (with ``configure(cache_key="support-v1")`` the first run records the answer and
    later runs replay it)::

        class SupportAdapter(sim.AgentAdapter):
            async def call(self, input: sim.AgentInput) -> str:
                return await self._answer(input.last_new_user_message_str())

            @sim.cache()
            async def _answer(self, question: str) -> str:
                return await expensive_model_call(question)
    """
    if isinstance(ignore, str) or callable(ignore):
        raise TypeError(
            "cache(ignore=...) takes a sequence of parameter names; "
            "write the decorator with parentheses: @cache() or @cache(ignore=[...])"
        )

    def decorator[**P, R](function: Callable[P, R]) -> Callable[P, R]:
        identity = (
            f"{getattr(function, '__module__', type(function).__module__)}."
            f"{getattr(function, '__qualname__', type(function).__qualname__)}"
        )
        signature = inspect.signature(function)
        parameters = list(signature.parameters)
        if unknown := [name for name in ignore if name not in signature.parameters]:
            raise ValueError(f"cache(ignore=...) names {unknown}, which are not parameters of {identity}")
        skipped = set(ignore)
        if parameters and parameters[0] in ("self", "cls"):
            skipped.add(parameters[0])

        def key_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, object]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return {name: value for name, value in bound.arguments.items() if name not in skipped}

        if inspect.iscoroutinefunction(function):
            asynchronous = cast("Callable[P, Awaitable[Any]]", function)

            @functools.wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                if (context := caching_context()) is None:
                    return await asynchronous(*args, **kwargs)
                return await cached_acall(
                    context, identity, key_arguments(args, kwargs), lambda: asynchronous(*args, **kwargs)
                )

            return cast("Callable[P, R]", async_wrapper)

        @functools.wraps(function)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if (context := caching_context()) is None:
                return function(*args, **kwargs)
            return cached_call(context, identity, key_arguments(args, kwargs), lambda: function(*args, **kwargs))

        return sync_wrapper

    return decorator
