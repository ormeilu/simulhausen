"""Ready-made adapters: agno Agent/Team/Workflow, and plain functions.

``agno_adapter`` builds the usual adapter (``arun`` in one session, an
``AgentTurn`` with content, tools and references) in one call, so tests do not
need a wrapper class each. ``function_adapter`` wraps any sync or async
callable. Subclass :class:`~simulhausen.types.AgentAdapter` for anything else.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

from simulhausen.exceptions import SimulationBackendError
from simulhausen.types import AgentAdapter, AgentInput, AgentReturnTypes, AgentTurn

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agno.agent import Agent
    from agno.team import Team
    from agno.workflow import Workflow

__all__ = ["agno_adapter", "function_adapter"]

logger = logging.getLogger(__name__)

# Transient backend failures (a 504 and the like) are retried a few times: runs
# go through a live inference gateway, and one timeout must not sink a scenario.
DEFAULT_BACKEND_RETRIES = 2
DEFAULT_BACKEND_RETRY_DELAY_S = 2.0

# agno error types that mean an input/output check refused the request (a
# guardrail), which is a product answer, not an infrastructure failure. See
# agno/exceptions.py: InputCheckError.type == "input_check_error".
_CHECK_ERROR_TYPES = frozenset({"input_check_error", "output_check_error"})

_ERROR_PAGE_MARKERS = ("gateway time-out", "bad gateway", "service unavailable", "gateway timeout")


class _Runnable(Protocol):
    async def arun(self, input: Any, **kwargs: Any) -> Any: ...


type AgnoTarget = Agent | Team | Workflow | Callable[[], Agent | Team | Workflow]


def _looks_like_error_page(text: str) -> bool:
    """Whether the content is a gateway's HTML error page rather than a model reply."""
    head = text.lstrip()[:300].lower()
    if head.startswith(("<!doctype", "<html", "<head")):
        return True
    return any(marker in head for marker in _ERROR_PAGE_MARKERS)


def _is_check_refusal(response: Any) -> bool:
    """Did the run end with an input/output check refusing the request?

    On InputCheckError agno sets RunStatus=error and puts the refusal text in
    ``content``. Formally an error, but for a scenario it is the product's
    answer ("Sorry, I can't help with that"), which is exactly what a guardrail
    scenario wants to check. Without this branch the adapter treated a
    guardrail as a backend failure, retried, and raised SimulationBackendError.
    """
    return any(
        getattr(event, "error_type", None) in _CHECK_ERROR_TYPES for event in getattr(response, "events", None) or []
    )


def _backend_error_reason(response: Any) -> str | None:
    """Why the run counts as a backend failure, or ``None`` when it does not.

    Two cases: agno reported RunStatus=error (guardrail refusals excluded, see
    ``_is_check_refusal``), or the model returned a raw HTML error page from a
    proxy instead of text or a structured object.
    """
    status = getattr(response, "status", None)
    status_name = getattr(status, "value", None) or getattr(status, "name", None) or str(status)
    if isinstance(status_name, str) and status_name.lower() == "error":
        if _is_check_refusal(response):
            return None
        return "agno RunStatus=error"
    content = getattr(response, "content", None)
    if isinstance(content, str) and _looks_like_error_page(content):
        return "backend returned a gateway HTML error page: " + " ".join(content.split())[:120]
    return None


def _collect_tools(response: Any) -> list[Any] | None:
    """Tool executions of the run, team members' included.

    For a Team, ``response.tools`` holds only the leader's calls (delegation),
    while the domain tools live in ``member_responses[*].tools``. Without
    merging them a team scenario cannot assert ``state.has_tool_call(...)`` on
    a member tool. Agents have no ``member_responses``, so nothing changes there.
    """
    own = list(getattr(response, "tools", None) or [])
    members = [
        tool
        for member in getattr(response, "member_responses", None) or []
        for tool in getattr(member, "tools", None) or []
    ]
    return (own + members) or None


class _AgnoAdapter(AgentAdapter):
    """Last user message -> ``target.arun`` -> ``AgentTurn``."""

    def __init__(
        self,
        target: AgnoTarget,
        arun_kwargs: dict[str, Any],
        content_fn: Callable[[Any], str] | None,
        retries: int,
        retry_delay: float,
    ) -> None:
        self._target = target
        self._resolved: _Runnable | None = None
        self._arun_kwargs = arun_kwargs
        self._content_fn = content_fn
        self._retries = retries
        self._retry_delay = retry_delay

    def _resolve(self) -> _Runnable:
        if self._resolved is None:
            target = self._target
            # A factory (callable without arun) is called once and cached.
            resolved = target() if callable(target) and not hasattr(target, "arun") else target
            self._resolved = cast("_Runnable", resolved)
        return self._resolved

    async def call(self, input: AgentInput) -> AgentTurn:
        target = self._resolve()
        message = input.last_new_user_message_str()
        attempts = self._retries + 1
        reason: str | None = None
        response: Any = None
        for attempt in range(1, attempts + 1):
            response = await target.arun(message, session_id=input.thread_id, **self._arun_kwargs)
            reason = _backend_error_reason(response)
            if reason is None:
                break
            logger.warning("backend error (attempt %d/%d): %s", attempt, attempts, reason)
            if attempt < attempts:
                await asyncio.sleep(self._retry_delay * attempt)
        else:
            # Every attempt hit the backend failure: raise a typed error instead
            # of passing an HTML page on as the reply and failing a tool assertion.
            raise SimulationBackendError(f"backend failed {attempts} time(s) in a row: {reason}")

        raw = response.content
        content = self._content_fn(raw) if self._content_fn is not None else str(raw or "")
        return AgentTurn(
            content=content,
            tools=_collect_tools(response),
            references=getattr(response, "references", None),
        )


def agno_adapter(
    target: AgnoTarget,
    *,
    content_fn: Callable[[Any], str] | None = None,
    retries: int = DEFAULT_BACKEND_RETRIES,
    retry_delay: float = DEFAULT_BACKEND_RETRY_DELAY_S,
    **arun_kwargs: Any,
) -> AgentAdapter:
    """Adapter for an agno Agent, Team or Workflow.

    ``target`` is the object itself or a lazy factory (``lambda: my_agent``
    with the import inside, so collecting tests does not build the agent).
    Extra kwargs go to ``arun`` unchanged: ``session_state=...``,
    ``user_id=...`` and so on. ``session_id`` is always the simulation's
    ``thread_id``, so the whole scenario runs in one session.

    ``content_fn`` renders ``response.content`` into the reply text. Agents
    with ``output_schema`` need it, since their content is a Pydantic model;
    otherwise the content goes through ``str(...)``.

    ``retries``/``retry_delay`` control how backend failures (RunStatus=error,
    a gateway HTML error page) are retried before ``SimulationBackendError``.

    >>> adapter = agno_adapter(lambda: support_agent, session_state={"plan": "pro"})
    """
    if retries < 0:
        raise ValueError("retries must be >= 0")
    return _AgnoAdapter(target, arun_kwargs, content_fn=content_fn, retries=retries, retry_delay=retry_delay)


class _FunctionAdapter(AgentAdapter):
    def __init__(self, fn: Callable[[AgentInput], AgentReturnTypes | Awaitable[AgentReturnTypes]]) -> None:
        self._fn = fn

    async def call(self, input: AgentInput) -> AgentReturnTypes:
        result = self._fn(input)
        if inspect.isawaitable(result):
            return await result
        return result


def function_adapter(
    fn: Callable[[AgentInput], AgentReturnTypes | Awaitable[AgentReturnTypes]],
) -> AgentAdapter:
    """Adapter around a sync or async function ``AgentInput -> str | AgentTurn``.

    >>> async def echo(input: AgentInput) -> str:
    ...     return input.last_new_user_message_str()
    >>> adapter = function_adapter(echo)
    """
    return _FunctionAdapter(fn)
