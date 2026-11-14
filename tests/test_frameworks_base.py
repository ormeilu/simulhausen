import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simulhausen import AgentInput, Usage
from simulhausen.frameworks._base import (
    Conversations,
    FrameworkAdapter,
    maybe_await,
    require,
    result_text,
    text_of,
    tool_execution,
    transcript_prompt,
    usage_of,
)


class _Reply(BaseModel):
    answer: str


def test_require_returns_the_module_and_names_a_missing_package() -> None:
    assert require("json", package="json", adapter="x_adapter").dumps({}) == "{}"
    with pytest.raises(ImportError, match=r"simulhausen\.x_adapter needs the 'no-such-pkg' package"):
        require("no_such_module_anywhere", package="no-such-pkg", adapter="x_adapter")


async def test_maybe_await() -> None:
    async def coro() -> int:
        return 2

    assert await maybe_await(1) == 1
    assert await maybe_await(coro()) == 2


def test_text_of() -> None:
    assert text_of(None) == ""
    assert text_of("hi") == "hi"
    assert text_of([{"type": "text", "text": "a"}, {"type": "tool_use"}, "b"]) == "ab"
    assert text_of(_Reply(answer="x")) == '{"answer":"x"}'
    assert text_of({"k": "ё"}) == '{"k": "ё"}'


def test_result_text() -> None:
    assert result_text(None) is None
    assert result_text("ok") == "ok"
    assert result_text([{"type": "text", "text": "a"}]) == "a"
    assert result_text([1, 2]) == "[1, 2]"
    assert result_text(_Reply(answer="x")) == '{"answer":"x"}'
    assert result_text({"n": 1}) == '{"n": 1}'

    class Loop:
        def __init__(self) -> None:
            self.me = self

    loop: Any = {}
    loop["self"] = loop
    assert result_text(loop) == str(loop)


def test_tool_execution_statuses() -> None:
    clean = tool_execution("lookup", '{"id": 1}', {"found": True}, call_id=7)
    assert (clean.tool_name, clean.tool_args, clean.result, clean.tool_call_id) == (
        "lookup",
        {"id": 1},
        '{"found": true}',
        "7",
    )
    assert not clean.tool_call_error
    assert not clean.is_paused
    failed = tool_execution("lookup", {"id": 1}, "boom", error=True)
    assert failed.tool_call_error is True
    asked = tool_execution("lookup", None, executed=False)
    assert asked.is_paused
    assert asked.tool_args == {}


def test_usage_of_reads_every_naming() -> None:
    assert usage_of(None) is None
    assert usage_of({}) is None
    assert usage_of({"prompt_tokens": 3, "completion_tokens": 4}) == Usage(3, 4, 7)
    assert usage_of(SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=5, requests=2)) == Usage(
        1, 2, 5, requests=2
    )
    assert usage_of({"prompt_token_count": 1, "candidates_token_count": 1, "total_token_count": 2}, requests=3) == (
        Usage(1, 1, 2, requests=3)
    )
    assert usage_of({"inputTokens": 2, "outputTokens": 1, "totalTokens": 3, "cost": 0.5}) == Usage(2, 1, 3, cost=0.5)
    # A boolean is not a count.
    assert usage_of({"input_tokens": True}) is None


def test_transcript_prompt() -> None:
    assert transcript_prompt([{"role": "user", "content": "hi"}]) == "hi"
    rendered = transcript_prompt(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "?"}]
    )
    assert "User: hi\nAssistant: yo\nUser: ?" in rendered


def _input(messages: list[tuple[str, str]], new: int = 1, thread_id: str = "t") -> AgentInput:
    transcript: Any = [{"role": role, "content": content} for role, content in messages]
    return AgentInput(thread_id=thread_id, messages=transcript, new_messages=transcript[len(transcript) - new :])


def test_conversations_follow_the_transcript() -> None:
    store: Conversations[list[str]] = Conversations()
    first = _input([("user", "hi")])
    assert store.recall(first) is None
    store.remember(first, "hello", ["native"])

    follow_up = _input([("user", "hi"), ("assistant", "hello"), ("user", "more")])
    assert store.recall(follow_up) == ["native"]
    # Another thread knows nothing.
    assert store.recall(_input([("user", "hi"), ("assistant", "hello"), ("user", "more")], thread_id="u")) is None
    # A rollback replaced the exchange: the stored state no longer matches.
    assert store.recall(_input([("user", "other")])) is None
    # A scripted reply the agent never produced.
    injected = _input([("user", "hi"), ("assistant", "hello"), ("assistant", "injected"), ("user", "more")])
    assert store.recall(injected) is None
    # Nothing new since the agent spoke: no way to tell what to send.
    assert store.recall(_input([("user", "hi"), ("assistant", "hello")], new=0)) is None

    store.forget("t")
    assert store.recall(follow_up) is None
    store.forget("t")


class _Adapter(FrameworkAdapter):
    def _is_target(self, target: Any) -> bool:
        return hasattr(target, "run")

    async def call(self, input: AgentInput) -> str:
        return self._render(self._resolve().run())


async def test_framework_adapter_resolves_factories_once_and_renders() -> None:
    built: list[int] = []

    class Target:
        def __call__(self) -> str:  # the agent is callable itself, like in several frameworks
            raise AssertionError("an agent must not be mistaken for a factory")

        def run(self) -> Any:
            return [{"type": "text", "text": "reply"}]

    def factory() -> Target:
        built.append(1)
        return Target()

    adapter = _Adapter(factory)
    agent_input = _input([("user", "hi")])
    assert await adapter.call(agent_input) == "reply"
    assert await adapter.call(agent_input) == "reply"
    assert built == [1]
    assert await _Adapter(Target()).call(agent_input) == "reply"
    assert await _Adapter(Target(), content_fn=lambda raw: f"{len(raw)} block").call(agent_input) == "1 block"


def test_framework_adapter_rejects_an_unknown_history_and_keeps_plain_objects() -> None:
    with pytest.raises(ValueError, match="history must be"):
        _Adapter(object(), history="nope")  # ty: ignore[invalid-argument-type]
    plain = object()
    # Neither a target by shape nor callable: used as it is.
    assert _Adapter(plain)._resolve() is plain


async def test_by_default_anything_callable_is_a_factory() -> None:
    class Plain(FrameworkAdapter):
        async def call(self, input: AgentInput) -> str:
            return str(self._resolve())

    agent_input = _input([("user", "hi")])
    assert await Plain(lambda: "built").call(agent_input) == "built"
    assert await Plain("as is").call(agent_input) == "as is"


FRAMEWORK_MODULES = {
    "ag2",
    "agent_framework",
    "agents",
    "autogen_agentchat",
    "camel",
    "claude_agent_sdk",
    "crewai",
    "deepagents",
    "dspy",
    "google",
    "haystack",
    "langchain",
    "langchain_core",
    "langgraph",
    "llama_index",
    "pydantic_ai",
    "smolagents",
    "strands",
}

ADAPTERS = [
    "ag2_adapter",
    "agent_framework_adapter",
    "autogen_adapter",
    "camel_adapter",
    "claude_agent_adapter",
    "crewai_adapter",
    "dspy_adapter",
    "google_adk_adapter",
    "haystack_adapter",
    "langchain_adapter",
    "llamaindex_adapter",
    "openai_agents_adapter",
    "pydantic_ai_adapter",
    "smolagents_adapter",
    "strands_adapter",
]


def test_importing_simulhausen_loads_no_framework() -> None:
    # A fresh interpreter: what this process has loaded says nothing.
    code = (
        "import sys, simulhausen\n"
        f"print(sorted({{name.split('.')[0] for name in sys.modules}} & set({sorted(FRAMEWORK_MODULES)!r})))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("name", ADAPTERS)
def test_every_framework_adapter_is_exported_and_builds_without_its_framework(name: str) -> None:
    import simulhausen as sim

    factory = getattr(sim, name)
    assert name in sim.__all__
    # Building an adapter around a lazy factory imports nothing and calls nothing.
    assert isinstance(factory(lambda: None), sim.AgentAdapter)
    with pytest.raises(ValueError, match="history must be"):
        factory(lambda: None, history="nope")
