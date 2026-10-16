import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from simulhausen import AgentAdapter, AgentInput, AgentTurn
from simulhausen.harbor import ConfigError, ScenarioSpec, build_target
from simulhausen.harbor import targets as targets_module
from simulhausen.harbor.targets import import_object, redact_target, resolve_target_config


async def _ask(adapter: AgentAdapter, text: str = "hello") -> Any:
    return await adapter.call(AgentInput(thread_id="t", messages=[{"role": "user", "content": text}]))


@pytest.fixture
def http_adapters(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """A stand-in for simulhausen.http_adapters whose factories record their keyword arguments."""
    calls: list[dict[str, Any]] = []
    module = ModuleType("simulhausen.http_adapters")

    def agentos_adapter(base_url: str, **options: Any) -> AgentAdapter:
        calls.append({"base_url": base_url, **options})
        from harbor_support import support_agent

        return support_agent()

    def strict_adapter(base_url: str) -> AgentAdapter:
        raise AssertionError("not reached")

    module.agentos_adapter = agentos_adapter  # ty: ignore[unresolved-attribute]
    module.openai_chat_adapter = strict_adapter  # ty: ignore[unresolved-attribute]
    module.a2a_adapter = "not callable"  # ty: ignore[unresolved-attribute]
    module.langgraph_adapter = lambda **options: "no adapter"  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "simulhausen.http_adapters", module)
    return calls


# -- python targets ------------------------------------------------------------


async def test_python_factory_gets_the_other_keys_as_keyword_arguments() -> None:
    adapter = build_target({"type": "python", "factory": "harbor_support:support_agent", "prefix": "bot"})

    turn = await _ask(adapter)

    assert isinstance(turn, AgentTurn)
    assert turn.content == "bot reply 1: hello"


async def test_python_factory_may_return_or_be_an_agno_runnable() -> None:
    from_factory = build_target({"type": "python", "factory": "harbor_support:agno_like_factory"})
    from_instance = build_target({"type": "python", "factory": "harbor_support:agno_like_instance"})
    adapter_attr = build_target({"type": "python", "factory": "harbor_support:adapter_instance"})

    assert (await _ask(from_factory)).content == "agno says: hello"
    assert (await _ask(from_instance)).content == "agno says: hello"
    assert (await _ask(adapter_attr)).content == "agent reply 1: hello"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"type": "python"}, 'needs `factory = "package.module:attribute"`'),
        ({"type": "python", "factory": "harbor_support"}, "is not an import string"),
        ({"type": "python", "factory": "no_such_module_xyz:make"}, "cannot import 'no_such_module_xyz'"),
        ({"type": "python", "factory": "harbor_support:nope"}, "has no attribute 'nope'"),
        ({"type": "python", "factory": "harbor_support:not_callable"}, "is a int, not a callable"),
        (
            {"type": "python", "factory": "harbor_support:broken_factory"},
            "returned str; it must return an AgentAdapter",
        ),
        ({"type": "python", "factory": "harbor_support:support_agent", "colour": "red"}, "rejected the options"),
        (
            {"type": "python", "factory": "harbor_support:agno_like_instance", "colour": "red"},
            "is not callable, so it takes no options: ['colour']",
        ),
        ({"base_url": "x"}, "a target needs a `type`"),
        ({"type": "carrier-pigeon"}, "unknown target type 'carrier-pigeon'; known types: agentos, openai-chat"),
    ],
)
def test_target_errors_say_what_to_fix(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError) as info:
        build_target(config)
    assert message in str(info.value)


def test_import_object_follows_dotted_attributes() -> None:
    assert import_object("harbor_support:AgnoLike.arun", what="x") is not None


# -- environment indirection ---------------------------------------------------


async def test_env_keys_are_replaced_by_the_variables_they_name(
    monkeypatch: pytest.MonkeyPatch, http_adapters: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("AGENTOS_KEY", "s3cret")

    build_target({"type": "agentos", "base_url": "http://agent-api:7777", "api_key_env": "AGENTOS_KEY"})

    assert http_adapters == [{"base_url": "http://agent-api:7777", "api_key": "s3cret"}]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"api_key_env": "MISSING_VAR_XYZ"}, "names the environment variable MISSING_VAR_XYZ, which is not set"),
        ({"api_key_env": ""}, "must name an environment variable"),
        ({"api_key_env": 3}, "must name an environment variable"),
        ({"api_key_env": "AGENTOS_KEY", "api_key": "x"}, "sets both api_key and api_key_env"),
    ],
)
def test_env_key_errors(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], message: str) -> None:
    monkeypatch.setenv("AGENTOS_KEY", "s3cret")
    monkeypatch.delenv("MISSING_VAR_XYZ", raising=False)
    with pytest.raises(ConfigError) as info:
        build_target({"type": "python", "factory": "harbor_support:support_agent", **config})
    assert message in str(info.value)


def test_a_key_named_just_env_is_an_ordinary_option(http_adapters: list[dict[str, Any]]) -> None:
    build_target({"type": "agentos", "base_url": "b", "_env": "x"})

    assert http_adapters == [{"base_url": "b", "_env": "x"}]


# -- built-in types ------------------------------------------------------------


def test_builtin_types_resolve_their_factory_at_call_time(http_adapters: list[dict[str, Any]]) -> None:
    with pytest.raises(ConfigError, match="target type 'openai-chat' rejected the options \\['base_url', 'model'\\]"):
        build_target({"type": "openai-chat", "base_url": "b", "model": "m"})
    with pytest.raises(ConfigError, match=r"target type 'a2a': simulhausen\.http_adapters:a2a_adapter is not callable"):
        build_target({"type": "a2a", "url": "u"})
    with pytest.raises(ConfigError, match="returned str"):
        build_target({"type": "langgraph", "base_url": "b"})


def test_a_missing_adapter_module_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(targets_module, "FACTORIES", {"agentos": "simulhausen_missing_module_xyz:agentos_adapter"})

    with pytest.raises(ConfigError, match="target type 'agentos': cannot import 'simulhausen_missing_module_xyz'"):
        build_target({"type": "agentos", "base_url": "b"})


@pytest.mark.parametrize(
    ("config", "adapter_class"),
    [
        ({"type": "agentos", "base_url": "http://agent-api:7777", "agent_id": "support"}, "_AgentOSAdapter"),
        ({"type": "agentos", "base_url": "http://agent-api:7777", "team_id": "t", "stream": False}, "_AgentOSAdapter"),
        ({"type": "openai-chat", "base_url": "http://llm:8000/v1", "model": "m"}, "_OpenAIChatAdapter"),
        ({"type": "openai-responses", "base_url": "http://llm:8000/v1", "model": "m"}, "_OpenAIResponsesAdapter"),
        ({"type": "a2a", "url": "http://agent-api:7777/a2a/agents/support/v1/message:send"}, "_A2AAdapter"),
        ({"type": "langgraph", "base_url": "http://graph:2024", "assistant_id": "agent"}, "_LangGraphAdapter"),
    ],
)
def test_builtin_types_build_the_real_http_adapters(config: dict[str, Any], adapter_class: str) -> None:
    """Every built-in type accepts a TOML-style table: the first argument by keyword, the rest as options."""
    assert type(build_target(config)).__name__ == adapter_class


async def test_agentos_target_reads_its_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "COMPLETED", "content": "hi there", "tools": []})

    monkeypatch.setenv("AGENTOS_KEY", "s3cret")
    adapter = build_target(
        {
            "type": "agentos",
            "base_url": "http://agent-api:7777",
            "agent_id": "support",
            "stream": False,
            "api_key_env": "AGENTOS_KEY",
            "transport": httpx.MockTransport(handler),
        }
    )

    turn = await _ask(adapter)

    assert isinstance(turn, AgentTurn)
    assert turn.content == "hi there"
    assert str(seen[0].url) == "http://agent-api:7777/agents/support/runs"
    assert seen[0].headers["authorization"] == "Bearer s3cret"


def test_agentos_target_option_errors_name_the_type() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        build_target({"type": "agentos", "base_url": "http://agent-api:7777"})
    with pytest.raises(ConfigError, match="target type 'openai-chat' rejected the options"):
        build_target({"type": "openai-chat", "base_url": "http://llm:8000/v1", "modle": "m"})


# -- redaction and precedence --------------------------------------------------


def test_redact_target_keeps_env_names_and_drops_secrets() -> None:
    config = {
        "type": "agentos",
        "base_url": "http://agent-api:7777",
        "agent_id": "support",
        "api_key": "s3cret",
        "api_key_env": "AGENTOS_KEY",
        "headers": {"Authorization": "Bearer x"},
        "token": "t",
    }

    assert redact_target(config) == {
        "type": "agentos",
        "base_url": "http://agent-api:7777",
        "agent_id": "support",
        "api_key_env": "AGENTOS_KEY",
    }


def _spec(target: dict[str, Any] | None = None) -> ScenarioSpec:
    return ScenarioSpec.model_validate({"id": "s", "name": "s", "description": "d", "target": target})


def test_target_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target_file = tmp_path / "target.json"
    target_file.write_text(json.dumps({"type": "from-file"}))
    monkeypatch.setenv("SIMULHAUSEN_TARGET", json.dumps({"type": "from-env"}))
    spec = _spec({"type": "from-scenario"})

    assert resolve_target_config(spec, target_json='{"type": "from-json"}', target_file=target_file) == {
        "type": "from-json"
    }
    assert resolve_target_config(spec, target_file=target_file) == {"type": "from-file"}
    assert resolve_target_config(spec) == {"type": "from-env"}
    monkeypatch.setenv("SIMULHAUSEN_TARGET", "")
    assert resolve_target_config(spec) == {"type": "from-scenario"}
    with pytest.raises(ConfigError, match="no target for scenario 's': set `target` in the scenario file"):
        resolve_target_config(_spec())


def test_target_source_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec()
    with pytest.raises(ConfigError, match="--target-json is not valid JSON"):
        resolve_target_config(spec, target_json="{")
    with pytest.raises(ConfigError, match="--target-json must be a JSON object"):
        resolve_target_config(spec, target_json="[1]")
    with pytest.raises(ConfigError, match="cannot read --target-file"):
        resolve_target_config(spec, target_file=tmp_path / "missing.json")
    monkeypatch.setenv("SIMULHAUSEN_TARGET", "nope")
    with pytest.raises(ConfigError, match="\\$SIMULHAUSEN_TARGET is not valid JSON"):
        resolve_target_config(spec)
