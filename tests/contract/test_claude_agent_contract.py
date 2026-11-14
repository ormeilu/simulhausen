"""``claude_agent_adapter`` against the real Claude Agent SDK, with a scripted transport in place of the CLI.

The SDK is not a dependency of the project, so the module is skipped in a
plain test run. Run it with the library overlaid:

    uv run --no-group harbor --with claude-agent-sdk pytest tests/contract/test_claude_agent_contract.py --no-cov -q

The SDK normally starts the ``claude`` CLI and talks to it over stdin and
stdout. Here ``ScriptedTransport`` stands in for the process: it answers the
SDK's ``initialize`` control request and replies to the prompt with the JSON
frames the CLI would print, so the SDK's own control protocol, message parser
and ``query()`` loop all run for real, with no CLI, no network and no API key.
What the tests cannot show is how the real CLI stores and resumes sessions:
``FakeClaude`` keeps them in memory, keyed by the ``resume`` option.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import ClaudeAgentOptions, ProcessError, ResultError, Transport

import simulhausen as sim
from simulhausen import ScenarioState, SimulationBackendError
from simulhausen.frameworks.claude_agent import claude_agent_adapter

USAGE = {"input_tokens": 10, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 5, "output_tokens": 4}


def _assistant(session_id: str, message_id: str, blocks: list[dict[str, Any]], parent: str | None = None) -> dict:
    return {
        "type": "assistant",
        "message": {"id": message_id, "model": "claude-scripted", "content": blocks, "usage": {}},
        "parent_tool_use_id": parent,
        "session_id": session_id,
    }


def _tool_result(tool_use_id: str, content: Any, *, is_error: bool = False, parent: str | None = None) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}
    return {"type": "user", "message": {"role": "user", "content": [block]}, "parent_tool_use_id": parent}


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_use(tool_use_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}


class FakeClaude:
    """The part of Claude Code the tests need: sessions that remember their prompts, and scripted turns."""

    def __init__(self) -> None:
        self.sessions: dict[str, list[str]] = {}
        self.options: list[Any] = []

    def turn(self, options: Any, prompt: str) -> list[dict[str, Any]]:
        self.options.append(options)
        resumed = options.resume
        session_id = resumed if resumed in self.sessions else str(uuid.uuid4())
        seen = self.sessions.setdefault(session_id, [])
        seen.append(prompt)
        frames: list[dict[str, Any]] = [{"type": "system", "subtype": "init", "session_id": session_id, "tools": []}]
        result: dict[str, Any] = {}
        text = f"{options.system_prompt}; prompts: {' | '.join(seen)}"
        if "where is my order" in prompt:
            frames += [
                _assistant(session_id, "msg_1", [_text("Let me look.")]),
                _assistant(session_id, "msg_1", [_tool_use("toolu_1", "mcp__shop__lookup_order", {"order_id": 7})]),
                _tool_result("toolu_1", [{"type": "text", "text": "shipped"}]),
            ]
            text = "Your order has shipped."
        elif "refund it" in prompt:
            frames += [
                _assistant(session_id, "msg_1", [_tool_use("toolu_2", "mcp__shop__refund", {"order_id": 7})]),
                _tool_result("toolu_2", "refunds are closed", is_error=True),
            ]
            text = "I could not refund it."
        elif "research the" in prompt:
            frames += [
                _assistant(session_id, "msg_1", [_tool_use("toolu_3", "Agent", {"subagent_type": "researcher"})]),
                _assistant(
                    session_id, "msg_sub", [_tool_use("toolu_4", "Grep", {"pattern": "TODO"})], parent="toolu_3"
                ),
                _tool_result("toolu_4", "3 matches", parent="toolu_3"),
                _assistant(session_id, "msg_sub2", [_text("subagent notes")], parent="toolu_3"),
                _tool_result("toolu_3", "three TODOs"),
            ]
            text = "There are three TODOs."
        elif "delete everything" in prompt:
            deferred = {"id": "toolu_5", "name": "Bash", "input": {"command": "rm -rf data"}}
            frames.append(_assistant(session_id, "msg_1", [_tool_use("toolu_5", "Bash", deferred["input"])]))
            result = {"deferred_tool_use": deferred, "result": None}
        elif "summary please" in prompt:
            result = {"structured_output": {"answer": "all good"}}
        elif prompt in ("fail", "crash"):
            result = {"is_error": True, "subtype": "error_max_turns", "errors": ["turn limit reached"], "result": None}
        if "deferred_tool_use" not in result and not result.get("is_error"):
            frames.append(_assistant(session_id, "msg_2", [_text(text)]))
        frames.append(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 12,
                "duration_api_ms": 10,
                "is_error": False,
                "num_turns": 2,
                "session_id": session_id,
                "total_cost_usd": 0.01,
                "usage": USAGE,
                "result": text,
                **result,
            }
        )
        return frames


class ScriptedTransport(Transport):
    """A transport that answers like the CLI process, fed by a :class:`FakeClaude`."""

    claude: FakeClaude

    def __init__(self, *, options: Any, prompt: Any = None) -> None:
        self._options = options
        self._frames: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._ready = False
        self._crash = False

    async def connect(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    async def write(self, data: str) -> None:
        frame = json.loads(data)
        if frame["type"] == "control_request":
            response = {"subtype": "success", "request_id": frame["request_id"], "response": {}}
            await self._frames.put({"type": "control_response", "response": response})
        elif frame["type"] == "user":
            prompt = frame["message"]["content"]
            self._crash = prompt == "crash"
            for reply in self.claude.turn(self._options, prompt):
                await self._frames.put(reply)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while (frame := await self._frames.get()) is not None:
            yield frame
        if self._crash:
            # The CLI exits non-zero after an error result; the SDK turns that into ResultError.
            raise ProcessError("Command failed", exit_code=1)

    async def end_input(self) -> None:
        await self._frames.put(None)

    async def close(self) -> None:
        self._ready = False
        await self._frames.put(None)


@pytest.fixture
def claude(monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    """``query()`` builds the scripted transport where it would build the subprocess one."""
    fake = FakeClaude()

    class Bound(ScriptedTransport):
        claude = fake

    monkeypatch.setattr("claude_agent_sdk._internal.client.SubprocessCLITransport", Bound)
    return fake


def _options(**kwargs: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(system_prompt="order desk", **kwargs)


async def _play(adapter: sim.AgentAdapter, script: list[Any]) -> sim.ScenarioResult:
    return await sim.run("contract", "A customer talks to the order desk.", [adapter], script)


async def test_reply_tool_call_memory_usage_and_cost_in_session_mode(claude: FakeClaude) -> None:
    states: list[ScenarioState] = []
    script = [sim.user("where is my order?"), sim.agent(), sim.user("thanks"), sim.agent(), states.append]

    result = await _play(claude_agent_adapter(_options), script)

    state = states[0]
    assert result.messages[1]["content"] == "Your order has shipped."
    assert state.has_tool_call("mcp__shop__lookup_order")
    call = state.tool_calls("mcp__shop__lookup_order").first
    assert (call.input, call.output, call.turn) == ({"order_id": 7}, "shipped", 0)
    assert call.execution is not None
    assert call.execution.tool_call_id == "toolu_1"
    # Second turn: the same SDK session is resumed and gets only the new message.
    assert result.messages[3]["content"] == "order desk; prompts: where is my order? | thanks"
    assert [options.resume for options in claude.options] == [None, next(iter(claude.sessions))]
    assert state.tool_calls(turn=1) == []
    # Two turns; the first made two model requests (msg_1 arrived as two messages, then msg_2), the second one.
    assert result.usage == sim.Usage(input_tokens=32, output_tokens=8, total_tokens=40, requests=3, cost=0.02)


async def test_a_failed_tool_is_an_error(claude: FakeClaude) -> None:
    states: list[ScenarioState] = []

    await _play(claude_agent_adapter(_options()), [sim.user("refund it"), sim.agent(), states.append])

    state = states[0]
    assert not state.has_tool_call("mcp__shop__refund")
    assert state.tool_calls() == []
    (failed,) = state.tool_calls(include_errors=True)
    assert (failed.name, failed.input, failed.output, failed.status) == (
        "mcp__shop__refund",
        {"order_id": 7},
        "refunds are closed",
        "error",
    )
    assert failed.error


async def test_subagent_tool_calls_are_recorded_and_its_text_is_not_the_reply(claude: FakeClaude) -> None:
    states: list[ScenarioState] = []

    result = await _play(claude_agent_adapter(_options()), [sim.user("research the TODOs"), sim.agent(), states.append])

    state = states[0]
    assert result.messages[1]["content"] == "There are three TODOs."
    assert state.tool_call_names() == ["Agent", "Grep"]
    assert state.tool_calls("Grep").first.output == "3 matches"
    assert state.tool_calls("Agent").first.output == "three TODOs"


async def test_a_deferred_tool_use_is_not_executed_and_the_reply_falls_back_to_the_last_text(
    claude: FakeClaude,
) -> None:
    states: list[ScenarioState] = []

    result = await _play(claude_agent_adapter(_options()), [sim.user("delete everything"), sim.agent(), states.append])

    state = states[0]
    assert result.messages[1]["content"] == ""
    assert not state.has_tool_call("Bash")
    (pending,) = state.tool_calls(include_errors=True)
    assert (pending.name, pending.input, pending.status) == ("Bash", {"command": "rm -rf data"}, "paused")


async def test_structured_output_is_rendered(claude: FakeClaude) -> None:
    plain = await _play(claude_agent_adapter(_options()), [sim.user("summary please"), sim.agent()])
    custom = await _play(
        claude_agent_adapter(_options(), content_fn=lambda output: output["answer"]),
        [sim.user("summary please"), sim.agent()],
    )

    assert plain.messages[-1]["content"] == '{"answer": "all good"}'
    assert custom.messages[-1]["content"] == "all good"


def _forget_the_secret(state: ScenarioState) -> None:
    state.rollback_messages_to(2)


ROLLBACK_SCRIPT = [
    sim.user("hello"),
    sim.agent(),
    sim.user("my secret is 42"),
    sim.agent(),
    _forget_the_secret,
    sim.user("what do you know?"),
    sim.agent(),
]


@pytest.mark.parametrize("history", ["transcript", "session"])
async def test_a_rollback_reaches_the_agent(claude: FakeClaude, history: Any) -> None:
    result = await _play(claude_agent_adapter(_options(), history=history), ROLLBACK_SCRIPT)

    reply = result.messages[-1]["content"]
    assert "secret" not in reply
    # A new session whose single prompt is the remaining transcript.
    assert reply.startswith("order desk; prompts: ")
    assert " | " not in reply
    assert "User: hello" in reply
    assert "User: what do you know?" in reply
    assert claude.options[-1].resume is None


async def test_transcript_mode_starts_a_new_session_on_every_turn(claude: FakeClaude) -> None:
    adapter = claude_agent_adapter(_options(), history="transcript")

    await _play(adapter, [sim.user("hello"), sim.agent(), sim.user("thanks"), sim.agent()])

    assert [options.resume for options in claude.options] == [None, None]
    assert len(claude.sessions) == 2


async def test_one_adapter_serves_scenarios_without_sharing_memory(claude: FakeClaude) -> None:
    adapter = claude_agent_adapter(_options())

    await _play(adapter, [sim.user("my secret is 42"), sim.agent()])
    result = await _play(adapter, [sim.user("what do you know?"), sim.agent()])

    assert result.messages[-1]["content"] == "order desk; prompts: what do you know?"


async def test_the_custom_transport_hook_gets_the_options_of_each_turn() -> None:
    fake = FakeClaude()

    class Remote(ScriptedTransport):
        claude = fake

    adapter = claude_agent_adapter(_options(), transport=lambda options: Remote(options=options))

    result = await _play(adapter, [sim.user("hello"), sim.agent(), sim.user("thanks"), sim.agent()])

    assert result.messages[3]["content"] == "order desk; prompts: hello | thanks"
    assert [options.resume for options in fake.options] == [None, next(iter(fake.sessions))]


async def test_an_error_result_raises(claude: FakeClaude) -> None:
    # The process exits non-zero after the error result: the SDK's own ResultError comes through.
    with pytest.raises(ResultError, match="turn limit reached"):
        await _play(claude_agent_adapter(_options()), [sim.user("crash"), sim.agent()])
    # A transport that just ends the stream: the adapter refuses to pass the error on as a reply.
    with pytest.raises(SimulationBackendError, match=r"error_max_turns\): turn limit reached"):
        await _play(claude_agent_adapter(_options()), [sim.user("fail"), sim.agent()])
