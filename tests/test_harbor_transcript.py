import json
from pathlib import Path
from typing import Any

import pytest
from agno.models.message import MessageReferences
from agno.models.response import ToolExecution

from simulhausen import ScenarioState
from simulhausen.harbor.transcript import (
    ATIF_VERSION,
    Checkpoint,
    ScenarioRef,
    Transcript,
    TranscriptError,
    build_transcript,
    read_transcript,
    trajectory,
    transcript_markdown,
    write_play_outputs,
)

SCENARIO = ScenarioRef(id="refund", name="refund: duplicated charge", description="Charged twice.")


def _state() -> ScenarioState:
    state = ScenarioState(thread_id="thread-1", name=SCENARIO.name, description=SCENARIO.description)
    state.add_user_message("i was charged twice")
    state.add_agent_message(
        "let me look",
        tools=[
            ToolExecution(tool_call_id="c1", tool_name="lookup_order", tool_args={"order": "1042"}, result="found"),
            ToolExecution(tool_name="issue_refund", tool_call_error=True, result="denied"),
            ToolExecution(tool_name="send_mail", requires_confirmation=True, tool_args={"to": object()}),
        ],
        references=[MessageReferences(query="refund", references=[{"content": "policy", "meta_data": {"url": "u"}}])],
    )
    state.add_user_message("thanks")
    state.add_agent_message("you're welcome")
    state.current_turn = 2
    return state


def _transcript(**overrides: Any) -> Transcript:
    options: dict[str, Any] = {
        "scenario": SCENARIO,
        "simulhausen_version": "1.2.3",
        "checkpoints": [Checkpoint(kind="check", step=2, message_count=2)],
        "stop_reason": "script_end",
        "total_time": 3.5,
        "agent_time": 1.25,
        "target": {"type": "agentos", "base_url": "http://agent-api:7777"},
    }
    options.update(overrides)
    return build_transcript(_state(), **options)


def test_build_transcript_records_messages_tools_and_references() -> None:
    transcript = _transcript()

    assert transcript.format == "simulhausen-transcript/1"
    assert transcript.thread_id == "thread-1"
    assert transcript.turns == 2
    assert [m.role for m in transcript.messages] == ["user", "assistant", "user", "assistant"]
    [turn] = transcript.agent_turns
    assert turn.message_index == 1
    assert [tool["tool_name"] for tool in turn.tools] == ["lookup_order", "issue_refund", "send_mail"]
    # Anything JSON cannot hold is kept as text, so the transcript is always writable.
    assert turn.tools[2]["tool_args"]["to"].startswith("<object object")
    assert turn.references[0]["references"] == [{"content": "policy", "meta_data": {"url": "u"}}]
    assert transcript.finished


def test_a_run_that_broke_before_it_began_has_an_empty_transcript() -> None:
    transcript = build_transcript(
        None,
        scenario=SCENARIO,
        simulhausen_version="1",
        checkpoints=[],
        stop_reason="harness_error",
        harness_error=True,
        error="boom",
    )

    assert transcript.messages == []
    assert transcript.thread_id == ""
    assert not transcript.finished


def test_round_trip_through_the_file_rebuilds_the_state(tmp_path: Path) -> None:
    original = _transcript()
    write_play_outputs(tmp_path, original)

    transcript = read_transcript(tmp_path / "transcript.json")
    state = transcript.to_state()

    assert transcript == original
    assert state.messages == _state().messages
    assert state.current_turn == 2
    assert state.tool_call_names() == ["lookup_order"]
    assert state.tool_calls("lookup_order").first.turn == 0
    assert state.has_tool_calls_in_order(["lookup_order"])
    assert "policy" in state.references_text()
    assert "Status: error" in state.tool_evidence_text()
    assert {path.name for path in tmp_path.iterdir()} == {"transcript.json", "trajectory.json", "transcript.md"}


def test_to_state_cuts_the_dialogue_at_a_limit() -> None:
    prefix = _transcript().to_state(1)

    assert prefix.messages == [{"role": "user", "content": "i was charged twice"}]
    assert prefix.tool_executions == []
    assert prefix.current_turn == 0
    assert _transcript().to_state(2).current_turn == 1


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return path


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "is not valid JSON"),
        ({"format": "other/1"}, "is not a simulhausen-transcript/1 transcript (format: 'other/1')"),
        ([], "(format: None)"),
        ({"format": "simulhausen-transcript/1"}, "is malformed"),
    ],
)
def test_unreadable_transcripts(tmp_path: Path, payload: object, message: str) -> None:
    with pytest.raises(TranscriptError) as info:
        read_transcript(_write(tmp_path / "transcript.json", payload))
    assert message in str(info.value)


def test_missing_transcript_says_the_agent_phase_did_not_run(tmp_path: Path) -> None:
    with pytest.raises(TranscriptError, match=r"no transcript at .*: the agent phase did not run"):
        read_transcript(tmp_path / "transcript.json")
    with pytest.raises(TranscriptError, match="cannot read the transcript"):
        read_transcript(tmp_path)


def test_inconsistent_indices_and_tool_records_are_rejected(tmp_path: Path) -> None:
    good = _transcript().model_dump(mode="json")
    wrong_index = {**good, "agent_turns": [{"message_index": 0, "tools": []}]}
    past_end = {**good, "checkpoints": [{"kind": "judge", "step": 1, "message_count": 9}]}
    bad_tool = {**good, "agent_turns": [{"message_index": 1, "tools": [{"metrics": "fast"}]}]}

    with pytest.raises(TranscriptError, match="message 0 is not an agent message"):
        read_transcript(_write(tmp_path / "a.json", wrong_index))
    with pytest.raises(TranscriptError, match="message_count 9 is past the dialogue"):
        read_transcript(_write(tmp_path / "b.json", past_end))
    with pytest.raises(TranscriptError, match="agent message 1 has a tool or reference record agno cannot read"):
        read_transcript(_write(tmp_path / "c.json", bad_tool))


def test_unknown_fields_from_newer_writers_are_ignored(tmp_path: Path) -> None:
    payload = {**_transcript().model_dump(mode="json"), "added_later": True}

    assert read_transcript(_write(tmp_path / "t.json", payload)).turns == 2


# -- trajectory ----------------------------------------------------------------


def test_trajectory_maps_messages_to_user_and_agent_steps() -> None:
    document = trajectory(_transcript())

    assert document["schema_version"] == ATIF_VERSION
    assert document["session_id"] == "thread-1"
    assert document["agent"] == {
        "name": "simulhausen",
        "version": "1.2.3",
        "extra": {"target": {"type": "agentos", "base_url": "http://agent-api:7777"}},
    }
    steps = document["steps"]
    assert [(s["step_id"], s["source"], s["message"]) for s in steps] == [
        (1, "user", "i was charged twice"),
        (2, "agent", "let me look"),
        (3, "user", "thanks"),
        (4, "agent", "you're welcome"),
    ]
    assert steps[1]["tool_calls"] == [
        {"tool_call_id": "c1", "function_name": "lookup_order", "arguments": {"order": "1042"}},
        {"tool_call_id": "call_1_1", "function_name": "issue_refund", "arguments": {}},
        {
            "tool_call_id": "call_1_2",
            "function_name": "send_mail",
            "arguments": {"to": steps[1]["tool_calls"][2]["arguments"]["to"]},
        },
    ]
    assert steps[1]["observation"]["results"][0] == {"source_call_id": "c1", "content": "found"}
    assert steps[1]["extra"]["tool_status"] == {"c1": "success", "call_1_1": "error", "call_1_2": "paused"}
    assert "tool_calls" not in steps[0]
    assert steps[3]["extra"] == {"message_index": 3}
    assert document["final_metrics"] == {"total_steps": 4}
    assert document["extra"]["stop_reason"] == "script_end"
    json.dumps(document)


def test_trajectory_of_an_empty_dialogue_has_a_system_step() -> None:
    transcript = build_transcript(
        None, scenario=SCENARIO, simulhausen_version="1", checkpoints=[], stop_reason="error", error="no target"
    )

    document = trajectory(transcript)

    assert document["steps"] == [{"step_id": 1, "source": "system", "message": "no dialogue was recorded: no target"}]
    assert "session_id" not in document
    assert "extra" not in document["agent"]
    no_error = transcript.model_copy(update={"error": None})
    assert trajectory(no_error)["steps"][0]["message"] == "no dialogue was recorded"


# -- markdown ------------------------------------------------------------------


def test_markdown_shows_the_dialogue_tools_checkpoints_and_outcome() -> None:
    text = transcript_markdown(_transcript(error="RuntimeError: boom", harness_error=True), "## Outcome: PASS\n")

    assert text.startswith("# refund: duplicated charge\n\nCharged twice.\n")
    assert "- Turns: 2, stop reason: `script_end`" in text
    assert "- Error (harness): RuntimeError: boom" in text
    assert text.index("## Outcome: PASS") < text.index("## Dialogue")
    assert "_Tools: `lookup_order` (success), `issue_refund` (error), `send_mail` (paused)_" in text
    assert text.index("> checkpoint: script[2] (check)") < text.index("thanks")


def test_markdown_of_an_empty_dialogue() -> None:
    transcript = build_transcript(
        None,
        scenario=SCENARIO,
        simulhausen_version="1",
        checkpoints=[],
        stop_reason="error",
        error="ValueError: x",
    )

    text = transcript_markdown(transcript)

    assert "_No messages were recorded._" in text
    assert "- Error: ValueError: x" in text
