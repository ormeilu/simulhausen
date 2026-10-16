"""The recorded dialogue: ``transcript.json``, Harbor's ``trajectory.json`` and ``transcript.md``.

In a Harbor task the dialogue and its grading run in different phases, often in
different processes: ``play`` talks to the agent under test in the agent phase and the
verifier's ``judge`` grades what was said. ``transcript.json`` is the hand-over between
them, so it carries everything the judge needs to rebuild the
:class:`~simulhausen.ScenarioState`: the messages, the tool executions and RAG references
of every agent turn (agno's own ``ToolExecution.to_dict()`` and ``MessageReferences``
shapes, so ``has_tool_call`` and the judge's evidence work as in a live run), where the
script's checkpoints sat, and how the dialogue ended.

``trajectory.json`` is the same dialogue in Harbor's ATIF trajectory format (the one
``harbor view`` renders from an agent's ``trajectory.json``), and ``transcript.md`` is for
people.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final, Literal, Self

from agno.models.message import MessageReferences
from agno.models.response import ToolExecution
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from simulhausen.harbor.paths import TRAJECTORY_FILE, TRANSCRIPT_FILE, TRANSCRIPT_MARKDOWN_FILE
from simulhausen.types import ScenarioState

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "ATIF_VERSION",
    "TRANSCRIPT_FORMAT",
    "AgentTurnRecord",
    "Checkpoint",
    "ScenarioRef",
    "StopReason",
    "Transcript",
    "TranscriptError",
    "TranscriptMessage",
    "build_transcript",
    "read_transcript",
    "trajectory",
    "transcript_markdown",
    "write_json",
    "write_play_outputs",
]

TRANSCRIPT_FORMAT: Final = "simulhausen-transcript/1"
# Oldest ATIF version with every field written here, so older Harbor viewers read it too.
ATIF_VERSION: Final = "ATIF-v1.6"
AGENT_NAME: Final = "simulhausen"

# Why the dialogue ended. max_turns / goal_achieved: the last step was a proceed that hit
# max_turns or whose simulator reported the goal reached; script_end: the script ran out;
# judge_step: a `judge = true` step; harness_error / error: the run raised (see `error`).
type StopReason = Literal["max_turns", "goal_achieved", "script_end", "judge_step", "harness_error", "error"]


class TranscriptError(Exception):
    """A transcript is missing or cannot be read; the judge reports the scenario as WARN."""


class _Record(BaseModel):
    # Newer writers may add fields; a reader that ignores them keeps working.
    model_config = ConfigDict(extra="ignore")


class ScenarioRef(_Record):
    id: str
    name: str
    description: str


class TranscriptMessage(_Record):
    role: Literal["user", "assistant"]
    content: str


class AgentTurnRecord(_Record):
    """What one agent message brought besides its text: tool executions and references."""

    message_index: int = Field(ge=0)
    # ToolExecution.to_dict() and MessageReferences.model_dump(mode="json") shapes.
    tools: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)


class Checkpoint(_Record):
    """A ``judge = { criteria }`` or ``check`` step: where in the dialogue it was reached.

    ``step`` is the 0-based index of the step in the scenario's ``script``; the judge
    phase takes its criteria or check from the scenario file by that index and grades the
    first ``message_count`` messages. ``criteria`` stays empty in ``play`` output.
    """

    kind: Literal["judge", "check"]
    step: int = Field(ge=0)
    message_count: int = Field(ge=0)
    criteria: list[str] = Field(default_factory=list)


class Transcript(_Record):
    """``transcript.json``: the dialogue ``play`` recorded, as ``judge`` reads it back."""

    format: Literal["simulhausen-transcript/1"] = TRANSCRIPT_FORMAT
    simulhausen_version: str
    scenario: ScenarioRef
    thread_id: str
    messages: list[TranscriptMessage] = Field(default_factory=list)
    agent_turns: list[AgentTurnRecord] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    # Agent turns played (ScenarioState.current_turn).
    turns: int = Field(default=0, ge=0)
    stop_reason: StopReason
    harness_error: bool = False
    error: str | None = None
    total_time: float = 0.0
    agent_time: float = 0.0
    # The target with secrets removed (see targets.redact_target).
    target: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _indices_in_range(self) -> Self:
        for turn in self.agent_turns:
            if turn.message_index >= len(self.messages) or self.messages[turn.message_index].role != "assistant":
                raise ValueError(f"agent_turns: message {turn.message_index} is not an agent message")
        for checkpoint in self.checkpoints:
            if checkpoint.message_count > len(self.messages):
                raise ValueError(f"checkpoints: message_count {checkpoint.message_count} is past the dialogue")
        return self

    @property
    def finished(self) -> bool:
        """The dialogue was played to its end (no harness failure, no runtime error)."""
        return not self.harness_error and self.stop_reason != "error"

    def to_state(self, limit: int | None = None) -> ScenarioState:
        """The dialogue as a :class:`ScenarioState`, cut to the first ``limit`` messages.

        Agent messages get their tool executions and references back through
        ``add_agent_message``, so tool turns and the judge's evidence match the live run.
        Raises :class:`TranscriptError` for tool or reference records agno cannot read.
        """
        messages = self.messages if limit is None else self.messages[:limit]
        state = ScenarioState(thread_id=self.thread_id, name=self.scenario.name, description=self.scenario.description)
        turns = {turn.message_index: turn for turn in self.agent_turns}
        for index, message in enumerate(messages):
            if message.role == "user":
                state.add_user_message(message.content)
                continue
            turn = turns.get(index)
            try:
                tools = [ToolExecution.from_dict(tool) for tool in turn.tools] if turn else None
                references = [MessageReferences.model_validate(ref) for ref in turn.references] if turn else None
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                raise TranscriptError(
                    f"agent message {index} has a tool or reference record agno cannot read: {exc}"
                ) from exc
            state.add_agent_message(message.content, tools=tools or None, references=references or None)
        played = sum(1 for message in messages if message.role == "assistant")
        state.current_turn = self.turns if limit is None else min(self.turns, played)
        return state


def _plain(value: Any) -> Any:
    """``value`` as plain JSON data; anything JSON cannot hold becomes its ``str``."""
    return json.loads(json.dumps(value, default=str))


def build_transcript(
    state: ScenarioState | None,
    *,
    scenario: ScenarioRef,
    simulhausen_version: str,
    checkpoints: Sequence[Checkpoint],
    stop_reason: StopReason,
    harness_error: bool = False,
    error: str | None = None,
    total_time: float = 0.0,
    agent_time: float = 0.0,
    target: dict[str, Any] | None = None,
) -> Transcript:
    """Record a played dialogue; ``state`` is ``None`` when the run broke before it began."""
    messages: list[TranscriptMessage] = []
    turns: dict[int, AgentTurnRecord] = {}
    thread_id = ""
    current_turn = 0
    if state is not None:
        thread_id = state.thread_id
        current_turn = state.current_turn
        messages = [TranscriptMessage(role=m["role"], content=m["content"]) for m in state.messages]
        for tool in state.tool_executions:
            # The runner records which agent message brought each execution.
            if (index := state._origins.get(id(tool))) is not None:
                turns.setdefault(index, AgentTurnRecord(message_index=index)).tools.append(_plain(tool.to_dict()))
        for ref in state.references:
            if (index := state._origins.get(id(ref))) is not None:
                turns.setdefault(index, AgentTurnRecord(message_index=index)).references.append(
                    ref.model_dump(mode="json")
                )
    return Transcript(
        simulhausen_version=simulhausen_version,
        scenario=scenario,
        thread_id=thread_id,
        messages=messages,
        agent_turns=[turns[index] for index in sorted(turns)],
        checkpoints=list(checkpoints),
        turns=current_turn,
        stop_reason=stop_reason,
        harness_error=harness_error,
        error=error,
        total_time=total_time,
        agent_time=agent_time,
        target=target,
    )


def read_transcript(path: Path) -> Transcript:
    """Read ``transcript.json``; raises :class:`TranscriptError` saying why it cannot be used."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise TranscriptError(
            f"no transcript at {path}: the agent phase did not run, or failed before the dialogue started "
            "(see the agent logs)"
        ) from None
    except OSError as exc:
        raise TranscriptError(f"cannot read the transcript {path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"the transcript {path} is not valid JSON: {exc}") from exc
    found = data.get("format") if isinstance(data, dict) else None
    if found != TRANSCRIPT_FORMAT:
        raise TranscriptError(f"{path} is not a {TRANSCRIPT_FORMAT} transcript (format: {found!r})")
    try:
        transcript = Transcript.model_validate(data)
    except ValidationError as exc:
        raise TranscriptError(f"the transcript {path} is malformed: {exc}") from exc
    transcript.to_state()  # surfaces unreadable tool records now, not halfway through grading
    return transcript


# -- Harbor trajectory ---------------------------------------------------------


def _tool_status(tool: ToolExecution) -> str:
    return "paused" if tool.is_paused else "error" if tool.tool_call_error else "success"


def trajectory(transcript: Transcript) -> dict[str, Any]:
    """The dialogue as a Harbor ATIF trajectory: one ``user`` or ``agent`` step per message.

    An agent step lists the tool executions of its turn as ``tool_calls`` with their
    results as the ``observation``; each step's ``extra`` holds the message index and the
    status of each call (``success``, ``error`` or ``paused``).
    """
    turns = {turn.message_index: turn for turn in transcript.agent_turns}
    steps: list[dict[str, Any]] = []
    for index, message in enumerate(transcript.messages):
        extra: dict[str, Any] = {"message_index": index}
        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": "user" if message.role == "user" else "agent",
            "message": message.content,
        }
        turn = turns.get(index)
        if turn is not None and turn.tools:
            calls: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []
            status: dict[str, str] = {}
            for number, record in enumerate(turn.tools):
                tool = ToolExecution.from_dict(record)
                call_id = tool.tool_call_id or f"call_{index}_{number}"
                calls.append(
                    {
                        "tool_call_id": call_id,
                        "function_name": tool.tool_name or "unknown",
                        "arguments": tool.tool_args if isinstance(tool.tool_args, dict) else {},
                    }
                )
                results.append({"source_call_id": call_id, "content": tool.result})
                status[call_id] = _tool_status(tool)
            step["tool_calls"] = calls
            step["observation"] = {"results": results}
            extra["tool_status"] = status
        step["extra"] = extra
        steps.append(step)
    if not steps:
        # ATIF needs at least one step; a run that broke before the first message gets a system note.
        note = "no dialogue was recorded" + (f": {transcript.error}" if transcript.error else "")
        steps.append({"step_id": 1, "source": "system", "message": note})
    agent: dict[str, Any] = {"name": AGENT_NAME, "version": transcript.simulhausen_version}
    if transcript.target is not None:
        agent["extra"] = {"target": transcript.target}
    document: dict[str, Any] = {"schema_version": ATIF_VERSION}
    if transcript.thread_id:
        document["session_id"] = transcript.thread_id
    return {
        **document,
        "agent": agent,
        "steps": steps,
        "notes": "simulhausen scenario dialogue: user steps come from the user simulator or the script, "
        "agent steps from the agent under test",
        "final_metrics": {"total_steps": len(steps)},
        "extra": {
            "scenario": transcript.scenario.model_dump(),
            "turns": transcript.turns,
            "stop_reason": transcript.stop_reason,
            "harness_error": transcript.harness_error,
            "error": transcript.error,
        },
    }


# -- markdown ------------------------------------------------------------------


def transcript_markdown(transcript: Transcript, outcome: str | None = None) -> str:
    """The dialogue for people; ``outcome`` (markdown, e.g. the verdicts) goes above it."""
    scenario = transcript.scenario
    lines = [
        f"# {scenario.name}",
        "",
        scenario.description,
        "",
        f"- Scenario: `{scenario.id}`",
        f"- Turns: {transcript.turns}, stop reason: `{transcript.stop_reason}`",
        f"- Time: {transcript.total_time:.1f}s ({transcript.agent_time:.1f}s in the agent)",
    ]
    if transcript.error:
        lines.append(f"- Error{' (harness)' if transcript.harness_error else ''}: {transcript.error}")
    if outcome:
        lines += ["", outcome.rstrip()]
    lines += ["", "## Dialogue", ""]
    if not transcript.messages:
        lines += ["_No messages were recorded._", ""]
    turns = {turn.message_index: turn for turn in transcript.agent_turns}
    marks: dict[int, list[Checkpoint]] = {}
    for checkpoint in transcript.checkpoints:
        marks.setdefault(checkpoint.message_count, []).append(checkpoint)
    for index in range(len(transcript.messages) + 1):
        lines += [f"> checkpoint: script[{c.step}] ({c.kind})\n" for c in marks.get(index, [])]
        if index == len(transcript.messages):
            break
        message = transcript.messages[index]
        lines += [f"**{'User' if message.role == 'user' else 'Agent'}**", "", message.content, ""]
        if (turn := turns.get(index)) and turn.tools:
            calls = [ToolExecution.from_dict(record) for record in turn.tools]
            listed = ", ".join(f"`{tool.tool_name or 'unknown'}` ({_tool_status(tool)})" for tool in calls)
            lines += [f"_Tools: {listed}_", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_play_outputs(out_dir: Path, transcript: Transcript) -> None:
    """Write ``transcript.json``, ``trajectory.json`` and ``transcript.md`` into ``out_dir``."""
    write_json(out_dir / TRANSCRIPT_FILE, transcript.model_dump(mode="json"))
    write_json(out_dir / TRAJECTORY_FILE, trajectory(transcript))
    (out_dir / TRANSCRIPT_MARKDOWN_FILE).write_text(transcript_markdown(transcript))
