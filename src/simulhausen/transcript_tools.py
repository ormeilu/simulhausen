"""Bounded discovery over a long transcript: a skeleton plus ``expand_transcript`` and ``grep_transcript``.

Adapted from LangWatch Scenario's ``_judge/transcript_tools.py`` and ``_judge/estimate_tokens.py``
(Apache-2.0, see NOTICE). A dialogue above the judge's token threshold is not pasted into the judge's
prompt. The judge gets a skeleton (one line per message) and reads what it needs through two lookups:
``expand_transcript`` returns whole messages by index, ``grep_transcript`` finds messages by text.

The lookups never raise on bad input: a wrong index or an empty pattern comes back as a readable message
the model can act on. Every result is capped, so a lookup cannot bring the whole dialogue back at once.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from simulhausen.types import Message

__all__ = [
    "DEFAULT_TOKEN_THRESHOLD",
    "MAX_GREP_MATCHES",
    "PREVIEW_CHARS",
    "TOOL_RESULT_CHAR_BUDGET",
    "TOOL_RESULT_TOKEN_BUDGET",
    "build_transcript_skeleton",
    "estimate_tokens",
    "expand_transcript",
    "grep_transcript",
    "message_line",
]

# Above this many estimated tokens the judge sees a skeleton instead of the dialogue.
DEFAULT_TOKEN_THRESHOLD: Final = 8192
# Cap for one lookup result and for the skeleton itself, in estimated tokens.
TOOL_RESULT_TOKEN_BUDGET: Final = 4096
# The same cap in characters, at the four-bytes-per-token estimate.
TOOL_RESULT_CHAR_BUDGET: Final = TOOL_RESULT_TOKEN_BUDGET * 4
# grep_transcript lists at most this many matching messages.
MAX_GREP_MATCHES: Final = 20
# How much of a message the skeleton shows.
PREVIEW_CHARS: Final = 80

_NO_MESSAGES: Final = "No messages recorded."
_TRUNCATED_NOTE: Final = (
    "\n\n[TRUNCATED] Output exceeded ~4000 token budget. "
    "Use grep_transcript(pattern) to search for specific content, "
    "or expand_transcript with fewer indices."
)
_SPEAKERS: Final = {"user": "User", "assistant": "Assistant"}


def estimate_tokens(text: str) -> int:
    """Rough token count: UTF-8 bytes divided by four, rounded up.

    Counting bytes rather than characters makes CJK text and emoji, which cost
    more tokens per character, weigh more. Characters that cannot be encoded
    (lone surrogates in broken model output) count as one byte.
    """
    return math.ceil(len(text.encode("utf-8", errors="replace")) / 4)


def message_line(message: Message) -> str:
    """One message as the transcript renders it: ``Speaker: text``.

    ``"\\n".join`` of these lines is exactly ``ScenarioState.transcript()`` with the
    default speaker names, so a lookup shows what the full transcript would have.
    """
    return f"{_SPEAKERS[message['role']]}: {message['content']}"


def _within_budget(text: str) -> str:
    if len(text) <= TOOL_RESULT_CHAR_BUDGET:
        return text
    return text[:TOOL_RESULT_CHAR_BUDGET] + _TRUNCATED_NOTE


def _skeleton_line(index: int, message: Message) -> str:
    preview = " ".join(message["content"].split())
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS].rstrip() + "..."
    return f"[{index}] {message['role']} (~{estimate_tokens(message_line(message))} tokens): {preview or '(empty)'}"


def _fit(lines: Iterable[str], budget: int) -> list[str]:
    """Leading lines that fit the token budget; at least one, so a window is never empty."""
    taken: list[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line)
        if taken and used + cost > budget:
            break
        taken.append(line)
        used += cost
    return taken


def build_transcript_skeleton(messages: Sequence[Message]) -> str:
    """One line per message: index, role, size in tokens and a short preview.

    The skeleton itself is capped at ``TOOL_RESULT_TOKEN_BUDGET``. A dialogue of
    hundreds of messages keeps a window of its first and last lines: the start
    frames the scenario, the end is what the judge rules on. The messages in
    between stay reachable through ``expand_transcript`` and ``grep_transcript``.
    """
    if not messages:
        return _NO_MESSAGES
    lines = [_skeleton_line(index, message) for index, message in enumerate(messages)]
    header = f"Messages: {len(lines)}"
    if estimate_tokens("\n".join(lines)) <= TOOL_RESULT_TOKEN_BUDGET:
        return f"{header}\n\n" + "\n".join(lines)

    half = TOOL_RESULT_TOKEN_BUDGET // 2
    head = _fit(lines, half)
    tail = _fit(reversed(lines[len(head) :]), half)[::-1]
    omitted = len(lines) - len(head) - len(tail)
    gap = (
        [f"... [{omitted} messages omitted - use grep_transcript(pattern) or expand_transcript(indices)] ..."]
        if omitted > 0
        else []
    )
    return f"{header}\n\n" + "\n".join(head + gap + tail)


def _as_index(raw: object) -> int | None:
    """An index from what a model sent: an int, a whole number or digits in a string."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def expand_transcript(messages: Sequence[Message], indices: Sequence[object]) -> str:
    """Whole messages by 0-based index, in transcript order, capped at the result budget.

    Duplicates collapse. Models sometimes send an index as a string or a whole
    float; those are read as numbers. Anything else, and indices outside the
    dialogue, are listed as ignored instead of failing the lookup.
    """
    if not messages:
        return _NO_MESSAGES
    if not indices:
        return "Error: provide at least one message index."

    wanted: set[int] = set()
    malformed: list[object] = []
    for raw in indices:
        index = _as_index(raw)
        if index is None:
            malformed.append(raw)
        else:
            wanted.add(index)

    selected = sorted(index for index in wanted if 0 <= index < len(messages))
    if not selected:
        return f"Error: no messages matched the given indices. Valid range: 0-{len(messages) - 1}."

    lines = [f"[{index}] {message_line(messages[index])}" for index in selected]
    if out_of_range := sorted(index for index in wanted if not 0 <= index < len(messages)):
        lines.append(f"\n[Ignored out-of-range indices: {out_of_range}]")
    if malformed:
        lines.append(f"\n[Ignored non-integer indices: {malformed}]")
    return _within_budget("\n".join(lines).rstrip())


def grep_transcript(messages: Sequence[Message], pattern: str) -> str:
    """Messages that contain ``pattern``, case-insensitively, as plain text and not a regex.

    The match runs over the rendered line (``Speaker: text``), the same as
    upstream. At most ``MAX_GREP_MATCHES`` messages are listed, then a note says
    how many more there are. The pattern is escaped, so no input is an invalid
    expression and none can stall the search with catastrophic backtracking.
    """
    if not messages:
        return _NO_MESSAGES
    if not pattern:
        return "Error: provide a non-empty search pattern."

    regex = re.compile(re.escape(pattern), re.IGNORECASE)
    matches = [(index, message) for index, message in enumerate(messages) if regex.search(message_line(message))]
    if not matches:
        roles = ", ".join(dict.fromkeys(message["role"] for message in messages))
        return f'No matches found for "{pattern}". Roles present: {roles}'

    lines: list[str] = []
    for index, message in matches[:MAX_GREP_MATCHES]:
        lines += [f"--- [{index}] {message['role']} ---", f"  {message_line(message)}", ""]
    if (more := len(matches) - MAX_GREP_MATCHES) > 0:
        lines.append(f"[{more} more matches omitted. Refine your search pattern for more specific results.]")
    return _within_budget("\n".join(lines).rstrip())
