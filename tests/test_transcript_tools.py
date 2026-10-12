import re

import pytest

from simulhausen import Message, ScenarioState
from simulhausen.transcript_tools import (
    DEFAULT_TOKEN_THRESHOLD,
    MAX_GREP_MATCHES,
    PREVIEW_CHARS,
    TOOL_RESULT_CHAR_BUDGET,
    TOOL_RESULT_TOKEN_BUDGET,
    build_transcript_skeleton,
    estimate_tokens,
    expand_transcript,
    grep_transcript,
    message_line,
)


def _dialogue(*texts: str) -> list[Message]:
    """Alternating user and assistant messages, starting with the user."""
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(texts)]


def _numbered_dialogue(count: int, filler: str = "") -> list[Message]:
    return _dialogue(*(f"message {i} {filler}" for i in range(count)))


# -- estimate_tokens ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("", 0),
        ("a", 1),
        ("abcd", 1),
        ("abcde", 2),  # 5 bytes / 4 = 1.25, rounded up
        ("abcdefgh", 2),
        pytest.param("a" * 4000, 1000, id="4000-ascii"),
        ("\U0001f600\U0001f601\U0001f602\U0001f603", 4),  # emoji: 4 bytes each
        ("你好世界", 3),  # CJK: 3 bytes each
        ("Привет", 3),  # Cyrillic: 2 bytes each
    ],
)
def test_estimate_tokens_is_utf8_bytes_over_four_rounded_up(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


def test_estimate_tokens_survives_text_that_cannot_be_encoded() -> None:
    assert estimate_tokens("\ud800") == 1  # a lone surrogate counts as one byte


def test_default_threshold_and_budgets_match_upstream() -> None:
    assert DEFAULT_TOKEN_THRESHOLD == 8192
    assert TOOL_RESULT_TOKEN_BUDGET == 4096
    assert TOOL_RESULT_CHAR_BUDGET == 4096 * 4
    assert MAX_GREP_MATCHES == 20


# -- message_line -------------------------------------------------------------


def test_message_lines_join_into_the_state_transcript() -> None:
    state = ScenarioState(thread_id="t", name="n", description="d")
    state.add_user_message("I was charged twice")
    state.add_agent_message("Let me check\nthe order")

    assert "\n".join(message_line(m) for m in state.messages) == state.transcript()


# -- build_transcript_skeleton ------------------------------------------------


def test_skeleton_has_one_line_per_message_with_index_role_size_and_preview() -> None:
    messages = _dialogue("I was charged twice", "Let me check")

    skeleton = build_transcript_skeleton(messages)

    assert skeleton == (
        "Messages: 2\n\n"
        f"[0] user (~{estimate_tokens('User: I was charged twice')} tokens): I was charged twice\n"
        f"[1] assistant (~{estimate_tokens('Assistant: Let me check')} tokens): Let me check"
    )
    assert estimate_tokens("User: I was charged twice") == 7


def test_skeleton_previews_are_one_line_and_cut_at_the_preview_length() -> None:
    long_text = "word " * 100
    messages = _dialogue("first line\n\n   second\tline", long_text)

    lines = build_transcript_skeleton(messages).splitlines()

    assert lines[2].endswith(": first line second line")
    preview = lines[3].split("): ", 1)[1]
    assert preview == ("word " * 100)[:PREVIEW_CHARS].rstrip() + "..."
    assert len(preview) <= PREVIEW_CHARS + 3


def test_skeleton_size_is_the_estimate_of_the_whole_rendered_message() -> None:
    messages = _dialogue("x" * 400)

    assert f"(~{estimate_tokens('User: ' + 'x' * 400)} tokens)" in build_transcript_skeleton(messages)


def test_skeleton_marks_an_empty_message() -> None:
    lines = build_transcript_skeleton(_dialogue("", "   ")).splitlines()

    assert lines[2] == "[0] user (~2 tokens): (empty)"
    assert lines[3].endswith("): (empty)")


def test_skeleton_of_no_messages() -> None:
    assert build_transcript_skeleton([]) == "No messages recorded."


def _first_number(line: str) -> int:
    found = re.search(r"\d+", line)
    assert found, line
    return int(found.group())


def test_skeleton_of_a_long_dialogue_keeps_a_head_and_a_tail_within_the_budget() -> None:
    messages = _numbered_dialogue(400, filler="lorem ipsum " * 20)

    skeleton = build_transcript_skeleton(messages)

    lines = skeleton.splitlines()
    assert lines[0] == "Messages: 400"
    assert lines[2].startswith("[0] user")
    assert lines[-1].startswith("[399] assistant")
    (gap,) = [line for line in lines if "omitted" in line]
    omitted = _first_number(gap.removeprefix("... ["))
    assert 0 < omitted < 400
    assert "grep_transcript(pattern)" in gap
    assert "expand_transcript(indices)" in gap
    assert estimate_tokens(skeleton) <= TOOL_RESULT_TOKEN_BUDGET + 64
    # every message is either listed or counted in the gap, and the gap sits between two listed indices
    listed = [line for line in lines[2:] if line != gap]
    assert len(listed) + omitted == 400
    gap_at = lines.index(gap)
    last_head = _first_number(lines[gap_at - 1])
    first_tail = _first_number(lines[gap_at + 1].removeprefix("["))
    assert first_tail - last_head - 1 == omitted


def test_skeleton_of_a_dialogue_that_fits_has_no_gap() -> None:
    skeleton = build_transcript_skeleton(_numbered_dialogue(100))

    assert "omitted" not in skeleton
    assert skeleton.splitlines()[-1].startswith("[99] assistant")


# -- expand_transcript --------------------------------------------------------


def test_expand_returns_whole_messages_by_index() -> None:
    messages = _dialogue("first", "second\nstill second", "third")

    assert expand_transcript(messages, [1]) == "[1] Assistant: second\nstill second"
    assert expand_transcript(messages, [0, 2]) == "[0] User: first\n[2] User: third"


def test_expand_lists_messages_in_dialogue_order_and_drops_duplicates() -> None:
    messages = _dialogue("a", "b", "c")

    assert expand_transcript(messages, [2, 0, 2, 0]) == "[0] User: a\n[2] User: c"


def test_expand_ignores_out_of_range_indices_and_says_so() -> None:
    messages = _dialogue("a", "b")

    result = expand_transcript(messages, [1, 7, -1, 99])

    assert result == "[1] Assistant: b\n\n[Ignored out-of-range indices: [-1, 7, 99]]"


@pytest.mark.parametrize("indices", [[5], [-1], [5, 6]])
def test_expand_with_no_valid_index_names_the_valid_range(indices: list[int]) -> None:
    assert expand_transcript(_dialogue("a", "b", "c"), indices) == (
        "Error: no messages matched the given indices. Valid range: 0-2."
    )


def test_expand_without_indices_asks_for_one() -> None:
    assert expand_transcript(_dialogue("a"), []) == "Error: provide at least one message index."


def test_expand_of_no_messages() -> None:
    assert expand_transcript([], [0]) == "No messages recorded."
    assert expand_transcript([], []) == "No messages recorded."


def test_expand_reads_numbers_that_models_send_as_strings_or_whole_floats() -> None:
    messages = _dialogue("a", "b", "c")

    assert expand_transcript(messages, ["1", " 2 ", 0.0]) == "[0] User: a\n[1] Assistant: b\n[2] User: c"


def test_expand_ignores_what_is_not_an_index_instead_of_failing() -> None:
    messages = _dialogue("a", "b")

    result = expand_transcript(messages, [0, "last", 1.5, True, None, float("nan"), float("inf"), [1]])

    assert result == ("[0] User: a\n\n[Ignored non-integer indices: ['last', 1.5, True, None, nan, inf, [1]]]")


def test_expand_with_only_malformed_indices_is_an_error_and_never_raises() -> None:
    assert expand_transcript(_dialogue("a"), ["x", None]) == (
        "Error: no messages matched the given indices. Valid range: 0-0."
    )


def test_expand_cuts_the_result_at_the_character_budget() -> None:
    messages = _dialogue("x" * (TOOL_RESULT_CHAR_BUDGET * 2), "short")

    result = expand_transcript(messages, [0, 1])

    assert result.startswith("[0] User: xxx")
    head, note = result.split("\n\n[TRUNCATED]")
    assert len(head) == TOOL_RESULT_CHAR_BUDGET
    assert "expand_transcript with fewer indices" in note
    assert "grep_transcript(pattern)" in note
    assert "[1] Assistant: short" not in result


def test_expand_result_at_exactly_the_budget_is_not_cut() -> None:
    prefix = "[0] User: "
    messages = _dialogue("x" * (TOOL_RESULT_CHAR_BUDGET - len(prefix)))

    result = expand_transcript(messages, [0])

    assert len(result) == TOOL_RESULT_CHAR_BUDGET
    assert "TRUNCATED" not in result


# -- grep_transcript ----------------------------------------------------------


def test_grep_lists_matching_messages_with_their_full_text() -> None:
    messages = _dialogue("I was charged twice", "Let me look up the order", "order 1042, please", "found it")

    result = grep_transcript(messages, "order")

    assert result == (
        "--- [1] assistant ---\n  Assistant: Let me look up the order\n\n--- [2] user ---\n  User: order 1042, please"
    )


def test_grep_ignores_case_in_both_directions() -> None:
    messages = _dialogue("Refund ME", "nothing here")

    assert grep_transcript(messages, "refund me").startswith("--- [0] user ---")
    assert grep_transcript(messages, "REFUND").startswith("--- [0] user ---")


def test_grep_matches_the_rendered_line_including_the_speaker() -> None:
    messages = _dialogue("hi", "hello")

    assert grep_transcript(messages, "assistant:").startswith("--- [1] assistant ---")


@pytest.mark.parametrize("pattern", ["a.c", "a|b", "(", "[", "*", "\\", "a{2}", "^abc$", "(?<"])
def test_grep_takes_the_pattern_literally_so_no_pattern_is_invalid(pattern: str) -> None:
    messages = _dialogue("abc", "aabb", f"literal {pattern} inside")

    result = grep_transcript(messages, pattern)

    assert result.startswith("--- [2] user ---")
    assert result.count("---") == 2  # only the message that holds the pattern as text


def test_grep_reports_what_it_searched_when_nothing_matches() -> None:
    messages = _dialogue("hi", "hello", "bye")

    assert grep_transcript(messages, "refund") == 'No matches found for "refund". Roles present: user, assistant'
    assert grep_transcript(_dialogue("only the user"), "x") == 'No matches found for "x". Roles present: user'


def test_grep_needs_a_pattern() -> None:
    assert grep_transcript(_dialogue("a"), "") == "Error: provide a non-empty search pattern."


def test_grep_of_no_messages() -> None:
    assert grep_transcript([], "x") == "No messages recorded."
    assert grep_transcript([], "") == "No messages recorded."


def test_grep_lists_at_most_twenty_messages_and_counts_the_rest() -> None:
    messages = _dialogue(*(f"refund {i}" for i in range(MAX_GREP_MATCHES + 5)))

    result = grep_transcript(messages, "refund")

    assert result.count("\n--- [") + result.startswith("--- [") == MAX_GREP_MATCHES
    assert f"--- [{MAX_GREP_MATCHES - 1}]" in result
    assert f"--- [{MAX_GREP_MATCHES}]" not in result
    assert result.endswith("[5 more matches omitted. Refine your search pattern for more specific results.]")


def test_grep_exactly_at_the_match_limit_has_no_omission_note() -> None:
    messages = _dialogue(*(f"refund {i}" for i in range(MAX_GREP_MATCHES)))

    assert "omitted" not in grep_transcript(messages, "refund")


def test_grep_cuts_the_result_at_the_character_budget() -> None:
    messages = _dialogue(*("refund " + "x" * 1500 for _ in range(MAX_GREP_MATCHES)))

    result = grep_transcript(messages, "refund")

    assert len(result) > TOOL_RESULT_CHAR_BUDGET
    head, note = result.split("\n\n[TRUNCATED]")
    assert len(head) == TOOL_RESULT_CHAR_BUDGET
    assert "expand_transcript with fewer indices" in note
