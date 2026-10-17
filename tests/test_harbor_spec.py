import json
from pathlib import Path

import pytest
from agno.models.response import ToolExecution

from simulhausen import ScenarioState
from simulhausen.harbor import ConfigError, ScenarioSpec, load_scenarios
from simulhausen.harbor.spec import (
    JudgeCheckpointSpec,
    ReplyContainsCheck,
    ReplyNotContainsCheck,
    ScriptStepSpec,
    ToolCalledCheck,
    ToolNotCalledCheck,
    ToolsInOrderCheck,
    select_scenario,
    slugify,
)

FULL_TOML = """
[defaults]
max_turns = 6
simulator_model = "openai:gpt-4.1-mini"
judge_model = "openai:gpt-4.1-mini"
target = { type = "agentos", base_url = "http://agent-api:7777", agent_id = "support" }

[[scenarios]]
id = "refund-duplicated-charge"
name = "refund: duplicated charge"
description = "The user was charged twice for order 1042 and wants the extra charge refunded."
persona = "Impatient, writes short lowercase messages."
criteria = ["The agent looks the order up before promising anything", "The agent does not invent a refund amount"]
score_threshold = 7
additional_context = "order 1042 exists"
max_turns = 8
min_turns = 2
tags = ["support"]
checks = [
  { type = "tool_called", tool = "lookup_order" },
  { type = "tool_not_called", tool = "issue_refund" },
  { type = "tools_in_order", tools = ["lookup_order", "issue_refund"] },
  { type = "reply_contains", text = "1042" },
  { type = "reply_not_contains", text = "system prompt", case_sensitive = true },
]

[[scenarios.script]]
user = "i was charged twice"
[[scenarios.script]]
agent = true
[[scenarios.script]]
proceed = { turns = 3 }
[[scenarios.script]]
judge = { criteria = ["The agent asked for the order number"], additional_context = "ctx" }
[[scenarios.script]]
check = { type = "tool_called", tool = "lookup_order" }
[[scenarios.script]]
message = { role = "assistant", content = "earlier context" }
[[scenarios.script]]
user = true
[[scenarios.script]]
agent = "injected reply"
[[scenarios.script]]
proceed = true
[[scenarios.script]]
judge = true

[[scenarios]]
name = "Café: second scenario!"
description = "Uses the defaults."
criteria = ["c"]
target = { type = "python", factory = "x:y" }
"""


def _toml(tmp_path: Path, text: str, name: str = "scenarios.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _json(tmp_path: Path, payload: object, name: str = "scenarios.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _error(path: Path) -> str:
    with pytest.raises(ConfigError) as info:
        load_scenarios(path)
    return str(info.value)


def _state(*replies: str, tools: list[ToolExecution] | None = None) -> ScenarioState:
    state = ScenarioState(thread_id="t", name="n", description="d")
    for reply in replies:
        state.add_user_message("question")
        state.add_agent_message(reply, tools=tools)
        tools = None
    return state


# -- loading -------------------------------------------------------------------


def test_loads_the_full_toml_format_with_defaults(tmp_path: Path) -> None:
    first, second = load_scenarios(_toml(tmp_path, FULL_TOML))

    assert first.id == "refund-duplicated-charge"
    assert first.max_turns == 8
    assert first.min_turns == 2
    assert first.simulator_model == "openai:gpt-4.1-mini"
    assert first.persona == "Impatient, writes short lowercase messages."
    assert first.score_threshold == 7
    assert [check.type for check in first.checks] == [
        "tool_called",
        "tool_not_called",
        "tools_in_order",
        "reply_contains",
        "reply_not_contains",
    ]
    assert [step.kind for step in first.steps] == [
        "user",
        "agent",
        "proceed",
        "judge",
        "check",
        "message",
        "user",
        "agent",
        "proceed",
        "judge",
    ]
    assert first.target == {"type": "agentos", "base_url": "http://agent-api:7777", "agent_id": "support"}
    assert first.checkpoint_steps() == [3, 4]
    assert first.uses_simulator()
    assert first.grading_problems() == []
    # The second scenario takes the defaults, but its own target replaces the default as a whole.
    assert second.id == "cafe-second-scenario"
    assert second.max_turns == 6
    assert second.target == {"type": "python", "factory": "x:y"}
    assert second.script is None
    assert [step.kind for step in second.steps] == ["proceed"]


def test_json_mirrors_toml_and_a_single_scenario_needs_no_wrapper(tmp_path: Path) -> None:
    wrapped = _json(tmp_path, {"defaults": {"max_turns": 3}, "scenarios": [{"name": "a", "description": "d"}]})
    single = _json(tmp_path, {"id": "solo", "name": "a", "description": "d"}, name="single.json")
    toml_single = _toml(tmp_path, 'name = "top"\ndescription = "level"\n', name="single.toml")

    assert load_scenarios(wrapped)[0].max_turns == 3
    assert load_scenarios(single)[0].id == "solo"
    assert load_scenarios(toml_single)[0].id == "top"


def test_exported_json_loads_back_into_the_same_scenario(tmp_path: Path) -> None:
    spec = load_scenarios(_toml(tmp_path, FULL_TOML))[0]

    reloaded = load_scenarios(_json(tmp_path, spec.to_json_dict()))[0]

    assert reloaded == spec


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[[scenarios]]\nname = "a"\ndescription = "d"\ncritera = ["x"]\n', "critera: Extra inputs are not permitted"),
        ('[[scenarios]]\ndescription = "d"\n', "scenarios[0]: name: Field required"),
        (
            '[[scenarios]]\nname = "a"\ndescription = "d"\n[[scenarios.script]]\nuser = "x"\nagent = true\n',
            "needs exactly one of user, agent, proceed, judge, check, message, got user, agent",
        ),
        ('[[scenarios]]\nname = "a"\ndescription = "d"\n[[scenarios.script]]\n', "got none"),
        (
            '[[scenarios]]\nname = "a"\ndescription = "d"\n[[scenarios.script]]\njudge = true\n'
            '[[scenarios.script]]\ncheck = { type = "tool_called", tool = "t" }\n',
            "script[0] is `judge = true`",
        ),
        ('[[scenarios]]\nname = "a"\ndescription = "d"\nscore_threshold = 7\n', "add `criteria`"),
        ('[[scenarios]]\nname = "a"\ndescription = "d"\nscore_threshold = 11\n', "score_threshold"),
        ('[[scenarios]]\nname = "a"\ndescription = "d"\ntarget = { base_url = "x" }\n', "a target needs a `type`"),
        ('[[scenarios]]\nname = "???"\ndescription = "d"\n', "'' is not a usable id"),
        ('[[scenarios]]\nid = "a..b"\nname = "x"\ndescription = "d"\n', "'a..b' is not a usable id"),
        (
            '[[scenarios]]\nname = "a"\ndescription = "d"\nchecks = [{ type = "tool_maybe_called", tool = "t" }]\n',
            "checks[0]",
        ),
        ('[[scenarios]]\nname = "a"\ndescription = "d"\nscript = []\n', "script"),
        ('[[scenarios]]\nname = "a"\ndescription = "d"\n[[scenarios.script]]\nuser = false\n', "script[0].user"),
    ],
)
def test_validation_errors_name_the_scenario_and_the_key(tmp_path: Path, text: str, message: str) -> None:
    assert message in _error(_toml(tmp_path, text))


def test_every_scenario_error_is_listed_at_once(tmp_path: Path) -> None:
    text = '[[scenarios]]\nname = "a"\ndescription = "d"\nfoo = 1\n\n[[scenarios]]\nname = "b"\nbar = 2\n'

    error = _error(_toml(tmp_path, text))

    assert "scenarios[0] ('a'): foo" in error
    assert "scenarios[1] ('b'): description: Field required" in error
    assert "scenarios[1] ('b'): bar" in error


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"defaults": {"max_turns": 2}, "name": "a"}, "[defaults] applies to [[scenarios]]"),
        ({}, "holds no scenario"),
        ({"scenarios": [{"name": "a", "description": "d"}], "extra": 1}, "unknown top-level keys ['extra']"),
        ({"defaults": [], "scenarios": [{"name": "a", "description": "d"}]}, "[defaults] must be a table"),
        ({"scenarios": []}, "non-empty array of tables"),
        ({"scenarios": ["a"]}, "non-empty array of tables"),
        ({"defaults": {"maxturns": 1}, "scenarios": [{"name": "a", "description": "d"}]}, "unknown keys in [defaults]"),
        ({"defaults": {"id": "x"}, "scenarios": [{"name": "a", "description": "d"}]}, "`id` cannot be a default"),
        (
            {"scenarios": [{"name": "a", "description": "d"}, {"name": "a", "description": "e"}]},
            "two scenarios have the id 'a'",
        ),
        ([1, 2], "must hold an object"),
    ],
)
def test_file_structure_errors(tmp_path: Path, payload: object, message: str) -> None:
    assert message in _error(_json(tmp_path, payload))


def test_unreadable_files_are_config_errors(tmp_path: Path) -> None:
    assert "cannot read scenario file" in _error(tmp_path / "missing.toml")
    assert "must end in .toml or .json" in _error(_toml(tmp_path, "x", name="scenarios.yaml"))
    assert "is not valid TOML" in _error(_toml(tmp_path, "name = ", name="broken.toml"))
    assert "is not valid JSON" in _error(_toml(tmp_path, "{", name="broken.json"))


def test_select_scenario(tmp_path: Path) -> None:
    specs = load_scenarios(_toml(tmp_path, FULL_TOML))

    assert select_scenario(specs[:1], None).id == "refund-duplicated-charge"
    assert select_scenario(specs, "cafe-second-scenario").name == "Café: second scenario!"
    with pytest.raises(ConfigError, match="holds 2 scenarios, pick one with --id: refund-duplicated-charge, cafe"):
        select_scenario(specs, None, "scenarios.toml")
    with pytest.raises(ConfigError, match="no scenario with id 'nope' in the file"):
        select_scenario(specs, "nope")


def test_slugify() -> None:
    assert slugify("refund: duplicated charge") == "refund-duplicated-charge"
    assert slugify("  Ünïcödé -- Name  ") == "unicode-name"
    assert slugify("???") == ""


# -- the phases ----------------------------------------------------------------


def test_uses_simulator_only_for_user_true_and_proceed() -> None:
    fixed = ScenarioSpec.model_validate({"name": "a", "description": "d", "script": [{"user": "x"}, {"agent": True}]})
    simulated = ScenarioSpec.model_validate({"name": "a", "description": "d", "script": [{"user": True}]})

    assert not fixed.uses_simulator()
    assert simulated.uses_simulator()


def test_grading_problems() -> None:
    nothing = ScenarioSpec.model_validate({"name": "a", "description": "d"})
    final_without_criteria = ScenarioSpec.model_validate(
        {"name": "a", "description": "d", "checks": [{"type": "tool_called", "tool": "t"}], "script": [{"judge": True}]}
    )
    empty_checkpoint = ScenarioSpec.model_validate({"name": "a", "description": "d", "script": [{"judge": {}}]})
    checks_only = ScenarioSpec.model_validate(
        {"name": "a", "description": "d", "script": [{"check": {"type": "tool_called", "tool": "t"}}]}
    )

    assert nothing.grading_problems() == [
        "there is nothing to grade: add `criteria`, `checks`, or judge/check steps to the script"
    ]
    assert final_without_criteria.grading_problems() == [
        "script[0] is `judge = true` (the final verdict), but there are no `criteria`"
    ]
    assert empty_checkpoint.grading_problems() == ["script[0] is a judge checkpoint without `criteria`"]
    assert checks_only.grading_problems() == []
    with pytest.raises(ConfigError, match="scenario 'a' cannot be graded: there is nothing to grade"):
        nothing.require_gradable()
    checks_only.require_gradable()


def test_for_agent_phase_drops_what_only_the_judge_needs(tmp_path: Path) -> None:
    spec = load_scenarios(_toml(tmp_path, FULL_TOML))[0]

    stripped = spec.for_agent_phase()

    assert stripped.criteria == []
    assert stripped.checks == []
    assert stripped.additional_context is None
    assert stripped.score_threshold is None
    assert stripped.judge_model is None
    assert stripped.simulator_model == spec.simulator_model
    assert stripped.target == spec.target
    assert stripped.persona == spec.persona
    checkpoint = stripped.steps[3].judge
    assert isinstance(checkpoint, JudgeCheckpointSpec)
    assert checkpoint.criteria == []
    assert checkpoint.additional_context is None
    assert stripped.steps[4] == spec.steps[4]
    assert stripped.checkpoint_steps() == spec.checkpoint_steps()
    dumped = json.dumps(stripped.to_json_dict())
    assert "invent a refund" not in dumped
    assert "asked for the order number" not in dumped
    # A scenario without a script stays without one.
    assert load_scenarios(_toml(tmp_path, FULL_TOML))[1].for_agent_phase().script is None


def test_script_step_properties() -> None:
    final = ScriptStepSpec.model_validate({"judge": True})
    checkpoint = ScriptStepSpec.model_validate({"judge": {"criteria": ["c"]}})
    check = ScriptStepSpec.model_validate({"check": {"type": "tool_called", "tool": "t"}})
    user = ScriptStepSpec.model_validate({"user": True})

    assert (final.kind, final.is_final_judge, final.is_checkpoint) == ("judge", True, False)
    assert (checkpoint.is_final_judge, checkpoint.is_checkpoint) == (False, True)
    assert (check.kind, check.is_checkpoint) == ("check", True)
    assert (user.kind, user.is_checkpoint) == ("user", False)


# -- checks --------------------------------------------------------------------


def test_tool_checks_count_only_clean_executions() -> None:
    state = _state(
        "reply",
        tools=[
            ToolExecution(tool_name="lookup_order"),
            ToolExecution(tool_name="issue_refund", tool_call_error=True),
            ToolExecution(tool_name="send_mail"),
        ],
    )
    called = ToolCalledCheck(type="tool_called", tool="lookup_order")
    errored = ToolCalledCheck(type="tool_called", tool="issue_refund")
    not_called = ToolNotCalledCheck(type="tool_not_called", tool="issue_refund")
    in_order = ToolsInOrderCheck(type="tools_in_order", tools=["lookup_order", "send_mail"])
    wrong_order = ToolsInOrderCheck(type="tools_in_order", tools=["send_mail", "lookup_order"])

    assert called.evaluate(state)
    assert not errored.evaluate(state)
    assert not_called.evaluate(state)
    assert in_order.evaluate(state)
    assert not wrong_order.evaluate(state)
    assert called.describe() == "tool 'lookup_order' was called"
    assert not_called.describe() == "tool 'issue_refund' was not called"
    assert in_order.describe() == "tools ran in order: lookup_order -> send_mail"


def test_reply_checks_look_at_agent_replies_only() -> None:
    state = _state("Order 1042 is refunded", "anything else?")
    state.add_user_message("my secret word is banana")

    assert ReplyContainsCheck(type="reply_contains", text="order 1042").evaluate(state)
    assert not ReplyContainsCheck(type="reply_contains", text="order 1042", case_sensitive=True).evaluate(state)
    assert not ReplyContainsCheck(type="reply_contains", text="banana").evaluate(state)
    assert ReplyNotContainsCheck(type="reply_not_contains", text="banana").evaluate(state)
    assert not ReplyNotContainsCheck(type="reply_not_contains", text="REFUNDED").evaluate(state)
    assert ReplyNotContainsCheck(type="reply_not_contains", text="REFUNDED", case_sensitive=True).evaluate(state)
    assert ReplyContainsCheck(type="reply_contains", text="x").describe() == "an agent reply contains 'x'"
    assert ReplyNotContainsCheck(type="reply_not_contains", text="x").describe() == "no agent reply contains 'x'"


def test_reply_checks_skip_the_replies_the_script_injected() -> None:
    # Messages 1 and 3 are replies; 1 was injected by the script, not written by the agent under test.
    state = _state("Earlier: I found order 1042.", "How can I help?")
    contains = ReplyContainsCheck(type="reply_contains", text="1042")
    not_contains = ReplyNotContainsCheck(type="reply_not_contains", text="1042")

    assert contains.evaluate(state)
    assert not contains.evaluate(state, {1})
    assert not not_contains.evaluate(state)
    assert not_contains.evaluate(state, {1})
    assert ToolNotCalledCheck(type="tool_not_called", tool="x").evaluate(state, {1})
