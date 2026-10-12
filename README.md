# simulhausen

[![PyPI](https://img.shields.io/pypi/v/simulhausen)](https://pypi.org/project/simulhausen/)
[![Python](https://img.shields.io/pypi/pyversions/simulhausen)](https://pypi.org/project/simulhausen/)
[![CI](https://github.com/ormeilu/simulhausen/actions/workflows/ci.yml/badge.svg)](https://github.com/ormeilu/simulhausen/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/ormeilu/simulhausen/blob/main/LICENSE)

Simulation testing for [agno](https://github.com/agno-agi/agno) agents that runs entirely on your side.
An LLM plays the user and holds a multi-turn dialogue with your agent, team or workflow. Then an LLM judge
grades the result against your criteria, one verdict and one reason per criterion.

The API follows [LangWatch Scenario](https://github.com/langwatch/scenario), minus the cloud: no LangWatch
account, no tracing backend, no litellm. The user simulator and the judge are plain agno agents on whatever
model you configure, and the reports land in your terminal, in a JSON file and in a markdown comment for your
pull request.

```python
import simulhausen as sim

result = await sim.run(
    name="refund: duplicated charge",
    description="The user was charged twice for order 1042 and wants the extra charge refunded.",
    agents=[
        sim.agno_adapter(lambda: support_agent),
        sim.UserSimulatorAgent(),
        sim.JudgeAgent(criteria=[
            "The agent looks the order up before promising anything",
            "The agent does not invent a refund amount",
        ]),
    ],
)
assert result.success, result.failure_summary()
```

## Install

```bash
uv add --dev simulhausen
```

or `pip install simulhausen`. Python 3.12+ and agno 3.x. The pytest plugin registers itself on install.

## Quick start

Configure the model for the simulator and the judge once, in `conftest.py`. Keep model construction inside the
fixture, so collecting tests never needs credentials:

```python
# conftest.py
import pytest

import simulhausen as sim


@pytest.fixture(scope="session", autouse=True)
def configure_simulations() -> None:
    from agno.models.openai import OpenAIChat

    sim.configure(model=OpenAIChat(id="gpt-4.1-mini"), max_turns=8)
```

Then write scenarios as ordinary async tests:

```python
# test_support.py
import pytest

import simulhausen as sim


def _support_agent():
    from myapp.agents import support_agent  # lazy: imported only when the test runs

    return support_agent


def _no_lookup_yet(state: sim.ScenarioState) -> None:
    assert not state.has_tool_call("lookup_order"), state.tool_call_names()


@pytest.mark.simulation
async def test_refund_asks_for_order_number() -> None:
    result = await sim.run(
        name="refund: missing order number",
        description="The user wants a refund but does not say which order it is about.",
        agents=[
            sim.agno_adapter(_support_agent, user_id="user-42"),
            sim.UserSimulatorAgent(),
            sim.JudgeAgent(criteria=["The agent asks for the order number before anything else"]),
        ],
        script=[
            sim.user("i want my money back"),
            sim.agent(),
            _no_lookup_yet,
            sim.judge(),
        ],
    )
    assert result.success, result.failure_summary()
```

`agno_adapter` takes the agent itself or a factory, and forwards extra keyword arguments to `arun`
(`session_state=...`, `user_id=...`). The whole scenario runs in one agno session. A runnable version lives in
[`examples/`](https://github.com/ormeilu/simulhausen/tree/main/examples).

## Scripts

Without `script`, a run is `[proceed()]` followed by the judge's verdict. With a script you decide what happens
and in which order:

| Step | What it does |
|---|---|
| `user("text")` | user message with fixed text |
| `user()` | user message written by the simulator |
| `agent()` | call the agent under test |
| `agent("text")` | inject an agent reply without calling the agent |
| `message(role, "text")` | append any message as is, to seed earlier context |
| `proceed(turns=None, on_turn=..., on_step=...)` | simulator and agent talk on their own (see below) |
| `judge()` | final verdict on the judge's criteria; later steps still run |
| `judge(criteria=[...], additional_context=...)` | checkpoint: fails the scenario on the spot if a criterion fails, goes on otherwise |
| `succeed(reasoning)` / `fail(reasoning)` | end the scenario right here |
| any callable | gets the `ScenarioState`; sync or async; raise `AssertionError` to fail, return `succeed()`/`fail()` to end the scenario |

`proceed()` stops at the first of these: `turns` turns played, `max_turns` reached, the simulator reporting the
goal as achieved, or the judge concluding it has seen enough. The judge is not consulted before `min_turns`
agent turns. If it concludes while some criteria are still inconclusive and turns remain, the dialogue goes on
to collect the missing evidence.

When a script has a judge with criteria but no `judge()` step, the judge runs after the last step. It also runs
again there if the dialogue went on after its last verdict, so the final verdict always covers the whole
dialogue; a new final verdict replaces the previous one, while checkpoint verdicts accumulate.
`additional_context` gives the judge evidence the dialogue cannot show, such as a database row the agent was
supposed to write.

## Verdicts and statuses

The judge answers per criterion: `passed`, `failed` or `inconclusive`, each with a restated requirement and its
reasoning. A criterion phrased as a prohibition ("the agent must not X") passes when X did not happen.
`inconclusive` means the evidence was not there, for example because the dialogue ended before the moment the
criterion depends on.

Every result gets one of three statuses:

- `PASS`: every criterion passed and no assertion failed.
- `FAIL`: a criterion failed, a script assertion failed, the judge score was below threshold, or `fail()` ran.
- `WARN`: the scenario could not be verified. Either the harness broke (`SimulationHarnessError`: the judge or
  the simulator returned no structured output, the model backend kept failing), or some criteria came back
  inconclusive and none failed.

WARN is kept out of the pass rate on purpose. A broken judge says nothing about your agent, and counting it as
a failure turns the pass rate into a measure of helper-LLM reliability. `pass_rate = PASS / (PASS + FAIL)`, and
the WARN count is reported next to it. Harness errors are still raised, so a plain pytest run fails on them.

`ScenarioResult` carries `verdicts`, `passed_criteria`, `failed_criteria`, `inconclusive_criteria`, the judge's
`reasoning`, `messages`, `tool_calls`, `turns`, `total_time` and `agent_time`.

For an overall quality gate on top of the criteria, `JudgeAgent(criteria, score_threshold=7)` also grades the
whole dialogue 1-10 with agno's `JudgeScorer`. Calibrate the threshold on known good and bad runs first.

### Long dialogues

A dialogue over `token_threshold` estimated tokens (UTF-8 bytes / 4, default 8192, roughly 32 KB of text) is not
pasted into the judge's prompt. The judge gets a skeleton instead, one line per message with its index, speaker,
size and a short preview, plus two tools: `expand_transcript(indices)` returns whole messages and
`grep_transcript(pattern)` finds messages by text (plain text, case-insensitive, not a regex). The prompt tells it
to read what a criterion depends on before ruling. What the tools return reaches the judge fenced as untrusted
data, like the rest of the dialogue. The continue/conclude decision of unscripted runs works the same way. Below
the threshold nothing changes.

```python
sim.JudgeAgent(criteria, token_threshold=16_000, max_discovery_steps=6)
```

`max_discovery_steps` (default 10) is passed to agno as `tool_call_limit` and caps the lookups per run. agno's
limit is soft: it answers a call past the limit with an error result and asks the model again, so the verdict is
whatever the model writes next. A decision that never came after the whole budget was spent concludes the
dialogue, and the verdict follows. Each lookup is a model round trip, so a long dialogue costs up to
`max_discovery_steps` extra calls per verdict and, in unscripted runs, per turn. Raise `token_threshold` to keep
discovery off, or lower `max_discovery_steps` to cap the cost. References, tool executions and
`additional_context` keep their caps and are not part of discovery, and `score_threshold` grading still reads
the whole dialogue.

## Checking tools

`agno_adapter` collects every `ToolExecution` of the run, including those of team members, so assertions work
for teams too:

```python
state.has_tool_call("lookup_order")         # ran cleanly at least once
state.tool_call_names()                     # names in call order, with repeats
state.has_tool_calls_in_order(["search", "lookup_order"])  # subsequence; other calls may sit between
state.last_tool_call("lookup_order")        # the ToolExecution, or None

calls = state.tool_calls("lookup_order")    # ToolCalls: a sequence of ToolCall(name, input, output, turn)
calls.last.input == {"order_id": "1042"}
calls.outputs                               # every result, in order
assert not state.tool_calls("refund").first # an empty, falsy call when the tool never ran
```

Only clean executions count: a call that errored, was rejected or is paused waiting for confirmation does not
satisfy `has_tool_call`. That matches `agno.scorer.ToolCallScorer`.

The judge sees the same evidence: the transcript, the RAG references the agent retrieved (with source URLs) and
a capped summary of tool executions, errors included. Each block is fenced with a random per-call tag and the
judge is told never to follow instructions inside them.

## Writing scenarios that hold up

One scenario checks one behaviour from one starting state. When it fails, the name alone should tell you what
broke.

- `name` is the outcome in a few words, readable in a PR comment: `refund: asks for the order number`.
- `description` is for the simulator: starting conditions, the user's goal, how they behave. It is not a
  criterion.
- Each criterion is a single observable requirement. "The answer is good" is not one.

Check in code whatever code can check: tool calls and their order, exact identifiers, an expected error, a call
that must not happen. Leave semantics to the judge: did the agent make something up, explain a limitation,
stick to what the tools returned.

Spell out the user messages that matter for the regression with `user("...")`. Use `user()` or `proceed()` only
where variety in the user's wording is part of what you test. Most regressions need one or two user/agent
pairs; every extra turn costs money and adds variance.

Do not paper over a reproducible failure with reruns. Look at `failure_summary()`, the tool evidence and the
transcript in the JSON report first: they tell an agent bug from a broken backend, a wrong assertion or a vague
criterion.

## Running

Simulation tests call real models, so keep them out of the default run:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not simulation'"
asyncio_mode = "auto"
```

```bash
uv run pytest -m simulation                    # failures fail the run
uv run pytest -m simulation --sim-benchmark \
  --sim-report reports/simulations.json \
  --sim-report-md reports/simulations.md       # always exits 0, writes reports
uv run pytest -m simulation -k refund -s --sim-debug  # you type the user's messages
```

Benchmark mode is for CI jobs that should stay green and post the outcome instead. After any run with
simulations, a rich table with every scenario, its verdicts, tools and dialogue is printed. With
`pytest-rerunfailures` only the last attempt is reported.

Parallel runs work with `pytest-xdist` (`-n 4`): each worker is its own process with its own event loop, and
the controller merges every worker's results into one report. Prefer that to `asyncio.gather` inside a test,
because agno agents and HTTP clients are usually process-wide singletons. When an agent's async clients bind to
the first event loop they see, run simulations on one session-scoped loop:

```python
# conftest.py
def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("simulation"):
            item.add_marker(pytest.mark.asyncio(loop_scope="session"), append=False)
```

### Reports in pull requests

The markdown report starts with a hidden `<!-- simulhausen-report -->` marker and a headline such as
`🟡 88% ▓▓▓▓▓▓▓▓▓░ · PASS 154 / FAIL 21`. The emoji follows the pass rate (🟢 from 90% with no WARN, 🟡 from
75%, 🔴 below), since one failure out of 175 is noise, not a regression. The bar shows passed, not verified and
failed scenarios out of all of them. On GitHub Actions:

```yaml
- run: uv run pytest -m simulation --sim-benchmark --sim-report-md reports/simulations.md
- if: github.event_name == 'pull_request'
  run: gh pr comment ${{ github.event.pull_request.number }} --body-file reports/simulations.md
  env:
    GH_TOKEN: ${{ github.token }}
```

The report links the CI job when `CI_JOB_URL` (GitLab) or the `GITHUB_*` run variables are set.

## Configuration

```python
sim.configure(
    model=...,             # agno Model for the simulator and the judge (required)
    parser_model=...,      # optional agno Model that turns their answers into the schema
    max_turns=10,          # ceiling for proceed() and unscripted runs
    min_turns=1,           # the judge may not end the dialogue earlier
    verbose=True,          # log every message and verdict (logger "simulhausen")
    debug=False,           # ask for the user's messages on stdin
    user_name="User",      # labels in reports
    agent_name="Agent",
)
```

`run()` accepts `max_turns`, `min_turns`, `verbose` and `debug` per scenario. `UserSimulatorAgent` and
`JudgeAgent` accept their own `model` and `parser_model`, plus `system_prompt` to replace the built-in prompt
([`simulhausen.prompts`](https://github.com/ormeilu/simulhausen/blob/main/src/simulhausen/prompts.py)). The simulator also takes a `persona`.

Set `parser_model` when your backend breaks structured output in thinking mode. Some vLLM setups with a
reasoning parser return empty or broken content when `response_format` meets reasoning; with a parser model,
agno sends `response_format` only to the parser call. The helper agents also retry an off-schema answer twice,
with a stricter instruction each time.

## Custom adapters

`agno_adapter` covers agno agents, teams and workflows. It retries backend failures (a run that ended in
`RunStatus.error`, a proxy's HTML error page) and raises `SimulationBackendError` when they persist, while a
guardrail refusal (`InputCheckError`/`OutputCheckError`) counts as the agent's answer. For agents with
`output_schema`, pass `content_fn` to render the structured reply as text.

Anything else is a function or an `AgentAdapter` subclass:

```python
async def call_my_api(input: sim.AgentInput) -> str:
    return await my_client.chat(input.thread_id, input.last_new_user_message_str())

adapter = sim.function_adapter(call_my_api)


class MyAdapter(sim.AgentAdapter):
    async def call(self, input: sim.AgentInput) -> sim.AgentTurn:
        reply = await my_agent.respond(input.messages)
        return sim.AgentTurn(content=reply.text, tools=reply.tool_executions)
```

`AgentInput` has the whole transcript (`messages`), what changed since the agent last spoke (`new_messages`)
and the live `scenario_state`.

`state.rollback_messages_to(index)` cuts the transcript back (with the tool calls and references of the removed
turns) and returns what it removed; rolled-back turns still count against `max_turns`. An agno agent keeps its
own session history, so for the rollback to reach the agent use `agno_adapter(..., history="transcript")`: every
call then replays the transcript as messages in a fresh session.

## Compared with LangWatch Scenario

Ported: the script steps (`user`, `agent`, `message`, `proceed`, `judge` with inline criteria and
`additional_context`, `succeed`, `fail`), per-criterion `passed`/`failed`/`inconclusive` verdicts, the two-phase
judge (an argument-free continue/conclude decision, then the verdict), `min_turns`, simulator `persona` and
`system_prompt`, debug mode, `total_time`/`agent_time`, long-transcript discovery (a skeleton plus
`expand_transcript` and `grep_transcript` once the dialogue passes `token_threshold`).

Different: models are agno `Model` objects instead of litellm strings; the simulator reports `goal_achieved` so
unscripted dialogues end without filler; harness failures and inconclusive scenarios are WARN and stay out of
the pass rate; a script that ends without a judge verdict passes if its assertions passed; the judge also gets
RAG references and tool executions; skeleton lines carry a short preview of each message; the discovery step
limit is agno's soft `tool_call_limit` and not a final verdict call with the tools stripped.

Not ported: anything that needs LangWatch (tracing, events, evaluators, remote traces), voice and realtime
agents, the red-team agents and the joblib cache.

## Development

```bash
uv sync
just ci          # ruff, ty, pytest with coverage
just precommit   # every prek hook
```

## License

Apache-2.0, see [LICENSE](https://github.com/ormeilu/simulhausen/blob/main/LICENSE). simulhausen began as a local reimplementation of LangWatch Scenario
(Apache-2.0, Copyright LangWatch); its API and parts of its prompts follow Scenario. See [NOTICE](https://github.com/ormeilu/simulhausen/blob/main/NOTICE).
