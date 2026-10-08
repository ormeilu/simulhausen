# simulhausen

Local simulation testing for agno agents: an LLM user simulator talks to the agent under test, an LLM judge
grades the dialogue per criterion. API modelled on LangWatch Scenario; user-facing docs live in `README.md`.

## Working here

- Gate before every commit: `just ci` (ruff, ruff format check, ty, pytest with coverage). `just precommit` runs
  the prek hooks.
- English everywhere: code, docstrings, prompts, reports, docs.
- Python 3.12 is the floor and ty targets it, so 3.13+ stdlib APIs fail the gate. PEP 695 generics and `type`
  aliases are fine.
- Runtime deps stay at agno, pydantic and rich. LangWatch, litellm and OpenTelemetry stay out: being local-only
  is the point of the package.

## Tests

- Tests never call a model. Fake the helper agents by subclassing `UserSimulatorAgent` / `JudgeAgent`, or
  monkeypatch `simulhausen.agents.Agent` with a stub whose `arun` returns `SimpleNamespace(content=...)`.
- `simulhausen.collector` and the `configure()` dataclass are process globals; the autouse fixture in
  `tests/conftest.py` resets both around each test.
- `addopts` carries `-p no:simulhausen` on purpose: the entry-point plugin would import the package before
  coverage starts. Plugin behaviour is tested through `pytester.runpytest_subprocess`, where the entry point loads
  normally.
- `examples/` has its own `pytest.ini` and calls a real model (`OPENAI_API_KEY=... uv run --with openai pytest
  examples`); CI only lints and type-checks it.

## Semantics to keep consistent

- Status rules live in `ScenarioResult.status` (`types.py`): PASS; FAIL; WARN when the harness broke
  (`SimulationHarnessError`) or criteria came back inconclusive with none failed. Pass rate is
  PASS / (PASS + FAIL). `reporting.py` and the README describe the same rules, so a change touches all three.
- `has_tool_call` and friends count only clean executions (no error, not paused), matching agno's
  `ToolCallScorer`.
- Prompt text in `prompts.py` goes to the model verbatim (hence its E501 exemption). A wording change is a
  behaviour change: note it in `CHANGELOG.md`.
- Untrusted data (transcript, RAG references, tool results, additional context) reaches the judge only through
  `_fence()` in `agents.py`.

## Licensing

Apache-2.0. `NOTICE` credits LangWatch Scenario, whose API and prompts this package adapts. Anything else ported
from upstream (`github.com/langwatch/scenario`, `python/scenario/`) gets a line in `NOTICE`.

## Releasing

1. `uv version --bump minor` (or `patch`), move the `Unreleased` notes in `CHANGELOG.md` under the new version.
2. Merge to `main`, then publish a GitHub release tagged `v<version>`.
3. `.github/workflows/release.yml` checks the tag against `pyproject.toml`, builds, and publishes to PyPI through
   Trusted Publishing (environment `pypi`, no token).

Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
