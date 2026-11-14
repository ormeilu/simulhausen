# simulhausen

Local simulation testing for AI agents: an LLM user simulator talks to the agent under test (agno, another
framework through `simulhausen/frameworks/`, or an HTTP API), an LLM judge grades the dialogue per criterion. API modelled on LangWatch Scenario; user-facing docs live in `README.md`.

## Working here

- Gate before every commit: `just ci` (ruff, ruff format check, ty, pytest with coverage). `just precommit` runs
  the prek hooks.
- English everywhere: code, docstrings, prompts, reports, docs.
- Python 3.12 is the floor and ty targets it, so 3.13+ stdlib APIs fail the gate. PEP 695 generics and `type`
  aliases are fine.
- Runtime deps stay at agno, pydantic and rich. LangWatch, litellm and OpenTelemetry stay out of a plain install:
  being local-only is the point of the package. The one exception is the `harbor` extra (Harbor requires
  litellm): only `simulhausen/harbor/agents.py` imports harbor, and `tests/test_harbor_agent_loading.py` checks
  that a plain import loads neither.

- The frameworks behind `simulhausen/frameworks/*.py` (LangChain, CrewAI, OpenAI Agents and the rest) are not
  dependencies and are not in `uv.lock`. An adapter module imports nothing from its framework at import time: it
  duck-types, or gets the module through `require()` from `frameworks/_base.py` inside the call.
  `tests/test_frameworks_base.py` checks that `import simulhausen` loads none of them.

## Tests

- Tests never call a model. Fake the helper agents by subclassing `UserSimulatorAgent` / `JudgeAgent`, or
  monkeypatch `simulhausen.agents.Agent` with a stub whose `arun` returns `SimpleNamespace(content=...)`.
- `simulhausen.collector` and the `configure()` dataclass are process globals; the autouse fixture in
  `tests/conftest.py` resets both around each test. Its autouse `cache_dir` fixture points the response cache at a
  per-test temporary directory, so no test touches `~/.cache/simulhausen`.
- `addopts` carries `-p no:simulhausen` on purpose: the entry-point plugin would import the package before
  coverage starts. Plugin behaviour is tested through `pytester.runpytest_subprocess`, where the entry point loads
  normally.
- Each framework adapter has two test files. `tests/test_frameworks_<name>.py` uses hand-written fakes, never
  imports the framework and covers the module. `tests/contract/test_<name>_contract.py` drives the real library
  with a scripted model (no network) and is skipped in the project environment; `just contract <name>` installs
  the framework into an overlay and runs it, and the `contract` CI job does that for every framework against its
  latest release. ty skips `tests/contract`. A new adapter gets both files, a line in the justfile's `contract`
  recipe and in the CI matrix, an export in `__init__.py` and a row in the README table.
- `examples/` has its own `pytest.ini` and calls a real model (`OPENAI_API_KEY=... uv run --with openai pytest
  examples`); CI only lints and type-checks it.

## Semantics to keep consistent

- Status rules live in `ScenarioResult.status` (`types.py`): PASS; FAIL; WARN when the harness broke
  (`SimulationHarnessError`) or criteria came back inconclusive with none failed. Pass rate is
  PASS / (PASS + FAIL). `reporting.py` and the README describe the same rules, so a change touches all three.
  The Harbor rewards in `harbor/judge.py` follow them too (PASS 1, FAIL 0, WARN no `reward.json` by default).
- `has_tool_call` and friends count only clean executions (no error, not paused), matching agno's
  `ToolCallScorer`. `tool_calls(include_errors=True)` and `match_trajectory(include_errors=True)` are the
  opt-in exceptions.
- Pass rates leave WARN out everywhere: `SummaryStats.pass_rate`, `TrialsResult` (`run_trials`), `diff.py` and the
  Harbor metric.
- A new field on `ScenarioResult` goes through `serialize_records`/`deserialize_records` in `reporting.py` (the
  xdist hop and the JSON report), and `diff.py` reads that JSON.
- Prompt text in `prompts.py` goes to the model verbatim (hence its E501 exemption). A wording change is a
  behaviour change: commit it as `feat:` or `fix:` so it reaches the changelog.
- Untrusted data (transcript, RAG references, tool results, additional context) reaches the judge only through
  `_fence()` in `agents.py`.
- The response cache (`_cache.py`) wraps the helper agents at the runner's call sites (`_helper_call`), keyed by
  the agent's `cache_fingerprint()` plus the call's arguments and the state. A new `UserSimulatorAgent` /
  `JudgeAgent` option that changes what the model is asked or answers goes into `cache_fingerprint()` (or into the
  call's arguments), or reruns replay stale answers. A subclass with hidden state sets `cacheable = False`.

## Licensing

Apache-2.0. `NOTICE` credits LangWatch Scenario, whose API and prompts this package adapts. Anything else ported
from upstream (`github.com/langwatch/scenario`, `python/scenario/`) gets a line in `NOTICE`.

## Releasing

release-please owns versions and the changelog: `pyproject.toml`'s version, the `simulhausen` entry in `uv.lock`,
`CHANGELOG.md` and `.release-please-manifest.json` change only through its Release PR.

- Commit types drive the bump (Conventional Commits): `fix:` patch, `feat:` minor, `feat!:` / `BREAKING CHANGE:`
  minor while below 1.0 (`bump-minor-pre-major`). `feat`, `fix`, `perf`, `revert`, `docs` and `deps` reach the
  changelog; `refactor`, `test`, `ci` and `chore` stay hidden.
- Every push to `main` updates the open Release PR. Merging it tags `v<version>`, creates the GitHub release, and
  `.github/workflows/release.yml` re-runs the tests, builds and publishes to PyPI through Trusted Publishing
  (environment `pypi`; the publisher is registered for `release.yml`, so the file keeps that name).
- A release PR opened with `GITHUB_TOKEN` gets no CI run of its own; its only changes are the version and the
  changelog, and the release job runs the tests again before publishing.
- To force a version, add a `Release-As: x.y.z` footer to a commit on `main`.
