# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-08

First public release, extracted from an in-house framework that replaced `langwatch-scenario` with a local,
agno-based implementation.

### Added

- `run()` with scripted and unscripted dialogues, script steps `user`, `agent`, `message`, `proceed`, `judge`
  (final verdict or inline checkpoint with `additional_context`), `succeed`, `fail`, and callable assertions.
- `UserSimulatorAgent` with `goal_achieved` early stop, `persona` and `system_prompt`.
- `JudgeAgent` with per-criterion `passed` / `failed` / `inconclusive` verdicts, a judge summary, a two-phase
  continue/conclude decision, `min_turns`, optional numeric grading through agno's `JudgeScorer`
  (`score_threshold`), and fenced untrusted evidence (transcript, RAG references, tool executions).
- `agno_adapter` for agno Agent, Team and Workflow: team member tool calls merged, backend failures retried and
  raised as `SimulationBackendError`, guardrail refusals treated as answers, `content_fn` for structured replies.
- `function_adapter` for plain sync or async functions.
- PASS / FAIL / WARN statuses: harness failures and inconclusive-only verdicts are WARN and stay out of the pass
  rate.
- pytest plugin: `--sim-benchmark`, `--sim-report`, `--sim-report-md`, `--sim-debug`, pytest-xdist aggregation,
  rich terminal summary, markdown report for pull/merge request comments.
- `configure()` with `model`, `parser_model`, `max_turns`, `min_turns`, `verbose`, `debug`, report labels.

[Unreleased]: https://github.com/ormeilu/simulhausen/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ormeilu/simulhausen/releases/tag/v0.1.0
