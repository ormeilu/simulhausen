# simulhausen-harbor

Harbor agents that evaluate an HTTP agent (agno AgentOS, an OpenAI-compatible endpoint, A2A, LangGraph) with
simulhausen's user simulator and judge. [Harbor](https://github.com/harbor-framework/harbor) runs the trials,
collects the rewards and shows the results; simulhausen plays and grades the dialogues.

This is a separate distribution because Harbor depends on litellm, and simulhausen itself stays at agno, pydantic
and rich. The only part of simulhausen that runs here is its CLI, inside the task container.

## How it fits together

1. `simulhausen harbor export scenarios.toml --output-dir tasks` turns each scenario into a Harbor task. The image
   gets simulhausen and the play-side scenario (no criteria); `tests/` gets the full scenario and a `test.sh` that
   runs the judge.
2. The agent under test runs next to the task container, usually as a compose sidecar named `agent-api`.
3. Agent phase: the Harbor agent from this package execs `simulhausen harbor play` in the task container. The user
   simulator talks to the agent under test over HTTP and the transcript lands in `/logs/agent/simulhausen/`.
4. Verifier: `tests/test.sh` runs `simulhausen harbor judge` on that transcript and writes
   `/logs/verifier/reward.json`. The criteria never reach the agent phase.

## Install

Until it is on PyPI, install it from git next to Harbor (0.24.x):

```bash
uv tool install harbor --with "simulhausen-harbor @ git+https://github.com/ormeilu/simulhausen#subdirectory=integrations/harbor"
```

`-a simulhausen_harbor:AgentOSAgent` must be importable by the `harbor` process, hence `--with`.

Inside the container, exported task images already have the `simulhausen` CLI. If the image lacks it, the agent
installs `simulhausen==0.3.0` plus `openai` with uv during setup (`simulhausen_spec`, `version` and `pip` change
that). simulhausen 0.3.0 is the first release with the Harbor CLI; until it is out, install it from a source
archive both when exporting and in the agent (a `git+` URL works too, but the slim images then need git first):

```bash
simulhausen harbor export scenarios.toml --output-dir tasks \
  --simulhausen-spec "simulhausen @ https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz"
harbor run ... --ak 'simulhausen_spec=simulhausen @ https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz'
```

## Option c: the agent under test is the Harbor agent

The Harbor agent column then names the protocol and `-m` names the agent under test, so results group per agent
and version.

```bash
export OPENAI_API_KEY=sk-...    # judge ([verifier.env]) and the sidecar read it from this shell
harbor run -p tasks \
  -a simulhausen_harbor:AgentOSAgent --ak base_url=http://agent-api:7777 --ak agent_id=support \
  -m support-agent@v1 \
  --extra-docker-compose examples/agentos/agentos-sidecar.yaml \
  --ae OPENAI_API_KEY=$OPENAI_API_KEY -y
```

- `--ak` sets the target and the play options (reference below; `harbor agent schema
  simulhausen_harbor:AgentOSAgent` prints it too).
- `--ae` passes environment variables to the agent phase only: the simulator's model key here, and whatever an
  `*_env` kwarg names.
- `--extra-docker-compose` adds the agent under test as a sidecar for this run. A task exported with
  `--compose FILE` carries the sidecar itself and needs no flag.
- Add `-k 3` for three attempts per task and Harbor's pass@k.

`examples/agentos/` is a runnable version of this, including a job config with the pass-rate metric.

## Option a: everything in the verifier

Export with `--mode verifier` and run Harbor's no-op agent. `tests/test.sh` then runs `simulhausen harbor run`:
play and judge, both in the verifier, against the target from the scenario file (or `SIMULHAUSEN_TARGET`). This
needs nothing from this package except, optionally, the metric.

```bash
simulhausen harbor export scenarios.toml --output-dir tasks-verifier --mode verifier
harbor run -p tasks-verifier -a nop \
  --extra-docker-compose examples/agentos/agentos-sidecar.yaml --job-name support-agent-v1 -y
```

The agent column says `nop`, so put the identity of the agent under test into the job name. The whole dialogue
runs inside `[verifier].timeout_sec`, and only in Harbor's default shared verifier mode: a separate verifier
environment has no sidecars.

## Agents

| Class (`-a simulhausen_harbor:...`) | `name()` | Target type | Flat kwargs |
|---|---|---|---|
| `SimulhausenAgent` | `simulhausen` | from `target` | none: `target`, or nothing to use `SIMULHAUSEN_TARGET` / the scenario's target |
| `AgentOSAgent` | `simulhausen-agentos` | `agentos` | `base_url` (required), `agent_id` / `team_id` / `workflow_id`, `api_key_env`, `user_id`, `session_state`, `history` |
| `OpenAIChatAgent` | `simulhausen-openai-chat` | `openai-chat` | `base_url` (required), `model` (default: `-m`), `api_key_env`, `system_prompt`, `extra_body` |
| `A2AAgent` | `simulhausen-a2a` | `a2a` | `url` (required) |
| `LangGraphAgent` | `simulhausen-langgraph` | `langgraph` | `base_url`, `assistant_id` (both required), `api_key_env`, `input_key` |

All HTTP agents also take `headers`, `timeout`, `retries` and `retry_delay`. The flat kwargs you set build the
target dict, which is written to `/installed-agent/simulhausen/target.json` and passed to play as `--target-file`. An
explicit `target` (a dict, or a JSON string) wins key by key, so it also reaches options without a flat kwarg,
such as extra AgentOS form fields. Other target types (`openai-responses`, `python`) go through
`SimulhausenAgent` with `--ak 'target={"type": "openai-responses", ...}'`.

Shared kwargs:

| Kwarg | Default | Meaning |
|---|---|---|
| `target` | none | Target config (see above). |
| `simulator_model` | none | agno model string for the user simulator, e.g. `openai:gpt-4.1-mini`. Unset: `SIMULHAUSEN_SIMULATOR_MODEL`, `SIMULHAUSEN_MODEL` (via `--ae`), then the scenario. |
| `scenario_path` | `/simulhausen/scenario.json` | Scenario file in the container (where export puts it). |
| `simulhausen_spec` | `simulhausen==0.3.0` | Requirement installed when the image has no `simulhausen` CLI. |
| `version` | none | Shorthand for `simulhausen_spec=simulhausen==<version>`. |
| `pip` | `["openai"]` | Extra requirements installed with it: the simulator model's SDK. |
| `play_timeout_sec` | none | Interrupt play after this many seconds (SIGINT, SIGKILL 30 s later). The verifier still runs. |

Kwargs are validated before Harbor queues any trial: a typo or a missing `base_url` stops the job at once.

The command the agent runs, as the task's agent user, with output kept in `play.log`:

```bash
simulhausen harbor play --scenario /simulhausen/scenario.json \
  --target-file /installed-agent/simulhausen/target.json --out /logs/agent/simulhausen [--model M]
```

A non-zero exit (2 means the harness broke; the transcript is still written) raises Harbor's
`NonZeroAgentExitCodeError`: Harbor records it and still runs the verifier, which reports the scenario as WARN.

## Identity

Harbor groups every statistic by `<agent>__<model>__<dataset>`, and an agent's name is fixed per class. Pass the
identity of the agent under test as the model, e.g. `-m support-agent@v1`; Harbor 0.24 records it as given
(there is no `--sm` flag in 0.24, and none is needed). `OpenAIChatAgent` also sends it as the endpoint's `model`
unless `--ak model=...` says otherwise.

## Secrets

- Never put a key into `--ak`: Harbor stores kwargs in the job's `config.json`, and the target is written to a
  file in the container. A literal `api_key` is refused. Name a variable instead and pass its value with `--ae`:
  `--ak api_key_env=AGENTOS_API_KEY --ae AGENTOS_API_KEY=$AGENTOS_API_KEY`. Credential-looking headers get a
  warning.
- Simulator key: `--ae OPENAI_API_KEY=...` (agent phase only).
- Judge key: exported tasks read `${OPENAI_API_KEY}` in `[verifier.env]` from the shell that runs `harbor run`
  (other names: `simulhausen harbor export --env NAME`), or `--ve OPENAI_API_KEY=...`.
- Sidecar key: compose interpolation, e.g. `OPENAI_API_KEY=${OPENAI_API_KEY}` in the overlay, also read from
  that shell.

## Where results land

```
jobs/<job>/
├── result.json                       # stats.evals["simulhausen-agentos__support-agent@v1__tasks"]: metrics, rewards
└── <task>__<id>/
    ├── result.json                   # agent_info, agent_result.metadata, verifier_result.rewards, exception_info
    ├── agent/
    │   ├── trajectory.json           # copy of play's ATIF trajectory, for `harbor view jobs`
    │   └── simulhausen/              # transcript.json, transcript.md, trajectory.json, play.log
    └── verifier/                     # reward.json, simulhausen-result.json, transcript.md, test-stdout.txt
```

`agent_result.metadata` summarises the transcript: `turns`, `messages`, `tool_calls` (clean executions only),
`failed_tool_calls`, `stop_reason`, `harness_error`, `error`, `agent_time`, `total_time`, the scenario id and the
simulhausen version. A missing or broken transcript shows up as `transcript_error`.

## Pass rate metric

`reward.json` holds `{"reward": 1}` for PASS and `{"reward": 0}` for FAIL. WARN writes no reward, so Harbor
records an error for that trial and its default mean counts it as 0. simulhausen keeps WARN out of the pass rate,
and `simulhausen_harbor/metric.py` (a Harbor `uv-script` metric, standard library only) does the same:

```yaml
metrics:
  - type: uv-script
    kwargs:
      script_path: src/simulhausen_harbor/metric.py   # relative to where `harbor run` runs
```

It reports `pass_rate` (PASS / (PASS + FAIL), 0-1), `n_pass`, `n_fail`, `n_warn`, and the mean of every other
numeric reward key over the trials that have it (`judge_score`, `criteria_pass_rate`, `criterion_<n>` with
`--rewards detailed`). `criterion_<n>` and `check_<n>` are numbered within each scenario, so their means mean
something only across trials of the same task (`-k`), not across a dataset of different scenarios. It replaces Harbor's default mean in that job. Outside a checkout, download the single file
from the repository. It also runs as `python -m simulhausen_harbor.metric -i rewards.jsonl [-o metric.json]`.

## Caveats

- Sidecars need a Harbor environment provider that runs Docker Compose. The default `docker` provider does; check
  Harbor's multi-container docs before switching `-e`.
- A separate verifier environment (`[verifier] environment_mode = "separate"`) has no `/logs/agent` and no
  sidecars. Add `artifacts = ["/logs/agent/simulhausen"]` to `task.toml` (or pass `--artifact
  /logs/agent/simulhausen`) so the transcript reaches the judge. Option a needs the shared mode.
- `--rewards detailed` writes several keys per trial, which turns off Harbor's pass@k (it wants exactly one 0/1
  key).
- With `network_mode = "no-network"` or an allowlist, allow the model API host and the agent under test, and PyPI
  if the image has to install simulhausen.
- The instruction Harbor hands the agent (the scenario description) is not used: play reads the scenario file.
- simulhausen's response cache lives in the container, so Harbor runs are uncached.

## Development

```bash
cd integrations/harbor
uv sync
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

From the repository root, `just harbor-ci` runs the same. The tests never start a container or call a model; they
load the agents through Harbor's own factory and `harbor run --dry-run`.
