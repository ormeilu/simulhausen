# AgentOS demo: a support agent evaluated by Harbor

The agent under test is a small agno support agent served with AgentOS (`app.py`): it looks orders up and refunds
duplicated charges. It runs as a sidecar named `agent-api` next to each task container. Three scenarios in
`scenarios.toml` check that it looks orders up before refunding, asks for a missing order number, and refuses to
refund a single charge.

| File | What it is |
|---|---|
| `app.py`, `Dockerfile` | The AgentOS app (port 7777, `GET /health`, `POST /agents/support/runs`) and its image |
| `agentos-sidecar.yaml` | Compose overlay for `--extra-docker-compose`: service `agent-api`, healthcheck, `OPENAI_API_KEY` |
| `scenarios.toml` | Scenarios in simulhausen's scenario-file format, one Harbor task each |
| `job.yaml` | The agent-mode run as a Harbor job config, with the pass-rate metric |

## Prerequisites

- Docker (Harbor's default `docker` provider runs the sidecar), uv, and an OpenAI API key. The simulator, the
  judge and the demo agent all use `gpt-4.1-mini`.
- Harbor with simulhausen's Harbor agents: `uv tool install harbor --with 'simulhausen[harbor]'`.
- simulhausen on the host for the export: `uv tool install simulhausen`.

Both need simulhausen 0.3.0 or later. Until it is out, install from a source archive instead:
`uv tool install harbor --with "simulhausen[harbor] @ https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz"`
and `uv tool install "simulhausen @ https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz"`.

Run every command from the repository root.

## 1. Build the agent under test

```bash
docker build -t simulhausen-agentos-demo examples/harbor/agentos
```

The overlay uses this image instead of a `build:` section, because relative build paths in an extra compose file
resolve against the task's `environment/` directory, not against the overlay.

## 2. Export the scenarios as Harbor tasks

```bash
simulhausen harbor export examples/harbor/agentos/scenarios.toml --output-dir tasks
```

One directory per scenario: `tasks/refund-duplicated-charge/`, `tasks/refund-missing-order-number/`,
`tasks/single-charge-not-refunded/`. With simulhausen installed from source, add
`--simulhausen-spec "simulhausen @ https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz"` so the task images install
the same version.

## 3. Run it (agent mode)

```bash
export OPENAI_API_KEY=sk-...
harbor run -p tasks \
  -a simulhausen.harbor:AgentOSAgent --ak base_url=http://agent-api:7777 --ak agent_id=support \
  -m support-agent@v1 \
  --extra-docker-compose examples/harbor/agentos/agentos-sidecar.yaml \
  --ae OPENAI_API_KEY=$OPENAI_API_KEY -y
```

Per trial, Harbor builds the task image, starts `agent-api` and waits for its healthcheck. The Harbor agent then
runs `simulhausen harbor play` in the task container, and the simulated user talks to `http://agent-api:7777`.
After that, `tests/test.sh` runs `simulhausen harbor judge` on the transcript.

The same run as a job config, plus the pass-rate metric:

```bash
harbor run -c examples/harbor/agentos/job.yaml -y
```

## 4. Or: everything in the verifier (verifier mode)

```bash
simulhausen harbor export examples/harbor/agentos/scenarios.toml --output-dir tasks-verifier --mode verifier
harbor run -p tasks-verifier -a nop \
  --extra-docker-compose examples/harbor/agentos/agentos-sidecar.yaml --job-name support-agent-v1 -y
```

`tests/test.sh` now runs `simulhausen harbor run`: play and judge in the verifier, against the target in
`scenarios.toml` (`[defaults] target`, the same `http://agent-api:7777`). The agent column says `nop`, hence the
job name.

## 5. Read the results

```
jobs/<job>/
├── result.json                                # per agent/model/dataset: trials, errors, metrics, rewards
└── refund-duplicated-charge__<id>/
    ├── result.json                            # agent_result.metadata (turns, tool_calls, stop_reason...), rewards
    ├── agent/simulhausen/transcript.json      # the dialogue, with the agent's tool calls
    ├── agent/simulhausen/play.log             # play's output
    └── verifier/
        ├── reward.json                        # {"reward": 1} PASS, {"reward": 0} FAIL
        └── simulhausen-result.json            # status, reason, per-criterion verdicts, checks
```

A trial without `reward.json` is a WARN: the harness broke (for example, the agent under test was unreachable)
or the judge found criteria inconclusive and none failed. `simulhausen-result.json` says which. Harbor lists these
trials as errors; the metric in `job.yaml` leaves them out of `pass_rate` and counts them in `n_warn`.

In agent mode, `harbor view jobs` shows each dialogue (play's trajectory) next to the rewards.

## 6. Evaluate your own agent

Build an image of your AgentOS app, point a copy of `agentos-sidecar.yaml` at it, and change the kwargs:
`--ak agent_id=...`, or `--ak team_id=...`, and `--ak api_key_env=AGENTOS_API_KEY --ae AGENTOS_API_KEY=...` if it
has a security key. Change `-m` with every version you evaluate, so Harbor keeps the results apart. For other
protocols, use `OpenAIChatAgent`, `A2AAgent` or `LangGraphAgent` (see the "Harbor" section of the
[README](../../../README.md#harbor)).
