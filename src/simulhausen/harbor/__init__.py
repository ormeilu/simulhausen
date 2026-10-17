"""Run simulhausen scenarios as Harbor tasks (https://github.com/harbor-framework/harbor).

Harbor evaluates an agent on tasks: a task is a directory with a container image and a
verifier script that writes ``/logs/verifier/reward.json``. This package is the part
that runs inside the task container, behind the ``simulhausen harbor`` commands:

* ``play`` (agent phase): the user simulator talks to the agent under test, reached
  over HTTP or in-process (see :func:`build_target`), and the dialogue is recorded in
  ``/logs/agent/simulhausen/transcript.json``;
* ``judge`` (verifier): the judge grades the transcript against the scenario's
  criteria, checkpoints and checks, and writes the reward;
* ``run``: both in one go, for tasks whose verifier does everything;
* ``export``: renders a scenario file into Harbor task directories.

Scenarios are data here, TOML or JSON (see :func:`load_scenarios`), since a task cannot
carry Python callables. Nothing in this package imports Harbor, which depends on
litellm: simulhausen stays agno, pydantic and rich. The Harbor agents that run ``play``
from the Harbor side live in the separate ``simulhausen-harbor`` distribution.
"""

from simulhausen.harbor.errors import ConfigError
from simulhausen.harbor.spec import ScenarioSpec, load_scenarios
from simulhausen.harbor.targets import build_target

__all__ = ["ConfigError", "ScenarioSpec", "build_target", "load_scenarios"]
