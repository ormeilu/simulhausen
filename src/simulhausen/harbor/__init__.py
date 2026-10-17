"""Run simulhausen scenarios as Harbor tasks (https://github.com/harbor-framework/harbor).

Harbor evaluates an agent on tasks: a task is a directory with a container image and a
verifier script that writes ``/logs/verifier/reward.json``. Most of this package runs
inside the task container, behind the ``simulhausen harbor`` commands:

* ``play`` (agent phase): the user simulator talks to the agent under test, reached
  over HTTP or in-process (see :func:`build_target`), and the dialogue is recorded in
  ``/logs/agent/simulhausen/transcript.json``;
* ``judge`` (verifier): the judge grades the transcript against the scenario's
  criteria, checkpoints and checks, and writes the reward;
* ``run``: both in one go, for tasks whose verifier does everything;
* ``export``: renders a scenario file into Harbor task directories.

Scenarios are data here, TOML or JSON (see :func:`load_scenarios`), since a task cannot
carry Python callables. None of that imports Harbor, which depends on litellm.

The Harbor side is the exception: the agent classes that run ``play`` from a Harbor job
(``harbor run -a simulhausen.harbor:AgentOSAgent``) live in :mod:`simulhausen.harbor.agents`
and need the extra, ``pip install 'simulhausen[harbor]'``. They are loaded on first
access, so importing this package never imports Harbor. :mod:`simulhausen.harbor.metric`
is Harbor's pass-rate metric, a standalone script.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Final

from simulhausen.harbor.errors import ConfigError
from simulhausen.harbor.spec import ScenarioSpec, load_scenarios
from simulhausen.harbor.targets import build_target

if TYPE_CHECKING:
    # What __getattr__ below returns, for type checkers.
    from simulhausen.harbor.agents import A2AAgent as A2AAgent
    from simulhausen.harbor.agents import AgentOSAgent as AgentOSAgent
    from simulhausen.harbor.agents import LangGraphAgent as LangGraphAgent
    from simulhausen.harbor.agents import OpenAIChatAgent as OpenAIChatAgent
    from simulhausen.harbor.agents import SimulhausenAgent as SimulhausenAgent

# Not in __all__: a star import would load Harbor.
__all__ = ["ConfigError", "ScenarioSpec", "build_target", "load_scenarios"]

_AGENT_CLASSES: Final = frozenset({"A2AAgent", "AgentOSAgent", "LangGraphAgent", "OpenAIChatAgent", "SimulhausenAgent"})


def __getattr__(name: str) -> Any:
    """The Harbor agent classes, imported from :mod:`simulhausen.harbor.agents` on first access."""
    if name not in _AGENT_CLASSES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        agents = importlib.import_module("simulhausen.harbor.agents")
    except ImportError as exc:
        raise ImportError(
            f"simulhausen.harbor.{name} needs Harbor, which a plain install leaves out: "
            f"pip install 'simulhausen[harbor]' ({exc})"
        ) from exc
    return getattr(agents, name)
