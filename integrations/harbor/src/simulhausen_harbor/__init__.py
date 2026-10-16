"""simulhausen-harbor: Harbor agents that evaluate HTTP agents with simulhausen's user simulator.

Harbor (https://github.com/harbor-framework/harbor) runs one trial per task. With these agents the task is a
simulhausen scenario exported by ``simulhausen harbor export``, the system under test is an HTTP agent next to the
task container, and the Harbor agent plays the dialogue in the container::

    harbor run -p tasks -a simulhausen_harbor:AgentOSAgent \\
      --ak base_url=http://agent-api:7777 --ak agent_id=support -m support-agent@v1 \\
      --extra-docker-compose agentos-sidecar.yaml --ae OPENAI_API_KEY=$OPENAI_API_KEY -y

The task's verifier grades the transcript with simulhausen's judge. ``simulhausen_harbor.metric`` is a Harbor
``uv-script`` metric that reports simulhausen's pass rate (WARN excluded).

This distribution depends on ``harbor`` (and through it on litellm); simulhausen itself does not, and only its CLI
runs inside the container.
"""

from importlib.metadata import PackageNotFoundError, version

from simulhausen_harbor._version import DEFAULT_SIMULHAUSEN_SPEC, SIMULHAUSEN_VERSION
from simulhausen_harbor.agents import (
    A2AAgent,
    AgentOSAgent,
    LangGraphAgent,
    OpenAIChatAgent,
    SimulhausenAgent,
)

try:
    __version__ = version("simulhausen-harbor")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without metadata
    __version__ = "0+unknown"

__all__ = [
    "DEFAULT_SIMULHAUSEN_SPEC",
    "SIMULHAUSEN_VERSION",
    "A2AAgent",
    "AgentOSAgent",
    "LangGraphAgent",
    "OpenAIChatAgent",
    "SimulhausenAgent",
    "__version__",
]
