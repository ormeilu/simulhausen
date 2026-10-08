"""simulhausen: local simulation testing for agno agents.

An LLM plays the user and holds a multi-turn dialogue with the agent under
test, then an LLM judge grades the outcome criterion by criterion (passed,
failed or inconclusive, each with its reasoning). No external services: the
helper agents run on whatever agno model you configure.

    import simulhausen as sim

    result = await sim.run(
        name="refund request",
        description="The user wants a refund for a duplicated charge.",
        agents=[
            sim.agno_adapter(lambda: support_agent),
            sim.UserSimulatorAgent(),
            sim.JudgeAgent(criteria=["The agent asks for the order number"]),
        ],
        script=[sim.user("I was charged twice"), sim.agent(), sim.proceed(turns=3), sim.judge()],
    )
    assert result.success, result.failure_summary()

The API follows LangWatch Scenario (Apache-2.0); see NOTICE.
"""

from importlib.metadata import PackageNotFoundError, version

from simulhausen.adapters import agno_adapter, function_adapter
from simulhausen.agents import JudgeAgent, JudgeVerdict, UserSimulatorAgent, UserTurn
from simulhausen.config import SimulationConfig, configure, get_config
from simulhausen.exceptions import (
    SimulationBackendError,
    SimulationHarnessError,
    SimulationStructuredOutputError,
)
from simulhausen.reporting import collector
from simulhausen.runner import run
from simulhausen.script import agent, fail, judge, message, proceed, succeed, user
from simulhausen.types import (
    AgentAdapter,
    AgentInput,
    AgentReturnTypes,
    AgentRole,
    AgentTurn,
    CriterionStatus,
    CriterionVerdict,
    JudgeScore,
    Message,
    Role,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
)

try:
    __version__ = version("simulhausen")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without metadata
    # Not an X.Y.Z literal on purpose: release-please rewrites those in this file.
    __version__ = "0+unknown"

__all__ = [
    "AgentAdapter",
    "AgentInput",
    "AgentReturnTypes",
    "AgentRole",
    "AgentTurn",
    "CriterionStatus",
    "CriterionVerdict",
    "JudgeAgent",
    "JudgeScore",
    "JudgeVerdict",
    "Message",
    "Role",
    "ScenarioResult",
    "ScenarioState",
    "ScenarioStatus",
    "SimulationBackendError",
    "SimulationConfig",
    "SimulationHarnessError",
    "SimulationStructuredOutputError",
    "UserSimulatorAgent",
    "UserTurn",
    "__version__",
    "agent",
    "agno_adapter",
    "collector",
    "configure",
    "fail",
    "function_adapter",
    "get_config",
    "judge",
    "message",
    "proceed",
    "run",
    "succeed",
    "user",
]
