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

from simulhausen._cache import cache, clear_cache
from simulhausen.adapters import agno_adapter, function_adapter
from simulhausen.agents import JudgeAgent, JudgeVerdict, UserSimulatorAgent, UserTurn
from simulhausen.config import SimulationConfig, configure, get_config
from simulhausen.exceptions import (
    SimulationBackendError,
    SimulationHarnessError,
    SimulationStructuredOutputError,
)
from simulhausen.http_adapters import (
    a2a_adapter,
    agentos_adapter,
    langgraph_adapter,
    openai_chat_adapter,
    openai_responses_adapter,
)
from simulhausen.red_team import CrescendoStrategy, GoatStrategy, RedTeamAgent, RedTeamStrategy
from simulhausen.reporting import collector
from simulhausen.runner import run, run_trials
from simulhausen.script import agent, fail, judge, message, proceed, succeed, user
from simulhausen.trajectory import ExpectedCall, TrajectoryMatch, expect_call, match_trajectory
from simulhausen.types import (
    AgentAdapter,
    AgentInput,
    AgentReturnTypes,
    AgentRole,
    AgentTurn,
    CriterionStatus,
    CriterionVerdict,
    JudgeCall,
    JudgeScore,
    Message,
    Role,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCall,
    ToolCalls,
    ToolCallStatus,
    TrialsResult,
    Usage,
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
    "CrescendoStrategy",
    "CriterionStatus",
    "CriterionVerdict",
    "ExpectedCall",
    "GoatStrategy",
    "JudgeAgent",
    "JudgeCall",
    "JudgeScore",
    "JudgeVerdict",
    "Message",
    "RedTeamAgent",
    "RedTeamStrategy",
    "Role",
    "ScenarioResult",
    "ScenarioState",
    "ScenarioStatus",
    "SimulationBackendError",
    "SimulationConfig",
    "SimulationHarnessError",
    "SimulationStructuredOutputError",
    "ToolCall",
    "ToolCallStatus",
    "ToolCalls",
    "TrajectoryMatch",
    "TrialsResult",
    "Usage",
    "UserSimulatorAgent",
    "UserTurn",
    "__version__",
    "a2a_adapter",
    "agent",
    "agentos_adapter",
    "agno_adapter",
    "cache",
    "clear_cache",
    "collector",
    "configure",
    "expect_call",
    "fail",
    "function_adapter",
    "get_config",
    "judge",
    "langgraph_adapter",
    "match_trajectory",
    "message",
    "openai_chat_adapter",
    "openai_responses_adapter",
    "proceed",
    "run",
    "run_trials",
    "succeed",
    "user",
]
