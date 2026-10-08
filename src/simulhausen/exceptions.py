"""Exceptions raised by the simulation harness."""

from __future__ import annotations

__all__ = [
    "SimulationBackendError",
    "SimulationHarnessError",
    "SimulationStructuredOutputError",
]


class SimulationHarnessError(RuntimeError):
    """The simulation harness broke, not the product under test.

    Reports mark such scenarios WARN instead of FAIL: what failed is the ability
    to check, not the check itself. Mixing the two makes the pass rate measure
    how reliable the helper LLMs (user simulator, judge) and the inference
    gateway are, instead of how good the agent is.
    """


class SimulationBackendError(SimulationHarnessError):
    """The model backend (LLM gateway, network) failed while a scenario ran.

    Separates an infrastructure failure (a 504 from a proxy, a dropped
    connection, an HTML error page instead of a model reply) from a real
    criterion or assertion failure. Without it a transient gateway timeout
    shows up as a failed tool-call assertion: the tool was never called because
    the model never answered, not because the agent misbehaved.
    """


class SimulationStructuredOutputError(SimulationHarnessError):
    """A helper agent (judge or user simulator) never returned schema-valid output.

    The product may have behaved fine; it is the model playing the user or the
    judge whose structured output failed.
    """
