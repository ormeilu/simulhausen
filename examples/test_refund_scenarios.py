"""Example scenarios: run with `OPENAI_API_KEY=... uv run --with openai pytest examples`."""

from __future__ import annotations

import pytest

import simulhausen as sim


def _agent() -> sim.AgentAdapter:
    from refund_agent import build_refund_agent

    return sim.agno_adapter(build_refund_agent)


def _looked_up_before_refund(state: sim.ScenarioState) -> None:
    assert state.has_tool_calls_in_order(["lookup_order", "issue_refund"]), state.tool_call_names()


@pytest.mark.simulation
async def test_duplicated_charge_is_refunded() -> None:
    result = await sim.run(
        name="refund: duplicated charge on order 1042",
        description="You were charged twice for order 1042 and want the second charge back. You know the order number.",
        agents=[
            _agent(),
            sim.UserSimulatorAgent(persona="Annoyed, types fast, no greetings."),
            sim.JudgeAgent(
                criteria=[
                    "The agent refunds exactly one charge of 49.00",
                    "The agent does not refund anything that was not charged twice",
                ]
            ),
        ],
        script=[
            sim.user("you charged me twice for 1042, fix it"),
            sim.agent(),
            sim.proceed(turns=2),
            _looked_up_before_refund,
            sim.judge(),
        ],
    )
    assert result.success, result.failure_summary()


@pytest.mark.simulation
async def test_missing_order_number_unscripted() -> None:
    result = await sim.run(
        name="refund: user forgets the order number",
        description=(
            "You want a refund for a double charge but you do not mention the order number until asked. "
            "Your order is 1042."
        ),
        agents=[
            _agent(),
            sim.UserSimulatorAgent(),
            sim.JudgeAgent(
                criteria=[
                    "The agent asks for the order number before promising a refund",
                    "The agent must not issue a refund for an order the user did not name",
                ]
            ),
        ],
        min_turns=2,
    )
    assert result.success, result.failure_summary()
