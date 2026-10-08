from __future__ import annotations

import os

import pytest

import simulhausen as sim


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        skip = pytest.mark.skip(reason="the examples call a real model: set OPENAI_API_KEY")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def configure_simulations() -> None:
    # Imported here so collecting the examples never needs the openai package.
    from agno.models.openai import OpenAIChat

    sim.configure(model=OpenAIChat(id="gpt-4.1-mini"), max_turns=6, agent_name="Refund desk")
