"""Where the Harbor commands read and write inside a task container.

Harbor gives every trial two log directories, ``/logs/agent`` for the agent phase and
``/logs/verifier`` for the verifier, and copies the task's ``tests/`` to ``/tests`` before
the verifier runs. These constants are the CLI defaults and what ``simulhausen harbor
export`` writes into the task files, so the generated tasks and the commands always agree.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "AGENT_LOGS_DIR",
    "AGENT_LOGS_SUBDIR",
    "DEFAULT_TRANSCRIPT",
    "IMAGE_SCENARIO_PATH",
    "RESULT_FILE",
    "REWARD_FILE",
    "TESTS_SCENARIO_PATH",
    "TRAJECTORY_FILE",
    "TRANSCRIPT_FILE",
    "TRANSCRIPT_MARKDOWN_FILE",
    "VERIFIER_LOGS_DIR",
]

# `play` writes here; Harbor copies /logs/agent to <trial>/agent on the host. The Harbor
# agents pass play the AGENT_LOGS_SUBDIR of the agent log directory Harbor gives them as --out.
AGENT_LOGS_SUBDIR: Final = "simulhausen"
AGENT_LOGS_DIR: Final = f"/logs/agent/{AGENT_LOGS_SUBDIR}"
# `judge` and `run` write here; Harbor reads reward.json from it.
VERIFIER_LOGS_DIR: Final = "/logs/verifier"

TRANSCRIPT_FILE: Final = "transcript.json"
# The dialogue in Harbor's ATIF trajectory format.
TRAJECTORY_FILE: Final = "trajectory.json"
TRANSCRIPT_MARKDOWN_FILE: Final = "transcript.md"
RESULT_FILE: Final = "simulhausen-result.json"
REWARD_FILE: Final = "reward.json"

DEFAULT_TRANSCRIPT: Final = f"{AGENT_LOGS_DIR}/{TRANSCRIPT_FILE}"

# The agent-phase scenario (no criteria) baked into the task image by `export`.
IMAGE_SCENARIO_PATH: Final = "/simulhausen/scenario.json"
# The full scenario, which reaches the container only with the verifier's tests/.
TESTS_SCENARIO_PATH: Final = "/tests/scenario.json"
