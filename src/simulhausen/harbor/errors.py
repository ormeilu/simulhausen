"""Errors of the Harbor commands that the scenario author has to fix."""

from __future__ import annotations

__all__ = ["ConfigError"]


class ConfigError(ValueError):
    """The scenario file, the target, the model or the command line is wrong.

    The message names what is wrong and how to fix it. The CLI prints it and exits
    with code 1 without playing or grading anything: a broken configuration is not
    a verdict on the agent, so it must not turn into a reward.
    """
