"""Versions this distribution pins inside the task container.

The Harbor agents exec the ``simulhausen`` CLI in the container. When the image lacks it, they install
``DEFAULT_SIMULHAUSEN_SPEC`` with uv, so the default must name a simulhausen release that ships
``simulhausen harbor play``. Bump it together with the release that changes the transcript contract.

Until that release is on PyPI, point ``--ak simulhausen_spec=...`` at a source archive, for example
``simulhausen @ https://github.com/ormeilu/simulhausen/archive/refs/heads/main.tar.gz``
(a ``git+`` URL works too, but makes the agent install git first).
"""

SIMULHAUSEN_VERSION = "0.3.0"
"""simulhausen release installed when the container has no ``simulhausen`` CLI."""

DEFAULT_SIMULHAUSEN_SPEC = f"simulhausen=={SIMULHAUSEN_VERSION}"
"""pip requirement installed by default (``simulhausen_spec`` overrides it)."""

UV_VERSION = "0.12.21"
"""uv release fetched from astral.sh when the container has no ``uv`` either."""
