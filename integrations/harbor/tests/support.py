"""Test doubles for Harbor: a recording environment that never starts a container.

``FakeEnvironment`` is a real ``BaseEnvironment`` subclass, so Harbor's own env merging (``--ae`` overlays) and
default-user handling apply; ``exec`` and ``upload_file`` only record calls and answer from a script.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

OK = ExecResult(stdout="", stderr="", return_code=0)


@dataclass
class ExecCall:
    command: str
    env: dict[str, str] | None
    user: str | int | None
    timeout_sec: int | None


@dataclass
class Upload:
    target_path: str
    content: str


class FakeEnvironment(BaseEnvironment):
    """Records every exec/upload. ``answer(command, result)`` scripts the result of one exact command."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[ExecCall] = []
        self.uploads: list[Upload] = []
        self._answers: list[tuple[Callable[[str], bool], ExecResult | Exception]] = []

    def answer(self, command: str, result: ExecResult | Exception, *, contains: bool = False) -> None:
        """Script a result (or an exception) for ``command``; ``contains`` matches a substring instead."""
        matcher: Callable[[str], bool] = (lambda c: command in c) if contains else (lambda c: c == command)
        self._answers.insert(0, (matcher, result))

    @property
    def commands(self) -> list[str]:
        return [call.command for call in self.calls]

    @staticmethod
    def type() -> str:
        return "fake"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self) -> None:
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool) -> None:
        pass

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads.append(Upload(target_path, _read_text(source_path)))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        raise NotImplementedError

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        raise NotImplementedError

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        raise NotImplementedError

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self.calls.append(ExecCall(command, self._merge_env(env), self._resolve_user(user), timeout_sec))
        for matches, result in self._answers:
            if matches(command):
                if isinstance(result, Exception):
                    raise result
                return result
        return OK


def _read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
