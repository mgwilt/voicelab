"""Prepare the workspace and rebuild browser assets only when sources change."""

import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class CommandRunner(Protocol):
    def run(self, command: Sequence[str], cwd: Path) -> None: ...


class SubprocessRunner:
    def run(self, command: Sequence[str], cwd: Path) -> None:
        environment = os.environ.copy()
        # The lightweight launcher has its own uv script environment. Setup must
        # target the repository workspace, not that bootstrap environment.
        environment.pop("VIRTUAL_ENV", None)
        subprocess.run(command, cwd=cwd, check=True, env=environment)


class Workspace:
    def __init__(self, root: Path, runner: CommandRunner | None = None) -> None:
        self.root = root
        self.runner = runner if runner is not None else SubprocessRunner()

    def prepare_python(self) -> None:
        self.runner.run(
            ["uv", "sync", "--all-packages", "--extra", "mlx", "--locked"], self.root
        )

    def prepare_frontend(self) -> None:
        if shutil.which("bun") is None:
            raise RuntimeError("Bun is required for the website: https://bun.sh")
        web = self.root / "apps/voicelab-web"
        self.runner.run(["bun", "install", "--frozen-lockfile"], web)
        digest = self._frontend_digest(web)
        stamp = self.root / ".local/frontend-build.sha256"
        assets = web / "src/voicelab_web/static/dist"
        built = all(
            (assets / name).is_file() for name in ("main.js", "capture-worklet.js")
        )
        if built and stamp.is_file() and stamp.read_text() == digest:
            return
        self.runner.run(["bun", "run", "check"], web)
        self.runner.run(["bun", "run", "build"], web)
        stamp.parent.mkdir(exist_ok=True)
        stamp.write_text(digest)

    @staticmethod
    def _frontend_digest(web: Path) -> str:
        paths = [
            web / name
            for name in ("package.json", "bun.lock", "tsconfig.json", "bunfig.toml")
        ]
        for directory in ("frontend", "types"):
            paths.extend(
                path for path in (web / directory).rglob("*") if path.is_file()
            )
        digest = hashlib.sha256()
        for path in sorted(paths):
            digest.update(str(path.relative_to(web)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def execute(self, executable: str, arguments: Sequence[str]) -> None:
        directory = "Scripts" if os.name == "nt" else "bin"
        command = self.root / ".venv" / directory / executable
        os.chdir(self.root)
        environment = os.environ.copy()
        environment["VIRTUAL_ENV"] = str(self.root / ".venv")
        os.execve(str(command), [str(command), *arguments], environment)
