"""A private native process; no user-managed or remote inference service."""

import contextlib
import os
import secrets
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO

from voicelab_models.minicpm.model import MODEL_FILE, Request


class NativeProcess:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.temporary: TemporaryDirectory[str] | None = None
        self.log: IO[bytes] | None = None
        self.token = secrets.token_urlsafe(32)
        self.port = 0

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def start(
        self, binary: Path, model: Path, request: Request, cancelled: Callable[[], bool]
    ) -> None:
        if self.process is not None:
            raise RuntimeError("MiniCPM process is already started.")
        self.temporary = TemporaryDirectory(prefix="voicelab-minicpm-")
        directory = Path(self.temporary.name)
        self.log = (directory / "native.log").open("wb")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        env = os.environ.copy()
        env.update(VOICELAB_NATIVE_TOKEN=self.token, TMPDIR=str(directory))
        command = [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--model",
            str(model / MODEL_FILE),
            "--ctx-size",
            "8192",
            "-ngl",
            "99",
            "--seed",
            str(request.seed),
            "--temp",
            str(request.temperature),
            "--top-p",
            str(request.top_p),
            "--top-k",
            str(request.top_k),
            "--repeat-penalty",
            "1.05",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=binary.parents[2],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 60
        try:
            while time.monotonic() < deadline:
                if cancelled():
                    raise InterruptedError("MiniCPM startup cancelled.")
                if self.process.poll() is not None:
                    raise RuntimeError("MiniCPM native process exited during startup.")
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{self.port}/health", headers=self.headers
                    )
                    with urllib.request.urlopen(req, timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError, urllib.error.URLError:
                    time.sleep(0.1)
            raise TimeoutError(
                "MiniCPM native process did not start within 60 seconds."
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None
