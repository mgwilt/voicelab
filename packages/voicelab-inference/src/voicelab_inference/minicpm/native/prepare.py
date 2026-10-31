"""Build pinned upstream code and download only the selected GGUF components."""

import contextlib
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from voicelab_models.minicpm.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MODEL_FILE,
    NATIVE_REVISION,
)

BUILD_OPTIONS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_CURL=OFF",
)

PATTERNS = [
    MODEL_FILE,
    "audio/*.gguf",
    "tts/*.gguf",
    "token2wav-gguf/*.gguf",
    "vision/*.gguf",
]


def cache_root() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "voicelab" / "minicpm"


def patch_hash() -> str:
    return hashlib.sha256(
        Path(__file__).with_name("voice-lab.patch").read_bytes()
    ).hexdigest()


def run_command(command: list[str], log: Path, cancelled: Callable[[], bool]) -> None:
    with log.open("ab") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if cancelled():
                    raise InterruptedError("MiniCPM preparation cancelled.")
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f"MiniCPM preparation failed; inspect {log}.")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
                # A child can ignore TERM after its parent exits. Reap the group.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def prepare(
    cancelled: Callable[[], bool], root: Path | None = None
) -> tuple[Path, Path]:
    import fcntl

    root = root or cache_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        while True:
            if cancelled():
                raise InterruptedError("MiniCPM preparation cancelled.")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        try:
            return _prepare(cancelled, root)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _prepare(cancelled: Callable[[], bool], root: Path) -> tuple[Path, Path]:
    source = root / f"{NATIVE_REVISION}-{patch_hash()[:12]}"
    binary = source / "build" / "bin" / "llama-omni-server"
    stamp = source / ".voicelab-build-complete"
    log = root / "prepare.log"
    signature = "\n".join((NATIVE_REVISION, patch_hash(), *BUILD_OPTIONS)) + "\n"
    if not binary.exists() or not stamp.exists() or stamp.read_text() != signature:
        if shutil.which("git") is None or shutil.which("cmake") is None:
            raise RuntimeError(
                "MiniCPM native inference needs Git, CMake, and a C++ compiler."
            )
        if not source.exists():
            run_command(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    "--filter=blob:none",
                    "https://github.com/OpenBMB/llama.cpp-omni.git",
                    str(source),
                ],
                log,
                cancelled,
            )
        run_command(
            ["git", "-C", str(source), "checkout", "--detach", NATIVE_REVISION],
            log,
            cancelled,
        )
        patch = Path(__file__).with_name("voice-lab.patch")
        # A failed build can be retried without reapplying an already applied patch.
        check = subprocess.run(
            ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
        )
        if check.returncode:
            run_command(
                ["git", "-C", str(source), "apply", "--check", str(patch)],
                log,
                cancelled,
            )
            run_command(["git", "-C", str(source), "apply", str(patch)], log, cancelled)
        run_command(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(source / "build"),
                *BUILD_OPTIONS,
            ],
            log,
            cancelled,
        )
        run_command(
            [
                "cmake",
                "--build",
                str(source / "build"),
                "--target",
                "llama-omni-server",
                "-j",
                "8",
            ],
            log,
            cancelled,
        )
        stamp.write_text(signature)
    if cancelled():
        raise InterruptedError("MiniCPM preparation cancelled.")
    try:
        checkpoint = snapshot_download(
            CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            allow_patterns=PATTERNS,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        run_command([sys.executable, "-m", __name__], log, cancelled)
        checkpoint = snapshot_download(
            CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            allow_patterns=PATTERNS,
            local_files_only=True,
        )
    model = Path(checkpoint)
    expected = [
        MODEL_FILE,
        "audio/MiniCPM-o-4_5-audio-F16.gguf",
        "tts/MiniCPM-o-4_5-tts-F16.gguf",
        "tts/MiniCPM-o-4_5-projector-F16.gguf",
        "vision/MiniCPM-o-4_5-vision-F16.gguf",
    ]
    expected += [
        f"token2wav-gguf/{name}.gguf"
        for name in (
            "encoder",
            "flow_extra",
            "flow_matching",
            "hifigan2",
            "prompt_cache",
        )
    ]
    if not all((model / name).is_file() for name in expected):
        run_command([sys.executable, "-m", __name__], log, cancelled)
    if not all((model / name).is_file() for name in expected):
        raise RuntimeError(
            "MiniCPM checkpoint download did not supply all required components."
        )
    return binary, model


if __name__ == "__main__":
    snapshot_download(
        CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
        allow_patterns=PATTERNS,
        max_workers=4,
    )
