"""Own the resident model and measure speech at its actual delivery boundary."""

import platform
import threading
from collections.abc import Callable
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from voicelab_inference.qwen_omni.mlx.contracts import Backend, Metrics, Response
from voicelab_models.qwen_omni.model import (
    MAX_OUTPUT_SECONDS,
    OUTPUT_SAMPLE_RATE,
    Request,
)


def never_cancel() -> bool:
    return False


def discard_text(text: str) -> None:
    del text


def discard_audio(pcm: bytes) -> None:
    del pcm


class QwenOmniMLX:
    implementation = "qwen-omni-mlx/thinker-talker-codec"

    def __init__(self, *, factory: Callable[[], Backend] | None = None) -> None:
        self._factory = factory
        self._backend: Backend | None = None
        self._lock = threading.Lock()
        self.load_seconds = 0.0

    def _load(self) -> Backend:
        if self._backend is None:
            start = perf_counter()
            if self._factory is not None:
                self._backend = self._factory()
            else:
                if platform.system() != "Darwin" or platform.machine() != "arm64":
                    raise RuntimeError("Qwen Omni MLX requires Apple Silicon macOS")
                from voicelab_inference.qwen_omni.mlx.vendor import Vendor

                self._backend = Vendor()
            self.load_seconds = perf_counter() - start
        return self._backend

    def load(self) -> None:
        with self._lock:
            self._load()

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                try:
                    self._backend.close()
                finally:
                    self._backend = None

    def respond(
        self,
        request: Request,
        emit_text: Callable[[str], None] = discard_text,
        emit_audio: Callable[[bytes], None] = discard_audio,
        should_cancel: Callable[[], bool] = never_cancel,
    ) -> Response:
        with self._lock:
            if should_cancel():
                raise InterruptedError("Qwen Omni experiment cancelled")
            backend = self._load()
            if should_cancel():
                raise InterruptedError("Qwen Omni experiment cancelled")
            start = perf_counter()
            samples = 0
            first: float | None = None

            def deliver(audio: NDArray[np.float32]) -> None:
                nonlocal samples, first
                if should_cancel():
                    raise InterruptedError("Qwen Omni experiment cancelled")
                if (
                    not request.return_audio
                    or audio.ndim != 1
                    or not audio.size
                    or not np.isfinite(audio).all()
                ):
                    raise ValueError("Expected finite nonempty mono speech")
                samples += audio.size
                if samples > MAX_OUTPUT_SECONDS * OUTPUT_SAMPLE_RATE:
                    raise ValueError("Qwen speech exceeds 120 seconds")
                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                if first is None:
                    first = perf_counter() - start
                emit_audio(pcm)

            result = backend.generate(request, emit_text, deliver, should_cancel)
            if should_cancel():
                raise InterruptedError("Qwen Omni experiment cancelled")
            if request.return_audio and result.completed and not samples:
                raise ValueError("Completed speech response contains no audio")
            if request.return_audio != (result.talker_eos is not None):
                raise ValueError("Completion does not match requested output mode")
            duration = samples / OUTPUT_SAMPLE_RATE
            elapsed = perf_counter() - start
            return Response(
                result.text,
                Metrics(
                    request.audio.duration_seconds if request.audio else 0,
                    duration,
                    elapsed,
                    elapsed / duration if duration else None,
                    first,
                    result.completed,
                    result.thinker_eos,
                    result.talker_eos,
                    result.thinker_tokens,
                    result.talker_tokens,
                ),
            )
