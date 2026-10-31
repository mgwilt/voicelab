"""Serialized local Ming Audio inference, measured after resident loading."""

import platform
import threading
from collections.abc import Callable
from time import perf_counter

import numpy as np

from voicelab_inference.ming_audio.mlx.contracts import Backend, Metrics, Response
from voicelab_models.ming_audio.model import (
    MAX_OUTPUT_SECONDS,
    OUTPUT_SAMPLE_RATE,
    Request,
    decode_input,
)


def never_cancel() -> bool:
    return False


class MingAudioMLX:
    implementation = "ming-audio-mlx/local-thinker-talker"
    device = "mlx"
    precision = "4-bit thinker; checkpoint talker and codec precision"

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
                    raise RuntimeError("Ming Audio MLX requires Apple Silicon macOS")
                from voicelab_inference.ming_audio.mlx.vendor import Vendor

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
        self, request: Request, should_cancel: Callable[[], bool] = never_cancel
    ) -> Response:
        input_seconds = (
            decode_input(request.audio_wav_base64).seconds
            if request.audio_wav_base64 is not None
            else 0.0
        )
        with self._lock:
            if should_cancel():
                raise InterruptedError("Audio conversation cancelled")
            backend = self._load()
            if should_cancel():
                raise InterruptedError("Audio conversation cancelled")
            start = perf_counter()
            output = backend.respond(request, should_cancel)
            if should_cancel():
                raise InterruptedError("Audio conversation cancelled")
            samples = output.audio
            speech_task = request.task in ("spoken_dialogue", "tts")
            if speech_task and output.completed and samples is None:
                raise ValueError("A completed speech task must return audio.")
            if not speech_task and samples is not None:
                raise ValueError("A text-only task must not return audio.")
            if samples is not None and (
                samples.ndim != 1
                or not 0 < samples.size <= OUTPUT_SAMPLE_RATE * MAX_OUTPUT_SECONDS
                or not np.isfinite(samples).all()
            ):
                raise ValueError("Ming Audio must return bounded finite mono speech")
            elapsed = perf_counter() - start
            duration = samples.size / OUTPUT_SAMPLE_RATE if samples is not None else 0.0
            pcm = (
                (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                if samples is not None
                else b""
            )
            return Response(
                text=output.text,
                pcm=pcm,
                reasoning=output.reasoning,
                language=output.language,
                metrics=Metrics(
                    input_seconds=input_seconds,
                    output_seconds=duration,
                    generation_seconds=elapsed,
                    rtf=elapsed / duration if duration else None,
                    first_pcm_seconds=elapsed if duration else None,
                    completed=output.completed,
                ),
            )
