"""Resident audio8 ownership and measured decoded-audio delivery."""

import platform
import threading
from collections.abc import Callable
from time import perf_counter

import numpy as np

from voicelab_core.synthesis import ChunkArrival, SynthesisMetrics
from voicelab_inference.audio8.mlx.contracts import Audio, Backend
from voicelab_models.audio8.model import SAMPLE_RATE, Request


class Audio8MLX:
    implementation = "audio8-mlx/vendor-adapter"
    precision = "float32"

    def __init__(self, *, factory: Callable[[], Backend] | None = None) -> None:
        self.device = "mlx"
        self.load_seconds = 0.0
        self._factory = factory
        self._backend: Backend | None = None
        self._lock = threading.Lock()

    def _load(self) -> Backend:
        if self._backend is None:
            start = perf_counter()
            if self._factory is not None:
                self._backend = self._factory()
            else:
                if platform.system() != "Darwin" or platform.machine() != "arm64":
                    raise RuntimeError("This runtime requires Apple Silicon macOS")
                from voicelab_inference.audio8.mlx.vendor import Vendor

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

    def synthesize(
        self, request: Request, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        with self._lock:
            backend = self._load()
            start = perf_counter()
            arrivals: list[ChunkArrival] = []
            samples = frames = 0
            codec_seconds = 0.0
            eos: bool | None = False
            completion = None
            chunks = backend.generate(request)
            try:
                for chunk in chunks:
                    audio = chunk.audio
                    if (
                        audio.ndim != 1
                        or audio.size == 0
                        or not np.isfinite(audio).all()
                    ):
                        raise ValueError(
                            "Model must produce finite nonempty mono audio"
                        )
                    elapsed = perf_counter() - start
                    arrivals.append(ChunkArrival(elapsed, int(audio.size)))
                    samples += int(audio.size)
                    frames += chunk.frames
                    codec_seconds += chunk.codec_seconds
                    eos = chunk.eos
                    completion = chunk.completion
                    emit(audio)
            finally:
                chunks.close()
            if samples == 0:
                raise RuntimeError("Model generated no audio")
            elapsed = perf_counter() - start
            duration = samples / SAMPLE_RATE
            return SynthesisMetrics(
                generation_seconds=elapsed,
                audio_seconds=duration,
                output_rtf=elapsed / duration,
                first_pcm_seconds=arrivals[0].seconds,
                codec_seconds=codec_seconds,
                frames=frames,
                eos=eos,
                arrivals=tuple(arrivals),
                completion=completion,
            )
