"""Resident native Maya inference and measured PCM delivery."""

import platform
import threading
import time
from collections.abc import Callable, Generator
from contextlib import closing
from typing import Protocol

from voicelab_core.synthesis import ChunkArrival, SynthesisMetrics
from voicelab_inference.maya.mlx.contracts import Audio, Chunk
from voicelab_models.maya.model import SAMPLE_RATE, Request


class Backend(Protocol):
    def chunks(self, request: Request) -> Generator[Chunk]: ...
    def close(self) -> None: ...


def _load_model() -> Backend:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Maya MLX requires Apple Silicon macOS")
    from voicelab_inference.maya.mlx.vendor import Vendor

    return Vendor()


class MayaMLX:
    implementation = "maya-mlx/native-streaming"
    precision = "8-bit backbone; BF16 remaining weights; FP16 depth"
    device = "mlx"

    def __init__(self, *, loader: Callable[[], Backend] = _load_model) -> None:
        self._loader = loader
        self._model: Backend | None = None
        self._lock = threading.Lock()

    def _load(self) -> Backend:
        if self._model is None:
            self._model = self._loader()
        return self._model

    def load(self) -> None:
        with self._lock:
            self._load()

    def synthesize(
        self, request: Request, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        with self._lock:
            model = self._load()
            started = time.perf_counter()
            samples = 0
            frames = 0
            arrivals: list[ChunkArrival] = []
            final = False
            eos: bool | None = None
            with closing(model.chunks(request)) as stream:
                for chunk in stream:
                    if final:
                        raise ValueError("Maya emitted data after completion")
                    frames += chunk.frames
                    if chunk.audio.size:
                        samples += chunk.audio.size
                        if samples > SAMPLE_RATE * 120:
                            raise ValueError("Maya output exceeded 120 seconds")
                        arrivals.append(
                            ChunkArrival(
                                time.perf_counter() - started, chunk.audio.size
                            )
                        )
                        emit(chunk.audio)
                    final = chunk.final
                    if final:
                        eos = chunk.eos
            elapsed = time.perf_counter() - started
            if not final:
                raise RuntimeError("Maya ended without a terminal chunk")
            if not samples:
                raise RuntimeError("Maya returned no audio")
            duration = samples / SAMPLE_RATE
            return SynthesisMetrics(
                generation_seconds=elapsed,
                audio_seconds=duration,
                output_rtf=elapsed / duration,
                first_pcm_seconds=arrivals[0].seconds,
                codec_seconds=None,
                frames=frames,
                eos=eos,
                arrivals=tuple(arrivals),
            )

    def close(self) -> None:
        with self._lock:
            if self._model is not None:
                self._model.close()
                self._model = None
