"""Resident KittenTTS 2 with measured sentence-chunk delivery."""

import threading
import time
from collections.abc import Callable, Generator
from contextlib import closing
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from voicelab_core.synthesis import ChunkArrival, SynthesisMetrics
from voicelab_inference.kitten.torch.contracts import Chunk
from voicelab_models.kitten.model import SAMPLE_RATE, Request

type Device = Literal["cpu", "mps", "cuda"]
type Audio = NDArray[np.float32]


class Backend(Protocol):
    def chunks(self, request: Request) -> Generator[Chunk]: ...
    def close(self) -> None: ...


def _load_model(device: Device) -> Backend:
    from voicelab_inference.kitten.torch.vendor import KittenModel

    return KittenModel(device)


class KittenTorch:
    implementation = "kittenml-0.9.7/torch-sentence-stream"
    precision = "bfloat16 language model; float32 codec"

    def __init__(
        self,
        device: Device = "cpu",
        *,
        loader: Callable[[Device], Backend] = _load_model,
    ) -> None:
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Kitten device must be cpu, mps, or cuda")
        self.device = device
        self._loader = loader
        self._model: Backend | None = None
        self._lock = threading.Lock()

    def _load(self) -> Backend:
        if self._model is None:
            self._model = self._loader(self.device)
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
            eos = True
            arrivals: list[ChunkArrival] = []
            with closing(model.chunks(request)) as stream:
                for chunk in stream:
                    frames += chunk.frames
                    eos = eos and chunk.eos
                    audio = chunk.audio
                    if audio.size == 0:
                        continue
                    samples += audio.size
                    if samples > SAMPLE_RATE * 120:
                        raise ValueError("Kitten output exceeded 120 seconds")
                    arrivals.append(
                        ChunkArrival(time.perf_counter() - started, audio.size)
                    )
                    emit(audio)
            elapsed = time.perf_counter() - started
            if not samples:
                raise RuntimeError("Kitten returned no audio")
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
