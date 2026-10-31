"""Resident OmniVoice diffusion inference with output-timeline measurements."""

import threading
import time
from collections.abc import Callable
from importlib.util import find_spec
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from voicelab_core.synthesis import ChunkArrival, SynthesisMetrics
from voicelab_models.omnivoice.model import SAMPLE_RATE, Request

type Device = Literal["cpu", "mps", "cuda"]
type Audio = NDArray[np.float32]


class Backend(Protocol):
    def generate(self, request: Request) -> Audio: ...
    def close(self) -> None: ...


def _load_model(device: Device) -> Backend:
    from voicelab_inference.omnivoice.torch.vendor import OmniVoiceModel

    return OmniVoiceModel(device)


class OmniVoiceTorch:
    implementation = "omnivoice-0.2.1/torch-buffered"

    def __init__(
        self,
        device: Device = "cpu",
        *,
        loader: Callable[[Device], Backend] = _load_model,
    ) -> None:
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("OmniVoice device must be cpu, mps, or cuda")
        self.device = device
        self.precision = "float16" if device == "cuda" else "float32"
        self._loader = loader
        self._model: Backend | None = None
        self._lock = threading.Lock()

    @staticmethod
    def validate_dependencies(request: Request) -> None:
        if request.normalize:
            # WeText is required for the native English/Chinese normalization path.
            language = (
                request.language.casefold()
                if request.language is not None
                else (
                    "zh"
                    if any(
                        "\u4e00" <= character <= "\u9fff" for character in request.text
                    )
                    else "en"
                )
            )
            module = (
                "tn"
                if language
                in {"en", "eng", "english", "zh", "zh-cn", "chinese", "mandarin"}
                else "num2words"
            )
            if find_spec(module) is None:
                raise RuntimeError(
                    f"OmniVoice normalization requires optional {module}; "
                    "disable normalization or install its upstream dependencies"
                )

    def load(self) -> None:
        with self._lock:
            self._load()

    def _load(self) -> Backend:
        if self._model is None:
            self._model = self._loader(self.device)
        return self._model

    def synthesize(
        self, request: Request, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        self.validate_dependencies(request)
        with self._lock:
            model = self._load()
            started = time.perf_counter()
            audio = model.generate(request)
            ready = time.perf_counter() - started
            # OmniVoice returns a full waveform. Do not simulate streaming chunks.
            emit(audio)
            duration = audio.size / SAMPLE_RATE
            if duration <= 0:
                raise ValueError("OmniVoice returned no audio")
            return SynthesisMetrics(
                generation_seconds=ready,
                audio_seconds=duration,
                output_rtf=ready / duration,
                first_pcm_seconds=ready,
                codec_seconds=None,
                frames=None,
                eos=None,
                arrivals=(ChunkArrival(ready, audio.size),),
                completion="complete",
            )

    def close(self) -> None:
        with self._lock:
            if self._model is not None:
                self._model.close()
                self._model = None
