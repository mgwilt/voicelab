"""Local audio-chat responses and a narrow vendor boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from voicelab_models.ming_audio.model import Request


@dataclass(frozen=True)
class Metrics:
    input_seconds: float
    output_seconds: float
    generation_seconds: float
    rtf: float | None
    first_pcm_seconds: float | None
    completed: bool


@dataclass(frozen=True)
class Response:
    text: str
    pcm: bytes
    metrics: Metrics
    reasoning: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class Output:
    text: str
    audio: NDArray[np.float32] | None
    completed: bool
    reasoning: str | None = None
    language: str | None = None


class Backend(Protocol):
    def respond(
        self, request: Request, should_cancel: Callable[[], bool]
    ) -> Output: ...
    def close(self) -> None: ...
