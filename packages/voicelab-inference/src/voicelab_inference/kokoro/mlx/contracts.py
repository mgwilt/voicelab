"""Validated decoded chunks at the kokoro vendor boundary."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from voicelab_models.kokoro.model import Request

Audio = NDArray[np.float32]


@dataclass(frozen=True)
class Chunk:
    audio: Audio
    frames: int | None
    eos: bool | None
    codec_seconds: float | None = None
    completion: Literal["complete", "incomplete", "unverified"] | None = None


class Backend(Protocol):
    def prepare(self, request: Request) -> None: ...
    def generate(self, request: Request) -> Generator[Chunk]: ...
    def close(self) -> None: ...
