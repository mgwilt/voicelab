"""Validated decoded chunks at the audio8 vendor boundary."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from voicelab_models.audio8.model import Request

Audio = NDArray[np.float32]


@dataclass(frozen=True)
class Chunk:
    audio: Audio
    frames: int
    eos: bool | None
    codec_seconds: float = 0.0
    completion: Literal["complete", "incomplete", "unverified"] | None = None


class Backend(Protocol):
    def generate(self, request: Request) -> Generator[Chunk]: ...
    def close(self) -> None: ...
