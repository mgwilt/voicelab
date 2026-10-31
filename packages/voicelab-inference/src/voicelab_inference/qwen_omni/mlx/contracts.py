"""Model-specific speech response and device boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from voicelab_models.qwen_omni.model import Request


@dataclass(frozen=True)
class Completion:
    text: str
    thinker_eos: bool
    talker_eos: bool | None
    thinker_tokens: int
    talker_tokens: int

    @property
    def completed(self) -> bool:
        return self.thinker_eos and self.talker_eos is not False


@dataclass(frozen=True)
class Metrics:
    input_seconds: float
    output_seconds: float
    generation_seconds: float
    rtf: float | None
    first_pcm_seconds: float | None
    completed: bool
    thinker_eos: bool
    talker_eos: bool | None
    thinker_tokens: int
    talker_tokens: int


@dataclass(frozen=True)
class Response:
    text: str
    metrics: Metrics


class Backend(Protocol):
    def generate(
        self,
        request: Request,
        emit_text: Callable[[str], None],
        emit_audio: Callable[[NDArray[np.float32]], None],
        should_cancel: Callable[[], bool],
    ) -> Completion: ...
    def close(self) -> None: ...
