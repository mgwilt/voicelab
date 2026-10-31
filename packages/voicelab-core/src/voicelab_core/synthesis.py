"""Output-timeline measurements, independent of the inference framework."""

from dataclasses import dataclass
from typing import Literal

type Completion = Literal["complete", "incomplete", "unverified"]


@dataclass(frozen=True)
class ChunkArrival:
    seconds: float
    samples: int


@dataclass(frozen=True)
class SynthesisMetrics:
    generation_seconds: float
    audio_seconds: float
    output_rtf: float | None
    first_pcm_seconds: float | None
    codec_seconds: float | None
    frames: int | None
    eos: bool | None
    arrivals: tuple[ChunkArrival, ...]
    completion: Completion | None = None
    preparation_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.eos is False and self.completion == "complete":
            raise ValueError("Completion contradicts the observed stop signal")
        if self.eos is True and self.completion == "incomplete":
            raise ValueError("Completion contradicts the observed stop signal")

    @property
    def completion_state(self) -> Completion:
        """Separate an observed stop token from algorithm or vendor completion."""
        if self.completion is not None:
            return self.completion
        if self.eos is None:
            return "unverified"
        return "complete" if self.eos else "incomplete"
