"""Output-timeline measurements, independent of the inference framework."""

from dataclasses import dataclass


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
    codec_seconds: float
    frames: int
    eos: bool
    arrivals: tuple[ChunkArrival, ...]
