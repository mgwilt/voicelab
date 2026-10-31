"""Runtime-independent contracts for offline speech experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GenerationSettings:
    prompt: str = "Be concise and answer in one sentence."
    seed: int = 0
    trailing_silence_seconds: float = 3.0

    def __post_init__(self) -> None:
        import math

        if (
            not math.isfinite(self.trailing_silence_seconds)
            or self.trailing_silence_seconds < 0
        ):
            raise ValueError("Trailing silence must be finite and non-negative.")


@dataclass(frozen=True)
class SpeechOutput:
    audio_path: Path
    text: str
    user_transcript: str | None
    sample_rate: int
    function_token_ids: tuple[int, ...] = ()


class OfflineSpeechBackend(Protocol):
    """A loaded backend; construction/loading is measured separately."""

    def generate(
        self, source: Path, destination: Path, settings: GenerationSettings
    ) -> SpeechOutput: ...

    def synchronize(self) -> None:
        """Wait for device work before recording wall-clock timing."""
        ...
