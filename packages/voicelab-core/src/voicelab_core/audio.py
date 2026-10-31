"""Validate the common PCM WAV input used for cross-backend comparisons."""

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WaveInfo:
    frames: int
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def inspect_input(path: Path) -> WaveInfo:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
            1,
            2,
            16000,
        ):
            raise ValueError("Input must be mono PCM16 WAV at 16000 Hz.")
        frames = audio.getnframes()
        if frames == 0:
            raise ValueError("Input WAV is empty.")
        if len(audio.readframes(frames)) != frames * 2:
            raise ValueError("Input WAV is truncated.")
        return WaveInfo(frames=frames, sample_rate=16000)
