"""Bounded reference audio shared by local and network request boundaries."""

import base64
import binascii
import io
import wave
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_REFERENCE_BASE64 = 4 * 1024 * 1024
MAX_REFERENCE_BYTES = MAX_REFERENCE_BASE64 * 3 // 4


class ReferenceAudio(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    wav_base64: str = Field(min_length=1, max_length=MAX_REFERENCE_BASE64)

    def wav_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.wav_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Reference audio must contain valid base64") from exc

    @model_validator(mode="after")
    def validate_wave(self) -> Self:
        try:
            with wave.open(io.BytesIO(self.wav_bytes()), "rb") as audio:
                rate = audio.getframerate()
                frames = audio.getnframes()
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                    raise ValueError("Reference audio must be mono PCM16 WAV")
                if not 8000 <= rate <= 48000:
                    raise ValueError("Reference sample rate must be 8000-48000 Hz")
                if not 0 < frames <= rate * 30:
                    raise ValueError("Reference audio must contain at most 30 seconds")
                if len(audio.readframes(frames)) != frames * 2:
                    raise ValueError("Reference WAV is truncated")
        except (wave.Error, EOFError) as exc:
            raise ValueError("Reference audio must be a valid PCM16 WAV") from exc
        return self

    @property
    def sample_rate(self) -> int:
        with wave.open(io.BytesIO(self.wav_bytes()), "rb") as audio:
            return audio.getframerate()

    @property
    def duration_seconds(self) -> float:
        with wave.open(io.BytesIO(self.wav_bytes()), "rb") as audio:
            return audio.getnframes() / audio.getframerate()

    @classmethod
    def from_path(cls, path: Path) -> Self:
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ValueError("Reference WAV exceeds the file size limit")
        return cls(wav_base64=base64.b64encode(path.read_bytes()).decode("ascii"))
