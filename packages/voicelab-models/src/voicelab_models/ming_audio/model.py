"""Pinned Ming Audio identity and bounded audio-chat input."""

import base64
import binascii
import io
import wave
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from voicelab_models.audio import ReferenceAudio

SOURCE_MODEL = "inclusionAI/Ming-3.0-audio-tiny"
REVISION = "401b1abc63de0862dfbc9cd2a40895019d742d09"
CHECKPOINT_ID = "mlx-community/Ming-3.0-audio-tiny-4bit"
CHECKPOINT_REVISION = "af0238886184f15fb81e28858df2a3bf43983145"
SOURCE_REVISION = "4af0410415c7b77dc1437f709a2ad1c9c8f1a19d"
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 44100
MAX_INPUT_SECONDS = 30
MAX_OUTPUT_SECONDS = 120


@dataclass(frozen=True)
class WaveAudio:
    pcm: bytes
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.pcm) / (2 * self.sample_rate)


def decode_wave(data: bytes, sample_rate: int, max_seconds: int) -> WaveAudio:
    """Reject malformed, truncated, empty, or oversized PCM16 WAV data."""
    if len(data) > sample_rate * max_seconds * 2 + 65536:
        raise ValueError("WAV exceeds the audio size limit.")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
                1,
                2,
                sample_rate,
            ):
                raise ValueError(f"Expected mono PCM16 WAV at {sample_rate} Hz.")
            frames = audio.getnframes()
            if not 0 < frames <= sample_rate * max_seconds:
                raise ValueError("WAV duration is outside the allowed range.")
            pcm = audio.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("WAV audio is truncated.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM WAV audio.") from exc
    return WaveAudio(pcm, sample_rate)


def decode_input(encoded: str) -> WaveAudio:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Audio must be valid base64 WAV data.") from exc
    return decode_wave(raw, INPUT_SAMPLE_RATE, MAX_INPUT_SECONDS)


Task = Literal["spoken_dialogue", "audio_understanding", "asr", "text", "tts"]


class HistoryTurn(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=4000)
    audio: ReferenceAudio | None = None

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if not self.content.strip() and self.audio is None:
            raise ValueError("A history turn needs text or audio.")
        return self


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    task: Task = "spoken_dialogue"
    audio_wav_base64: str | None = Field(default=None, min_length=1, max_length=1400000)
    system_prompt: str | None = Field(default=None, min_length=1, max_length=2000)
    history: tuple[HistoryTurn, ...] = Field(default=(), max_length=20)
    thinking: bool = False
    reference_audio: ReferenceAudio | None = None
    trim_reference_silence: bool = True
    reference_text: str | None = Field(default=None, min_length=1, max_length=2000)
    prompt: str = Field(
        default="Please answer the question in the audio.",
        min_length=1,
        max_length=2000,
    )
    voice: str = Field(default="US_Stella", min_length=1, max_length=80)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_decode_steps: int = Field(
        default=375,
        ge=1,
        le=1500,
        description="Total acoustic-step budget across all speech fragments.",
    )
    cfg_scale: float = Field(default=2.0, ge=0, le=10, allow_inf_nan=False)
    sigma: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    flow_steps: int = Field(default=10, ge=1, le=100)

    @field_validator("audio_wav_base64")
    @classmethod
    def validate_audio(cls, value: str | None) -> str | None:
        if value is not None:
            decode_input(value)
        return value

    @model_validator(mode="after")
    def validate_workflow(self) -> Self:
        if (
            self.task in ("asr", "audio_understanding")
            and self.audio_wav_base64 is None
        ):
            raise ValueError(f"{self.task} requires input audio.")
        if self.task == "tts" and (
            self.audio_wav_base64 is not None
            or self.history
            or self.system_prompt is not None
            or self.thinking
        ):
            raise ValueError(
                "TTS accepts text and voice settings, without conversation input."
            )
        if self.reference_text is not None and self.reference_audio is None:
            raise ValueError("A reference transcript requires reference audio.")
        if self.reference_audio is not None:
            if self.task not in ("spoken_dialogue", "tts"):
                raise ValueError("Reference voices require a speech-output task.")
            if self.reference_audio.duration_seconds < 0.4:
                raise ValueError("Reference audio must contain at least 0.4 seconds.")
        if sum(len(turn.content) for turn in self.history) > 20000:
            raise ValueError("Conversation history exceeds 20000 characters.")
        audio_size = sum(
            len(turn.audio.wav_base64)
            for turn in self.history
            if turn.audio is not None
        )
        audio_size += len(self.audio_wav_base64 or "")
        audio_size += (
            len(self.reference_audio.wav_base64)
            if self.reference_audio is not None
            else 0
        )
        if audio_size > 6 * 1024 * 1024 - 65536:
            raise ValueError("Combined conversation audio exceeds the request limit.")
        for index, turn in enumerate(self.history):
            expected = "user" if index % 2 == 0 else "assistant"
            if turn.role != expected:
                raise ValueError("History must alternate user and assistant turns.")
        if self.history and self.history[-1].role != "assistant":
            raise ValueError("History must finish with the previous assistant answer.")
        return self
