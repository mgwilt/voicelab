"""Pinned ming_tts identity and validated synthesis input."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio
from voicelab_models.ming_tts.voices import VOICES

MODEL_ID = "inclusionAI/Ming-3.0-tts-0.5B"
MODEL_REVISION = "7247f4c94a5b2e987404ed28a5cfec41b1a4ca61"
SAMPLE_RATE = 44100
CHECKPOINT_ID = "mlx-community/Ming-3.0-tts-0.5B-8bit"
CHECKPOINT_REVISION = "d0d1da6f627e4b3f20cc2db6ba4aee86d45343ad"


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    instruction: str | None = Field(default=None, min_length=1, max_length=500)
    language: Literal["zh", "en", "ja", "ko", "de", "fr", "es"] = "en"
    ref_audio: ReferenceAudio | None = None
    ref_text: str | None = Field(default=None, min_length=1, max_length=500)
    trim_reference_silence: bool = True
    voice: str | None = Field(default=None, min_length=1, max_length=100)
    style: str | None = Field(default=None, min_length=1, max_length=500)
    emotion: Literal["高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性"] | None = (
        None
    )
    dialect: Literal["cantonese", "chuanyu dialect"] | None = None
    speed: Literal["慢速", "中速", "快速"] | None = None
    pitch: Literal["低", "中", "高"] | None = None
    volume: Literal["低", "中", "高"] | None = None
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_frames: int = Field(default=120, ge=1, le=120)
    cfg_scale: float = Field(default=2.0, gt=0, le=10, allow_inf_nan=False)
    flow_steps: int = Field(default=10, ge=1, le=100)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    sigma: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        values = (self.text, self.instruction, self.ref_text, self.voice, self.style)
        if any(value is not None and not value.strip() for value in values):
            raise ValueError("Content must include non-whitespace characters")
        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError(
                "Reference audio and its exact transcript are required together"
            )
        controls = (
            self.voice,
            self.style,
            self.emotion,
            self.dialect,
            self.speed,
            self.pitch,
            self.volume,
        )
        if self.instruction is not None and any(
            value is not None for value in controls
        ):
            raise ValueError(
                "Freeform instruction cannot combine with structured controls"
            )
        if self.voice is not None and (
            self.ref_audio is not None or self.style is not None
        ):
            raise ValueError(
                "Select a built-in voice, voice design, or reference audio"
            )
        if self.style is not None and self.ref_audio is not None:
            raise ValueError("Voice design cannot combine with reference audio")
        if self.voice is not None and self.voice not in VOICES:
            raise ValueError("Unknown Ming built-in voice; see the voice catalog")
        return self
