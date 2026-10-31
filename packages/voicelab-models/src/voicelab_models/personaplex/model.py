"""Pinned PersonaPlex identity and native audio timeline."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

SOURCE_MODEL = "nvidia/personaplex-7b-v1"
REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"
SOURCE_REVISION = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
MAX_FRAMES = 1500


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    prompt: str = Field(
        default="You are a helpful assistant. Answer clearly and concisely.",
        min_length=1,
        max_length=2000,
    )
    voice: str = Field(
        default="NATF2.pt", pattern=r"^(?:NAT[FM][0-3]|VAR[FM][0-4])\.pt$"
    )

    reference_audio: ReferenceAudio | None = None
    seed: int = Field(default=0, ge=0, le=4294967295)
    text_temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    audio_temperature: float = Field(default=0.8, ge=0, le=2, allow_inf_nan=False)
    text_top_k: int = Field(default=25, ge=1, le=32000)
    audio_top_k: int = Field(default=250, ge=1, le=2048)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if (
            self.reference_audio is not None
            and self.reference_audio.duration_seconds < 0.4
        ):
            raise ValueError("A voice reference must contain at least 0.4 seconds.")
        return self
