"""Pinned phonon identity and validated synthesis input."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.phonon.voices import VOICES

MODEL_ID = "gradium/phonon-31m-en"
MODEL_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f"
SAMPLE_RATE = 24000
CHECKPOINT_ID = MODEL_ID
CHECKPOINT_REVISION = MODEL_REVISION
CODEC_ID = "gradium/phonon-mimi"
CODEC_REVISION = "33519be1a8e61c6fa00c6e154e2d38e76a367919"


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_frames: int = Field(default=750, ge=1, le=750)
    voice: str = "default"
    temperature: float = Field(default=0.3, ge=0, le=2, allow_inf_nan=False)
    streaming: bool = True

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if not self.text.strip():
            raise ValueError("Content must include non-whitespace characters")
        if self.voice != "default" and self.voice not in VOICES:
            raise ValueError("Select a released English voice from the Phonon catalog")
        return self
