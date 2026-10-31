"""Pinned audio8 identity and validated synthesis input."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

MODEL_ID = "Edge0/Audio8-TTS-Realistic-0.6b"
MODEL_REVISION = "1e857568bfd4f02931aca3988ee773a44d2d26d3"
SAMPLE_RATE = 44100
CHECKPOINT_ID = MODEL_ID
CHECKPOINT_REVISION = MODEL_REVISION


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    temperature: float = Field(default=0.7, gt=0, le=2, allow_inf_nan=False)
    max_frames: int = Field(default=512, ge=1, le=1024)
    ref_audio: ReferenceAudio | None = None
    ref_text: str | None = Field(default=None, min_length=1, max_length=2000)
    top_p: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=50, ge=0, le=4096)
    do_sample: bool = True

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if not self.text.strip():
            raise ValueError("Content must include non-whitespace characters")
        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError(
                "Reference audio and its exact transcript are required together"
            )
        if self.ref_text is not None and not self.ref_text.strip():
            raise ValueError("Reference transcript cannot be blank")
        return self
