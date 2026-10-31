"""Framework-independent CSM identity and request boundary."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

MODEL_ID = "sesame/csm-1b"
REVISION = "c92a71e1c419772e25be7dc14d952c2521a740ab"
SAMPLE_RATE = 24_000
CONTEXT_LENGTH = 2048
Device = Literal["cpu", "mps", "cuda"]
Engine = Literal["baseline", "direct-depth"]


class ContextSegment(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    audio: ReferenceAudio
    text: str = Field(min_length=1, max_length=8000)
    speaker: Literal[0, 1]

    @model_validator(mode="after")
    def validate_text(self) -> Self:
        if not self.text.strip():
            raise ValueError("Context transcript must not be blank")
        return self


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=8000)
    speaker: Literal[0, 1] = 0
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_frames: int = Field(default=250, ge=4, le=1500)
    ref_audio: ReferenceAudio | None = None
    ref_text: str | None = Field(default=None, min_length=1, max_length=8000)
    temperature: float = Field(default=0.9, gt=0, le=2, allow_inf_nan=False)
    top_k: int = Field(default=50, ge=0, le=2048)
    context: tuple[ContextSegment, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def validate_text(self) -> Self:
        if not self.text.strip():
            raise ValueError("Text must contain non-whitespace characters")
        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError("Reference audio and transcript must be supplied together")
        if self.ref_text is not None and not self.ref_text.strip():
            raise ValueError("Reference transcript must not be blank")
        if self.context and self.ref_audio is not None:
            raise ValueError(
                "Use either a single voice reference or conversation context"
            )
        if sum(segment.audio.duration_seconds for segment in self.context) > 30:
            raise ValueError("Combined context audio exceeds 30 seconds")
        return self
