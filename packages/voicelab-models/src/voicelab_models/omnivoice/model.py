"""Pinned Omnivoice checkpoint and validated synthesis input."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

MODEL_ID = "k2-fsa/OmniVoice"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
CHECKPOINT_ID = MODEL_ID
CHECKPOINT_REVISION = MODEL_REVISION
SAMPLE_RATE = 24_000


class Request(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(default="en", min_length=1, max_length=80)
    instruction: str | None = Field(default=None, min_length=1, max_length=1000)
    steps: int = Field(default=32, ge=1, le=100)
    speed: float = Field(default=1.0, ge=0.5, le=2.0, allow_inf_nan=False)
    ref_audio: ReferenceAudio | None = None
    ref_text: str | None = Field(default=None, min_length=1, max_length=4000)
    duration: float | None = Field(default=None, gt=0, le=120, allow_inf_nan=False)
    normalize: bool = False
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    guidance_scale: float = Field(default=2.0, ge=0, le=10, allow_inf_nan=False)
    t_shift: float = Field(default=0.1, gt=0, le=5, allow_inf_nan=False)
    layer_penalty_factor: float = Field(default=5.0, ge=0, le=20, allow_inf_nan=False)
    position_temperature: float = Field(default=5.0, ge=0, le=20, allow_inf_nan=False)
    class_temperature: float = Field(default=0.0, ge=0, le=5, allow_inf_nan=False)
    audio_chunk_duration: float = Field(default=15.0, ge=0, le=120, allow_inf_nan=False)
    audio_chunk_threshold: float = Field(
        default=30.0, ge=0, le=120, allow_inf_nan=False
    )
    pad_duration: float = Field(default=0.1, ge=0, le=2, allow_inf_nan=False)
    fade_duration: float = Field(default=0.1, ge=0, le=2, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError(
                "Voice cloning requires reference audio and its transcript"
            )
        if self.ref_audio is not None and self.instruction is not None:
            raise ValueError("Choose reference cloning or voice-design instructions")
        values = (self.text, self.language, self.instruction, self.ref_text)
        if any(value is not None and not value.strip() for value in values):
            raise ValueError("Text and controls must contain non-whitespace characters")
        return self
