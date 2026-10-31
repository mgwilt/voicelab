"""Instruction-conditioned synthesis request boundary."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=4000)
    instruction: str | None = Field(
        default="A clear, natural conversational voice.", min_length=1, max_length=2000
    )
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    cfg_scale: float = Field(default=4.0, gt=0, le=20, allow_inf_nan=False)
    max_frames: int = Field(default=750, ge=1, le=750)
    ref_audio: ReferenceAudio | None = None
    ref_text: str | None = Field(default=None, min_length=1, max_length=4000)
    temperature: float = Field(default=0.9, gt=0, le=2, allow_inf_nan=False)
    top_k: int = Field(default=50, ge=0, le=2048)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if not self.text.strip() or (
            self.instruction is not None and not self.instruction.strip()
        ):
            raise ValueError(
                "Text and instruction must contain non-whitespace characters"
            )
        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError("Reference audio and transcript must be supplied together")
        if self.ref_text is not None and not self.ref_text.strip():
            raise ValueError("Reference transcript must not be blank")
        return self
