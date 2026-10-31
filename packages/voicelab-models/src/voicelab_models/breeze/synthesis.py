"""Instruction-conditioned synthesis request boundary."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=4000)
    instruction: str = Field(
        default="A clear, natural conversational voice.", min_length=1, max_length=2000
    )
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    cfg_scale: float = Field(default=4.0, gt=0, le=20, allow_inf_nan=False)
    max_frames: int = Field(default=750, ge=1, le=750)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if not self.text.strip() or not self.instruction.strip():
            raise ValueError(
                "Text and instruction must contain non-whitespace characters"
            )
        return self
