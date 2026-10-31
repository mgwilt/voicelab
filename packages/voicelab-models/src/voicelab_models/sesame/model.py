"""Framework-independent CSM identity and request boundary."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_ID = "sesame/csm-1b"
REVISION = "c92a71e1c419772e25be7dc14d952c2521a740ab"
SAMPLE_RATE = 24_000
CONTEXT_LENGTH = 2048
Device = Literal["cpu", "mps", "cuda"]
Engine = Literal["baseline", "direct-depth"]


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=8000)
    speaker: Literal[0, 1] = 0
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_frames: int = Field(default=250, ge=4, le=1500)

    @model_validator(mode="after")
    def validate_text(self) -> Self:
        if not self.text.strip():
            raise ValueError("Text must contain non-whitespace characters")
        return self
