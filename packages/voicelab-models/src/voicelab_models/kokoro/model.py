"""Pinned kokoro identity and validated synthesis input."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.kokoro.voices import VOICES

MODEL_ID = "hexgrad/Kokoro-82M"
MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SAMPLE_RATE = 24000
CHECKPOINT_ID = "mlx-community/Kokoro-82M-bf16"
CHECKPOINT_REVISION = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    voice: str = Field(default="af_heart", min_length=1, max_length=500)
    lang_code: Literal["a", "b", "e", "f", "h", "i", "j", "p", "z"] | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if not self.text.strip():
            raise ValueError("Content must include non-whitespace characters")
        voices = self.voice.split(",")
        if any(voice not in VOICES for voice in voices):
            raise ValueError(
                "Select supported voice IDs, separated by commas for blending"
            )
        if len({voice[0] for voice in voices}) != 1:
            raise ValueError("Blended voices must belong to the same language")
        if self.lang_code is not None and self.lang_code != voices[0][0]:
            raise ValueError("Language must match the selected voice")
        return self
