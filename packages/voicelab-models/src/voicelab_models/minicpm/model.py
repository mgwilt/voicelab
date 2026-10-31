"""Pinned MiniCPM-o 4.5 identity and bounded voice workflows."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from voicelab_models.audio import ReferenceAudio

SOURCE_MODEL = "openbmb/MiniCPM-o-4_5"
SOURCE_REVISION = "eb4141de24550a3a1a4082f718be58f46b761728"
CHECKPOINT_ID = "openbmb/MiniCPM-o-4_5-gguf"
CHECKPOINT_REVISION = "db25077c33951fe163b42986fba0132e279872a2"
NATIVE_REVISION = "a64932ec615e1463158129feaf2f0752d0c07a2e"
INPUT_RATE = 16000
OUTPUT_RATE = 24000
FRAME_BYTES = INPUT_RATE * 2
MAX_SECONDS = 120
MODEL_FILE = "MiniCPM-o-4_5-Q4_K_M.gguf"


class HistoryTurn(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=1200)


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    mode: Literal["full_duplex", "turn_based"] = "full_duplex"
    prompt: str = Field(
        default=(
            "You are a helpful voice assistant. "
            "Answer in English, briefly and naturally."
        ),
        min_length=1,
        max_length=2000,
    )
    reference_audio: ReferenceAudio | None = None
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=20, ge=1, le=200)
    tts_temperature: float = Field(default=0.8, gt=0, le=2, allow_inf_nan=False)
    length_penalty: float = Field(default=1.1, ge=0.1, le=2, allow_inf_nan=False)
    listen_prob_scale: float = Field(default=1.0, gt=0, le=3, allow_inf_nan=False)
    max_speak_tokens: int = Field(default=20, ge=1, le=100)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    text: str = Field(default="", max_length=1200)
    audio: ReferenceAudio | None = None
    history: tuple[HistoryTurn, ...] = Field(default=(), max_length=6)

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("System instructions must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "full_duplex" and (self.audio or self.text or self.history):
            raise ValueError("Live duplex uses microphone frames, not turn inputs.")
        if self.mode == "turn_based" and not self.text.strip() and self.audio is None:
            raise ValueError("A turn needs recorded audio or text.")
        for index, turn in enumerate(self.history):
            if turn.role != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("History must alternate user and assistant turns.")
        if len(self.history) % 2:
            raise ValueError("History must end with the previous assistant answer.")
        if self.reference_audio and self.reference_audio.duration_seconds < 1:
            raise ValueError("Reference voice must contain at least one second.")
        return self
