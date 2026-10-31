"""Pinned Qwen3-Omni speech workflows and validated controls."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

SOURCE_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SOURCE_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
CHECKPOINT_ID = "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"
CHECKPOINT_REVISION = "93b3cbddd65ed4babff8f22fba491cdba7a21778"
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
MAX_OUTPUT_SECONDS = 120
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
Speaker = Literal["Ethan", "Chelsie", "Aiden"]


class HistoryTurn(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=4000)
    audio: ReferenceAudio | None = None

    @model_validator(mode="after")
    def validate_turn(self) -> Self:
        if not self.content.strip() and self.audio is None:
            raise ValueError("A history turn needs text or audio.")
        if self.role == "assistant" and self.audio is not None:
            raise ValueError(
                "Assistant history uses text; audio input belongs to user turns."
            )
        return self


class Request(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    audio: ReferenceAudio | None = None
    prompt: str = Field(default="Answer briefly.", max_length=4000)
    system_prompt: str = Field(default=SYSTEM_PROMPT, min_length=1, max_length=4000)
    history: tuple[HistoryTurn, ...] = Field(default=(), max_length=20)
    return_audio: bool = True
    speaker: Speaker = "Ethan"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=0, ge=0, le=200)
    talker_max_tokens: int = Field(default=1500, ge=1, le=1500)
    talker_temperature: float = Field(default=0.9, ge=0, le=2, allow_inf_nan=False)
    talker_top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    talker_top_k: int = Field(default=50, ge=0, le=200)
    talker_repetition_penalty: float = Field(
        default=1.05, ge=1, le=2, allow_inf_nan=False
    )
    stream_audio: bool = True
    chunk_frames: int = Field(default=25, ge=1, le=300)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.audio is None and not self.prompt.strip():
            raise ValueError("Supply audio or a text prompt.")
        if not self.system_prompt.strip():
            raise ValueError("System instructions must not be blank.")
        if len(self.history) % 2:
            raise ValueError("History must contain complete user/assistant pairs.")
        for index, turn in enumerate(self.history):
            if turn.role != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("History must alternate user and assistant turns.")
        if sum(len(turn.content) for turn in self.history) > 20000:
            raise ValueError("History exceeds 20000 characters.")
        recordings = [turn.audio for turn in self.history if turn.audio is not None]
        if self.audio is not None:
            recordings.append(self.audio)
        if sum(item.duration_seconds for item in recordings) > 60:
            raise ValueError("Combined input audio exceeds 60 seconds.")
        if sum(len(item.wav_base64) for item in recordings) > 12 * 1024 * 1024:
            raise ValueError("Combined input audio exceeds 12 MiB.")
        return self
