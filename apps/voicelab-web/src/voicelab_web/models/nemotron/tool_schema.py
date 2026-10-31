"""Bounded upload boundary for the offline supplied-response experiment."""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from voicelab_models.audio import ReferenceAudio
from voicelab_models.nemotron.tools import ToolSettings


class ToolStart(ToolSettings):
    type: Literal["start"]
    audio: ReferenceAudio
    prompt: str = Field(default="Use the supplied tools when needed.", max_length=2000)
    seed: int = Field(default=0, ge=0, le=2147483647)
    trailing_silence: float = Field(default=16.0, ge=0, le=30, allow_inf_nan=False)

    @field_validator("prompt")
    @classmethod
    def ascii_prompt(cls, prompt: str) -> str:
        if not prompt.isascii():
            raise ValueError("Nemotron system prompts must be ASCII.")
        return prompt

    @model_validator(mode="after")
    def input_format(self) -> Self:
        if self.audio.sample_rate != 16000:
            raise ValueError("Tool input must be mono PCM16 WAV at 16000 Hz.")
        return self

    def settings(self) -> ToolSettings:
        return ToolSettings(tools=self.tools, api_response=self.api_response)
