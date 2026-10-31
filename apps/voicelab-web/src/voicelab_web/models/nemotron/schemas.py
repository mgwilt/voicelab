"""Nemotron control messages."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StartMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["start"]
    prompt: str = Field(
        default="Be concise and answer in one sentence.", max_length=2000
    )
    seed: int = Field(default=0, ge=0, le=2147483647)

    @field_validator("prompt")
    @classmethod
    def ascii_prompt(cls, value: str) -> str:
        if not value.isascii():
            raise ValueError("Use an ASCII prompt for this model.")
        return value


class StopMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["stop"]
