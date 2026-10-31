"""Pinned Drama checkpoint and validated synthesis input."""

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

MODEL_ID = "fishaudio/drama-3"
MODEL_REVISION = "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
CHECKPOINT_ID = "mlx-community/fish-audio-drama-3-4bit"
CHECKPOINT_REVISION = "1ac1b2108b63c2ef5bade463a121569363e293ce"
SAMPLE_RATE = 44_100


class Reference(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    audio: ReferenceAudio
    text: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def minimum_duration(self) -> Self:
        if self.audio.duration_seconds < 0.1:
            raise ValueError("Drama reference audio must be at least 0.1 seconds")
        return self


class PreviousTake(Reference):
    """An earlier utterance, supplied as audio and its exact transcript."""


class Request(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=4000)
    references: list[Reference] = Field(default_factory=list, max_length=4)
    previous_takes: list[PreviousTake] = Field(default_factory=list, max_length=2)
    top_k: int | None = Field(default=None, ge=1, le=1024)
    streaming_interval: float = Field(default=0.32, ge=0.05, le=5, allow_inf_nan=False)
    first_chunk_frames: int | None = Field(default=None, ge=1, le=256)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_new_tokens: int = Field(default=1024, ge=1, le=2048)
    temperature: float = Field(default=1.0, gt=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if len(self.references) + len(self.previous_takes) > 4:
            raise ValueError(
                "Use at most four recordings across references and prior takes"
            )
        values = (
            self.text,
            *(reference.text for reference in self.references),
            *(take.text for take in self.previous_takes),
        )
        if any(value is not None and not value.strip() for value in values):
            raise ValueError("Text and controls must contain non-whitespace characters")
        for token in (
            "<|im_start|>",
            "<|im_end|>",
            "<|voice|>",
            "<|text|>",
            "<|interleave|>",
            "<|semantic:",
        ):
            if any(token in value for value in values):
                raise ValueError("Text contains a reserved Drama structure token")
        for text in values:
            for match in re.finditer(r"<\|speaker", text):
                speaker = re.match(r"<\|speaker:([0-9]+)\|>", text[match.start() :])
                if speaker is None:
                    raise ValueError("Invalid Drama speaker tag")
                identifier = speaker.group(1)
                if identifier != str(int(identifier)) or int(identifier) > 15:
                    raise ValueError("Drama speaker IDs must be between 0 and 15")
        return self
