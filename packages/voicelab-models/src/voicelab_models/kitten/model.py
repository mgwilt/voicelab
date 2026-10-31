"""Pinned Kitten checkpoint and validated synthesis input."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

MODEL_ID = "KittenML/kitten-tts-2"
MODEL_REVISION = "c93cb7c271baa51d9ae8204d5277988906f235a2"
CHECKPOINT_ID = MODEL_ID
CHECKPOINT_REVISION = MODEL_REVISION
SAMPLE_RATE = 24_000
CODEC_ID = "ResembleAI/chatterbox-turbo"
CODEC_REVISION = "749d1c1a46eb10492095d68fbcf55691ccf137cd"
CODEC_FILE = "s3gen_meanflow.safetensors"


class Request(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=4000)
    voice: str | None = Field(default=None, min_length=1, max_length=100)
    preset: Literal["stable", "expressive"] = "stable"
    max_tokens: int = Field(default=1000, ge=1, le=2000)
    ref_audio: ReferenceAudio | None = None
    ref_text: str | None = Field(default=None, min_length=1, max_length=4000)
    use_reference_prompt: bool | None = None
    temperature: float | None = Field(default=None, gt=0, le=2, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=0, le=1024)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    min_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.1, ge=1, le=3, allow_inf_nan=False)
    repetition_window: int | None = Field(default=None, ge=1, le=2000)
    token_run_penalty: float = Field(default=1.3, ge=1, le=5, allow_inf_nan=False)
    token_run_grace: int = Field(default=10, ge=1, le=100)
    chunk_chars: int | None = Field(default=None, ge=1, le=4000)
    chunk_min_chars: int | None = Field(default=None, ge=0, le=4000)
    use_emotion: bool | None = None
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    normalize: bool = False

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError(
                "Voice cloning requires reference audio and its transcript"
            )
        if self.ref_audio is not None and self.voice is not None:
            raise ValueError("Choose a named voice or reference audio, not both")
        if self.ref_audio is not None and self.ref_audio.duration_seconds < 0.1:
            raise ValueError("Kitten reference audio must be at least 0.1 seconds")
        if (
            self.chunk_chars is not None
            and self.chunk_min_chars is not None
            and self.chunk_min_chars > self.chunk_chars
        ):
            raise ValueError("Minimum chunk size must not exceed chunk size")
        values = (self.text, self.voice, self.ref_text)
        if any(value is not None and not value.strip() for value in values):
            raise ValueError("Text and controls must contain non-whitespace characters")
        return self
