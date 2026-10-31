"""Pinned Maya checkpoint and validated synthesis input."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voicelab_models.audio import ReferenceAudio

MODEL_ID = "maya-research/maya2-native"
MODEL_REVISION = "76aedafc325fb72413569a0e15121d58ea2ac312"
CHECKPOINT_ID = "aruntemme/maya2-native-mlx-8bit"
CHECKPOINT_REVISION = "ee18411e5f4f69dcd526b66d2b18dbb8e2f21c8e"
SPEAKER_ID = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
SPEAKER_REVISION = "e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77"
SAMPLE_RATE = 24_000
CODEC_ID = "kyutai/moshiko-pytorch-bf16"
CODEC_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
CODEC_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"


class Request(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=4000)
    language: Literal[
        "as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"
    ] = "en"
    voice: (
        Literal[
            "Ahana",
            "Vaidehi",
            "Malini",
            "Charvi",
            "Jahnavi",
            "Anirban",
            "Nakul",
            "Vasant",
            "Siddhant",
            "Satvik",
            "Abhay",
            "Nivedita",
            "Ishwari",
            "Ritvik",
            "Vrinda",
            "Sreekar",
            "Kayal",
            "Shaunak",
            "Akshat",
            "Tanushree",
        ]
        | None
    ) = None
    ref_audio: ReferenceAudio | None = None
    ref_text: str | None = Field(default=None, min_length=1, max_length=4000)
    ref_language: (
        Literal["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
        | None
    ) = None
    conditioning: Literal["icl", "spk", "icl_spk"] = "icl_spk"
    temperature: float = Field(default=0.9, ge=0, le=2, allow_inf_nan=False)
    top_k: int = Field(default=64, ge=0, le=2048)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    depth_temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    depth_top_k: int = Field(default=64, ge=0, le=2048)
    presence_penalty: float = Field(default=0.8, ge=0, le=5, allow_inf_nan=False)
    presence_window: int = Field(default=12, ge=0, le=512)
    min_frames: int = Field(default=2, ge=0, le=1500)
    streaming_interval: float = Field(default=0.32, ge=0.08, le=5, allow_inf_nan=False)
    max_frames: int = Field(default=750, ge=1, le=1500)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    normalize: bool = True

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if self.ref_audio is not None:
            if self.voice is not None:
                raise ValueError("Choose a named voice or reference audio, not both")
            if self.conditioning != "spk" and self.ref_text is None:
                raise ValueError("In-context cloning requires the reference transcript")
            if self.ref_audio.duration_seconds < 0.1:
                raise ValueError("Maya reference audio must be at least 0.1 seconds")
        elif self.ref_text is not None or self.ref_language is not None:
            raise ValueError("Reference text and language require reference audio")
        values = (self.text, self.voice, self.ref_text)
        if any(value is not None and not value.strip() for value in values):
            raise ValueError("Text and controls must contain non-whitespace characters")
        return self
