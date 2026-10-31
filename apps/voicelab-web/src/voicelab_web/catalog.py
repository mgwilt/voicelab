"""Explicit model capabilities presented by the lab, not a plugin registry."""

import platform
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from voicelab_web.schemas import ServerConfig

ModelId = Literal["nemotron", "sesame", "breeze"]


class ModelInfo(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    label: str
    description: str
    backend: str
    available: bool
    reason: str | None = None


class NemotronInfo(ModelInfo):
    id: Literal["nemotron"] = "nemotron"
    kind: Literal["duplex"] = "duplex"
    input_sample_rate: Literal[16000] = 16000
    output_sample_rate: Literal[22050] = 22050
    endpoint: Literal["/api/nemotron/session"] = "/api/nemotron/session"


class SesameInfo(ModelInfo):
    id: Literal["sesame"] = "sesame"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/sesame/session"] = "/api/sesame/session"


class BreezeInfo(ModelInfo):
    id: Literal["breeze"] = "breeze"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/breeze/session"] = "/api/breeze/session"


ModelInfoUnion = Annotated[
    NemotronInfo | SesameInfo | BreezeInfo, Field(discriminator="id")
]


def models(config: ServerConfig) -> tuple[NemotronInfo, SesameInfo, BreezeInfo]:
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    return (
        NemotronInfo(
            label="Nemotron VoiceChat 11B",
            description="Speak naturally. Listen and talk at the same time.",
            backend="MLX · 8-bit",
            available=apple,
            reason=None if apple else "This backend requires Apple Silicon and MLX.",
        ),
        SesameInfo(
            label="Sesame CSM-1B",
            description="Turn text into speech with contextual voice synthesis.",
            backend=f"PyTorch · {config.sesame_device.upper()}",
            available=True,
        ),
        BreezeInfo(
            label="Breeze TTS",
            description="Shape a voice with natural-language instructions.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This backend requires Apple Silicon and MLX.",
        ),
    )
