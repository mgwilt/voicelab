"""Strict, bounded interpretation of native WebSocket output."""

import base64
import binascii
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class NativeMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    type: Literal[
        "session.created",
        "session.closed",
        "response.output.delta",
        "response.done",
        "response.flush.done",
    ]
    kind: Literal["text", "audio", "listen"] | None = None
    text: str = Field(default="", max_length=100000)
    audio: str | None = Field(default=None, max_length=16 * 1024 * 1024)
    reason: str = ""
    drained: bool | None = None


@dataclass(frozen=True)
class Event:
    kind: Literal["text", "audio", "listen"]
    text: str = ""
    pcm: bytes = b""


def decode_audio(encoded: str) -> bytes:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("MiniCPM returned invalid audio base64.") from exc
    if not raw or len(raw) % 4 or len(raw) > 24000 * 4 * 120:
        raise ValueError("MiniCPM returned invalid or excessive float32 PCM.")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("MiniCPM returned nonfinite speech.")
    return (np.clip(values, -1, 1) * 32767).astype("<i2").tobytes()


def encode_audio(pcm: bytes) -> str:
    if not pcm or len(pcm) % 2:
        raise ValueError("Expected nonempty mono PCM16.")
    values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    return base64.b64encode(values.astype("<f4").tobytes()).decode("ascii")
