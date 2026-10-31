"""Nemotron's native timeline contract, independent of accelerator libraries."""

from dataclasses import dataclass
from typing import Literal, Protocol

FRAME_SAMPLES = 1280
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = 0.08


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal[
        "audio", "assistant_text_delta", "user_transcript_delta", "function_delta"
    ]
    pcm: bytes = b""
    text: str = ""
    frame_index: int | None = None


class DuplexSession(Protocol):
    def push(self, pcm: bytes) -> tuple[StreamEvent, ...]: ...
    def close(self) -> None: ...
