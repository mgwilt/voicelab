"""Application-owned audio and conversation interfaces; no Pipecat types."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict


class Notice(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    type: Literal["state", "text", "metrics", "error", "interrupted"]
    text: str = ""
    speaker: Literal["user", "assistant"] = "assistant"
    final: bool = False
    turn: int = 0
    stage: str = ""
    seconds: float | None = None
    output_rtf: float | None = None
    input_rtf: float | None = None


@dataclass(frozen=True)
class Audio:
    pcm: bytes
    sample_rate: int


type Emit = Callable[[Audio | Notice], None]
type Cancelled = Callable[[], bool]


class Backend(Protocol):
    input_rate: int
    frame_bytes: int
    duplex: bool

    def load(self, emit: Emit, cancelled: Cancelled) -> None: ...
    def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None: ...
    def played(self) -> None: ...
    def interrupt(self) -> None: ...
    def close(self) -> None: ...
