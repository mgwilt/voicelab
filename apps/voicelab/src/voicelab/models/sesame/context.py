"""Load an ordered local conversation manifest without exposing paths over HTTP."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from voicelab_models.audio import ReferenceAudio
from voicelab_models.sesame.model import ContextSegment


class ContextFile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    audio: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=8000)
    speaker: Literal[0, 1]


def load_context(path: Path) -> tuple[ContextSegment, ...]:
    if path.stat().st_size > 65536:
        raise ValueError("Context manifest is too large")
    files = TypeAdapter(list[ContextFile]).validate_json(path.read_text())
    if not 1 <= len(files) <= 4:
        raise ValueError("Provide one to four context turns")
    return tuple(
        ContextSegment(
            audio=ReferenceAudio.from_path(path.parent / turn.audio),
            text=turn.text,
            speaker=turn.speaker,
        )
        for turn in files
    )
