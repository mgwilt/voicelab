"""Load ordered earlier utterances for continuation-conditioned editing."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from voicelab_models.audio import ReferenceAudio
from voicelab_models.drama.model import PreviousTake


class TakeFile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    audio: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4000)


def load_previous_takes(path: Path) -> list[PreviousTake]:
    if path.stat().st_size > 65536:
        raise ValueError("Prior-take manifest exceeds 64 KiB")
    files = TypeAdapter(list[TakeFile]).validate_json(path.read_text())
    if not 1 <= len(files) <= 2:
        raise ValueError("Provide one or two prior takes")
    return [
        PreviousTake(
            audio=ReferenceAudio.from_path(path.parent / item.audio), text=item.text
        )
        for item in files
    ]
