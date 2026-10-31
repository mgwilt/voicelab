"""Local Whisper transcription with pinned assets and explicit residency."""

import numpy as np
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict

from voicelab_models.whisper.model import CHECKPOINT_ID, REVISION


class Transcript(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    text: str


class WhisperMLX:
    def __init__(self) -> None:
        self.path: str | None = None

    def load(self) -> None:
        from mlx_whisper.transcribe import ModelHolder

        self.path = snapshot_download(CHECKPOINT_ID, revision=REVISION)
        import mlx.core as mx

        ModelHolder.get_model(self.path, mx.float16)

    def transcribe(self, pcm: bytes) -> str:
        from mlx_whisper import transcribe

        if self.path is None:
            raise RuntimeError("Whisper is not loaded")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        result: object = transcribe(audio, path_or_hf_repo=self.path)
        return Transcript.model_validate(result).text.strip()

    def close(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        self.path = None
        mx.clear_cache()
