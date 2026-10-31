"""Pinned Kokoro MLX-Audio boundary with local voice packs."""

from collections.abc import Generator
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.kokoro.kokoro import Model
from mlx_audio.tts.utils import load_model

from voicelab_inference.kokoro.mlx.contracts import Chunk
from voicelab_models.kokoro.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    SAMPLE_RATE,
    Request,
)


class Vendor:
    def __init__(self) -> None:
        path = Path(snapshot_download(CHECKPOINT_ID, revision=CHECKPOINT_REVISION))
        model = load_model(path, strict=True)
        if not isinstance(model, Model) or model.sample_rate != SAMPLE_RATE:
            raise ValueError("Checkpoint must contain a 24 kHz Kokoro model")
        self.model = model
        self.path = path

    def prepare(self, request: Request) -> None:
        for voice in request.voice.split(","):
            if not (self.path / "voices" / f"{voice}.safetensors").is_file():
                raise ValueError("Selected voice is missing from the pinned checkpoint")
        language = request.lang_code or request.voice[0]
        if language == "j":
            from voicelab_inference.kokoro.mlx.japanese import install_pipeline

            install_pipeline(self.model, self.path)
        else:
            self.model._get_pipeline(language)

    def generate(self, request: Request) -> Generator[Chunk]:
        language = request.lang_code or request.voice[0]
        chunks = self.model.generate(
            request.text, voice=request.voice, speed=request.speed, lang_code=language
        )
        try:
            for result in chunks:
                if (
                    not isinstance(result, GenerationResult)
                    or result.sample_rate != SAMPLE_RATE
                ):
                    raise ValueError(
                        "Kokoro returned an incompatible generation result"
                    )
                audio: object = result.audio
                if not isinstance(audio, mx.array) or audio.ndim != 1:
                    raise ValueError("Kokoro must return mono audio")
                mx.eval(audio)
                yield Chunk(
                    np.asarray(audio, dtype=np.float32),
                    frames=None,
                    eos=None,
                    completion="complete",
                )
        finally:
            chunks.close()
            mx.clear_cache()

    def close(self) -> None:
        self.model._pipelines.clear()
        mx.clear_cache()
