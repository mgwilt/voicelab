"""Pinned native Drama architecture with per-segment cap accounting."""

from collections.abc import Generator
from contextlib import ExitStack, closing
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.codec.models.fish_s1_dac import DAC
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.fish_drama3.drama3 import AudioInput, Model
from mlx_audio.tts.utils import load_model

from voicelab_inference.audio import reference_file, reference_pcm
from voicelab_inference.drama.mlx.contracts import Chunk
from voicelab_inference.drama.mlx.generation import generate
from voicelab_inference.drama.mlx.sampling import seeded
from voicelab_models.drama.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    SAMPLE_RATE,
    Request,
)


class Vendor:
    def __init__(self) -> None:
        path = snapshot_download(
            CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            allow_patterns=["*.json", "*.safetensors"],
        )
        model = load_model(Path(path), strict=True)
        if not isinstance(model, Model) or model.sample_rate != SAMPLE_RATE:
            raise TypeError("Checkpoint must contain a native 44.1 kHz Drama model")
        if not isinstance(model.codec, DAC):
            raise TypeError("Drama checkpoint did not load its DAC codec")
        self._model: Model | None = model

    def chunks(self, request: Request) -> Generator[Chunk]:
        if self._model is None or not isinstance(self._model.codec, DAC):
            raise RuntimeError("Drama has been closed")
        codec = self._model.codec
        with seeded(request.seed), ExitStack() as resources:
            try:
                paths = [
                    resources.enter_context(reference_file(item.audio))
                    for item in request.references
                ]
                references: list[AudioInput] = [
                    path for path in paths if path is not None
                ]
                segments: dict[int, int] = {}
                previous = [
                    (
                        take.text,
                        self._model._encode_audio(
                            reference_pcm(take.audio, SAMPLE_RATE)
                        ),
                    )
                    for take in request.previous_takes
                ]
                iterator = generate(self._model, codec, request, references, previous)
                if not isinstance(iterator, Generator):
                    raise TypeError("Drama did not return a closable generator")
                with closing(iterator):
                    for value in iterator:
                        if not isinstance(value, GenerationResult):
                            raise TypeError(
                                "Drama returned an incompatible generation result"
                            )
                        if (
                            value.sample_rate != SAMPLE_RATE
                            or not isinstance(value.audio, mx.array)
                            or value.audio.ndim != 1
                        ):
                            raise ValueError("Drama returned an incompatible waveform")
                        if value.token_count < 0 or value.segment_idx < 0:
                            raise ValueError("Drama returned invalid frame metadata")
                        segments[value.segment_idx] = (
                            segments.get(value.segment_idx, 0) + value.token_count
                        )
                        mx.eval(value.audio)
                        audio = np.asarray(value.audio, dtype=np.float32)
                        if not np.isfinite(audio).all():
                            raise ValueError("Drama returned non-finite PCM")
                        # The pinned _frames loop returns early only for sampled EOS.
                        # At a cap it emits every frame, including a held tail.
                        eos = (
                            all(
                                frames < request.max_new_tokens
                                for frames in segments.values()
                            )
                            if value.is_final_chunk
                            else None
                        )
                        yield Chunk(audio, value.token_count, eos, value.is_final_chunk)
            finally:
                codec.reset_streaming_state()
                mx.clear_cache()

    def close(self) -> None:
        self._model = None
        mx.clear_cache()
