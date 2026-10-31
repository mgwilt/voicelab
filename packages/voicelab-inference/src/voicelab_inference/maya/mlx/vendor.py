"""Pinned native Maya2 architecture, codec, and terminal-frame interpretation."""

from collections.abc import Generator
from contextlib import closing
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from mlx_audio.codec.models.ecapa_tdnn.speaker import EcapaSpeakerEmbeddingExtractor
from mlx_audio.codec.models.mimi import Mimi
from mlx_audio.codec.models.mimi.mimi import mimi_202407
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.maya2.maya2 import Model
from mlx_audio.tts.utils import load_model

from voicelab_inference.audio import reference_file
from voicelab_inference.maya.mlx.contracts import Chunk
from voicelab_inference.maya.mlx.sampling import seeded
from voicelab_models.maya.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    CODEC_FILE,
    CODEC_ID,
    CODEC_REVISION,
    SAMPLE_RATE,
    SPEAKER_ID,
    SPEAKER_REVISION,
    Request,
)


class Vendor:
    def __init__(self) -> None:
        path = snapshot_download(
            CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            allow_patterns=["*.json", "*.safetensors", "backbone/*", "voices/*"],
        )
        model = load_model(Path(path), strict=True)
        if not isinstance(model, Model) or model.sample_rate != SAMPLE_RATE:
            raise TypeError("Checkpoint must contain a native 24 kHz Maya2 model")
        codec_path = hf_hub_download(CODEC_ID, CODEC_FILE, revision=CODEC_REVISION)
        codec = Mimi(mimi_202407(32))
        codec.load_pytorch_weights(codec_path, strict=True)
        codec.eval()
        mx.eval(codec.parameters())
        model._codec = codec
        model._speaker_encoder = EcapaSpeakerEmbeddingExtractor.from_pretrained(
            SPEAKER_ID, revision=SPEAKER_REVISION
        )
        self._model: Model | None = model

    def chunks(self, request: Request) -> Generator[Chunk]:
        if self._model is None:
            raise RuntimeError("Maya has been closed")
        with seeded(request.seed), reference_file(request.ref_audio) as reference:
            frames = 0
            iterator = self._model.generate(
                request.text,
                voice=(request.voice or "Siddhant") if reference is None else None,
                ref_audio=reference,
                ref_text=request.ref_text,
                ref_lang=request.ref_language,
                conditioning=request.conditioning,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                depth_temperature=request.depth_temperature,
                depth_top_k=request.depth_top_k,
                presence_penalty=request.presence_penalty,
                presence_window=request.presence_window,
                min_tokens=request.min_frames,
                lang_code=request.language,
                max_tokens=request.max_frames,
                normalize=request.normalize,
                stream=True,
                streaming_interval=request.streaming_interval,
            )
            if not isinstance(iterator, Generator):
                raise TypeError("Maya did not return a closable generator")
            try:
                with closing(iterator):
                    for value in iterator:
                        if not isinstance(value, GenerationResult):
                            raise TypeError(
                                "Maya returned an incompatible generation result"
                            )
                        if (
                            value.sample_rate != SAMPLE_RATE
                            or not isinstance(value.audio, mx.array)
                            or value.audio.ndim != 1
                        ):
                            raise ValueError("Maya returned an incompatible waveform")
                        if value.token_count < frames:
                            raise ValueError("Maya frame count moved backwards")
                        new_frames = value.token_count - frames
                        frames = value.token_count
                        mx.eval(value.audio)
                        audio = np.asarray(value.audio, dtype=np.float32)
                        if not np.isfinite(audio).all():
                            raise ValueError("Maya returned non-finite PCM")
                        # The pinned loop has one early exit: its EOS probability guard.
                        # Natural iterator exhaustion at the frame cap is incomplete.
                        eos = (
                            frames < request.max_frames
                            if value.is_final_chunk
                            else None
                        )
                        yield Chunk(audio, new_frames, eos, value.is_final_chunk)
            finally:
                mx.clear_cache()

    def close(self) -> None:
        self._model = None
        mx.clear_cache()
