"""Pinned Ming architecture with explicit stop-head completion."""

from collections.abc import Generator
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.tts.models.bailingmm.bailingmm import (
    _add_period,
    _split_fragments,
    _trim_trailing_silence,
)
from mlx_audio.tts.models.dense.dense import Model
from mlx_audio.tts.utils import load_model

from voicelab_inference.audio import reference_pcm
from voicelab_inference.ming_tts.mlx.contracts import Chunk
from voicelab_inference.ming_tts.mlx.preparation import trim_reference_tail
from voicelab_inference.ming_tts.mlx.reference import encode_reference
from voicelab_inference.ming_tts.mlx.sampling import sample
from voicelab_inference.ming_tts.mlx.speaker import SpeakerEncoder
from voicelab_models.ming_tts.model import (
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
            raise ValueError("Checkpoint must contain a 44.1 kHz Ming dense model")
        self.model = model
        self.speaker = SpeakerEncoder()

    def generate(self, request: Request) -> Generator[Chunk]:
        prompt_latent = None
        speaker = None
        if request.ref_audio is not None:
            audio = reference_pcm(request.ref_audio, SAMPLE_RATE)
            if request.trim_reference_silence:
                audio = trim_reference_tail(audio, SAMPLE_RATE)
            prompt_latent = encode_reference(self.model, mx.array(audio), request.seed)
            speaker = mx.array(
                self.speaker.encode(reference_pcm(request.ref_audio, 16000))
            )
        fragments = (
            [_add_period(" ".join(request.text.split()))]
            if request.instruction is not None or request.style is not None
            else _split_fragments(_add_period(request.text), 24)
        )
        try:
            for segment, text in enumerate(fragments):
                frames: list[mx.array] = []
                eos = False
                sampler = sample(
                    self.model,
                    request,
                    text=text,
                    prompt_latent=prompt_latent,
                    speaker=speaker,
                    segment=segment,
                )
                try:
                    for latent, last in sampler:
                        mx.eval(latent)
                        frames.append(latent)
                        eos = last
                finally:
                    sampler.close()
                if not frames:
                    raise RuntimeError("Ming generated no audio latents")
                start = perf_counter()
                waveform, _, _ = self.model.audio.decode(
                    mx.concatenate(frames, axis=1), use_cache=False, last_chunk=True
                )
                if (
                    not isinstance(waveform, mx.array)
                    or waveform.ndim != 2
                    or waveform.shape[0] != 1
                ):
                    raise ValueError("Ming decoder must return mono [1, samples] audio")
                audio = _trim_trailing_silence(waveform[0], SAMPLE_RATE)
                mx.eval(audio)
                yield Chunk(
                    np.asarray(audio, dtype=np.float32),
                    frames=len(frames),
                    eos=eos,
                    codec_seconds=perf_counter() - start,
                )
                if not eos:
                    return
        finally:
            mx.clear_cache()

    def close(self) -> None:
        mx.clear_cache()
