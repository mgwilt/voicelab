"""Codec lifecycle and synchronized output-timeline measurements."""

import time
from collections.abc import Callable, Iterator
from typing import Protocol

import mlx.core as mx
import numpy as np

from voicelab_core.synthesis import ChunkArrival, SynthesisMetrics
from voicelab_inference.breeze.mlx.audio import Audio, validate
from voicelab_models.breeze.model import SAMPLE_RATE
from voicelab_models.breeze.synthesis import SynthesisRequest


class StreamingModel(Protocol):
    def reset_codec(self) -> None: ...
    def decode(self, frames: list[mx.array]) -> mx.array: ...
    def frames(self, request: SynthesisRequest) -> Iterator[mx.array | None]: ...


def synthesize(
    model: StreamingModel, request: SynthesisRequest, emit: Callable[[Audio], None]
) -> SynthesisMetrics:
    """Exclude loading; include prefill, decoding, synchronization and emit time."""
    started = time.perf_counter()
    arrivals: list[ChunkArrival] = []
    samples = frames = 0
    codec_seconds = 0.0
    eos = False
    pending: list[mx.array] = []

    def flush() -> None:
        nonlocal samples, codec_seconds
        codec_started = time.perf_counter()
        audio = model.decode(pending)
        mx.eval(audio)
        pcm = np.array(audio, dtype=np.float32)
        validate(pcm)
        codec_seconds += time.perf_counter() - codec_started
        arrivals.append(ChunkArrival(time.perf_counter() - started, pcm.size))
        emit(pcm)
        samples += pcm.size
        pending.clear()

    model.reset_codec()
    try:
        for frame in model.frames(request):
            if frame is None:
                eos = True
                break
            frames += 1
            pending.append(frame)
            if len(pending) == 3:
                flush()
        if pending:
            flush()
        if not samples:
            raise RuntimeError("Breeze produced no audio frames")
    finally:
        model.reset_codec()
    elapsed = time.perf_counter() - started
    duration = samples / SAMPLE_RATE
    return SynthesisMetrics(
        generation_seconds=elapsed,
        audio_seconds=duration,
        output_rtf=elapsed / duration,
        first_pcm_seconds=arrivals[0].seconds,
        codec_seconds=codec_seconds,
        frames=frames,
        eos=eos,
        arrivals=tuple(arrivals),
    )
