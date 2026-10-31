"""Bounded frame delivery, EOS, flushing, and output-timeline measurements."""

import time
from collections.abc import Callable
from typing import Protocol, override

import torch
from torch import Tensor
from transformers.generation.streamers import BaseStreamer

from voicelab_core.synthesis import ChunkArrival, SynthesisMetrics
from voicelab_inference.sesame.torch.audio import Audio, validate_audio
from voicelab_models.sesame.model import SAMPLE_RATE


class AudioDecoder(Protocol):
    def decode(self, codes: Tensor) -> Audio: ...
    def close(self) -> None: ...


class AudioStreamer(BaseStreamer):
    """Consume Transformers frames; callers own cleanup even after cancellation."""

    def __init__(
        self,
        codec: AudioDecoder,
        emit: Callable[[Audio], None],
        *,
        device: str,
        codebooks: int,
        eos_token: int,
        max_frames: int,
        chunk_frames: int = 4,
    ) -> None:
        if chunk_frames < 4:
            raise ValueError("Mimi reflection padding needs four initial frames")
        self.codec = codec
        self.emit = emit
        self.device = device
        self.codebooks = codebooks
        self.eos_token = eos_token
        self.max_frames = max_frames
        self.chunk_frames = chunk_frames
        self.pending: list[Tensor] = []
        self.frame_count = 0
        self.prompt_pending = True
        self.saw_eos = False
        self.ended = False
        self.emitted = 0
        self.started = time.perf_counter()
        self.finished: float | None = None
        self.arrivals: list[ChunkArrival] = []
        self.codec_seconds = 0.0

    @override
    def put(self, value: object) -> None:
        if not isinstance(value, Tensor):
            raise TypeError("CSM streamer requires a tensor")
        if self.prompt_pending:
            self.prompt_pending = False
            return
        if self.ended or self.saw_eos:
            raise RuntimeError("Received frame after stream ended")
        if tuple(value.shape) != (1, self.codebooks):
            raise ValueError("Unexpected generated frame shape")
        if bool((value == self.eos_token).all()):
            self.saw_eos = True
            return
        if self.frame_count >= self.max_frames:
            raise ValueError("Frame budget exceeded")
        self.pending.append(value.to(self.device))
        self.frame_count += 1
        if len(self.pending) >= self.chunk_frames:
            self._decode()

    def _decode(self) -> None:
        if not self.pending:
            return
        if self.frame_count < 4:
            raise ValueError("Generated fewer than four frames; cannot stream Mimi")
        started = time.perf_counter()
        chunk = validate_audio(self.codec.decode(torch.stack(self.pending, dim=1)))
        self.codec_seconds += time.perf_counter() - started
        self.arrivals.append(
            ChunkArrival(time.perf_counter() - self.started, len(chunk))
        )
        self.emit(chunk.copy())
        self.emitted += len(chunk)
        self.pending.clear()

    @override
    def end(self) -> None:
        if not self.ended:
            if self.frame_count == 0:
                raise ValueError("CSM generated no audio frames")
            self._decode()
            self.ended = True
            self.finished = time.perf_counter()

    def metrics(self) -> SynthesisMetrics:
        total = (self.finished or time.perf_counter()) - self.started
        audio_seconds = self.emitted / SAMPLE_RATE
        return SynthesisMetrics(
            total,
            audio_seconds,
            total / audio_seconds if audio_seconds else None,
            self.arrivals[0].seconds if self.arrivals else None,
            self.codec_seconds,
            self.frame_count,
            self.saw_eos,
            tuple(self.arrivals),
        )

    def close(self) -> None:
        self.codec.close()
