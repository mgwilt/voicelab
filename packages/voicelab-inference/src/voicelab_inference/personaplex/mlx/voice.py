"""NVIDIA voice conditioning: saved embeddings or loudness-normalized audio."""

import io
import math
import tarfile
from typing import TYPE_CHECKING

import numpy as np
import pyloudnorm as pyln
from numpy.typing import NDArray

from voicelab_inference.audio import reference_pcm
from voicelab_models.personaplex.model import FRAME_SAMPLES, SAMPLE_RATE, Request

if TYPE_CHECKING:
    from mlx_audio.codec.models.mimi.mimi import Mimi

    from voicelab_inference.personaplex.mlx.scheduler import Scheduler


def reference_wave(request: Request) -> NDArray[np.float32] | None:
    if request.reference_audio is None:
        return None
    waveform = reference_pcm(request.reference_audio, SAMPLE_RATE)
    loudness = float(pyln.Meter(SAMPLE_RATE).integrated_loudness(waveform))
    if not math.isfinite(loudness):
        raise ValueError("Voice reference must contain audible speech, not silence.")
    normalized = np.asarray(
        pyln.normalize.loudness(waveform, loudness, -24.0), dtype=np.float32
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Voice normalization produced non-finite audio.")
    return normalized


def condition_voice(
    scheduler: Scheduler,
    codec: Mimi,
    voices: str,
    request: Request,
    waveform: NDArray[np.float32] | None,
    text_tokens: tuple[int, ...],
) -> None:
    import mlx.core as mx

    from voicelab_inference.personaplex.mlx.scheduler import SINE

    if waveform is not None:
        codec.reset_state()
        codec.downsample.reset_state()
        codec.upsample.reset_state()
        padded = np.pad(waveform, (0, (-len(waveform)) % FRAME_SAMPLES))
        sine = mx.array(SINE, dtype=mx.int32).reshape(1, 8, 1)
        for offset in range(0, len(padded), FRAME_SAMPLES):
            frame = mx.array(padded[offset : offset + FRAME_SAMPLES])[None, None, :]
            codes = codec.encode_step(frame)[:, :8, :]
            scheduler.step(sine, codes, 3)
        scheduler.instructions(text_tokens)
        return
    import torch

    with tarfile.open(voices, "r:gz") as archive:
        member = archive.getmember(f"voices/{request.voice}")
        if not member.isfile() or member.size > 64 * 1024 * 1024:
            raise ValueError("Invalid voice prompt archive member.")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Voice prompt is missing.")
        with stream:
            state: object = torch.load(
                io.BytesIO(stream.read()), map_location="cpu", weights_only=True
            )
    if not isinstance(state, dict):
        raise ValueError("Voice prompt must contain tensors.")
    embeddings: object = state.get("embeddings")
    cache: object = state.get("cache")
    if not isinstance(embeddings, torch.Tensor) or not isinstance(cache, torch.Tensor):
        raise ValueError("Voice prompt embeddings and cache are required.")
    if cache.dtype not in (torch.int32, torch.int64) or cache.shape != (1, 17, 4):
        raise ValueError(
            "Voice cache must contain integer tokens with shape [1, 17, 4]."
        )
    if not ((cache[:, 0] >= -2) & (cache[:, 0] <= 32000)).all().item():
        raise ValueError("Voice cache text token is out of range.")
    if not ((cache[:, 1:] >= -2) & (cache[:, 1:] <= 2048)).all().item():
        raise ValueError("Voice cache audio token is out of range.")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("Voice prompt embeddings are non-finite.")
    scheduler.prompt(
        mx.array(embeddings.float().numpy()).astype(mx.bfloat16),
        mx.array(cache.numpy()).astype(mx.int32),
        text_tokens,
    )
