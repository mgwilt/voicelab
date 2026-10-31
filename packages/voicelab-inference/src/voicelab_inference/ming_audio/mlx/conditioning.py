"""Prepare a pinned voice or reference once for all Talker text fragments."""

from dataclasses import dataclass

import mlx.core as mx
from mlx_audio.sts.models.ming_v3_audio.talker import Talker
from mlx_audio.tts.models.bailingmm.bailingmm import ming3_instruction
from mlx_audio.tts.models.lingv3_tts.speaker import SpeakerEncoder

from voicelab_inference.audio import reference_pcm
from voicelab_inference.ming_audio.mlx.reference import (
    encode_reference,
    trim_reference_tail,
)
from voicelab_models.ming_audio.model import OUTPUT_SAMPLE_RATE, Request


@dataclass(frozen=True)
class Conditioning:
    instruction: str | None
    speaker: list[mx.array] | None
    latent: mx.array | None


def prepare(talker: Talker, request: Request) -> Conditioning:
    instruction: str | None = ming3_instruction(ip=request.voice)
    speaker: list[mx.array] | None = None
    prompt_latent: mx.array | None = None
    if request.reference_audio is not None:
        instruction = None
        samples = reference_pcm(request.reference_audio, OUTPUT_SAMPLE_RATE)
        trimmed = trim_reference_tail(samples, OUTPUT_SAMPLE_RATE)
        waveform = mx.array(samples)
        if not talker._speaker_path.is_file():
            raise ValueError(
                "The pinned checkpoint lacks its reference speaker encoder."
            )
        encoder = SpeakerEncoder(talker._speaker_path)
        extracted: object = encoder(waveform, OUTPUT_SAMPLE_RATE)
        if not isinstance(extracted, mx.array) or extracted.size != 192:
            raise ValueError("Reference speaker encoder must return 192 values.")
        projected: object = talker._speaker_embeddings(extracted, False)
        if not isinstance(projected, list):
            raise TypeError("Speaker projection must return an embedding list.")
        speaker = []
        for embedding in projected:
            if not isinstance(embedding, mx.array):
                raise TypeError("Projected speaker embeddings must be MLX arrays.")
            speaker.append(embedding)
        if request.reference_text is not None:
            continuation = (
                mx.array(trimmed) if request.trim_reference_silence else waveform
            )
            prompt_latent = encode_reference(talker, continuation)
            if not isinstance(prompt_latent, mx.array):
                raise TypeError("Reference encoder must return MLX latents.")
    return Conditioning(instruction, speaker, prompt_latent)
